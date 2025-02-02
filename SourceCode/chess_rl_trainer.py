"""
Optimized Parallel Chess RL Trainer
Hardware: NVIDIA RTX 4060 (8GB), Intel Core i7-14700F, 32GB RAM
Opponent: Stockfish (adaptive skill)
Agent: Deep Q-Network (DQN) for chess

This version:
  - Pre-generates a full move mapping (including promotion moves) so that the action space size is 20480.
  - Uses per-move reward calculation based on Stockfish evaluation differences.
  - Adds extra rewards for tactical moves (check, capture, castling).
  - Ensures the final game outcome reward is added (not overwriting intermediate rewards).
  - Implements a simple Prioritized Experience Replay (PER) mechanism.
  - Runs CPU simulations in parallel and adjusts GPU training updates accordingly.
  - Adjusts Stockfish difficulty gradually over a rolling window of 100 episodes.

Before running:
  - Install dependencies:
      pip install torch torchvision torchaudio python-chess matplotlib psutil
  - Ensure your Stockfish binary is at:
      stockfish/stockfish-windows-x86-64-avx2.exe
"""

import os
import time
import random
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import chess
import chess.engine
import psutil  # For CPU monitoring

# =============================================================================
# Global Move Mapping and Helper Functions
# =============================================================================

MOVE_INDEX_MAP = {}
INDEX_MOVE_MAP = {}
ACTION_SPACE_SIZE = None  # Will be set by init_move_mapping()

def init_move_mapping():
    """
    Pre-generate a full move mapping:
      - For each from-square (64) and each to-square (64), add the non-promotion move.
      - For each from-square and to-square, add promotion moves for each promotion piece in ['q', 'r', 'b', 'n'].
    Total action space size becomes 4096 (non-promotion) + 4096*4 (promotion) = 20480.
    """
    global MOVE_INDEX_MAP, INDEX_MOVE_MAP, ACTION_SPACE_SIZE
    MOVE_INDEX_MAP = {}
    INDEX_MOVE_MAP = {}
    idx = 0
    # Non-promotion moves.
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            move = chess.Move(from_sq, to_sq)
            uci_str = move.uci()  # e.g., "e2e4" (length==4)
            if len(uci_str) == 4:
                MOVE_INDEX_MAP[uci_str] = idx
                INDEX_MOVE_MAP[idx] = uci_str
                idx += 1
    # Promotion moves.
    promotion_pieces = ['q', 'r', 'b', 'n']
    prom_map = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            for prom in promotion_pieces:
                move = chess.Move(from_sq, to_sq, promotion=prom_map[prom])
                uci_str = move.uci()  # e.g., "e7e8q" (length==5)
                MOVE_INDEX_MAP[uci_str] = idx
                INDEX_MOVE_MAP[idx] = uci_str
                idx += 1
    ACTION_SPACE_SIZE = idx
    logging.info(f"Initialized move mapping with ACTION_SPACE_SIZE = {ACTION_SPACE_SIZE}")

def move_to_index(move):
    """Convert a chess.Move to its corresponding index using the pre-generated mapping."""
    uci_str = move.uci()
    if uci_str in MOVE_INDEX_MAP:
        return MOVE_INDEX_MAP[uci_str]
    else:
        raise ValueError(f"Move {uci_str} not found in mapping.")

def index_to_move(index):
    """Convert an index back to a chess.Move using the pre-generated mapping."""
    uci_str = INDEX_MOVE_MAP.get(index)
    if uci_str is None:
        raise ValueError(f"Index {index} not found in mapping.")
    return chess.Move.from_uci(uci_str)

def board_to_tensor(board):
    """
    Convert a python-chess Board to a 12x8x8 numpy tensor.
    Channels: 6 piece types for white and 6 for black.
    """
    tensor_board = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        channel = piece.piece_type - 1 if piece.color == chess.WHITE else piece.piece_type - 1 + 6
        tensor_board[channel, row, col] = 1.0
    return tensor_board

# =============================================================================
# Configuration and Device Setup
# =============================================================================

STOCKFISH_PATH = os.path.join("stockfish", "stockfish-windows-x86-64-avx2.exe")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

# =============================================================================
# DQN Model Definition
# =============================================================================

class DQN(nn.Module):
    def __init__(self, action_space_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.dropout = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(1024 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, action_space_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# =============================================================================
# Prioritized Experience Replay Buffer
# =============================================================================

class PERReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []  # list of (experience, priority)

    def push(self, experience):
        # experience: (state, action, reward, next_state, done, legal_moves)
        priority = abs(experience[2]) + 0.0001  # using |reward| + epsilon as priority
        self.buffer.append((experience, priority))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        priorities = np.array([p for (_, p) in self.buffer])
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i][0] for i in indices]
        # Unpack the batch into separate arrays
        state, action, reward, next_state, done, legal_moves = zip(*batch)
        return (np.array(state), list(action), list(reward),
                np.array(next_state), list(done), list(legal_moves))

    def __len__(self):
        return len(self.buffer)

# Use PERReplayBuffer in place of the simple ReplayBuffer.
ReplayBuffer = PERReplayBuffer

# =============================================================================
# DQN Agent
# =============================================================================

class DQNAgent:
    def __init__(self, action_space_size, lr=3e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_final=0.1, epsilon_decay=20000):
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.policy_net = DQN(action_space_size).to(DEVICE)
        self.target_net = DQN(action_space_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state, legal_moves):
        self.steps_done += 1
        eps_threshold = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                        np.exp(-1. * self.steps_done / self.epsilon_decay)
        if random.random() < eps_threshold:
            move = random.choice(legal_moves)
            return move_to_index(move), eps_threshold
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
            legal_indices = [move_to_index(m) for m in legal_moves]
            mask = np.full(self.action_space_size, -np.inf)
            mask[legal_indices] = q_values[legal_indices]
            best_action = np.argmax(mask)
            return best_action, eps_threshold

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones, legal_moves_batch = replay_buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values_policy = self.policy_net(next_states)
            next_q_values_target = self.target_net(next_states)
            next_q_values = []
            for i in range(batch_size):
                legal_moves = legal_moves_batch[i]
                legal_indices = [move_to_index(m) for m in legal_moves]
                if legal_indices:
                    q_policy = next_q_values_policy[i].cpu().numpy()
                    mask = np.full(self.action_space_size, -np.inf)
                    mask[legal_indices] = q_policy[legal_indices]
                    best_next_action = np.argmax(mask)
                    next_q_val = next_q_values_target[i, best_next_action].item()
                else:
                    next_q_val = 0.0
                next_q_values.append(next_q_val)
            next_q_values = torch.tensor(next_q_values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# =============================================================================
# Chess Game Simulation (Parallelized on CPU)
# =============================================================================

def simulate_game(agent_state_dict, stockfish_skill, simulation_id, max_moves=200, time_limit=0.1):
    # Limit CPU threads in this process.
    torch.set_num_threads(1)
    global ACTION_SPACE_SIZE
    if ACTION_SPACE_SIZE is None:
        init_move_mapping()

    local_model = DQN(ACTION_SPACE_SIZE)
    local_model.load_state_dict(agent_state_dict)
    local_model.to("cpu")
    local_model.eval()

    board = chess.Board()
    transitions = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({"Skill Level": stockfish_skill})
    except Exception as e:
        logging.error(f"Simulation {simulation_id}: Could not start Stockfish: {e}")
        return transitions

    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            # Get evaluation before move.
            pre_state = board_to_tensor(board)
            try:
                pre_analysis = engine.analyse(board, chess.engine.Limit(depth=1))
                pre_eval = pre_analysis['score'].white().score(mate_score=10000)
                if pre_eval is None:
                    pre_eval = 0
            except Exception as e:
                pre_eval = 0

            legal_moves = list(board.legal_moves)
            state_tensor = torch.tensor(pre_state, dtype=torch.float32).unsqueeze(0).to("cpu")
            with torch.no_grad():
                q_values = local_model(state_tensor).cpu().numpy().flatten()
            legal_indices = [move_to_index(m) for m in legal_moves]
            mask = np.full(ACTION_SPACE_SIZE, -np.inf)
            mask[legal_indices] = q_values[legal_indices]
            best_action_index = np.argmax(mask)
            best_move = chess.Move.from_uci(INDEX_MOVE_MAP[best_action_index])
            if best_move not in legal_moves:
                best_move = random.choice(legal_moves)
            action_index = move_to_index(best_move)
            board.push(best_move)

            # Get evaluation after move.
            try:
                post_analysis = engine.analyse(board, chess.engine.Limit(depth=1))
                post_eval = post_analysis['score'].white().score(mate_score=10000)
                if post_eval is None:
                    post_eval = 0
            except Exception as e:
                post_eval = 0

            # Base reward from evaluation change.
            reward = (post_eval - pre_eval) / 100.0

            # Additional adjustments:
            if (post_eval - pre_eval) < -200:
                reward -= 1.0  # strong penalty for blunder
            elif (post_eval - pre_eval) > 200:
                reward += 1.0  # bonus for brilliant move

            # Check if move gives check.
            if board.is_check():
                reward += 0.05

            # Check if move captures a piece.
            if board.is_capture(best_move):
                reward += 0.1

            # Check if move is castling.
            if board.is_castling(best_move):
                reward += 0.1

            # If the game ends after the move, add final outcome reward.
            if board.is_game_over():
                result = board.result()
                if result == "1-0":
                    reward += 1.0
                elif result == "0-1":
                    reward -= 1.0
                else:
                    reward += 0.2

            next_state = board_to_tensor(board)
            done = board.is_game_over()
            transitions.append((pre_state, action_index, reward, next_state, done, list(board.legal_moves)))
        else:
            try:
                result = engine.play(board, chess.engine.Limit(time=time_limit))
                board.push(result.move)
            except Exception as e:
                logging.error(f"Simulation {simulation_id}: Stockfish error: {e}")
                break
        move_count += 1

    # (Optional) If the game ended, add a final outcome reward to the last move.
    if board.is_game_over() and transitions:
        result = board.result()
        final_reward = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.2
        s, a, old_reward, ns, d, lm = transitions[-1]
        transitions[-1] = (s, a, old_reward + final_reward, ns, True, lm)
    engine.quit()
    return transitions

# =============================================================================
# Training Loop
# =============================================================================

def training_loop():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Starting training...")

    init_move_mapping()
    agent = DQNAgent(ACTION_SPACE_SIZE)
    replay_buffer = ReplayBuffer(capacity=100000)

    num_episodes = 10000
    batch_size = 128
    target_update_frequency = 10
    simulation_processes = min(4, mp.cpu_count() - 1)
    stockfish_skill = 1
    win_threshold = 0.55  # Increase if win rate > 55%
    lose_threshold = 0.35   # Decrease if win rate < 35%
    results_window = deque(maxlen=100)  # Rolling window of last 100 episodes

    episode_rewards = []
    loss_history = []

    model_save_path = "dqn_chess_model.pth"
    logs_save_path = "training_logs.pkl"
    if os.path.exists(model_save_path):
        try:
            loaded_state = torch.load(model_save_path, map_location=DEVICE)
            if loaded_state['fc3.weight'].shape == agent.policy_net.state_dict()['fc3.weight'].shape:
                agent.policy_net.load_state_dict(loaded_state)
                agent.target_net.load_state_dict(loaded_state)
                logging.info("Resumed training from saved model.")
            else:
                logging.warning("Saved model dimensions do not match current architecture. Starting fresh training.")
        except Exception as e:
            logging.error(f"Error loading saved model: {e}. Starting fresh training.")

    pool = mp.Pool(processes=simulation_processes)

    for episode in range(num_episodes):
        start_time = time.time()
        policy_state = {k: v.cpu() for k, v in agent.policy_net.state_dict().items()}

        async_results = []
        for sim_id in range(simulation_processes):
            async_result = pool.apply_async(simulate_game,
                                            args=(policy_state, stockfish_skill, sim_id))
            async_results.append(async_result)

        episode_transitions = []
        episode_reward = 0.0
        wins = 0
        losses_count = 0
        draws = 0

        for async_result in async_results:
            transitions = async_result.get()
            episode_transitions.extend(transitions)
            if transitions:
                final_reward = transitions[-1][2]
                episode_reward += final_reward
                if final_reward > 0:
                    wins += 1
                elif final_reward < 0:
                    losses_count += 1
                else:
                    draws += 1

        # Update steps_done by the number of agent moves in this episode.
        agent.steps_done += len(episode_transitions)

        for trans in episode_transitions:
            replay_buffer.push(trans)

        # Perform several training updates per episode to use GPU fully.
        update_iterations = 3
        for _ in range(update_iterations):
            loss = agent.update(replay_buffer, batch_size)
            if loss is not None:
                loss_history.append(loss)

        episode_rewards.append(episode_reward)

        # Record outcome for rolling window:
        # We'll consider the final move reward sign as outcome (win if >0, loss if <0, draw otherwise)
        if episode_transitions:
            final_outcome = 1 if episode_transitions[-1][2] > 0 else (-1 if episode_transitions[-1][2] < 0 else 0.5)
            results_window.append(final_outcome)
        else:
            results_window.append(0.5)

        # Adjust Stockfish skill based on win rate over last 100 episodes.
        if len(results_window) == 100:
            win_rate = sum(1 for outcome in results_window if outcome == 1) / 100.0
            if win_rate > win_threshold:
                stockfish_skill = min(stockfish_skill + 1, 20)
            elif win_rate < lose_threshold:
                stockfish_skill = max(stockfish_skill - 1, 1)

        current_eps = agent.epsilon_final + (agent.epsilon_start - agent.epsilon_final) * \
                      np.exp(-1. * agent.steps_done / agent.epsilon_decay)
        logging.info(f"Episode {episode}: Reward={episode_reward:.2f} | Wins={wins} | "
                     f"Losses={losses_count} | Draws={draws} | Loss={loss if loss else 0:.4f} | "
                     f"Stockfish Level={stockfish_skill} | Epsilon={current_eps:.4f} | "
                     f"Duration={time.time()-start_time:.2f}s")

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        if episode % 2 == 0:
            torch.save(agent.policy_net.state_dict(), model_save_path)
            with open(logs_save_path, "wb") as f:
                pickle.dump({
                    "episode_rewards": episode_rewards,
                    "loss_history": loss_history,
                    "stockfish_skill": stockfish_skill
                }, f)

            # Dynamic plotting
            plt.figure(figsize=(12, 10))

            plt.subplot(2, 2, 1)
            plt.plot(episode_rewards, label="Episode Reward", color='blue')
            plt.title("Reward Progress")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(loss_history, label="Loss", color='black')
            plt.title("Loss Over Time")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.legend()

            # Cumulative counts for wins, losses, and draws (using final outcomes)
            wins_list = [sum(1 for r in results_window if r == 1)] * len(episode_rewards)
            losses_list = [sum(1 for r in results_window if r == -1)] * len(episode_rewards)
            draws_list = [sum(1 for r in results_window if r == 0.5)] * len(episode_rewards)

            plt.subplot(2, 2, 3)
            plt.plot(wins_list, label="Wins", color='blue')
            plt.plot(losses_list, label="Losses", color='magenta')
            plt.plot(draws_list, label="Draws", color='yellow')
            plt.title("Game Outcomes (Last 100 Episodes)")
            plt.xlabel("Episode")
            plt.ylabel("Count")
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot([stockfish_skill]*len(episode_rewards), label="Stockfish Skill", color='cyan')
            plt.title("Stockfish Skill Level Over Time")
            plt.xlabel("Episode")
            plt.ylabel("Skill Level")
            plt.legend()

            plt.tight_layout()
            plt.savefig("training_progress.png")
            plt.show()

            print(f"\nEpisode {episode}: Reward={episode_reward:.2f} | Wins={wins} | "
                  f"Losses={losses_count} | Draws={draws} | Loss={loss if loss else 0:.4f} | "
                  f"Stockfish Level={stockfish_skill} | Epsilon={current_eps:.4f} | "
                  f"Duration={time.time()-start_time:.2f}s")
            print(f"CPU usage: {psutil.cpu_percent()}%")

        logging.info(f"CPU usage: {psutil.cpu_percent()}%")

        if episode > 0 and episode % 1000 == 0:
            logging.info("Cooldown: Pausing training for 5 minutes to manage GPU load.")
            time.sleep(300)

    pool.close()
    pool.join()
    logging.info("Training completed.")

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    training_loop()
