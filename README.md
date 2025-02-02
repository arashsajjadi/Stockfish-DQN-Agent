# Chess Reinforcement Learning (DQN vs Stockfish)
An experimental project using **Deep Q-Networks (DQN)** to train a chess-playing AI against **Stockfish** on a **Windows 11** machine. The goal is to explore whether reinforcement learning can develop a playable chess agent using consumer hardware.

## Features
- **Per-Move Rewards**: Uses Stockfish evaluations to provide feedback for each move.
- **Prioritized Experience Replay (PER)**: Learns more efficiently by prioritizing impactful moves.
- **Dynamic Stockfish Skill Level**: Adjusts based on agent's performance.
- **Parallelized Training**: CPU-based game simulations, GPU-based model training.
- **Home Research**: Running on a personal PC to see if it can learn chess.

## Hardware & Environment
- **OS**: Windows 11 (64-bit)
- **CPU**: Intel Core i7-14700F (20 Cores, 28 Threads)
- **GPU**: NVIDIA RTX 4060 (8GB VRAM)
- **RAM**: 32GB DDR4
- **Deep Learning Framework**: PyTorch 2.5.1 with CUDA 12.4
- **Stockfish Version**: Latest

## Installation
Ensure you have **Anaconda** installed. Then, run:

```sh
# Create a new environment
conda create -n chess-rl python=3.12

# Activate environment
conda activate chess-rl

# Install dependencies
pip install -r requirements.txt
```

## Running the Trainer
```sh
python chess_rl_trainer.py
```

## Dependencies (`requirements.txt`)
```sh
torch==2.5.1
torchvision
torchaudio
numpy
matplotlib
python-chess
psutil
stockfish
```

## Notes
- This is a **home research project** exploring RL training feasibility on a home PC.
- The AI may **not** become strong without **significantly more compute**.
