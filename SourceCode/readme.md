# Stockfish-DQN-Agent  

## **Overview**  
This project is a **home research experiment** to explore whether a Deep Q-Network (DQN) can learn to play chess **move-by-move** using **Stockfish evaluations** as feedback. The goal is to determine if a reinforcement learning agent can improve over time with limited hardware and without large-scale datasets or supervised training.

## **Purpose of the Project**  
- 🏠 **A Home AI Experiment**: This is **not a professional chess engine** but a test to see if a **DQN can learn purely from Stockfish evaluation values**.  
- 🎯 **Move-by-Move Learning**: The agent learns **gradually**, not just from the final game result but from **Stockfish's centipawn evaluations after each move**.  
- 🏎️ **Optimized for Home PC**: The project is built to run on a **mid-range gaming PC** (RTX 4060, Intel i7-14700F, 32GB RAM) to test if reinforcement learning in chess can be effective **without massive computation**.  
- 🔬 **Exploring RL in Chess**: Unlike traditional self-play RL (e.g., AlphaZero), this agent is trained **against Stockfish** using **DQN and Prioritized Experience Replay (PER)**.

## **How It Works**  
### **1️⃣ Pre-Processing & Move Mapping**  
- The full **chess move space** (20,480 moves) is mapped to **DQN action indices**.  
- The agent uses a **12x8x8 tensor representation of the board** as state input.  

### **2️⃣ Reward System**  
- **Stockfish Evaluates Every Move**  
  - Before the agent moves, **Stockfish assigns a centipawn score**.  
  - After the move, **Stockfish evaluates the new position**.  
  - **Reward = (New Evaluation - Old Evaluation) / 100**.  
- **Extra Adjustments**  
  - **-1.0 penalty for blunders** (if the evaluation drops >200 centipawns).  
  - **+1.0 bonus for brilliant moves** (if the evaluation improves >200 centipawns).  
  - **Bonus for tactical moves** (captures, castling, checks).  
  - **Final game result gives an extra reward** (Win = +1.0, Loss = -1.0, Draw = +0.2).  

### **3️⃣ Training Process**  
- **Deep Q-Network (DQN)** predicts move values.  
- **Experience replay (PER)** ensures important moves are learned first.  
- **Epsilon decay** allows the agent to explore at first, then exploit learned knowledge.  
- **Stockfish adjusts difficulty dynamically** based on the agent’s win rate.  

### **4️⃣ Optimization for Home Training**  
- **Parallel CPU simulations** speed up training.  
- **GPU is only used for DQN training** while Stockfish runs on the CPU.  
- **Adaptive difficulty** prevents Stockfish from being too weak or too strong.  

## **Current Progress & Challenges**  
- ✅ The reward function is now based on **Stockfish evaluation differences**.  
- ✅ The loss is decreasing, but the agent is **still losing every game**.  
- 🔧 CPU/GPU efficiency is not optimal—Stockfish is **bottlenecking the GPU**.  
- ❌ The agent has **not won or drawn a single game yet**.  

## **Future Goals**  
- Improve **move selection** (ensure the agent understands long-term strategy).  
- Fix CPU bottlenecking to **use the GPU more efficiently**.  
- Train **until at least one game is won or drawn** to confirm learning.  

🚀 This project is **a fun experiment** to see if a home PC can train an AI to play chess **without a massive dataset or cloud computing**.  
