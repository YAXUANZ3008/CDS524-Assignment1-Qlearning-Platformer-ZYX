# CDS524 Assignment 1 Report (1000–1500 words)

## Title
**Learning to Solve a 2D Platformer with DQN (Web Canvas Implementation)**

---

## 1. Introduction
Describe the assignment goal, why reinforcement learning is suitable for this game, and what success means.

**To fill:**
- Problem motivation
- Course requirement alignment
- Summary of final outcome

---

## 2. Game Design
### 2.1 Objective and Rules
Explain player/AI objective, episode termination conditions, controls, and game states.

### 2.2 State Space and Action Space
Document the 16-dimensional normalized state vector and 5 discrete actions.

### 2.3 Reward Function Design
Explain staged reward shaping and why each reward term is needed.

**To fill:**
- Final reward table
- Why negative penalties were reduced
- How this improves learning stability

---

## 3. DQN Algorithm Implementation
### 3.1 Network Architecture
Describe 16->256->128->64->5 model with ReLU.

### 3.2 Replay Memory and Training Loop
Explain buffer size, batch sampling, and training step order.

### 3.3 Target Network and Stability
Explain target synchronization every 10 steps.

### 3.4 Epsilon-Greedy Strategy
Explain decay schedule and forced exploration reset.

**To fill:**
- Hyperparameter table
- Pseudocode of train loop
- Key implementation details

---

## 4. Experimental Setup and Evaluation
### 4.1 Setup
Number of episodes, speed mode, browser/runtime conditions.

### 4.2 Metrics
- Episode reward trend
- Pass rate over last 50 episodes
- First successful pass episode
- Final pass rate after 300 episodes

### 4.3 Results
Insert chart screenshots and discuss trends.

**To fill:**
- Figure 1: Reward curve
- Figure 2: Pass rate curve
- Figure 3: Action distribution
- Quantitative summary table

---

## 5. Challenges and Solutions
List major issues encountered and how they were solved:
- Non-learning behavior
- Reward imbalance
- Action execution mismatch
- Visualization scaling issues
- Episode reset/loop bugs

---

## 6. Conclusion
Summarize contributions and final performance. Mention possible future improvements.

---

## 7. References
- CDS524 course materials
- DQN foundational papers/tutorials
- Chart.js documentation

---

## Appendix (Optional)
- Additional logs
- Extra screenshots
- Ablation comparison notes
