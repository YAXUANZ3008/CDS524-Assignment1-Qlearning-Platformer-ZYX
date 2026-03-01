# CDS524 Assignment 1 – Platformer DQN (Web)

## 1) Project Overview
This assignment implements a browser-based side-scrolling platformer with a DQN-controlled AI agent.  
The objective is to train the agent to maximize cumulative reward and reach the goal flag under changing maps.

## 2) Rubric Mapping
- **Game Design (10 marks)**: explicit objective/rules, 16-dim state space, 5-action space, reward shaping with clear termination rules.
- **Q-Learning Implementation (10 marks)**: DQN with epsilon-greedy exploration, replay memory, online/target networks, and periodic target sync.
- **Game Interaction (10 marks)**: HTML5 Canvas gameplay, keyboard controls, and real-time dashboards.
- **Documentation & Delivery (10 marks)**: reproducible run path, report template, and deliverable-ready project structure.

## 3) Runtime and Dependencies
- Browser-based (Chrome / Edge / Firefox)
- No build step required
- External dependency via CDN:
  - Chart.js (training dashboard charts)

## 4) Run Instructions
1. Open `code/index.html` in a browser (or run a local static server and open it).
2. Controls:
   - `← / →`: move
   - `↑`: jump
   - `Y`: start
   - `R`: restart
   - `ESC`: back to start
3. Use the top controls to switch mode, speed, and max episodes.

## 5) Environment Design

### 5.1 State Space
- 16-dimensional normalized state vector (`stateDim = 16`), including:
  - AI position and velocity
  - nearest spike/platform/coin features
  - progress and step-related context

### 5.2 Action Space
- 5 discrete mutually exclusive actions:
  - `0`: LEFT
  - `1`: RIGHT
  - `2`: JUMP
  - `3`: LEFT_JUMP
  - `4`: RIGHT_JUMP

### 5.3 Curriculum / Level Sampling
- Early episodes: `classic` dominant
- Mid episodes: `classic + zigzag`
- Later episodes: random among `classic / zigzag / gaps`

### 5.4 Teacher Guidance
- Enabled in early training and linearly decays by episode index
- Current configuration:
  - `teacherAssistMaxEpisodes = 180`
  - `teacherOverrideProbStart = 0.35`
  - `teacherOverrideProbEnd = 0.06`
  - imitation bonus applied when action matches teacher suggestion

## 6) DQN Implementation Summary

### 6.1 Network & Hyperparameters (current code)
- Network: `16 -> 256 -> 128 -> 64 -> 5` (ReLU hidden layers)
- Optimizer: Adam
- Learning rate: `0.001`
- Discount factor (`gamma`): `0.95`
- Epsilon-greedy: `epsilon 0.99 -> 0.1`, decay `0.999`
- Batch size: `32`
- Replay memory size: `10000`
- Target network sync interval: every `10` train steps

### 6.2 Training Mechanisms
- Replay memory random sampling to reduce temporal correlation
- Periodic target network synchronization for stability
- Per-episode epsilon decay
- Negative-streak safeguard: if 10 consecutive episode rewards are negative, epsilon can be lifted to improve exploration

## 7) Dashboard and Outputs
- Reward curve + moving average
- Pass rate curve (recent-window based)
- Action distribution (recent episodes)
- Real-time core metrics: epsilon, loss, reward, total steps, success ratio
- Built-in save/load:
  - localStorage training persistence
  - model export/import (JSON)

## 8) Validation Checklist
- Console logs print episode reward / epsilon / loss / success flag
- Weight checksum delta printed periodically to verify network updates
- Auto-training can continuously generate episodes up to configured maximum

## 9) Submission Checklist
Required files in this assignment package:
- `code/index.html`
- `code/styles.css`
- `code/app.js`
- `report/report.md`
- `README.md`

Before final submission, ensure report screenshots and numeric results are based on your latest run.

