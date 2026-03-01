# CDS524 Assignment 1 - Q-Learning Implementation Documentation (Module 2)

## 1. DQN Network Structure (子项2.1)

The implementation uses a 4-layer fully connected DQN:
- Input layer: 16-dimensional state vector
- Hidden layer 1: 256 neurons + ReLU
- Hidden layer 2: 128 neurons + ReLU
- Hidden layer 3: 64 neurons + ReLU
- Output layer: 5 Q-values (LEFT, RIGHT, JUMP, LEFT_JUMP, RIGHT_JUMP)

### Why this structure
- 16D compact state avoids feature noise.
- 256/128/64 balances representation and browser-side runtime cost.
- 5 outputs map directly to the discrete action space.

---

## 2. Hyperparameters and Design Logic (子项2.3)

- Learning rate (`lr`): 0.001
- Discount factor (`gamma`): 0.95
- Batch size: 32
- Replay memory size: 10000
- Target sync interval: every 10 update steps

### Epsilon-greedy
- `epsilon` init: 0.99
- `epsilon_min`: 0.10
- `epsilon_decay`: 0.999 per episode
- Forced exploration: if recent 10 episode rewards are all negative, epsilon is temporarily reset to at least 0.9.

Rationale:
- High initial epsilon ensures broad exploration.
- Slow decay keeps exploration active for a 300-episode training horizon.
- Forced exploration avoids local optima when policy collapses.

---

## 3. Experience Replay and Target Network (子项2.1 / 2.4)

### Replay mechanism
1. Store transitions `(s, a, r, s', done)` into replay memory.
2. During update, sample a random batch (size 32, shuffled without replacement).
3. Validate sample dimensionality before training.
4. Train online network on each sampled transition.

### Target network
- A frozen target network is used to compute `max(next_q)` for stable targets.
- Every 10 optimization steps:
  - `target <- online`

---

## 4. Training Update Formula and Pipeline (子项2.1)

### Target Q formula
For each transition:

`target_q = reward + gamma * max(next_q) * (1 - done)`

This explicitly masks the bootstrap term at terminal states.

### Loss and optimizer
- Loss: Mean Squared Error (MSE), implemented as TD error squared per sample and averaged over batch.
- Optimizer: Adam.
- Adam time step `t` is incremented once per training sample and shared across all layer updates in that step.

### Backprop pipeline (strict order)
1. Zero gradients (fresh zero arrays)
2. Forward pass
3. Compute target and loss
4. Backward pass
5. Adam update on all layers

---

## 5. End-to-end Training Flow (子项2.4)

Per frame in running episode:
1. Build normalized 16D state
2. Select action by epsilon-greedy
3. Execute action in environment
4. Compute reward and done
5. Store transition to replay memory
6. Run DQN update if replay size >= 32
7. If done: finalize episode, update dashboard, decay epsilon, log metrics

Per episode logs include:
- episode index
- episode reward
- epsilon
- loss
- success flag
- end reason

---

## 6. Recorded Training Data (子项2.2)

Each episode stores:
- cumulative reward
- success flag
- survival steps
- end reason
- action distribution

These records are used by dashboard charts:
- reward + moving average
- pass rate over recent 50 episodes
- action distribution over recent 100 episodes

---

## 7. Notes on Performance Targets

The implementation includes mechanisms to improve learning reliability (reward shaping, anti-loop termination, teacher assist, forced exploration), but stochastic RL outcomes can still vary by random seed and browser runtime.
Practical evaluation should be done with repeated runs and smoothed metrics from the dashboard.
