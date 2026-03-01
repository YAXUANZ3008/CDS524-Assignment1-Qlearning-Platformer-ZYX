# CDS524 Assignment 1 - Game Design Documentation (Module 1)

## 1. Game Objective and Rules (子项1.1)

### 1.1 Core Objective
- Player and AI race in a side-scrolling platform map.
- Win condition: first to touch the goal flag.
- Lose condition: touching spikes or falling out of map bounds.

### 1.2 Episode Lifecycle (Closed-loop Design)
Each episode has deterministic start/end to avoid infinite loops:
1. Initialize level, player, AI, reward trackers, behavior trackers.
2. Per frame loop:
   - Observe state
   - Select action
   - Execute action
   - Compute reward + done
   - Store transition
   - Train DQN (if replay size >= batch size)
3. Episode terminates immediately when any condition is met:
   - Success: AI reaches goal
   - Failure: AI dies (spike/out-of-map)
   - Timeout: 60 seconds (3600 frames)
   - Stagnation: no new max progress for 300 frames
   - Loop behavior: oscillation/no net forward progress for 200 frames
   - Ineffective action repetition for 200 frames
4. End-of-episode:
   - Persist metrics
   - Update charts
   - Start next episode (auto mode) / show game-over panel (manual mode)

### 1.3 Reset Rules
On reset, all episode-only variables are reinitialized:
- positions, velocity, on_ground
- reward accumulators
- progress milestones
- anti-loop trackers (max progress, no-progress frames, action windows)

---

## 2. State Space and Action Space (子项1.2)

## 2.1 Action Space (5 discrete actions)
- 0: LEFT (horizontal left)
- 1: RIGHT (horizontal right)
- 2: JUMP (vertical jump if on_ground)
- 3: LEFT_JUMP (left + jump if on_ground)
- 4: RIGHT_JUMP (right + jump if on_ground)

All actions are mapped 1:1 to execution logic in `execute_action()`.

## 2.2 State Space (fixed 16 dimensions, normalized 0-1)
1. AI x
2. AI y
3. AI vx
4. AI vy
5. on_ground
6. goal x
7. goal y
8. nearest forward spike x
9. nearest forward spike y
10. distance to nearest forward spike
11. nearest forward platform x
12. nearest forward platform y
13. nearest coin x
14. nearest coin y
15. current map progress
16. historical max progress

Notes:
- “forward” means object.x > ai.x.
- Missing objects are padded with value 1.
- Final vector length is always exactly 16.

---

## 3. Reward Function Design (子项1.3)

## 3.1 Positive Rewards
- +2000 goal reached
- +200 each 10% progress milestone (one-time per milestone)
- +5 per frame when moving forward
- +1 survival reward per alive frame
- +100 successful spike bypass (one-time per spike)
- +50 coin collected
- +100 re-climb reward after falling back to ground and climbing again

## 3.2 Negative Rewards
- -50 death (spike/fall)
- -20 timeout
- -3 per frame when falling behind historical max progress
- -5 per frame when no progress lasts >=100 frames

## 3.3 Forced Termination for Stability
- timeout @ 3600 frames
- stagnation @ 300 frames without max-progress improvement
- loop @ 200-frame oscillation/no net displacement
- ineffective action @ 200 repeated ineffective frames

---

## 4. Class and System Design (子项1.4)

## 4.1 Main Classes
- `Entity`: base rectangle object
- `Character`: physics, collisions, movement states
- `Level`: map templates, platforms/spikes/coins/goal
- `WebDQNAgent`: DQN model, replay memory, policy, reward logic, optimization
- `Game`: runtime loop, episode management, UI synchronization, persistence

## 4.2 High-level Architecture Diagram (Text)
- Input Layer: keyboard / auto training controls
- Environment Layer: Level + Character physics + collisions
- RL Layer: state extraction -> action -> reward -> replay -> train
- Visualization Layer: HUD panels, charts, logs
- Persistence Layer: localStorage snapshots and training histories

## 4.3 Sequence Diagram (Simplified)
1. `Game.update()` gets current state from `WebDQNAgent.get_state()`
2. `WebDQNAgent.select_action()` returns action via epsilon-greedy
3. `execute_action()` applies velocity/jump
4. physics updates and collisions resolve on_ground/death/goal
5. `get_reward()` computes reward and done
6. transition stored in replay buffer
7. `update()` trains online net and periodically syncs target net
8. if done: finalize episode, persist data, move to next episode

---

## 5. Validation Checklist (Module 1)
- [x] Clear objective and explicit rules
- [x] Explicit episode start/end closure
- [x] Fixed 16-dim normalized state space
- [x] Explicit 5-action mapping
- [x] Balanced positive/negative reward with anti-stuck guidance
- [x] Written game design doc including class/system description
