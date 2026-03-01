const SCREEN_WIDTH = 800;
const SCREEN_HEIGHT = 400;
const FPS = 60;
const PLAYER_SIZE = 30;
const MOVE_SPEED = 5;
const JUMP_FORCE = -15;
const GRAVITY = 0.8;

const WHITE = "#ffffff";
const BLACK = "#000000";
const RED = "#ff0000";
const GREEN = "#00ff00";
const YELLOW = "#ffff00";
const BLUE = "#0000ff";
const PURPLE = "#800080";
const ORANGE = "#ffa500";

const ACTION_NAMES = ["LEFT", "RIGHT", "JUMP", "LEFT_JUMP", "RIGHT_JUMP"];

// 角色边界判定参数：
// - 跳到屏幕上方不再判死（允许高跳后自然落回）；
// - 只有明显坠落到屏幕下方一定距离才算出界死亡。
const FALL_DEATH_MARGIN = 80;

// 手动对战模式下的自动下一回合延时（帧）：
// 回合结束后短暂停留结果页，再自动进入下一回合，避免“卡在结束页不训练”。
const MANUAL_NEXT_EPISODE_DELAY_FRAMES = Math.floor(FPS * 1.2);

// 16维状态向量语义标签：严格对应简化后的 get_state 顺序。
const STATE_DIMENSION_LABELS = [
  "AI位置X(归一化)",
  "AI位置Y(归一化)",
  "AI速度X(归一化)",
  "AI速度Y(归一化)",
  "AI是否在地面(0/1)",
  "终点位置X(归一化)",
  "终点位置Y(归一化)",
  "前方最近陷阱X",
  "前方最近陷阱Y",
  "到前方陷阱水平距离",
  "前方最近平台X",
  "前方最近平台Y",
  "最近金币-X",
  "最近金币-Y",
  "当前地图进度",
  "回合步数归一化",
];

const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const overlay = document.getElementById("overlay");

const hud = {
  state: document.getElementById("hudState"),
  playerScore: document.getElementById("hudPlayerScore"),
  playerProgress: document.getElementById("hudPlayerProgress"),
  aiProgress: document.getElementById("hudAiProgress"),
  aiAction: document.getElementById("hudAiAction"),
  aiReward: document.getElementById("hudAiReward"),
  playerAlive: document.getElementById("hudPlayerAlive"),
  aiAlive: document.getElementById("hudAiAlive"),
};

// 自动训练模式UI控件：用于模式切换、速度控制、最大回合设置与进度展示。
const uiControls = {
  btnSaveModel: document.getElementById("btnSaveModel"),
  btnLoadModel: document.getElementById("btnLoadModel"),
  modelFileInput: document.getElementById("modelFileInput"),
  btnMode: document.getElementById("btnMode"),
  btnSpeed: document.getElementById("btnSpeed"),
  maxEpisodes: document.getElementById("trainMaxEpisodes"),
  trainingInfo: document.getElementById("trainingInfo"),
  btnClearTrainingData: document.getElementById("btnClearTrainingData"),
};

// localStorage键：分别保存训练统计和Agent参数，刷新后可继续训练。
const STORAGE_KEYS = {
  trainingStats: "cds524_web_training_stats_v1",
  agentSnapshot: "cds524_web_agent_snapshot_v1",
};

// RL面板元素：用于展示强化学习核心参数（实时更新）。
const rlHud = {
  epsilon: document.getElementById("rlEpsilon"),
  epsilonPct: document.getElementById("rlEpsilonPct"),
  stepReward: document.getElementById("rlStepReward"),
  episodeReward: document.getElementById("rlEpisodeReward"),
  avgReward: document.getElementById("rlAvgReward"),
  episodeSteps: document.getElementById("rlEpisodeSteps"),
  totalSteps: document.getElementById("rlTotalSteps"),
  successRatio: document.getElementById("rlSuccessRatio"),
  stateVector: document.getElementById("rlStateVector"),
  q0: document.getElementById("q0"),
  q1: document.getElementById("q1"),
  q2: document.getElementById("q2"),
  q3: document.getElementById("q3"),
  q4: document.getElementById("q4"),
  qBarPanel: document.getElementById("qBarPanel"),
  stateHeatmap: document.getElementById("stateHeatmap"),
  decisionLog: document.getElementById("decisionLog"),
  dashboardEpisodeReward: document.getElementById("dashboardEpisodeReward"),
  dashboardSuccessRate: document.getElementById("dashboardSuccessRate"),
  dashboardTotalEpisodes: document.getElementById("dashboardTotalEpisodes"),
};

// Q值条形图DOM缓存：初始化一次后每帧仅更新样式和文本，减少DOM重建开销。
const qBarEls = [];

// 状态热力图DOM缓存：16个状态格子固定存在，每帧更新颜色强度和值文本。
const stateCellEls = [];

// 初始化Q值动态条形图结构。
function initQBarPanel() {
  if (!rlHud.qBarPanel) return;
  rlHud.qBarPanel.innerHTML = "";

  for (let i = 0; i < ACTION_NAMES.length; i++) {
    const row = document.createElement("div");
    row.className = "qbar-row";

    const label = document.createElement("span");
    label.className = "qbar-label";
    label.textContent = ACTION_NAMES[i];

    const track = document.createElement("div");
    track.className = "qbar-track";

    const fill = document.createElement("div");
    fill.className = "qbar-fill";
    track.appendChild(fill);

    const value = document.createElement("span");
    value.className = "qbar-value";
    value.textContent = "0.000";

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);

    rlHud.qBarPanel.appendChild(row);
    qBarEls.push({ row, fill, value });
  }
}

// 初始化状态热力面板结构：每个维度显示“标签 + 当前值”。
function initStateHeatmap() {
  if (!rlHud.stateHeatmap) return;
  rlHud.stateHeatmap.innerHTML = "";

  for (let i = 0; i < STATE_DIMENSION_LABELS.length; i++) {
    const cell = document.createElement("div");
    cell.className = "state-cell";

    const label = document.createElement("span");
    label.className = "state-label";
    label.textContent = `${i}. ${STATE_DIMENSION_LABELS[i]}`;

    const value = document.createElement("span");
    value.className = "state-value";
    value.textContent = "0.000";

    cell.appendChild(label);
    cell.appendChild(value);
    rlHud.stateHeatmap.appendChild(cell);
    stateCellEls.push({ cell, value });
  }
}

// 根据Q值数组实时刷新横向动态条形图，并高亮当前最大Q值动作。
function updateQBarPanel(qValues) {
  if (!qBarEls.length) return;
  const safeQ = Array.isArray(qValues) ? qValues : [0, 0, 0, 0, 0];
  const maxAbs = Math.max(0.0001, ...safeQ.map((v) => Math.abs(v)));

  let maxIndex = 0;
  for (let i = 1; i < safeQ.length; i++) {
    if (safeQ[i] > safeQ[maxIndex]) maxIndex = i;
  }

  for (let i = 0; i < qBarEls.length; i++) {
    const value = Number(safeQ[i] || 0);
    const percent = Math.max(4, (Math.abs(value) / maxAbs) * 100);
    qBarEls[i].fill.style.width = `${percent.toFixed(2)}%`;
    qBarEls[i].value.textContent = value.toFixed(3);
    qBarEls[i].row.classList.toggle("active", i === maxIndex);
  }
}

// 根据16维状态值刷新热力面板：值越大背景越亮，帮助快速识别高激活特征。
function updateStateHeatmap(stateVector) {
  if (!stateCellEls.length) return;
  const safeState = Array.isArray(stateVector) ? stateVector : new Array(16).fill(0);

  for (let i = 0; i < stateCellEls.length; i++) {
    const value = Number(safeState[i] || 0);
    const clamped = Math.max(0, Math.min(1, value));
    const alpha = 0.06 + clamped * 0.42;
    stateCellEls[i].cell.style.background = `rgba(126, 231, 255, ${alpha.toFixed(3)})`;
    stateCellEls[i].value.textContent = clamped.toFixed(3);
  }
}

// 刷新底部决策日志显示：仅渲染最近12条，保持可读与性能。
function renderDecisionLog(logs) {
  if (!rlHud.decisionLog) return;
  const recentLogs = Array.isArray(logs) ? logs.slice(-12) : [];

  rlHud.decisionLog.innerHTML = "";
  for (const item of recentLogs) {
    const line = document.createElement("div");
    line.className = "decision-item";
    line.textContent = item;
    rlHud.decisionLog.appendChild(line);
  }
  rlHud.decisionLog.scrollTop = rlHud.decisionLog.scrollHeight;
}

initQBarPanel();
initStateHeatmap();

// Tab切换：不改布局，只在右侧面板内切换可见模块。
const tabButtons = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");
tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const targetId = btn.dataset.tab;
    tabButtons.forEach((b) => b.classList.remove("active"));
    tabContents.forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    const targetEl = document.getElementById(targetId);
    if (targetEl) targetEl.classList.add("active");
  });
});

// 图表对象：用于训练看板三大核心图表。
let rewardChart = null;
let passRateChart = null;
let actionDistChart = null;

// 看板配置常量：
// - 奖励移动平均窗口用于平滑波动；
// - 通关率窗口固定最近50回合；
// - 动作分布窗口固定最近100回合。
const REWARD_MOVING_AVG_WINDOW = 10;
const PASS_RATE_WINDOW = 50;
const ACTION_DIST_WINDOW = 100;

// 计算移动平均：返回与原序列等长数组，前期数据不足时仍基于可用样本计算。
function calculateMovingAverage(values, windowSize) {
  if (!Array.isArray(values) || values.length === 0) return [];
  const output = [];
  let rollingSum = 0;

  for (let i = 0; i < values.length; i++) {
    rollingSum += values[i];
    if (i >= windowSize) rollingSum -= values[i - windowSize];
    const divisor = Math.min(windowSize, i + 1);
    output.push(rollingSum / divisor);
  }

  return output;
}

// 计算每个回合对应的最近50回合通关率曲线（百分比）。
function buildPassRateSeries(history, windowSize) {
  if (!Array.isArray(history) || history.length === 0) return [];
  const passRates = [];
  let rollingSuccess = 0;

  for (let i = 0; i < history.length; i++) {
    rollingSuccess += history[i].success ? 1 : 0;
    if (i >= windowSize) rollingSuccess -= history[i - windowSize].success ? 1 : 0;
    const divisor = Math.min(windowSize, i + 1);
    passRates.push((rollingSuccess / divisor) * 100);
  }

  return passRates;
}

// 汇总最近100回合动作分布：将每回合的5动作计数累加成一个环形图数据向量。
function buildActionDistribution(history, windowSize) {
  const totals = [0, 0, 0, 0, 0];
  if (!Array.isArray(history) || history.length === 0) return totals;

  const recentHistory = history.slice(-windowSize);
  for (const episode of recentHistory) {
    const counts = Array.isArray(episode.actionCounts) ? episode.actionCounts : [0, 0, 0, 0, 0];
    for (let i = 0; i < ACTION_NAMES.length; i++) {
      totals[i] += Number(counts[i] || 0);
    }
  }

  return totals;
}

// 将训练历史同步到三张图表：仅在“回合结束/加载/清空”时调用，避免逐帧更新造成卡顿。
function refreshTrainingDashboard(history) {
  const safeHistory = Array.isArray(history) ? history : [];
  const labels = safeHistory.map((entry) => String(entry.episode));
  const rewards = safeHistory.map((entry) => Number(entry.reward || 0));
  const movingAvg = calculateMovingAverage(rewards, REWARD_MOVING_AVG_WINDOW);
  const passRates = buildPassRateSeries(safeHistory, PASS_RATE_WINDOW);
  const actionDist = buildActionDistribution(safeHistory, ACTION_DIST_WINDOW);

  if (rewardChart) {
    rewardChart.data.labels = labels;
    rewardChart.data.datasets[0].data = rewards;
    rewardChart.data.datasets[1].data = movingAvg;
    rewardChart.update("none");
  }

  if (passRateChart) {
    passRateChart.data.labels = labels;
    passRateChart.data.datasets[0].data = passRates;
    passRateChart.update("none");
  }

  if (actionDistChart) {
    actionDistChart.data.labels = ACTION_NAMES;
    actionDistChart.data.datasets[0].data = actionDist;
    actionDistChart.update("none");
  }
}

function initCharts() {
  // 若Chart.js未加载，则跳过图表初始化，避免主循环报错。
  if (typeof Chart === "undefined") return;

  const rewardCtx = document.getElementById("rewardChart");
  const passRateCtx = document.getElementById("passRateChart");
  const actionDistCtx = document.getElementById("actionDistChart");
  if (!rewardCtx || !passRateCtx || !actionDistCtx) return;

  // 图表1：累计奖励曲线 + 移动平均线，核心用于观察“不会玩 -> 会玩”的趋势。
  rewardChart = new Chart(rewardCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "单回合累计奖励",
          data: [],
          borderColor: "#7ee7ff",
          backgroundColor: "rgba(126, 231, 255, 0.15)",
          tension: 0.25,
          pointRadius: 1.5,
          borderWidth: 2,
          fill: true,
        },
        {
          label: `移动平均(${REWARD_MOVING_AVG_WINDOW})`,
          data: [],
          borderColor: "#ffd166",
          backgroundColor: "rgba(255, 209, 102, 0.08)",
          tension: 0.2,
          pointRadius: 0,
          borderWidth: 2,
          fill: false,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#dbe6ff" } },
        tooltip: { mode: "index", intersect: false },
      },
      scales: {
        x: {
          ticks: { color: "#a9b3d6", maxTicksLimit: 8 },
          grid: { color: "rgba(170,190,255,0.12)" },
          title: { display: true, text: "回合数", color: "#c9d6ff" },
        },
        y: {
          min: -200,
          max: 2500,
          ticks: { color: "#a9b3d6" },
          grid: { color: "rgba(170,190,255,0.12)" },
          title: { display: true, text: "累计奖励", color: "#c9d6ff" },
        },
      },
    },
  });

  // 图表2：通关率曲线（最近50回合滚动），直观展示学习成功率演进。
  passRateChart = new Chart(passRateCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: `最近${PASS_RATE_WINDOW}回合通关率`,
          data: [],
          borderColor: "#8bebc2",
          backgroundColor: "rgba(139, 235, 194, 0.16)",
          tension: 0.25,
          pointRadius: 0,
          borderWidth: 2,
          fill: true,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#dbe6ff" } },
      },
      scales: {
        x: {
          ticks: { color: "#a9b3d6", maxTicksLimit: 8 },
          grid: { color: "rgba(170,190,255,0.12)" },
          title: { display: true, text: "回合数", color: "#c9d6ff" },
        },
        y: {
          min: 0,
          max: 100,
          ticks: {
            color: "#a9b3d6",
            callback: (value) => `${value}%`,
          },
          grid: { color: "rgba(170,190,255,0.12)" },
          title: { display: true, text: "通关率", color: "#c9d6ff" },
        },
      },
    },
  });

  // 图表3：动作分布（最近100回合），用环形图查看策略偏好变化。
  actionDistChart = new Chart(actionDistCtx, {
    type: "doughnut",
    data: {
      labels: ACTION_NAMES,
      datasets: [
        {
          label: `最近${ACTION_DIST_WINDOW}回合动作频次`,
          data: [0, 0, 0, 0, 0],
          backgroundColor: [
            "rgba(121, 195, 255, 0.75)",
            "rgba(139, 235, 194, 0.75)",
            "rgba(255, 213, 128, 0.75)",
            "rgba(201, 171, 255, 0.75)",
            "rgba(255, 158, 171, 0.75)",
          ],
          borderColor: ["#79c3ff", "#8bebc2", "#ffd580", "#c9abff", "#ff9eab"],
          borderWidth: 1,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#dbe6ff" } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const value = Number(ctx.raw || 0);
              const total = ctx.dataset.data.reduce((sum, v) => sum + Number(v || 0), 0);
              const pct = total > 0 ? (value / total) * 100 : 0;
              return `${ctx.label}: ${value} (${pct.toFixed(1)}%)`;
            },
          },
        },
      },
      cutout: "56%",
    },
  });
}

initCharts();

const keys = new Set();
const blockedKeys = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "KeyY", "KeyR"]);

function intersects(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

function randomFloat(min, max) {
  return Math.random() * (max - min) + min;
}

class Entity {
  constructor(x, y, w, h, color) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.color = color;
  }
  get rect() {
    return { x: this.x, y: this.y, w: this.w, h: this.h };
  }
  draw(cameraX = 0) {
    ctx.fillStyle = this.color;
    ctx.fillRect(this.x - cameraX, this.y, this.w, this.h);
  }
}

class Particle {
  constructor(x, y, color, vx, vy, life, size) {
    this.x = x;
    this.y = y;
    this.color = color;
    this.vx = vx;
    this.vy = vy;
    this.life = life;
    this.maxLife = life;
    this.size = size;
  }

  update() {
    this.x += this.vx;
    this.y += this.vy;
    this.vy += 0.08;
    this.life -= 1;
  }

  draw(cameraX = 0) {
    if (this.life <= 0) return;
    const alpha = this.life / this.maxLife;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(this.x - cameraX, this.y, this.size, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

class Character extends Entity {
  constructor(x, y, color) {
    super(x, y, PLAYER_SIZE, PLAYER_SIZE, color);
    this.velX = 0;
    this.velY = 0;
    this.onGround = false;
    this.isAlive = true;
    this.reachedGoal = false;
    this.score = 0;
  }

  reset(x, y) {
    this.x = x;
    this.y = y;
    this.velX = 0;
    this.velY = 0;
    this.onGround = true;
    this.isAlive = true;
    this.reachedGoal = false;
    this.score = 0;
  }

  applyPhysics(level) {
    if (!this.isAlive || this.reachedGoal) return;

    this.velY += GRAVITY;

    this.x += this.velX;
    for (const p of level.platforms) {
      if (intersects(this.rect, p.rect)) {
        if (this.velX > 0) this.x = p.x - this.w;
        if (this.velX < 0) this.x = p.x + p.w;
        this.velX = 0;
      }
    }

    this.y += this.velY;
    this.onGround = false;
    for (const p of level.platforms) {
      if (intersects(this.rect, p.rect)) {
        if (this.velY > 0) {
          this.y = p.y - this.h;
          this.velY = 0;
          this.onGround = true;
        } else if (this.velY < 0) {
          this.y = p.y + p.h;
          this.velY = 0;
        }
      }
    }

    if (this.x < 0) this.x = 0;
    const worldWidth = level.worldWidth || SCREEN_WIDTH;
    if (this.x + this.w > worldWidth) this.x = worldWidth - this.w;

    // 出界判定修正：
    // 1) 不再因为“跳到屏幕上方”直接死亡；
    // 2) 仅在坠落到屏幕下方较深位置时判死，避免误判。
    if (this.y > SCREEN_HEIGHT + FALL_DEATH_MARGIN) {
      this.isAlive = false;
      return;
    }

    for (const s of level.spikes) {
      if (intersects(this.rect, s.rect)) {
        this.isAlive = false;
        return;
      }
    }

    for (let i = level.coins.length - 1; i >= 0; i--) {
      if (intersects(this.rect, level.coins[i].rect)) {
        level.coins.splice(i, 1);
        this.score += 30;
      }
    }

    if (intersects(this.rect, level.goal.rect)) {
      this.reachedGoal = true;
    }
  }

  draw(cameraX = 0, time = 0) {
    const drawX = this.x - cameraX;
    const drawY = this.y;

    // 角色动画说明：
    // 1) 行走时用sin波驱动上下摆动与腿部摆幅；
    // 2) 跳跃时做轻微拉伸；
    // 3) 仅改变渲染，不改变碰撞盒，保证逻辑稳定。
    const walking = this.onGround && Math.abs(this.velX) > 0.1;
    const walkPhase = walking ? Math.sin(time * 0.02 + this.x * 0.1) : 0;
    const bodyYOffset = walking ? walkPhase * 1.5 : 0;
    const isJumping = !this.onGround;

    const bodyW = isJumping ? this.w * 0.9 : this.w;
    const bodyH = isJumping ? this.h * 1.08 : this.h;
    const bodyX = drawX + (this.w - bodyW) / 2;
    const bodyY = drawY + (this.h - bodyH) / 2 + bodyYOffset;

    ctx.save();
    ctx.shadowColor = this.color;
    ctx.shadowBlur = 14;

    const grad = ctx.createLinearGradient(bodyX, bodyY, bodyX, bodyY + bodyH);
    grad.addColorStop(0, "#ffffff");
    grad.addColorStop(0.12, this.color);
    grad.addColorStop(1, "#1b1b1b");

    ctx.fillStyle = grad;
    ctx.fillRect(bodyX, bodyY, bodyW, bodyH);

    ctx.shadowBlur = 0;
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.fillRect(bodyX + 3, bodyY + 3, bodyW - 6, 4);

    ctx.fillStyle = "#0b0b0b";
    const eyeY = bodyY + 9;
    ctx.fillRect(bodyX + 7, eyeY, 4, 4);
    ctx.fillRect(bodyX + bodyW - 11, eyeY, 4, 4);

    const legSwing = walking ? walkPhase * 3.2 : 0;
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bodyX + 8, bodyY + bodyH - 1);
    ctx.lineTo(bodyX + 8 - legSwing, bodyY + bodyH + 5);
    ctx.moveTo(bodyX + bodyW - 8, bodyY + bodyH - 1);
    ctx.lineTo(bodyX + bodyW - 8 + legSwing, bodyY + bodyH + 5);
    ctx.stroke();

    ctx.restore();
  }
}

class Level {
  constructor(mode = "classic") {
    this.worldWidth = 1500;
    this.mode = mode;

    // 关卡模式工厂：每个模式定义平台/陷阱/金币/终点，
    // 再经过轻微随机扰动，避免AI只背固定路径。
    const spec = this._buildModeSpec(mode);
    this.platforms = spec.platforms.map((obj) => new Entity(obj.x, obj.y, obj.w, obj.h, GREEN));
    this.spikes = spec.spikes.map((obj) => new Entity(obj.x, obj.y, obj.w, obj.h, RED));
    this.coins = spec.coins.map((obj) => new Entity(obj.x, obj.y, obj.w, obj.h, YELLOW));
    this.goal = new Entity(spec.goal.x, spec.goal.y, spec.goal.w, spec.goal.h, ORANGE);
  }

  static get MODE_NAMES() {
    return ["classic", "zigzag", "gaps"];
  }

  _jitter(value, range) {
    return value + randomFloat(-range, range);
  }

  _buildClassicSpec() {
    return {
      platforms: [
        { x: 0, y: SCREEN_HEIGHT - 40, w: this.worldWidth, h: 40 },
        { x: 150, y: 305, w: 140, h: 16 },
        { x: 330, y: 275, w: 135, h: 16 },
        { x: 510, y: 245, w: 130, h: 16 },
        { x: 690, y: 220, w: 130, h: 16 },
        { x: 860, y: 195, w: 130, h: 16 },
        { x: 1030, y: 165, w: 125, h: 16 },
        { x: 1190, y: 140, w: 120, h: 16 },
        { x: 1320, y: 120, w: 110, h: 16 },
      ],
      spikes: [
        { x: 720, y: SCREEN_HEIGHT - 50, w: 42, h: 10 },
      ],
      coins: [
        { x: 240, y: 275, w: 14, h: 14 },
        { x: 420, y: 245, w: 14, h: 14 },
        { x: 600, y: 215, w: 14, h: 14 },
        { x: 780, y: 185, w: 14, h: 14 },
        { x: 960, y: 155, w: 14, h: 14 },
        { x: 1130, y: 130, w: 14, h: 14 },
        { x: 1290, y: 110, w: 14, h: 14 },
      ],
      goal: { x: 1410, y: 85, w: 20, h: 55 },
    };
  }

  _buildZigzagSpec() {
    return {
      platforms: [
        { x: 0, y: SCREEN_HEIGHT - 40, w: this.worldWidth, h: 40 },
        { x: 160, y: 310, w: 140, h: 16 },
        { x: 340, y: 240, w: 130, h: 16 },
        { x: 520, y: 300, w: 130, h: 16 },
        { x: 700, y: 230, w: 130, h: 16 },
        { x: 880, y: 285, w: 125, h: 16 },
        { x: 1060, y: 215, w: 125, h: 16 },
        { x: 1235, y: 170, w: 120, h: 16 },
        { x: 1365, y: 125, w: 105, h: 16 },
      ],
      spikes: [
        { x: 760, y: SCREEN_HEIGHT - 50, w: 42, h: 10 },
      ],
      coins: [
        { x: 255, y: 280, w: 14, h: 14 },
        { x: 435, y: 210, w: 14, h: 14 },
        { x: 615, y: 270, w: 14, h: 14 },
        { x: 795, y: 200, w: 14, h: 14 },
        { x: 975, y: 255, w: 14, h: 14 },
        { x: 1145, y: 185, w: 14, h: 14 },
        { x: 1310, y: 105, w: 14, h: 14 },
      ],
      goal: { x: 1415, y: 85, w: 20, h: 55 },
    };
  }

  _buildGapsSpec() {
    return {
      platforms: [
        { x: 0, y: SCREEN_HEIGHT - 40, w: this.worldWidth, h: 40 },
        { x: 150, y: 315, w: 125, h: 16 },
        { x: 330, y: 270, w: 120, h: 16 },
        { x: 520, y: 230, w: 120, h: 16 },
        { x: 710, y: 265, w: 120, h: 16 },
        { x: 900, y: 220, w: 120, h: 16 },
        { x: 1085, y: 180, w: 120, h: 16 },
        { x: 1260, y: 140, w: 115, h: 16 },
        { x: 1380, y: 100, w: 105, h: 16 },
      ],
      spikes: [
        { x: 740, y: SCREEN_HEIGHT - 50, w: 42, h: 10 },
      ],
      coins: [
        { x: 210, y: 295, w: 14, h: 14 },
        { x: 390, y: 250, w: 14, h: 14 },
        { x: 580, y: 210, w: 14, h: 14 },
        { x: 770, y: 245, w: 14, h: 14 },
        { x: 960, y: 200, w: 14, h: 14 },
        { x: 1140, y: 160, w: 14, h: 14 },
        { x: 1320, y: 120, w: 14, h: 14 },
      ],
      goal: { x: 1410, y: 85, w: 20, h: 55 },
    };
  }

  _applyJitter(spec) {
    // 关卡随机扰动：在不破坏可通关性的前提下，轻微改变元素位置。
    const jittered = {
      platforms: spec.platforms.map((p, index) => {
        if (index === 0) return { ...p };
        return {
          x: this._jitter(p.x, 12),
          y: this._jitter(p.y, 10),
          w: p.w,
          h: p.h,
        };
      }),
      spikes: spec.spikes.map((s) => ({
        x: this._jitter(s.x, 18),
        y: s.y,
        w: s.w,
        h: s.h,
      })),
      coins: spec.coins.map((c) => ({
        x: this._jitter(c.x, 16),
        y: this._jitter(c.y, 10),
        w: c.w,
        h: c.h,
      })),
      goal: {
        x: this._jitter(spec.goal.x, 10),
        y: this._jitter(spec.goal.y, 6),
        w: spec.goal.w,
        h: spec.goal.h,
      },
    };

    return jittered;
  }

  _buildModeSpec(mode) {
    let baseSpec;
    if (mode === "zigzag") baseSpec = this._buildZigzagSpec();
    else if (mode === "gaps") baseSpec = this._buildGapsSpec();
    else baseSpec = this._buildClassicSpec();

    return this._applyJitter(baseSpec);
  }

  draw(cameraX = 0, time = 0) {
    // 平台升级：渐变 + 阴影 + 顶部高光边。
    for (const p of this.platforms) {
      const x = p.x - cameraX;

      ctx.save();
      ctx.shadowColor = "rgba(0, 255, 180, 0.25)";
      ctx.shadowBlur = 10;

      const grad = ctx.createLinearGradient(x, p.y, x, p.y + p.h);
      grad.addColorStop(0, "#77ffd2");
      grad.addColorStop(0.45, "#2ecf8f");
      grad.addColorStop(1, "#12805a");

      ctx.fillStyle = grad;
      ctx.fillRect(x, p.y, p.w, p.h);

      ctx.shadowBlur = 0;
      ctx.strokeStyle = "rgba(0, 0, 0, 0.45)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, p.y + 0.5, p.w - 1, p.h - 1);

      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.fillRect(x + 2, p.y + 2, Math.max(0, p.w - 4), 2);
      ctx.restore();
    }

    // 陷阱升级：尖刺三角图案。
    for (const s of this.spikes) {
      const x = s.x - cameraX;
      const spikeCount = Math.max(3, Math.floor(s.w / 8));
      const step = s.w / spikeCount;

      ctx.save();
      for (let i = 0; i < spikeCount; i++) {
        const sx = x + i * step;
        ctx.beginPath();
        ctx.moveTo(sx, s.y + s.h);
        ctx.lineTo(sx + step * 0.5, s.y);
        ctx.lineTo(sx + step, s.y + s.h);
        ctx.closePath();

        const sg = ctx.createLinearGradient(sx, s.y, sx, s.y + s.h);
        sg.addColorStop(0, "#ffb0b0");
        sg.addColorStop(0.4, "#ff4040");
        sg.addColorStop(1, "#8b0000");
        ctx.fillStyle = sg;
        ctx.fill();
      }
      ctx.restore();
    }

    // 金币升级：旋转 + 发光。
    for (const c of this.coins) {
      const cx = c.x - cameraX + c.w / 2;
      const cy = c.y + c.h / 2;
      const pulse = 0.8 + 0.2 * Math.sin(time * 0.01 + c.x * 0.05);
      const spinScaleX = 0.35 + 0.65 * Math.abs(Math.sin(time * 0.015 + c.x * 0.03));

      ctx.save();
      ctx.translate(cx, cy);
      ctx.scale(spinScaleX, 1);

      ctx.shadowColor = "rgba(255, 230, 80, 0.85)";
      ctx.shadowBlur = 14 * pulse;

      const cg = ctx.createRadialGradient(0, 0, 2, 0, 0, c.w / 2);
      cg.addColorStop(0, "#fff8c9");
      cg.addColorStop(0.45, "#ffd84a");
      cg.addColorStop(1, "#b98f00");
      ctx.fillStyle = cg;

      ctx.beginPath();
      ctx.arc(0, 0, c.w / 2, 0, Math.PI * 2);
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.strokeStyle = "rgba(120, 90, 0, 0.75)";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      ctx.restore();
    }

    // 终点旗帜升级：旗面飘动动画。
    {
      const poleX = this.goal.x - cameraX;
      const poleY = this.goal.y;

      ctx.save();
      ctx.fillStyle = "#dcdcdc";
      ctx.fillRect(poleX, poleY, 4, this.goal.h);

      ctx.strokeStyle = "rgba(0,0,0,0.35)";
      ctx.strokeRect(poleX, poleY, 4, this.goal.h);

      const wave = Math.sin(time * 0.02) * 5;
      ctx.beginPath();
      ctx.moveTo(poleX + 4, poleY + 6);
      ctx.quadraticCurveTo(poleX + 16 + wave, poleY + 13, poleX + 34, poleY + 8);
      ctx.quadraticCurveTo(poleX + 18 + wave, poleY + 2, poleX + 4, poleY + 6);
      const fg = ctx.createLinearGradient(poleX + 4, poleY, poleX + 34, poleY + 14);
      fg.addColorStop(0, "#ffd39b");
      fg.addColorStop(1, ORANGE);
      ctx.fillStyle = fg;
      ctx.fill();
      ctx.restore();
    }
  }
}

class WebDQNAgent {
  constructor() {
    this.stateDim = 16;
    this.actionDim = 5;

    this.lr = 0.001;
    this.gamma = 0.95;
    this.epsilon = 0.99;
    this.epsilonMin = 0.1;
    this.epsilonDecay = 0.999;
    this.batchSize = 32;
    this.memorySize = 10000;
    this.updateTargetStep = 10;

    this.hiddenDim1 = 256;
    this.hiddenDim2 = 128;
    this.hiddenDim3 = 64;

    // Adam优化器超参数：与常见DQN实现一致。
    this.adamBeta1 = 0.9;
    this.adamBeta2 = 0.999;
    this.adamEps = 1e-8;
    this.optimizerStep = 0;

    this.online = this._initNet();
    this.target = this._cloneNet(this.online);
    this.optim = this._initOptimizerState(this.online);

    this.memory = [];
    this.stepCount = 0;

    this.prevCoinCount = 0;
    this.lastActionWasJump = false;
    this.episodeSteps = 0;
    this.stuckSteps = 0;
    this.progressMilestones = new Set();
    this.passedSpikeIndices = new Set();
    this.recentEpisodeRewards = [];
  }

  _initNet() {
    const W1 = Array.from({ length: this.hiddenDim1 }, () =>
      Array.from({ length: this.stateDim }, () => randomFloat(-0.05, 0.05))
    );
    const b1 = Array.from({ length: this.hiddenDim1 }, () => 0);

    const W2 = Array.from({ length: this.hiddenDim2 }, () =>
      Array.from({ length: this.hiddenDim1 }, () => randomFloat(-0.05, 0.05))
    );
    const b2 = Array.from({ length: this.hiddenDim2 }, () => 0);

    const W3 = Array.from({ length: this.hiddenDim3 }, () =>
      Array.from({ length: this.hiddenDim2 }, () => randomFloat(-0.05, 0.05))
    );
    const b3 = Array.from({ length: this.hiddenDim3 }, () => 0);

    const W4 = Array.from({ length: this.actionDim }, () =>
      Array.from({ length: this.hiddenDim3 }, () => randomFloat(-0.05, 0.05))
    );
    const b4 = Array.from({ length: this.actionDim }, () => 0);

    return { W1, b1, W2, b2, W3, b3, W4, b4 };
  }

  _initOptimizerState(net) {
    // Adam一阶矩/二阶矩缓存，结构与网络参数一一对应。
    const zerosLike2D = (mat) => mat.map((row) => row.map(() => 0));
    const zerosLike1D = (arr) => arr.map(() => 0);
    return {
      mW1: zerosLike2D(net.W1),
      vW1: zerosLike2D(net.W1),
      mb1: zerosLike1D(net.b1),
      vb1: zerosLike1D(net.b1),
      mW2: zerosLike2D(net.W2),
      vW2: zerosLike2D(net.W2),
      mb2: zerosLike1D(net.b2),
      vb2: zerosLike1D(net.b2),
      mW3: zerosLike2D(net.W3),
      vW3: zerosLike2D(net.W3),
      mb3: zerosLike1D(net.b3),
      vb3: zerosLike1D(net.b3),
      mW4: zerosLike2D(net.W4),
      vW4: zerosLike2D(net.W4),
      mb4: zerosLike1D(net.b4),
      vb4: zerosLike1D(net.b4),
    };
  }

  _cloneNet(net) {
    return {
      W1: net.W1.map((row) => [...row]),
      b1: [...net.b1],
      W2: net.W2.map((row) => [...row]),
      b2: [...net.b2],
      W3: net.W3.map((row) => [...row]),
      b3: [...net.b3],
      W4: net.W4.map((row) => [...row]),
      b4: [...net.b4],
    };
  }

  _isValidNetShape(net) {
    if (
      !net ||
      !Array.isArray(net.W1) ||
      !Array.isArray(net.b1) ||
      !Array.isArray(net.W2) ||
      !Array.isArray(net.b2) ||
      !Array.isArray(net.W3) ||
      !Array.isArray(net.b3) ||
      !Array.isArray(net.W4) ||
      !Array.isArray(net.b4)
    ) {
      return false;
    }

    if (net.W1.length !== this.hiddenDim1 || net.b1.length !== this.hiddenDim1) return false;
    if (net.W2.length !== this.hiddenDim2 || net.b2.length !== this.hiddenDim2) return false;
    if (net.W3.length !== this.hiddenDim3 || net.b3.length !== this.hiddenDim3) return false;
    if (net.W4.length !== this.actionDim || net.b4.length !== this.actionDim) return false;

    for (const row of net.W1) {
      if (!Array.isArray(row) || row.length !== this.stateDim) return false;
    }
    for (const row of net.W2) {
      if (!Array.isArray(row) || row.length !== this.hiddenDim1) return false;
    }
    for (const row of net.W3) {
      if (!Array.isArray(row) || row.length !== this.hiddenDim2) return false;
    }
    for (const row of net.W4) {
      if (!Array.isArray(row) || row.length !== this.hiddenDim3) return false;
    }

    return true;
  }

  _relu(value) {
    return value > 0 ? value : 0;
  }

  _forward(net, state) {
    const z1 = new Array(this.hiddenDim1).fill(0);
    const h1 = new Array(this.hiddenDim1).fill(0);
    const z2 = new Array(this.hiddenDim2).fill(0);
    const h2 = new Array(this.hiddenDim2).fill(0);
    const z3 = new Array(this.hiddenDim3).fill(0);
    const h3 = new Array(this.hiddenDim3).fill(0);
    const q = new Array(this.actionDim).fill(0);

    for (let j = 0; j < this.hiddenDim1; j++) {
      let sum = net.b1[j];
      for (let i = 0; i < this.stateDim; i++) {
        sum += net.W1[j][i] * state[i];
      }
      z1[j] = sum;
      h1[j] = this._relu(sum);
    }

    for (let j = 0; j < this.hiddenDim2; j++) {
      let sum = net.b2[j];
      for (let i = 0; i < this.hiddenDim1; i++) {
        sum += net.W2[j][i] * h1[i];
      }
      z2[j] = sum;
      h2[j] = this._relu(sum);
    }

    for (let j = 0; j < this.hiddenDim3; j++) {
      let sum = net.b3[j];
      for (let i = 0; i < this.hiddenDim2; i++) {
        sum += net.W3[j][i] * h2[i];
      }
      z3[j] = sum;
      h3[j] = this._relu(sum);
    }

    for (let a = 0; a < this.actionDim; a++) {
      let sum = net.b4[a];
      for (let j = 0; j < this.hiddenDim3; j++) {
        sum += net.W4[a][j] * h3[j];
      }
      q[a] = sum;
    }

    return { z1, h1, z2, h2, z3, h3, q };
  }

  _argmax(values) {
    let index = 0;
    let best = values[0];
    for (let i = 1; i < values.length; i++) {
      if (values[i] > best) {
        best = values[i];
        index = i;
      }
    }
    return index;
  }

  resetEpisode(game) {
    this.episodeSteps = 0;
    this.lastActionWasJump = false;
    this.prevCoinCount = game.level.coins.length;
    this.stuckSteps = 0;
    this.progressMilestones.clear();
    this.passedSpikeIndices.clear();
  }

  get_state(game, aiCharacter) {
    // 状态重构为16维：仅保留决策强相关特征，全部归一化到0~1。
    const worldWidth = Math.max(1, game.level.worldWidth || SCREEN_WIDTH);
    const worldHeight = SCREEN_HEIGHT;
    const goalDistance = Math.max(1, game.level.goal.x - 40);

    const clamp01 = (value) => Math.max(0, Math.min(1, value));
    const normX = (value) => clamp01(value / worldWidth);
    const normY = (value) => clamp01(value / worldHeight);

    const aiX = aiCharacter.x;
    const aiY = aiCharacter.y;

    // 只看前方元素（x > AI当前x），避免身后信息干扰。
    const forwardSpikes = game.level.spikes.filter((s) => s.x > aiX).sort((a, b) => a.x - b.x);
    const forwardPlatforms = game.level.platforms
      .filter((p, idx) => idx > 0 && p.x > aiX)
      .sort((a, b) => a.x - b.x);

    const nearestSpike = forwardSpikes.length ? forwardSpikes[0] : null;
    const nearestPlatform = forwardPlatforms.length ? forwardPlatforms[0] : null;

    let nearestCoin = null;
    let nearestCoinDist = Number.POSITIVE_INFINITY;
    for (const coin of game.level.coins) {
      const dx = coin.x - aiX;
      const dy = coin.y - aiY;
      const dist = dx * dx + dy * dy;
      if (dist < nearestCoinDist) {
        nearestCoinDist = dist;
        nearestCoin = coin;
      }
    }

    const state = [
      normX(aiX),
      normY(aiY),
      clamp01((aiCharacter.velX + MOVE_SPEED) / (2 * MOVE_SPEED)),
      clamp01((aiCharacter.velY + 20) / 40),
      aiCharacter.onGround ? 1 : 0,
      normX(game.level.goal.x),
      normY(game.level.goal.y),
      nearestSpike ? normX(nearestSpike.x) : 1,
      nearestSpike ? normY(nearestSpike.y) : 1,
      nearestSpike ? clamp01((nearestSpike.x - aiX) / worldWidth) : 1,
      nearestPlatform ? normX(nearestPlatform.x) : 1,
      nearestPlatform ? normY(nearestPlatform.y) : 1,
      nearestCoin ? normX(nearestCoin.x) : 1,
      nearestCoin ? normY(nearestCoin.y) : 1,
      clamp01((aiX - 40) / goalDistance),
      clamp01(this.episodeSteps / 3600),
    ];

    while (state.length < 16) state.push(1);
    if (state.length > 16) state.length = 16;

    return state.map((value) => clamp01(value));
  }

  execute_action(aiCharacter, action) {
    // 动作执行严格映射5离散动作，确保不会出现“动作选了但没执行”。
    aiCharacter.velX = 0;

    if (action === 0) {
      aiCharacter.velX = -MOVE_SPEED;
    } else if (action === 1) {
      aiCharacter.velX = MOVE_SPEED;
    } else if (action === 2) {
      if (aiCharacter.onGround) {
        aiCharacter.velY = JUMP_FORCE;
        aiCharacter.onGround = false;
      }
    } else if (action === 3) {
      if (aiCharacter.onGround) {
        aiCharacter.velX = -MOVE_SPEED;
        aiCharacter.velY = JUMP_FORCE;
        aiCharacter.onGround = false;
      } else {
        aiCharacter.velX = -MOVE_SPEED;
      }
    } else if (action === 4) {
      if (aiCharacter.onGround) {
        aiCharacter.velX = MOVE_SPEED;
        aiCharacter.velY = JUMP_FORCE;
        aiCharacter.onGround = false;
      } else {
        aiCharacter.velX = MOVE_SPEED;
      }
    }
  }

  get_reward(game, aiCharacter, oldState) {
    let reward = 0;
    let done = false;
    let endReason = "running";

    this.episodeSteps += 1;

    const currentState = this.get_state(game, aiCharacter);
    const worldWidth = Math.max(1, game.level.worldWidth || SCREEN_WIDTH);
    const oldX = (oldState[0] || 0) * worldWidth;
    const newX = (currentState[0] || 0) * worldWidth;
    const goalDistance = Math.max(1, game.level.goal.x - 40);

    // 1) 终极通关奖励：一次性+2000并结束回合。
    if (aiCharacter.reachedGoal) {
      reward += 2000;
      done = true;
      endReason = "goal";
    }

    // 7) 死亡惩罚：一次性-50并结束回合。
    if (!aiCharacter.isAlive) {
      reward -= 50;
      done = true;
      endReason = "death";
    }

    // 6) 金币奖励：每枚+50。
    const currentCoinCount = game.level.coins.length;
    if (currentCoinCount < this.prevCoinCount) {
      reward += (this.prevCoinCount - currentCoinCount) * 50;
    }

    // 2) 里程碑进度奖励：每10%一次性+200。
    const progress = Math.max(0, Math.min(1, (newX - 40) / goalDistance));
    const milestone = Math.floor(progress * 10);
    for (let step = 1; step <= milestone; step++) {
      if (!this.progressMilestones.has(step)) {
        this.progressMilestones.add(step);
        reward += 200;
      }
    }

    // 3) 持续前进奖励：向右推进+5，否则-0.5。
    if (newX > oldX) reward += 5;
    else reward -= 0.5;

    // 4) 生存奖励：存活且未通关每帧+1。
    if (aiCharacter.isAlive && !aiCharacter.reachedGoal) reward += 1;

    // 5) 跳跃避障奖励：成功从陷阱左侧跨越到右侧且安全存活，每个陷阱+100仅一次。
    for (let i = 0; i < game.level.spikes.length; i++) {
      if (this.passedSpikeIndices.has(i)) continue;
      const spike = game.level.spikes[i];
      const passed = oldX + aiCharacter.w <= spike.x && newX >= spike.x + spike.w;
      const safelyAbove = aiCharacter.y + aiCharacter.h <= spike.y + 4;
      if (passed && safelyAbove && aiCharacter.isAlive) {
        this.passedSpikeIndices.add(i);
        reward += 100;
      }
    }

    // 卡住终止：若连续较长时间几乎没有前进，直接结束回合并给惩罚，
    // 防止AI在原地反复上下跳导致“看起来回合不结束”。
    if (Math.abs(newX - oldX) < 0.4) this.stuckSteps += 1;
    else this.stuckSteps = 0;

    if (!done && this.stuckSteps >= 240) {
      reward -= 20;
      done = true;
      endReason = "stuck";
    }

    // 8) 超时惩罚：60秒未通关，-20并结束回合。
    if (this.episodeSteps >= 60 * FPS && !done) {
      reward -= 20;
      done = true;
      endReason = "timeout";
    }

    this.prevCoinCount = currentCoinCount;

    // 在重置回合计数前先缓存本回合生存步数，供训练统计记录使用。
    const survivalSteps = this.episodeSteps;

    if (done) {
      this.episodeSteps = 0;
      this.lastActionWasJump = false;
      this.stuckSteps = 0;
    }

    return { reward, done, survivalSteps, endReason };
  }

  select_action(state) {
    let action;

    if (Math.random() < this.epsilon) {
      action = Math.floor(Math.random() * this.actionDim);
    } else {
      const { q } = this._forward(this.online, state);
      action = this._argmax(q);
    }

    this.lastActionWasJump = action === 2 || action === 3 || action === 4;
    return action;
  }

  onEpisodeEnd(episodeReward) {
    // 强制探索机制：连续10回合奖励为负，则把epsilon临时拉回0.9，防止陷入局部最优。
    this.recentEpisodeRewards.push(episodeReward);
    if (this.recentEpisodeRewards.length > 10) this.recentEpisodeRewards.shift();

    if (this.recentEpisodeRewards.length === 10 && this.recentEpisodeRewards.every((value) => value < 0)) {
      this.epsilon = Math.max(this.epsilon, 0.9);
      this.recentEpisodeRewards = [];
    }

    // epsilon按回合衰减：更符合“每回合探索率更新”的训练节奏。
    if (this.epsilon > this.epsilonMin) {
      this.epsilon *= this.epsilonDecay;
      if (this.epsilon < this.epsilonMin) this.epsilon = this.epsilonMin;
    }
  }

  store_memory(state, action, reward, nextState, done) {
    if (this.memory.length >= this.memorySize) this.memory.shift();
    this.memory.push({ state: [...state], action, reward, nextState: [...nextState], done });
  }

  _adamUpdate1D(param, grad, m, v) {
    this.optimizerStep += 1;
    const t = this.optimizerStep;
    const b1 = this.adamBeta1;
    const b2 = this.adamBeta2;

    for (let i = 0; i < param.length; i++) {
      m[i] = b1 * m[i] + (1 - b1) * grad[i];
      v[i] = b2 * v[i] + (1 - b2) * grad[i] * grad[i];

      const mHat = m[i] / (1 - Math.pow(b1, t));
      const vHat = v[i] / (1 - Math.pow(b2, t));
      param[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.adamEps);
    }
  }

  _adamUpdate2D(param, grad, m, v) {
    this.optimizerStep += 1;
    const t = this.optimizerStep;
    const b1 = this.adamBeta1;
    const b2 = this.adamBeta2;

    for (let r = 0; r < param.length; r++) {
      for (let c = 0; c < param[r].length; c++) {
        m[r][c] = b1 * m[r][c] + (1 - b1) * grad[r][c];
        v[r][c] = b2 * v[r][c] + (1 - b2) * grad[r][c] * grad[r][c];

        const mHat = m[r][c] / (1 - Math.pow(b1, t));
        const vHat = v[r][c] / (1 - Math.pow(b2, t));
        param[r][c] -= this.lr * mHat / (Math.sqrt(vHat) + this.adamEps);
      }
    }
  }

  _trainSingle(sample) {
    const { state, action, reward, nextState, done } = sample;

    // 1) 前向传播：在线网络输出当前Q。
    const onlineForward = this._forward(this.online, state);
    const currentQ = onlineForward.q[action];

    // 2) 目标Q计算：target = reward + gamma * max(nextQ) * (1 - done)
    const targetForward = this._forward(this.target, nextState);
    const maxNextQ = Math.max(...targetForward.q);
    const doneMask = done ? 0 : 1;
    const targetQ = reward + this.gamma * maxNextQ * doneMask;

    // 3) MSE损失梯度（对所选动作输出节点）。
    const tdError = currentQ - targetQ;

    const gW4 = this.online.W4.map((row) => row.map(() => 0));
    const gb4 = new Array(this.actionDim).fill(0);
    const gW3 = this.online.W3.map((row) => row.map(() => 0));
    const gb3 = new Array(this.hiddenDim3).fill(0);
    const gW2 = this.online.W2.map((row) => row.map(() => 0));
    const gb2 = new Array(this.hiddenDim2).fill(0);
    const gW1 = this.online.W1.map((row) => row.map(() => 0));
    const gb1 = new Array(this.hiddenDim1).fill(0);

    // 4) 反向传播：输出层 -> 隐藏层3 -> 隐藏层2 -> 隐藏层1 -> 输入。
    gb4[action] = tdError;
    for (let j = 0; j < this.hiddenDim3; j++) {
      gW4[action][j] = tdError * onlineForward.h3[j];
    }

    const dH3 = new Array(this.hiddenDim3).fill(0);
    for (let j = 0; j < this.hiddenDim3; j++) {
      dH3[j] = this.online.W4[action][j] * tdError;
    }

    const dZ3 = new Array(this.hiddenDim3).fill(0);
    for (let j = 0; j < this.hiddenDim3; j++) {
      dZ3[j] = onlineForward.z3[j] > 0 ? dH3[j] : 0;
      gb3[j] = dZ3[j];
      for (let i = 0; i < this.hiddenDim2; i++) {
        gW3[j][i] = dZ3[j] * onlineForward.h2[i];
      }
    }

    const dH2 = new Array(this.hiddenDim2).fill(0);
    for (let i = 0; i < this.hiddenDim2; i++) {
      let sum = 0;
      for (let j = 0; j < this.hiddenDim3; j++) {
        sum += this.online.W3[j][i] * dZ3[j];
      }
      dH2[i] = sum;
    }

    const dZ2 = new Array(this.hiddenDim2).fill(0);
    for (let j = 0; j < this.hiddenDim2; j++) {
      dZ2[j] = onlineForward.z2[j] > 0 ? dH2[j] : 0;
      gb2[j] = dZ2[j];
      for (let i = 0; i < this.hiddenDim1; i++) {
        gW2[j][i] = dZ2[j] * onlineForward.h1[i];
      }
    }

    const dH1 = new Array(this.hiddenDim1).fill(0);
    for (let i = 0; i < this.hiddenDim1; i++) {
      let sum = 0;
      for (let j = 0; j < this.hiddenDim2; j++) {
        sum += this.online.W2[j][i] * dZ2[j];
      }
      dH1[i] = sum;
    }

    for (let j = 0; j < this.hiddenDim1; j++) {
      const dZ1 = onlineForward.z1[j] > 0 ? dH1[j] : 0;
      gb1[j] = dZ1;
      for (let i = 0; i < this.stateDim; i++) {
        gW1[j][i] = dZ1 * state[i];
      }
    }

    // 5) Adam参数更新：按“梯度清零->反向->step”流程对所有层参数执行更新。
    this._adamUpdate2D(this.online.W4, gW4, this.optim.mW4, this.optim.vW4);
    this._adamUpdate1D(this.online.b4, gb4, this.optim.mb4, this.optim.vb4);
    this._adamUpdate2D(this.online.W3, gW3, this.optim.mW3, this.optim.vW3);
    this._adamUpdate1D(this.online.b3, gb3, this.optim.mb3, this.optim.vb3);
    this._adamUpdate2D(this.online.W2, gW2, this.optim.mW2, this.optim.vW2);
    this._adamUpdate1D(this.online.b2, gb2, this.optim.mb2, this.optim.vb2);
    this._adamUpdate2D(this.online.W1, gW1, this.optim.mW1, this.optim.vW1);
    this._adamUpdate1D(this.online.b1, gb1, this.optim.mb1, this.optim.vb1);

    return tdError * tdError;
  }

  update() {
    // 经验不足时不训练，确保采样批次有效。
    if (this.memory.length < this.batchSize) return null;

    let lossSum = 0;
    for (let i = 0; i < this.batchSize; i++) {
      const idx = Math.floor(Math.random() * this.memory.length);
      lossSum += this._trainSingle(this.memory[idx]);
    }

    this.stepCount += 1;

    if (this.stepCount % this.updateTargetStep === 0) {
      this.target = this._cloneNet(this.online);
    }

    return lossSum / this.batchSize;
  }

  getWeightChecksum() {
    // 计算权重绝对值和，用于检测网络是否确实在持续更新。
    const sumAbs2D = (mat) => mat.reduce((acc, row) => acc + row.reduce((s, v) => s + Math.abs(v), 0), 0);
    const sumAbs1D = (arr) => arr.reduce((acc, value) => acc + Math.abs(value), 0);
    return (
      sumAbs2D(this.online.W1) + sumAbs1D(this.online.b1) +
      sumAbs2D(this.online.W2) + sumAbs1D(this.online.b2) +
      sumAbs2D(this.online.W3) + sumAbs1D(this.online.b3) +
      sumAbs2D(this.online.W4) + sumAbs1D(this.online.b4)
    );
  }

  getSnapshot() {
    // 序列化Agent核心参数与网络权重，供localStorage持久化使用。
    return {
      epsilon: this.epsilon,
      stepCount: this.stepCount,
      optimizerStep: this.optimizerStep,
      online: this._cloneNet(this.online),
      target: this._cloneNet(this.target),
    };
  }

  loadSnapshot(snapshot) {
    // 从持久化数据恢复Agent状态，确保刷新页面后可继续训练。
    if (!snapshot) return;
    if (typeof snapshot.epsilon === "number") this.epsilon = snapshot.epsilon;
    if (typeof snapshot.stepCount === "number") this.stepCount = snapshot.stepCount;
    if (typeof snapshot.optimizerStep === "number") this.optimizerStep = snapshot.optimizerStep;

    if (this._isValidNetShape(snapshot.online) && this._isValidNetShape(snapshot.target)) {
      this.online = this._cloneNet(snapshot.online);
      this.target = this._cloneNet(snapshot.target);
      this.optim = this._initOptimizerState(this.online);
    } else {
      // 维度不兼容（例如旧30维模型）时忽略加载，防止训练崩溃。
      console.warn("[DQN] 模型加载被跳过：权重维度与当前状态空间不匹配");
    }
  }
}

class Game {
  constructor() {
    this.state = "start";
    this.levelMode = "classic";
    this.level = new Level(this.levelMode);
    this.player = new Character(40, SCREEN_HEIGHT - 40 - PLAYER_SIZE, BLUE);
    this.ai = new Character(40, SCREEN_HEIGHT - 40 - PLAYER_SIZE, PURPLE);

    this.aiAction = 1;
    this.aiReward = 0;
    this.resultText = "";
    this.focused = false;

    this.cameraX = 0;
    this.cameraLerp = 0.12;

    this.agent = new WebDQNAgent();

    // 训练统计：用于面板Tab2/Tab3实时展示强化学习效果。
    this.episodeReward = 0;
    this.episodeRewards = [];
    this.totalEpisodes = 0;
    this.successEpisodes = 0;
    this.lastQValues = [0, 0, 0, 0, 0];
    this.lastStateVector = new Array(16).fill(0);
    this.lastLoss = 0;
    this.lastWeightChecksum = this.agent.getWeightChecksum();
    this.currentEpisodeActionCounts = [0, 0, 0, 0, 0];
    this.decisionLogs = [];
    this.lastDecisionMessage = "";
    this.decisionLogDirty = true;

    // 自动训练模式控制参数。
    this.autoTrainingMode = false;
    this.trainingSpeed = 1;
    this.maxTrainingEpisodes = 300;
    this.trainingEpisodeIndex = 0;
    this.trainingEpisodeHistory = [];
    this.manualNextEpisodeCountdown = 0;

    // 教师引导（自动玩家）参数：
    // - 前期用启发式动作给AI“示范”，帮助更快学会基础通关动作；
    // - 随着回合增加，引导概率逐步衰减，最终让AI独立决策。
    this.enableTeacherAssist = true;
    this.teacherAssistMaxEpisodes = 180;
    this.teacherOverrideProbStart = 0.35;
    this.teacherOverrideProbEnd = 0.06;
    this.teacherImitationBonus = 0.28;
    this.lastTeacherAction = null;

    this.particles = [];
    this.maxParticles = 180;

    this.starsFar = Array.from({ length: 70 }, () => ({
      x: randomFloat(0, this.level.worldWidth),
      y: randomFloat(10, 210),
      r: randomFloat(0.6, 1.6),
      tw: randomFloat(0, Math.PI * 2),
    }));

    this.starsNear = Array.from({ length: 45 }, () => ({
      x: randomFloat(0, this.level.worldWidth),
      y: randomFloat(15, 240),
      r: randomFloat(1.0, 2.3),
      tw: randomFloat(0, Math.PI * 2),
    }));

    // 启动时加载历史训练统计与Agent参数，实现刷新后继续训练。
    this.loadTrainingData();
  }

  reset() {
    // 每回合刷新关卡：采用课程学习+随机模式，避免只在单一地图过拟合。
    this.levelMode = this.pickLevelMode();
    this.level = new Level(this.levelMode);
    this.player.reset(40, SCREEN_HEIGHT - 40 - PLAYER_SIZE);
    this.ai.reset(40, SCREEN_HEIGHT - 40 - PLAYER_SIZE);
    this.aiAction = 1;
    this.aiReward = 0;
    this.resultText = "";
    this.cameraX = 0;
    this.particles = [];
    this.agent.resetEpisode(this);
    this.manualNextEpisodeCountdown = 0;

    // 回合重置时清空累计奖励计数，不清空历史曲线。
    this.episodeReward = 0;
    this.currentEpisodeActionCounts = [0, 0, 0, 0, 0];
    this.lastQValues = [0, 0, 0, 0, 0];
    this.lastStateVector = new Array(16).fill(0);
    this.lastLoss = 0;
    this.lastWeightChecksum = this.agent.getWeightChecksum();
    this.lastTeacherAction = null;

    // 每次重开都写入一条关键日志，方便观察新回合决策起点。
    this.appendDecisionLog(`回合重置：模式=${this.levelMode}，AI开始新一轮探索`, true);
  }

  appendDecisionLog(message, force = false) {
    // 决策日志防刷屏策略：默认仅在消息变化时写入；force=true用于关键事件强制写入。
    if (!message) return;
    if (!force && message === this.lastDecisionMessage) return;

    const prefix = `#${this.agent.stepCount}`;
    this.decisionLogs.push(`${prefix} ${message}`);
    if (this.decisionLogs.length > 160) this.decisionLogs.shift();

    this.lastDecisionMessage = message;
    this.decisionLogDirty = true;
  }

  describeDecision(action) {
    // 基于当前环境构造“可解释决策语句”，只读环境信息，不参与训练更新。
    const aiCenterX = this.ai.x + this.ai.w / 2;
    const aiCenterY = this.ai.y + this.ai.h / 2;

    let nearestSpikeDist = Number.POSITIVE_INFINITY;
    for (const s of this.level.spikes) {
      const dx = s.x + s.w / 2 - aiCenterX;
      const dy = s.y + s.h / 2 - aiCenterY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < nearestSpikeDist) nearestSpikeDist = dist;
    }

    let nearestCoinDx = Number.POSITIVE_INFINITY;
    for (const c of this.level.coins) {
      const dx = c.x + c.w / 2 - aiCenterX;
      if (Math.abs(dx) < Math.abs(nearestCoinDx)) nearestCoinDx = dx;
    }

    const goalDx = this.level.goal.x - aiCenterX;
    const actionName = ACTION_NAMES[action] || "RIGHT";

    if (!this.ai.isAlive) return `发生失误，动作${actionName}后角色死亡`;
    if (this.ai.reachedGoal) return `到达终点，动作${actionName}完成通关`;

    const jumpAction = action === 2 || action === 3 || action === 4;
    if (nearestSpikeDist < 95 && jumpAction) {
      return `前方陷阱距离${nearestSpikeDist.toFixed(0)}，选择${actionName}规避`;
    }

    if (Math.abs(nearestCoinDx) < 130) {
      if (nearestCoinDx > 0 && (action === 1 || action === 4)) {
        return `检测到右侧金币，选择${actionName}尝试接近`;
      }
      if (nearestCoinDx < 0 && (action === 0 || action === 3)) {
        return `检测到左侧金币，选择${actionName}尝试接近`;
      }
    }

    if (goalDx > 0 && (action === 1 || action === 4)) {
      return `终点在前方，选择${actionName}推进进度`;
    }

    if (jumpAction) return `执行${actionName}探索垂直路径`;
    return `执行${actionName}继续探索地形`;
  }

  pickLevelMode() {
    // 多模式课程学习策略：
    // - 0~39回合：以classic为主，先稳定学会前进与基础跳跃；
    // - 40~119回合：混入zigzag，提高地形适应性；
    // - 120回合后：classic/zigzag/gaps全随机，强化泛化能力。
    const episode = this.trainingEpisodeIndex;
    if (episode < 40) return "classic";
    if (episode < 120) return Math.random() < 0.7 ? "classic" : "zigzag";

    const modes = Level.MODE_NAMES;
    return modes[Math.floor(Math.random() * modes.length)];
  }

  getTeacherAction() {
    // 启发式自动玩家（教师）：
    // 只读取当前环境信息给出建议动作，不修改DQN网络结构。
    // 规则优先级：避障 > 防坠落 > 吃金币 > 向终点推进。
    const aiCenterX = this.ai.x + this.ai.w / 2;
    const feetY = this.ai.y + this.ai.h;

    // 1) 前方近距离陷阱：优先执行右跳规避。
    for (const spike of this.level.spikes) {
      const spikeCenterX = spike.x + spike.w / 2;
      const dx = spikeCenterX - aiCenterX;
      const nearSameHeight = Math.abs(spike.y - feetY) < 60;
      if (dx > 18 && dx < 95 && nearSameHeight) {
        return this.ai.onGround ? 4 : 1;
      }
    }

    // 2) 前方落脚空档：若即将踩空且在地面，提前右跳。
    const probeX = this.ai.x + this.ai.w + 24;
    const probeY = feetY + 6;
    const hasFutureFloor = this.level.platforms.some((p) =>
      probeX >= p.x && probeX <= p.x + p.w && probeY >= p.y && probeY <= p.y + p.h + 6
    );
    if (!hasFutureFloor && this.ai.onGround) {
      return 4;
    }

    // 3) 金币吸引：附近有金币时优先向金币方向移动。
    let nearestCoin = null;
    let bestCoinDist = Number.POSITIVE_INFINITY;
    for (const coin of this.level.coins) {
      const dx = coin.x + coin.w / 2 - aiCenterX;
      const dy = coin.y + coin.h / 2 - (this.ai.y + this.ai.h / 2);
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < bestCoinDist) {
        bestCoinDist = dist;
        nearestCoin = coin;
      }
    }
    if (nearestCoin && bestCoinDist < 150) {
      const dx = nearestCoin.x + nearestCoin.w / 2 - aiCenterX;
      if (dx > 10) return 1;
      if (dx < -10) return 0;
    }

    // 4) 默认策略：向终点方向推进；若终点较高且当前在地面，则尝试右跳。
    const goalDx = this.level.goal.x - aiCenterX;
    const goalDy = this.level.goal.y - this.ai.y;
    if (goalDx > 0 && goalDy < -70 && this.ai.onGround) return 4;
    return goalDx >= 0 ? 1 : 0;
  }

  getTeacherOverrideProbability() {
    // 教师介入概率衰减：早期高、后期低，避免永远“代打”。
    if (!this.enableTeacherAssist) return 0;
    if (this.trainingEpisodeIndex >= this.teacherAssistMaxEpisodes) return 0;

    const ratio = this.trainingEpisodeIndex / Math.max(1, this.teacherAssistMaxEpisodes);
    return this.teacherOverrideProbStart + (this.teacherOverrideProbEnd - this.teacherOverrideProbStart) * ratio;
  }

  startGame() {
    // 自动训练达到最大回合后，点击“开始/重开”应能直接重新开训。
    // 这里重置训练统计与Agent，避免再次立即触发“训练完成”导致看起来无法开始。
    if (this.autoTrainingMode && this.trainingEpisodeIndex >= this.maxTrainingEpisodes) {
      this.totalEpisodes = 0;
      this.successEpisodes = 0;
      this.trainingEpisodeIndex = 0;
      this.episodeReward = 0;
      this.episodeRewards = [];
      this.trainingEpisodeHistory = [];
      this.currentEpisodeActionCounts = [0, 0, 0, 0, 0];
      this.lastQValues = [0, 0, 0, 0, 0];
      this.lastStateVector = new Array(16).fill(0);
      this.lastLoss = 0;

      this.agent = new WebDQNAgent();
      refreshTrainingDashboard(this.trainingEpisodeHistory);
      this.appendDecisionLog("自动训练已重置：开始新一轮300回合训练", true);
      this.saveTrainingData();
    }

    this.reset();
    this.state = "running";
  }

  saveTrainingData() {
    // 持久化训练统计与Agent快照，页面刷新后不丢失训练进度。
    const statsPayload = {
      maxTrainingEpisodes: this.maxTrainingEpisodes,
      totalEpisodes: this.totalEpisodes,
      successEpisodes: this.successEpisodes,
      episodeRewards: this.episodeRewards,
      trainingEpisodeIndex: this.trainingEpisodeIndex,
      trainingEpisodeHistory: this.trainingEpisodeHistory,
      trainingSpeed: this.trainingSpeed,
      autoTrainingMode: this.autoTrainingMode,
    };

    localStorage.setItem(STORAGE_KEYS.trainingStats, JSON.stringify(statsPayload));
    localStorage.setItem(STORAGE_KEYS.agentSnapshot, JSON.stringify(this.agent.getSnapshot()));
  }

  loadTrainingData() {
    // 恢复训练统计。
    const rawStats = localStorage.getItem(STORAGE_KEYS.trainingStats);
    if (rawStats) {
      try {
        this.applyTrainingStats(JSON.parse(rawStats));
      } catch (_err) {
        // 存档损坏时忽略并回退默认值。
      }
    }

    // 恢复Agent参数，支持继续训练。
    const rawAgent = localStorage.getItem(STORAGE_KEYS.agentSnapshot);
    if (rawAgent) {
      try {
        this.agent.loadSnapshot(JSON.parse(rawAgent));
      } catch (_err) {
        // 快照损坏时忽略。
      }
    }

    // 初始化后立刻恢复训练看板数据，确保刷新页面图表不丢失。
    refreshTrainingDashboard(this.trainingEpisodeHistory);
  }

  applyTrainingStats(stats) {
    // 统一训练统计恢复逻辑：
    // 1) 供页面启动时读取 localStorage 使用；
    // 2) 供“加载模型”从JSON文件恢复统计使用；
    // 3) 只恢复可校验的字段，避免坏数据污染运行状态。
    if (!stats || typeof stats !== "object") return;

    if (typeof stats.maxTrainingEpisodes === "number") {
      this.maxTrainingEpisodes = Math.max(1, Math.floor(stats.maxTrainingEpisodes));
    }
    if (typeof stats.totalEpisodes === "number") this.totalEpisodes = Math.max(0, Math.floor(stats.totalEpisodes));
    if (typeof stats.successEpisodes === "number") this.successEpisodes = Math.max(0, Math.floor(stats.successEpisodes));
    if (Array.isArray(stats.episodeRewards)) this.episodeRewards = stats.episodeRewards.slice(-150).map((v) => Number(v || 0));
    if (typeof stats.trainingEpisodeIndex === "number") {
      this.trainingEpisodeIndex = Math.max(0, Math.floor(stats.trainingEpisodeIndex));
    }
    if (typeof stats.trainingSpeed === "number") this.trainingSpeed = [1, 2, 4].includes(stats.trainingSpeed) ? stats.trainingSpeed : 1;
    if (typeof stats.autoTrainingMode === "boolean") this.autoTrainingMode = stats.autoTrainingMode;

    if (Array.isArray(stats.trainingEpisodeHistory)) {
      this.trainingEpisodeHistory = stats.trainingEpisodeHistory.slice(-300).map((entry, idx) => ({
        episode: Number(entry.episode || idx + 1),
        reward: Number(entry.reward || 0),
        success: entry.success ? 1 : 0,
        survivalSteps: Number(entry.survivalSteps || 0),
        actionCounts: Array.isArray(entry.actionCounts)
          ? entry.actionCounts.slice(0, ACTION_NAMES.length).map((v) => Number(v || 0))
          : [0, 0, 0, 0, 0],
      }));
    }

    if (uiControls.maxEpisodes) {
      uiControls.maxEpisodes.value = String(this.maxTrainingEpisodes);
    }
  }

  saveModelSnapshot() {
    // 模型保存功能（核心）：
    // - 将DQN权重+关键训练统计打包为JSON；
    // - 同步写入localStorage，满足“无需文件也能恢复”；
    // - 自动触发浏览器下载，满足“导出模型文件”需求。
    const exportPayload = {
      format: "cds524-dqn-model",
      version: 1,
      createdAt: new Date().toISOString(),
      agentSnapshot: this.agent.getSnapshot(),
      trainingStats: {
        maxTrainingEpisodes: this.maxTrainingEpisodes,
        totalEpisodes: this.totalEpisodes,
        successEpisodes: this.successEpisodes,
        episodeRewards: this.episodeRewards,
        trainingEpisodeIndex: this.trainingEpisodeIndex,
        trainingEpisodeHistory: this.trainingEpisodeHistory,
        trainingSpeed: this.trainingSpeed,
        autoTrainingMode: this.autoTrainingMode,
      },
    };

    // 先同步到localStorage，确保即使用户不下载文件也能通过“加载模型”恢复。
    localStorage.setItem(STORAGE_KEYS.agentSnapshot, JSON.stringify(exportPayload.agentSnapshot));
    localStorage.setItem(STORAGE_KEYS.trainingStats, JSON.stringify(exportPayload.trainingStats));

    // 导出JSON文件：文件名带时间戳，便于多版本模型管理。
    const timestamp = new Date().toISOString().replace(/[\:\.]/g, "-");
    const fileName = `cds524_dqn_model_${timestamp}.json`;
    const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileName;
    anchor.click();
    URL.revokeObjectURL(url);

    this.appendDecisionLog("模型已保存：已导出JSON并同步到localStorage", true);
    overlay.textContent = "模型保存成功：已下载JSON并写入本地存储";
  }

  loadModelFromLocalStorage() {
    // 从localStorage恢复模型：用于快速继续训练，无需重新训练。
    const rawSnapshot = localStorage.getItem(STORAGE_KEYS.agentSnapshot);
    if (!rawSnapshot) {
      overlay.textContent = "未找到可加载模型：localStorage为空";
      this.appendDecisionLog("模型加载失败：localStorage中没有可用模型", true);
      return false;
    }

    try {
      const snapshot = JSON.parse(rawSnapshot);
      this.agent.loadSnapshot(snapshot);

      const rawStats = localStorage.getItem(STORAGE_KEYS.trainingStats);
      if (rawStats) {
        const stats = JSON.parse(rawStats);
        this.applyTrainingStats(stats);
      }

      refreshTrainingDashboard(this.trainingEpisodeHistory);
      this.reset();
      this.state = "running";
      overlay.textContent = "模型加载成功：已从localStorage恢复训练状态";
      this.appendDecisionLog("模型已从localStorage恢复，AI进入已训练状态", true);
      return true;
    } catch (_err) {
      overlay.textContent = "模型加载失败：localStorage数据格式错误";
      this.appendDecisionLog("模型加载失败：localStorage快照损坏", true);
      return false;
    }
  }

  loadModelFromJsonPayload(payload) {
    // 从JSON载荷恢复模型：兼容“完整导出格式”与“仅agentSnapshot格式”。
    if (!payload || typeof payload !== "object") return false;

    const snapshot = payload.agentSnapshot || payload;
    if (!snapshot.online || !snapshot.target) return false;

    this.agent.loadSnapshot(snapshot);

    const stats = payload.trainingStats;
    if (stats && typeof stats === "object") {
      this.applyTrainingStats(stats);
    }

    // 加载成功后立即刷新环境和看板，让AI直接以训练后状态运行。
    this.saveTrainingData();
    refreshTrainingDashboard(this.trainingEpisodeHistory);
    this.reset();
    this.state = "running";
    this.appendDecisionLog("模型已从JSON文件恢复，AI进入已训练状态", true);
    overlay.textContent = "模型加载成功：已从JSON文件恢复";
    return true;
  }

  finalizeEpisode(aiSuccess, survivalSteps, episodeReward) {
    // 回合结束统计：记录奖励、通关与生存步数。
    this.totalEpisodes += 1;
    this.trainingEpisodeIndex += 1;
    if (aiSuccess) this.successEpisodes += 1;

    // 将回合奖励反馈给Agent，用于“连续负回合强制探索”机制。
    this.agent.onEpisodeEnd(episodeReward);

    this.episodeRewards.push(episodeReward);
    if (this.episodeRewards.length > 150) this.episodeRewards.shift();

    this.trainingEpisodeHistory.push({
      episode: this.trainingEpisodeIndex,
      reward: episodeReward,
      success: aiSuccess ? 1 : 0,
      survivalSteps,
      actionCounts: [...this.currentEpisodeActionCounts],
    });
    if (this.trainingEpisodeHistory.length > 300) this.trainingEpisodeHistory.shift();

    // 回合结束后刷新图表：每回合只更新一次，避免影响主循环帧率。
    refreshTrainingDashboard(this.trainingEpisodeHistory);
    this.saveTrainingData();

    // 训练调试日志：每回合输出累计奖励、epsilon、loss和是否通关。
    console.log(
      `[TRAIN] ep=${this.trainingEpisodeIndex} reward=${episodeReward.toFixed(2)} epsilon=${this.agent.epsilon.toFixed(3)} loss=${this.lastLoss.toFixed(5)} success=${aiSuccess ? 1 : 0}`
    );

    // 每10回合打印一次权重变化量，验证网络确实在更新。
    if (this.trainingEpisodeIndex % 10 === 0) {
      const checksum = this.agent.getWeightChecksum();
      const delta = checksum - this.lastWeightChecksum;
      console.log(`[TRAIN] ep=${this.trainingEpisodeIndex} weight_checksum=${checksum.toFixed(3)} delta=${delta.toFixed(3)}`);
      this.lastWeightChecksum = checksum;
    }
  }

  clearTrainingData() {
    // 一键清空训练数据：统计、历史、Agent参数全部重置，用于重新开始训练。
    this.totalEpisodes = 0;
    this.successEpisodes = 0;
    this.episodeReward = 0;
    this.episodeRewards = [];
    this.trainingEpisodeIndex = 0;
    this.trainingEpisodeHistory = [];
    this.currentEpisodeActionCounts = [0, 0, 0, 0, 0];
    this.lastQValues = [0, 0, 0, 0, 0];
    this.lastStateVector = new Array(16).fill(0);
    this.lastLoss = 0;
    this.decisionLogs = [];
    this.lastDecisionMessage = "";
    this.decisionLogDirty = true;

    // 重建Agent，确保真正从“不会玩”重新学习。
    this.agent = new WebDQNAgent();

    localStorage.removeItem(STORAGE_KEYS.trainingStats);
    localStorage.removeItem(STORAGE_KEYS.agentSnapshot);

    // 立即持久化干净状态并刷新图表。
    this.saveTrainingData();
    refreshTrainingDashboard(this.trainingEpisodeHistory);

    // 保持当前模式不变，仅重置回合环境。
    this.reset();
    this.state = "running";
  }

  toggleTrainingMode() {
    // 两种模式自由切换，不修改已有手动玩法逻辑。
    this.autoTrainingMode = !this.autoTrainingMode;
    this.reset();
    this.state = "running";
    this.saveTrainingData();
  }

  cycleTrainingSpeed() {
    // 训练速度倍率循环：1x -> 2x -> 4x -> 1x。
    if (this.trainingSpeed === 1) this.trainingSpeed = 2;
    else if (this.trainingSpeed === 2) this.trainingSpeed = 4;
    else this.trainingSpeed = 1;
    this.saveTrainingData();
  }

  spawnBurst(x, y, color, count = 12, speed = 2.2, size = 2.2, life = 28) {
    for (let i = 0; i < count; i++) {
      const angle = randomFloat(0, Math.PI * 2);
      const magnitude = randomFloat(0.4, speed);
      const vx = Math.cos(angle) * magnitude;
      const vy = Math.sin(angle) * magnitude - 1.0;

      this.particles.push(
        new Particle(
          x,
          y,
          color,
          vx,
          vy,
          Math.floor(randomFloat(life * 0.6, life)),
          randomFloat(size * 0.7, size * 1.35)
        )
      );
    }

    if (this.particles.length > this.maxParticles) {
      this.particles.splice(0, this.particles.length - this.maxParticles);
    }
  }

  updateParticles() {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      this.particles[i].update();
      if (this.particles[i].life <= 0) this.particles.splice(i, 1);
    }
  }

  drawParticles(cameraX) {
    for (const p of this.particles) p.draw(cameraX);
  }

  handlePlayerInput() {
    this.player.velX = 0;
    if (keys.has("ArrowLeft")) this.player.velX = -MOVE_SPEED;
    if (keys.has("ArrowRight")) this.player.velX = MOVE_SPEED;

    if (keys.has("ArrowUp") && this.player.onGround) {
      this.player.velY = JUMP_FORCE;
      this.player.onGround = false;

      // 跳跃粒子：脚底扬尘，不影响任何物理判定。
      this.spawnBurst(this.player.x + this.player.w / 2, this.player.y + this.player.h, "#88c8ff", 8, 1.5, 1.8, 18);
    }
  }

  updateCamera() {
    // 自动训练时跟随AI，手动模式跟随玩家。
    const targetCharacter = this.autoTrainingMode ? this.ai : this.player;
    const targetCameraX = targetCharacter.x + targetCharacter.w / 2 - SCREEN_WIDTH / 2;
    const maxCameraX = Math.max(0, this.level.worldWidth - SCREEN_WIDTH);
    const clampedTarget = Math.max(0, Math.min(targetCameraX, maxCameraX));
    this.cameraX += (clampedTarget - this.cameraX) * this.cameraLerp;
    this.cameraX = Math.max(0, Math.min(this.cameraX, maxCameraX));
  }

  update() {
    if (this.state !== "running") {
      // 手动模式回合自动续跑：
      // 游戏结束后进入短暂倒计时，倒计时到0后自动重置并进入下一回合，
      // 从而持续产生episode数据，让AI在多回合中逐步变强。
      if (!this.autoTrainingMode && this.state === "game_over" && this.manualNextEpisodeCountdown > 0) {
        this.manualNextEpisodeCountdown -= 1;
        if (this.manualNextEpisodeCountdown <= 0) {
          this.reset();
          this.state = "running";
          this.appendDecisionLog("自动进入下一回合：继续训练", true);
        }
      }

      this.updateParticles();
      return;
    }

    const prevPlayerAlive = this.player.isAlive;
    const prevPlayerGoal = this.player.reachedGoal;
    const prevPlayerScore = this.player.score;

    const prevAiAlive = this.ai.isAlive;
    const prevAiGoal = this.ai.reachedGoal;
    const prevAiScore = this.ai.score;

    // 手动模式保留原玩家逻辑；自动训练模式隐藏玩家并跳过玩家输入与更新。
    if (!this.autoTrainingMode) {
      this.handlePlayerInput();
      this.player.applyPhysics(this.level);
    }

    const oldState = this.agent.get_state(this, this.ai);
    // 记录当前状态向量，供Tab2实时展示16维状态值。
    this.lastStateVector = [...oldState];

    // 计算动作Q值：用于Tab2和Tab3展示实时决策依据。
    this.lastQValues = this.agent._forward(this.agent.online, oldState).q;

    let action = this.agent.select_action(oldState);

    // 教师引导注入：
    // 在训练早期按衰减概率覆盖一次动作，让AI有机会观察到“更像通关策略”的行为轨迹。
    const teacherAction = this.getTeacherAction();
    this.lastTeacherAction = teacherAction;
    const teacherOverrideProbability = this.getTeacherOverrideProbability();
    if (this.autoTrainingMode && Math.random() < teacherOverrideProbability) {
      action = teacherAction;
      this.agent.lastActionWasJump = action === 2 || action === 3 || action === 4;
    }

    this.aiAction = action;
    this.currentEpisodeActionCounts[action] += 1;

    const aiJumpingAction = action === 2 || action === 3 || action === 4;
    if (aiJumpingAction && this.ai.onGround) {
      this.spawnBurst(this.ai.x + this.ai.w / 2, this.ai.y + this.ai.h, "#cb88ff", 8, 1.5, 1.8, 18);
    }

    this.agent.execute_action(this.ai, action);

    // 动作执行日志：实时输出动作和on_ground状态，便于排查“不会跳”问题。
    console.debug(`[ACTION] step=${this.agent.episodeSteps} action=${ACTION_NAMES[action]} on_ground=${this.ai.onGround}`);

    this.ai.applyPhysics(this.level);

    // 每帧写入可解释决策日志（去重后落盘到面板），实时展示AI“为何这么选”。
    this.appendDecisionLog(this.describeDecision(action));

    const nextState = this.agent.get_state(this, this.ai);
    const rewardResult = this.agent.get_reward(this, this.ai, oldState);

    // 手动模式“竞速结果”奖励修正：
    // 当玩家先到终点而AI尚未完成时，本回合会被界面判定为结束。
    // 若不在这里补一个终止惩罚，AI很难学到“输掉回合”的代价。
    // 该修正仅改变可视化玩法下的奖励信号，不改DQN结构本身。
    if (!this.autoTrainingMode && this.player.reachedGoal && !this.ai.reachedGoal && !rewardResult.done) {
      rewardResult.reward -= 80;
      rewardResult.done = true;
    }

    // 对称地，当玩家死亡但AI仍存活时，给AI一个小额终止奖励，强化“活下来更好”的策略。
    if (!this.autoTrainingMode && !this.player.isAlive && this.ai.isAlive && !rewardResult.done) {
      rewardResult.reward += 20;
      rewardResult.done = true;
    }

    // 教师一致性奖励：自动训练且教师可用时，动作与教师建议一致给小额奖励，
    // 用于加速前期学习；不一致给极小惩罚，避免策略发散。
    if (this.autoTrainingMode && teacherOverrideProbability > 0) {
      if (action === teacherAction) rewardResult.reward += this.teacherImitationBonus;
      else rewardResult.reward -= this.teacherImitationBonus * 0.25;
    }

    this.aiReward = rewardResult.reward;

    this.agent.store_memory(oldState, action, rewardResult.reward, nextState, rewardResult.done);
    const loss = this.agent.update();
    if (loss !== null) this.lastLoss = loss;

    // 累计回合奖励：用于展示单回合学习效果。
    this.episodeReward += rewardResult.reward;

    if (this.player.score > prevPlayerScore) {
      this.spawnBurst(this.player.x + this.player.w / 2, this.player.y + 8, "#ffe066", 16, 2.1, 2.2, 24);
    }
    if (this.ai.score > prevAiScore) {
      this.spawnBurst(this.ai.x + this.ai.w / 2, this.ai.y + 8, "#ffe066", 16, 2.1, 2.2, 24);
    }

    if (prevPlayerAlive && !this.player.isAlive) {
      this.spawnBurst(this.player.x + this.player.w / 2, this.player.y + this.player.h / 2, "#ff5f5f", 26, 2.8, 2.5, 30);
    }
    if (prevAiAlive && !this.ai.isAlive) {
      this.spawnBurst(this.ai.x + this.ai.w / 2, this.ai.y + this.ai.h / 2, "#ff5f5f", 26, 2.8, 2.5, 30);
    }

    if (!prevPlayerGoal && this.player.reachedGoal) {
      this.spawnBurst(this.player.x + this.player.w / 2, this.player.y + this.player.h / 2, "#8aff9c", 34, 3.0, 2.7, 34);
    }
    if (!prevAiGoal && this.ai.reachedGoal) {
      this.spawnBurst(this.ai.x + this.ai.w / 2, this.ai.y + this.ai.h / 2, "#8aff9c", 34, 3.0, 2.7, 34);
    }

    this.updateParticles();
    this.updateCamera();

    const pWin = this.player.reachedGoal;
    const aWin = this.ai.reachedGoal;
    const pDead = !this.player.isAlive;
    const aDead = !this.ai.isAlive;

    if (pWin && !aWin) {
      this.state = "game_over";
      this.resultText = "玩家获胜";
    } else if (aWin && !pWin) {
      this.state = "game_over";
      this.resultText = "AI获胜";
    } else if (pWin && aWin) {
      this.state = "game_over";
      this.resultText = "平局：双方通关";
    } else if (pDead && aDead) {
      this.state = "game_over";
      this.resultText = "平局：双方死亡";
    }

    // 手动模式补充终止判定：
    // 超时或卡住虽然会让rewardResult.done=true，但不一定触发上面的胜负分支。
    // 这里统一将环境终止事件映射为game_over，保证回合总能收尾。
    if (!this.autoTrainingMode && this.state === "running" && rewardResult.done) {
      this.state = "game_over";
      if (rewardResult.endReason === "timeout") this.resultText = "回合结束：AI超时";
      else if (rewardResult.endReason === "stuck") this.resultText = "回合结束：AI卡住";
      else if (rewardResult.endReason === "death") this.resultText = "回合结束：AI死亡";
      else if (rewardResult.endReason === "goal") this.resultText = "回合结束：AI通关";
      else this.resultText = "回合结束";
    }

    // 自动训练模式下：
    // 1) 只要AI回合结束（通关/死亡/超时）就自动记录并重开；
    // 2) 达到最大回合数后自动停止在结束页。
    if (this.autoTrainingMode && rewardResult.done) {
      const aiSuccess = this.ai.reachedGoal;
      const survivalSteps = rewardResult.survivalSteps;
      const episodeRewardSnapshot = this.episodeReward;
      this.finalizeEpisode(aiSuccess, survivalSteps, episodeRewardSnapshot);
      this.episodeReward = 0;

      if (this.trainingEpisodeIndex >= this.maxTrainingEpisodes) {
        this.state = "game_over";
        this.resultText = `训练完成：${this.trainingEpisodeIndex}/${this.maxTrainingEpisodes}`;
      } else {
        this.reset();
        this.state = "running";
      }
    }

    // 手动模式保持原结束行为：回合结束后记录一次统计。
    if (!this.autoTrainingMode && this.state === "game_over") {
      this.finalizeEpisode(aWin, rewardResult.survivalSteps, this.episodeReward);
      this.episodeReward = 0;

      // 手动模式同样按“回合制训练”推进：结束后自动进入下一回合，避免停在game_over不再学习。
      this.manualNextEpisodeCountdown = MANUAL_NEXT_EPISODE_DELAY_FRAMES;
    }

    // 回合结束事件作为关键节点强制记录，保证日志中可见成功/失败转折。
    if (rewardResult.done) {
      if (this.ai.reachedGoal) this.appendDecisionLog("关键事件：AI成功通关", true);
      else if (!this.ai.isAlive) this.appendDecisionLog("关键事件：AI触发危险导致回合终止", true);
      else if (rewardResult.endReason === "stuck") this.appendDecisionLog("关键事件：AI长时间卡住，回合强制终止", true);
      else if (rewardResult.endReason === "timeout") this.appendDecisionLog("关键事件：AI超过时间上限，回合终止", true);
      else this.appendDecisionLog("关键事件：回合结束（环境终止）", true);
    }
  }

  drawText(text, x, y, color = BLACK, size = 20, align = "left") {
    ctx.fillStyle = color;
    ctx.font = `${size}px Segoe UI`;
    ctx.textAlign = align;
    ctx.fillText(text, x, y);
    ctx.textAlign = "left";
  }

  drawParallaxBackground(time = 0) {
    const bg = ctx.createLinearGradient(0, 0, 0, SCREEN_HEIGHT);
    bg.addColorStop(0, "#0a0f1f");
    bg.addColorStop(0.55, "#131a34");
    bg.addColorStop(1, "#1c2441");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

    for (const star of this.starsFar) {
      const sx = ((star.x - this.cameraX * 0.2) % this.level.worldWidth + this.level.worldWidth) % this.level.worldWidth;
      if (sx < -10 || sx > SCREEN_WIDTH + 10) continue;
      const twinkle = 0.45 + 0.55 * Math.sin(time * 0.002 + star.tw);
      ctx.fillStyle = `rgba(190,210,255,${twinkle * 0.6})`;
      ctx.beginPath();
      ctx.arc(sx, star.y, star.r, 0, Math.PI * 2);
      ctx.fill();
    }

    for (const star of this.starsNear) {
      const sx = ((star.x - this.cameraX * 0.35) % this.level.worldWidth + this.level.worldWidth) % this.level.worldWidth;
      if (sx < -10 || sx > SCREEN_WIDTH + 10) continue;
      const twinkle = 0.55 + 0.45 * Math.sin(time * 0.003 + star.tw);
      ctx.fillStyle = `rgba(235,245,255,${twinkle * 0.75})`;
      ctx.beginPath();
      ctx.arc(sx, star.y, star.r, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.save();
    ctx.fillStyle = "rgba(80, 95, 140, 0.38)";
    ctx.beginPath();
    ctx.moveTo(0, SCREEN_HEIGHT);
    for (let x = 0; x <= SCREEN_WIDTH + 20; x += 40) {
      const wx = x + this.cameraX * 0.25;
      const y = 250 + Math.sin(wx * 0.01) * 18 + Math.sin(wx * 0.004) * 24;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(SCREEN_WIDTH, SCREEN_HEIGHT);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    ctx.save();
    ctx.fillStyle = "rgba(56, 72, 116, 0.58)";
    ctx.beginPath();
    ctx.moveTo(0, SCREEN_HEIGHT);
    for (let x = 0; x <= SCREEN_WIDTH + 20; x += 30) {
      const wx = x + this.cameraX * 0.45;
      const y = 285 + Math.sin(wx * 0.016) * 18 + Math.cos(wx * 0.007) * 15;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(SCREEN_WIDTH, SCREEN_HEIGHT);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  drawFullScreenPanel(title, subtitleLines) {
    const panelGrad = ctx.createLinearGradient(0, 0, 0, SCREEN_HEIGHT);
    panelGrad.addColorStop(0, "rgba(6, 10, 24, 0.76)");
    panelGrad.addColorStop(1, "rgba(6, 10, 24, 0.87)");
    ctx.fillStyle = panelGrad;
    ctx.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

    ctx.save();
    ctx.shadowColor = "rgba(106, 166, 255, 0.55)";
    ctx.shadowBlur = 20;
    this.drawText(title, SCREEN_WIDTH / 2, 132, "#eaf0ff", 42, "center");
    ctx.restore();

    let y = 182;
    for (const line of subtitleLines) {
      this.drawText(line, SCREEN_WIDTH / 2, y, "#c8d6ff", 20, "center");
      y += 32;
    }
  }

  draw(time = 0) {
    this.drawParallaxBackground(time);

    if (this.state === "running" || this.state === "game_over") {
      this.level.draw(this.cameraX, time);
      this.drawParticles(this.cameraX);
      if (!this.autoTrainingMode) {
        // 自动训练模式下隐藏玩家角色，仅展示AI与环境交互。
        this.player.draw(this.cameraX, time);
      }
      this.ai.draw(this.cameraX, time);
    }

    if (this.state === "start") {
      this.drawFullScreenPanel("CDS524 Web Platformer Race", [
        "← / → 移动，↑ 跳跃",
        "按 ↑ 或 Y 开始游戏",
        "按 ↓ 或 R 重开，ESC 返回开始页",
      ]);
      overlay.textContent = this.focused
        ? "画布已聚焦，可直接键盘操作"
        : "请先点击游戏画布激活键盘输入";
    }

    if (this.state === "running") {
      overlay.textContent = this.autoTrainingMode
        ? `自动训练中：回合 ${this.trainingEpisodeIndex} / ${this.maxTrainingEpisodes}（${this.trainingSpeed}x） | 关卡=${this.levelMode} | 教师引导=${this.getTeacherOverrideProbability().toFixed(2)}`
        : `游戏进行中（AI每帧决策中） | 关卡=${this.levelMode}`;
    }

    if (this.state === "game_over") {
      const autoNextLine = !this.autoTrainingMode && this.manualNextEpisodeCountdown > 0
        ? `自动下一回合：${Math.max(0, (this.manualNextEpisodeCountdown / FPS)).toFixed(1)}s`
        : "按 ↓ / R 重开";

      this.drawFullScreenPanel("Game Over", [
        this.resultText,
        autoNextLine,
        "按 ESC 返回开始页",
      ]);
      // 回合结束提示：若已启用自动下一回合，则明确告知仍在持续训练。
      overlay.textContent = !this.autoTrainingMode && this.manualNextEpisodeCountdown > 0
        ? "回合结束：即将自动进入下一回合"
        : "回合结束";
    }

    this.updateHud();
  }

  updateHud() {
    const gp = ((this.player.x / Math.max(this.level.goal.x, 1)) * 100).toFixed(1);
    const ap = ((this.ai.x / Math.max(this.level.goal.x, 1)) * 100).toFixed(1);

    hud.state.textContent = this.state;
    hud.playerScore.textContent = String(this.player.score);
    hud.playerProgress.textContent = `${gp}%`;
    hud.aiProgress.textContent = `${ap}%`;
    hud.aiAction.textContent = ACTION_NAMES[this.aiAction] || "RIGHT";
    hud.aiReward.textContent = `${this.aiReward >= 0 ? "+" : ""}${this.aiReward.toFixed(3)}`;
    hud.playerAlive.textContent = String(this.player.isAlive);
    hud.aiAlive.textContent = String(this.ai.isAlive);

    // 顶部训练模式信息同步：
    // - 模式按钮显示当前模式；
    // - 速度按钮显示当前倍率；
    // - 回合信息显示当前回合/总回合。
    uiControls.btnMode.textContent = this.autoTrainingMode
      ? "切换训练模式：自动训练"
      : "切换训练模式：手动对战";
    uiControls.btnSpeed.textContent = `训练速度：${this.trainingSpeed}x`;
    uiControls.trainingInfo.textContent = `回合 ${this.trainingEpisodeIndex} / ${this.maxTrainingEpisodes}`;

    // 允许用户在运行中调整最大训练回合数。
    const uiMaxValue = Number(uiControls.maxEpisodes.value);
    if (Number.isFinite(uiMaxValue) && uiMaxValue > 0) {
      this.maxTrainingEpisodes = Math.floor(uiMaxValue);
    }

    // -----------------------
    // Tab2：强化学习核心参数
    // -----------------------
    const epsilonRange = this.agent.epsilon - this.agent.epsilonMin;
    const epsilonMaxRange = 0.99 - this.agent.epsilonMin;
    const decayProgress = epsilonMaxRange > 0 ? ((epsilonMaxRange - Math.max(0, epsilonRange)) / epsilonMaxRange) * 100 : 0;

    rlHud.epsilon.textContent = this.agent.epsilon.toFixed(3);
    rlHud.epsilonPct.textContent = `${decayProgress.toFixed(1)}%`;
    rlHud.stepReward.textContent = `${this.aiReward >= 0 ? "+" : ""}${this.aiReward.toFixed(3)}`;
    rlHud.episodeReward.textContent = this.episodeReward.toFixed(3);

    const avgReward = this.episodeRewards.length > 0
      ? this.episodeRewards.reduce((a, b) => a + b, 0) / this.episodeRewards.length
      : 0;
    rlHud.avgReward.textContent = avgReward.toFixed(3);

    rlHud.episodeSteps.textContent = String(this.agent.episodeSteps);
    rlHud.totalSteps.textContent = String(this.agent.stepCount);
    rlHud.successRatio.textContent = `${this.successEpisodes} / ${this.totalEpisodes}`;

    rlHud.q0.textContent = this.lastQValues[0].toFixed(3);
    rlHud.q1.textContent = this.lastQValues[1].toFixed(3);
    rlHud.q2.textContent = this.lastQValues[2].toFixed(3);
    rlHud.q3.textContent = this.lastQValues[3].toFixed(3);
    rlHud.q4.textContent = this.lastQValues[4].toFixed(3);

    // 组件1：Q值实时条形图（每帧更新，并高亮最大Q对应动作）。
    updateQBarPanel(this.lastQValues);

    // 组件2：16维状态向量热力面板（每帧更新，颜色表示特征激活强度）。
    updateStateHeatmap(this.lastStateVector);

    // 训练看板顶部实时奖励数字：不用看曲线也能看到当前回合奖励变化。
    if (rlHud.dashboardEpisodeReward) {
      rlHud.dashboardEpisodeReward.textContent = this.episodeReward.toFixed(3);
    }
    if (rlHud.dashboardSuccessRate) {
      const liveSuccessRate = this.totalEpisodes > 0 ? (this.successEpisodes / this.totalEpisodes) * 100 : 0;
      rlHud.dashboardSuccessRate.textContent = `${liveSuccessRate.toFixed(1)}%`;
    }
    if (rlHud.dashboardTotalEpisodes) {
      rlHud.dashboardTotalEpisodes.textContent = String(this.totalEpisodes);
    }

    // 保留字符串状态输出（隐藏DOM，仅作兼容保底调试）。
    rlHud.stateVector.textContent = `[${this.lastStateVector.map((v) => v.toFixed(3)).join(", ")}]`;

    // 底部决策日志：仅在日志变化时重绘DOM，既满足实时性又避免不必要重排。
    if (this.decisionLogDirty) {
      renderDecisionLog(this.decisionLogs);
      this.decisionLogDirty = false;
    }

    // 训练看板图表改为“回合结束时刷新”，此处不做逐帧图表更新，确保渲染稳定。
  }
}

const game = new Game();

window.addEventListener("keydown", (e) => {
  if (blockedKeys.has(e.code)) e.preventDefault();
  keys.add(e.code);

  if (e.code === "Escape") {
    game.state = "start";
    game.reset();
  }

  if ((e.code === "ArrowUp" || e.code === "KeyY") && game.state === "start") {
    game.startGame();
  }

  if (e.code === "ArrowDown" || e.code === "KeyR") {
    game.startGame();
  }
});

window.addEventListener("keyup", (e) => {
  if (blockedKeys.has(e.code)) e.preventDefault();
  keys.delete(e.code);
});

canvas.addEventListener("click", () => {
  canvas.focus();
  game.focused = true;
});

document.getElementById("btnStart").addEventListener("click", () => game.startGame());
document.getElementById("btnRestart").addEventListener("click", () => game.startGame());
if (uiControls.btnSaveModel) {
  // 点击“保存模型”：导出JSON文件并同步到localStorage。
  uiControls.btnSaveModel.addEventListener("click", () => game.saveModelSnapshot());
}
if (uiControls.btnLoadModel) {
  // 点击“加载模型”：
  // - 确认使用localStorage时，直接快速恢复；
  // - 否则打开文件选择器，允许导入离线JSON模型文件。
  uiControls.btnLoadModel.addEventListener("click", () => {
    const useLocalStorage = window.confirm("点击“确定”：从localStorage加载模型\n点击“取消”：选择本地JSON模型文件");
    if (useLocalStorage) {
      game.loadModelFromLocalStorage();
      return;
    }

    if (uiControls.modelFileInput) {
      uiControls.modelFileInput.value = "";
      uiControls.modelFileInput.click();
    }
  });
}

if (uiControls.modelFileInput) {
  // 文件导入流程：读取JSON文本 -> 解析 -> 校验 -> 恢复模型。
  uiControls.modelFileInput.addEventListener("change", () => {
    const file = uiControls.modelFileInput.files && uiControls.modelFileInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = JSON.parse(String(reader.result || "{}"));
        const loaded = game.loadModelFromJsonPayload(payload);
        if (!loaded) {
          overlay.textContent = "模型加载失败：文件内容不符合模型格式";
          game.appendDecisionLog("模型加载失败：JSON结构无效", true);
        }
      } catch (_err) {
        overlay.textContent = "模型加载失败：JSON解析错误";
        game.appendDecisionLog("模型加载失败：JSON解析异常", true);
      }
    };

    reader.onerror = () => {
      overlay.textContent = "模型加载失败：文件读取异常";
      game.appendDecisionLog("模型加载失败：读取文件出错", true);
    };

    reader.readAsText(file, "utf-8");
  });
}
uiControls.btnMode.addEventListener("click", () => game.toggleTrainingMode());
uiControls.btnSpeed.addEventListener("click", () => game.cycleTrainingSpeed());
if (uiControls.btnClearTrainingData) {
  // 清空按钮仅重置训练数据，不删除游戏UI配置。
  uiControls.btnClearTrainingData.addEventListener("click", () => game.clearTrainingData());
}
uiControls.maxEpisodes.addEventListener("change", () => {
  const value = Number(uiControls.maxEpisodes.value);
  if (Number.isFinite(value) && value > 0) {
    game.maxTrainingEpisodes = Math.floor(value);
    game.saveTrainingData();
  }
});

let lastTime = 0;
function loop(ts) {
  if (ts - lastTime >= 1000 / FPS) {
    // 自动训练模式支持加速：同一渲染帧内执行多次逻辑更新。
    const updateTimes = game.autoTrainingMode ? game.trainingSpeed : 1;
    for (let i = 0; i < updateTimes; i++) {
      game.update();
    }
    game.draw(ts);
    lastTime = ts;
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
