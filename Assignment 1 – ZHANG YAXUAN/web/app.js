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

// 性能参数：降低主线程负载，减少训练+渲染并发导致的卡顿。
const PERF_CONFIG = {
  hudHeavyUpdateStride: 4,
  autoRenderStride2x: 2,
  autoRenderStride4x: 4,
  autoRenderStride8x: 8,
  autoRenderStrideLite8x: 12,
};

// 代码级加速配置（无需点击UI按钮）
const CODE_ACCEL_PROFILE = {
  enabled: true,
  forceAutoTrainSpeed: 8,
  forceLiteUi: true,
  dashboardUpdateEpisodeStride: 20,
  muteVerboseDecisionLogs: true,
};

// 角色边界判定参数：
// - 跳到屏幕上方不再判死（允许高跳后自然落回）；
// - 只有明显坠落到屏幕下方一定距离才算出界死亡。
const FALL_DEATH_MARGIN = 80;

// 手动对战模式下的自动下一回合延时（帧）：
// 回合结束后短暂停留结果页，再自动进入下一回合，避免“卡在结束页不训练”。
const MANUAL_NEXT_EPISODE_DELAY_FRAMES = Math.floor(FPS * 1.2);

// 17维状态向量语义标签：在原16维基础上新增“前方最近陷阱宽度”。
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
  "前方最近陷阱宽度",
  "前方最近平台X",
  "前方最近平台Y",
  "最近金币-X",
  "最近金币-Y",
  "当前地图进度",
  "历史最高进度",
];

// 状态维度统一常量：避免代码里散落硬编码16，后续扩维时更安全。
const STATE_DIM = STATE_DIMENSION_LABELS.length;

// 回合强制终止阈值：用于杜绝“掉回地面后无限循环”。
const EPISODE_LIMITS = {
  maxFrames: 60 * FPS,
  noProgressTerminateFrames: 300,
  noProgressPenaltyStartFrames: 100,
  loopTerminateFrames: 200,
  ineffectiveActionTerminateFrames: 200,
};

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
  remainTime: document.getElementById("hudRemainTime"),
  aiMaxProgress: document.getElementById("hudAiMaxProgress"),
  aiBehavior: document.getElementById("hudAiBehavior"),
  mapSignature: document.getElementById("hudMapSignature"),
};

// 自动训练模式UI控件：用于模式切换、速度控制、最大回合设置与进度展示。
const uiControls = {
  btnSaveModel: document.getElementById("btnSaveModel"),
  btnLoadModel: document.getElementById("btnLoadModel"),
  btnBackToStart: document.getElementById("btnBackToStart"),
  modelFileInput: document.getElementById("modelFileInput"),
  btnMode: document.getElementById("btnMode"),
  speedLabel: document.getElementById("speedLabel"),
  speedSelect: document.getElementById("trainSpeedSelect"),
  toggleLiteUi: document.getElementById("toggleLiteUi"),
  btnExportRequiredShots: document.getElementById("btnExportRequiredShots"),
  btnExportReportData: document.getElementById("btnExportReportData"),
  btnExportRewardPng: document.getElementById("btnExportRewardPng"),
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
  loss: document.getElementById("rlLoss"),
  weightDelta: document.getElementById("rlWeightDelta"),
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
  dashboardRecent50Rate: document.getElementById("dashboardRecent50Rate"),
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

// 根据17维状态值刷新热力面板：值越大背景越亮，帮助快速识别高激活特征。
function updateStateHeatmap(stateVector) {
  if (!stateCellEls.length) return;
  const safeState = Array.isArray(stateVector) ? stateVector : new Array(STATE_DIM).fill(0);

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
const STAGE_SNAPSHOT_EPISODES = [100, 200, 300];

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
  // 通关率计算修复：
  // 规则必须为“最近windowSize回合内成功数 / 最近windowSize内总回合数”。
  // 当history为空时返回空数组；显示基线由外层统一补0，避免长度错位。
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
  // 图表起点修复：第0回合强制显示0，避免“首回合直接100%”视觉误导。
  const labels = ["0", ...safeHistory.map((entry) => String(entry.episode))];
  const episodeRewards = safeHistory.map((entry) => Number(entry.reward || 0));
  const episodeMovingAvg = calculateMovingAverage(episodeRewards, REWARD_MOVING_AVG_WINDOW);
  const rewards = [0, ...episodeRewards];
  const movingAvg = [0, ...episodeMovingAvg];
  const passRates = buildPassRateSeries(safeHistory, PASS_RATE_WINDOW);
  const passLabels = labels;
  const passRatesWithBaseline = [0, ...passRates];
  const actionDist = buildActionDistribution(safeHistory, ACTION_DIST_WINDOW);

  if (rewardChart) {
    // 显示优化：奖励曲线Y轴按当前数据自适应，避免高奖励阶段被固定上限截断。
    const finiteValues = rewards.concat(movingAvg).filter((v) => Number.isFinite(v));
    if (finiteValues.length > 0) {
      const minValue = Math.min(...finiteValues);
      const maxValue = Math.max(...finiteValues);
      const spread = Math.max(1, maxValue - minValue);
      const padding = Math.max(80, spread * 0.12);
      let yMin = Math.floor((minValue - padding) / 50) * 50;
      let yMax = Math.ceil((maxValue + padding) / 50) * 50;
      if (yMin === yMax) yMax = yMin + 100;
      rewardChart.options.scales.y.min = yMin;
      rewardChart.options.scales.y.max = yMax;
    }

    rewardChart.data.labels = labels;
    rewardChart.data.datasets[0].data = rewards;
    rewardChart.data.datasets[1].data = movingAvg;
    rewardChart.update("none");
  }

  if (passRateChart) {
    passRateChart.data.labels = passLabels;
    passRateChart.data.datasets[0].data = passRatesWithBaseline;
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
          label: `全回合通关率`,
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

    // 先记录移动前的垂直边界，供“落地缓冲判定”使用。
    const prevTop = this.y;
    const prevBottom = this.y + this.h;

    this.velY += GRAVITY;

    // -----------------------
    // 水平移动与碰撞（先水平）
    // -----------------------
    this.x += this.velX;
    for (const p of level.platforms) {
      if (intersects(this.rect, p.rect)) {
        if (this.velX > 0) this.x = p.x - this.w;
        if (this.velX < 0) this.x = p.x + p.w;
        this.velX = 0;
      }
    }

    // -----------------------
    // 垂直移动与碰撞（后垂直）
    // -----------------------
    this.y += this.velY;
    this.onGround = false;
    const landTolerance = 5;

    for (const p of level.platforms) {
      const horizontalOverlap = this.x + this.w > p.x && this.x < p.x + p.w;
      if (!horizontalOverlap) continue;

      // 落地判定：角色向下、上一帧底边在平台顶附近、这一帧穿过平台顶。
      const crossedTop = prevBottom <= p.y + landTolerance && this.y + this.h >= p.y;
      if (this.velY >= 0 && crossedTop) {
        this.y = p.y - this.h;
        this.velY = 0;
        this.onGround = true;
        continue;
      }

      // 顶碰判定：角色向上、上一帧顶边在平台底附近、这一帧穿过平台底。
      const crossedBottom = prevTop >= p.y + p.h - landTolerance && this.y <= p.y + p.h;
      if (this.velY < 0 && crossedBottom) {
        this.y = p.y + p.h;
        this.velY = 0;
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
  constructor(mode = "classic", episodeNumber = 1, totalEpisodes = 300) {
    this.worldWidth = 1500;
    this.mode = mode;
    this.episodeNumber = Math.max(1, Number(episodeNumber || 1));
    this.totalEpisodes = Math.max(1, Number(totalEpisodes || 300));
    this.randomizationMeta = {
      difficulty: "低难度",
      spikeXJitter: 0,
      platformGapJitter: 0,
      platformYJitter: 0,
      coinPosJitter: 0,
      avgPlatformGap: 0,
      firstSpikeX: "none",
      extraSmallSpike: 0,
    };

    // 关卡模式工厂：每个模式定义平台/陷阱/金币/终点，
    // 再经过轻微随机扰动，避免AI只背固定路径。
    const spec = this._buildModeSpec(mode, this.episodeNumber, this.totalEpisodes);
    this.platforms = spec.platforms.map((obj) => new Entity(obj.x, obj.y, obj.w, obj.h, GREEN));
    this.spikes = spec.spikes.map((obj) => new Entity(obj.x, obj.y, obj.w, obj.h, RED));
    this.coins = spec.coins.map((obj) => new Entity(obj.x, obj.y, obj.w, obj.h, YELLOW));
    this.goal = new Entity(spec.goal.x, spec.goal.y, spec.goal.w, spec.goal.h, ORANGE);
  }

  getSpawnPoint() {
    // 出生点规则（无地面模式）：
    // 始终在最左侧起始平台上方生成，保证“开局就在一块板子上”。
    const sorted = [...this.platforms].sort((a, b) => a.x - b.x);
    const startPlatform = sorted.length ? sorted[0] : new Entity(30, SCREEN_HEIGHT - 80, 120, 16, GREEN);
    return {
      x: startPlatform.x + 12,
      y: startPlatform.y - PLAYER_SIZE,
    };
  }

  static get MODE_NAMES() {
    // 训练关卡集合：
    // tutorial仅用于前期课程学习，帮助AI先学会“向右推进+基础跳跃”。
    return ["tutorial", "classic", "zigzag", "gaps"];
  }

  static getDifficultyStage(episodeNumber, totalEpisodes = 300) {
    // 难度阶段定义（按总回合百分比）：
    // - 前35%：低难度
    // - 35%~75%：中难度
    // - 后25%：高难度
    const maxEpisodes = Math.max(1, Number(totalEpisodes || 300));
    const progress = Math.min(1, Math.max(0, Number(episodeNumber || 1) / maxEpisodes));
    if (progress <= 0.35) return "low";
    if (progress <= 0.75) return "mid";
    return "high";
  }

  _buildTutorialSpec() {
    // 新手训练图（简化版）：
    // - 平台更宽、上升更平缓；
    // - 不放置地面尖刺；
    // - 终点更容易抵达；
    // 目的：让DQN在早期快速获得“可通关正反馈”，打破长期0通关。
    return {
      platforms: [
        { x: 20, y: 342, w: 150, h: 16 },
        { x: 170, y: 318, w: 175, h: 16 },
        { x: 380, y: 290, w: 165, h: 16 },
        { x: 590, y: 262, w: 160, h: 16 },
        { x: 790, y: 234, w: 155, h: 16 },
        { x: 980, y: 205, w: 150, h: 16 },
        { x: 1160, y: 175, w: 145, h: 16 },
        { x: 1315, y: 145, w: 135, h: 16 },
      ],
      spikes: [],
      coins: [
        { x: 250, y: 294, w: 14, h: 14 },
        { x: 455, y: 266, w: 14, h: 14 },
        { x: 665, y: 238, w: 14, h: 14 },
        { x: 860, y: 210, w: 14, h: 14 },
        { x: 1045, y: 181, w: 14, h: 14 },
        { x: 1225, y: 151, w: 14, h: 14 },
      ],
      goal: { x: 1392, y: 108, w: 20, h: 55 },
    };
  }

  _jitter(value, range) {
    return value + randomFloat(-range, range);
  }

  _buildClassicSpec() {
    return {
      platforms: [
        { x: 20, y: 338, w: 145, h: 16 },
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
        { x: 20, y: 344, w: 145, h: 16 },
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
        { x: 20, y: 346, w: 140, h: 16 },
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

  _buildModeSpec(mode, episodeNumber, totalEpisodes) {
    let baseSpec;
    if (mode === "tutorial") baseSpec = this._buildTutorialSpec();
    else if (mode === "zigzag") baseSpec = this._buildZigzagSpec();
    else if (mode === "gaps") baseSpec = this._buildGapsSpec();
    else baseSpec = this._buildClassicSpec();

    return this._randomizeWithinDifficulty(baseSpec, episodeNumber, totalEpisodes);
  }

  _clamp(value, minValue, maxValue) {
    return Math.max(minValue, Math.min(maxValue, value));
  }

  _isSpikeAbovePlatform(spike, platforms) {
    // 限制：陷阱不能生成在平台正上方，避免“必死点”。
    return platforms.some((p) => {
      const overlapX = spike.x < p.x + p.w && spike.x + spike.w > p.x;
      const abovePlatform = spike.y + spike.h <= p.y + 2;
      return overlapX && abovePlatform;
    });
  }

  _isCoinReachable(coin, platforms, spikes) {
    const coinRect = { x: coin.x, y: coin.y, w: coin.w, h: coin.h };
    const coinCenterX = coin.x + coin.w * 0.5;
    const coinBottom = coin.y + coin.h;

    const supportPlatform = platforms.find((platform) => {
      const withinPlatformX = coinCenterX >= platform.x - 4 && coinCenterX <= platform.x + platform.w + 4;
      const nearPlatformTop = coinBottom >= platform.y - 48 && coinBottom <= platform.y + 24;
      return withinPlatformX && nearPlatformTop;
    });
    if (!supportPlatform) return false;

    const standingTopY = supportPlatform.y - PLAYER_SIZE;
    const verticalReachable = coin.y >= standingTopY - 72 && coin.y <= supportPlatform.y + 6;
    if (!verticalReachable) return false;

    const horizontalReachable = coin.x + coin.w >= supportPlatform.x - 10 && coin.x <= supportPlatform.x + supportPlatform.w + 10;
    if (!horizontalReachable) return false;

    const overlapSpike = spikes.some((spike) => intersects(coinRect, spike));
    return !overlapSpike;
  }

  _isCoinTooCloseToSpike(coin, spikes) {
    const coinBottom = coin.y + coin.h;
    return spikes.some((spike) => {
      const overlapX = coin.x < spike.x + spike.w + 18 && coin.x + coin.w > spike.x - 18;
      const closeY = Math.abs(coinBottom - spike.y) < 42;
      return overlapX && closeY;
    });
  }

  _placeCoinOnSafePlatformEdge(coin, platforms, spikes) {
    const platformCandidates = [...platforms]
      .filter((p) => p.w >= 96)
      .sort((a, b) => a.x - b.x);

    for (const platform of platformCandidates) {
      const coinY = this._clamp(platform.y - coin.h - 14, 50, SCREEN_HEIGHT - 60);
      const leftX = this._clamp(platform.x + 8, platform.x + 8, platform.x + platform.w - coin.w - 8);
      const rightX = this._clamp(platform.x + platform.w - coin.w - 10, platform.x + 8, platform.x + platform.w - coin.w - 8);

      // 优先放在右侧边缘，鼓励“通过障碍后拿奖励”的路线。
      const candidates = [rightX, leftX];
      for (const x of candidates) {
        const candidate = { ...coin, x, y: coinY };
        if (!this._isCoinTooCloseToSpike(candidate, spikes) && this._isCoinReachable(candidate, platforms, spikes)) {
          return candidate;
        }
      }
    }

    return { ...coin };
  }

  _fallbackReachableCoin(coin, platforms, spikes) {
    const sortedPlatforms = [...platforms].sort((a, b) => a.x - b.x);
    for (const platform of sortedPlatforms) {
      const candidate = {
        ...coin,
        x: platform.x + Math.max(0, (platform.w - coin.w) * 0.5),
        y: this._clamp(platform.y - coin.h - 14, 50, SCREEN_HEIGHT - 60),
      };
      if (this._isCoinReachable(candidate, platforms, spikes)) return candidate;
    }
    return {
      ...coin,
      x: this._clamp(coin.x, 40, SCREEN_WIDTH - coin.w - 20),
      y: this._clamp(coin.y, 70, SCREEN_HEIGHT - 70),
    };
  }

  _anchorSpikeToPlatform(spike, platforms, targetX = null) {
    const TAKEOFF_CLEARANCE = 34;
    const LANDING_CLEARANCE = 42;
    const candidates = platforms
      .slice(1)
      .filter((p) => p.w >= spike.w + TAKEOFF_CLEARANCE + LANDING_CLEARANCE && p.x > 140)
      .sort((a, b) => a.x - b.x);
    if (!candidates.length) return null;

    const preferredX = Number.isFinite(targetX) ? targetX : spike.x;
    const platform = candidates.reduce((best, current) => {
      const bestDist = best ? Math.abs((best.x + best.w * 0.5) - preferredX) : Number.POSITIVE_INFINITY;
      const currDist = Math.abs((current.x + current.w * 0.5) - preferredX);
      return currDist < bestDist ? current : best;
    }, null);

    if (!platform) return null;

    const minX = platform.x + TAKEOFF_CLEARANCE;
    const maxX = platform.x + platform.w - spike.w - LANDING_CLEARANCE;
    if (maxX < minX) return null;
    const anchoredX = maxX >= minX
      ? this._clamp(preferredX, minX, maxX)
      : platform.x + Math.max(0, (platform.w - spike.w) * 0.5);

    return {
      ...spike,
      x: anchoredX,
      y: platform.y - spike.h,
    };
  }

  _getRandomProfile(episodeNumber, totalEpisodes) {
    // 同难度阶段随机化参数：只调小幅扰动，不改核心结构（平台总数/终点/主路线）。
    const stage = Level.getDifficultyStage(episodeNumber, totalEpisodes);
    if (stage === "low") {
      return {
        stageLabel: "低难度",
        spikeXJitter: 50,
        platformGapJitter: 30,
        platformYJitter: 0,
        coinPosJitter: 40,
        placeSpikesOnPlatforms: true,
        disableSpikes: true,
        coinCountMin: null,
        coinCountMax: null,
        addSmallSpike: false,
      };
    }
    if (stage === "mid") {
      return {
        stageLabel: "中难度",
        spikeXJitter: 80,
        platformGapJitter: 50,
        platformYJitter: 20,
        coinPosJitter: 40,
        placeSpikesOnPlatforms: true,
        disableSpikes: false,
        coinCountMin: 3,
        coinCountMax: 5,
        addSmallSpike: false,
      };
    }
    return {
      stageLabel: "高难度",
      spikeXJitter: 100,
      platformGapJitter: 70,
      platformYJitter: 30,
      coinPosJitter: 40,
      placeSpikesOnPlatforms: true,
      disableSpikes: false,
      coinCountMin: null,
      coinCountMax: null,
      addSmallSpike: true,
    };
  }

  _randomizeWithinDifficulty(spec, episodeNumber, totalEpisodes) {
    const profile = this._getRandomProfile(episodeNumber, totalEpisodes);
    const randomized = {
      platforms: spec.platforms.map((p) => ({ ...p })),
      spikes: spec.spikes.map((s) => ({ ...s })),
      coins: spec.coins.map((c) => ({ ...c })),
      // 硬性限制：终点位置保持不变。
      goal: { ...spec.goal },
    };

    // 平台随机化（同难度内小幅扰动）：
    // - 低难度只调水平间距；
    // - 中/高难度额外调高度；
    // - 起点和终点平台x固定，保证核心路线与总长度不变。
    const basePlatforms = spec.platforms;
    const newX = basePlatforms.map((p) => p.x);
    const newY = basePlatforms.map((p) => p.y);
    for (let i = 1; i < basePlatforms.length; i++) {
      const baseGap = basePlatforms[i].x - basePlatforms[i - 1].x;
      const jitterGap = this._jitter(baseGap, profile.platformGapJitter);
      newX[i] = newX[i - 1] + jitterGap;
      if (i < basePlatforms.length - 1 && profile.platformYJitter > 0) {
        newY[i] = this._jitter(basePlatforms[i].y, profile.platformYJitter);
      }
    }

    // 保持末端平台锚点不变：把中间平台做线性回拉，维持总长度稳定。
    const last = basePlatforms.length - 1;
    const drift = newX[last] - basePlatforms[last].x;
    for (let i = 1; i < last; i++) {
      const ratio = i / last;
      newX[i] -= drift * ratio;
    }
    newX[0] = basePlatforms[0].x;
    newX[last] = basePlatforms[last].x;

    // 可达性约束：限制平台间边缘间距不超过可跳跃范围，避免生成不可达地图。
    const MIN_EDGE_GAP = 18;
    const MAX_EDGE_GAP = 140;
    for (let i = 1; i < basePlatforms.length; i++) {
      const prev = randomized.platforms[i - 1];
      const width = randomized.platforms[i].w;
      const minX = prev.x + prev.w + MIN_EDGE_GAP;
      const maxX = prev.x + prev.w + MAX_EDGE_GAP;
      newX[i] = this._clamp(newX[i], minX, maxX);
      randomized.platforms[i].x = newX[i];
      randomized.platforms[i].y = newY[i];
      randomized.platforms[i].w = width;
      randomized.platforms[i].h = basePlatforms[i].h;
    }
    randomized.platforms[0] = { ...basePlatforms[0] };

    // 陷阱随机化：中高难度将陷阱锚定到平台顶部，避免“生成在平台下方碰不到”。
    randomized.spikes = randomized.spikes.map((s) => {
      const jitteredSpike = {
        ...s,
        x: this._clamp(this._jitter(s.x, profile.spikeXJitter), 70, randomized.goal.x - 120),
      };
      if (profile.placeSpikesOnPlatforms) {
        return this._anchorSpikeToPlatform(jitteredSpike, randomized.platforms, jitteredSpike.x);
      }
      return this._isSpikeAbovePlatform(jitteredSpike, randomized.platforms) ? { ...s } : jitteredSpike;
    }).filter(Boolean);

    if (profile.disableSpikes) {
      randomized.spikes = [];
    }

    // 金币随机化：
    // - 低/高难度保持数量不变，仅位置小幅随机；
    // - 中难度数量随机到3~5（若原数量不足，按可用上限）；
    // - 关键修正：金币最终锚定到“最近平台上方”，避免与平台视觉/可达性错位。
    let coinPool = randomized.coins.map((c) => {
      const jitteredX = this._clamp(this._jitter(c.x, profile.coinPosJitter), 40, randomized.goal.x - 40);

      // 找到x方向最近的平台，并将金币放置在该平台上方。
      const nearestPlatform = randomized.platforms.reduce((best, platform) => {
        const bestDist = best ? Math.abs(jitteredX - (best.x + best.w * 0.5)) : Number.POSITIVE_INFINITY;
        const dist = Math.abs(jitteredX - (platform.x + platform.w * 0.5));
        return dist < bestDist ? platform : best;
      }, null);

      if (!nearestPlatform) {
        return {
          ...c,
          x: jitteredX,
          y: this._clamp(this._jitter(c.y, profile.coinPosJitter), 70, SCREEN_HEIGHT - 55),
        };
      }

      const marginX = Math.max(8, c.w * 0.5);
      const minAnchorX = nearestPlatform.x + marginX;
      const maxAnchorX = nearestPlatform.x + nearestPlatform.w - c.w - marginX * 0.5;
      const anchoredX = maxAnchorX >= minAnchorX
        ? this._clamp(jitteredX, minAnchorX, maxAnchorX)
        : nearestPlatform.x + Math.max(0, (nearestPlatform.w - c.w) * 0.5);
      const aboveOffset = this._clamp(randomFloat(10, 22), 10, 24);
      const anchoredY = this._clamp(nearestPlatform.y - c.h - aboveOffset, 50, SCREEN_HEIGHT - 60);

      const candidate = {
        ...c,
        x: anchoredX,
        y: anchoredY,
      };

      if (this._isCoinReachable(candidate, randomized.platforms, randomized.spikes)) {
        return candidate;
      }

      return this._fallbackReachableCoin(candidate, randomized.platforms, randomized.spikes);
    });

    if (profile.coinCountMin !== null && profile.coinCountMax !== null) {
      coinPool = coinPool.sort(() => Math.random() - 0.5);
      const maxCount = Math.min(profile.coinCountMax, coinPool.length);
      const minCount = Math.min(profile.coinCountMin, maxCount);
      const targetCount = Math.floor(randomFloat(minCount, maxCount + 0.999));
      coinPool = coinPool.slice(0, Math.max(minCount, targetCount));
    }

    // 金币安全约束：避免放在陷阱正上方或陷阱附近“必死点”。
    coinPool = coinPool.map((coin) => {
      if (!this._isCoinTooCloseToSpike(coin, randomized.spikes)) return coin;
      return this._placeCoinOnSafePlatformEdge(coin, randomized.platforms, randomized.spikes);
    });
    randomized.coins = coinPool;

    // 高难度附加：新增1个小陷阱（不影响核心路线）。
    let extraSmallSpike = 0;
    if (profile.addSmallSpike) {
      const smallSpikeSeed = {
        x: this._clamp(randomFloat(120, randomized.goal.x - 160), 70, randomized.goal.x - 120),
        y: SCREEN_HEIGHT - 50,
        w: 24,
        h: 8,
      };
      const smallSpike = profile.placeSpikesOnPlatforms
        ? this._anchorSpikeToPlatform(smallSpikeSeed, randomized.platforms, smallSpikeSeed.x)
        : smallSpikeSeed;
      if (smallSpike && (!this._isSpikeAbovePlatform(smallSpike, randomized.platforms) || profile.placeSpikesOnPlatforms)) {
        randomized.spikes.push(smallSpike);
        extraSmallSpike = 1;
      }
    }

    // 记录本回合随机化参数，供控制台打印和报告说明。
    const gaps = [];
    for (let i = 1; i < randomized.platforms.length; i++) {
      gaps.push(randomized.platforms[i].x - randomized.platforms[i - 1].x);
    }
    const avgGap = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : 0;
    this.randomizationMeta = {
      difficulty: profile.stageLabel,
      spikeXJitter: profile.spikeXJitter,
      platformGapJitter: profile.platformGapJitter,
      platformYJitter: profile.platformYJitter,
      coinPosJitter: profile.coinPosJitter,
      avgPlatformGap: avgGap,
      firstSpikeX: randomized.spikes.length ? randomized.spikes[0].x : "none",
      extraSmallSpike,
    };

    return randomized;
  }

  addPlatformSpikes(extraCount = 2) {
    // 后期难度增强：在“非地面平台”上动态加陷阱。
    // 只在中后期课程调用，前期学习阶段不触发，避免破坏基础收敛。
    const candidates = this.platforms
      .slice(1)
      .filter((p) => p.w >= 90 && p.x > 220 && p.x < this.goal.x - 140)
      .sort(() => Math.random() - 0.5);

    let placed = 0;
    for (const platform of candidates) {
      if (placed >= extraCount) break;

      const spikeW = 32;
      const spikeH = 10;
      const spikeX = platform.x + platform.w * 0.5 - spikeW * 0.5;
      const spikeY = platform.y - spikeH;

      // 避免与已有陷阱过近重叠，保证可读性与可通关性。
      const tooClose = this.spikes.some((s) => Math.abs(s.x - spikeX) < 40 && Math.abs(s.y - spikeY) < 14);
      if (tooClose) continue;

      this.spikes.push(new Entity(spikeX, spikeY, spikeW, spikeH, RED));
      placed += 1;
    }
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
    // 输入层维度与状态空间严格一致：由16维扩展为17维。
    this.stateDim = STATE_DIM;
    this.actionDim = 5;

    // 超参数微调：降低学习率与目标网络同步频率，提升训练稳定性。
    this.lr = 0.0008;
    this.gamma = 0.95;
    this.epsilon = 0.99;
    // epsilon衰减修复：
    // - 按要求保留初始0.99；
    // - 最小值恢复为0.1；
    // - 衰减系数改为0.993，确保300回合可接近下限并更早进入利用阶段。
    this.epsilonMin = 0.1;
    this.epsilonDecay = 0.993;
    this.batchSize = 32;
    this.memorySize = 10000;
    this.updateTargetStep = 15;

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
    this.progressMilestones = new Set();
    this.passedSpikeIndices = new Set();
    this.recentEpisodeRewards = [];

    // 防卡死与行为识别状态。
    this.maxProgressX = 40;
    this.noProgressFrames = 0;
    this.ineffectiveActionFrames = 0;
    this.recentXWindow = [];
    this.recentActionWindow = [];
    this.prevAction = null;
    this.behaviorState = "normal";

    // 重新登顶奖励状态：记录是否掉回地面并再次爬上平台。
    // 无地面模式：将“地面参考”改为屏幕底边，用于掉落相关行为识别。
    this.groundYRef = SCREEN_HEIGHT - PLAYER_SIZE;
    this.fallenToGround = false;
    this.reClimbRewardedAfterFall = false;
    // 新规则状态：记录AI是否曾登上高平台。
    // 一旦登上后又掉回地面，将直接判定本回合失败。
    this.reachedHighPlatform = false;

    // on_ground调试日志开关：排查时可改为true。
    this.debugGroundLog = false;
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
    this.progressMilestones.clear();
    this.passedSpikeIndices.clear();

    this.maxProgressX = 40;
    this.noProgressFrames = 0;
    this.ineffectiveActionFrames = 0;
    this.recentXWindow = [];
    this.recentActionWindow = [];
    this.prevAction = null;
    this.behaviorState = "normal";

    this.groundYRef = SCREEN_HEIGHT - PLAYER_SIZE;
    this.fallenToGround = false;
    this.reClimbRewardedAfterFall = false;
    this.reachedHighPlatform = false;
  }

  get_state(game, aiCharacter) {
    // 状态重构为17维：仅保留决策强相关特征，全部归一化到0~1。
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
      // 新增维度：前方最近陷阱宽度（0~1）。
      // 采用120像素作为归一化上界，覆盖当前关卡常见陷阱宽度并保留区分度。
      nearestSpike ? clamp01(nearestSpike.w / 120) : 1,
      nearestPlatform ? normX(nearestPlatform.x) : 1,
      nearestPlatform ? normY(nearestPlatform.y) : 1,
      nearestCoin ? normX(nearestCoin.x) : 1,
      nearestCoin ? normY(nearestCoin.y) : 1,
      clamp01((aiX - 40) / goalDistance),
      clamp01((this.maxProgressX - 40) / goalDistance),
    ];

    while (state.length < STATE_DIM) state.push(1);
    if (state.length > STATE_DIM) state.length = STATE_DIM;

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

    // 规则增强：掉下横板（高平台）即失败。
    // 触发条件：
    // 1) AI曾到达高平台高度（不要求必须onGround，避免漏判）；
    // 2) 当前又掉回地面层附近；
    // 3) 未通关。
    // 这样可避免AI“反复爬上去又掉下去”拖长回合并污染学习信号。
    const highPlatformThresholdY = this.groundYRef - 45;
    if (aiCharacter.y <= highPlatformThresholdY) {
      this.reachedHighPlatform = true;
    }
    const droppedBackToGround = aiCharacter.y >= this.groundYRef - 2;
    if (!aiCharacter.reachedGoal && this.reachedHighPlatform && droppedBackToGround) {
      aiCharacter.isAlive = false;
    }

    // 更新历史最高进度与停滞计数：
    // 评分子项1.1要求回合必须可终止，后续所有“停滞/循环”判定都依赖该统计。
    if (newX > this.maxProgressX + 0.2) {
      this.maxProgressX = newX;
      this.noProgressFrames = 0;
      this.behaviorState = "forward";
    } else {
      this.noProgressFrames += 1;
    }

    // 1) 终极通关奖励：一次性+300并结束回合。
    // 量级下调后，奖励曲线更平稳，便于分析各子奖励对策略的贡献。
    if (aiCharacter.reachedGoal) {
      reward += 300;
      done = true;
      endReason = "goal";
    }

    // 7) 死亡惩罚：一次性-120并结束回合。
    if (!aiCharacter.isAlive) {
      reward -= 120;
      done = true;
      endReason = "death";
    }

    // 6) 金币奖励：每枚+20。
    const currentCoinCount = game.level.coins.length;
    if (currentCoinCount < this.prevCoinCount) {
      reward += (this.prevCoinCount - currentCoinCount) * 20;
    }

    // 2) 里程碑进度奖励：每10%一次性+25。
    const progress = Math.max(0, Math.min(1, (newX - 40) / goalDistance));
    const milestone = Math.floor(progress * 10);
    for (let step = 1; step <= milestone; step++) {
      if (!this.progressMilestones.has(step)) {
        this.progressMilestones.add(step);
        reward += 25;
      }
    }

    // 3) 持续前进奖励：向右推进+1.2，否则-0.3。
    // 惩罚减弱是为了允许AI在障碍前进行小幅微调，降低策略震荡。
    if (newX > oldX) reward += 1.2;
    else reward -= 0.3;

    // 新增：进度倒退惩罚。低于历史最高进度时每帧-1.2，
    // 强制AI从“乱走”回到“重新向终点推进”的行为轨道。
    if (newX + 0.2 < this.maxProgressX) {
      reward -= 1.2;
      this.behaviorState = "backtracking";
    }

    // 新增：进度停滞惩罚。连续100帧无前进后每帧-2，
    // 防止AI在地面或局部平台长时间试探但没有任何有效推进。
    if (this.noProgressFrames >= EPISODE_LIMITS.noProgressPenaltyStartFrames) {
      reward -= 2;
      this.behaviorState = "stagnant";
    }

    // 4) 生存奖励：存活且未通关每帧+0.2。
    if (aiCharacter.isAlive && !aiCharacter.reachedGoal) reward += 0.2;

    // 无效跳跃惩罚：在前方有可走平台时仍频繁跳跃，会产生轻微负反馈。
    const lookAheadX = aiCharacter.x + aiCharacter.w + 26;
    const lookAheadY = aiCharacter.y + aiCharacter.h + 4;
    const hasWalkableFloorAhead = game.level.platforms.some((p) =>
      lookAheadX >= p.x &&
      lookAheadX <= p.x + p.w &&
      lookAheadY >= p.y &&
      lookAheadY <= p.y + p.h + 8
    );
    if (this.lastActionWasJump && aiCharacter.onGround && hasWalkableFloorAhead) {
      reward -= 0.7;
    }

    // 5) 跳跃避障奖励：成功从陷阱左侧跨越到右侧且安全存活，每个陷阱+35仅一次。
    for (let i = 0; i < game.level.spikes.length; i++) {
      if (this.passedSpikeIndices.has(i)) continue;
      const spike = game.level.spikes[i];
      const passed = oldX + aiCharacter.w <= spike.x && newX >= spike.x + spike.w;
      const safelyAbove = aiCharacter.y + aiCharacter.h <= spike.y + 4;
      if (passed && safelyAbove && aiCharacter.isAlive) {
        this.passedSpikeIndices.add(i);
        reward += 35;
      }
    }

    // 新增：重新登顶奖励。掉回地面后再次登上高平台，一次性+15。
    if (!this.fallenToGround && aiCharacter.y >= this.groundYRef - 2) {
      this.fallenToGround = true;
      this.reClimbRewardedAfterFall = false;
    }
    const climbedHighPlatform = aiCharacter.onGround && aiCharacter.y <= this.groundYRef - 45;
    if (this.fallenToGround && !this.reClimbRewardedAfterFall && climbedHighPlatform) {
      reward += 15;
      this.reClimbRewardedAfterFall = true;
      this.fallenToGround = false;
      this.behaviorState = "reclimb";
    }

    // 行为窗口：用于识别左右乱走与无效动作循环。
    this.recentXWindow.push(newX);
    if (this.recentXWindow.length > EPISODE_LIMITS.loopTerminateFrames) this.recentXWindow.shift();

    const actionTag = this.lastActionWasJump
      ? "jump"
      : (aiCharacter.velX > 0 ? "right" : (aiCharacter.velX < 0 ? "left" : "idle"));
    this.recentActionWindow.push(actionTag);
    if (this.recentActionWindow.length > EPISODE_LIMITS.loopTerminateFrames) this.recentActionWindow.shift();

    if (this.prevAction === actionTag && Math.abs(newX - oldX) < 0.15) {
      this.ineffectiveActionFrames += 1;
    } else {
      this.ineffectiveActionFrames = Math.max(0, this.ineffectiveActionFrames - 2);
    }
    this.prevAction = actionTag;

    // 强制终止A：连续300帧无前进（作业要求的停滞终止）。
    if (!done && this.noProgressFrames >= EPISODE_LIMITS.noProgressTerminateFrames) {
      done = true;
      endReason = "stagnation";
      this.behaviorState = "force_end_stagnation";
    }

    // 强制终止B：连续200帧左右来回、净前进极小（作业要求的无效循环终止）。
    if (!done && this.recentXWindow.length >= EPISODE_LIMITS.loopTerminateFrames) {
      const netDisplacement = Math.abs(this.recentXWindow[this.recentXWindow.length - 1] - this.recentXWindow[0]);
      let leftCount = 0;
      let rightCount = 0;
      for (const act of this.recentActionWindow) {
        if (act === "left") leftCount += 1;
        if (act === "right") rightCount += 1;
      }
      const oscillating = netDisplacement < 20 && leftCount > 45 && rightCount > 45;
      if (oscillating) {
        done = true;
        endReason = "loop";
        this.behaviorState = "force_end_loop";
      }
    }

    // 强制终止C：无效动作连续200帧（例如地面反复同动作且几乎无位移）。
    if (!done && this.ineffectiveActionFrames >= EPISODE_LIMITS.ineffectiveActionTerminateFrames) {
      done = true;
      endReason = "ineffective_action";
      this.behaviorState = "force_end_ineffective";
    }

    // 8) 超时惩罚：60秒未通关，-30并结束回合。
    if (this.episodeSteps >= EPISODE_LIMITS.maxFrames && !done) {
      reward -= 30;
      done = true;
      endReason = "timeout";
      this.behaviorState = "force_end_timeout";
    }

    this.prevCoinCount = currentCoinCount;

    // 在重置回合计数前先缓存本回合生存步数，供训练统计记录使用。
    const survivalSteps = this.episodeSteps;

    if (done) {
      this.episodeSteps = 0;
      this.lastActionWasJump = false;
      this.noProgressFrames = 0;
      this.ineffectiveActionFrames = 0;
      this.recentXWindow = [];
      this.recentActionWindow = [];
      this.prevAction = null;
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

  _adamUpdate1D(param, grad, m, v, t) {
    // Adam更新（向量参数）：
    // t 由外层训练步骤统一维护，确保同一次反向传播的所有参数共享同一时间步。
    // 这样偏置校正项 mHat/vHat 与标准Adam定义一致，避免“每层各算一次t”带来的数值偏移。
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

  _adamUpdate2D(param, grad, m, v, t) {
    // Adam更新（矩阵参数）：与1D版本共享同一训练时间步 t。
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

    // 训练样本维度守卫：
    // 1) 防止历史脏数据/旧版本缓存导致维度不匹配；
    // 2) 一旦维度异常，跳过该样本并返回0损失，避免污染梯度。
    if (
      !Array.isArray(state) ||
      !Array.isArray(nextState) ||
      state.length !== this.stateDim ||
      nextState.length !== this.stateDim ||
      action < 0 ||
      action >= this.actionDim
    ) {
      return 0;
    }

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

    // 5) 优化器更新：
    // 训练流程严格遵循“梯度清零(本函数中以新建0梯度数组实现) -> 前向 -> 损失 -> 反向 -> Adam更新”。
    // 关键修复：同一个样本的所有参数共享同一个Adam时间步 t（标准实现）。
    this.optimizerStep += 1;
    const t = this.optimizerStep;

    this._adamUpdate2D(this.online.W4, gW4, this.optim.mW4, this.optim.vW4, t);
    this._adamUpdate1D(this.online.b4, gb4, this.optim.mb4, this.optim.vb4, t);
    this._adamUpdate2D(this.online.W3, gW3, this.optim.mW3, this.optim.vW3, t);
    this._adamUpdate1D(this.online.b3, gb3, this.optim.mb3, this.optim.vb3, t);
    this._adamUpdate2D(this.online.W2, gW2, this.optim.mW2, this.optim.vW2, t);
    this._adamUpdate1D(this.online.b2, gb2, this.optim.mb2, this.optim.vb2, t);
    this._adamUpdate2D(this.online.W1, gW1, this.optim.mW1, this.optim.vW1, t);
    this._adamUpdate1D(this.online.b1, gb1, this.optim.mb1, this.optim.vb1, t);

    return tdError * tdError;
  }

  update() {
    // 经验不足时不训练，确保采样批次有效。
    if (this.memory.length < this.batchSize) return null;

    // 经验回放采样：
    // - 采用“随机无放回”子集，减少同一batch内重复样本比例；
    // - 每条样本都经过_trainSingle维度守卫，确保输入与网络维度一致。
    let lossSum = 0;
    const indices = Array.from({ length: this.memory.length }, (_, idx) => idx);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const temp = indices[i];
      indices[i] = indices[j];
      indices[j] = temp;
    }

    for (let i = 0; i < this.batchSize; i++) {
      lossSum += this._trainSingle(this.memory[indices[i]]);
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
    this.level = new Level(this.levelMode, 1, 300);
    const spawn = this.level.getSpawnPoint();
    this.player = new Character(spawn.x, spawn.y, BLUE);
    this.ai = new Character(spawn.x, spawn.y, PURPLE);

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
    this.lastStateVector = new Array(STATE_DIM).fill(0);
    this.lastLoss = 0;
    this.lastWeightChecksum = this.agent.getWeightChecksum();
    this.lastWeightDelta = 0;
    this.currentEpisodeActionCounts = [0, 0, 0, 0, 0];
    this.decisionLogs = [];
    this.lastDecisionMessage = "";
    this.decisionLogDirty = true;
    this.hudTick = 0;
    this.debugActionLog = false;
    this.liteUiMode = CODE_ACCEL_PROFILE.enabled ? !!CODE_ACCEL_PROFILE.forceLiteUi : true;

    // 自动训练模式控制参数。
    this.autoTrainingMode = false;
    this.trainingSpeed = CODE_ACCEL_PROFILE.enabled
      ? ([1, 2, 4, 8].includes(CODE_ACCEL_PROFILE.forceAutoTrainSpeed) ? CODE_ACCEL_PROFILE.forceAutoTrainSpeed : 1)
      : 1;
    this.maxTrainingEpisodes = 300;
    this.trainingEpisodeIndex = 0;
    this.trainingEpisodeHistory = [];
    this.stageSnapshots = {};
    this.manualNextEpisodeCountdown = 0;
    this.lastBatchedPanelEpisode = -1;
    this.levelSignature = "-";
    this.pendingAiWinCapture = false;
    this.lastAiWinFrameDataUrl = "";
    this.lastAiWinEpisode = 0;
    this.dashboardUpdateEpisodeStride = CODE_ACCEL_PROFILE.enabled
      ? Math.max(1, Number(CODE_ACCEL_PROFILE.dashboardUpdateEpisodeStride || 10))
      : 10;

    // 最近一回合摘要：用于结束页展示“双方进度+回合时长”，满足交互评分项可视化要求。
    this.lastEpisodeDurationSec = 0;
    this.lastEpisodePlayerProgressPct = 0;
    this.lastEpisodeAiProgressPct = 0;

    // 玩家侧“掉下横板即失败”状态：
    // 与AI规则保持一致，避免手动模式与AI模式判定不一致。
    this.playerGroundYRef = SCREEN_HEIGHT - PLAYER_SIZE;
    this.playerReachedHighPlatform = false;

    // 教师引导（自动玩家）参数：
    // - 前期用启发式动作给AI“示范”，帮助更快学会基础通关动作；
    // - 随着回合增加，引导概率逐步衰减，最终让AI独立决策。
    this.enableTeacherAssist = true;
    // 提升前期引导强度：
    // 目标是在50~100回合尽快看到首次通关，再逐步降低引导让策略独立。
    this.teacherAssistMaxEpisodes = 240;
    this.teacherOverrideProbStart = 0.55;
    this.teacherOverrideProbEnd = 0.08;
    this.teacherImitationBonus = 0.45;
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
    this.applyLiteUiMode();
  }

  isLiteTrainingMode() {
    // 简洁加速判定：仅在“自动训练 + 高倍率”时启用，避免影响手动演示体验。
    return this.liteUiMode && this.autoTrainingMode && this.trainingSpeed >= 4;
  }

  applyLiteUiMode() {
    // 通过body类名切换简洁视图（缩减非核心面板显示）。
    document.body.classList.toggle("lite-ui", this.liteUiMode);
    if (uiControls.toggleLiteUi) uiControls.toggleLiteUi.checked = this.liteUiMode;
  }

  setTrainingSpeed(multiplier, options = {}) {
    // 倍速核心状态统一入口：
    // 1) 限制倍率只能是1/2/4/8；
    // 2) 更新全局倍率变量 currentSpeedMultiplier；
    // 3) 同步下拉框与标题文本；
    // 4) 倍速>1自动启用“简洁加速”；
    // 5) 打印控制台切换日志，便于验收。
    const { save = true, log = true } = options;
    const next = [1, 2, 4, 8].includes(Number(multiplier)) ? Number(multiplier) : 1;

    this.trainingSpeed = next;
    currentSpeedMultiplier = next;

    if (uiControls.speedSelect) uiControls.speedSelect.value = String(next);
    if (uiControls.speedLabel) uiControls.speedLabel.textContent = `速度：${next}x`;

    if (next > 1 && !this.liteUiMode) {
      this.liteUiMode = true;
      this.applyLiteUiMode();
    }

    if (log) {
      console.log(`[倍速切换] 当前训练速度已切换为 ${next}x`);
    }

    if (save) this.saveTrainingData({ includeAgentSnapshot: false });
  }

  buildLevelSignature(level) {
    // 地图签名（可视化调试）：
    // 将关键平台/陷阱/终点坐标组合成短字符串，快速判断回合地图是否变化。
    const platforms = level.platforms.slice(1, 4).map((e) => `${Math.round(e.x)}-${Math.round(e.y)}`).join("|");
    const spikes = level.spikes.slice(0, 2).map((e) => `${Math.round(e.x)}-${Math.round(e.y)}`).join("|") || "none";
    const goal = `${Math.round(level.goal.x)}-${Math.round(level.goal.y)}`;
    return `${level.mode}:${platforms}:${spikes}:${goal}`;
  }

  reset() {
    // 每回合刷新关卡：采用课程学习+随机模式，避免只在单一地图过拟合。
    this.levelMode = this.pickLevelMode();
    const episodeNo = this.trainingEpisodeIndex + 1;
    this.level = new Level(this.levelMode, episodeNo, this.maxTrainingEpisodes);

    // 反复同图保护：自动训练下若签名与上回合一致，最多重采样2次。
    // 这样能提高“每回合地图都有变化”的概率，减少策略对单一地图记忆。
    let signature = this.buildLevelSignature(this.level);
    if (this.autoTrainingMode && signature === this.levelSignature) {
      for (let attempt = 0; attempt < 2; attempt++) {
        this.level = new Level(this.levelMode, episodeNo, this.maxTrainingEpisodes);
        signature = this.buildLevelSignature(this.level);
        if (signature !== this.levelSignature) break;
      }
    }
    this.levelSignature = signature;

    // 每回合打印难度与随机化参数，便于训练过程记录与报告复现。
    const meta = this.level.randomizationMeta;
    console.log(
      `[MAP] 当前难度：${meta.difficulty}，随机化参数：陷阱x=${meta.firstSpikeX}（±${meta.spikeXJitter}），平台间距均值=${meta.avgPlatformGap.toFixed(1)}（±${meta.platformGapJitter}），平台高度抖动=±${meta.platformYJitter}，金币抖动=±${meta.coinPosJitter}，高难小陷阱=${meta.extraSmallSpike}`
    );

    const spawn = this.level.getSpawnPoint();
    this.player.reset(spawn.x, spawn.y);
    this.ai.reset(spawn.x, spawn.y);
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
    this.lastStateVector = new Array(STATE_DIM).fill(0);
    this.lastLoss = 0;
    this.lastWeightChecksum = this.agent.getWeightChecksum();
    this.lastTeacherAction = null;
    this.playerReachedHighPlatform = false;

    if (CODE_ACCEL_PROFILE.enabled && this.autoTrainingMode && CODE_ACCEL_PROFILE.forceLiteUi) {
      this.liteUiMode = true;
      this.applyLiteUiMode();
    }

    // 每次重开都写入一条关键日志，方便观察新回合决策起点。
    this.appendDecisionLog(`回合重置：模式=${this.levelMode}，地图签名=${this.levelSignature}，AI开始新一轮探索`, true);
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
    // 多模式课程学习策略（按最大回合百分比）：
    // - 前15%：tutorial 70% + classic 30%（避免过度“保姆图”）；
    // - 15%~45%：classic 75% + zigzag 25%；
    // - 45%~75%：classic 50% + zigzag 35% + gaps 15%；
    // - 后25%：classic 35% + zigzag 35% + gaps 30%。
    const total = Math.max(1, this.maxTrainingEpisodes);
    const progress = Math.min(1, Math.max(0, this.trainingEpisodeIndex / total));
    const roll = Math.random();

    if (progress < 0.15) {
      return roll < 0.7 ? "tutorial" : "classic";
    }

    if (progress < 0.45) {
      return roll < 0.75 ? "classic" : "zigzag";
    }

    if (progress < 0.75) {
      if (roll < 0.5) return "classic";
      if (roll < 0.85) return "zigzag";
      return "gaps";
    }

    if (roll < 0.35) return "classic";
    if (roll < 0.7) return "zigzag";
    return "gaps";
  }

  getCurriculumStageText() {
    // 与pickLevelMode保持同一套分段，确保提示文案与真实采样策略一致。
    const total = Math.max(1, this.maxTrainingEpisodes);
    const progress = Math.min(1, Math.max(0, this.trainingEpisodeIndex / total));

    if (progress < 0.15) return "课程=前15%(tutorial70/classic30)";
    if (progress < 0.45) return "课程=15~45%(classic75/zigzag25)";
    if (progress < 0.75) return "课程=45~75%(classic50/zigzag35/gaps15)";
    return "课程=后25%(classic35/zigzag35/gaps30)";
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
      if (dx > 10) {
        const coinProbeX = this.ai.x + this.ai.w + 24;
        const coinProbeY = feetY + 6;
        const hasFloorForCoin = this.level.platforms.some((p) =>
          coinProbeX >= p.x && coinProbeX <= p.x + p.w && coinProbeY >= p.y && coinProbeY <= p.y + p.h + 6
        );
        if (hasFloorForCoin) return 1;
      }
      if (dx < -10) return 0;
    }

    // 4) 平台上优先横向移动：有前方落脚面时尽量走，不做无意义跳跃。
    if (this.ai.onGround && hasFutureFloor) {
      const goalDxSafeRun = this.level.goal.x - aiCenterX;
      return goalDxSafeRun >= 0 ? 1 : 0;
    }

    // 5) 默认策略：向终点方向推进；仅在确实需要跨越时再跳。
    const goalDx = this.level.goal.x - aiCenterX;
    const longGapProbeX = this.ai.x + this.ai.w + 42;
    const longGapProbeY = feetY + 8;
    const hasLongFutureFloor = this.level.platforms.some((p) =>
      longGapProbeX >= p.x && longGapProbeX <= p.x + p.w && longGapProbeY >= p.y && longGapProbeY <= p.y + p.h + 6
    );
    if (goalDx > 0 && this.ai.onGround && !hasLongFutureFloor) return 4;
    return goalDx >= 0 ? 1 : 0;
  }

  normalizeActionForGroundRun(action) {
    // 硬约束：在前方存在连续落脚面时，将跳跃动作归一为左右移动。
    // 仅在前方缺口或近距离陷阱出现时保留跳跃。
    const isJumpAction = action === 2 || action === 3 || action === 4;
    if (!isJumpAction || !this.ai.onGround) return action;

    const feetY = this.ai.y + this.ai.h;
    const nearProbeX = this.ai.x + this.ai.w + 26;
    const farProbeX = this.ai.x + this.ai.w + 46;
    const probeY = feetY + 6;

    const hasNearFloor = this.level.platforms.some((p) =>
      nearProbeX >= p.x && nearProbeX <= p.x + p.w && probeY >= p.y && probeY <= p.y + p.h + 6
    );
    const hasFarFloor = this.level.platforms.some((p) =>
      farProbeX >= p.x && farProbeX <= p.x + p.w && probeY >= p.y && probeY <= p.y + p.h + 6
    );

    const aiCenterX = this.ai.x + this.ai.w * 0.5;
    const dangerAhead = this.level.spikes.some((spike) => {
      const dx = spike.x + spike.w * 0.5 - aiCenterX;
      const nearSameHeight = Math.abs(spike.y - feetY) < 60;
      return dx > 16 && dx < 96 && nearSameHeight;
    });

    if (hasNearFloor && hasFarFloor && !dangerAhead) {
      return this.level.goal.x >= aiCenterX ? 1 : 0;
    }

    return action;
  }

  getTeacherOverrideProbability() {
    // 教师介入概率衰减：早期高、后期低，避免永远“代打”。
    if (!this.enableTeacherAssist) return 0;
    if (this.trainingEpisodeIndex >= this.teacherAssistMaxEpisodes) return 0;

    const ratio = this.trainingEpisodeIndex / Math.max(1, this.teacherAssistMaxEpisodes);
    return this.teacherOverrideProbStart + (this.teacherOverrideProbEnd - this.teacherOverrideProbStart) * ratio;
  }

  startGame() {
    // 自动训练达到最大回合后，点击“开始/重开”时继续在当前模型基础上训练。
    // 不再重置统计与Agent，避免刷新后看起来“从第一回合重新开始”。
    if (this.autoTrainingMode && this.trainingEpisodeIndex >= this.maxTrainingEpisodes) {
      this.maxTrainingEpisodes = this.trainingEpisodeIndex + 300;
      if (uiControls.maxEpisodes) uiControls.maxEpisodes.value = String(this.maxTrainingEpisodes);
      this.appendDecisionLog("已基于当前模型继续训练：最大回合数自动+300", true);
      this.saveTrainingData({ includeAgentSnapshot: true });
    }

    this.reset();
    this.state = "running";
  }

  saveTrainingData(options = {}) {
    // 持久化优化：
    // - 训练统计始终保存；
    // - 模型快照按需保存，减少每回合大对象写入localStorage导致的卡顿。
    const { includeAgentSnapshot = false } = options;
    const statsPayload = {
      maxTrainingEpisodes: this.maxTrainingEpisodes,
      totalEpisodes: this.totalEpisodes,
      successEpisodes: this.successEpisodes,
      episodeRewards: this.episodeRewards,
      trainingEpisodeIndex: this.trainingEpisodeIndex,
      trainingEpisodeHistory: this.trainingEpisodeHistory,
      stageSnapshots: this.stageSnapshots,
      trainingSpeed: this.trainingSpeed,
      autoTrainingMode: this.autoTrainingMode,
      liteUiMode: this.liteUiMode,
      agentCoreState: {
        epsilon: this.agent.epsilon,
        stepCount: this.agent.stepCount,
        optimizerStep: this.agent.optimizerStep,
      },
    };

    localStorage.setItem(STORAGE_KEYS.trainingStats, JSON.stringify(statsPayload));
    if (includeAgentSnapshot) {
      localStorage.setItem(STORAGE_KEYS.agentSnapshot, JSON.stringify(this.agent.getSnapshot()));
    }
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
    if (typeof stats.trainingSpeed === "number") this.trainingSpeed = [1, 2, 4, 8].includes(stats.trainingSpeed) ? stats.trainingSpeed : 1;
    if (typeof stats.autoTrainingMode === "boolean") this.autoTrainingMode = stats.autoTrainingMode;
    if (typeof stats.liteUiMode === "boolean") this.liteUiMode = stats.liteUiMode;
    if (stats.agentCoreState && typeof stats.agentCoreState === "object") {
      const core = stats.agentCoreState;
      if (typeof core.epsilon === "number") this.agent.epsilon = core.epsilon;
      if (typeof core.stepCount === "number") this.agent.stepCount = Math.max(0, Math.floor(core.stepCount));
      if (typeof core.optimizerStep === "number") this.agent.optimizerStep = Math.max(0, Math.floor(core.optimizerStep));
    }

    if (Array.isArray(stats.trainingEpisodeHistory)) {
      this.trainingEpisodeHistory = stats.trainingEpisodeHistory.slice(-300).map((entry, idx) => ({
        episode: Number(entry.episode || idx + 1),
        reward: Number(entry.reward || 0),
        success: entry.success ? 1 : 0,
        survivalSteps: Number(entry.survivalSteps || 0),
        endReason: typeof entry.endReason === "string" ? entry.endReason : "unknown",
        actionCounts: Array.isArray(entry.actionCounts)
          ? entry.actionCounts.slice(0, ACTION_NAMES.length).map((v) => Number(v || 0))
          : [0, 0, 0, 0, 0],
      }));
    }

    if (stats.stageSnapshots && typeof stats.stageSnapshots === "object") {
      this.stageSnapshots = { ...stats.stageSnapshots };
    }

    if (uiControls.maxEpisodes) {
      uiControls.maxEpisodes.value = String(this.maxTrainingEpisodes);
    }

    if (CODE_ACCEL_PROFILE.enabled) {
      if ([1, 2, 4, 8].includes(CODE_ACCEL_PROFILE.forceAutoTrainSpeed)) {
        this.trainingSpeed = CODE_ACCEL_PROFILE.forceAutoTrainSpeed;
      }
      if (CODE_ACCEL_PROFILE.forceLiteUi) this.liteUiMode = true;
    }

    this.setTrainingSpeed(this.trainingSpeed, { save: false, log: false });
    this.applyLiteUiMode();
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
        stageSnapshots: this.stageSnapshots,
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
    // 高频回合结束只存统计；每10回合再额外存一次模型快照。
    this.saveTrainingData({ includeAgentSnapshot: this.trainingEpisodeIndex % 10 === 0 });
    refreshTrainingDashboard(this.trainingEpisodeHistory);
    this.reset();
    this.state = "running";
    this.appendDecisionLog("模型已从JSON文件恢复，AI进入已训练状态", true);
    overlay.textContent = "模型加载成功：已从JSON文件恢复";
    return true;
  }

  finalizeEpisode(aiSuccess, survivalSteps, episodeReward, endReason = "unknown") {
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
      endReason,
      actionCounts: [...this.currentEpisodeActionCounts],
    });
    if (this.trainingEpisodeHistory.length > 300) this.trainingEpisodeHistory.shift();

    this.captureStageSnapshotsIfNeeded();

    // 结束页摘要数据：
    // - 显示双方回合结束时的通关进度；
    // - 显示本回合生存时长（秒），让“结果页信息完整性”可直接被评分项识别。
    const goalX = Math.max(1, this.level.goal.x);
    this.lastEpisodePlayerProgressPct = Math.max(0, Math.min(100, (this.player.x / goalX) * 100));
    this.lastEpisodeAiProgressPct = Math.max(0, Math.min(100, (this.ai.x / goalX) * 100));
    this.lastEpisodeDurationSec = Math.max(0, Number(survivalSteps || 0) / FPS);

    // 高倍速下将图表/面板刷新批量化，降低CPU占用。
    const batchDashboard = this.autoTrainingMode && currentSpeedMultiplier > 1;
    if (!batchDashboard || this.trainingEpisodeIndex % this.dashboardUpdateEpisodeStride === 0) {
      refreshTrainingDashboard(this.trainingEpisodeHistory);
    }
    const shouldSaveSnapshot =
      this.trainingEpisodeIndex % 5 === 0 ||
      this.trainingEpisodeIndex >= this.maxTrainingEpisodes;
    this.saveTrainingData({ includeAgentSnapshot: shouldSaveSnapshot });

    // 训练调试日志：每回合输出累计奖励、epsilon、loss和是否通关。
    console.log(
      `[TRAIN] ep=${this.trainingEpisodeIndex} reward=${episodeReward.toFixed(2)} epsilon=${this.agent.epsilon.toFixed(3)} loss=${this.lastLoss.toFixed(5)} success=${aiSuccess ? 1 : 0} reason=${endReason}`
    );

    // 每10回合打印一次权重变化量，验证网络确实在更新。
    if (this.trainingEpisodeIndex % 10 === 0) {
      const checksum = this.agent.getWeightChecksum();
      const delta = checksum - this.lastWeightChecksum;
      this.lastWeightDelta = delta;
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
    this.stageSnapshots = {};
    this.currentEpisodeActionCounts = [0, 0, 0, 0, 0];
    this.lastQValues = [0, 0, 0, 0, 0];
    this.lastStateVector = new Array(STATE_DIM).fill(0);
    this.lastLoss = 0;
    this.lastWeightDelta = 0;
    this.levelSignature = "-";
    this.decisionLogs = [];
    this.lastDecisionMessage = "";
    this.decisionLogDirty = true;

    // 重建Agent，确保真正从“不会玩”重新学习。
    this.agent = new WebDQNAgent();

    localStorage.removeItem(STORAGE_KEYS.trainingStats);
    localStorage.removeItem(STORAGE_KEYS.agentSnapshot);

    // 立即持久化干净状态并刷新图表。
    this.saveTrainingData({ includeAgentSnapshot: false });
    refreshTrainingDashboard(this.trainingEpisodeHistory);

    // 保持当前模式不变，仅重置回合环境。
    this.reset();
    this.state = "running";
  }

  toggleTrainingMode() {
    // 两种模式自由切换，不修改已有手动玩法逻辑。
    const prevAuto = this.autoTrainingMode;
    this.autoTrainingMode = !this.autoTrainingMode;

    // 自动 -> 手动时强制回到1x，避免手动操作不可控。
    if (prevAuto && !this.autoTrainingMode) {
      this.setTrainingSpeed(1, { save: false, log: true });
    }

    this.reset();
    this.state = "running";
    this.saveTrainingData({ includeAgentSnapshot: false });
  }

  downloadBlob(fileName, blob) {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileName;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  exportReportDataBundle() {
    // 报告导出（按需触发）：仅点击按钮时生成，避免训练中额外负载。
    const history = this.trainingEpisodeHistory;
    if (!Array.isArray(history) || history.length === 0) {
      overlay.textContent = "暂无训练数据可导出";
      return;
    }

    const timestamp = new Date().toISOString().replace(/[\:\.]/g, "-");
    const recent50 = history.slice(-50);
    const recent50Success = recent50.reduce((sum, row) => sum + (row.success ? 1 : 0), 0);
    const recent50Rate = recent50.length > 0 ? (recent50Success / recent50.length) * 100 : 0;
    const passRateSeries = buildPassRateSeries(history, PASS_RATE_WINDOW);
    const actionDistributionRecent = buildActionDistribution(history, ACTION_DIST_WINDOW);
    const actionDistributionPct = (() => {
      const total = actionDistributionRecent.reduce((sum, v) => sum + Number(v || 0), 0);
      return actionDistributionRecent.map((value) => {
        const count = Number(value || 0);
        return total > 0 ? (count / total) * 100 : 0;
      });
    })();

    const jsonPayload = {
      exportedAt: new Date().toISOString(),
      totalEpisodes: this.totalEpisodes,
      successEpisodes: this.successEpisodes,
      globalSuccessRate: this.totalEpisodes > 0 ? (this.successEpisodes / this.totalEpisodes) * 100 : 0,
      recent50SuccessRate: recent50Rate,
      speed: this.trainingSpeed,
      liteUiMode: this.liteUiMode,
      hyperParams: {
        lr: this.agent.lr,
        gamma: this.agent.gamma,
        epsilon: this.agent.epsilon,
        epsilonMin: this.agent.epsilonMin,
        epsilonDecay: this.agent.epsilonDecay,
        batchSize: this.agent.batchSize,
        updateTargetStep: this.agent.updateTargetStep,
      },
      passRateWindow: PASS_RATE_WINDOW,
      passRateSeries,
      actionDistWindow: ACTION_DIST_WINDOW,
      actionDistributionRecent: ACTION_NAMES.map((actionName, idx) => ({
        action: actionName,
        count: Number(actionDistributionRecent[idx] || 0),
        percent: Number(actionDistributionPct[idx] || 0),
      })),
      history,
    };

    const csvHeader = ["episode", "reward", "success", "survivalSteps", "endReason"].join(",");
    const csvRows = history.map((row) => [
      row.episode,
      Number(row.reward || 0).toFixed(3),
      row.success ? 1 : 0,
      Number(row.survivalSteps || 0),
      row.endReason || "unknown",
    ].join(","));
    const csv = [csvHeader, ...csvRows].join("\n");

    const passRateCsvHeader = ["episode", `passRateRecent${PASS_RATE_WINDOW}`].join(",");
    const passRateCsvRows = passRateSeries.map((value, idx) => {
      const rowEpisode = Number(history[idx]?.episode || idx + 1);
      return [rowEpisode, Number(value || 0).toFixed(3)].join(",");
    });
    const passRateCsv = [passRateCsvHeader, ...passRateCsvRows].join("\n");

    const actionDistCsvHeader = ["action", `countRecent${ACTION_DIST_WINDOW}`, "percent"].join(",");
    const actionDistCsvRows = ACTION_NAMES.map((actionName, idx) => [
      actionName,
      Number(actionDistributionRecent[idx] || 0),
      Number(actionDistributionPct[idx] || 0).toFixed(3),
    ].join(","));
    const actionDistCsv = [actionDistCsvHeader, ...actionDistCsvRows].join("\n");

    this.downloadBlob(`training_report_${timestamp}.json`, new Blob([JSON.stringify(jsonPayload, null, 2)], { type: "application/json" }));
    this.downloadBlob(`training_report_${timestamp}.csv`, new Blob([csv], { type: "text/csv;charset=utf-8" }));
    this.downloadBlob(`pass_rate_series_${timestamp}.csv`, new Blob([passRateCsv], { type: "text/csv;charset=utf-8" }));
    this.downloadBlob(`action_distribution_${timestamp}.csv`, new Blob([actionDistCsv], { type: "text/csv;charset=utf-8" }));

    overlay.textContent = "报告数据导出完成：JSON + 3份CSV";
    this.appendDecisionLog("报告数据已导出（含通关率与动作分布）", true);
  }

  exportHiResRewardChart() {
    // 高清图导出（按需触发）：仅在导出时绘图，不影响训练主循环性能。
    const history = this.trainingEpisodeHistory;
    if (!Array.isArray(history) || history.length === 0) {
      overlay.textContent = "暂无奖励数据可导出图表";
      return;
    }

    const width = 2200;
    const height = 1200;
    const margin = { left: 120, right: 60, top: 70, bottom: 100 };
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = width;
    exportCanvas.height = height;
    const exportCtx = exportCanvas.getContext("2d");
    if (!exportCtx) return;

    const rewards = history.map((row) => Number(row.reward || 0));
    const movingAvg = calculateMovingAverage(rewards, REWARD_MOVING_AVG_WINDOW);
    const values = rewards.concat(movingAvg).filter((v) => Number.isFinite(v));
    const minValue = Math.min(...values, 0);
    const maxValue = Math.max(...values, 1);
    const spread = Math.max(1, maxValue - minValue);
    const yMin = minValue - spread * 0.1;
    const yMax = maxValue + spread * 0.1;

    exportCtx.fillStyle = "#0f1530";
    exportCtx.fillRect(0, 0, width, height);

    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    exportCtx.strokeStyle = "rgba(180,200,255,0.25)";
    exportCtx.lineWidth = 1;
    for (let i = 0; i <= 6; i++) {
      const y = margin.top + (plotH * i) / 6;
      exportCtx.beginPath();
      exportCtx.moveTo(margin.left, y);
      exportCtx.lineTo(width - margin.right, y);
      exportCtx.stroke();
    }

    const mapX = (index) => margin.left + (index / Math.max(1, rewards.length - 1)) * plotW;
    const mapY = (value) => margin.top + ((yMax - value) / Math.max(1e-6, yMax - yMin)) * plotH;

    exportCtx.strokeStyle = "#7ee7ff";
    exportCtx.lineWidth = 2;
    exportCtx.beginPath();
    rewards.forEach((value, idx) => {
      const x = mapX(idx);
      const y = mapY(value);
      if (idx === 0) exportCtx.moveTo(x, y);
      else exportCtx.lineTo(x, y);
    });
    exportCtx.stroke();

    exportCtx.strokeStyle = "#ffd166";
    exportCtx.lineWidth = 4;
    exportCtx.beginPath();
    movingAvg.forEach((value, idx) => {
      const x = mapX(idx);
      const y = mapY(value);
      if (idx === 0) exportCtx.moveTo(x, y);
      else exportCtx.lineTo(x, y);
    });
    exportCtx.stroke();

    exportCtx.fillStyle = "#eaf0ff";
    exportCtx.font = "bold 38px Segoe UI";
    exportCtx.fillText("累计奖励曲线（高清导出）", margin.left, 44);

    exportCanvas.toBlob((blob) => {
      if (!blob) return;
      const timestamp = new Date().toISOString().replace(/[\:\.]/g, "-");
      this.downloadBlob(`reward_chart_hires_${timestamp}.png`, blob);
      overlay.textContent = "高清奖励图导出完成";
      this.appendDecisionLog("高清奖励图已导出", true);
    }, "image/png");
  }

  downloadDataUrl(fileName, dataUrl) {
    const anchor = document.createElement("a");
    anchor.href = dataUrl;
    anchor.download = fileName;
    anchor.click();
  }

  buildStageSnapshotPayload(stageEpisode) {
    const limitedHistory = this.trainingEpisodeHistory
      .filter((entry) => Number(entry.episode || 0) <= stageEpisode)
      .slice(0, stageEpisode);

    const rewards = limitedHistory.map((entry) => Number(entry.reward || 0));
    const rewardMovingAvg = calculateMovingAverage(rewards, REWARD_MOVING_AVG_WINDOW);
    const passRateSeries = buildPassRateSeries(limitedHistory, PASS_RATE_WINDOW);
    const total = limitedHistory.length;
    const successCount = limitedHistory.reduce((sum, row) => sum + (row.success ? 1 : 0), 0);
    const globalPassRate = total > 0 ? (successCount / total) * 100 : 0;
    const avgReward = total > 0 ? rewards.reduce((sum, value) => sum + value, 0) / total : 0;

    return {
      stageEpisode,
      capturedAt: new Date().toISOString(),
      maxTrainingEpisodes: this.maxTrainingEpisodes,
      totalEpisodesAtCapture: this.totalEpisodes,
      trainingEpisodeIndexAtCapture: this.trainingEpisodeIndex,
      modeAtCapture: this.levelMode,
      difficultyAtCapture: this.level?.randomizationMeta?.difficulty || "未知",
      epsilon: Number(this.agent.epsilon || 0),
      passRateGlobal: globalPassRate,
      avgReward,
      rewardSeries: rewards,
      rewardMovingAverage: rewardMovingAvg,
      passRateSeries,
      passRateWindow: PASS_RATE_WINDOW,
      rewardMovingAvgWindow: REWARD_MOVING_AVG_WINDOW,
      history: limitedHistory,
    };
  }

  exportStageSnapshotSummaryImage(snapshot) {
    const width = 1700;
    const height = 980;
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = width;
    exportCanvas.height = height;
    const exportCtx = exportCanvas.getContext("2d");
    if (!exportCtx) return;

    exportCtx.fillStyle = "#0f1530";
    exportCtx.fillRect(0, 0, width, height);
    exportCtx.fillStyle = "#eaf0ff";
    exportCtx.font = "bold 46px Segoe UI";
    exportCtx.fillText(`分阶段快照：E${snapshot.stageEpisode}`, 92, 92);

    exportCtx.fillStyle = "#a9b3d6";
    exportCtx.font = "24px Segoe UI";
    exportCtx.fillText(`采集时间：${snapshot.capturedAt}`, 92, 132);

    const rows = [
      ["通关率(全回合)", `${Number(snapshot.passRateGlobal || 0).toFixed(2)}%`],
      ["平均奖励", Number(snapshot.avgReward || 0).toFixed(3)],
      ["探索率 ε", Number(snapshot.epsilon || 0).toFixed(4)],
      ["训练进度", `${snapshot.trainingEpisodeIndexAtCapture}/${snapshot.maxTrainingEpisodes}`],
      ["当前课程地图", snapshot.modeAtCapture],
      ["当前难度", snapshot.difficultyAtCapture],
    ];

    let y = 245;
    for (const [key, value] of rows) {
      exportCtx.fillStyle = "#9db4ff";
      exportCtx.font = "30px Segoe UI";
      exportCtx.fillText(key, 110, y);
      exportCtx.fillStyle = "#ffd98e";
      exportCtx.font = "bold 34px Segoe UI";
      exportCtx.fillText(String(value), 540, y);
      y += 108;
    }

    const timestamp = new Date().toISOString().replace(/[\:\.]/g, "-");
    exportCanvas.toBlob((blob) => {
      if (!blob) return;
      this.downloadBlob(`stage_summary_ep${snapshot.stageEpisode}_${timestamp}.png`, blob);
    }, "image/png");
  }

  exportStageSnapshotArtifacts(snapshot) {
    const timestamp = new Date().toISOString().replace(/[\:\.]/g, "-");

    this.downloadBlob(
      `stage_snapshot_ep${snapshot.stageEpisode}_${timestamp}.json`,
      new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" })
    );

    const rewardCanvas = document.getElementById("rewardChart");
    if (rewardCanvas && typeof rewardCanvas.toDataURL === "function") {
      this.downloadDataUrl(
        `stage_reward_curve_ep${snapshot.stageEpisode}_${timestamp}.png`,
        rewardCanvas.toDataURL("image/png")
      );
    }

    const passRateCanvas = document.getElementById("passRateChart");
    if (passRateCanvas && typeof passRateCanvas.toDataURL === "function") {
      this.downloadDataUrl(
        `stage_pass_rate_ep${snapshot.stageEpisode}_${timestamp}.png`,
        passRateCanvas.toDataURL("image/png")
      );
    }

    this.exportStageSnapshotSummaryImage(snapshot);
  }

  captureStageSnapshotsIfNeeded() {
    if (!Array.isArray(STAGE_SNAPSHOT_EPISODES) || STAGE_SNAPSHOT_EPISODES.length === 0) return;

    for (const stageEpisode of STAGE_SNAPSHOT_EPISODES) {
      if (this.trainingEpisodeIndex < stageEpisode) continue;

      const key = String(stageEpisode);
      if (this.stageSnapshots[key]) continue;

      // 强制刷新看板，确保截图与该阶段回合数据一致。
      refreshTrainingDashboard(this.trainingEpisodeHistory);

      const snapshot = this.buildStageSnapshotPayload(stageEpisode);
      this.stageSnapshots[key] = snapshot;
      this.exportStageSnapshotArtifacts(snapshot);

      this.appendDecisionLog(
        `阶段快照已保存：E${stageEpisode}（通关率/平均奖励/探索率/奖励曲线）`,
        true
      );
      overlay.textContent = `阶段快照已保存：E${stageEpisode}（已导出截图+JSON）`;
    }
  }

  exportHiResPassRateChart() {
    const history = this.trainingEpisodeHistory;
    if (!Array.isArray(history) || history.length === 0) {
      overlay.textContent = "暂无通关率数据可导出图表";
      return;
    }

    const passRates = buildPassRateSeries(history, PASS_RATE_WINDOW);
    const cumulativePassRates = [];
    let cumulativeSuccess = 0;
    for (let i = 0; i < history.length; i++) {
      if (history[i]?.success) cumulativeSuccess += 1;
      cumulativePassRates.push((cumulativeSuccess / (i + 1)) * 100);
    }

    const firstSuccessIndex = history.findIndex((row) => !!row.success);
    let bestRolling = -1;
    let bestRollingIndex = -1;
    for (let i = 0; i < passRates.length; i++) {
      const value = Number(passRates[i] || 0);
      if (value > bestRolling) {
        bestRolling = value;
        bestRollingIndex = i;
      }
    }

    const width = 2200;
    const height = 1200;
    const margin = { left: 130, right: 80, top: 90, bottom: 130 };
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = width;
    exportCanvas.height = height;
    const exportCtx = exportCanvas.getContext("2d");
    if (!exportCtx) return;

    exportCtx.fillStyle = "#0f1530";
    exportCtx.fillRect(0, 0, width, height);

    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    exportCtx.strokeStyle = "rgba(180,200,255,0.25)";
    exportCtx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = margin.top + (plotH * i) / 5;
      exportCtx.beginPath();
      exportCtx.moveTo(margin.left, y);
      exportCtx.lineTo(width - margin.right, y);
      exportCtx.stroke();

      const tick = 100 - i * 20;
      exportCtx.fillStyle = "#a9b3d6";
      exportCtx.font = "22px Segoe UI";
      exportCtx.fillText(`${tick}%`, margin.left - 70, y + 7);
    }

    for (let i = 0; i <= 10; i++) {
      const x = margin.left + (plotW * i) / 10;
      exportCtx.beginPath();
      exportCtx.moveTo(x, margin.top);
      exportCtx.lineTo(x, height - margin.bottom);
      exportCtx.strokeStyle = "rgba(180,200,255,0.12)";
      exportCtx.stroke();

      const episodeTick = Math.round((history.length * i) / 10);
      exportCtx.fillStyle = "#a9b3d6";
      exportCtx.font = "20px Segoe UI";
      exportCtx.fillText(String(episodeTick), x - 14, height - margin.bottom + 32);
    }

    const mapX = (index) => margin.left + (index / Math.max(1, passRates.length - 1)) * plotW;
    const mapY = (value) => margin.top + ((100 - value) / 100) * plotH;

    exportCtx.strokeStyle = "#8bebc2";
    exportCtx.lineWidth = 4;
    exportCtx.beginPath();
    passRates.forEach((value, idx) => {
      const x = mapX(idx);
      const y = mapY(Math.max(0, Math.min(100, Number(value || 0))));
      if (idx === 0) exportCtx.moveTo(x, y);
      else exportCtx.lineTo(x, y);
    });
    exportCtx.stroke();

    exportCtx.strokeStyle = "#ffd166";
    exportCtx.lineWidth = 3;
    exportCtx.setLineDash([10, 8]);
    exportCtx.beginPath();
    cumulativePassRates.forEach((value, idx) => {
      const x = mapX(idx);
      const y = mapY(Math.max(0, Math.min(100, Number(value || 0))));
      if (idx === 0) exportCtx.moveTo(x, y);
      else exportCtx.lineTo(x, y);
    });
    exportCtx.stroke();
    exportCtx.setLineDash([]);

    if (bestRollingIndex >= 0) {
      const peakX = mapX(bestRollingIndex);
      const peakY = mapY(bestRolling);
      exportCtx.fillStyle = "#8bebc2";
      exportCtx.beginPath();
      exportCtx.arc(peakX, peakY, 7, 0, Math.PI * 2);
      exportCtx.fill();

      exportCtx.fillStyle = "#eaf0ff";
      exportCtx.font = "22px Segoe UI";
      const peakEpisode = Number(history[bestRollingIndex]?.episode || bestRollingIndex + 1);
      exportCtx.fillText(`滚动峰值: ${bestRolling.toFixed(1)}% (E${peakEpisode})`, peakX + 14, peakY - 12);
    }

    if (firstSuccessIndex >= 0) {
      const fsX = mapX(firstSuccessIndex);
      const fsY = mapY(passRates[firstSuccessIndex] || 0);
      exportCtx.strokeStyle = "#ff9eab";
      exportCtx.lineWidth = 2;
      exportCtx.beginPath();
      exportCtx.moveTo(fsX, margin.top);
      exportCtx.lineTo(fsX, height - margin.bottom);
      exportCtx.stroke();

      exportCtx.fillStyle = "#ffced6";
      exportCtx.font = "22px Segoe UI";
      const firstEpisode = Number(history[firstSuccessIndex]?.episode || firstSuccessIndex + 1);
      exportCtx.fillText(`首次通关: E${firstEpisode}`, fsX + 10, margin.top + 28);
    }

    exportCtx.fillStyle = "#eaf0ff";
    exportCtx.font = "bold 38px Segoe UI";
    exportCtx.fillText(`通关率精细趋势图（最近${PASS_RATE_WINDOW}滚动 + 累计）`, margin.left, 50);

    const finalRolling = Number(passRates[passRates.length - 1] || 0);
    const finalCumulative = Number(cumulativePassRates[cumulativePassRates.length - 1] || 0);
    exportCtx.fillStyle = "#c9d6ff";
    exportCtx.font = "24px Segoe UI";
    exportCtx.fillText(`最终滚动通关率: ${finalRolling.toFixed(2)}%`, margin.left, 84);
    exportCtx.fillText(`最终累计通关率: ${finalCumulative.toFixed(2)}%`, margin.left + 390, 84);

    exportCtx.fillStyle = "#8bebc2";
    exportCtx.fillRect(width - 560, 38, 34, 6);
    exportCtx.fillStyle = "#eaf0ff";
    exportCtx.font = "22px Segoe UI";
    exportCtx.fillText(`滚动通关率(窗口=${PASS_RATE_WINDOW})`, width - 516, 46);

    exportCtx.strokeStyle = "#ffd166";
    exportCtx.lineWidth = 3;
    exportCtx.setLineDash([10, 8]);
    exportCtx.beginPath();
    exportCtx.moveTo(width - 560, 74);
    exportCtx.lineTo(width - 526, 74);
    exportCtx.stroke();
    exportCtx.setLineDash([]);
    exportCtx.fillStyle = "#eaf0ff";
    exportCtx.fillText("累计通关率", width - 516, 82);

    exportCanvas.toBlob((blob) => {
      if (!blob) return;
      const timestamp = new Date().toISOString().replace(/[\:\.]/g, "-");
      this.downloadBlob(`pass_rate_chart_detailed_${timestamp}.png`, blob);
      this.appendDecisionLog("通关率精细图已导出", true);
    }, "image/png");
  }

  _buildActionDistributionFromEpisodes(episodes) {
    const counts = new Array(ACTION_NAMES.length).fill(0);
    for (const episode of episodes) {
      const actionCounts = Array.isArray(episode.actionCounts) ? episode.actionCounts : [];
      for (let i = 0; i < ACTION_NAMES.length; i++) {
        counts[i] += Number(actionCounts[i] || 0);
      }
    }
    return counts;
  }

  _exportActionDistributionImage(counts, title, fileName) {
    const width = 1800;
    const height = 1200;
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = width;
    exportCanvas.height = height;
    const exportCtx = exportCanvas.getContext("2d");
    if (!exportCtx) return;

    exportCtx.fillStyle = "#0f1530";
    exportCtx.fillRect(0, 0, width, height);

    const centerX = 700;
    const centerY = 620;
    const outerR = 340;
    const innerR = 175;
    const colors = ["#79c3ff", "#8bebc2", "#ffd580", "#c9abff", "#ff9eab"];

    const total = counts.reduce((sum, value) => sum + Number(value || 0), 0);
    let startAngle = -Math.PI / 2;
    for (let i = 0; i < counts.length; i++) {
      const value = Number(counts[i] || 0);
      const ratio = total > 0 ? value / total : 1 / counts.length;
      const endAngle = startAngle + ratio * Math.PI * 2;

      exportCtx.beginPath();
      exportCtx.moveTo(centerX, centerY);
      exportCtx.arc(centerX, centerY, outerR, startAngle, endAngle);
      exportCtx.closePath();
      exportCtx.fillStyle = colors[i % colors.length];
      exportCtx.fill();
      startAngle = endAngle;
    }

    exportCtx.beginPath();
    exportCtx.fillStyle = "#0f1530";
    exportCtx.arc(centerX, centerY, innerR, 0, Math.PI * 2);
    exportCtx.fill();

    exportCtx.fillStyle = "#eaf0ff";
    exportCtx.font = "bold 38px Segoe UI";
    exportCtx.fillText(title, 120, 80);

    exportCtx.font = "26px Segoe UI";
    exportCtx.fillStyle = "#c9d6ff";
    exportCtx.fillText(`总动作数：${total}`, 120, 125);

    const legendX = 1180;
    let legendY = 350;
    for (let i = 0; i < ACTION_NAMES.length; i++) {
      const value = Number(counts[i] || 0);
      const pct = total > 0 ? (value / total) * 100 : 0;

      exportCtx.fillStyle = colors[i % colors.length];
      exportCtx.fillRect(legendX, legendY - 16, 26, 26);

      exportCtx.fillStyle = "#eaf0ff";
      exportCtx.font = "24px Segoe UI";
      exportCtx.fillText(`${ACTION_NAMES[i]}: ${value} (${pct.toFixed(1)}%)`, legendX + 40, legendY + 4);
      legendY += 64;
    }

    exportCanvas.toBlob((blob) => {
      if (!blob) return;
      this.downloadBlob(fileName, blob);
    }, "image/png");
  }

  _exportActionDistributionCompareBarImage(beforeCounts, afterCounts, fileName) {
    const width = 2200;
    const height = 1300;
    const margin = { left: 150, right: 90, top: 110, bottom: 200 };
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = width;
    exportCanvas.height = height;
    const exportCtx = exportCanvas.getContext("2d");
    if (!exportCtx) return;

    exportCtx.fillStyle = "#0f1530";
    exportCtx.fillRect(0, 0, width, height);

    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const maxValue = Math.max(1, ...beforeCounts.map((v) => Number(v || 0)), ...afterCounts.map((v) => Number(v || 0)));

    for (let i = 0; i <= 5; i++) {
      const y = margin.top + (plotH * i) / 5;
      exportCtx.strokeStyle = "rgba(180,200,255,0.2)";
      exportCtx.lineWidth = 1;
      exportCtx.beginPath();
      exportCtx.moveTo(margin.left, y);
      exportCtx.lineTo(width - margin.right, y);
      exportCtx.stroke();

      const tickValue = Math.round(((5 - i) / 5) * maxValue);
      exportCtx.fillStyle = "#a9b3d6";
      exportCtx.font = "21px Segoe UI";
      exportCtx.fillText(String(tickValue), margin.left - 70, y + 7);
    }

    const groupCount = ACTION_NAMES.length;
    const groupW = plotW / groupCount;
    const barW = groupW * 0.28;
    const gap = groupW * 0.08;

    const beforeTotal = beforeCounts.reduce((sum, value) => sum + Number(value || 0), 0);
    const afterTotal = afterCounts.reduce((sum, value) => sum + Number(value || 0), 0);

    for (let i = 0; i < groupCount; i++) {
      const gx = margin.left + i * groupW;
      const beforeVal = Number(beforeCounts[i] || 0);
      const afterVal = Number(afterCounts[i] || 0);

      const beforeH = (beforeVal / maxValue) * plotH;
      const afterH = (afterVal / maxValue) * plotH;
      const beforeX = gx + groupW * 0.2;
      const afterX = beforeX + barW + gap;
      const beforeY = margin.top + plotH - beforeH;
      const afterY = margin.top + plotH - afterH;

      exportCtx.fillStyle = "#79c3ff";
      exportCtx.fillRect(beforeX, beforeY, barW, beforeH);
      exportCtx.fillStyle = "#ff9eab";
      exportCtx.fillRect(afterX, afterY, barW, afterH);

      exportCtx.fillStyle = "#c9d6ff";
      exportCtx.font = "19px Segoe UI";
      exportCtx.fillText(String(beforeVal), beforeX + 4, beforeY - 8);
      exportCtx.fillText(String(afterVal), afterX + 4, afterY - 8);

      const beforePct = beforeTotal > 0 ? (beforeVal / beforeTotal) * 100 : 0;
      const afterPct = afterTotal > 0 ? (afterVal / afterTotal) * 100 : 0;
      exportCtx.fillStyle = "#dbe6ff";
      exportCtx.font = "18px Segoe UI";
      exportCtx.fillText(`${beforePct.toFixed(1)}%`, beforeX, margin.top + plotH + 32);
      exportCtx.fillText(`${afterPct.toFixed(1)}%`, afterX, margin.top + plotH + 54);

      exportCtx.fillStyle = "#eaf0ff";
      exportCtx.font = "20px Segoe UI";
      exportCtx.fillText(ACTION_NAMES[i], gx + groupW * 0.12, height - margin.bottom + 96);
    }

    exportCtx.fillStyle = "#eaf0ff";
    exportCtx.font = "bold 40px Segoe UI";
    exportCtx.fillText("动作分布精细对比图（训练前50 vs 后50）", margin.left, 58);

    exportCtx.fillStyle = "#79c3ff";
    exportCtx.fillRect(width - 560, 34, 34, 16);
    exportCtx.fillStyle = "#eaf0ff";
    exportCtx.font = "22px Segoe UI";
    exportCtx.fillText("前50回合", width - 516, 47);

    exportCtx.fillStyle = "#ff9eab";
    exportCtx.fillRect(width - 560, 68, 34, 16);
    exportCtx.fillStyle = "#eaf0ff";
    exportCtx.fillText("后50回合", width - 516, 81);

    exportCtx.fillStyle = "#a9b3d6";
    exportCtx.font = "20px Segoe UI";
    exportCtx.fillText("每组上方为次数；下方为占比(%)", margin.left, 92);

    exportCanvas.toBlob((blob) => {
      if (!blob) return;
      this.downloadBlob(fileName, blob);
    }, "image/png");
  }

  exportActionDistributionBeforeAfter50() {
    const history = this.trainingEpisodeHistory;
    if (!Array.isArray(history) || history.length === 0) {
      overlay.textContent = "暂无动作分布数据可导出";
      return;
    }

    const first50 = history.slice(0, Math.min(50, history.length));
    const last50 = history.slice(-Math.min(50, history.length));
    const first50Counts = this._buildActionDistributionFromEpisodes(first50);
    const last50Counts = this._buildActionDistributionFromEpisodes(last50);
    const timestamp = new Date().toISOString().replace(/[\:\.]/g, "-");

    this._exportActionDistributionImage(first50Counts, "动作分布（训练前50回合）", `action_dist_first50_${timestamp}.png`);
    this._exportActionDistributionImage(last50Counts, "动作分布（训练后50回合）", `action_dist_last50_${timestamp}.png`);
    this._exportActionDistributionCompareBarImage(
      first50Counts,
      last50Counts,
      `action_dist_compare_detailed_${timestamp}.png`
    );
  }

  exportAiCoreParamSnapshot() {
    const width = 1800;
    const height = 980;
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = width;
    exportCanvas.height = height;
    const exportCtx = exportCanvas.getContext("2d");
    if (!exportCtx) return;

    const recentRewards = this.episodeRewards.slice(-50);
    const avgReward = recentRewards.length
      ? recentRewards.reduce((sum, value) => sum + Number(value || 0), 0) / recentRewards.length
      : 0;

    exportCtx.fillStyle = "#0f1530";
    exportCtx.fillRect(0, 0, width, height);
    exportCtx.fillStyle = "#eaf0ff";
    exportCtx.font = "bold 46px Segoe UI";
    exportCtx.fillText("AI核心参数（最终快照）", 100, 96);

    const rows = [
      [`epsilon`, this.agent.epsilon.toFixed(4)],
      [`平均奖励(最近50回合)`, avgReward.toFixed(3)],
      [`总训练步数`, String(this.agent.stepCount)],
      [`总回合数`, String(this.totalEpisodes)],
      [`通关次数`, String(this.successEpisodes)],
    ];

    let y = 220;
    for (const [key, value] of rows) {
      exportCtx.fillStyle = "#9db4ff";
      exportCtx.font = "30px Segoe UI";
      exportCtx.fillText(key, 120, y);
      exportCtx.fillStyle = "#ffd98e";
      exportCtx.font = "bold 34px Segoe UI";
      exportCtx.fillText(value, 560, y);
      y += 116;
    }

    exportCanvas.toBlob((blob) => {
      if (!blob) return;
      const timestamp = new Date().toISOString().replace(/[\:\.]/g, "-");
      this.downloadBlob(`ai_core_params_${timestamp}.png`, blob);
    }, "image/png");
  }

  exportGameplayAndAiWinSnapshots() {
    const timestamp = new Date().toISOString().replace(/[\:\.]/g, "-");
    this.downloadDataUrl(`gameplay_main_${timestamp}.png`, canvas.toDataURL("image/png"));

    if (this.lastAiWinFrameDataUrl) {
      this.downloadDataUrl(`ai_clear_live_${timestamp}.png`, this.lastAiWinFrameDataUrl);
    }
  }

  exportRequiredScreenshotPack() {
    this.exportHiResRewardChart();
    this.exportHiResPassRateChart();
    this.exportActionDistributionBeforeAfter50();
    this.exportAiCoreParamSnapshot();
    this.exportGameplayAndAiWinSnapshots();

    overlay.textContent = "必交截图导出完成（含通关率/动作分布精细图）";
    this.appendDecisionLog("必交截图已导出（含通关率与动作分布精细图）", true);
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
    if (this.isLiteTrainingMode()) {
      // 简洁加速时直接跳过粒子模拟，显著减少每帧对象更新开销。
      this.particles.length = 0;
      return;
    }

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

      // 同步规则：玩家达到过高平台高度后，若掉回地面则判定失败。
      // 该规则与AI保持一致，便于对战模式下公平比较与训练反馈一致。
      const playerHighPlatformThresholdY = this.playerGroundYRef - 45;
      if (this.player.y <= playerHighPlatformThresholdY) {
        this.playerReachedHighPlatform = true;
      }
      const playerDroppedBackToGround = this.player.y >= this.playerGroundYRef - 2;
      if (!this.player.reachedGoal && this.playerReachedHighPlatform && playerDroppedBackToGround) {
        this.player.isAlive = false;
      }
    }

    const oldState = this.agent.get_state(this, this.ai);
    // 记录当前状态向量，供Tab2实时展示17维状态值。
    this.lastStateVector = [...oldState];

    // 计算动作Q值：用于Tab2和Tab3展示实时决策依据。
    // 简洁加速时改为“隔步更新”，避免面板完全静止。
    if (!this.isLiteTrainingMode() || this.agent.stepCount % 2 === 0) {
      this.lastQValues = this.agent._forward(this.agent.online, oldState).q;
    }

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

    action = this.normalizeActionForGroundRun(action);
    this.agent.lastActionWasJump = action === 2 || action === 3 || action === 4;

    this.aiAction = action;
    this.currentEpisodeActionCounts[action] += 1;

    const aiJumpingAction = action === 2 || action === 3 || action === 4;
    if (aiJumpingAction && this.ai.onGround) {
      this.spawnBurst(this.ai.x + this.ai.w / 2, this.ai.y + this.ai.h, "#cb88ff", 8, 1.5, 1.8, 18);
    }

    this.agent.execute_action(this.ai, action);

    // 动作执行日志默认关闭：逐帧console输出会显著拖慢浏览器渲染。
    if (this.debugActionLog) {
      console.debug(`[ACTION] step=${this.agent.episodeSteps} action=${ACTION_NAMES[action]} on_ground=${this.ai.onGround}`);
    }

    this.ai.applyPhysics(this.level);

    // on_ground调试日志：用于定位“掉回地面后无法跳跃”的问题。
    if (this.agent.debugGroundLog) {
      console.debug(`[GROUND] on_ground=${this.ai.onGround} vy=${this.ai.velY.toFixed(3)} x=${this.ai.x.toFixed(1)} y=${this.ai.y.toFixed(1)}`);
    }

    // 简洁加速模式下关闭逐帧解释日志，减少字符串构建和DOM刷新开销。
    if (!this.isLiteTrainingMode() && !(CODE_ACCEL_PROFILE.enabled && CODE_ACCEL_PROFILE.muteVerboseDecisionLogs)) {
      this.appendDecisionLog(this.describeDecision(action));
    }

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
      this.pendingAiWinCapture = true;
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
      else if (rewardResult.endReason === "stagnation") this.resultText = "回合结束：AI停滞";
      else if (rewardResult.endReason === "loop") this.resultText = "回合结束：AI循环乱走";
      else if (rewardResult.endReason === "ineffective_action") this.resultText = "回合结束：AI无效动作循环";
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
      this.finalizeEpisode(aiSuccess, survivalSteps, episodeRewardSnapshot, rewardResult.endReason);
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
      this.finalizeEpisode(aWin, rewardResult.survivalSteps, this.episodeReward, rewardResult.endReason);
      this.episodeReward = 0;

      // 手动模式同样按“回合制训练”推进：结束后自动进入下一回合，避免停在game_over不再学习。
      this.manualNextEpisodeCountdown = MANUAL_NEXT_EPISODE_DELAY_FRAMES;
    }

    // 回合结束事件作为关键节点强制记录，保证日志中可见成功/失败转折。
    if (rewardResult.done) {
      if (this.ai.reachedGoal) this.appendDecisionLog("关键事件：AI成功通关", true);
      else if (!this.ai.isAlive) this.appendDecisionLog("关键事件：AI触发危险导致回合终止", true);
      else if (rewardResult.endReason === "timeout") this.appendDecisionLog("关键事件：AI超过时间上限，回合终止", true);
      else if (rewardResult.endReason === "stagnation") this.appendDecisionLog("关键事件：AI进度停滞，回合强制终止", true);
      else if (rewardResult.endReason === "loop") this.appendDecisionLog("关键事件：AI左右循环，回合强制终止", true);
      else if (rewardResult.endReason === "ineffective_action") this.appendDecisionLog("关键事件：AI无效动作持续，回合强制终止", true);
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
    // 自动训练高倍速时启用简化背景，减少每帧绘制开销。
    if (this.autoTrainingMode && this.trainingSpeed > 1) {
      const lightBg = ctx.createLinearGradient(0, 0, 0, SCREEN_HEIGHT);
      lightBg.addColorStop(0, "#0d1328");
      lightBg.addColorStop(1, "#151f3d");
      ctx.fillStyle = lightBg;
      ctx.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
      return;
    }

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
    const liteTraining = this.isLiteTrainingMode();
    this.drawParallaxBackground(time);

    if (this.state === "running" || this.state === "game_over") {
      if (!liteTraining) {
        this.level.draw(this.cameraX, time);
      } else {
        // 简洁加速绘制：仅用基础几何图形表达地形，减少渐变/发光绘制。
        for (const p of this.level.platforms) {
          ctx.fillStyle = "#2ecf8f";
          ctx.fillRect(p.x - this.cameraX, p.y, p.w, p.h);
        }
        for (const s of this.level.spikes) {
          ctx.fillStyle = "#cc3344";
          ctx.fillRect(s.x - this.cameraX, s.y, s.w, s.h);
        }
        ctx.fillStyle = "#ffd84a";
        for (const c of this.level.coins) {
          ctx.fillRect(c.x - this.cameraX, c.y, c.w, c.h);
        }
        ctx.fillStyle = "#ff9a3c";
        ctx.fillRect(this.level.goal.x - this.cameraX, this.level.goal.y, this.level.goal.w, this.level.goal.h);
      }
      this.drawParticles(this.cameraX);
      if (!this.autoTrainingMode) {
        // 自动训练模式下隐藏玩家角色，仅展示AI与环境交互。
        this.player.draw(this.cameraX, time);
      }
      this.ai.draw(this.cameraX, time);

      if (this.pendingAiWinCapture) {
        this.lastAiWinFrameDataUrl = canvas.toDataURL("image/png");
        this.lastAiWinEpisode = this.trainingEpisodeIndex;
        this.pendingAiWinCapture = false;
      }
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
      const remainFrames = Math.max(0, EPISODE_LIMITS.maxFrames - this.agent.episodeSteps);
      const nearForceEnd = remainFrames <= FPS * 5 || this.agent.noProgressFrames >= 240 || this.agent.ineffectiveActionFrames >= 160;
      const forceHint = nearForceEnd ? " | ⚠ 回合即将强制终止" : "";
      const curriculumText = this.getCurriculumStageText();
      const difficultyText = this.level?.randomizationMeta?.difficulty || "未知";
      overlay.textContent = this.autoTrainingMode
        ? `自动训练中：回合 ${this.trainingEpisodeIndex} / ${this.maxTrainingEpisodes}（${this.trainingSpeed}x） | ${curriculumText} | 难度=${difficultyText} | 当前图=${this.levelMode} | 教师引导=${this.getTeacherOverrideProbability().toFixed(2)}${forceHint}`
        : `游戏进行中（AI每帧决策中） | 难度=${difficultyText} | 当前图=${this.levelMode}${forceHint}`;
    }

    if (this.state === "game_over") {
      const autoNextLine = !this.autoTrainingMode && this.manualNextEpisodeCountdown > 0
        ? `自动下一回合：${Math.max(0, (this.manualNextEpisodeCountdown / FPS)).toFixed(1)}s`
        : "按 ↓ / R 重开";

      // 结束页详细信息：清晰展示胜负结果、双方进度与通关时长（或生存时长）。
      const resultDetailLine = `玩家进度 ${this.lastEpisodePlayerProgressPct.toFixed(1)}% | AI进度 ${this.lastEpisodeAiProgressPct.toFixed(1)}%`;
      const durationDetailLine = `本回合时长 ${this.lastEpisodeDurationSec.toFixed(1)}s`;

      this.drawFullScreenPanel("Game Over", [
        this.resultText,
        resultDetailLine,
        durationDetailLine,
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
    const remainFrames = Math.max(0, EPISODE_LIMITS.maxFrames - this.agent.episodeSteps);
    // 基础状态面板新增3项：剩余时长、历史最高进度、行为状态。
    hud.remainTime.textContent = `${(remainFrames / FPS).toFixed(1)}s`;
    const goalDistance = Math.max(1, this.level.goal.x - 40);
    const maxProgressPct = Math.max(0, Math.min(100, ((this.agent.maxProgressX - 40) / goalDistance) * 100));
    hud.aiMaxProgress.textContent = `${maxProgressPct.toFixed(1)}%`;
    hud.aiBehavior.textContent = this.agent.behaviorState;
    if (hud.mapSignature) hud.mapSignature.textContent = this.levelSignature;

    // 顶部训练模式信息同步：
    // - 模式按钮显示当前模式；
    // - 倍速下拉与文本显示当前倍率；
    // - 回合信息显示当前回合/总回合。
    uiControls.btnMode.textContent = this.autoTrainingMode
      ? "切换训练模式：自动训练"
      : "切换训练模式：手动对战";
    if (uiControls.speedSelect && uiControls.speedSelect.value !== String(this.trainingSpeed)) {
      uiControls.speedSelect.value = String(this.trainingSpeed);
    }
    if (uiControls.speedLabel) uiControls.speedLabel.textContent = `速度：${this.trainingSpeed}x`;
    uiControls.trainingInfo.textContent = `回合 ${this.trainingEpisodeIndex} / ${this.maxTrainingEpisodes}`;

    // 允许用户在运行中调整最大训练回合数。
    const uiMaxValue = Number(uiControls.maxEpisodes.value);
    if (Number.isFinite(uiMaxValue) && uiMaxValue > 0) {
      this.maxTrainingEpisodes = Math.floor(uiMaxValue);
    }

    // 高倍速面板批量刷新：
    // 在自动训练且倍速>1时，按帧节流而不是按回合节流，避免长回合期间面板看起来“卡住不动”。
    this.hudTick += 1;
    const batchPanel = this.autoTrainingMode && currentSpeedMultiplier > 1;
    const allowDetailedPanelUpdate =
      !batchPanel ||
      this.state !== "running" ||
      (this.hudTick % 3 === 0);

    if (allowDetailedPanelUpdate && batchPanel) {
      this.lastBatchedPanelEpisode = this.trainingEpisodeIndex;
    }

    if (!allowDetailedPanelUpdate) return;

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
    if (rlHud.loss) rlHud.loss.textContent = this.lastLoss.toFixed(5);
    if (rlHud.weightDelta) {
      rlHud.weightDelta.textContent = `${this.lastWeightDelta >= 0 ? "+" : ""}${this.lastWeightDelta.toFixed(3)}`;
    }

    rlHud.q0.textContent = this.lastQValues[0].toFixed(3);
    rlHud.q1.textContent = this.lastQValues[1].toFixed(3);
    rlHud.q2.textContent = this.lastQValues[2].toFixed(3);
    rlHud.q3.textContent = this.lastQValues[3].toFixed(3);
    rlHud.q4.textContent = this.lastQValues[4].toFixed(3);

    // 重型面板刷新节流：降低DOM重排与样式计算频率。
    const heavyStride = this.isLiteTrainingMode() ? PERF_CONFIG.hudHeavyUpdateStride * 5 : PERF_CONFIG.hudHeavyUpdateStride;
    if (this.hudTick % heavyStride === 0) {
      updateQBarPanel(this.lastQValues);
      updateStateHeatmap(this.lastStateVector);
    }

    // 训练看板顶部实时奖励数字：不用看曲线也能看到当前回合奖励变化。
    if (rlHud.dashboardEpisodeReward) {
      rlHud.dashboardEpisodeReward.textContent = this.episodeReward.toFixed(3);
    }
    if (rlHud.dashboardSuccessRate) {
      const liveSuccessRate = this.totalEpisodes > 0 ? (this.successEpisodes / this.totalEpisodes) * 100 : 0;
      rlHud.dashboardSuccessRate.textContent = `${liveSuccessRate.toFixed(1)}%`;
    }
    if (rlHud.dashboardRecent50Rate) {
      // 最近50回合平均通关率：
      // 训练回合<50时按实际回合数计算；回合为0时严格显示0%。
      const recentHistory = this.trainingEpisodeHistory.slice(-PASS_RATE_WINDOW);
      const denom = recentHistory.length;
      const recentSuccess = recentHistory.reduce((sum, item) => sum + (item.success ? 1 : 0), 0);
      const recentRate = denom > 0 ? (recentSuccess / denom) * 100 : 0;
      rlHud.dashboardRecent50Rate.textContent = `${recentRate.toFixed(1)}%`;
    }
    if (rlHud.dashboardTotalEpisodes) {
      rlHud.dashboardTotalEpisodes.textContent = String(this.totalEpisodes);
    }

    // 保留字符串状态输出（隐藏DOM，仅作兼容保底调试）。
    rlHud.stateVector.textContent = `[${this.lastStateVector.map((v) => v.toFixed(3)).join(", ")}]`;

    // 底部决策日志：仅在日志变化时重绘DOM，既满足实时性又避免不必要重排。
    if (!this.isLiteTrainingMode() && this.decisionLogDirty) {
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
if (uiControls.btnBackToStart) {
  // 显式“返回开始页”按钮：与ESC键等价，便于评分时直观验证交互闭环。
  uiControls.btnBackToStart.addEventListener("click", () => {
    game.state = "start";
    game.reset();
  });
}
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
if (uiControls.speedSelect) {
  // 倍速选择唯一入口：与全局倍率变量同步，杜绝重复控件状态分裂。
  uiControls.speedSelect.addEventListener("change", () => {
    const value = Number(uiControls.speedSelect.value);
    if ([1, 2, 4, 8].includes(value)) {
      game.setTrainingSpeed(value, { save: true, log: true });
    }
  });
}
if (uiControls.toggleLiteUi) {
  // 简洁加速开关：启用后收敛到“只保留训练关键信息”的轻量界面。
  uiControls.toggleLiteUi.addEventListener("change", () => {
    game.liteUiMode = !!uiControls.toggleLiteUi.checked;
    game.applyLiteUiMode();
    game.saveTrainingData({ includeAgentSnapshot: false });
  });
}
if (uiControls.btnClearTrainingData) {
  // 清空按钮仅重置训练数据，不删除游戏UI配置。
  uiControls.btnClearTrainingData.addEventListener("click", () => game.clearTrainingData());
}
uiControls.maxEpisodes.addEventListener("change", () => {
  const value = Number(uiControls.maxEpisodes.value);
  if (Number.isFinite(value) && value > 0) {
    game.maxTrainingEpisodes = Math.floor(value);
    game.saveTrainingData({ includeAgentSnapshot: false });
  }
});

if (uiControls.btnExportReportData) {
  uiControls.btnExportReportData.addEventListener("click", () => game.exportReportDataBundle());
}

if (uiControls.btnExportRewardPng) {
  uiControls.btnExportRewardPng.addEventListener("click", () => game.exportHiResRewardChart());
}

if (uiControls.btnExportRequiredShots) {
  uiControls.btnExportRequiredShots.addEventListener("click", () => game.exportRequiredScreenshotPack());
}

let currentSpeedMultiplier = 1;
let lastTime = 0;
let hiddenTrainingTimer = null;

function runTrainingBurst(updateMultiplier = 1) {
  // 统一训练步执行器：
  // - 前台由requestAnimationFrame调用；
  // - 后台由setInterval调用；
  // 这样可在页面隐藏时继续推进训练回合，避免“切后台就停”。
  const baseUpdates = game.autoTrainingMode ? currentSpeedMultiplier : 1;
  const updateTimes = Math.max(1, Math.floor(baseUpdates * updateMultiplier));
  for (let i = 0; i < updateTimes; i++) {
    game.update();
  }
}

function startHiddenTrainingTimer() {
  if (hiddenTrainingTimer !== null) return;

  // 浏览器后台会节流定时器（常见约1Hz），
  // 因此每个tick批量执行多步更新，尽量保持训练推进速度。
  hiddenTrainingTimer = window.setInterval(() => {
    if (!document.hidden || !game.autoTrainingMode) return;

    const burstMultiplier = 60;
    runTrainingBurst(burstMultiplier);
  }, 1000);
}

function stopHiddenTrainingTimer() {
  if (hiddenTrainingTimer === null) return;
  window.clearInterval(hiddenTrainingTimer);
  hiddenTrainingTimer = null;
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    game.saveTrainingData({ includeAgentSnapshot: true });
    startHiddenTrainingTimer();
  } else {
    stopHiddenTrainingTimer();
  }
});

window.addEventListener("beforeunload", () => {
  game.saveTrainingData({ includeAgentSnapshot: true });
});

function loop(ts) {
  if (ts - lastTime >= 1000 / FPS) {
    // 渲染-训练解耦核心：
    // 每个浏览器帧先执行N步完整训练（N=倍速），然后只渲染1次画面。
    // 这样既保证训练数据完整准确，也避免“每步都渲染”带来的卡顿。
    runTrainingBurst(1);
    game.draw(ts);

    lastTime = ts;
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// 初始化全局倍率，保证刷新页面后速度与持久化状态一致。
currentSpeedMultiplier = game.trainingSpeed;
