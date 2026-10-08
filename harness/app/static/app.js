"use strict";

const state = {
  snapshot: null,
  costmap: null,
  cursor: 0,
  events: [],
  filter: "all",
  cameraStream: null,
  observerStream: null,
  toastTimer: null,
  resetWasActive: false,
  sceneCatalogKey: "",
  sceneSelectionDirty: false,
  manualKeys: new Set(),
  manualCommandInFlight: false,
  manualCommandPending: false,
  manualTimer: null,
  lastManualErrorAt: 0,
  commandPending: false,
  commandError: "",
  statePollPending: false,
  eventsPollPending: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function setStatus(element, label, mode) {
  element.classList.remove("online", "offline", "busy");
  element.classList.add(mode);
  element.querySelector("strong").textContent = label;
}

function formatNumber(value, digits = 3) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "--";
}

function localTime(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? "--:--:--"
    : date.toLocaleTimeString("zh-CN", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("visible");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove("visible"), 3200);
}


async function requestJson(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(url, { cache: "no-store", ...options, signal: controller.signal });
    let payload = {};
    try {
      payload = await response.json();
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw new Error(`控制台返回了无效响应（HTTP ${response.status}）`);
    }
    if (!response.ok) {
      throw new Error(payload.error || payload.message || `HTTP ${response.status}`);
    }
    return payload;
  } catch (error) {
    if (error.name === "AbortError") throw new Error(options.method === "POST"
      ? "请求超时，受理结果未知。请等待任务状态更新，不要重复提交。"
      : "控制台状态请求超时，请检查服务是否正常运行。");
    throw error;
  } finally { clearTimeout(timer); }
}

function commandFeedback(message, error = false) {
  const element = $("#command-feedback");
  element.textContent = message;
  element.classList.toggle("error", error);
}

function updateCommandFeedback(agent, runtime) {
  if (state.commandPending) return;
  if (agent.busy) commandFeedback(agent.execution_mode === "composed"
    ? "Harness 正在处理组合任务；生成目标后会在此显示确认按钮。"
    : "Harness 正在处理任务，执行过程见下方记录。");
  else if (agent.proposed_goal) commandFeedback("目标已生成，请核对上方目标并点击“确认目标并执行”。");
  else if (agent.last_error) commandFeedback(`任务失败：${agent.last_error}`, true);
  else if (state.commandError) commandFeedback(state.commandError, true);
  else if (runtime && runtime.state !== "READY") commandFeedback(`机器人状态：${runtime.state}，暂不能提交任务。`, true);
  else if (agent.last_task_result?.task_status) commandFeedback(`最近任务：${agent.last_task_result.task_status}。${agent.last_response || ""}`);
  else commandFeedback("已连接 Harness，可以提交任务。组合任务需先确认目标。");
}

function updateSnapshot(snapshot) {
  state.snapshot = snapshot;
  const simulation = snapshot.simulation;
  const backend = simulation.backend || "mujoco";
  const isaacBackend = backend === "isaac-g1";
  const go2Backend = backend === "mujoco-go2";
  const capabilities = simulation.capabilities || {
    navigation: true,
    costmap: true,
    lidar_safety: true,
    third_person: Boolean(simulation.third_person_enabled),
    scene_seed: true,
    scene_person: true,
  };
  const agent = snapshot.agent;
  const world = snapshot.world;
  const navigation = snapshot.navigation_bridge || {};
  const recovery = snapshot.safety_recovery || {
    enabled: false,
    state: "idle",
    active: false,
    safety_hold: false,
    reconsideration_required: false,
  };
  const taskRecovery = snapshot.task_recovery || {
    state: "idle",
    pending: false,
    resume_attempts: 0,
    max_resumes: 0,
  };
  const longTask = snapshot.long_task || {
    supported: false,
    active: false,
    job: null,
  };
  const mapping = snapshot.costmap || { available: false };
  const manual = snapshot.manual_control || {
    supported: false,
    enabled: false,
    active: false,
    target: "robot",
    person_available: false,
    person_mode: "auto",
  };
  const reset = snapshot.reset || { active: false, phase: "idle", last_error: "" };
  const resetActive = Boolean(reset.active);
  const resetPhaseLabels = {
    stopping: "复位·停车",
    clearing: "复位·清理",
    starting: "复位·启动",
  };

  $("#backend-stream").textContent = isaacBackend
    ? "ISAAC SIM 5.1 / ATOMIC RGB-D"
    : go2Backend
      ? "MUJOCO GO2 / HIKROBOT RGB + MID-360"
      : "MUJOCO / LOCAL STREAM";
  $(".vision-grid").classList.toggle("isaac", isaacBackend && !capabilities.third_person);
  $(".vision-grid").classList.toggle("go2", go2Backend);
  $("#observer-view").hidden = !capabilities.third_person;
  $("#primary-camera-title").textContent = go2Backend ? "Go2-01 第一视角" : "机器人第一视角";
  const observerProfile = simulation.third_person_profile || {
    width: isaacBackend ? 320 : 640,
    height: isaacBackend ? 180 : 360,
    hz: isaacBackend ? 1 : 10,
  };
  $("#observer-profile").textContent = `${observerProfile.width} × ${observerProfile.height} · ${observerProfile.hz} FPS`;
  $("#start-sim").textContent = isaacBackend
    ? "启动 Isaac"
    : go2Backend
      ? "启动 Go2 MuJoCo"
      : "启动仿真";
  const costmapDisplay = Boolean(capabilities.costmap || capabilities.costmap_display);
  $("#map-panel-title").textContent = go2Backend
    ? "Go2 轨迹与 MID-360 安全态势"
    : capabilities.costmap
    ? "实时建图与空间态势"
    : costmapDisplay
      ? capabilities.recovery_costmap
        ? "RTX 雷达安全回撤 Costmap"
        : "RTX 雷达在线 Costmap"
      : "里程计轨迹与能力边界";
  $("#world-data-boundary").textContent = go2Backend
    ? "单 Go2 使用 MID-360 在线传感器地图；不存在第二机器人或共享覆盖率验收。"
    : capabilities.costmap
    ? "栅格与距离只来自本轮 DimOS 在线传感器；隐藏场景几何不会进入 UI 或 Agent 观测。"
    : costmapDisplay
      ? capabilities.recovery_costmap
        ? "栅格只由已验证 RTX 雷达在线投影，供操作者查看，并仅用于沿里程计面包屑安全回撤；不开放 Agent 导航或任意路径规划。"
        : "栅格只由已验证 RTX 雷达在线投影，供操作者查看；当前不作为 Agent 导航、路径规划或安全回溯依据。"
      : "Isaac Agent 仅使用新鲜第一视角 RGB 与里程计；第三人称只供操作者查看。";

  if (resetActive) {
    setStatus($("#sim-status"), resetPhaseLabels[reset.phase] || "复位中", "busy");
  } else if (simulation.process_alive || (simulation.mcp && simulation.command_center)) {
    setStatus($("#sim-status"), simulation.starting ? "启动中" : "运行中", simulation.starting ? "busy" : "online");
  } else if (simulation.starting) {
    setStatus($("#sim-status"), "启动中", "busy");
  } else {
    setStatus($("#sim-status"), "未运行", "offline");
  }
  $("#mcp-status span").textContent = go2Backend ? "TASK BRIDGE" : "MCP";
  setStatus($("#mcp-status"), simulation.mcp ? "已连接" : "未连接", simulation.mcp ? "online" : "offline");
  if (!capabilities.navigation) {
    setStatus($("#nav-status"), "未开放", "offline");
  } else setStatus(
    $("#nav-status"),
    recovery.active
      ? "安全回撤"
      : recovery.safety_hold
        ? "安全锁定"
        : navigation.active
          ? "规划控制"
          : navigation.running
            ? "安全桥在线"
            : "桥未连接",
    recovery.active || navigation.active
      ? "busy"
      : recovery.safety_hold
        ? "offline"
        : navigation.running
          ? "online"
          : "offline",
  );
  setStatus(
    $("#agent-status"),
    resetActive ? "复位中" : agent.busy ? "执行中" : agent.available ? "待机" : "不可用",
    resetActive || agent.busy ? "busy" : agent.available ? "online" : "offline",
  );
  const agentLabel = "Harness";
  const readyLabel = $("#agent-link-label");
  if (readyLabel) readyLabel.textContent = `${agentLabel.toUpperCase()} · ${agent.model || agent.model_provider || "模型未配置"} · ${agent.available ? "READY" : "UNAVAILABLE"}`;
  $("#agent-submit-label").textContent = `交给 ${agentLabel}`;
  $("#agent-orbit").classList.toggle("busy", agent.busy);
  const recoveryHeld = recovery.safety_hold && !recovery.reconsideration_required;
  $("#send-command").disabled = state.commandPending || agent.busy || longTask.active || resetActive || recovery.active || recoveryHeld || (snapshot.robot_runtime && snapshot.robot_runtime.state !== "READY");
  updateCommandFeedback(agent, snapshot.robot_runtime);
  $("#start-sim").hidden = resetActive || simulation.process_alive || (simulation.mcp && simulation.command_center);
  $("#start-sim").disabled = resetActive || Boolean(simulation.starting);
  $("#save-map").hidden = !capabilities.costmap;
  $("#save-map").disabled = resetActive || !capabilities.costmap;
  const manualPanel = $("#manual-control");
  const manualToggle = $("#manual-mode-toggle");
  const manualSupported = Boolean(manual.supported && (isaacBackend || go2Backend));
  manualPanel.hidden = !manualSupported;
  manualPanel.classList.toggle("active", Boolean(manual.enabled));
  const manualTarget = manual.target === "person" ? "person" : "robot";
  const manualTargetSelect = $("#manual-control-target");
  manualTargetSelect.value = manualTarget;
  manualTargetSelect.querySelector('option[value="person"]').disabled = !manual.person_available;
  manualTargetSelect.querySelector('option[value="robot"]').disabled = manual.robot_available === false || agent.busy;
  manualTargetSelect.disabled = resetActive;
  $("#person-control-actions").hidden = manualTarget !== "person";
  $("#person-pause").disabled = !manual.person_available || resetActive;
  $("#person-resume").disabled = !manual.person_available || resetActive;
  manualToggle.classList.toggle("active", Boolean(manual.enabled));
  manualToggle.textContent = manual.enabled
    ? "退出键盘控制"
    : `控制${manualTarget === "person" ? "人物" : "机器人"}`;
  manualToggle.disabled = !manualSupported
    || (
      !manual.enabled
      && (
        resetActive
        || !simulation.bridge_ready
        || (
          manualTarget === "robot"
          && (agent.busy || recovery.active || recovery.safety_hold)
        )
        || (manualTarget === "person" && !manual.person_available)
      )
    );
  $("#manual-control-state").textContent = manual.enabled
    ? manual.active
      ? `${manualTarget === "person" ? "人物" : "机器人"}键盘控制中`
      : `${manualTarget === "person" ? "人物" : "机器人"}键盘控制已就绪`
    : manualTarget === "person"
      ? `人物${manual.person_mode === "auto" ? "自动路线" : manual.person_mode === "paused" ? "已暂停" : "手动待机"}`
      : "机器人键盘控制关闭";
  $("#manual-control-help").textContent = manualTarget === "person"
    ? go2Backend
      ? "人物命令 0.35 秒失效；松键、窗口失焦或网络中断都会停车。"
      : "人物命令 0.35 秒失效；移动受公寓边界、墙体和家具碰撞约束。"
    : "按住移动，松键停车；窗口失焦或 0.35 秒无刷新自动归零。";
  $("#manual-control-keys-help").textContent = manualTarget === "person"
    ? "W/S 前后行走 · A/D 原地转向 · Esc 退出并暂停"
    : "W/S 前后 · A/D 踏步转向 · Esc 退出";
  if (!manual.enabled && state.manualKeys.size) clearManualKeys(false);
  $$('[data-requires-navigation]').forEach((element) => {
    element.hidden = !capabilities.navigation;
  });
  $$('[data-requires-go2]').forEach((element) => {
    element.hidden = !go2Backend;
  });
  $$('.suggestions [data-command]:not([data-requires-go2])').forEach((element) => {
    element.hidden = go2Backend;
  });
  $$('[data-command]').forEach((button) => {
    button.disabled = longTask.active || resetActive || recovery.active || recoveryHeld;
  });
  const resetButton = $("#reset-experiment");
  resetButton.disabled = resetActive;
  resetButton.classList.toggle("busy", resetActive);
  resetButton.setAttribute("aria-busy", String(resetActive));
  $("#reset-label").textContent = resetActive
    ? (resetPhaseLabels[reset.phase] || "复位中")
    : isaacBackend ? "重启 Isaac" : go2Backend ? "重启 Go2" : "复位实验";
  if (resetActive) {
    state.resetWasActive = true;
  } else if (state.resetWasActive) {
    state.resetWasActive = false;
    showToast(
      reset.phase === "ready"
        ? (isaacBackend ? "Isaac 已重启，新鲜 RGB 与里程计已就绪" : "实验复位完成，新地图已就绪")
        : `复位失败：${reset.last_error || "未知错误"}`,
      reset.phase !== "ready",
    );
  }

  const pose = world.pose;
  if (pose) {
    const yawDegrees = (pose.yaw * 180) / Math.PI;
    $("#pose-hud").textContent = `X ${formatNumber(pose.x, 2)} · Y ${formatNumber(pose.y, 2)} · YAW ${formatNumber(yawDegrees, 1)}°`;
    $("#head-pose-hud").textContent = `X ${formatNumber(pose.x, 2)} · Y ${formatNumber(pose.y, 2)} · YAW ${formatNumber(yawDegrees, 1)}°`;
  } else {
    $("#pose-hud").textContent = "X -- · Y -- · YAW --";
    $("#head-pose-hud").textContent = "X -- · Y -- · YAW --";
  }

  const command = world.command || [0, 0, 0, 0, 0, 0];
  $("#cmd-x").textContent = formatNumber(command[0]);
  $("#cmd-y").textContent = formatNumber(command[1]);
  $("#cmd-yaw").textContent = formatNumber(command[5]);
  $("#measured-speed").textContent = formatNumber(world.motion?.planar_speed);
  $("#measured-speed").closest(".telemetry-item").classList.toggle("alert", Boolean(world.motion?.unexpected_motion));
  const personDistance = world.metrics.person_distance;
  $("#person-distance").textContent = formatNumber(personDistance, 2);
  $("#person-telemetry").hidden = !Number.isFinite(personDistance);
  $$('[data-requires-person]').forEach((element) => {
    element.hidden = !Number.isFinite(personDistance);
  });
  $("#obstacle-distance").textContent = formatNumber(world.metrics.nearest_obstacle_distance, 2);

  const risk = world.metrics.risk || "unknown";
  const badge = $("#risk-badge");
  badge.className = `risk-badge ${risk}`;
  badge.textContent = risk.toUpperCase();
  const headBadge = $("#head-risk-badge");
  headBadge.className = `risk-badge ${risk}`;
  headBadge.textContent = capabilities.lidar_safety ? risk.toUpperCase() : "LIDAR N/A";

  updateRecoveryStatus(recovery, taskRecovery, Boolean(agent.busy));
  updateLongTask(longTask);

  $("#frame-clock").textContent = localTime(world.sampled_at);
  $("#server-clock").textContent = `SERVER ${localTime(snapshot.server_time)}`;

  refreshHeadCamera();
  refreshObserverCamera();
  $("#instruction").placeholder = go2Backend
    ? "例如：巡检仓库南区、寻找蓝色球、跟随这个人 8 秒"
    : "例如：先观察四周，然后在空旷处停下。";

  updateSceneControl(simulation.scene, resetActive, Boolean(simulation.starting), capabilities);

  updateExecutionModes(agent);
  updateAgentResponse(agent);
  renderAgentProcess();
  updateMapStatus(mapping, capabilities);
  updateObservations(world.observations || [], mapping);
  drawWorldMap(world, state.costmap);
}

function updateLongTask(longTask) {
  const card = $("#long-task-card");
  const job = longTask?.job;
  card.hidden = !job;
  if (!job) return;
  const stageLabels = {
    starting: "启动工具",
    navigate_to_pickup: "导航到取物点",
    locate_entity: "视觉定位物体",
    approach_entity: "接近物体",
    grasp_entity: "接触并抓取",
    prepare_carry_entity: "切换携带姿态",
    carry_entity: "携带到目的地",
    waiting_for_idle: "确认 Agent / 导航 / 手臂空闲",
    stopping: "协作取消与停车",
    completed: "完成",
    failed: "失败",
    cancelled: "已取消",
  };
  const result = job.last_structured_result;
  $("#long-task-title").textContent = `${job.tool || "long task"} · ${job.state || "unknown"}`;
  $("#long-task-id").textContent = String(job.job_id || "--").slice(0, 13);
  $("#long-task-stage").textContent = stageLabels[job.stage] || job.stage || "--";
  $("#long-task-result").textContent = result?.task_status
    || job.idle_error
    || job.execution_state
    || "--";
  const cancelRow = $("#long-task-cancel-row");
  cancelRow.hidden = !job.cancel_reason;
  $("#long-task-cancel-reason").textContent = job.cancel_reason || "--";
  const cancelButton = $("#cancel-long-task");
  cancelButton.disabled = !longTask.active || job.state === "cancelling";
  cancelButton.textContent = job.state === "cancelling" ? "正在取消" : "取消任务";
}

function updateRecoveryStatus(recovery, taskRecovery, agentBusy) {
  const banner = $("#recovery-banner");
  const stateName = recovery.state || "idle";
  const continuationState = taskRecovery.state || "idle";
  const continuationVisible = taskRecovery.pending
    || Boolean(taskRecovery.reason)
    || continuationState === "resume_limit_reached"
    || (continuationState === "replanning_original_task" && agentBusy);
  const visible = recovery.enabled && (stateName !== "idle" || continuationVisible);
  banner.hidden = !visible;
  banner.className = `recovery-banner ${stateName}`;
  if (!visible) return;

  const labels = {
    stopping: "立即停车并确认静止",
    retreating: "沿已验证轨迹低速回撤",
    settling: "安全位置停车复核",
    recovered_waiting_replan: "已到达非 critical 且视野良好的位置",
    held: "无法证明回撤安全，保持停车",
  };
  $("#recovery-title").textContent = labels[stateName] || "安全恢复锁定";
  const distance = Number.isFinite(recovery.retreat_distance_m)
    ? ` · 已回撤 ${Number(recovery.retreat_distance_m).toFixed(2)} m`
    : "";
  const recoveryReasons = {
    no_active_task: "安全中断时没有活动任务",
    task_already_finished: "登记恢复时原任务已经结束",
    task_replaced: "原任务已被新任务替换",
    instruction_unavailable: "原任务指令缺失",
    task_identity_unavailable: "原任务标识或指令缺失",
    goal_not_confirmed: "任务目标尚未确认",
    fixed_task_not_resumable: "该预定义任务不支持暂停续跑",
    remaining_budget_exhausted: "原任务剩余执行预算已耗尽",
    task_not_cancelled: "原任务未以可恢复的取消状态结束",
    execution_error: "原任务执行异常，未保存可恢复进度",
    progress_save_failed: "原任务进度保存失败",
    progress_not_saved: "原任务已结束，但未生成恢复进度",
    recovery_event_write_failed: "恢复记录写入失败",
    cancellation_callback_failed: "任务取消回调失败",
    recovery_registration_failed: "安全恢复意图登记失败",
    recovery_state_unavailable: "无法读取或交接原任务恢复状态",
    confirmed_goal_unavailable: "缺少与原任务匹配的已确认目标",
    independent_long_task: "独立长运动任务已取消，不支持自动续跑",
    handoff_rejected: "安全控制权交接被拒绝",
    agent_cancel_timeout: "等待原任务结束超时",
    resume_limit_reached: "自动恢复次数已达上限",
  };
  if (continuationState === "replanning_original_task" && agentBusy) {
    $("#recovery-title").textContent = "已基于新观测继续原任务";
    $("#recovery-detail").textContent = `Agent 正在重新规划原任务 · 自动续跑 ${taskRecovery.resume_attempts}/${taskRecovery.max_resumes} 次`;
  } else if (taskRecovery.reason) {
    const reason = recoveryReasons[taskRecovery.reason] || taskRecovery.reason;
    const detail = taskRecovery.detail ? `（${taskRecovery.detail}）` : "";
    const motion = stateName === "recovered_waiting_replan" || stateName === "held"
      ? "机器人保持停车" : "安全停车与回撤流程继续";
    $("#recovery-detail").textContent = `原任务无法自动续跑：${reason}${detail}；${motion}${distance}`;
  } else if (stateName === "recovered_waiting_replan") {
    if (taskRecovery.pending) {
      $("#recovery-detail").textContent = `已登记原任务恢复意图；等待停车与进度保存完成后重新规划${distance}`;
    } else {
      $("#recovery-detail").textContent = `未登记自动续跑任务；机器人保持停车，等待新指令${distance}`;
    }
  } else if (stateName === "held") {
    $("#recovery-detail").textContent = `原因：${recovery.reason || "安全证据不足"}；新任务已锁定，请急停检查或复位${distance}`;
  } else {
    $("#recovery-detail").textContent = `恢复速度上限 ${formatNumber(recovery.recovery_speed_limit_mps, 2)} m/s${distance}`;
  }
}

function selectedSceneDescriptor(scene) {
  const sceneId = $("#scene-select").value;
  return (scene?.catalog || []).find((item) => item.scene_id === sceneId) || null;
}

function renderPendingScene(scene) {
  const descriptor = selectedSceneDescriptor(scene);
  const seedInput = $("#scene-seed");
  const personInput = $("#scene-person");
  if (!descriptor) {
    $("#scene-current").textContent = "未知场景";
    $("#scene-description").textContent = "当前场景不在受支持目录中。";
    return;
  }
  seedInput.disabled = !descriptor.seedable || Boolean(state.snapshot?.reset?.active);
  personInput.disabled = !descriptor.supports_person || Boolean(state.snapshot?.reset?.active);
  if (!descriptor.supports_person) personInput.checked = false;
  const complexity = descriptor.complexity === "complex" ? "复杂" : descriptor.complexity === "baseline" ? "基线" : "标准";
  $("#scene-current").textContent = `${descriptor.label} · ${complexity}`;
  $("#scene-description").textContent = descriptor.description;
}

function updateSceneControl(scene, resetActive, simulationStarting, capabilities) {
  const control = $("#scene-control");
  const switchable = Boolean(scene?.switchable);
  control.hidden = !switchable;
  if (!switchable) return;

  const catalog = scene.catalog || [];
  const catalogKey = JSON.stringify(catalog.map((item) => [item.scene_id, item.label]));
  const selector = $("#scene-select");
  if (catalogKey !== state.sceneCatalogKey) {
    selector.replaceChildren(...catalog.map((descriptor) => {
      const option = document.createElement("option");
      option.value = descriptor.scene_id;
      option.textContent = `${descriptor.complexity === "complex" ? "复杂 · " : ""}${descriptor.label}`;
      return option;
    }));
    state.sceneCatalogKey = catalogKey;
    state.sceneSelectionDirty = false;
  }
  if (!state.sceneSelectionDirty) {
    selector.value = scene.selected_id;
    $("#scene-seed").value = String(scene.seed ?? 0);
    $("#scene-person").checked = Boolean(scene.include_person);
  }
  selector.disabled = resetActive || simulationStarting;
  $("#switch-scene").disabled = resetActive || simulationStarting;
  $("#scene-seed-field").hidden = !capabilities.scene_seed;
  $("#scene-person-field").hidden = !capabilities.scene_person;
  $("#switch-scene").textContent = capabilities.costmap
    ? "切换并重建地图"
    : "切换并重启 Isaac";
  renderPendingScene(scene);
}

function updateAgentResponse(agent) {
  const container = $("#agent-response");
  if (agent.last_response) {
    const message = document.createElement("div");
    message.className = "response-message";
    message.textContent = agent.last_response;
    if (agent.last_error) message.textContent += `\n原因：${agent.last_error}`;
    const task = agent.last_task_result || {};
    if (task.task_status) {
      const outcome = document.createElement("div");
      outcome.className = `task-outcome ${task.completed ? "complete" : "incomplete"}`;
      const facts = [
        `STATUS ${task.task_status}`,
        `完成 ${task.completed ? "是" : "否"}`,
      ];
      if (typeof task.planner_goal_reached === "boolean") {
        facts.push(`规划到达 ${task.planner_goal_reached ? "是" : "否"}`);
      }
      if (Number.isFinite(task.target_distance_m)) {
        facts.push(`目标距离 ${formatNumber(task.target_distance_m, 2)} m`);
      }
      if (Number.isFinite(task.physical_stop_latency_ms)) {
        facts.push(`物理停车 ${formatNumber(task.physical_stop_latency_ms, 0)} ms`);
      }
      if (Number.isFinite(task.stop_command_publish_latency_ms)) {
        facts.push(`零速发布 ${formatNumber(task.stop_command_publish_latency_ms, 0)} ms`);
      }
      if (Number.isFinite(task.planning_steps) && Number.isFinite(task.tool_calls)) {
        facts.push(`规划/工具 ${task.planning_steps}/${task.tool_calls}`);
      }
      if (Number.isFinite(task.elapsed_s)) {
        facts.push(`耗时 ${formatNumber(task.elapsed_s, 1)} s`);
      }
      outcome.textContent = facts.join(" · ");
      container.replaceChildren(message, outcome);
    } else {
      container.replaceChildren(message);
    }
  } else if (agent.busy) {
    const pending = document.createElement("div");
    pending.className = "response-empty";
    const title = document.createElement("span");
    const agentLabel = "Harness";
    title.textContent = `${agentLabel.toUpperCase()} / WORKING`;
    const text = document.createElement("p");
    text.textContent = "代理正在观测并调用工具。详细步骤会实时出现在工具链记录中。";
    pending.append(title, text);
    container.replaceChildren(pending);
  } else if (agent.last_error) {
    const failed = document.createElement("div");
    failed.className = "response-empty";
    const title = document.createElement("span");
    title.textContent = "AGENT / ERROR";
    const text = document.createElement("p");
    text.textContent = agent.last_error;
    failed.append(title, text);
    container.replaceChildren(failed);
  } else {
    const ready = document.createElement("div");
    ready.className = "response-empty";
    const title = document.createElement("span");
    const agentLabel = "Harness";
    title.textContent = `${agentLabel.toUpperCase()} LINK READY`;
    const text = document.createElement("p");
    text.textContent = "输入自然语言指令。代理会观测、调用 DimOS 工具并记录执行过程。";
    ready.append(title, text);
    container.replaceChildren(ready);
  }
}

function processOutput(event) {
  const output = event.data?.output;
  if (typeof output !== "string") return null;
  try {
    const parsed = JSON.parse(output);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch (_error) {
    return null;
  }
}

function compactProcessArguments(argumentsValue) {
  if (!argumentsValue || typeof argumentsValue !== "object") return "无参数";
  const entries = Object.entries(argumentsValue);
  if (!entries.length) return "无参数";
  const text = entries.map(([key, value]) => {
    const rendered = typeof value === "string" ? value : JSON.stringify(value);
    return `${key}=${rendered}`;
  }).join(" · ");
  return text.length > 220 ? `${text.slice(0, 217)}…` : text;
}

function processDuration(data) {
  return Number.isFinite(data?.duration_ms) ? `${Math.max(0, data.duration_ms)} ms` : "";
}

function processResultSummary(event) {
  const data = event.data || {};
  const result = processOutput(event) || {};
  if (result.error) return String(result.error);
  if (result.motion_evidence) {
    const evidence = result.motion_evidence;
    const facts = [];
    if (Number.isFinite(evidence.planar_displacement_m)) {
      facts.push(`实测位移 ${formatNumber(evidence.planar_displacement_m, 3)} m`);
    }
    if (Number.isFinite(evidence.immediate_yaw_change_rad) && Number.isFinite(evidence.yaw_change_rad)) {
      facts.push(
        `偏航 瞬时 ${formatNumber(evidence.immediate_yaw_change_rad * 180 / Math.PI, 1)}°`
        + ` / 稳定 ${formatNumber(evidence.yaw_change_rad * 180 / Math.PI, 1)}°`,
      );
    } else if (Number.isFinite(evidence.yaw_change_rad)) {
      facts.push(`实测偏航 ${formatNumber(evidence.yaw_change_rad * 180 / Math.PI, 1)}°`);
    }
    facts.push(evidence.motion_observed ? "检测到动作" : "未检测到有效动作");
    return facts.join(" · ");
  }
  if (typeof result.found === "boolean") {
    const facts = [result.found ? "已发现目标" : "未发现目标"];
    if (result.confidence !== undefined) facts.push(`置信度 ${result.confidence}`);
    if (result.evidence) facts.push(String(result.evidence));
    return facts.join(" · ");
  }
  if (result.observation?.pose) {
    const pose = result.observation.pose;
    return `位姿 x=${formatNumber(pose.x, 2)} · y=${formatNumber(pose.y, 2)} · yaw=${formatNumber(pose.yaw, 3)}`;
  }
  const status = result.task_status || data.task_status;
  const distanceFacts = [];
  if (status) distanceFacts.push(String(status));
  if (Number.isFinite(result.actual_distance_m)) {
    distanceFacts.push(`实走 ${formatNumber(result.actual_distance_m, 2)} m`);
  }
  if (Number.isFinite(result.distance_error_m)) {
    distanceFacts.push(`误差 ${formatNumber(result.distance_error_m, 2)} m`);
  }
  if (distanceFacts.length) return distanceFacts.join(" · ");
  if (data.ok === false || result.ok === false) return "工具返回失败";
  if (!data.output && event.message) {
    return event.message.length > 240 ? `${event.message.slice(0, 237)}…` : event.message;
  }
  return data.completed || result.completed ? "工具已完成并通过验收" : "工具调用已返回";
}

function processEventView(event) {
  const data = event.data || {};
  const duration = processDuration(data);
  const callId = typeof data.call_id === "string" ? data.call_id.slice(-10) : "";
  if ((event.source === "agent" && event.kind === "instruction") || event.title === "Language instruction") {
    return { type: "instruction", label: "指令", title: "用户任务", summary: event.message, meta: [] };
  }
  if (event.source === "agent" && event.kind === "response") {
    if (data.task_status === "goal_proposed") return { type: "lifecycle", label: "待确认",
      title: "组合目标已生成", summary: "请核对目标并确认执行。", meta: [] };
    const failed = Boolean(data.error) || data.completed === false;
    return { type: failed ? "failed" : "complete", label: failed ? "未完成" : "结果",
      title: event.title, summary: data.error || event.message || data.task_status || "任务已返回", meta: [] };
  }
  if (event.title === "Harness model step") {
    const round = data.round || event.message.match(/round=(\d+)/)?.[1] || "?";
    const finish = data.finish_reason || event.message.match(/finish=([^ ·]+)/)?.[1] || "unknown";
    const usage = data.usage || {};
    const summary = finish === "tool_calls" ? "模型选择调用工具" : "模型生成最终答复";
    const meta = [duration];
    if (Number.isFinite(usage.total_tokens)) meta.push(`${usage.total_tokens} tokens`);
    return { type: "model", label: "规划", title: `模型步骤 ${round}`, summary, meta };
  }
  if (
    event.source === "tool"
    && (event.kind === "call" || (event.kind === "mcp" && data.arguments))
  ) {
    return {
      type: "call",
      label: data.automatic ? "预检" : "调用",
      title: data.tool || event.title,
      summary: compactProcessArguments(data.arguments),
      meta: [callId && `#${callId}`],
    };
  }
  if (event.source === "tool" && ["result", "mcp"].includes(event.kind)) {
    const ok = data.ok !== false && data.tool_ok !== false && !data.error && event.level !== "danger" && event.level !== "warning";
    return {
      type: ok ? "result" : "failed",
      label: ok ? "结果" : "失败",
      title: data.tool || data.name || event.title,
      summary: processResultSummary(event),
      meta: [duration, callId && `#${callId}`],
    };
  }
  if (event.source === "tool" && event.kind === "progress") {
    return { type: "lifecycle", label: "进度", title: data.tool || event.title, summary: event.message, meta: [duration] };
  }
  if (event.title === "Agent response" || event.title === "AgentOS response") {
    return { type: "complete", label: "答复", title: "最终答复已生成", summary: "完整内容见下方 Agent 回复。", meta: [] };
  }
  if (/turn started|reasoning started/i.test(event.title)) {
    return { type: "lifecycle", label: "开始", title: "Agent 开始处理", summary: event.message || "建立本轮执行上下文", meta: [] };
  }
  if (/turn finished|turn completed/i.test(event.title)) {
    return { type: "complete", label: "结束", title: "本轮执行结束", summary: event.message || "Agent 已转为空闲", meta: [] };
  }
  if (event.source === "agent" && ["error", "warning", "budget"].includes(event.kind)) {
    return { type: "failed", label: "异常", title: event.title, summary: event.message, meta: [duration] };
  }
  return null;
}

function currentAgentProcessEvents() {
  let start = -1;
  for (let index = state.events.length - 1; index >= 0; index -= 1) {
    if ((state.events[index].source === "agent" && state.events[index].kind === "instruction") || state.events[index].title === "Language instruction") {
      start = index;
      break;
    }
  }
  if (start < 0) return [];
  return state.events.slice(start).map((event) => ({ event, view: processEventView(event) }))
    .filter((item) => item.view)
    .slice(-24);
}

function renderAgentProcess() {
  const container = $("#agent-process");
  const indicator = $("#agent-process-state");
  if (!container || !indicator) return;
  const items = currentAgentProcessEvents();
  const busy = Boolean(state.snapshot?.agent?.busy);
  const last = items.at(-1)?.view;
  indicator.classList.remove("idle", "busy", "failed", "complete");
  if (busy && last?.type === "call") {
    indicator.textContent = `${last.title} 执行中`;
    indicator.classList.add("busy");
  } else if (busy) {
    indicator.textContent = "模型规划中";
    indicator.classList.add("busy");
  } else if (last?.type === "failed") {
    indicator.textContent = "最近一轮有异常";
    indicator.classList.add("failed");
  } else if (items.length) {
    indicator.textContent = "最近一轮已结束";
    indicator.classList.add("complete");
  } else {
    indicator.textContent = "等待指令";
    indicator.classList.add("idle");
  }

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "agent-process-empty";
    empty.textContent = "模型步骤、工具调用、结果与耗时会实时显示在这里。";
    container.replaceChildren(empty);
    return;
  }

  const rows = items.map(({ event, view }) => {
    const row = document.createElement("div");
    row.className = `agent-process-item ${view.type}`;
    const marker = document.createElement("span");
    marker.className = "agent-process-marker";
    const body = document.createElement("div");
    body.className = "agent-process-body";
    const heading = document.createElement("div");
    heading.className = "agent-process-row-heading";
    const label = document.createElement("span");
    label.textContent = view.label;
    const title = document.createElement("strong");
    title.textContent = view.title;
    const time = document.createElement("time");
    time.textContent = localTime(event.timestamp);
    heading.append(label, title, time);
    const summary = document.createElement("p");
    summary.textContent = view.summary || "--";
    body.append(heading, summary);
    const metaValues = view.meta.filter(Boolean);
    if (metaValues.length) {
      const meta = document.createElement("div");
      meta.className = "agent-process-meta";
      metaValues.forEach((value) => {
        const badge = document.createElement("span");
        badge.textContent = value;
        meta.append(badge);
      });
      body.append(meta);
    }
    row.append(marker, body);
    return row;
  });
  container.replaceChildren(...rows);
  container.scrollTop = container.scrollHeight;
}

function updateMapStatus(mapping, capabilities = { costmap: true }) {
  const indicator = $("#map-state");
  indicator.classList.remove("online", "saved", "offline");
  const costmapDisplay = Boolean(capabilities.costmap || capabilities.costmap_display);
  if (!costmapDisplay) {
    indicator.classList.add("offline");
    indicator.textContent = "NAV / LIDAR 未开放";
    $("#map-summary").textContent = "新鲜里程计轨迹";
    $("#map-scale").textContent = "ODOM LOCAL ±4 m";
    return;
  }
  if (!mapping.available) {
    indicator.classList.add("offline");
    indicator.textContent = mapping.running ? "等待实时地图" : "地图接口离线";
    $("#map-summary").textContent = "结构化观测";
    $("#map-scale").textContent = "LOCAL ±4 m";
    return;
  }

  const go2LidarMap = mapping.source === "go2_mid360_live";
  const fastLioEstimators = Object.values(mapping.estimators || {});
  const fastLioReady = fastLioEstimators.length > 0
    && fastLioEstimators.every((estimator) => estimator?.ready && estimator?.process_alive);
  const fastLioMap = go2LidarMap && String(mapping.fusion || "").includes("fastlio2");
  const live = mapping.source === "live" || mapping.source === "isaac_lidar_live" || go2LidarMap;
  indicator.classList.add(live ? "online" : "saved");
  indicator.textContent = live
    ? go2LidarMap
      ? fastLioMap
        ? fastLioReady
          ? "MID-360 + FAST-LIO2"
          : "FAST-LIO2 初始化中"
        : "MID-360 LIVE MAP"
      : capabilities.recovery_costmap
      ? "RETRACE COSTMAP"
      : "LIVE COSTMAP"
    : "已保存快照";
  const known = mapping.cells?.known_reachable ?? mapping.cells?.known ?? 0;
  const total = mapping.cells?.reachable_total ?? mapping.cells?.total ?? 0;
  const percent = total ? (known / total) * 100 : 0;
  $("#map-summary").textContent = `${mapping.width}×${mapping.height} · 可达空间 ${formatNumber(percent, 1)}%`;
  const span = Math.max(mapping.width, mapping.height) * mapping.resolution;
  $("#map-scale").textContent = `${formatNumber(span, 1)} m ${fastLioMap ? "FAST-LIO2 SHARED" : "SHARED"} · ${formatNumber(mapping.resolution * 100, 1)} cm/cell · 拖动滚动条查看全图`;
}


function updateObservations(observations, mapping) {
  const list = $("#observation-list");
  const combined = [...observations];
  if (mapping?.available) {
    const total = mapping.cells?.reachable_total ?? mapping.cells?.total ?? 0;
    const known = mapping.cells?.known_reachable ?? mapping.cells?.known ?? 0;
    combined.unshift(
      {
        label: "mapped known area",
        value: total ? (known / total) * 100 : 0,
        unit: "%",
        severity: "normal",
      },
      {
        label: "saturated HeightCost cells",
        value: mapping.cells?.saturated_height_cost ?? mapping.cells?.occupied ?? 0,
        unit: "cells",
        severity: "normal",
      },
    );
  }
  if (!combined.length) {
    const empty = document.createElement("div");
    empty.className = "observation-empty";
    empty.textContent = "等待位姿与距离数据";
    list.replaceChildren(empty);
    return;
  }
  const cards = combined.map((observation) => {
    const card = document.createElement("div");
    card.className = `observation-card ${observation.severity || "normal"}`;
    const label = document.createElement("span");
    label.textContent = observation.label;
    const value = document.createElement("strong");
    value.textContent = formatNumber(observation.value, 2);
    const unit = document.createElement("em");
    unit.textContent = observation.unit || "";
    card.append(label, value, unit);
    return card;
  });
  list.replaceChildren(...cards);
}

function decodedCostmap(costmap) {
  if (!costmap?.available || !costmap.data) return null;
  if (costmap._decoded instanceof Uint8Array) return costmap._decoded;
  try {
    const binary = atob(costmap.data);
    const decoded = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      decoded[index] = binary.charCodeAt(index);
    }
    costmap._decoded = decoded;
    return decoded;
  } catch (error) {
    console.error("Could not decode costmap", error);
    return null;
  }
}

function drawCostmap(context, costmap, toScreen, scale) {
  const cells = decodedCostmap(costmap);
  if (!cells || cells.length !== costmap.width * costmap.height) return;
  const origin = costmap.origin || { x: 0, y: 0, yaw: 0 };
  const yaw = origin.yaw || 0;
  const cosine = Math.cos(yaw);
  const sine = Math.sin(yaw);
  const cellSize = Math.max(1, costmap.resolution * scale + 0.35);

  const drawPass = (minimum, maximum, fillStyle) => {
    context.fillStyle = fillStyle;
    for (let row = 0; row < costmap.height; row += 1) {
      for (let column = 0; column < costmap.width; column += 1) {
        const value = cells[row * costmap.width + column];
        if (value < minimum || value > maximum) continue;
        const localX = (column + 0.5) * costmap.resolution;
        const localY = (row + 0.5) * costmap.resolution;
        const worldX = origin.x + localX * cosine - localY * sine;
        const worldY = origin.y + localX * sine + localY * cosine;
        const screen = toScreen(worldX, worldY);
        context.fillRect(
          screen.x - cellSize / 2,
          screen.y - cellSize / 2,
          cellSize,
          cellSize,
        );
      }
    }
  };

  drawPass(0, 0, "rgba(93, 145, 133, 0.18)");
  drawPass(1, 49, "rgba(255, 202, 105, 0.24)");
  drawPass(50, 100, "rgba(255, 99, 95, 0.72)");

  const corners = [
    [0, 0],
    [costmap.width * costmap.resolution, 0],
    [costmap.width * costmap.resolution, costmap.height * costmap.resolution],
    [0, costmap.height * costmap.resolution],
  ].map(([x, y]) => toScreen(
    origin.x + x * cosine - y * sine,
    origin.y + x * sine + y * cosine,
  ));
  context.strokeStyle = "rgba(102, 203, 231, 0.34)";
  context.lineWidth = 1;
  context.beginPath();
  corners.forEach((point, index) => {
    if (index === 0) context.moveTo(point.x, point.y);
    else context.lineTo(point.x, point.y);
  });
  context.closePath();
  context.stroke();
}

function drawWorldMap(world, costmap = null) {
  const canvas = $("#world-map");
  const scrollViewport = $("#map-scroll");
  const detailedSharedMap = Boolean(
    costmap?.available
    && costmap?.source === "go2_mid360_live"
    && Number(costmap.width) > 0
    && Number(costmap.height) > 0
  );
  if (detailedSharedMap && scrollViewport) {
    // Keep every 10 cm grid cell legible. The viewport stays bounded while
    // native scrollbars expose the complete warehouse map in both axes.
    const pixelsPerCell = Math.max(
      6,
      scrollViewport.clientWidth / Number(costmap.width),
    );
    canvas.style.width = `${Math.ceil(Number(costmap.width) * pixelsPerCell)}px`;
    canvas.style.height = `${Math.ceil(Number(costmap.height) * pixelsPerCell)}px`;
  } else {
    canvas.style.width = "100%";
    canvas.style.height = "100%";
  }
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(rect.width * ratio);
  const height = Math.round(rect.height * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);

  const pose = world.pose || { x: 0, y: 0, yaw: 0 };
  const mapOrigin = costmap?.origin || {};
  const mapCenterX = Number(mapOrigin.x) + Number(costmap?.width || 0) * Number(costmap?.resolution || 0) / 2;
  const mapCenterY = Number(mapOrigin.y) + Number(costmap?.height || 0) * Number(costmap?.resolution || 0) / 2;
  const hasMapExtent = Boolean(
    costmap?.available
    && Number.isFinite(mapCenterX)
    && Number.isFinite(mapCenterY)
    && Number(costmap.width) > 0
    && Number(costmap.height) > 0
  );
  const centerX = hasMapExtent ? mapCenterX : pose.x;
  const centerY = hasMapExtent ? mapCenterY : pose.y;
  const mapWidthM = Number(costmap?.width || 0) * Number(costmap?.resolution || 0);
  const mapHeightM = Number(costmap?.height || 0) * Number(costmap?.resolution || 0);
  const radius = hasMapExtent ? Math.max(4, mapWidthM, mapHeightM) * 0.53 : 4;
  const scale = hasMapExtent
    ? Math.max(1, Math.min(
      (rect.width - 42) / Math.max(mapWidthM, 0.1),
      (rect.height - 42) / Math.max(mapHeightM, 0.1),
    ))
    : Math.min(rect.width, rect.height) / (radius * 2.25);
  const toScreen = (x, y) => ({
    x: rect.width / 2 + (x - centerX) * scale,
    y: rect.height / 2 - (y - centerY) * scale,
  });

  drawCostmap(context, costmap, toScreen, scale);

  context.lineWidth = 1;
  context.strokeStyle = "rgba(113, 241, 200, 0.075)";
  context.fillStyle = "rgba(113, 241, 200, 0.28)";
  context.font = "8px monospace";
  for (let offset = -4; offset <= 4; offset += 1) {
    const x = toScreen(centerX + offset, centerY).x;
    const y = toScreen(centerX, centerY + offset).y;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, rect.height);
    context.stroke();
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(rect.width, y);
    context.stroke();
    if (offset !== 0) context.fillText(`${offset > 0 ? "+" : ""}${offset}m`, x + 3, rect.height / 2 - 4);
  }

  const trail = world.trail || [];
  if (trail.length > 1) {
    context.strokeStyle = "rgba(113, 241, 200, 0.42)";
    context.lineWidth = 1.5;
    context.beginPath();
    trail.forEach((point, index) => {
      const screen = toScreen(point.x, point.y);
      if (index === 0) context.moveTo(screen.x, screen.y);
      else context.lineTo(screen.x, screen.y);
    });
    context.stroke();
  }

  const robots = [{
        robotId: state.snapshot?.simulation?.backend === "mujoco-go2" ? "Go2" : "G1",
        x: pose.x,
        y: pose.y,
        yaw: pose.yaw,
      }];
  robots.forEach((item, index) => {
    const robot = toScreen(item.x, item.y);
    const color = index === 0 ? "#71f1c8" : "#79a7ff";
    context.save();
    context.translate(robot.x, robot.y);
    context.rotate(-item.yaw + Math.PI / 2);
    context.fillStyle = color;
    context.shadowColor = color;
    context.shadowBlur = 12;
    context.beginPath();
    context.moveTo(0, -10);
    context.lineTo(7, 8);
    context.lineTo(0, 5);
    context.lineTo(-7, 8);
    context.closePath();
    context.fill();
    context.restore();
    context.shadowBlur = 0;
    context.fillStyle = "#b8d8cf";
    context.fillText(
      `${item.robotId}  ${formatNumber(item.x, 2)}, ${formatNumber(item.y, 2)}`,
      robot.x + 12,
      robot.y + 4,
    );
  });
}

function eventMatchesFilter(event) {
  if (state.filter === "all") return true;
  if (state.filter === "tool") return event.source === "tool";
  if (state.filter === "agent") return event.source === "agent" || event.source === "user";
  if (state.filter === "perception") return event.source === "perception" || event.source === "mapping";
  if (state.filter === "system") return ["ui", "simulation", "safety", "navigation"].includes(event.source);
  return true;
}

function eventDetail(event) {
  const data = event.data || {};
  if (data.output) return data.output;
  const keys = Object.keys(data);
  return keys.length ? JSON.stringify(data, null, 2) : "";
}

function renderEvents() {
  const list = $("#trace-list");
  const filtered = state.events.filter(eventMatchesFilter).slice(-240).reverse();
  if (!filtered.length) {
    const empty = document.createElement("div");
    empty.className = "trace-empty";
    empty.textContent = "当前筛选下没有事件";
    list.replaceChildren(empty);
    return;
  }
  const rows = filtered.map((event) => {
    const row = document.createElement("div");
    row.className = `trace-row ${event.level || "info"}`;

    const time = document.createElement("span");
    time.className = "trace-time";
    time.textContent = localTime(event.timestamp);

    const source = document.createElement("span");
    source.className = "trace-source";
    source.textContent = event.source;

    const content = document.createElement("div");
    content.className = "trace-content";
    const title = document.createElement("strong");
    title.textContent = event.title;
    content.append(title);
    if (event.message) {
      const message = document.createElement("p");
      message.textContent = event.message;
      message.title = event.message;
      content.append(message);
    }
    const detailText = eventDetail(event);
    if (detailText) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "查看原始结果";
      const pre = document.createElement("pre");
      pre.textContent = detailText;
      details.append(summary, pre);
      content.append(details);
    }

    const status = document.createElement("span");
    status.className = "trace-state";
    status.textContent = event.level || "info";
    row.append(time, source, content, status);
    return row;
  });
  list.replaceChildren(...rows);
}

async function pollState() {
  if (state.statePollPending) return;
  state.statePollPending = true;
  try {
    const snapshot = await requestJson("/api/state");
    updateSnapshot(snapshot);
  } catch (error) {
    setStatus($("#sim-status"), "UI 离线", "offline");
    setStatus($("#mcp-status"), "未知", "offline");
    $("#send-command").disabled = true;
    if (!state.commandPending) commandFeedback(error.message, true);
    console.error(error);
  } finally { state.statePollPending = false; }
}

async function pollEvents() {
  if (state.eventsPollPending) return;
  state.eventsPollPending = true;
  try {
    const payload = await requestJson(`/api/events?after=${state.cursor}`);
    if (payload.events.length) {
      state.events.push(...payload.events);
      state.events = state.events.slice(-800);
      state.cursor = payload.cursor;
      renderEvents();
      renderAgentProcess();
    }
  } catch (error) {
    console.error(error);
  } finally { state.eventsPollPending = false; }
}

async function pollCostmap() {
  const expected = state.snapshot?.costmap;
  if (!expected?.available) {
    if (state.costmap) {
      state.costmap = null;
      if (state.snapshot) drawWorldMap(state.snapshot.world, null);
    }
    return;
  }
  if (
    state.costmap?.revision === expected.revision
    && state.costmap?.source === expected.source
  ) return;
  try {
    const payload = await requestJson("/api/costmap");
    state.costmap = payload.available ? payload : null;
    if (state.snapshot) drawWorldMap(state.snapshot.world, state.costmap);
  } catch (error) {
    console.error(error);
  }
}

// One request per camera, including image decoding. A capability permits trying
// the endpoint; the producer and the image load event decide frame readiness.
function createCameraStream({ image, placeholder, indicator, path, permitted, label, waitingLabel = () => "等待画面" }) {
  let enabled = false;
  let loaded = false;
  let pending = false;
  let timer = null;
  let controller = null;
  let displayedUrl = null;
  let generation = 0;
  let epoch = null;

  function render() {
    placeholder.classList.toggle("hidden", enabled && loaded);
    indicator.textContent = enabled && loaded ? label() : waitingLabel();
  }

  function discard() {
    generation += 1;
    loaded = false;
    clearTimeout(timer);
    timer = null;
    controller?.abort();
    image.removeAttribute("src");
    if (displayedUrl) URL.revokeObjectURL(displayedUrl);
    displayedUrl = null;
    render();
  }

  async function requestFrame() {
    if (!enabled || pending) return;
    pending = true;
    const version = generation;
    const began = performance.now();
    const requestController = new AbortController();
    controller = requestController;
    // Bound both transport and decoding, so a hung frame cannot stall retries.
    const deadline = setTimeout(() => requestController.abort(), 2000);
    let nextUrl = null;
    let succeeded = false;
    try {
      const response = await fetch(path, { cache: "no-store", signal: requestController.signal });
      if (!response.ok) throw new Error(`camera HTTP ${response.status}`);
      const blob = await response.blob();
      if (version !== generation || !enabled || requestController.signal.aborted) return;
      nextUrl = URL.createObjectURL(blob);
      await new Promise((resolve, reject) => {
        function cleanup() {
          image.removeEventListener("load", complete);
          image.removeEventListener("error", failed);
          requestController.signal.removeEventListener("abort", aborted);
        }
        function complete() {
          cleanup();
          if (image.naturalWidth > 0) resolve();
          else reject(new Error("empty camera image"));
        }
        function failed() { cleanup(); reject(new Error("camera decode failed")); }
        function aborted() { cleanup(); image.removeAttribute("src"); reject(new Error("camera timeout")); }
        image.addEventListener("load", complete);
        image.addEventListener("error", failed);
        requestController.signal.addEventListener("abort", aborted, { once: true });
        image.src = nextUrl;
      });
      if (version !== generation || !enabled) return;
      if (displayedUrl) URL.revokeObjectURL(displayedUrl);
      displayedUrl = nextUrl;
      nextUrl = null;
      loaded = true;
      succeeded = true;
    } catch (_error) {
      if (version === generation) loaded = false;
    } finally {
      clearTimeout(deadline);
      if (nextUrl) URL.revokeObjectURL(nextUrl);
      if (controller === requestController) controller = null;
      pending = false;
      render();
      if (enabled) {
        const interval = document.hidden ? 1000 : succeeded ? 100 : 250;
        timer = setTimeout(requestFrame, Math.max(0, interval - (performance.now() - began)));
      }
    }
  }

  function sync() {
    const snapshot = state.snapshot;
    const nextEpoch = snapshot ? `${snapshot.simulation.backend}:${snapshot.robot_runtime?.boot_epoch || ""}:${Boolean(snapshot.reset?.active)}` : null;
    const nextEnabled = Boolean(snapshot && !snapshot.reset?.active && permitted(snapshot));
    const changed = nextEpoch !== epoch || nextEnabled !== enabled;
    enabled = nextEnabled;
    epoch = nextEpoch;
    if (changed) discard();
    render();
    if (enabled && !pending && timer === null) void requestFrame();
  }

  function wake() {
    clearTimeout(timer);
    timer = null;
    sync();
  }

  return { sync, wake };
}

function refreshHeadCamera() { state.cameraStream?.sync(); }
function refreshObserverCamera() { state.observerStream?.sync(); }

function bindCameraStreams() {
  state.cameraStream = createCameraStream({
    image: $("#camera-feed"), placeholder: $("#camera-placeholder"), indicator: $("#camera-state"),
    path: "/api/camera.jpg",
    permitted: snapshot => snapshot.simulation.capabilities?.rgb !== false,
    label: () => state.snapshot?.simulation.backend === "isaac-g1" ? "实时 Isaac RGB"
      : state.snapshot?.simulation.backend === "mujoco-go2" ? "实时海康 MV-CU013-A0UC RGB" : "实时共享内存",
  });
  state.observerStream = createCameraStream({
    image: $("#third-person-feed"), placeholder: $("#third-person-placeholder"), indicator: $("#observer-state"),
    path: "/api/third-person.jpg",
    permitted: snapshot => !snapshot.evaluation?.blind_mode
      && Boolean(snapshot.simulation.third_person_enabled)
      && snapshot.simulation.capabilities?.third_person !== false,
    label: () => "实时跟随",
    waitingLabel: () => state.snapshot && !state.snapshot.simulation.third_person_enabled ? "默认关闭" : "等待画面",
  });
}

function updateExecutionModes(agent) {
  const select = $("#execution-mode");
  const supported = (agent.execution_modes || ["terminal"]).includes("composed");
  const option = select.querySelector('option[value="composed"]');
  option.disabled = !supported;
  option.textContent = supported ? "自主组合" : "自主组合（当前后端未开放）";
  select.disabled = Boolean(agent.busy);
  if (!select.dataset.initialized) {
    select.value = supported && agent.dynamic_composition ? "composed" : "terminal";
    select.dataset.initialized = "true";
  }
  if (!supported) select.value = "terminal";
  const tasks = $("#composed-task");
  const catalog = [...(agent.dynamic_composition ? [{ task_key: "", instruction: "输入自定义复杂任务" }] : []), ...(agent.composed_tasks || [])];
  const signature = JSON.stringify(catalog);
  if (tasks.dataset.catalog !== signature) {
    tasks.replaceChildren(...catalog.map(task => {
      const item = document.createElement("option");
      item.value = task.task_key;
      item.textContent = task.instruction;
      return item;
    }));
    tasks.dataset.catalog = signature;
  }
  const composed = select.value === "composed";
  tasks.hidden = !composed;
  tasks.disabled = Boolean(agent.busy);
  $("#composed-task-label").hidden = !composed;
  $("#instruction").readOnly = composed && Boolean(tasks.value);
  $("#pause-composed").hidden = !composed || !agent.busy;
  $("#resume-composed").hidden = !composed || agent.busy || !agent.paused_goal;
  const proposal = agent.proposed_goal;
  $("#composed-goal-preview").hidden = !composed || !proposal;
  $("#confirm-composed-goal").disabled = Boolean(agent.busy);
  const predicates = { acquired: "曾在取物点取得对象", placed_on: "已放在指定桌面并稳定", visited: "到访", at: "最终到达", holding: "在取物点取得并持续持有", released: "放置并确认静止" };
  $("#composed-goal-conditions").replaceChildren(...(proposal?.conditions || []).map(goal => {
    const item = document.createElement("li");
    const pose = proposal.references[goal.target];
    const surface = goal.predicate === "placed_on" ? proposal.placement_surfaces?.[goal.target] : null;
    const requireHeading = goal.require_heading ?? true; // Historical proposals require the full pose.
    const heading = goal.target_source === "visual" ? "；水瓶位置与取物朝向由实时视觉确定" : requireHeading ? `；朝向 ${pose[2]} 弧度` : "；不限制终点朝向";
    const location = surface ? `桌面放物点 (${surface.place_point.join(", ")})；机器人站位 (${pose.slice(0, 2).join(", ")})` : `(${pose.slice(0, 2).join(", ")}${goal.target_source === "visual" ? "，厨房观察点" : ""})`;
    item.textContent = `${goal.id}：${predicates[goal.predicate]} ${goal.entity_id || ""} · ${goal.target} ${location}${heading}${goal.depends_on.length ? `；依赖 ${goal.depends_on.join(", ")}` : ""}`;
    return item;
  }));
  const progress = agent.task_progress || {};
  const progressList = $("#composed-progress");
  progressList.hidden = progress.execution_mode !== "composed" || !(progress.plan || []).length;
  const labels = { step_verified: "步骤已验证", executing: "执行中", action_verified: "动作已验证", action_unverified: "动作待核实" };
  progressList.replaceChildren(...(progress.plan || []).map((goal, index) => {
    const item = document.createElement("li");
    const status = progress.subgoal_progress?.[String(index)]?.status;
    const verified = typeof goal === "object" && (!goal.completion || goal.completion.kind === "goal") && goal.goal_ids.every(id => progress.goal_status?.[id]);
    item.textContent = `${typeof goal === "object" ? goal.label : goal} · ${verified ? "子目标已验证" : labels[status] || "待执行"}`;
    return item;
  }));
}

function selectExecutionTask() {
  updateExecutionModes(state.snapshot?.agent || {});
  if ($("#execution-mode").value === "composed" && $("#composed-task").value) {
    $("#instruction").value = $("#composed-task").selectedOptions[0]?.textContent || "";
  }
}

// 提交自然语言及执行模式；动态首次提交不携带确认令牌，由后端生成待确认目标。
async function submitInstruction(event) {
  event.preventDefault();
  if (state.commandPending) return;
  const textarea = $("#instruction");
  const instruction = textarea.value.trim();
  if (!instruction) {
    showToast("请先输入一条指令", true);
    textarea.focus();
    return;
  }
  $("#send-command").disabled = true;
  state.commandPending = true;
  state.commandError = "";
  commandFeedback("正在提交给 Harness…");
  try {
    const payload = await requestJson("/api/commands", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        instruction,
        execution_mode: $("#execution-mode").value,
        ...($("#execution-mode").value === "composed" && $("#composed-task").value ? { task_key: $("#composed-task").value } : {}),
      }),
    });
    if ($("#execution-mode").value === "terminal") textarea.value = "";
    showToast(payload.message || "指令已发送");
    commandFeedback("Harness 已受理，正在准备任务。组合目标生成后需要你确认。");
    await pollState();
  } catch (error) {
    showToast(error.message, true);
    state.commandError = error.message;
    commandFeedback(error.message, true);
  } finally {
    state.commandPending = false;
    if (!state.snapshot?.agent?.busy && !state.snapshot?.reset?.active && state.snapshot?.robot_runtime?.state === "READY") {
      $("#send-command").disabled = false;
    }
  }
}

// 发送目标原文和 task_key 确认令牌；编辑过指令时须重新生成目标，不能确认旧提议。
// resume 使用暂停目标令牌，后端会重新检查物理状态和剩余预算。
async function confirmComposedGoal(resume = false) {
  const proposal = resume ? state.snapshot?.agent?.paused_goal : state.snapshot?.agent?.proposed_goal;
  if (!proposal || state.snapshot?.agent?.busy) return;
  if (!resume && $("#instruction").value.trim() !== proposal.instruction) {
    showToast("指令已修改，请重新发送以生成目标。", true);
    return;
  }
  $("#confirm-composed-goal").disabled = true;
  try {
    const payload = await requestJson("/api/commands", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction: proposal.instruction, execution_mode: "composed", task_key: proposal.task_key }) });
    showToast(payload.message || "目标已确认");
    await pollState();
  } catch (error) { showToast(error.message, true); }
  finally { $("#confirm-composed-goal").disabled = Boolean(state.snapshot?.agent?.busy); }
}

async function emergencyStop() {
  const button = $("#emergency-stop");
  button.disabled = true;
  clearManualKeys(false);
  try {
    const payload = await requestJson("/api/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    showToast(payload.message || "急停已触发");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setTimeout(() => { button.disabled = false; }, 900);
  }
}

async function cancelLongTask() {
  const jobId = state.snapshot?.long_task?.job?.job_id;
  if (!jobId) return;
  const button = $("#cancel-long-task");
  button.disabled = true;
  try {
    await requestJson(`/api/mcp-jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    showToast("长任务取消与停车已触发");
    await pollState();
  } catch (error) {
    showToast(error.message, true);
  }
}

function manualAxes() {
  const forward = (state.manualKeys.has("KeyW") ? 1 : 0)
    - (state.manualKeys.has("KeyS") ? 1 : 0);
  const turn = (state.manualKeys.has("KeyA") ? 1 : 0)
    - (state.manualKeys.has("KeyD") ? 1 : 0);
  return { forward, turn };
}

function renderManualKeys() {
  $$("[data-manual-key]").forEach((key) => {
    key.classList.toggle(
      "active",
      state.manualKeys.has(key.dataset.manualKey),
    );
  });
}

function clearManualKeys(sendStop = true) {
  const hadKeys = state.manualKeys.size > 0;
  state.manualKeys.clear();
  renderManualKeys();
  if (sendStop && hadKeys && state.snapshot?.manual_control?.enabled) {
    sendManualCommand();
  }
}

async function sendManualCommand() {
  if (!state.snapshot?.manual_control?.enabled) return;
  if (state.manualCommandInFlight) {
    state.manualCommandPending = true;
    return;
  }
  const axes = manualAxes();
  state.manualCommandInFlight = true;
  try {
    await requestJson("/api/manual-control/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(axes),
    });
  } catch (error) {
    clearManualKeys(false);
    const now = Date.now();
    if (now - state.lastManualErrorAt > 1500) {
      showToast(error.message, true);
      state.lastManualErrorAt = now;
    }
  } finally {
    state.manualCommandInFlight = false;
    if (state.manualCommandPending) {
      state.manualCommandPending = false;
      sendManualCommand();
    }
  }
}

async function toggleManualMode() {
  const enabled = !Boolean(state.snapshot?.manual_control?.enabled);
  clearManualKeys(false);
  try {
    const payload = await requestJson("/api/manual-control/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    showToast(
      payload.message
      || (enabled ? "键盘控制已开启" : "键盘控制已关闭"),
    );
    await pollState();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function setManualTarget(event) {
  clearManualKeys(false);
  try {
    const payload = await requestJson("/api/manual-control/target", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target: event.target.value }),
    });
    showToast(payload.message || "控制对象已切换");
    await pollState();
  } catch (error) {
    showToast(error.message, true);
    await pollState();
  }
}

async function personControlAction(action) {
  clearManualKeys(false);
  try {
    const payload = await requestJson("/api/person-control/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    showToast(payload.message || "人物状态已更新");
    await pollState();
  } catch (error) {
    showToast(error.message, true);
  }
}

function manualKeyDown(event) {
  if (event.code === "Escape" && state.snapshot?.manual_control?.enabled) {
    event.preventDefault();
    toggleManualMode();
    return;
  }
  if (!["KeyW", "KeyA", "KeyS", "KeyD"].includes(event.code)) return;
  const tagName = event.target?.tagName?.toLowerCase();
  if (["input", "textarea", "select"].includes(tagName)) return;
  if (!state.snapshot?.manual_control?.enabled) return;
  event.preventDefault();
  if (!state.manualKeys.has(event.code)) {
    state.manualKeys.add(event.code);
    renderManualKeys();
    sendManualCommand();
  }
}

function manualKeyUp(event) {
  if (!["KeyW", "KeyA", "KeyS", "KeyD"].includes(event.code)) return;
  if (!state.manualKeys.has(event.code)) return;
  event.preventDefault();
  state.manualKeys.delete(event.code);
  renderManualKeys();
  sendManualCommand();
}

async function startSimulation() {
  try {
    await requestJson("/api/simulation/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    showToast("仿真启动请求已提交");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function resetExperiment() {
  const isaacBackend = state.snapshot?.simulation?.backend === "isaac-g1";
  const confirmed = window.confirm(
    isaacBackend
      ? "确定重启当前 Isaac 实验吗？\n\n机器人会先归零速度，清空当前轨迹和 Agent 会话，再重新启动 Isaac G1。导航与 lidar 安全仍保持关闭。"
      : "确定复位当前实验吗？\n\n机器人会先急停；当前建图、轨迹、命名位置和空间记忆将被清空，然后重新启动 MuJoCo 与建图。工具调用记录会保留。",
  );
  if (!confirmed) return;

  const button = $("#reset-experiment");
  button.disabled = true;
  button.classList.add("busy");
  try {
    const payload = await requestJson("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    state.costmap = null;
    showToast(payload.message || "实验复位已开始");
    await pollState();
  } catch (error) {
    button.disabled = false;
    button.classList.remove("busy");
    showToast(error.message, true);
  }
}

async function switchScene() {
  const sceneId = $("#scene-select").value;
  const seed = Number.parseInt($("#scene-seed").value, 10);
  if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) {
    showToast("Seed 必须是 0 到 2147483647 之间的整数", true);
    return;
  }
  const confirmed = window.confirm(
    state.snapshot?.simulation?.backend === "isaac-g1"
      ? "切换 Isaac 场景会先归零速度并重启当前 G1 仿真。只有新场景产出新鲜 Stable-ID lidar、且目标位于已观测自由区域时才允许自主导航；未知区域会被拒绝。\n\n确定继续吗？"
      : "切换场景会先停车，并清空当前地图、轨迹、地点标签与空间记忆。工具调用记录会保留。\n\n确定继续吗？",
  );
  if (!confirmed) return;

  const button = $("#switch-scene");
  button.disabled = true;
  try {
    const payload = await requestJson("/api/scene/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scene_id: sceneId,
        seed,
        include_person: $("#scene-person").checked,
      }),
    });
    state.costmap = null;
    state.sceneSelectionDirty = false;
    showToast(payload.message || "场景切换已开始");
    await pollState();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = Boolean(state.snapshot?.reset?.active);
  }
}

async function saveCostmap() {
  const button = $("#save-map");
  button.disabled = true;
  try {
    const payload = await requestJson("/api/costmap/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    showToast(payload.message || "地图快照已保存");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = Boolean(state.snapshot?.reset?.active);
  }
}

function bindInteractions() {
  $("#execution-mode").addEventListener("change", selectExecutionTask);
  $("#composed-task").addEventListener("change", selectExecutionTask);
  $("#confirm-composed-goal").addEventListener("click", () => confirmComposedGoal());
  $("#resume-composed").addEventListener("click", () => confirmComposedGoal(true));
  $("#pause-composed").addEventListener("click", async () => {
    try { const result = await requestJson("/api/composed/pause", { method: "POST" }); showToast(result.message); await pollState(); }
    catch (error) { showToast(error.message, true); }
  });
  $("#command-form").addEventListener("submit", submitInstruction);
  $("#instruction").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      $("#command-form").requestSubmit();
    }
  });
  $$("[data-command]").forEach((button) => {
    button.addEventListener("click", () => {
      if (state.snapshot?.agent?.busy) return;
      $("#execution-mode").value = "terminal";
      selectExecutionTask();
      $("#instruction").value = button.dataset.command;
      $("#instruction").focus();
    });
  });
  $$("[data-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      $$("[data-filter]").forEach((item) => item.classList.toggle("active", item === button));
      renderEvents();
    });
  });
  $("#emergency-stop").addEventListener("click", emergencyStop);
  $("#cancel-long-task").addEventListener("click", cancelLongTask);
  $("#reset-experiment").addEventListener("click", resetExperiment);
  $("#scene-select").addEventListener("change", () => {
    state.sceneSelectionDirty = true;
    renderPendingScene(state.snapshot?.simulation?.scene);
  });
  $("#scene-seed").addEventListener("input", () => { state.sceneSelectionDirty = true; });
  $("#scene-person").addEventListener("change", () => { state.sceneSelectionDirty = true; });
  $("#switch-scene").addEventListener("click", switchScene);
  $("#start-sim").addEventListener("click", startSimulation);
  $("#save-map").addEventListener("click", saveCostmap);
  $("#manual-mode-toggle").addEventListener("click", toggleManualMode);
  $("#manual-control-target").addEventListener("change", setManualTarget);
  $("#person-pause").addEventListener("click", () => personControlAction("pause"));
  $("#person-resume").addEventListener("click", () => personControlAction("resume"));
  window.addEventListener("keydown", manualKeyDown);
  window.addEventListener("keyup", manualKeyUp);
  window.addEventListener("blur", () => clearManualKeys());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearManualKeys();
    state.cameraStream?.wake();
    state.observerStream?.wake();
  });
  window.addEventListener("resize", () => state.snapshot && drawWorldMap(state.snapshot.world, state.costmap));
}

async function boot() {
  bindCameraStreams();
  bindInteractions();
  void pollState();
  void pollEvents();
  void pollCostmap();
  setInterval(pollState, 750);
  setInterval(pollEvents, 650);
  setInterval(pollCostmap, 800);
  state.manualTimer = setInterval(() => {
    if (state.manualKeys.size) sendManualCommand();
  }, 120);
}

boot();
