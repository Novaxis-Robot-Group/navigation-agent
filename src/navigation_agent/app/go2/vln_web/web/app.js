(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const ui = {
    web: $("web-pill"), camera: $("camera-pill"), vln: $("vln-pill"), robot: $("robot-pill"), source: $("source-pill"),
    image: $("camera-stream"), overlay: $("trajectory"), pathNote: $("path-note"),
    instruction: $("instruction"), taskMode: $("vln-task-mode"), vlnToggle: $("vln-toggle"), vlnMode: $("vln-mode"),
    serverUrl: $("server-url"), serverUrlSave: $("server-url-save"),
    count: $("waypoint-count"), latency: $("latency"), rate: $("result-rate"), sequence: $("sequence"), vlnStatus: $("vln-status"),
    robotMode: $("robot-mode"), battery: $("battery"), imu: $("imu"), motor: $("motor"),
    motorTemp: $("motor-temp"), motorTemps: $("motor-temps"), robotStatus: $("robot-status"),
    control: $("control-toggle"), stop: $("stop-button"), walkAction: $("walk-action"), sitAction: $("sit-action"), policyToggle: $("policy-toggle"), linearTarget: $("linear-target"),
    linearFeedback: $("linear-feedback"), angularTarget: $("angular-target"), angularFeedback: $("angular-feedback"),
    linearChart: $("linear-chart"), angularChart: $("angular-chart"), toast: $("toast"),
    wifiButton: $("wifi-button"), wifiModal: $("wifi-modal"), wifiClose: $("wifi-close"),
    wifiCurrent: $("wifi-current"), wifiSsid: $("wifi-ssid"), wifiNetworks: $("wifi-networks"),
    wifiPassword: $("wifi-password"), wifiStatus: $("wifi-status"),
    wifiRefresh: $("wifi-refresh"), wifiConnect: $("wifi-connect"),
    manualConfigForm: $("manual-config-form"), manualConfigSave: $("manual-config-save"),
    manualConfigStatus: $("manual-config-status"),
    mpcConfigForm: $("mpc-config-form"), mpcConfigSave: $("mpc-config-save"),
    mpcConfigStatus: $("mpc-config-status")
  };
  const manualConfigInputs = Array.from(document.querySelectorAll("[data-manual-param]"));
  const mpcConfigInputs = Array.from(document.querySelectorAll("[data-mpc-param]"));
  const state = {
    socket: null, reconnect: null, connected: false,
    instructionTouched: false, modeTouched: false, serverUrlTouched: false,
    vln: {available:false,enabled:false,connected:false,mode:"track",server_url:"",message:"Waiting for vln_client"},
    mpc: {available:false,enabled:false,active:false,reason:"disabled",message:"Waiting for vln_mpc"},
    mpcConfig: {available:false,error:"Waiting for vln_mpc"},
    mpcConfigTouched: false, mpcConfigPending: false,
    manualConfigTouched: false, manualConfigPending: false,
    robot: {available:false,connected:false,mode:"UNKNOWN",message:"Waiting for robot adapter"},
    wifi: {available:false,current_ssid:"",networks:[],scanning:false,connecting:false,error:""},
    camera: {received:false,age_ms:null,fps:0}, responseHz: 0, pathFrame: "", waypoints: [],
    owner: false, autoOwner: false, locked: false, source: "disabled", keys: new Set(),
    limits: {linear:1.5,angular:3,linearAccel:1,angularAccel:2}, target: {linear:0,angular:0}, feedback: {linear:0,angular:0},
    toastTimer: null
  };
  let cameraRequestActive = false;
  let cameraObjectUrl = null;
  let wifiNetworkSignature = "";
  const manualRamp = {
    linear: 0,
    angular: 0,
    updatedMs: performance.now()
  };
  const MOTOR_TEMP_LABELS = [
    "FR hip", "FR thigh", "FR calf", "FL hip", "FL thigh", "FL calf",
    "RR hip", "RR thigh", "RR calf", "RL hip", "RL thigh", "RL calf"
  ];
  const MOTOR_TEMP_WARM_C = 55;
  const MOTOR_TEMP_HOT_C = 75;
  let motorTempsSignature = "";

  function setText(element, text) {
    const value = String(text);
    if (element.textContent !== value) element.textContent = value;
  }
  function setDisabled(element, disabled) {
    const value = Boolean(disabled);
    if (element.disabled !== value) element.disabled = value;
  }
  function setPill(element, online, error=false, label="") {
    element.classList.toggle("online", online);
    element.classList.toggle("error", error);
    if (label && element.lastChild.textContent !== label) element.lastChild.textContent = label;
  }
  function connect() {
    clearTimeout(state.reconnect);
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${location.host}/ws`);
    state.socket = socket;
    socket.onopen = () => { state.connected = true; render(); };
    socket.onclose = () => {
      state.connected = false;
      state.owner = false;
      clearKeys(false);
      render();
      state.reconnect = setTimeout(connect, 1000);
    };
    socket.onerror = () => socket.close();
    socket.onmessage = (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch (_) { return; }
      handleMessage(message);
    };
  }
  async function loadCameraFrame() {
    if (cameraRequestActive) return;
    cameraRequestActive = true;
    let objectUrl = null;
    try {
      const response = await fetch("/api/camera.jpg", {cache:"no-store"});
      if (!response.ok) throw new Error(`camera request failed: ${response.status}`);
      objectUrl = URL.createObjectURL(await response.blob());
      await new Promise((resolve, reject) => {
        ui.image.onload = resolve;
        ui.image.onerror = reject;
        ui.image.src = objectUrl;
      });
      const previousUrl = cameraObjectUrl;
      cameraObjectUrl = objectUrl;
      objectUrl = null;
      if (previousUrl) URL.revokeObjectURL(previousUrl);
    } catch (_) {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    } finally {
      cameraRequestActive = false;
    }
  }
  function handleMessage(message) {
    if (message.type === "odom") {
      setOdom(message.data || {});
      return;
    }
    if (message.type === "sent_command") {
      setSent(message.data || {});
      return;
    }
    if (message.type === "path") {
      setPath(message.data || {});
      return;
    }
    if (message.type === "snapshot" || message.type === "runtime") update(message.data || {});
    else if (message.type === "vln_status") {
      state.vln = message.data || state.vln;
      if (!state.serverUrlTouched && typeof state.vln.server_url === "string") {
        ui.serverUrl.value = state.vln.server_url;
      }
    }
    else if (message.type === "mpc_status") state.mpc = message.data || state.mpc;
    else if (message.type === "mpc_config") setMpcConfigState(message.data || {});
    else if (message.type === "manual_limits") setManualLimits(message.data || {});
    else if (message.type === "robot_diagnostics") state.robot = message.data || state.robot;
    else if (message.type === "wifi_status") state.wifi = message.data || state.wifi;
    else if (message.type === "control_source") state.source = String(message.source || "disabled");
    else if (message.type === "control_state") {
      const wasOwner = state.owner;
      state.owner = Boolean(message.owner);
      state.autoOwner = Boolean(message.auto_owner);
      state.locked = Boolean(message.locked);
      if (state.owner !== wasOwner) resetManualRamp();
      if (!state.owner) clearKeys(false);
    } else if (message.type === "command_result") {
      if (message.ok && message.command === "server_url") {
        state.serverUrlTouched = false;
        state.vln.server_url = String(message.server_url || "");
        ui.serverUrl.value = state.vln.server_url;
      }
      if (message.command === "mpc_config") {
        state.mpcConfigPending = false;
        if (message.ok) {
          state.mpcConfigTouched = false;
          fillMpcConfigInputs();
        }
      }
      if (message.command === "manual_limits") {
        state.manualConfigPending = false;
        if (message.ok) {
          state.manualConfigTouched = false;
          fillManualConfigInputs();
        }
      }
      toast(message.message || (message.ok ? "Done" : "Failed"), !message.ok);
    }
    render();
  }
  function update(data) {
    if (data.vln) state.vln = data.vln;
    if (data.mpc) state.mpc = data.mpc;
    if (data.mpc_config) setMpcConfigState(data.mpc_config);
    if (data.robot) state.robot = data.robot;
    if (data.wifi) state.wifi = data.wifi;
    if (data.camera) state.camera = data.camera;
    if (data.path) setPath(data.path);
    if (data.sent_command) setSent(data.sent_command);
    if (data.odom) setOdom(data.odom);
    if (typeof data.control_source === "string") state.source = data.control_source;
    if (Number.isFinite(Number(data.response_hz))) state.responseHz = Number(data.response_hz);
    if (data.manual_limits) setManualLimits(data.manual_limits);
    if (!state.instructionTouched && state.vln.instruction) ui.instruction.value = state.vln.instruction;
    if (!state.serverUrlTouched && typeof state.vln.server_url === "string") {
      ui.serverUrl.value = state.vln.server_url;
    }
  }
  function setPath(path) {
    const points = path.body_waypoints;
    state.waypoints = Array.isArray(points)
      ? points.filter((point) => Array.isArray(point) && point.length >= 3 && point.every(Number.isFinite))
      : [];
    if (typeof path.frame_id === "string") state.pathFrame = path.frame_id;
    ui.pathNote.textContent = state.waypoints.length
      ? `${state.waypoints.length} waypoints · ${state.pathFrame || "unknown frame"}`
      : "Waiting for body-frame waypoints";
    drawWaypoints();
  }
  function finiteArray(data, name, index) {
    const value = Number(Array.isArray(data[name]) ? data[name][index] : NaN);
    return Number.isFinite(value) ? value : 0;
  }
  function setOdom(data) {
    state.feedback.linear = finiteArray(data, "twist_linear", 0);
    state.feedback.angular = finiteArray(data, "twist_angular", 2);
    ui.linearFeedback.textContent = state.feedback.linear.toFixed(2);
    ui.angularFeedback.textContent = state.feedback.angular.toFixed(2);
    const stamp = Number(data.stamp_ns) / 1e9;
    if (!Number.isFinite(stamp) || stamp <= 0) return;
    linearChart.append(stamp, state.target.linear, state.feedback.linear);
    angularChart.append(stamp, state.target.angular, state.feedback.angular);
  }
  function setSent(data) {
    const linear = Number(data.linear), angular = Number(data.angular);
    state.target.linear = Number.isFinite(linear) ? linear : 0;
    state.target.angular = Number.isFinite(angular) ? angular : 0;
    ui.linearTarget.textContent = state.target.linear.toFixed(2);
    ui.angularTarget.textContent = state.target.angular.toFixed(2);
  }
  function send(payload) {
    if (state.socket?.readyState !== WebSocket.OPEN) {
      toast("WebSocket not connected", true);
      return false;
    }
    state.socket.send(JSON.stringify(payload));
    return true;
  }
  function setVln(enabled) {
    const instruction = ui.instruction.value.trim();
    if (enabled && !instruction) return toast("Enter a VLN instruction first", true);
    return send({type:"set_vln", enabled, instruction, mode:ui.taskMode.value});
  }
  function setServerUrl() {
    const serverUrl = ui.serverUrl.value.trim();
    if (!serverUrl) return toast("Enter the VLN server URL", true);
    send({type:"set_server_url", server_url:serverUrl});
  }
  function setMpcConfigState(config) {
    state.mpcConfig = config;
    if (!state.mpcConfigTouched) fillMpcConfigInputs();
  }
  function setManualLimits(config) {
    const linear = Number(config.linear);
    const angular = Number(config.angular);
    const linearAccel = Number(config.linear_accel);
    const angularAccel = Number(config.angular_accel);
    if (Number.isFinite(linear) && linear > 0) state.limits.linear = linear;
    if (Number.isFinite(angular) && angular > 0) state.limits.angular = angular;
    if (Number.isFinite(linearAccel) && linearAccel > 0) state.limits.linearAccel = linearAccel;
    if (Number.isFinite(angularAccel) && angularAccel > 0) state.limits.angularAccel = angularAccel;
    if (!state.manualConfigTouched) fillManualConfigInputs();
  }
  function fillManualConfigInputs() {
    const values = {
      linear: state.limits.linear,
      angular: state.limits.angular,
      linear_accel: state.limits.linearAccel,
      angular_accel: state.limits.angularAccel
    };
    manualConfigInputs.forEach((input) => {
      input.value = String(values[input.dataset.manualParam]);
    });
  }
  function submitManualConfig() {
    const config = {};
    for (const input of manualConfigInputs) {
      const value = Number(input.value);
      if (!Number.isFinite(value) || value <= 0) {
        input.focus();
        toast(`${input.dataset.manualParam} must be greater than 0`, true);
        return;
      }
      config[input.dataset.manualParam] = value;
    }
    if (send({type:"set_manual_limits", config})) {
      state.manualConfigPending = true;
      render();
    }
  }
  function fillMpcConfigInputs() {
    mpcConfigInputs.forEach((input) => {
      const value = Number(state.mpcConfig[input.dataset.mpcParam]);
      if (Number.isFinite(value)) input.value = String(value);
    });
  }
  function submitMpcConfig() {
    const config = {};
    for (const input of mpcConfigInputs) {
      const value = Number(input.value);
      const minimum = Number(input.min);
      if (!Number.isFinite(value) || value < minimum) {
        input.focus();
        toast(`${input.dataset.mpcParam} must be at least ${minimum}`, true);
        return;
      }
      config[input.dataset.mpcParam] = value;
    }
    if (send({type:"set_mpc_config", config})) {
      state.mpcConfigPending = true;
      render();
    }
  }
  function manualCommand() {
    if (!state.owner || state.robot.mode !== "WALK") return {linear:0, angular:0};
    return {
      linear: ((state.keys.has("w") ? 1 : 0) - (state.keys.has("s") ? 1 : 0)) * state.limits.linear,
      angular: ((state.keys.has("a") ? 1 : 0) - (state.keys.has("d") ? 1 : 0)) * state.limits.angular
    };
  }
  function resetManualRamp() {
    manualRamp.linear = 0;
    manualRamp.angular = 0;
    manualRamp.updatedMs = performance.now();
  }
  function approach(current, target, maximumStep) {
    return current + Math.max(-maximumStep, Math.min(maximumStep, target - current));
  }
  function publishManual() {
    const now = performance.now();
    const dt = Math.min(0.2, Math.max(0, (now - manualRamp.updatedMs) / 1000));
    manualRamp.updatedMs = now;
    if (state.owner && state.robot.mode === "WALK") {
      const target = manualCommand();
      manualRamp.linear = approach(manualRamp.linear, target.linear, state.limits.linearAccel * dt);
      manualRamp.angular = approach(manualRamp.angular, target.angular, state.limits.angularAccel * dt);
      send({type:"twist", x:manualRamp.linear, y:0, z:manualRamp.angular});
    } else {
      resetManualRamp();
      if (state.owner) send({type:"twist", x:0, y:0, z:0});
    }
    document.querySelectorAll(".key").forEach((key) => key.classList.toggle("active", state.keys.has(key.dataset.key)));
  }
  function clearKeys(sendZero=true) {
    state.keys.clear();
    if (sendZero) publishManual();
  }
  function releaseControl() {
    if (!state.owner) return;
    clearKeys();
    send({type:"release_control"});
    state.owner = false;
    render();
  }
  function render() {
    const vln = state.vln, mpc = state.mpc, robot = state.robot;
    const mpcConfigLocked = Boolean(vln.enabled || mpc.enabled || state.autoOwner);
    const mpcConfigEditable = Boolean(
      state.connected && state.mpcConfig.available
        && !state.mpcConfigPending && !mpcConfigLocked
    );
    const manualConfigEditable = Boolean(
      state.connected && !state.locked && !state.manualConfigPending
    );
    if (["objnav", "track"].includes(vln.mode) && (!state.modeTouched || vln.enabled)) {
      ui.taskMode.value = vln.mode;
      if (vln.enabled) state.modeTouched = false;
    }
    const cameraOnline = Boolean(state.camera.received) && Number(state.camera.age_ms) < 1500;
    setPill(ui.web, state.connected, false, state.connected ? "WEB" : "WEB OFFLINE");
    setPill(ui.camera, cameraOnline, false, "CAMERA");
    setPill(ui.vln, Boolean(vln.connected), vln.available && vln.level >= 2, "VLN");
    setPill(ui.robot, Boolean(robot.connected), robot.available && robot.level >= 2, "ROBOT");
    const source = String(state.source || "disabled").toUpperCase();
    setPill(ui.source, source !== "DISABLED", false, source);
    setPill(ui.wifiButton, Boolean(state.wifi.current_ssid), Boolean(state.wifi.error), "WIFI");
    ui.wifiButton.title = state.wifi.current_ssid || "Switch Wi-Fi";

    ui.vlnMode.textContent = `${vln.enabled ? "VLN ON" : "VLN OFF"} · ${mpc.enabled ? (mpc.active ? "MPC ACTIVE" : "MPC WAIT") : "MPC OFF"}`;
    const pipelineEnabled = Boolean(vln.enabled || mpc.enabled || state.autoOwner);
    const pipelineStarting = state.autoOwner && (!vln.enabled || !mpc.enabled);
    setDisabled(
      ui.vlnToggle,
      !state.connected || pipelineStarting || (
        !pipelineEnabled && (!vln.available || !mpc.available || !robot.connected)
      )
    );
    setDisabled(ui.taskMode, !state.connected || vln.enabled || mpc.enabled);
    setDisabled(ui.serverUrlSave, !state.connected || !ui.serverUrl.value.trim());
    setDisabled(ui.mpcConfigSave, !mpcConfigEditable);
    mpcConfigInputs.forEach((input) => setDisabled(input, !mpcConfigEditable));
    ui.mpcConfigStatus.textContent = state.mpcConfig.available
      ? (mpcConfigLocked
          ? "Editable after stopping VLN/MPC"
          : (state.mpcConfigPending ? "Applying…" : "Connected · hot reload"))
      : (state.mpcConfig.error || "Waiting for vln_mpc");
    setDisabled(ui.manualConfigSave, !manualConfigEditable);
    manualConfigInputs.forEach((input) => setDisabled(input, !manualConfigEditable));
    ui.manualConfigStatus.textContent = !state.connected
      ? "Waiting for web connection"
      : (state.manualConfigPending
          ? "Applying…"
          : (state.locked ? "Editable after releasing control" : "Takes effect immediately in the current process"));
    setText(ui.vlnToggle, pipelineStarting ? "Starting…" : (pipelineEnabled ? "Stop VLN" : "Start VLN"));
    ui.vlnToggle.classList.toggle("active", pipelineEnabled);
    ui.count.textContent = vln.waypoint_count ?? state.waypoints.length;
    ui.latency.textContent = Number.isFinite(Number(vln.last_latency_ms)) ? `${Number(vln.last_latency_ms).toFixed(0)} ms` : "—";
    ui.rate.textContent = `${state.responseHz.toFixed(1)} Hz`;
    ui.sequence.textContent = vln.last_sequence ?? 0;
    const vlnDetail = [vln.message || "Waiting for vln_client"];
    if (vln.visible === false) vlnDetail.push("target not visible");
    if (vln.stop) vlnDetail.push("VLN STOP");
    if (mpc.enabled) {
      vlnDetail.push(mpc.active
        ? "MPC tracking"
        : `MPC waiting: ${mpc.reason || mpc.message || "not ready"}`);
    }
    if (mpc.error) vlnDetail.push(`MPC: ${mpc.error}`);
    ui.vlnStatus.textContent = vlnDetail.join(" · ");
    ui.robotMode.textContent = robot.mode || "UNKNOWN";
    ui.battery.textContent = Number.isFinite(Number(robot.battery)) ? `${Math.round(Number(robot.battery))}%` : "—";
    ui.imu.textContent = robot.imu || "—";
    ui.motor.textContent = robot.motor || "—";
    renderMotorTemps(robot);
    ui.robotStatus.textContent = robot.connected
      ? `${robot.adapter || "robot"} · ${robot.robot_id || "online"}`
      : (robot.message || "Waiting for robot adapter");
    const isGo2 = String(robot.adapter || "").toLowerCase() === "go2";
    ui.walkAction.hidden = isGo2;
    setText(ui.sitAction, isGo2 ? "Lie down" : "Crouch");
    setDisabled(ui.control, !state.connected || !robot.connected || (state.locked && !state.owner));
    setText(ui.control, state.owner ? "Release manual control" : (state.autoOwner ? "MPC auto control" : (state.locked ? "Another page is in control" : "Take manual control")));
    ui.control.classList.toggle("active", state.owner);
    setDisabled(ui.stop, !state.connected);
    document.querySelectorAll("[data-action]").forEach((button) => setDisabled(button, !state.owner || !robot.connected));
    ui.policyToggle.hidden = !robot.vln_policy;
    if (!ui.policyToggle.hidden) {
      const policy = String(robot.policy || "");
      const isVlnPolicy = policy === robot.vln_policy;
      const label = isVlnPolicy ? "VLN" : (policy === "locomotion" ? "loco" : policy);
      setText(ui.policyToggle, `Policy: ${label || "—"}`);
      ui.policyToggle.classList.toggle("active", isVlnPolicy);
      setDisabled(ui.policyToggle, !state.connected || !robot.connected);
    }
    document.querySelectorAll(".key").forEach((button) => setDisabled(button, !state.owner || robot.mode !== "WALK"));
    renderWifi();
  }

  function renderMotorTemps(robot) {
    const temps = Array.isArray(robot.motor_temps)
      ? robot.motor_temps.map((value) => Number(value))
      : [];
    const validTemps = temps.filter((value) => Number.isFinite(value) && value > 0);
    let maxTemp = Number(robot.motor_temp_max);
    if (!Number.isFinite(maxTemp) || maxTemp <= 0) {
      maxTemp = validTemps.length ? Math.max(...validTemps) : NaN;
    }
    const hot = maxTemp >= MOTOR_TEMP_HOT_C;
    const warm = !hot && maxTemp >= MOTOR_TEMP_WARM_C;
    ui.motorTemp.textContent = Number.isFinite(maxTemp) && maxTemp > 0 ? `${Math.round(maxTemp)}°C` : "—";
    ui.motorTemp.classList.toggle("warm", warm);
    ui.motorTemp.classList.toggle("hot", hot);
    ui.motorTemps.hidden = validTemps.length === 0;
    const signature = JSON.stringify(temps);
    if (signature !== motorTempsSignature) {
      motorTempsSignature = signature;
      ui.motorTemps.replaceChildren(...MOTOR_TEMP_LABELS.map((label, index) => {
        const cell = document.createElement("div");
        cell.className = "motor-cell";
        const value = temps[index];
        const valid = Number.isFinite(value) && value > 0;
        if (valid && value >= MOTOR_TEMP_HOT_C) cell.classList.add("hot");
        else if (valid && value >= MOTOR_TEMP_WARM_C) cell.classList.add("warm");
        const name = document.createElement("small");
        name.textContent = label;
        const reading = document.createElement("strong");
        reading.textContent = valid ? `${Math.round(value)}°` : "—";
        cell.append(name, reading);
        return cell;
      }));
    }
  }

  function renderWifi() {
    const wifi = state.wifi || {};
    ui.wifiCurrent.textContent = wifi.current_ssid || "Not connected";
    const networks = Array.isArray(wifi.networks) ? wifi.networks : [];
    const signature = JSON.stringify(networks.map((network) => [
      network.ssid, network.signal, network.security
    ]));
    if (signature !== wifiNetworkSignature) {
      wifiNetworkSignature = signature;
      ui.wifiNetworks.replaceChildren(...networks.map((network) => {
        const option = document.createElement("option");
        option.value = String(network.ssid || "");
        option.label = `${Number(network.signal || 0)}% · ${network.security || "OPEN"}`;
        return option;
      }));
    }
    const busy = Boolean(wifi.scanning || wifi.connecting);
    setDisabled(ui.wifiRefresh, !state.connected || busy);
    setDisabled(ui.wifiConnect, !state.connected || busy || !ui.wifiSsid.value.trim());
    setText(ui.wifiConnect, wifi.connecting ? "Connecting…" : "Connect");
    ui.wifiStatus.textContent = wifi.error
      ? wifi.error
      : (wifi.connecting ? "Switching network…" : (wifi.scanning ? "Scanning…" : `${networks.length} networks available`));
  }

  function drawWaypoints() {
    const canvas = ui.overlay, width = 480, height = 270, context = canvas.getContext("2d");
    context.clearRect(0, 0, width, height);
    if (!state.vln.enabled) return;
    const inside = (point) => point && point.x >= 0 && point.x < width && point.y >= 0 && point.y < height;
    const focal = (width / 2) / Math.tan(Math.PI / 3);
    const project = (point) => {
      const depth = Number(point[0]) + 0.65;
      return depth > 0.05 ? {x:width / 2 - focal * Number(point[1]) / depth, y:height / 2 + focal * 0.5 / depth} : null;
    };
    const pixels = state.waypoints.map(project);
    const color = (ratio) => `rgb(${Math.round(50 + 160 * ratio)},${Math.round(230 - 170 * ratio)},${Math.round(80 - 30 * ratio)})`;
    context.lineWidth = 2;
    context.lineCap = "round";
    for (let index = 1; index < pixels.length; index += 1) {
      if (!inside(pixels[index - 1]) || !inside(pixels[index])) continue;
      context.strokeStyle = color((index - 1) / Math.max(1, pixels.length - 1));
      context.beginPath();
      context.moveTo(pixels[index - 1].x, pixels[index - 1].y);
      context.lineTo(pixels[index].x, pixels[index].y);
      context.stroke();
    }
    pixels.forEach((point, index) => {
      if (!inside(point)) return;
      context.fillStyle = color(index / Math.max(1, pixels.length - 1));
      context.beginPath();
      context.arc(point.x, point.y, 3.5, 0, Math.PI * 2);
      context.fill();
    });

    const drawSemanticMarker = (pixel, channel, semanticState, color) => {
      if (!Array.isArray(pixel) || pixel.length !== 2) return;
      const point = {x:Number(pixel[0]), y:Number(pixel[1])};
      if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !inside(point)) return;
      const label = semanticState ? `${channel} · ${semanticState}` : channel;
      context.save();
      context.strokeStyle = color;
      context.fillStyle = color;
      context.lineWidth = 2;
      context.beginPath();
      context.arc(point.x, point.y, 6, 0, Math.PI * 2);
      context.stroke();
      context.beginPath();
      context.moveTo(point.x - 9, point.y);
      context.lineTo(point.x + 9, point.y);
      context.moveTo(point.x, point.y - 9);
      context.lineTo(point.x, point.y + 9);
      context.stroke();
      context.font = "700 9px ui-monospace, monospace";
      const textWidth = context.measureText(label).width;
      const labelX = Math.max(3, Math.min(width - textWidth - 9, point.x + 9));
      const labelY = point.y < 18 ? point.y + 18 : point.y - 8;
      context.fillStyle = "rgba(2, 8, 6, .78)";
      context.fillRect(labelX - 3, labelY - 10, textWidth + 6, 13);
      context.fillStyle = color;
      context.fillText(label, labelX, labelY);
      context.restore();
    };
    drawSemanticMarker(state.vln.apos_px, "APOS", state.vln.apos_state, "#ffb454");
    drawSemanticMarker(state.vln.opos_px, "OPOS", state.vln.opos_state, "#58cbe8");
  }
  function createBufferedChart(canvas, minimumRange, limitName) {
    const context = canvas.getContext("2d");
    const samples = [];
    const windowSeconds = 5;
    let width = 1, height = 1, dpr = 1;

    function resize() {
      const rect = canvas.getBoundingClientRect();
      const nextDpr = Math.min(devicePixelRatio || 1, 2);
      const nextWidth = Math.max(1, rect.width);
      const nextHeight = Math.max(1, rect.height);
      if (nextWidth === width && nextHeight === height && nextDpr === dpr) return;
      width = nextWidth;
      height = nextHeight;
      dpr = nextDpr;
      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function drawTrace(startStamp, range, field, color) {
      if (!samples.length) return;
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.lineJoin = "round";
      context.lineCap = "round";
      context.beginPath();
      samples.forEach((sample, index) => {
        const x = ((sample.stamp - startStamp) / windowSeconds) * width;
        const value = Math.max(-range, Math.min(range, sample[field]));
        const y = ((range - value) / (2 * range)) * height;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.stroke();
    }

    function draw() {
      resize();
      context.clearRect(0, 0, width, height);
      if (!samples.length) return;
      const outputScale = Number(
        state.mpcConfig[
          limitName === "linear" ? "v_output_scale" : "w_output_scale"
        ]
      );
      const mpcLimit = limitName === "linear"
        ? Math.max(
            Number(state.mpcConfig.track_v_max) || 0,
            Number(state.mpcConfig.objnav_v_max) || 0
          ) * (Number.isFinite(outputScale) ? outputScale : 1)
        : (Number(state.mpcConfig.w_max) || 0)
          * (Number.isFinite(outputScale) ? outputScale : 1);
      const range = Math.max(
        minimumRange,
        Number(state.limits[limitName]) || 0,
        mpcLimit
      ) * 1.15;
      const startStamp = samples[samples.length - 1].stamp - windowSeconds;
      drawTrace(startStamp, range, "target", "#78f2ac");
      drawTrace(startStamp, range, "feedback", "#58cbe8");
    }

    function append(stamp, target, feedback) {
      const lastStamp = samples.length ? samples[samples.length - 1].stamp : null;
      if (lastStamp !== null && stamp <= lastStamp) {
        if (stamp < lastStamp) samples.length = 0;
        else return;
      }
      samples.push({stamp, target, feedback});
      const oldestStamp = stamp - windowSeconds;
      while (samples.length && samples[0].stamp < oldestStamp) samples.shift();
    }

    resize();
    setInterval(draw, 50);
    return {append};
  }
  const linearChart = createBufferedChart(ui.linearChart, .5, "linear");
  const angularChart = createBufferedChart(ui.angularChart, 1, "angular");
  function toast(message, error=false) {
    if (!message) return;
    clearTimeout(state.toastTimer);
    ui.toast.textContent = message;
    ui.toast.className = error ? "visible error" : "visible";
    state.toastTimer = setTimeout(() => ui.toast.className = "", 2800);
  }

  ui.instruction.addEventListener("input", () => state.instructionTouched = true);
  ui.serverUrl.addEventListener("input", () => {
    state.serverUrlTouched = true;
    render();
  });
  ui.serverUrl.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); setServerUrl(); }
  });
  ui.serverUrlSave.addEventListener("click", setServerUrl);
  manualConfigInputs.forEach((input) => input.addEventListener("input", () => {
    state.manualConfigTouched = true;
  }));
  ui.manualConfigForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitManualConfig();
  });
  mpcConfigInputs.forEach((input) => input.addEventListener("input", () => {
    state.mpcConfigTouched = true;
  }));
  ui.mpcConfigForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitMpcConfig();
  });
  ui.taskMode.addEventListener("change", () => state.modeTouched = true);
  ui.instruction.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      if (setVln(true)) ui.instruction.blur();
    }
  });
  ui.vlnToggle.addEventListener("click", () => {
    setVln(!(state.vln.enabled || state.mpc.enabled || state.autoOwner));
  });
  ui.wifiButton.addEventListener("click", () => {
    ui.wifiModal.hidden = false;
    send({type:"wifi_scan"});
    renderWifi();
  });
  ui.wifiClose.addEventListener("click", () => { ui.wifiModal.hidden = true; });
  ui.wifiModal.addEventListener("click", (event) => {
    if (event.target === ui.wifiModal) ui.wifiModal.hidden = true;
  });
  ui.wifiRefresh.addEventListener("click", () => send({type:"wifi_scan"}));
  ui.wifiSsid.addEventListener("input", renderWifi);
  ui.wifiConnect.addEventListener("click", () => {
    const ssid = ui.wifiSsid.value.trim();
    if (!ssid) return;
    if (!confirm(`Switch to ${ssid}? This page may disconnect.`)) return;
    send({type:"wifi_connect", ssid, password:ui.wifiPassword.value});
    ui.wifiPassword.value = "";
  });
  ui.control.addEventListener("click", () => state.owner ? releaseControl() : send({type:"acquire_control"}));
  ui.stop.addEventListener("click", () => { clearKeys(false); send({type:"stop"}); });
  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => send({type:"robot_action", action:button.dataset.action}));
  });
  addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target?.tagName)) return;
    const key = event.key.toLowerCase();
    if (key === " ") {
      event.preventDefault();
      if (event.repeat) return;
      if (state.vln.enabled) setVln(false);
      else ui.stop.click();
      return;
    }
    if (state.owner && state.robot.mode === "WALK" && "wasd".includes(key)) {
      event.preventDefault(); state.keys.add(key); publishManual();
    }
  });
  addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    if ("wasd".includes(key)) { state.keys.delete(key); publishManual(); }
  });
  document.querySelectorAll(".key").forEach((button) => {
    const down = (event) => {
      if (button.disabled) return;
      event.preventDefault(); state.keys.add(button.dataset.key); publishManual();
    };
    const up = (event) => {
      event.preventDefault(); state.keys.delete(button.dataset.key); publishManual();
    };
    button.addEventListener("pointerdown", down);
    button.addEventListener("pointerup", up);
    button.addEventListener("pointercancel", up);
    button.addEventListener("lostpointercapture", up);
  });
  addEventListener("blur", releaseControl);
  document.addEventListener("visibilitychange", () => { if (document.hidden) releaseControl(); });
  setInterval(publishManual, 100);
  setInterval(loadCameraFrame, 100);
  render();
  connect();
  loadCameraFrame();
})();
