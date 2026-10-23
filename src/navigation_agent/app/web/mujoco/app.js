(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  // 页面状态：连接、当前任务、事件游标和手动按键
  const state = {
    socket: null,
    connected: false,
    data: null,
    services: null,
    tasks: [],
    vlnTasks: [],
    vlnAutoSelected: null,
    active: null,
    selected: null,
    events: [],
    corrections: new Set(),
    completionSteps: new Set(),
    cursor: 0,
    taskApi: false,
    agentSettingsLoaded: false,
    navigationTimeLimitEnabled: false,
    busy: false,
    view: 'first',
    keys: new Set(),
    urls: {},
    cameraReady: false,
    polling: false,
    eventTask: null,
    stopEpoch: 0,
  };
  const labels = {
    running: '执行中',
    completed: '已完成',
    failed: '未完成',
    cancelled: '已取消',
    budget_exhausted: '预算用尽',
    interrupted: '已中断',
    stopped: '已停止 · 未评定',
  };
  const phases = {
    starting: '准备任务',
    observing: '正在观察环境',
    deciding: '正在选择下一步',
    acting: '正在执行动作',
    stopped: '任务已结束',
  };
  const actions = {
    look: '环视',
    orient: '转回确认',
    navigate: '导航',
    recover: '退回脱困',
    finish: '完成判断',
    give_up: '结束探索',
  };
  let toastTimer;
  function toast(message, error = false) {
    $('toast').textContent = message;
    $('toast').classList.toggle('error', error);
    $('toast').hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => ($('toast').hidden = true), 6000);
  }
  const node = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const duration = (seconds) => {
    seconds = Math.max(0, Math.floor(seconds || 0));
    return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  };
  const date = (value) =>
    value && !Number.isNaN(Date.parse(value))
      ? new Date(value).toLocaleString('zh-CN', { hour12: false })
      : '历史记录';
  function friendly(message) {
    const known = {
      'A task is already running; cancel it first': '已有任务正在运行，请先结束当前任务。',
      '请配置 Agent 云端模型密钥':
        '请先填写 deploy/.env 中的百炼 Key，并重启 Agent 服务。',
      'Cancelled by user or service shutdown': '任务已取消，机器人已停止。',
      'Cancelled before start': '任务在开始前已取消。',
      'Maximum decision steps reached': '已达到决策轮数上限，可调整预算后重试。',
      'Task or operation timed out': '任务或操作等待超时，已停止。',
      'Robot control was interrupted': '机器人控制已被打断，任务结束。',
      'Three consecutive navigation attempts failed': '连续三次导航未能推进，已结束任务。',
      'Repeated navigation instruction limit reached': '导航指令连续重复，已结束任务。',
      'Cloud model returned an invalid or incomplete action':
        '模型返回的动作不完整或不符合要求，本轮没有执行运动。',
      'Simulator control connection closed': '仿真连接已断开。',
      'Simulator control connection failed': '仿真控制连接异常。',
      'Navigation was interrupted by another controller': '导航被其他控制操作打断。',
    };
    if (known[message]) return known[message];
    if (String(message).startsWith('Cloud model HTTP '))
      return `云模型调用失败（${String(message).match(/HTTP \d+/)?.[0] || '请求异常'}），请检查百炼 Key、模型权限和地域。`;
    if (String(message).startsWith('Execution error:'))
      return `任务执行异常：${message.slice(16)}。请检查服务连接后重试。`;
    return message || '';
  }
  async function api(path, method = 'GET', body) {
    let response;
    try {
      response = await fetch('/api/agent/' + path, {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : {},
        body: body ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(15000),
        cache: 'no-store',
      });
    } catch {
      throw new Error('Agent 服务连接超时，请检查服务窗口。');
    }
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error('服务接口不可用，请重启 Agent 和仿真服务。');
    }
    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? '输入不符合要求，请检查目标和执行预算。'
        : data.detail;
      throw new Error(friendly(detail) || `请求失败（${response.status}）`);
    }
    return data;
  }
  async function vlnApi(path) {
    const response = await fetch('/api/vln/runs' + path, {
      cache: 'no-store', signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) throw new Error('VLN 记录读取失败，请检查仿真服务。');
    return response.json();
  }
  function records() {
    return [...state.tasks, ...state.vlnTasks].sort((a, b) =>
      String(b.started_at).localeCompare(String(a.started_at)));
  }
  function logBase(task) {
    return task.mode === 'vln' ? '/api/vln/runs/' + task.id + '/logs/'
      : '/api/agent/tasks/' + task.id + '/logs/';
  }
  function readEvents(task, cursor) {
    return task.mode === 'vln' ? vlnApi('/' + task.id + '/events?after=' + cursor)
      : api('tasks/' + task.id + '/events?after=' + cursor);
  }
  function send(payload) {
    if (state.socket?.readyState !== WebSocket.OPEN)
      throw new Error('仿真尚未连接，无法发送控制指令。');
    state.socket.send(JSON.stringify(payload));
  }
  // 连接仿真，断线后自动重连
  function connect() {
    const socket = new WebSocket(
      `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`,
    );
    state.socket = socket;
    socket.onopen = () => {
      state.connected = true;
      render();
    };
    socket.onclose = () => {
      state.connected = false;
      state.keys.clear();
      render();
      setTimeout(connect, 1500);
    };
    socket.onerror = () => socket.close();
    socket.onmessage = (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      if (['runtime', 'snapshot'].includes(message.type)) {
        state.data = message.data;
        if (!state.data.control?.manual) state.keys.clear();
      }
      if (message.type === 'command_result' && !message.ok)
        toast(
          message.message === 'another page owns robot control'
            ? '其他控制端正在使用机器人，请先释放控制权。'
            : message.message,
          true,
        );
      render();
    };
  }
  function setService(id, good, text, warning = false) {
    const element = $(id);
    element.classList.toggle('good', good && !warning);
    element.classList.toggle('bad', !good);
    element.classList.toggle('warn', warning);
    element.querySelector('b').textContent = text;
  }
  function readiness() {
    if (!state.connected) return '仿真未连接，请先启动仿真服务';
    if (!state.services?.agent?.online) return 'Agent 未连接，请先启动 Agent 服务';
    if (!state.services.agent.configured) return '请填写百炼 Key，并重启 Agent 服务';
    if (!state.agentSettingsLoaded) return '正在读取 Agent 任务配置';
    if (!state.taskApi) return '正在连接任务接口；若一直未就绪，请重启 Agent';
    if (!state.services?.vln?.online) return 'LightNav 未就绪，请等待模型窗口显示 READY';
    if (state.active) return '任务正在执行，可查看下方实时进展';
    if (state.data?.control?.source !== 'idle' && !state.data?.control?.manual)
      return '机器人正在被其他控制操作使用';
    return '';
  }
  // 根据最新快照更新画面和控制状态
  function render() {
    const data = state.data || {},
      sim = data.simulation || {},
      control = data.control || {},
      vln = data.vln || {};
    setService('service-sim', state.connected, state.connected ? '在线' : '离线');
    setService(
      'service-agent',
      !!state.services?.agent?.online,
      state.services?.agent?.online
        ? state.services.agent.configured
          ? '在线'
          : '待配置'
        : '离线',
      state.services?.agent?.online && !state.services.agent.configured,
    );
    setService(
      'service-vln',
      !!state.services?.vln?.online,
      state.services?.vln?.online ? '就绪' : '未就绪',
    );
    const problem = readiness();
    $('start-task').disabled = !!problem || state.busy;
    $('start-task').textContent = state.busy
      ? '正在处理…'
      : state.active
        ? '探索进行中'
        : '↗ 开始探索';
    $('start-hint').textContent = problem || '准备就绪 · 点击后开始观察与自主探索';
    const alert = !state.connected
      ? '仿真连接已断开，正在自动重连。'
      : !state.services?.agent?.online
        ? 'Agent 服务未连接。手动驾驶仍可使用，请启动 Agent 服务。'
        : !state.services.agent.configured
          ? '百炼 Key 尚未配置。请填写 deploy/.env 后重启 Agent，即可开始探索。'
          : !state.taskApi
            ? '任务接口尚未就绪。若持续出现，请重启 Agent 服务。'
            : !state.services.vln?.online
              ? 'LightNav 尚未就绪，等待模型加载后即可开始导航。'
              : '';
    $('connection-alert').hidden = !alert;
    $('connection-alert').textContent = alert;
    $('goal').disabled = !!state.active || state.busy;
    $('max-steps').disabled = $('max-seconds').disabled = !!state.active || state.busy;
    $('navigation-time-limit-enabled').disabled = !!state.active || state.busy;
    $('navigation-seconds').disabled = !state.navigationTimeLimitEnabled || !!state.active || state.busy;
    document
      .querySelectorAll('[data-goal]')
      .forEach((button) => (button.disabled = !!state.active || state.busy));
    $('emergency').disabled = !state.connected && !state.active;
    $('takeover').textContent = control.manual
      ? '释放控制'
      : state.active
        ? '停止并接管'
        : '手动接管';
    $('takeover').disabled = !state.connected || state.busy || (control.locked && !state.active);
    $('reset').disabled = !state.connected || state.busy;
    let directBlocked = '';
    if (!state.connected) directBlocked = '仿真连接已断开，重连后可执行。';
    else if (state.active) directBlocked = '请先结束当前 Agent 任务。';
    else if (data.vln_run) directBlocked = data.vln_run.status === 'running'
      ? 'LightNav 正在执行导航。' : '正在保存日志和录像，请稍候。';
    else if (control.locked) directBlocked = '其他控制端正在使用机器人。';
    else if (!state.services?.vln?.online) directBlocked = 'LightNav 尚未就绪。';
    else if (state.busy) directBlocked = '正在处理上一操作。';
    $('direct-start').disabled = !!directBlocked;
    $('direct-start').title = directBlocked;
    $('direct-instruction').disabled = !!state.active || !!data.vln_run;
    $('direct-seconds').disabled = $('direct-instruction').disabled;
    $('manual-hint').textContent = control.manual
      ? '已接管 · WASD / 方向键控制，松开即停'
      : state.active
        ? 'Agent 正在控制，接管会结束任务'
        : control.locked
          ? '其他控制端正在使用机器人'
          : '接管后使用 WASD 或方向按钮';
    document.querySelectorAll('[data-key]').forEach((button) => {
      button.disabled = !control.manual || !state.connected;
      button.classList.toggle('pressed', state.keys.has(button.dataset.key));
    });
    $('metric-speed').textContent = state.connected
      ? Number(sim.velocity?.linear || 0).toFixed(2)
      : '—';
    $('metric-yaw').textContent = state.connected
      ? String(Math.round(((Number(sim.pose?.yaw || 0) * 180) / Math.PI + 360) % 360))
      : '—';
    $('metric-fps').textContent = sim.camera?.ready ? Number(sim.camera.fps).toFixed(1) : '—';
    $('metric-latency').textContent =
      vln.latency_ms != null ? String(Math.round(vln.latency_ms)) : '—';
    $('clock').textContent = duration(sim.sim_time);
    $('scene').textContent = data.scene ? `MuJoCo · ${data.scene.name}` : 'MuJoCo · 室内仿真';
    $('motion-state').textContent = !state.connected
      ? '离线'
      : sim.backend?.blocked
        ? '前方受阻'
        : Math.abs(sim.velocity?.linear || 0) > 0.01
          ? '移动中'
          : Math.abs(sim.velocity?.angular || 0) > 0.01
            ? '转向中'
            : '已停止';
    $('direct-status').textContent =
      directBlocked ||
      vln.error ||
      data.mpc?.error ||
      (vln.stop === true
        ? '导航模型请求停止，请查看画面确认结果。'
        : control.auto
          ? 'LightNav 正在执行导航。'
          : '此模式直接调用导航模型，不使用上层 Agent。');
    $('snapshot').disabled = !state.cameraReady;
    renderTask();
  }
  function selectedRecord() {
    return records().find((task) => task.id === state.selected) || null;
  }
  function renderTask() {
    const task = selectedRecord();
    $('task-empty').hidden = !!task;
    $('task-progress').hidden = !task;
    $('task-badge').textContent = task ? labels[task.status] || task.status : '待命';
    $('task-badge').className = 'status-badge ' + (task?.status || '');
    $('export').disabled = !task;
    // 当前任务和历史任务共用报告入口，所有资源都通过同源代理读取。
    const replayLink = $('task-replay-link');
    replayLink.hidden = !task;
    if (task) replayLink.href = `/replay/${task.id}`;
    else replayLink.removeAttribute('href');
    const logLink = $('task-log-link');
    const memoryLink = $('task-memory-link');
    logLink.hidden = !task;
    logLink.textContent = task?.mode === 'vln' ? '运行日志 ↗' : '任务摘要 ↗';
    memoryLink.hidden = !task?.spatial_memory_database;
    if (task) {
      const reportBase = logBase(task);
      logLink.href = reportBase + 'index.html';
      memoryLink.href = reportBase + 'memory_visualize.html';
    } else {
      logLink.removeAttribute('href');
      memoryLink.removeAttribute('href');
    }
    $('return-current').hidden = !state.active || state.active.id === state.selected;
    $('viewing-label').textContent = task
      ? task.status === 'running'
        ? '实时更新'
        : '历史记录'
      : '实时更新';
    if (!task) return;
    const isVln = task.mode === 'vln';
    $('task-meter').hidden = isVln;
    document.querySelector('.phases').hidden = isVln;
    $('cancel-task').textContent = isVln ? '结束当前导航' : '结束当前任务';
    $('current-goal').textContent = task.goal;
    if (isVln) {
      $('step-label').textContent = '预测 ' + task.prediction_count + ' 次';
      const end = task.status === 'running' ? Date.now() : Date.parse(task.ended_at);
      $('elapsed').textContent = duration((end - Date.parse(task.started_at)) / 1000);
      $('decision-label').textContent = 'VLN 执行状态';
      $('decision-reason').textContent = task.summary;
      $('decision-instruction').textContent = '导航指令：' + task.instruction;
      $('result-summary').hidden = true;
      $('task-model').textContent = 'LightNav · 任务成功与否未评定' +
        (task.log_error ? ' · 日志异常：' + task.log_error : '') +
        (task.dropped_log_entries ? ' · 部分预测日志未保存' : '');
      $('cancel-task').hidden = task.status !== 'running';
      $('cancel-task').disabled = state.busy;
      return;
    }
    $('step-label').textContent = `第 ${task.step || 0} / ${task.max_steps} 轮`;
    $('task-meter').value = task.step || 0;
    $('task-meter').max = task.max_steps;
    const end =
      task.status === 'running' ? Date.now() : Date.parse(task.ended_at || task.started_at);
    $('elapsed').textContent = task.started_at
      ? duration((end - Date.parse(task.started_at)) / 1000)
      : '—';
    document
      .querySelectorAll('[data-phase]')
      .forEach((e) => e.classList.toggle('active', task.phase === e.dataset.phase));
    const decision = task.last_decision;
    $('decision-label').textContent =
      task.status === 'running'
        ? phases[task.phase] || '运行中'
        : decision
          ? actions[decision.action] || '最终决策'
          : '最终状态';
    $('decision-reason').textContent =
      decision?.reason ||
      (task.status === 'running' ? '正在连接机器人，准备获取第一帧画面。' : friendly(task.summary));
    $('decision-instruction').textContent = decision?.instruction
      ? `导航指令：${decision.instruction}`
      : decision?.action === 'look'
        ? `环视方向：${decision.direction === 'left' ? '左侧' : '右侧'}`
        : '';
    $('result-summary').hidden = task.status === 'running' || !task.summary;
    $('result-summary').textContent = friendly(task.summary);
    $('task-model').textContent = task.model
      ? `${task.model}${task.success_basis === 'model_visual_judgment' ? ' · 视觉模型判断' : ''}`
      : '';
    $('cancel-task').hidden = task.status !== 'running';
    $('cancel-task').disabled = state.busy;
  }
  // 切换历史任务时，重置事件游标
  async function selectTask(id) {
    if (state.selected === id) return;
    state.selected = id;
    state.events = [];
    state.corrections.clear();
    logFrames.clear(); detectionFrames.clear(); currentLogFrame=null; detectorWaiting=false;
    state.completionSteps.clear();
    state.cursor = 0;
    state.eventTask = id;
    $('events').replaceChildren();
    $('events-empty').hidden = false;
    $('event-count').textContent = '0';
    renderTask();
    await refreshEvents();
  }
  // 增量读取选中任务的日志
  async function refreshEvents() {
    const id = state.selected;
    const task = selectedRecord();
    if (!task || (task.mode !== 'vln' && !state.taskApi)) return;
    const cursor = state.cursor;
    const result = await readEvents(task, cursor);
    if (state.selected !== id || state.cursor !== cursor) return;
    for (const entry of result.events) {
      state.events.push(entry);
      appendEvent(entry, id);
    }
    state.cursor = result.next;
    $('event-count').textContent = String($('events').children.length);
    $('events-empty').hidden = !!$('events').children.length;
  }


  const logFrames = new Map(), detectionFrames = new Map();
  let currentLogFrame = null, detectorWaiting = false;
  function evidenceFrames(entry) {
    const safe = path => typeof path === 'string' && /^images\/(?:detection-)?[0-9]{3,}\.jpg$/.test(path);
    const make = (image,caption) => safe(image) ? {image,caption} : null;
    if (['observation','passive_observation','decision_input'].includes(entry.type)) {
      let frame=make(entry.image,'决策时画面');
      if (!frame && entry.type==='decision_input') {
        const current=(entry.images || []).find(item => (item.uses || []).some(use => use.purpose==='当前图像 A'));
        frame=make(current?.image,'决策时画面');
      }
      if (frame) {
        logFrames.set(entry.step,frame);
        if (entry.type!=='passive_observation') currentLogFrame=frame;
      }
    }
    if (entry.type==='target_detection_hit') {
      const frame=make(entry.boxed_image || entry.image,'检测触发画面');
      if (frame) detectionFrames.set(entry.candidate_id,frame);
      return frame ? [frame] : [];
    }
    if (entry.type==='decision') {
      const frame=logFrames.get(entry.step);
      return frame ? [frame] : [];
    }
    if (entry.type==='target_detection_review') {
      const frames=[], detected=detectionFrames.get(entry.candidate_id);
      if (detected) frames.push(detected);
      if (currentLogFrame && currentLogFrame.image!==detected?.image)
        frames.push({...currentLogFrame,caption:'停车复核画面'});
      return frames;
    }
    if (['completion_check','finished'].includes(entry.type) && currentLogFrame)
      return [{...currentLogFrame,caption:entry.type==='finished'?'最后记录的观察':'核验时画面'}];
    return [];
  }

  function keyEvent(entry) {
    if (selectedRecord()?.mode === 'vln')
      return entry.key ? {label: entry.label, description: entry.summary} : null;
    const actionNames = {look:'环视观察',orient:'转回确认',navigate:'执行导航',recover:'退回脱困',finish:'确认完成',give_up:'结束探索'};
    const statusNames = {running:'执行中',completed:'已完成',failed:'未完成',cancelled:'已取消',budget_exhausted:'达到执行上限',interrupted:'已中断'};
    const resultNames = {blocked:'前方受阻，已停止导航',navigation_error:'导航服务异常',timeout:'导航超时',
      agent_interrupt:'新观察触发重规划，已停车待复核',path_repeated:'导航绕回旧位置，停车复核',
      no_progress:'原地无进展，停止并准备脱困',recovered:'已沿原路径退回',
      recovery_unavailable:'缺少连续退路或碰撞保护',recovery_blocked:'退回途中受阻，已停止',
      recovery_timeout:'脱困超时，已停止',recovery_no_progress:'脱困未产生足够位移'};
    const d = entry.decision || {}, r = entry.result || {};
    let label = '', description = '';
    switch (entry.type) {
      case 'started': label = '任务开始'; description = entry.goal; break;
      case 'decision':
        if (['finish','give_up'].includes(d.action)) return null;
        label = actionNames[d.action] || '选择下一步'; description = d.reason; break;
      case 'finished': label = statusNames[entry.status] || '任务结束'; description = entry.summary; break;
      case 'action_result':
        if (['turned','oriented','model_stop','agent_interrupt','completed','success',null,undefined].includes(r.reason) && !r.startup_failed) return null;
        label = '动作反馈'; description = r.startup_failed ? '导航服务启动失败，未执行接近' : resultNames[r.reason] || r.reason; break;
      case 'completion_check':
        if (entry.confirmed) return null;
        label = entry.confirmed ? '完成核验通过' : '尚未满足完成条件';
        description = entry.evidence || entry.error || '继续观察或接近。'; break;
      case 'decision_error':
        if (entry.category === 'completion_verification') return {label:'尚未满足完成条件',description:'本次完成判断未通过核验，后续处理见下一节点。'};
        label = '调整下一步';
        description = ['scene_rule','exploration_rule'].includes(entry.category) ?
          '动作未通过检查，未执行；后续处理见下一节点。' : '模型输出不完整，本次动作未执行；后续处理见下一节点。'; break;
      case 'decision_replan': label = '重新观察'; description = '本轮未形成可执行动作，重新观察后再决定。'; break;
      case 'target_detection_hit': label = '检测到目标线索'; description = '已请求停车，等待千问核实目标。'; break;
      case 'target_detection_review':
        label = {confirmed:'已确认目标',rejected:'排除误检',uncertain:'目标尚待确认'}[entry.verdict] || '复核目标线索';
        description = entry.evidence || '根据停车画面重新判断。'; break;
      case 'target_watch_degraded':
        if (['TimeoutError','ServerTimeoutError'].includes(entry.reason)) {
          label='本次目标检测超时'; description='没有及时收到检测结果；继续导航，等待后续检测。';
        } else if (entry.reason==='unsupported') {
          label='当前类别不支持检测'; description='继续使用视觉模型观察。';
        } else if (['ClientConnectorError','ClientConnectionError'].includes(entry.reason)) {
          label='无法连接目标检测服务'; description='本次未获得检测结果，继续原有观察流程。';
        } else {
          label='本次目标检测失败'; description='本次未获得检测结果，继续原有观察流程。';
        }
        break;
      case 'navigation_interrupt_requested':
        if (!entry.accepted) return null;
        label = '停车复核';
        description = {target_seen:'行进中发现目标线索。',subgoal_reached:'当前局部目标可能已到达。',route_invalid:'当前路线需要重新判断。'}[entry.reason] || '行进中有新线索，停车后重新判断。'; break;
      default: return null;
    }
    description = String(description || '').replace(/\s+/g, ' ').trim();
    return {label,description};
  }

  function appendEvent(entry, id) {
    if (entry.type === 'action_input' && entry.action === 'navigate' && entry.instruction) {
      const row = [...$('events').children].find(item => item.dataset.kind === 'decision' && item.dataset.step === String(entry.step));
      if (row) {
        const instruction = node('p', 'instruction', 'VLN 指令：' + entry.instruction);
        row.querySelector('.event-content > p').after(instruction);
      }
      return;
    }
    const frames = evidenceFrames(entry);
    let summary = keyEvent(entry);
    if (entry.type==='target_detection_frame' && detectorWaiting) {
      detectorWaiting=false; summary={label:'目标检测已恢复',description:'已重新收到有效检测结果。'};
    }
    if (entry.type==='target_watch_degraded') detectorWaiting=true;
    if (!summary) return;
    if (entry.type === 'completion_check') state.completionSteps.add(entry.step);
    if (entry.category === 'completion_verification' && state.completionSteps.has(entry.step)) return;
    if (entry.type === 'target_watch_degraded' && $('events').lastElementChild?.dataset.kind === entry.type) return;
    const signature = entry.type === 'decision_error' && entry.step != null ?
      String(entry.step) + ':' + entry.category : null;
    if (signature && state.corrections.has(signature)) return;
    if (signature) state.corrections.add(signature);
    const item = node('li');
    item.dataset.kind = entry.type;
    item.dataset.step = String(entry.step);
    const time = node('span', 'event-time', entry.time ?
      new Date(entry.time).toLocaleTimeString('zh-CN', {hour12:false}) : '—');
    const content = node('div', 'event-content'), head = node('div', 'event-head');
    head.append(node('b', '', summary.label));
    content.append(head, node('p', '', summary.description));
    const gallery=node('div','event-frames');
    for (const frame of frames) {
      const button=node('button','event-frame'), image=node('img');
      image.src=logBase(selectedRecord())+frame.image;
      image.alt=frame.caption; image.loading='lazy';
      button.append(image,node('span','',frame.caption+' · 点击放大'));
      button.onclick=()=> {
        $('observation-image').src=image.src;
        $('observation-caption').textContent=frame.caption;
        $('image-dialog').showModal();
      };
      gallery.append(button);
    }
    if (frames.length) content.append(gallery);
    item.append(time, node('span', 'event-symbol', '·'), content);
    $('events').append(item);
  }
  function renderHistory() {
    const all = records();
    const filter = $('history-mode').value;
    const visible = all.filter(task => filter === 'all' || (task.mode || 'agent') === filter);
    $('history-count').textContent = String(all.length);
    $('history-empty').hidden = !!visible.length;
    const signature = JSON.stringify([filter, visible.map(t => [t.id, t.status, t.step, t.prediction_count])]);
    if (renderHistory.last === signature) return;
    renderHistory.last = signature;
    const fragment = document.createDocumentFragment();
    visible.forEach((task) => {
      const button = node('button', 'history-item'),
        text = node('div');
      text.append(
        node('strong', '', (task.mode === 'vln' ? 'VLN · ' : 'Agent · ') + task.goal),
        node(
          'small',
          '',
          `${date(task.started_at)} · ${task.mode === 'vln' ? (task.prediction_count + ' 次预测') : ((task.step || 0) + ' 轮')}${task.model ? ' · ' + task.model : ''}`,
        ),
      );
      button.append(
        text,
        node('span', 'status-badge ' + task.status, labels[task.status] || task.status),
        node('span', 'arrow', '↗'),
      );
      button.onclick = () => {
        selectTask(task.id).catch((e) => toast(e.message, true));
        showTab('activity');
      };
      fragment.append(button);
    });
    $('history-list').replaceChildren(fragment);
  }
  function showTab(name) {
    for (const tab of ['activity', 'history']) {
      $('tab-' + tab).classList.toggle('selected', tab === name);
      $('tab-' + tab).setAttribute('aria-selected', String(tab === name));
      $(tab + '-view').hidden = tab !== name;
    }
  }
  // 轮询服务与任务状态，避免并发重复请求
  async function poll() {
    if (state.polling) return;
    state.polling = true;
    try {
      const response = await fetch('/api/services', {
        signal: AbortSignal.timeout(5000),
        cache: 'no-store',
      });
      if (!response.ok) throw new Error();
      state.services = await response.json();
      if (state.services.agent.online) {
        try {
          const data = await api('tasks?limit=50');
          state.tasks = data.tasks;
          state.taskApi = true;
          state.active = state.tasks.find((t) => t.status === 'running') || null;
          renderHistory();
          if (!state.selected && state.active) await selectTask(state.active.id);
          else await refreshEvents();
        } catch {
          state.taskApi = false;
        }
      } else state.taskApi = false;
    } catch {
      state.services = null;
      state.taskApi = false;
    } finally {
      try {
        const data = await vlnApi('');
        state.vlnTasks = data.tasks;
        const active = state.vlnTasks.find(task => task.status === 'running');
        renderHistory();
        if (active && active.id !== state.vlnAutoSelected) {
          state.vlnAutoSelected = active.id;
          await selectTask(active.id);
        } else if (selectedRecord()?.mode === 'vln') await refreshEvents();
      } catch (error) {
        if (selectedRecord()?.mode === 'vln') $('viewing-label').textContent = error.message;
      }
      render();
      state.polling = false;
      setTimeout(poll, 1500);
    }
  }
  async function operation(fn) {
    if (state.busy) return;
    state.busy = true;
    render();
    try {
      await fn();
    } catch (e) {
      toast(e.message, true);
    } finally {
      state.busy = false;
      render();
    }
  }
  // 先结束自动任务，再允许手动接管
  async function cancelActive() {
    if (selectedRecord()?.mode === 'vln' && selectedRecord().status === 'running') {
      send({type: 'set_vln', enabled: false});
      return;
    }
    if (!state.active) return;
    const record = await api(`tasks/${state.active.id}/cancel`, 'POST');
    state.tasks = state.tasks.map((t) => (t.id === record.id ? record : t));
    state.active = null;
    renderHistory();
    render();
  }
  async function confirmAction(title, text) {
    $('confirm-title').textContent = title;
    $('confirm-text').textContent = text;
    $('confirm-dialog').returnValue = '';
    $('confirm-dialog').showModal();
    return new Promise((resolve) => {
      $('confirm-yes').onclick = () => {
        $('confirm-dialog').close('yes');
      };
      $('confirm-no').onclick = () => {
        $('confirm-dialog').close('no');
      };
      $('confirm-dialog').addEventListener(
        'close',
        () => resolve($('confirm-dialog').returnValue === 'yes'),
        { once: true },
      );
    });
  }
  $('task-form').onsubmit = (event) => {
    event.preventDefault();
    if (!state.agentSettingsLoaded || !$('task-form').reportValidity()) return;
    operation(async () => {
      const epoch = state.stopEpoch;
      const goal = $('goal').value.trim();
      if (!goal) throw new Error('请先输入一个目标。');
      if (state.data?.control?.manual) {
        send({ type: 'release_control' });
        await new Promise((resolve) => setTimeout(resolve, 150));
      }
      const task = await api('tasks', 'POST', {
        goal,
        max_steps: Number($('max-steps').value),
        max_seconds: Number($('max-seconds').value),
        navigation_time_limit_enabled: state.navigationTimeLimitEnabled,
        ...(state.navigationTimeLimitEnabled
          ? { navigation_seconds: Number($('navigation-seconds').value) }
          : {}),
      });
      if (epoch !== state.stopEpoch) {
        await api(`tasks/${task.id}/cancel`, 'POST');
        toast('启动请求已取消。');
        return;
      }
      state.tasks = [task, ...state.tasks];
      state.active = task;
      await selectTask(task.id);
      showTab('activity');
      renderHistory();
      toast('探索已开始，可以在现场视图中观察机器人。');
    });
  };
  $('cancel-task').onclick = () =>
    operation(async () => {
      await cancelActive();
      toast('任务已结束，机器人已停止。');
    });
  $('emergency').onclick = () => {
    state.stopEpoch++;
    state.keys.clear();
    try {
      send({ type: 'stop' });
    } catch (e) {
      toast(e.message, true);
    }
    (async () => {
      try {
        const data = await api('tasks?limit=50');
        for (const task of data.tasks.filter((t) => t.status === 'running'))
          await api(`tasks/${task.id}/cancel`, 'POST');
        state.active = null;
        toast('已请求停止全部运动。');
      } catch (e) {
        toast('已发送停车指令；' + e.message, true);
      } finally {
        render();
      }
    })();
  };
  $('takeover').onclick = () =>
    operation(async () => {
      state.keys.clear();
      if (state.data?.control?.manual) {
        send({ type: 'release_control' });
        return;
      }
      if (state.active) {
        if (
          !(await confirmAction(
            '停止任务并接管？',
            '当前探索会结束，随后你可以使用方向按钮或 WASD 驾驶。',
          ))
        )
          return;
        await cancelActive();
      }
      send({ type: 'acquire_control' });
    });
  $('reset').onclick = () =>
    operation(async () => {
      if (
        !(await confirmAction(
          '将机器人放回起点？',
          '当前动作会停止，正在运行的 Agent 任务也会结束。历史执行记录会保留。',
        ))
      )
        return;
      state.keys.clear();
      send({ type: 'stop' });
      await cancelActive();
      send({ type: 'reset' });
      toast('已请求重置场景。');
    });
  $('direct-form').onsubmit = (event) => {
    event.preventDefault();
    operation(async () => {
      const instruction = $('direct-instruction').value.trim();
      if (!instruction) throw new Error('请输入导航指令。');
      if (state.active) throw new Error('请先结束 Agent 任务。');
      state.keys.clear();
      if (!$('direct-form').reportValidity()) return;
      send({ type: 'set_vln', enabled: true, instruction, record_run: true,
        max_seconds: $('direct-seconds').value === '' ? null : Number($('direct-seconds').value) });
    });
  };
  // 把当前按键合成为速度；松开后发送停车
  function publishManual() {
    if (state.connected && state.data?.control?.manual) {
      try {
        send({
          type: 'twist',
          linear: ((state.keys.has('w') ? 1 : 0) - (state.keys.has('s') ? 1 : 0)) * 0.5,
          angular: ((state.keys.has('a') ? 1 : 0) - (state.keys.has('d') ? 1 : 0)) * 1,
        });
      } catch {}
    }
  }
  function key(value, pressed) {
    if (!['w', 'a', 's', 'd'].includes(value)) return;
    if (pressed && (!state.data?.control?.manual || !state.connected)) return;
    pressed ? state.keys.add(value) : state.keys.delete(value);
    publishManual();
    render();
  }
  document.querySelectorAll('[data-key]').forEach((button) => {
    button.onpointerdown = (e) => {
      e.preventDefault();
      button.setPointerCapture(e.pointerId);
      key(button.dataset.key, true);
    };
    button.onpointerup = button.onpointercancel = () => key(button.dataset.key, false);
  });
  window.addEventListener('keydown', (e) => {
    if (
      ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName) ||
      document.querySelector('dialog[open]')
    )
      return;
    if (e.code === 'Space') {
      e.preventDefault();
      if (!e.repeat) $('emergency').click();
      return;
    }
    const k = e.key.toLowerCase();
    if (['w', 'a', 's', 'd'].includes(k)) {
      e.preventDefault();
      key(k, true);
    }
  });
  window.addEventListener('keyup', (e) => key(e.key.toLowerCase(), false));
  const releaseKeys = () => {
    state.keys.clear();
    publishManual();
    render();
  };
  window.addEventListener('blur', releaseKeys);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) releaseKeys();
  });
  setInterval(publishManual, 100);
  // 持续刷新相机，替换后释放旧图片资源
  async function cameraLoop() {
    const view = state.view;
    await Promise.all(
      ['main', 'secondary'].map(async (name) => {
        const first = (name === 'main') === (view === 'first');
        const img = $(name === 'main' ? 'camera' : 'camera-secondary');
        let next;
        try {
          const response = await fetch(
            (first ? '/api/camera.jpg' : '/api/third-person.jpg') + '?t=' + Date.now(),
            { cache: 'no-store', signal: AbortSignal.timeout(5000) },
          );
          if (!response.ok) throw new Error();
          next = URL.createObjectURL(await response.blob());
          if (state.view !== view) return;
          await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
            img.src = next;
          });
          const old = state.urls[name];
          state.urls[name] = next;
          next = null;
          if (old) URL.revokeObjectURL(old);
          if (name === 'main') {
            state.cameraReady = true;
            $('camera-empty').hidden = true;
          }
        } catch {
          if (name === 'main') {
            state.cameraReady = false;
            $('camera-empty').hidden = false;
          }
        } finally {
          if (next) URL.revokeObjectURL(next);
        }
      }),
    );
    setTimeout(cameraLoop, 120);
  }
  function switchView(view) {
    state.view = view;
    $('view-first').classList.toggle('selected', view === 'first');
    $('view-third').classList.toggle('selected', view === 'third');
    $('camera-label').textContent = view === 'first' ? '机器人第一视角' : '机器人第三视角';
    $('camera').alt = view === 'first' ? '机器人实时第一视角画面' : '机器人实时第三视角画面';
    $('camera-secondary').alt = view === 'first' ? '辅助第三视角画面' : '辅助第一视角画面';
    $('pip-label').textContent = view === 'first' ? '第三视角 ↗' : '第一视角 ↗';
    $('camera-caption').textContent =
      view === 'first'
        ? '当前第一视角也会提供给 Agent 进行观察'
        : '第三视角仅供观察，Agent 仍使用第一视角';
  }
  $('view-first').onclick = () => switchView('first');
  $('view-third').onclick = () => switchView('third');
  $('pip').onclick = () => switchView(state.view === 'first' ? 'third' : 'first');
  $('fullscreen').onclick = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await $('camera-stage').requestFullscreen();
    } catch {
      toast('当前浏览器不支持全屏显示。', true);
    }
  };
  function download(url, name) {
    const link = node('a');
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
  }
  $('snapshot').onclick = () => {
    if (state.urls.main) download(state.urls.main, `nav-${state.view}-${Date.now()}.jpg`);
  };
  $('export').onclick = () =>
    operation(async () => {
      const task = selectedRecord();
      if (!task) return;
      let cursor = 0,
        events = [];
      while (true) {
        const part = await readEvents(task, cursor);
        events.push(...part.events);
        if (part.next === cursor || part.events.length < 200) break;
        cursor = part.next;
      }
      const url = URL.createObjectURL(
        new Blob([JSON.stringify({ task, events }, null, 2)], { type: 'application/json' }),
      );
      download(url, `nav-task-${task.id.slice(0, 8)}.json`);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
  $('history-mode').onchange = renderHistory;
  $('tab-activity').onclick = () => showTab('activity');
  $('tab-history').onclick = () => showTab('history');
  $('return-current').onclick = () => {
    if (state.active) selectTask(state.active.id).catch((e) => toast(e.message, true));
  };
  $('close-image').onclick = () => $('image-dialog').close();
  document.querySelectorAll('[data-goal]').forEach(
    (button) =>
      (button.onclick = () => {
        $('goal').value = button.dataset.goal;
        $('goal').dispatchEvent(new Event('input'));
        $('goal').focus();
      }),
  );
  try {
    $('goal').value = localStorage.getItem('nav-agent-goal') || '';
  } catch {}
  $('goal').oninput = () => {
    try {
      localStorage.setItem('nav-agent-goal', $('goal').value);
    } catch {}
  };
  const budget = () => {
    $('budget-summary').textContent =
      `${$('max-steps').value} 轮 · ${Math.round((Number($('max-seconds').value) / 60) * 10) / 10} 分钟`;
  };
  $('max-steps').oninput = $('max-seconds').oninput = budget;
  connect();
  poll();
  cameraLoop();
  render();

  function updateNavigationTimeLimit() {
    state.navigationTimeLimitEnabled = $('navigation-time-limit-enabled').checked;
    $('navigation-limit').hidden = !state.navigationTimeLimitEnabled;
    $('navigation-seconds').disabled = !state.navigationTimeLimitEnabled;
    $('navigation-unlimited').hidden = state.navigationTimeLimitEnabled;
  }
  $('navigation-time-limit-enabled').checked = false;
  $('navigation-time-limit-enabled').addEventListener('change', updateNavigationTimeLimit);
  updateNavigationTimeLimit();

  async function loadAgentSettings() {
    try {
      const config = await api('settings/agent');
      for (const [id, value] of [['max-steps', config.ui_task_defaults.max_steps],
                                ['max-seconds', config.ui_task_defaults.max_seconds]]) {
        if (!Number.isFinite(value)) throw new Error('Agent 配置返回值无效');
        $(id).value = value;
      }
      $('max-seconds').min = config.task_limits.min_seconds;
      $('max-seconds').max = config.task_limits.max_seconds;
      // 秒数只作为开启后的初始值，不用配置覆盖页面开关。
      if (!Number.isFinite(config.navigation_seconds))
        throw new Error('Agent 导航时限返回值无效');
      $('navigation-seconds').value = config.navigation_seconds;
      $('navigation-seconds').min = config.task_limits.min_navigation_seconds;
      $('navigation-seconds').max = config.task_limits.max_navigation_seconds;
      updateNavigationTimeLimit();
      state.agentSettingsLoaded = true;
      budget();
      render();

    } catch (error) {
      toast('正在等待 Agent 任务配置：' + error.message, true);
      setTimeout(loadAgentSettings, 2000);
    }
  }
  loadAgentSettings();

})();
