(() => {
  const $ = (id) => document.getElementById(id);
  const labels = {running:'执行中', failed:'失败', cancelled:'已取消', interrupted:'已中断',
    completed:'已完成', budget_exhausted:'达到任务预算', look:'环视', orient:'转回确认', navigate:'导航', recover:'恢复',
    finish:'完成', give_up:'放弃', left:'向左', right:'向右'};
  let taskId = '', cursor = 0, cameraOK = false, freshCamera = false, paused = false;
  let selectionVersion = 0, activeId = '', posting = false, online = false;
  const corrections = new Set(), completionSteps = new Set();
  const taskDraft = new Set();
  let navigationTimeLimitEnabled = false;
  let configuredNavigationTimeLimitEnabled = false;
  const canvas = $('camera'), context = canvas.getContext('2d');
  function text(id, value) { if ($(id).textContent !== value) $(id).textContent = value; }
  async function request(path, body) {
    const response = await fetch(path, {cache:'no-store', signal:AbortSignal.timeout(body === undefined ? 5000 : 15000),
      ...(body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)})});
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(typeof detail.detail === 'string' ? detail.detail : Array.isArray(detail.detail) ? detail.detail.map(e=>`${e.loc?.slice(1).join('.')}: ${e.msg}`).join('；') : `HTTP ${response.status}`);
    }
    return response;
  }
  function buttons() {
    $('start').disabled = posting || !online || !!activeId;
    $('cancel-task').disabled = posting || !activeId;
    settingButtons();
  }
  function updateCamera() {
    const visible = cameraOK && freshCamera;
    canvas.classList.toggle('stale', !visible);
    $('camera-message').hidden = visible;
    text('camera-message', '相机未连接或画面已过期，等待新画面…');
  }
  async function cameraLoop() {
    try {
      if (!document.hidden && !paused) {
        const response = await request('/monitor/camera.jpg');
        const bitmap = await createImageBitmap(await response.blob());
        try {
          if (!paused && !document.hidden) {
            // 固定画布，解码完成后再绘制，不替换 img src、不清空上一帧。
            const scale = Math.min(canvas.width / bitmap.width, canvas.height / bitmap.height);
            const width = bitmap.width * scale, height = bitmap.height * scale;
            context.fillStyle = '#080e16'; context.fillRect(0, 0, canvas.width, canvas.height);
            context.drawImage(bitmap, (canvas.width-width)/2, (canvas.height-height)/2, width, height);
            cameraOK = true;
            text('camera-time', `实时画面 · ${new Date().toLocaleTimeString()}`);
          }
        } finally { bitmap.close(); }
      }
    } catch { cameraOK = false; }
    if (!paused) updateCamera();
    setTimeout(cameraLoop, 200);
  }
  async function robotLoop() {
    try {
      if (!document.hidden && !paused) {
        const r = await (await request('/monitor/robot')).json();
        if (!paused) {
          freshCamera = r.camera?.received === true && Number.isFinite(r.camera?.age_ms) && r.camera.age_ms < 1000;
          const yaw = r.odom?.pose_valid ? `${(r.odom.pose.yaw * 180 / Math.PI).toFixed(1)}°` : '未知';
          text('robot', `本体 ${r.robot?.connected ? '在线' : '离线'} · 模式 ${r.robot?.mode || '未知'} · 朝向 ${yaw} · 相机 ${Number(r.camera?.fps || 0).toFixed(1)} fps · 控制来源 ${r.control_source || '未知'}`);
          showTemperatures(r.robot); showActual(r);
          const checks = {verified:'通过', failed:'失败', not_checked:'尚未检查'};
          const limit = Number.isFinite(r.robot?.linear_limit) ? `${r.robot.linear_limit} m/s` : '未上报';
          text('safety', `最近接管时避障检查：${checks[r.robot?.avoidance_check] || '未上报'} · 本体限速：${limit} · 自定义安全距离：${r.robot?.custom_clearance_available === true ? '已启用' : '未启用'}${r.robot?.safety_error ? '\n' + r.robot.safety_error : ''}`);
        }
      }
    } catch { freshCamera = false; if (!paused) { text('robot', '本体连接失败，正在重试'); showTemperatures(null,true); showActual(null); } }
    if (!paused) updateCamera();
    setTimeout(robotLoop, 1000);
  }
  function useTask(id) {
    if (taskId === id) return;
    taskId = id; cursor = 0; logFrames.clear(); detectionFrames.clear(); currentLogFrame=null; detectorWaiting=false; corrections.clear(); completionSteps.clear(); $('events').replaceChildren();
  }
  function showTask(record) {
    text('goal', record?.goal || '暂无任务');
    text('status', record ? `${labels[record.status] || record.status} · 第 ${record.step} / ${record.max_steps} 步 · 总预算 ${record.max_seconds} 秒 · ${record.phase}` : '');
    text('summary', record?.summary || '');
    const decision = record?.last_decision;
    text('decision', decision ? [labels[decision.action] || decision.action, labels[decision.direction] || '', decision.instruction, decision.reason].filter(Boolean).join('\n') : '等待模型决策');
    $('record-link').hidden = !record;
    if (record) $('record-link').href = `/tasks/${record.id}/logs/index.html`;
    $('replay-link').hidden = !record;
    if (record) $('replay-link').href = `/replay/${record.id}`;
    const observation = record?.latest_image ? `/tasks/${record.id}/images/${record.latest_image}` : '';
    $('observation').hidden = !observation;
    if (observation && $('observation').getAttribute('src') !== observation) $('observation').src = observation;
  }
  function updateOptions(tasks) {
    // 保留已有 option 节点及用户选择，避免关闭正在操作的下拉菜单。
    const select = $('task-select');
    for (const t of tasks) {
      const label = `${new Date(t.started_at).toLocaleString()} · ${labels[t.status] || t.status} · ${t.goal}`;
      let option = Array.from(select.options).find(o => o.value === t.id);
      if (!option) { option = new Option(label, t.id); select.append(option); }
      else if (option.textContent !== label) option.textContent = label;
    }
    for (const option of Array.from(select.options)) {
      if (option.value && option.value !== select.value && !tasks.some(t => t.id === option.value)) option.remove();
    }
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
        if (r.motion_limit_unavailable) return {label:'位移限制未生效',description:'此段导航没有执行指定的位移上限。'};
        if (r.detail==='observation_unavailable') return {label:'位置观测失效，已停车',description:'无法继续检查位移上限，需要重新获取有效观测。'};
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
        if (entry.reason==='backend_unsupported') {
          label='途中检测不可用'; description='当前适配层不支持行进采图，未启动检测。';
        } else if (entry.reason==='unsupported') {
          label='当前类别不支持检测'; description='继续使用视觉模型观察。';
        } else if (entry.retrying===false || entry.reason==='AttributeError') {
          label='本段途中检测已停止'; description='采样任务异常退出，本段导航不再检测目标。';
        } else if (entry.stage==='observation') {
          label='行进采图暂不可用'; description='正在重试采图，当前尚无检测结果。';
        } else if (['TimeoutError','ServerTimeoutError'].includes(entry.reason)) {
          label='本次目标检测超时'; description='没有及时收到检测结果；继续导航，等待后续检测。';
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
  async function taskLoop() {
    const version = selectionVersion;
    try {
      if (!document.hidden) {
        // 暂停显示时仍检查运行状态，以便取消任务，避免提交重复任务。
        const {tasks} = await (await request('/tasks')).json();
        if (version !== selectionVersion || posting) return;
        online = true; activeId = tasks.find(t => t.status === 'running')?.id || ''; buttons();
        text('connection', 'Agent 已连接'); $('connection').classList.remove('error');
        if (paused) return;
        updateOptions(tasks);
        const selected = $('task-select').value;
        const nextId = selected || tasks[0]?.id || '';
        const record = nextId ? await (await request(`/tasks/${nextId}`)).json() : null;
        if (version !== selectionVersion || paused || posting) return;
        useTask(nextId); showTask(record);
        if (taskId) {
          const page = await (await request(`/tasks/${taskId}/events?after=${cursor}`)).json();
          if (version !== selectionVersion || paused || posting) return;
          const list = $('events'), atBottom = list.scrollHeight-list.scrollTop-list.clientHeight < 40;
          let added = false;
          for (const e of page.events) {
            if (e.type === 'action_input' && e.action === 'navigate' && e.instruction) {
              const row = [...list.children].find(item => item.dataset.kind === 'decision' && item.dataset.step === String(e.step));
              if (row) {
                const instruction = document.createElement('p');
                instruction.className = 'instruction'; instruction.textContent = 'VLN 指令：' + e.instruction;
                const gallery = row.querySelector('.event-frames');
                row.insertBefore(instruction, gallery); added = true;
              }
              continue;
            }
            const frames=evidenceFrames(e);
            let summary = keyEvent(e);
            if (e.type==='target_detection_frame' && detectorWaiting) {
              detectorWaiting=false; summary={label:'目标检测已恢复',description:'已重新收到有效检测结果。'};
            }
            if (e.type==='target_watch_degraded') detectorWaiting=true;
            if (!summary) continue;
            if (e.type === 'completion_check') completionSteps.add(e.step);
            if (e.category === 'completion_verification' && completionSteps.has(e.step)) continue;
            if (e.type === 'target_watch_degraded' && list.lastElementChild?.dataset.kind === e.type && list.lastElementChild?.dataset.label === summary.label) continue;
            const signature = e.type === 'decision_error' && e.step != null ? String(e.step) + ':' + e.category : null;
            if (signature && corrections.has(signature)) continue;
            if (signature) corrections.add(signature);
            const li = document.createElement('li');
            li.dataset.kind = e.type;
            li.dataset.label = summary.label;
            li.dataset.step = String(e.step);
            li.textContent = new Date(e.time).toLocaleTimeString() + ' · ' + summary.label +
              '\n' + summary.description;
            const gallery=document.createElement('div'); gallery.className='event-frames';
            for (const frame of frames) {
              const link=document.createElement('a'), img=document.createElement('img'), caption=document.createElement('span');
              link.href='/tasks/'+taskId+'/logs/'+frame.image; link.target='_blank'; link.rel='noopener';
              img.src=link.href; img.alt=frame.caption; img.loading='lazy'; caption.textContent=frame.caption+' · 点击放大';
              link.append(img,caption); gallery.append(link);
            }
            if (frames.length) li.append(gallery);
            list.append(li); added = true;
          }
          cursor = page.next;
          // 有新事件且用户本就在底部时才跟随；查看旧日志时不移动位置。
          if (added && atBottom) list.scrollTop = list.scrollHeight;
        }
        text('events-state', taskId ? '关键行动、当时画面与结果' : '等待任务');
      }
    } catch {
      online = false; buttons();
      text('connection', 'Agent 连接中断，展示上次记录，正在重试'); $('connection').classList.add('error');
    } finally { setTimeout(taskLoop, 1000); }
  }
  $('task-select').addEventListener('change', () => { selectionVersion++; });
  $('pause').addEventListener('click', () => {
    paused = !paused; selectionVersion++;
    $('pause').setAttribute('aria-pressed', String(paused));
    text('pause', paused ? '恢复画面与日志更新' : '暂停画面与日志更新');
    text('monitor-mode', paused ? '画面与日志已暂停更新 · 机器人任务仍会继续，可使用取消按钮停止任务' : '实时监控中 · 页面仅更新变化内容');
  });
  $('task-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!settingsLoaded || settingsBusy || posting || !online || activeId || !$('task-form').reportValidity()) return;
    const goal = $('goal-input').value.trim();
    if (!goal) { text('command-message', '请输入任务目标'); return; }
    posting = true; selectionVersion++; buttons();
    text('command-message', '正在提交任务…');
    try {
      const record = await (await request('/tasks', {
        goal,
        max_steps: Number($('max-steps').value),
        max_seconds: Number($('max-seconds').value),
        navigation_time_limit_enabled: navigationTimeLimitEnabled,
        ...(navigationTimeLimitEnabled ? {navigation_seconds: Number($('navigation-seconds').value)} : {}),
      })).json();
      activeId = record.id; $('task-select').value = ''; useTask(record.id); showTask(record);
      text('command-message', '任务已接收，正在执行。可在下方查看进度。');
    } catch (error) {
      online = false;
      text('command-message', `提交未确认：${error.message}。请查看最新任务状态后再重试。`);
    } finally { posting = false; selectionVersion++; buttons(); }
  });
  $('cancel-task').addEventListener('click', async () => {
    if (posting || !activeId) return;
    const cancelling = activeId;
    posting = true; selectionVersion++; buttons(); text('command-message', '正在取消任务并等待停车…');
    try {
      const record = await (await request(`/tasks/${cancelling}/cancel`, {})).json();
      activeId = record.status === 'running' ? record.id : '';
      if (taskId === record.id) showTask(record);
      text('command-message', `任务状态：${labels[record.status] || record.status}`);
    } catch (error) { text('command-message', `取消未确认：${error.message}。请检查本体状态。`); }
    finally { posting = false; selectionVersion++; buttons(); }
  });
  const settingTitles = {mpc:'导航速度与 MPC 权重',body:'本体最终限速',look:'环视与单段导航',
    agent:'Agent 决策与默认任务预算',prompt:'VLN 指令指南',exploration:'探索与失败重试',memory:'空间记忆与历史检索',protection:'停车、观测与导航保护'};
  let settingsBusy = false, settingsSupported = false, settingsLoaded = false;
  const settingInputs = {}, actualLabels = {};
  function buildSettings(schema) {
    if (Object.keys(settingInputs).length) return;
    for (const [group, prop] of Object.entries(schema.properties)) {
      const definition = schema.$defs[prop.$ref.split('/').pop()];
      const box=document.createElement('fieldset'),legend=document.createElement('legend'),grid=document.createElement('div');
      legend.textContent=settingTitles[group] || group;grid.className='setting-grid';box.append(legend,grid);
      for (const [key, field] of Object.entries(definition.properties)) {
        const item=document.createElement('label'),input=document.createElement(field.type==='string'?'textarea':'input'),actual=document.createElement('span');
        const name=`${group}.${key}`;item.textContent=field.title || key;
        input.required=true;input.name=name;
        if(field.type!=='string') {
          input.type='number';
          input.step=field.type==='integer'?'1':'any';
          if(field.minimum!==undefined)input.min=String(field.minimum);
          else if(field.exclusiveMinimum!==undefined)input.min=String(field.exclusiveMinimum+.000001);
          if(field.maximum!==undefined)input.max=String(field.maximum);
        } else {input.maxLength=field.maxLength || 500;input.minLength=field.minLength || 0;item.className='wide-setting';}
        input.id=`setting-${group}-${key}`;input.setAttribute('aria-label',field.title || key);
        actual.className='actual';actual.textContent=['mpc','body'].includes(group)?'本体当前：—':'下次任务生效';
        settingInputs[name]=input;actualLabels[name]=actual;item.append(input,actual);
        if(field.description){const hint=document.createElement('span');hint.className='actual';hint.textContent=field.description;item.append(hint);}
        grid.append(item);
      }
      $('settings-fields').append(box);
    }
    settingInputs['look.navigation_seconds'].addEventListener('input',previewPrompt);
  }
  function previewPrompt() {
    const seconds=settingInputs['look.navigation_seconds']?.value || '—';
    const limit=configuredNavigationTimeLimitEnabled ? `${seconds} 秒（编辑值）` : '不限时，仍受任务总预算与导航保护约束';
    text('prompt-preview',`指令指南在保存后用于下次任务。配置中的单段导航：${limit}`);
  }

  function syncTaskDefaults(values) {
    text('budget-note', taskDraft.size ? '保留本次任务的预算修改；保存运行参数不会覆盖这里。' : '预算从当前配置读取；此处可覆盖本次任务。');
    if (!taskDraft.has('navigation-seconds')) $('navigation-seconds').value=values.look.navigation_seconds;
    for(const [key,id] of [['max_steps','max-steps'],['max_seconds','max-seconds']]) {
      if (!taskDraft.has(id)) $(id).value=values.agent[key];
      settingInputs[`agent.${key}`].value=values.agent[key];
    }
  }
  function settingButtons() {
    $('settings-save').disabled = !settingsLoaded || !settingsSupported || settingsBusy || !!activeId || posting || !online;
    $('settings-load').disabled = settingsBusy;
    $('start').disabled = posting || !online || !!activeId || settingsBusy || !settingsLoaded;
    $('navigation-time-limit-enabled').disabled = posting || !!activeId || settingsBusy;
    $('navigation-seconds').disabled = !navigationTimeLimitEnabled || posting || !!activeId || settingsBusy;
  }
  function showActual(state) {
    for (const group of ['mpc','body']) {
      const values = group==='mpc' ? state?.mpc_config : state?.robot;
      for (const name of Object.keys(actualLabels).filter(name=>name.startsWith(group+'.'))) {
        const key=name.split('.')[1];
        const value = values?.[key];
        const label = `本体当前：${Number.isFinite(value)?value:'—'}`;
        if (actualLabels[`${group}.${key}`].textContent!==label) actualLabels[`${group}.${key}`].textContent=label;
      }
    }
  }
  function updateNavigationTimeLimit() {
    navigationTimeLimitEnabled = $('navigation-time-limit-enabled').checked;
    $('navigation-limit').hidden = !navigationTimeLimitEnabled;
    $('navigation-seconds').disabled = !navigationTimeLimitEnabled;
    $('navigation-unlimited').hidden = navigationTimeLimitEnabled;
  }
  $('navigation-time-limit-enabled').checked = false;
  $('navigation-time-limit-enabled').addEventListener('change', updateNavigationTimeLimit);
  updateNavigationTimeLimit();

  async function loadSettings() {
    if (settingsBusy) return;
    settingsLoaded=false; settingsBusy=true; settingButtons();
    try {
      const agent=await (await request('/settings/agent')).json();
      if (!taskDraft.has('max-steps')) $('max-steps').value=agent.ui_task_defaults.max_steps;
      if (!taskDraft.has('max-seconds')) $('max-seconds').value=agent.ui_task_defaults.max_seconds;
      if (!taskDraft.has('navigation-seconds')) $('navigation-seconds').value=agent.navigation_seconds;
      configuredNavigationTimeLimitEnabled=agent.navigation_time_limit_enabled;
      updateNavigationTimeLimit();
      $('max-seconds').min=agent.task_limits.min_seconds;$('max-seconds').max=agent.task_limits.max_seconds;
      $('navigation-seconds').min=agent.task_limits.min_navigation_seconds;
      $('navigation-seconds').max=agent.task_limits.max_navigation_seconds;
      settingsLoaded=true;
      const data=await (await request('/settings/motion')).json();
      buildSettings(data.schema);
      const durationInput=settingInputs['look.navigation_seconds'];
      durationInput.closest('label').hidden=!configuredNavigationTimeLimitEnabled;
      durationInput.disabled=!configuredNavigationTimeLimitEnabled;
      for (const [name,input] of Object.entries(settingInputs)) {const [group,key]=name.split('.');input.value=data.values[group][key];}
      syncTaskDefaults(data.values);previewPrompt();

      settingsSupported=data.supported;
      if(data.current) showActual({mpc_config:data.current.mpc,robot:data.current.body});
      text('settings-message',data.error || (data.saved?'已读取保存值；可对照“本体当前”查看实际值。':'已读取本体参数和 Agent 默认设置，尚未保存自定义设置。'));
    } catch(error) {settingsSupported=false;text('settings-message',`参数读取失败：${error.message}`);}
    finally {settingsBusy=false;settingButtons();}
  }
  $('settings-load').addEventListener('click',loadSettings);
  $('settings-form').addEventListener('submit',async event=>{
    event.preventDefault();
    if(settingsBusy || activeId || posting || !settingsSupported || !$('settings-form').reportValidity())return;
    const values={};
    for(const [name,input] of Object.entries(settingInputs)){const [group,key]=name.split('.');(values[group] ||= {})[key]=input.type==='number'?Number(input.value):input.value;}
    settingsBusy=true;settingButtons();text('settings-message','正在应用并确认本体参数…');
    try {
      const result=await (await request('/settings/motion',values)).json();
      syncTaskDefaults(result.values);
      for(const [name,input] of Object.entries(settingInputs)){const [group,key]=name.split('.');input.value=result.values[group][key];}
      previewPrompt();
      text('settings-message','已保存；本体参数已读回确认。Agent、指令指南和保护参数用于下次任务。');
    } catch(error) {text('settings-message',`保存未确认：${error.message}。可重新读取确认状态。`);}
    finally {settingsBusy=false;settingButtons();}
  });
  const motorNames=['右前·髋','右前·大腿','右前·小腿','左前·髋','左前·大腿','左前·小腿',
    '右后·髋','右后·大腿','右后·小腿','左后·髋','左后·大腿','左后·小腿'];
  const motorCells=motorNames.map(name=>{
    const cell=document.createElement('div'),value=document.createElement('strong');
    cell.className='motor';cell.append(document.createTextNode(name));value.textContent='—';cell.append(value);$('motor-grid').append(cell);
    return {cell,value};
  });
  function showTemperatures(robot, offline=false) {
    const age=robot?.motor_temps_age_s;
    const stale=offline || !robot?.connected || (Number.isFinite(age) && age>3);
    const temps=Array.isArray(robot?.motor_temps)?robot.motor_temps:[];
    let maximum=null;
    motorCells.forEach(({cell,value},i)=>{
      const t=temps[i],valid=typeof t==='number'&&Number.isFinite(t)&&t>=0;
      const label=valid?`${t}°C`:'—';if(value.textContent!==label)value.textContent=label;
      cell.classList.toggle('warm',valid&&t>=55&&t<75);cell.classList.toggle('hot',valid&&t>=75);
      cell.classList.toggle('stale-data',stale);
      if(valid)maximum=maximum===null?t:Math.max(maximum,t);
    });
    text('motor-max',maximum===null?'最高温度 —':`最高温度 ${maximum}°C`);
    $('motor-max').classList.toggle('hot',maximum!==null&&maximum>=75);
    $('motor-max').classList.toggle('warm',maximum!==null&&maximum>=55&&maximum<75);
    text('motor-status',stale?'温度数据离线或已过期':maximum===null?'尚无有效电机温度':Number.isFinite(age)?`温度数据正常 · ${age.toFixed(1)} 秒前采样`:'已收到温度，本体未提供采样时间');
  }
  for(const id of ['max-steps','max-seconds','navigation-seconds']) {
    $(id).addEventListener('input',()=>{
      taskDraft.add(id);
      text('budget-note','本次任务预算已修改；如需设为默认值，请在参数面板修改默认预算并保存。');
    });
  }
  loadSettings();

  cameraLoop(); robotLoop(); taskLoop();
})();
