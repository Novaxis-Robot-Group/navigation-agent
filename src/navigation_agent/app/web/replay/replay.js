(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const taskId = location.pathname.split('/').pop();
  let logBase;
  const videoBase = `/api/recordings/${taskId}/`;
  let data, time = 0, playing = false, lastTick, activeNode = -1, videoUrl = '';
  let bounds, lastDraw = -Infinity;
  const video = $('video');
  function text(tag, value, className = '') {
    const element = document.createElement(tag);
    element.textContent = value;
    element.className = className;
    return element;
  }
  function clock(t) {
    t = Math.max(0, t);
    return `${String(Math.floor(t / 60)).padStart(2, '0')}:${(t % 60).toFixed(1).padStart(4, '0')}`;
  }
  // 二分定位到当前时刻之前的最后一条记录；缺失时不使用未来画面。
  function before(items, t, key = 't') {
    let low = 0, high = items.length;
    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      if (items[mid][key] <= t) low = mid + 1;
      else high = mid;
    }
    return low - 1;
  }
  function hasVideo() {
    return data?.frames.length && data.recording.views?.length &&
      ['ready', 'interrupted'].includes(data.recording.status);
  }
  function mappedTime(t) {
    const i = before(data.frames, t);
    if (i < 0) return {v: 0, hold: true};
    const frame = data.frames[i], next = data.frames[i + 1];
    const gap = next ? next.t - frame.t : Infinity;
    if (!next || gap > .5 || gap <= 0) return {v: frame.v, hold: true};
    return {v: frame.v + (t - frame.t) / gap * (next.v - frame.v),
      rate: (next.v - frame.v) / gap, hold: false};
  }
  function setPlaying(value) {
    playing = value;
    lastTick = undefined;
    $('play').textContent = playing ? '暂停' : '播放';
    if (!playing) video.pause();
  }
  function seek(t, play = false) {
    time = Math.max(0, Math.min(data.duration, t));
    setPlaying(play);
    render(true);
  }
  function showView() {
    if (hasVideo()) {
      if (!data.recording.views.includes($('view').value)) $('view').value = data.recording.views[0];
      const url = videoBase + $('view').value + '.mp4';
      if (videoUrl !== url) {
        videoUrl = url;
        video.src = url;
        $('download').href = url;
        $('download').download = `${taskId}-${$('view').value}.mp4`;
      }
      video.hidden = false;
      $('image').hidden = true;
      $('empty').hidden = true;
      $('download').hidden = false;
      const views = data.recording.views;
      for (const option of $('view').options) option.hidden = !views.includes(option.value);
      $('view').disabled = views.length < 2;
    } else {
      video.hidden = true;
      $('download').hidden = true;
      $('view').value = 'first';
      $('view').disabled = true;
    }
    render(true);
  }
  function render(force = false) {
    if (!data) return;
    $('time').textContent = `${clock(time)} / ${clock(data.duration)}`;
    $('seek').value = time;
    if (hasVideo()) {
      const mapping = mappedTime(time);
      if (video.readyState && (force || Math.abs(video.currentTime - mapping.v) > .2)) {
        video.currentTime = mapping.v;
      }
      if (playing && !mapping.hold) {
        video.playbackRate = Math.max(.1, Math.min(8, Number($('speed').value) * mapping.rate));
        if (video.paused) video.play().catch(error => {
          if (error.name !== 'AbortError') setPlaying(false);
        });
      } else video.pause();
      $('frame-note').textContent = mapping.hold ? '该时段没有连续录制帧，保持最近画面。' : '按原始采图时间同步画面与任务节点。';
    } else {
      const i = before(data.images, time);
      $('image').hidden = i < 0;
      $('empty').hidden = i >= 0;
      if (i >= 0) {
        const frame = data.images[i];
        const source = logBase + frame.image;
        if ($('image').getAttribute('src') !== source) $('image').src = source;
        $('frame-note').textContent = `抽帧回放 · 画面拍摄于 ${clock(frame.t)} · 距当前时刻 ${(time - frame.t).toFixed(1)} 秒`;
      } else $('frame-note').textContent = '尚未到第一张已保存画面的拍摄时刻。';
    }
    let i = before(data.nodes, time);
    while (i >= 0 && !visible(data.nodes[i])) i--;
    if (force || i !== activeNode) {
      activeNode = i;
      document.querySelectorAll('#events li').forEach((row) => row.classList.toggle('active', Number(row.dataset.index) === i));
      $('caption').textContent = i >= 0 ? `${data.nodes[i].label} · ${data.nodes[i].description}` : '';
    }
    if (force || Math.abs(time - lastDraw) >= .1) {
      lastDraw = time;
      drawTrajectory();
    }
  }
  function drawTrajectory() {
    const canvas = $('trajectory'), ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const points = data.trajectory;
    if (!points.length) {
      ctx.fillStyle = '#716f66'; ctx.fillText('此任务未保存有效位姿', 20, 30); return;
    }
    const {minX, maxX, minY, maxY} = bounds;
    const scale = Math.min(740 / Math.max(1, maxX - minX), 240 / Math.max(1, maxY - minY));
    const xy = (p) => [30 + (p.pose[0] - minX) * scale, 270 - (p.pose[1] - minY) * scale];
    const last = before(points, time);
    function line(end, color) {
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
      for (let i = 0; i <= end; i++) {
        const [x, y] = xy(points[i]);
        // 较长空档断开连线，避免把未记录的移动画成确定路径。
        if (!i || points[i].t - points[i - 1].t > 2) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    line(points.length - 1, '#d4cfc4'); line(last, '#796347');
    if (last >= 0) {
      const [x, y] = xy(points[last]);
      ctx.save();ctx.translate(x, y);ctx.rotate(-points[last].pose[2]);
      ctx.beginPath();ctx.moveTo(10, 0);ctx.lineTo(-7, -6);ctx.lineTo(-7, 6);ctx.closePath();
      ctx.fillStyle = '#506e65';ctx.fill();ctx.restore();
    }
    ctx.fillStyle = '#716f66';ctx.font = '12px system-ui';
    ctx.fillText(`x: ${minX.toFixed(1)}–${maxX.toFixed(1)} m    y: ${minY.toFixed(1)}–${maxY.toFixed(1)} m`, 20, 20);
  }
  function visible(node) {
    return node.key !== undefined ? node.key :
      !['observation', 'action_input', 'action_result', 'skill_selected', 'target_detection_navigation_returned'].includes(node.type) ||
      (node.type === 'action_result' && /受阻|失败|异常|无进展|超时|退回/.test(node.description));
  }
  function renderNodes() {
    $('events').replaceChildren(); $('markers').replaceChildren();
    data.nodes.forEach((node, i) => {
      if (!visible(node)) return;
      const row = text('li', ''); row.dataset.index = i;
      const jump = text('button', `${clock(node.t)} · ${node.label}${node.step ? ' · O' + node.step : ''}`, 'jump');
      jump.onclick = () => seek(node.t, node.type === 'action_input');
      row.append(jump, text('p', node.description));
      if (node.instruction) row.append(text('p', 'VLN 指令：' + node.instruction, 'instruction'));
      const links = text('div', '', 'links');
      if (node.type === 'decision' && node.observation_t != null) {
        const observation = text('button', '看当时画面');
        observation.onclick = () => seek(node.observation_t);
        links.append(observation);
        const action = data.nodes.find(n => n.type === 'action_input' && n.step === node.step && n.t >= node.t);
        if (action) {
          const execution = text('button', '看动作执行');execution.onclick = () => seek(action.t, true);links.append(execution);
        }
      }
      const log = text('a', data.task.mode === 'vln' ? '运行日志 ↗' : '任务摘要 ↗');log.href = logBase + 'index.html' + (node.type === 'decision' && node.step ? `#step-${node.step}` : '');log.target = '_blank';log.rel = 'noopener';links.append(log);
      row.append(links);
      $('events').append(row);
      const marker = text('button', '');marker.style.left = `${100 * node.t / Math.max(1, data.duration)}%`;
      marker.className = node.type === 'decision' ? 'decision' : /error/.test(node.type) ? 'error' : '';
      marker.title = `${clock(node.t)} · ${node.label}`;marker.setAttribute('aria-label', marker.title);marker.onclick = jump.onclick;
      $('markers').append(marker);
    });
  }
  async function load() {
    setPlaying(false);$('refresh').disabled = true;
    try {
      const response = await fetch(`/api/replay/${taskId}`, {cache: 'no-store'});
      if (!response.ok) throw new Error(await response.text());
      data = await response.json();logBase = data.log_base;time = Math.min(time, data.duration);
      bounds = data.trajectory.reduce((b, p) => ({minX: Math.min(b.minX, p.pose[0]), maxX: Math.max(b.maxX, p.pose[0]),
        minY: Math.min(b.minY, p.pose[1]), maxY: Math.max(b.maxY, p.pose[1])}),
        {minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity});
      $('goal').textContent = data.task.goal;
      const isVln = data.task.mode === 'vln';
      $('replay-kind').textContent = isVln ? 'NAV / VLN · REPLAY' : 'NAV / AGENT · REPLAY';
      $('events-heading').textContent = isVln ? 'VLN 执行事件' : '关键节点';
      $('logs').textContent = isVln ? '运行日志 ↗' : '任务摘要 ↗';
      $('summary').textContent = `${new Date(data.task.started_at).toLocaleString('zh-CN')} · ${data.task.summary || data.task.status}`;
      $('seek').max = data.duration;$('play').disabled = false;
      $('logs').href = logBase + 'index.html';
      const states = {ready: data.recording.views?.length === 1 ? '第一视角录像已保存' : '双视角录像已保存', recording: '正在录像，任务结束后刷新即可播放视频', missing: '此任务没有连续录像，使用已有观察画面回放', disabled: '录像未开启', failed: '录像失败', interrupted: '录像曾中断'};
      $('notice').textContent = (states[data.recording.status] || data.recording.status) + (data.recording.error ? '：' + data.recording.error : '') + (data.recording.dropped_frames ? ` · 编码队列丢弃 ${data.recording.dropped_frames} 帧` : '');
      $('mode').textContent = hasVideo() ? '连续录像 · 原始时间映射' : '历史观察抽帧回放';
      $('subtitles').hidden = !hasVideo();
      $('subtitles').href = `/api/replay/${taskId}/subtitles.vtt`;
      renderNodes();showView();
    } catch (error) {$('notice').textContent = '回放加载失败：' + error.message;}
    finally {$('refresh').disabled = false;}
  }
  $('play').onclick = () => {if (time >= data.duration) seek(0);setPlaying(!playing);};
  $('seek').oninput = () => seek(Number($('seek').value));
  $('view').onchange = showView;
  $('refresh').onclick = load;
  $('previous').onclick = () => {if (data) seek(data.nodes.slice(0, activeNode).reverse().find(visible)?.t || 0);};
  $('next').onclick = () => {if (data) seek(data.nodes.slice(activeNode + 1).find(visible)?.t ?? data.duration);};
  video.onloadedmetadata = () => render(true);
  video.onerror = () => {setPlaying(false);$('notice').textContent = '视频无法读取或浏览器不支持该编码，可下载 MP4 检查。';};
  function tick(now) {
    if (playing && data) {
      if (lastTick != null) time = Math.min(data.duration, time + (now - lastTick) / 1000 * Number($('speed').value));
      lastTick = now;render();if (time >= data.duration) setPlaying(false);
    }
    requestAnimationFrame(tick);
  }
  document.addEventListener('visibilitychange', () => {if (document.hidden) setPlaying(false);});
  requestAnimationFrame(tick);load();
})();
