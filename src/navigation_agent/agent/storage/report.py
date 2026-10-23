"""生成离线日志页；命令行入口由 navigation_agent.app.reports.main 装配。"""

from datetime import datetime, timedelta, timezone
from html import escape
import json
from pathlib import Path
import re

from navigation_agent.agent.storage.memory_sections import read_events

LABELS = {
    'running': '执行中',
    'completed': '已完成',
    'failed': '未完成',
    'cancelled': '已取消',
    'budget_exhausted': '达到执行上限',
    'interrupted': '已中断',
}
ACTIONS = {
    'watch_target': '配置目标监测',
    'look': '环视观察',
    'orient': '转回确认',
    'oriented': '已转回观察方向',
    'navigate': '执行导航',
    'recover': '退回脱困',
    'finish': '确认完成',
    'give_up': '结束探索',
}
NAMES = {
    'target_watch_configured': '目标监测配置',
    'target_watch_degraded': '目标检测不可用，继续原有流程',
    'target_detection_interrupt': '检测候选申请停车',
    'target_detection_navigation_returned': '检测后导航已返回',
    'target_detection_hit': '检测候选证据',
    'target_detection_review': '检测候选复核',
    'execution_goal': '导航英文任务',
    'selective_transition': '调度切换',
    'effective_execution': '实际执行配置',
    'started': '任务开始',
    'observation': '观察环境',
    'decision_input': '决策上下文',
    'decision': 'Agent 决策',
    'decision_error': '决策异常',
    'observation_report': '观察报告',
    'passive_observation': '行进采图',
    'monitor_report': '行进观察报告',
    'monitor_request': '行进观察请求',
    'navigation_interrupt_requested': '申请停车复核',
    'monitor_error': '行进观察失败',
    'monitor_frame_error': '行进采图失败',
    'subgoal_changed': '探索目的更新',
    'observation_gain': '观察所得线索',
    'decision_latency': '模型决策用时',
    'completion_check': '独立完成核验',
    'region_comparison': '停车房间对照',
    'region_comparison_error': '停车房间对照不可用',
    'stopped_observation_latency': '停车与新帧等待用时',
    'action_latency': '动作执行用时',
    'action_input': '执行动作',
    'action_result': '动作反馈',
    'finished': '任务结束',
    'exploration_session_finished': '探索阶段结束',
}
OUTCOMES = {
    'turned': '已完成转向观察',
    'blocked': '前方受阻，已停止导航',
    'navigation_error': '导航服务异常',
    'timeout': '导航超时',
    'model_stop': '导航模型发出停止信号',
    'agent_interrupt': '新观察触发重规划，已停车待复核',
    'path_repeated': '导航绕回旧位置，停车复核',
    'no_progress': '原地无进展，停止并准备脱困',
    'recovered': '已沿原路径退回',
    'recovery_unavailable': '缺少连续退路或碰撞保护',
    'recovery_blocked': '退回途中受阻，已停止',
    'recovery_timeout': '脱困超时，已停止',
    'recovery_no_progress': '脱困未产生足够位移',
}


def text(value):
    return escape(str(value if value is not None else ''), quote=True)


def timestamp(value, utc_offset_minutes):
    try:
        return (
            datetime.fromisoformat(value)
            .astimezone(timezone(timedelta(minutes=utc_offset_minutes)))
            .strftime('%Y-%m-%d %H:%M:%S')
        )
    except (ValueError, TypeError):
        return str(value or '—')



def key_event(entry):
    """把原始记录压缩为面向读者的节点；不改变事件或任务判断。"""
    kind = entry.get('type')
    decision = entry.get('decision') or {}
    result = entry.get('result') or {}
    label, description = '', ''
    if kind == 'started':
        label, description = '任务开始', entry.get('goal', '')
    elif kind == 'execution_goal':
        label, description = '导航英文任务', entry.get('instruction', '')
    elif kind == 'selective_transition':
        label = '调度切换'
        description = str(entry.get('previous', '')) + ' → ' + str(entry.get('mode', ''))
        description += '；触发原因：' + str(entry.get('trigger', ''))
    elif kind == 'decision':
        if decision.get('action') in ('finish', 'give_up'):
            return None
        label = ACTIONS.get(decision.get('action'), '选择下一步')
        description = decision.get('reason', '')
    elif kind == 'finished':
        label = LABELS.get(entry.get('status'), '任务结束')
        description = entry.get('summary', '')
    elif kind == 'action_result':
        if result.get('motion_limit_unavailable') or result.get('detail') == 'observation_unavailable':
            missing = result.get('motion_limit_unavailable')
            return {'type': kind, 'time': entry.get('time'), 'step': entry.get('step'),
                    'label': '位移限制未生效' if missing else '位置观测失效，已停车',
                    'description': '此段导航没有执行指定的位移上限。' if missing
                                   else '无法继续检查位移上限，需要重新获取有效观测。'}
        reason = result.get('reason')
        if reason in ('turned', 'oriented', 'model_stop', 'agent_interrupt', 'completed', 'success', None) and not result.get('startup_failed'):
            return None
        label = '动作反馈'
        description = OUTCOMES.get(reason, reason)
        if result.get('startup_failed'):
            description = '导航服务启动失败，未执行接近'
    elif kind == 'completion_check':
        if entry.get('confirmed'):
            return None
        label = '完成核验通过' if entry.get('confirmed') else '尚未满足完成条件'
        description = entry.get('evidence') or entry.get('error') or '继续观察或接近。'
    elif kind == 'decision_error':
        if entry.get('category') == 'completion_verification':
            return {'type': kind, 'time': entry.get('time'), 'step': entry.get('step'),
                    'label': '尚未满足完成条件', 'description': '本次完成判断未通过核验，后续处理见下一节点。'}
        label = '调整下一步'
        description = ('动作未通过检查，未执行；后续处理见下一节点。'
                       if entry.get('category') in ('scene_rule', 'exploration_rule')
                       else '模型输出不完整，本次动作未执行；后续处理见下一节点。')
    elif kind == 'decision_replan':
        label, description = '重新观察', '本轮未形成可执行动作，重新观察后再决定。'
    elif kind == 'target_detection_hit':
        label, description = '检测到目标线索', '已请求停车，等待千问核实目标。'
    elif kind == 'target_detection_review':
        label = {'confirmed': '已确认目标', 'rejected': '排除误检', 'uncertain': '目标尚待确认'}.get(entry.get('verdict'), '复核目标线索')
        description = entry.get('evidence') or '根据停车画面重新判断。'
    elif kind == 'target_watch_degraded':
        reason = entry.get('reason', '')
        if reason == 'backend_unsupported':
            label, description = '途中检测不可用', '当前适配层不支持行进采图，未启动检测。'
        elif reason == 'unsupported':
            label, description = '当前类别不支持检测', '继续使用视觉模型观察。'
        elif entry.get('retrying') is False or reason == 'AttributeError':
            label, description = '本段途中检测已停止', '采样任务异常退出，本段导航不再检测目标。'
        elif entry.get('stage') == 'observation':
            label, description = '行进采图暂不可用', '正在重试采图，当前尚无检测结果。'
        elif reason in ('TimeoutError', 'ServerTimeoutError'):
            label, description = '本次目标检测超时', '没有及时收到检测结果；继续导航，等待后续检测。'
        elif reason in ('ClientConnectorError', 'ClientConnectionError'):
            label, description = '无法连接目标检测服务', '本次未获得检测结果，继续原有观察流程。'
        else:
            label, description = '本次目标检测失败', '本次未获得检测结果，继续原有观察流程。'
    elif kind == 'navigation_interrupt_requested':
        if not entry.get('accepted'):
            return None
        label = '停车复核'
        description = {'target_seen': '行进中发现目标线索。', 'subgoal_reached': '当前局部目标可能已到达。',
                       'route_invalid': '当前路线需要重新判断。'}.get(entry.get('reason'), '行进中有新线索，停车后重新判断。')
    else:
        return None
    description = ' '.join(str(description or '').split())
    return {'type': kind, 'time': entry.get('time'), 'step': entry.get('step'),
            'label': label, 'description': description}


def event_frame(path, caption, time=None):
    """只引用日志中明确记录的任务图片，不猜文件名或补用未来画面。"""
    if isinstance(path, str) and re.fullmatch(r'images/(?:detection-)?[0-9]{3,}\.jpg', path):
        return {'image': path, 'caption': caption, 'time': time}
    return None


def key_events(events):
    """把关键行动、当时证据和真实结果组成图文节点，完整事件仍独立保存。"""
    nodes, observations, detections, actions = [], {}, {}, {}
    seen, completion_steps = set(), set()
    current = None
    detector_waiting = False
    previous = None
    for entry in events:
        kind, step = entry.get('type'), entry.get('step')
        if kind in ('observation', 'passive_observation'):
            frame = event_frame(entry.get('image'), '当时画面', entry.get('time'))
            if frame:
                observations[step] = frame
                if kind == 'observation':
                    current = frame
        elif kind == 'decision_input':
            # 当前图优先用明确字段；附图清单可能同时含历史参考，不能拿第一张代替。
            frame = event_frame(entry.get('image'), '决策时画面', entry.get('time'))
            if frame is None:
                for item in entry.get('images') or []:
                    if any(use.get('purpose') == '当前图像 A' for use in item.get('uses', [])):
                        frame = event_frame(item.get('image'), '决策时画面', entry.get('time'))
                        break
            if frame:
                observations[step] = current = frame
        elif kind == 'target_detection_hit':
            detections[entry.get('candidate_id')] = event_frame(
                entry.get('boxed_image') or entry.get('image'), '检测触发画面', entry.get('time'))
        elif kind == 'action_input' and entry.get('action') == 'navigate' and step in actions:
            # 只展示实际交给导航工具的原文，不把未执行的模型草稿当成已发送。
            actions[step]['instruction'] = entry.get('instruction', '')
        elif kind == 'action_result' and step in actions:
            result = entry.get('result') or {}
            outcome = OUTCOMES.get(result.get('reason'), result.get('reason', '动作结束'))
            if result.get('detail') == 'target_detected':
                outcome = '检测到候选，导航中断并返回复核'
            if result.get('startup_failed'):
                outcome = '导航服务启动失败，动作未执行'
            actions[step]['outcome'] = outcome
        if kind == 'target_detection_frame' and detector_waiting:
            # 只在真实收到检测响应后显示恢复，不把服务探活当成检测恢复。
            detector_waiting = False
            previous = kind
            nodes.append({'type': kind, 'time': entry.get('time'), 'step': None,
                          'label': '目标检测已恢复', 'description': '已重新收到有效检测结果。',
                          'frames': []})
        node = key_event(entry)
        if node is None:
            continue
        if kind == 'completion_check':
            completion_steps.add(step)
        if entry.get('category') == 'completion_verification' and step in completion_steps:
            continue
        if kind == 'target_watch_degraded':
            detector_waiting = True
            if previous == kind and nodes and nodes[-1]['label'] == node['label']:
                continue
        if kind == 'decision_error' and step is not None:
            signature = (step, entry.get('category'))
            if signature in seen:
                continue
            seen.add(signature)
        node['frames'] = []
        if kind == 'decision':
            frame = observations.get(step)
            if frame:
                node['frames'] = [{**frame, 'caption': '决策时画面'}]
            decision = entry.get('decision') or {}
            node['mode'] = {'search_navigation': '搜索', 'approach_target': '接近 / 核实'}.get(decision.get('skill'), '')
            review = decision.get('target_review') or {}
            if review.get('status') == 'temporarily_lost':
                node['review'] = '目标暂时不可见：' + review.get('evidence', '')
            actions[step] = node
        elif kind == 'target_detection_hit':
            frame = detections.get(entry.get('candidate_id'))
            if frame:
                node['frames'] = [frame]
        elif kind == 'target_detection_review':
            detected = detections.get(entry.get('candidate_id'))
            if detected:
                node['frames'].append(detected)
            if current and (not detected or current['image'] != detected['image']):
                node['frames'].append({**current, 'caption': '停车复核画面'})
        elif kind in ('completion_check', 'finished') and current:
            node['frames'] = [{**current, 'caption': '最后记录的观察' if kind == 'finished' else '核验时画面'}]
        previous = kind
        nodes.append(node)
    return nodes


def render_key_event(node, utc_offset_minutes):
    step = node.get('step')
    anchor = f' id="step-{text(step)}"' if node['type'] == 'decision' and step is not None else ''
    alias = f'<span id="decision-input-{text(step)}"></span>' if anchor else ''
    frames = []
    for frame in node.get('frames', []):
        stamp = timestamp(frame.get('time'), utc_offset_minutes)
        caption = frame['caption'] + ((' · ' + stamp[11:]) if len(stamp) > 11 else '')
        path = text(frame['image'])
        frames.append(
            f'<figure><a class="frame" href="{path}" data-caption="{text(caption)}">'
            f'<img src="{path}" loading="lazy" alt="{text(caption)}"></a>'
            f'<figcaption>{text(caption)} · 点击放大</figcaption></figure>'
        )
    pictures = '<div class="frames">' + ''.join(frames) + '</div>' if frames else ''
    outcome = f'<p class="outcome">结果：{text(node["outcome"])}</p>' if node.get('outcome') else ''
    instruction = (f'<p class="instruction"><b>VLN 指令：</b><span lang="en">{text(node["instruction"])}</span></p>'
                   if node.get('instruction') else '')
    review = f'<p class="review">{text(node["review"])}</p>' if node.get('review') else ''
    mode = f'<span class="badge">{text(node["mode"])}</span>' if node.get('mode') else ''
    stamp = timestamp(node.get('time'), utc_offset_minutes)
    return (
        f'<article{anchor} data-kind="{text(node["type"])}">{alias}'
        f'<time>{text(stamp[11:] if len(stamp) > 11 else stamp)}</time>'
        f'<div class="content"><h3>{text(node["label"])}{mode}</h3>'
        f'<p>{text(node["description"])}</p>{instruction}{review}{outcome}{pictures}</div></article>'
    )


def write_report(directory, record=None, *, utc_offset_minutes):
    directory = Path(directory)
    if record is None:
        record = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
    events, invalid = read_events(directory)
    nodes = key_events(events)
    rows = ''.join(render_key_event(node, utc_offset_minutes) for node in nodes)
    notice = f'<p>有 {invalid} 行日志不完整或无效，已跳过；原始文件仍保留。</p>' if invalid else ''

    # 把时间线和任务摘要填入离线页面
    html = (
        TEMPLATE.replace(
            '<!--HEADER-->',
            f'<div class="eyebrow">VLN AGENT · 任务日志</div><h1>{text(record.get("goal", "任务记录"))}</h1>'
            f'<div class="meta"><span class="badge">{text(LABELS.get(record.get("status"), record.get("status", "未知")))}</span>'
            f'<span>{text(record.get("step", 0))} 轮决策</span></div>'
            f'<p class="muted">{text(timestamp(record.get("started_at"), utc_offset_minutes))} → {text(timestamp(record.get("ended_at"), utc_offset_minutes))}</p>'
            f'<p>{text(record.get("summary", ""))}</p>'
            f'<p class="muted">关键行动、当时画面与结果 · 图片可点击放大</p>{notice}',
        )
        .replace('<!--COUNT-->', str(len(nodes)))
        .replace('<!--ROWS-->', rows or '<p class="empty">暂无执行记录</p>')
    )
    destination = directory / 'index.html'
    temporary = directory / 'index.html.tmp'
    temporary.write_text(html, encoding='utf-8')
    temporary.replace(destination)
    return destination


TEMPLATE = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>任务图文记录 · Navigation Agent</title>
<style>
:root {color-scheme:light;font:14px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#f3f2ee;color:#39362f}
* {box-sizing:border-box}
body {margin:0;padding:24px 16px}
main {max-width:860px;margin:auto}
header,.panel {border:1px solid #d9d4c8;border-radius:14px;background:#faf8f2}
header {padding:20px 24px;margin-bottom:16px}
h1 {font-size:24px;margin:6px 0 12px}
h3 {font-size:15px;margin:0 0 6px}
p {margin:6px 0;overflow-wrap:anywhere}
.eyebrow,.muted,time,figcaption {font-size:12px;color:#777267}
.meta,.toolbar {display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.badge {font-size:11px;background:#e7e1d5;border-radius:4px;padding:2px 7px;display:inline-block}
h3 .badge {margin-left:10px;font-weight:400}
.toolbar {padding:12px 18px;border-bottom:1px solid #d9d4c8}
input,select,button {font:inherit;color:inherit;background:#f1ede3;border:1px solid #cdc5b6;border-radius:6px;padding:6px 10px}
input {min-width:0;width:180px;margin-left:auto}
article {display:grid;grid-template-columns:65px minmax(0,1fr);gap:12px;margin:0 18px;padding:16px 0;border-bottom:1px solid #e2ddd3;scroll-margin-top:12px}
article:last-child {border:0}
article:target {background:#f1ede3}
article > span[id] {display:none}
time {font-variant-numeric:tabular-nums;padding-top:2px}
.outcome {color:#53685c;font-size:13px}
.instruction {color:#655340;font-size:13px;white-space:pre-wrap}
.review {color:#856347;font-size:13px}
.frames {display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
figure {margin:0;width:190px;max-width:100%}
.frame {display:block;border:1px solid #ccc4b5;border-radius:7px;overflow:hidden;background:#ece9e1}
.frame img {display:block;width:100%;aspect-ratio:16/9;object-fit:contain}
figcaption {font-size:11px;margin-top:3px}
.empty,footer {padding:14px;color:#777267;font-size:12px}
dialog {max-width:min(1000px,96vw);padding:12px;border:1px solid #cfc3ae;border-radius:12px;background:#faf8f2;color:#39362f}
dialog::backdrop {background:#282521b8}
dialog img {display:block;max-width:100%;max-height:75vh;object-fit:contain;margin:10px auto}
dialog .toolbar {padding:0;border:0;justify-content:space-between}
[hidden] {display:none!important}
@media(max-width:600px) {
body {padding:10px} header {padding:16px} h1 {font-size:22px}
article {grid-template-columns:1fr;gap:3px;margin:0 14px}
time {font-size:11px} figure {width:calc(50% - 5px);min-width:135px}
input {width:100%;margin:0}
}
</style>
</head>
<body>
<main>
<header><!--HEADER--></header>
<section class="panel">
<div class="toolbar"><b>关键节点 <span class="badge"><!--COUNT--></span></b>
<select aria-label="记录类型"><option value="">全部节点</option><option value="decision">行动</option><option value="target_detection_hit">目标线索</option><option value="target_detection_review">目标复核</option><option value="completion_check">完成核验</option><option value="finished">结果</option></select>
<input type="search" placeholder="搜索关键节点…" aria-label="搜索日志"></div>
<div id="events"><!--ROWS--></div><p id="empty" class="empty" hidden>没有匹配的记录</p>
</section>
<footer>图像来自当时保存的记录；缺少画面时不补用其他时刻的图片。</footer>
</main>
<dialog id="picture"><div class="toolbar"><span id="picture-caption"></span><button id="close-picture" aria-label="关闭大图">关闭</button></div><img id="picture-image" alt=""></dialog>
<script>
const search=document.querySelector('input'), filter=document.querySelector('select');
const rows=[...document.querySelectorAll('#events > article')];
function apply() {
  let count=0;
  for (const row of rows) {
    row.hidden=!!((filter.value && row.dataset.kind!==filter.value) || !row.textContent.toLowerCase().includes(search.value.toLowerCase()));
    if (!row.hidden) count++;
  }
  document.querySelector('#empty').hidden=count>0;
}
search.addEventListener('input',apply); filter.addEventListener('change',apply);
function revealHash() {
  const target=document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (target) {search.value='';filter.value='';apply();target.scrollIntoView();}
}
window.addEventListener('hashchange',revealHash); if (location.hash) revealHash();
const picture=document.querySelector('#picture'), image=document.querySelector('#picture-image');
document.querySelectorAll('a.frame').forEach(link => link.addEventListener('click',event => {
  event.preventDefault();image.src=link.getAttribute('href');image.alt=link.dataset.caption;
  document.querySelector('#picture-caption').textContent=link.dataset.caption;picture.showModal();
}));
document.querySelector('#close-picture').onclick=()=>picture.close();
picture.addEventListener('click',event=>{if(event.target===picture) picture.close();});
</script>
</body>
</html>
'''
