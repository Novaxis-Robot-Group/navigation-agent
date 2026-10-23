"""导出观察、轨迹和决策输入的记忆阅读页；命令行入口由 navigation_agent.app.reports.main 装配。"""

from contextlib import closing
from html import escape
import json
from pathlib import Path
import re
import sqlite3

from navigation_agent.agent.memory.trajectory import ACTION_LABELS, RESULT_LABELS, finite_number
from navigation_agent.agent.storage.memory_sections import details, read_events, render_memory_input, render_reviews, step_number


def text(value):
    return escape(str(value if value is not None else '—'), quote=True)


def decode(value, default=None):
    return json.loads(value) if value else default


def field(label, value):
    if isinstance(value, list):
        value = '、'.join(str(item) for item in value)
    value = '—' if value is None or value == '' else value
    return f'<div><dt>{text(label)}</dt><dd>{text(value)}</dd></div>'


def frame_card(frame, objects, directory):
    scene = decode(frame['scene_json'])
    observation = decode(frame['observation_json'], {})
    review = decode(frame['review_json'])
    image = frame['image']
    picture = '<div class="missing">观察图片不可用</div>'
    # 图片只引用任务目录内的标准观察图，目录改名或移动后仍可打开。
    if re.fullmatch(r'images/\d+\.jpg', image) and (directory / image).is_file():
        picture = f'<a href="{text(image)}"><img src="{text(image)}" loading="lazy" alt="第 {frame["step"]} 步观察图"></a>'
    heading = frame['heading_deg']
    heading_text = f'{heading:.1f}°' if heading is not None else '未知'
    pose = observation.get('pose') or {}
    position = [pose.get('x'), pose.get('y')] if observation.get('pose_valid', True) else '定位无效'
    body = field('拍摄时间', frame['observed_at']) + field('机器人位置 / 朝向', f'{position} / {heading_text}')
    if scene is None:
        body += field('记录状态', '仅原始观察：尚无模型视觉描述')
    else:
        source = '模型视觉报告'
        if frame.get('report_status') == 'rejected_action':
            source += '（对应动作被拒绝，未执行；线索待复核）'
        if frame.get('report_status') == 'passive_observation':
            source += '（行进观察，仅作线索，未直接授权动作）'
        body += field('描述来源', source)
        body += field('地点推测', scene.get('place_label'))
        body += field('可见参照物', scene.get('landmarks'))
        body += field('可见地面', scene.get('visible_ground'))
        body += field('遮挡区域', scene.get('occluded'))
        body += field('是否看见目标', {'yes': '看见', 'no': '未看见', 'uncertain': '不确定'}.get(scene.get('target_seen'), '未知'))
    sightings = ''.join(
        f'<li><strong>{text(item["name"])}</strong> · {text(item["description"])} '
        f'<small>（相对方向 {text(item["bearing_deg"])}° · {text(item["id"])}）</small></li>'
        for item in objects
    )
    body += field('物体目击数', len(objects))
    review_html = ''
    if review:
        outcome = {'progress': '有进展', 'no_new_information': '没有新信息', 'uncertain': '不确定'}.get(review.get('outcome'), '未知')
        review_html = (
            f'<aside>对第 {text(frame["previous_action_step"])} 步动作的复核：<b>{text(outcome)}</b>'
            f'<p>{text(review.get("evidence"))}</p></aside>'
        )
    raw = dict(frame)
    for key in ('observation_json', 'scene_json', 'review_json'):
        raw[key] = decode(raw[key])
    raw['objects'] = objects
    return (
        f'<article class="memory-record" id="frame-{frame["step"]}"><h3>第 {frame["step"]} 步 <span>Semantic Frame</span></h3>'
        f'<div class="frame">{picture}<div><dl>{body}</dl>'
        f'<ul>{sightings}</ul>{review_html}</div></div>'
        f'<details><summary>查看原始记录</summary><pre>{text(json.dumps(raw, ensure_ascii=False, indent=2))}</pre></details></article>'
    )



def frame_reference(frames, step, label):
    step = step_number(step)
    frame = frames.get(step)
    if frame is None:
        return f'<p>{text(label)}：尚无对应观察</p>'
    observation = decode(frame['observation_json'], {})
    pose = observation.get('pose') or {}
    position = [pose.get('x'), pose.get('y')]
    if not observation.get('pose_valid', True) or not all(finite_number(value) for value in position):
        position = '未知'
    heading = frame.get('heading_deg')
    heading = f'{heading:.1f}°' if finite_number(heading) else '未知'
    return (
        f'<p><a href="#frame-{step}">{text(label)} · 第 {step} 步观察</a>'
        f' · {text(frame["place_id"])} · 坐标 {text(position)} · 朝向 {text(heading)}</p>'
    )


def trajectory_card(row, frames):
    step = row['action_step']
    result = decode(row['result_json'])
    review = decode(row['review_json'], {})
    arguments = decode(row['arguments_json'], {})
    action = ACTION_LABELS.get(row['action'], row['action'])
    body = f'<h3>第 {step} 步 · {text(action)}</h3><p><b>行动目的：</b>{text(row["intent"])}</p>'
    body += frame_reference(frames, step, '起点')
    body += frame_reference(frames, row['observed_step'], '终点')
    if result is None:
        description = '执行结果未返回，状态未知'
    elif result.get('startup_failed'):
        description = '导航服务启动失败，未执行接近'
    else:
        reason = result.get('reason')
        description = RESULT_LABELS.get(reason, reason or '未知结果')
    body += f'<div class="execution-result"><b>工具执行结果：</b>{text(description)}'
    displacement = (result or {}).get('motion', {}).get('displacement_m')
    if displacement is None:
        displacement = (result or {}).get('displacement_m')
    if finite_number(displacement):
        body += f' · 净位移 {displacement:.2f} m'
    points = ((result or {}).get('motion') or {}).get('sampled_positions')
    if points:
        body += f' · 实测轨迹 {len(points)} 点（按行走顺序）'
    body += '</div>'
    body += render_reviews(review.get('action'), review.get('target'))
    if not review.get('action') and not review.get('target'):
        body += '<p class="muted">尚无通过校验的视觉复核；动作停止不表示任务完成。</p>'
    body += details('查看动作参数及原始结果', {
        'action_step': step, 'action': row['action'], 'intent': row['intent'],
        'arguments': arguments, 'candidate_origin_id': row['candidate_origin_id'],
        'result': result, 'observed_step': row['observed_step'], 'review': review,
    })
    return f'<article class="memory-record" id="action-{step}">{body}</article>'


def write_memory_visualization(source):
    """只读 SQLite 和事件快照；兼容尚无轨迹表的旧任务。"""
    source = Path(source)
    database = source / 'memory.sqlite3' if source.is_dir() else source
    if not database.is_file():
        return None
    directory = database.parent
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        places = [dict(row) for row in db.execute('SELECT * FROM places ORDER BY rowid')]
        frames = [dict(row) for row in db.execute('SELECT * FROM semantic_frames ORDER BY step')]
        objects = [dict(row) for row in db.execute('SELECT * FROM objects ORDER BY frame_step, id')]
        has_trajectory = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trajectory_actions'"
        ).fetchone() is not None
        actions = [dict(row) for row in db.execute(
            'SELECT * FROM trajectory_actions ORDER BY action_step'
        )] if has_trajectory else []
    events, invalid = read_events(directory)
    inputs = [event for event in events if event.get('type') == 'decision_input']
    record_path = directory / 'result.json'
    record = json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else {}
    sections, navigation = [], []
    for place in places:
        place_frames = [frame for frame in frames if frame['place_id'] == place['id']]
        navigation.append(f'<a href="#{text(place["id"])}">{text(place["id"])} · {text(place["label"] or "未命名地点")}</a>')
        cards = ''.join(frame_card(frame, [item for item in objects if item['frame_step'] == frame['step']], directory)
                        for frame in place_frames)
        sections.append(
            f'<section class="place" id="{text(place["id"])}"><h3>{text(place["id"])} · {text(place["label"] or "未命名观察地点")}</h3>'
            f'<p class="muted">观察点坐标 {text(decode(place["position_json"]))} · {len(place_frames)} 帧'
            f' · 标签来源：第 {text(place["label_frame"])} 步</p>{cards}</section>'
        )
    observations = ''.join(sections) or '<p>暂无记忆：这个任务尚未记录观察。</p>'
    frames_by_step = {frame['step']: frame for frame in frames}
    timeline = ''.join(trajectory_card(row, frames_by_step) for row in actions)
    if not timeline:
        timeline = '<p class="muted">' + (
            '本任务尚无工具动作记录。' if has_trajectory else '旧版记录未保存轨迹表，不能还原完整动作经历。'
        ) + '</p>'
    input_cards = []
    for index, entry in enumerate(inputs):
        step = step_number(entry.get('step'))
        anchor = f'input-{step}' if step is not None else f'input-event-{index}'
        backlink = f'<a href="index.html#decision-input-{step}">查看本轮日志 →</a>' if step is not None else ''
        input_cards.append(
            f'<article class="memory-record" id="{anchor}"><h3>第 {text(step)} 轮 · 决策前快照</h3>'
            + backlink + render_memory_input(entry) + '</article>'
        )
    snapshots = ''.join(input_cards) or '<p class="muted">未记录决策输入快照，无法判断当时模型读到了什么。</p>'
    notice = f'<p class="notice">{invalid} 行事件不完整或无效，已跳过；原始记录未修改。</p>' if invalid else ''
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>任务记忆 · 观察与轨迹</title><style>' + STYLE + '</style><main>'
        '<header><div class="muted">AGENT MEMORY / 离线阅读页</div><h1>任务记忆</h1>'
        f'<p>任务：{text(record.get("goal", "未提供目标"))}</p>'
        '<p class="muted">观察记忆记录看到了什么，动作轨迹记录做过什么；决策快照记录当时模型拿到了什么。'
        '坐标属于观察者，场景和复核属于模型判断。</p>'
        '<nav aria-label="页面导航"><a href="index.html">← 返回任务日志</a>'
        '<a href="#trajectory">动作时间线</a><a href="#inputs">当轮读取的记忆</a>'
        '<a href="#observations">观察与空间记忆</a></nav></header>'
        f'<div class="stats"><div><b>{len(places)}</b>观察地点</div><div><b>{len(frames)}</b>观察帧</div>'
        f'<div><b>{len(actions)}</b>动作经历</div><div><b>{len(inputs)}</b>决策快照</div></div>'
        + notice +
        '<div class="toolbar"><label for="category">展示内容</label><select id="category">'
        '<option value="">全部记忆</option><option value="trajectory">动作时间线</option>'
        '<option value="inputs">当轮读取的记忆</option><option value="observations">观察与空间记忆</option>'
        '</select><label for="search">搜索记录</label>'
        '<input id="search" type="search" placeholder="例如：沙发、无新信息、P1">'
        '<button id="reset" type="button">清除筛选</button></div>'
        '<p id="empty" hidden>没有匹配的记录。</p>'
        '<section class="memory-section" id="trajectory"><h2>动作时间线</h2>'
        '<p class="muted">本地保存的完整经历；最后一次动作可能尚无终点观察或视觉复核。</p>'
        + timeline + '</section>'
        '<section class="memory-section" id="inputs"><h2>当轮读取的记忆</h2>'
        '<p class="muted">直接展示当轮决策输入，不将后来写入的复核补进过去的上下文。</p>'
        + snapshots + '</section>'
        '<section class="memory-section" id="observations"><h2>观察与空间记忆</h2>'
        f'<nav aria-label="地点目录">{"".join(navigation)}</nav>' + observations + '</section>'
        '<footer>数据来源：本任务 SQLite 与事件日志。历史目击不代表当前可见，导航停止不表示到达。'
        '图片使用相对路径，请保留 images 文件夹。</footer></main><script>' + SCRIPT + '</script></html>'
    )
    destination = directory / 'memory_visualize.html'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(page, encoding='utf-8')
    temporary.replace(destination)
    return destination


STYLE = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f7f4ed;color:#393832;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:40px 24px}h1,h2,h3{line-height:1.35}h1{font-family:Georgia,serif;font-size:36px;margin:8px 0}h2{font-size:24px}h3{font-size:18px}h3 span{font-size:12px;font-weight:400;color:#777}a{color:#796347}header,.place{margin-bottom:28px}.muted,small{color:#716f66}.stats{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}.stats div{background:#ece9de;border:1px solid #dcd5c5;border-radius:10px;padding:14px 28px}.stats b{font-size:28px;display:block;color:#746044}nav{display:flex;gap:10px;flex-wrap:wrap}nav a{background:#e9e5db;border-radius:20px;padding:5px 13px;text-decoration:none}input{width:100%;padding:14px;border:1px solid #cfc5b2;border-radius:8px;background:#fffdf7;font:inherit;margin:20px 0}section{scroll-margin-top:20px}article{background:#fffdf8;border:1px solid #ded7ca;border-radius:12px;padding:22px;margin:16px 0}.frame{display:grid;grid-template-columns:minmax(250px,42%) 1fr;gap:24px;align-items:start}img{display:block;width:100%;border-radius:8px}.missing{background:#e9e7de;text-align:center;padding:70px 12px;color:#777}dl{margin:0}dl>div{margin-bottom:10px}dt{font-size:12px;color:#797569}dd{margin:0;overflow-wrap:anywhere}ul{padding-left:22px}aside{background:#e9eee7;border-left:3px solid #9aab98;padding:12px 16px}aside p{margin:5px 0}details{margin-top:16px;border-top:1px solid #e6e0d5;padding-top:10px}summary{cursor:pointer;color:#796347}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:420px;overflow:auto}footer{margin-top:40px;color:#777;font-size:13px}[hidden]{display:none!important}@media(max-width:700px){.frame{grid-template-columns:1fr}main{padding:24px 16px}.stats div{padding:10px 18px}}

h4{font-size:15px;margin:18px 0 8px}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:16px;background:#ece9de;border-radius:10px}.toolbar input{flex:1;min-width:160px;width:auto;margin:0}select,button{font:inherit;background:#fffdf7;color:#393832;border:1px solid #cfc5b2;border-radius:7px;padding:10px}button{cursor:pointer}.memory-section{margin-top:36px}.execution-result{background:#ede8df;padding:12px 16px;border-radius:6px;margin:12px 0}.notice{background:#eee7d9;padding:14px;border-left:3px solid #a28c66}aside{margin:12px 0}li,p{overflow-wrap:anywhere}article{scroll-margin-top:20px}article:target{outline:2px solid #9b805a}.memory-input ul{margin-top:6px}
"""

SCRIPT = """
const search=document.getElementById('search'),category=document.getElementById('category');
function apply(){
 let visible=0;
 document.querySelectorAll('.memory-section').forEach(section=>{
  const selected=!category.value||category.value===section.id;
  const cards=[...section.querySelectorAll('.memory-record')];
  let count=0;
  cards.forEach(card=>{card.hidden=!card.textContent.toLocaleLowerCase().includes(search.value.trim().toLocaleLowerCase());if(!card.hidden)count++;});
  section.querySelectorAll('.place').forEach(place=>{place.hidden=![...place.querySelectorAll('.memory-record')].some(card=>!card.hidden);});
  section.hidden=!selected||(cards.length>0?count===0:!!search.value.trim());
  if(!section.hidden)visible++;
 });
 document.getElementById('empty').hidden=visible>0;
}
function reset(){search.value='';category.value='';apply();}
function revealHash(){
 if(!location.hash)return;
 const target=document.getElementById(decodeURIComponent(location.hash.slice(1)));
 if(target){reset();target.scrollIntoView();}
}
search.addEventListener('input',apply);category.addEventListener('change',apply);
document.getElementById('reset').addEventListener('click',reset);
document.querySelectorAll('a[href^="#"]').forEach(link=>link.addEventListener('click',reset));
window.addEventListener('hashchange',revealHash);revealHash();
"""
