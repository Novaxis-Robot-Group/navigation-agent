"""记忆页与日志页共用的只读摘要；只展示记录中的内容，不重新推导历史。"""

from html import escape
import json
from pathlib import Path
import re

from navigation_agent.agent.memory.trajectory import REVIEW_LABELS, TARGET_LABELS


def text(value):
    return escape(str(value if value is not None else '—'), quote=True)


def details(label, value):
    body = json.dumps(value, ensure_ascii=False, indent=2)
    return f'<details><summary>{text(label)}</summary><pre>{text(body)}</pre></details>'


def read_events(directory):
    events, invalid = [], 0
    path = Path(directory) / 'events.jsonl'
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError('事件应为对象')
                events.append(entry)
            except ValueError:
                invalid += 1
    return events, invalid


def step_number(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def image_link(path, label):
    if isinstance(path, str) and re.fullmatch(r'images/\d+\.jpg', path):
        return f'<a href="{text(path)}">{text(label)}</a>'
    return text(label)


def list_html(items, empty):
    if not items:
        return f'<p class="muted">{text(empty)}</p>'
    return '<ul>' + ''.join(f'<li>{text(item)}</li>' for item in items) + '</ul>'


def render_reviews(action_review=None, target_review=None):
    parts = []
    if action_review:
        outcome = REVIEW_LABELS.get(action_review.get('outcome'), action_review.get('outcome'))
        parts.append(f'<aside><b>模型进展复核 · {text(outcome)}</b>'
                     f'<p>{text(action_review.get("evidence"))}</p></aside>')
    if target_review:
        status = TARGET_LABELS.get(target_review.get('status'), target_review.get('status'))
        step = target_review.get('observation_step')
        parts.append(f'<aside><b>模型目标复核 · {text(status)}</b>'
                     f'<p>观察步：{text(step)} · {text(target_review.get("evidence"))}</p></aside>')
    return ''.join(parts)


def render_memory_input(entry):
    """直接读取当轮 decision_input，避免把后来写入的复核当成当时已知。"""
    context = entry.get('model_context') or {}
    execution = context.get('execution') or {}
    scan = execution.get('scan') or {}
    if scan.get('recovery_required'):
        return '<p class="notice">程序强制脱困，本轮未调用模型，也未读取语义记忆。</p>'
    if not context:
        return '<p class="muted">这条旧日志未记录模型记忆上下文。</p>'
    spatial = context.get('spatial_memory')
    trajectory = context.get('trajectory_memory')
    target = execution.get('target_tracking')
    session = execution.get('exploration_session') or {}
    phases = {
        'observing': '收集观察', 'selecting': '比较方向', 'aligning': '转回方向',
        'confirming': '重新确认路径', 'exploring': '接近探索位置',
        'target_confirmation': '目标局部确认', 'approaching': '接近目标',
        'verifying_arrival': '复核到达', 'temporarily_lost': '目标暂时不可见', 'arrived': '已确认到达',
    }
    phase = phases.get(session.get('phase'), session.get('phase', '未记录'))
    parts = [f'<div class="memory-input"><p><b>本轮阶段：</b>{text(phase)}'
             f' · 已观察转向 {text(scan.get("look_count"))} 次'
             f' · 当前位置 {text((spatial or {}).get("current_place_id"))}</p>']
    if target:
        phase = phases.get(target.get('phase'), target.get('phase'))
        parts.append(f'<p><b>跟进目标：</b>{text(target.get("description"))} · {text(phase)}'
                     f' · 已进行局部确认 {text(target.get("confirmation_turns"))} 次</p>')
    subgoal = execution.get('active_subgoal')
    if subgoal:
        parts.append(f'<p><b>当前探索目的：</b>{text(subgoal.get("description"))}'
                     f' · 待确认：{text(subgoal.get("question"))}</p>')
    topology = context.get('exploration_memory')
    if topology:
        previous_region = topology.get('previous_region', topology.get('current_region'))
        parts.append(f'<h4>区域与待探索路线 · 上次区域 {text(previous_region)}</h4>')
        if topology.get('recent_region_path'):
            parts.append('<p><b>最近经过：</b>' + text(' → '.join(topology['recent_region_path'])) + '</p>')
        frontier = topology.get('next_frontier')
        if frontier:
            parts.append('<p><b>尚未尝试入口：</b>' + text(frontier['id']) + ' · '
                         + text(frontier['region_id']) + ' · ' + text(frontier['description']) + '</p>')
        routes = topology.get('routes') or []
        candidates = {item['id']: item for key in ('local_candidates', 'unfinished_elsewhere')
                      for item in execution.get(key, [])}
        route_labels = []
        for route in routes:
            candidate = candidates.get(route.get('id'), {})
            description = route.get('description') or candidate.get('description', route.get('id'))
            route_labels.append(f'{route.get("region_id")} · {description} · {route.get("exploration_status")}')
        parts.append(list_html(route_labels, '当前没有登记的未解决路线。'))
        parts.append(details('区域判断、已走连接与证据来源', topology))
    reports = context.get('navigation_observations')
    if reports:
        parts.append(details('最近行进观察（停车后须复核）', reports))
    parts.append('<h4>近期动作摘要 · 当轮读取</h4>')
    if trajectory is None:
        parts.append('<p class="muted">此日志未记录轨迹记忆输入。</p>')
    else:
        parts.append(list_html(trajectory.get('recent'), '当轮尚无历史动作。'))
        parts.append('<h4>重复尝试提醒 · 当轮读取</h4>')
        parts.append(list_html(trajectory.get('repeated_attempts'), '当轮无重复尝试提醒。'))
        if trajectory.get('local_paths'):
            parts.append(details('附近已走路径 · 相对本轮停车朝向', trajectory['local_paths']))
    parts.append('<h4>空间证据与环视引用</h4>')
    if spatial is None:
        parts.append('<p class="muted">此日志未记录空间记忆输入。</p>')
    else:
        frames = spatial.get('semantic_frames') or []
        frame_labels = [f'第 {frame.get("step", "—")} 步 · {frame.get("place_id", "—")}'
                        for frame in frames]
        parts.append(list_html(frame_labels, '当轮无检索到的历史观察帧。'))
    views = session.get('views') or []
    parts.append(list_html(
        [f'{view.get("id", "—")} · 第 {view.get("step", "—")} 步方向图' for view in views],
        '当轮无环视方向引用。',
    ))
    parts.append('<h4>本轮实际附图</h4>')
    images = entry.get('images')
    if images is None:
        parts.append('<p class="muted">旧日志未保存完整请求图片清单，请查看日志中的观察图片。</p>')
    elif not images:
        parts.append('<p class="muted">记录的附图清单为空。</p>')
    else:
        parts.append('<ul>' + ''.join(
            '<li>' + image_link(frame.get('image'), frame.get('image') or '未知图片')
            + ' · ' + text(' / '.join(use.get('purpose', '') for use in frame.get('uses', []))) + '</li>'
            for frame in images
        ) + '</ul>')
    parts.append(details('查看当轮完整记忆与技能输入', {
        'observation_step': context.get('observation_step'),
        'spatial_memory': spatial, 'trajectory_memory': trajectory, 'execution': execution,
        'exploration_memory': context.get('exploration_memory'),
        'navigation_observations': context.get('navigation_observations'),
    }))
    parts.append('</div>')
    return ''.join(parts)
