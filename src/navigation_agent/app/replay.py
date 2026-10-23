"""Go2 与 MuJoCo 共用的回放时间轴和字幕。"""

from bisect import bisect_right
from html import escape
from datetime import datetime
import json
import math
from pathlib import Path
import re

from navigation_agent.agent.storage.report import NAMES, ACTIONS, OUTCOMES, key_events

WEB_DIR = Path(__file__).resolve().parent / 'web/replay'


def seconds(value):
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return None


def rows(value):
    result = []
    for line in value.splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                result.append(row)
        except ValueError:
            pass
    return result


def position(pose):
    if not isinstance(pose, dict):
        return None
    values = [pose.get(key) for key in ('x', 'y', 'yaw')]
    return values if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) else None


def build_timeline(task, events, samples, recording, frames):
    start = seconds(task['started_at'])
    end = seconds(task.get('ended_at'))
    duration = max(0, (end or start) - start)
    observations = {}
    is_vln = task.get('mode') == 'vln'
    summaries = {} if is_vln else {(node['type'], node['time'], node['step']): node for node in key_events(events)}
    completion_rejections = set()
    images, nodes, trajectory = [], [], []
    wanted = {'started', 'observation', 'decision', 'action_input', 'action_result', 'finished',
              'decision_error', 'navigation_interrupt_requested', 'completion_check', 'monitor_report', 'observation_gain',
              'target_detection_hit', 'target_detection_review', 'target_watch_degraded', 'decision_replan'}
    for entry in events:
        stamp = seconds(entry.get('time'))
        if stamp is None:
            continue
        t = max(0, stamp - start)
        duration = max(duration, t)
        kind, step = entry.get('type'), entry.get('step')
        image = entry.get('image')
        if kind in ('observation', 'passive_observation'):
            captured = (entry.get('observation') or {}).get('camera_stamp_ns')
            captured = captured / 1e9 - start if isinstance(captured, (int, float)) else t
            captured = max(0, captured)
            observations[step] = captured
            if re.fullmatch(r'images/\d+\.jpg', image or ''):
                images.append({'t': captured, 'image': image})
            if kind == 'observation':
                t = captured
        signature = (kind, entry.get('time'), step)
        if (kind not in wanted and signature not in summaries and not (is_vln and entry.get('key'))) or (kind == 'monitor_report' and entry.get('event') == 'none'):
            continue
        if kind == 'observation_gain' and entry.get('target_seen') != 'yes':
            continue
        decision = entry.get('decision') or {}
        result = entry.get('result') or {}
        description = (decision.get('reason') or entry.get('evidence') or entry.get('summary')
                       or entry.get('instruction') or entry.get('goal') or entry.get('reason') or '')
        if kind == 'decision':
            description = ACTIONS.get(decision.get('action'), decision.get('action', '')) + ' · ' + description
        if kind == 'action_result':
            description = OUTCOMES.get(result.get('reason'), result.get('reason', '动作结束'))
        if kind == 'action_input':
            description = ACTIONS.get(entry.get('action'), entry.get('action', '')) + ' · ' + description
        label = entry.get('label', NAMES.get(kind, kind)) if is_vln else NAMES.get(kind, kind)
        if kind == 'decision_error':
            if entry.get('category') == 'completion_verification':
                # 同一次未通过核验已展示原因，不再重复画成红色异常节点。
                if step is not None and step in completion_rejections:
                    continue
                kind = 'completion_check'
                label = '尚未满足完成条件'
                description = '本次完成判断未通过核验，需要继续观察或接近；后续处理见下一决策。'
            else:
                label = {
                    'output_format': '输出格式错误',
                    'scene_rule': '动作未通过场景检查',
                    'exploration_rule': '动作需要调整',
                }.get(entry.get('category'), label)
                description = '本次动作未执行；具体原因见原始日志。'
        elif kind == 'completion_check':
            label = '完成核验通过' if entry.get('confirmed') else '尚未满足完成条件'
            if entry.get('confirmed') is False and step is not None:
                completion_rejections.add(step)
        if kind == 'observation_gain' or (kind == 'monitor_report' and entry.get('event') == 'target_seen'):
            label = '目标线索（待复核）'
        signature = (entry.get('type'), entry.get('time'), entry.get('step'))
        summary = summaries.pop(signature, None)
        if summary:
            label, description = summary['label'], summary['description']
        nodes.append({'t': t, 'type': kind, 'label': label, 'step': step, 'key': bool(entry.get('key')) if is_vln else summary is not None,
                      'description': description, 'observation_t': observations.get(step),
                      'decision': decision if kind == 'decision' else None,
                      'frames': summary.get('frames', []) if summary else [],
                      'outcome': summary.get('outcome', '') if summary else '',
                      'instruction': summary.get('instruction', '') if summary else ''})
    for sample in samples:
        pose = position(sample.get('pose'))
        stamp = sample.get('time_ns')
        if pose and isinstance(stamp, (int, float)):
            trajectory.append({'t': max(0, stamp / 1e9 - start), 'pose': pose})
    mapped = []
    for frame in frames:
        stamp, video_s = frame.get('stamp_ns'), frame.get('video_s')
        if not isinstance(stamp, (int, float)) or not isinstance(video_s, (int, float)):
            continue
        t = max(0, stamp / 1e9 - start)
        mapped.append({'t': t, 'v': video_s})
        if isinstance(frame.get('pose'), list) and len(frame['pose']) == 3:
            pose = position(dict(zip(('x', 'y', 'yaw'), frame['pose'])))
            if pose:
                trajectory.append({'t': t, 'pose': pose})
    duration = max(duration, max((row['t'] for row in mapped), default=0))
    return {'task': {key: task.get(key) for key in ('id', 'mode', 'goal', 'status', 'started_at', 'summary')},
            'duration': duration, 'nodes': sorted(nodes, key=lambda r: r['t']),
            'images': sorted(images, key=lambda r: r['t']),
            'trajectory': sorted(trajectory, key=lambda r: r['t']),
            'recording': recording, 'frames': sorted(mapped, key=lambda r: r['t'])}


def event_subtitles(data):
    """字幕使用视频时间，不能把任务时间直接写入 MP4 的字幕轨。"""
    frames = data['frames']
    if not frames:
        return 'WEBVTT\n\n'
    times = [frame['t'] for frame in frames]

    def video_time(t):
        index = max(0, bisect_right(times, t) - 1)
        frame = frames[index]
        if index + 1 < len(frames):
            after = frames[index + 1]
            gap = after['t'] - frame['t']
            if 0 < gap <= .5:
                return frame['v'] + max(0, t - frame['t']) / gap * (after['v'] - frame['v'])
        return frame['v']

    def stamp(value):
        milliseconds = max(0, round(value * 1000))
        return (f'{milliseconds // 3600000:02d}:{milliseconds // 60000 % 60:02d}:'
                f'{milliseconds // 1000 % 60:02d}.{milliseconds % 1000:03d}')

    subtitle_types = {'decision', 'action_input', 'finished', 'completion_check'}
    if data.get('task', {}).get('mode') == 'vln':
        subtitle_types.update(('started', 'vln_prediction', 'vln_error'))
    nodes = [node for node in data['nodes'] if node['type'] in subtitle_types]
    cues = ['WEBVTT\n']
    for index, node in enumerate(nodes):
        start = video_time(node['t'])
        end = video_time(nodes[index + 1]['t']) if index + 1 < len(nodes) else start + 3
        end = min(data['recording']['video_duration_s'], end)
        if end <= start:
            continue
        caption = escape(node['label'] + ' · ' + node['description']).replace('\n', ' ')
        cues.append(f'{stamp(start)} --> {stamp(end)}\n{caption}\n')
    return '\n'.join(cues)
