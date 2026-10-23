"""单次任务的动作经历：引用观察帧，区分执行结果与模型复核。"""

import json
import math

from navigation_agent.agent.memory.spatial import angle_distance, encode


ACTION_LABELS = {'navigate': '导航', 'look': '观察', 'orient': '转回', 'recover': '脱困'}
RESULT_LABELS = {
    'agent_interrupt': '上层发现线索，停车复核',
    'model_stop': '底层停止，到达未确认', 'timeout': '超时', 'blocked': '受阻',
    'path_repeated': '同一指令绕回旧点，已停车待复核',
    'no_progress': '执行无进展', 'navigation_error': '导航服务错误',
    'turned': '已转向', 'oriented': '已转回', 'recovered': '已退回',
    'recovery_unavailable': '无法沿原路退回',
}
REVIEW_LABELS = {'progress': '有进展', 'no_new_information': '无新信息', 'uncertain': '进展不确定'}
TARGET_LABELS = {
    'visible': '目标当前可见', 'arrived': '当前证据支持完成',
    'temporarily_lost': '目标暂时不可见', 'rejected': '原目标线索已否定',
}


def compact(text, limit=64):
    value = ' '.join(str(text or '').split())
    return value if len(value) <= limit else value[:limit] + '…'


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_position(point):
    return (isinstance(point, (list, tuple)) and len(point) == 2
            and all(finite_number(value) for value in point))


class TrajectoryMemory:
    def __init__(self, spatial):
        self.spatial = spatial
        self.settings = spatial.settings
        with spatial.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS trajectory_actions (
                    action_step INTEGER PRIMARY KEY REFERENCES semantic_frames(step),
                    action TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    candidate_origin_id TEXT,
                    result_json TEXT,
                    observed_step INTEGER REFERENCES semantic_frames(step),
                    review_json TEXT
                )
            """)

    def start(self, step, action, intent, arguments, candidate=None):
        """只在动作通过校验并即将执行时写入；无返回结果保持未知。"""
        origin = None
        if candidate:
            origin = candidate.get('origin_candidate_id', candidate['id'])
        with self.spatial.connect() as db:
            db.execute(
                """INSERT INTO trajectory_actions
                   (action_step, action, intent, arguments_json, candidate_origin_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (step, action, intent, encode(arguments), origin),
            )

    def result(self, step, result):
        with self.spatial.connect() as db:
            db.execute('UPDATE trajectory_actions SET result_json = ? WHERE action_step = ?',
                       (encode(result), step))

    def observe(self, step):
        """新观察补齐上个动作的终点，不推算不存在的最后一帧。"""
        with self.spatial.connect() as db:
            db.execute(
                """UPDATE trajectory_actions SET observed_step = ?
                   WHERE action_step = (
                       SELECT max(action_step) FROM trajectory_actions WHERE action_step < ?
                   ) AND observed_step IS NULL""",
                (step, step),
            )

    def review(self, step, action_review=None, target_review=None):
        if action_review is None and target_review is None:
            return
        report = {'action': action_review, 'target': target_review}
        with self.spatial.connect() as db:
            db.execute('UPDATE trajectory_actions SET review_json = ? WHERE observed_step = ?',
                       (encode(report), step))

    def _read(self, before_step):
        with self.spatial.connect() as db:
            rows = db.execute(
                'SELECT * FROM trajectory_actions WHERE action_step < ? ORDER BY action_step',
                (before_step,),
            ).fetchall()
        steps = [row['action_step'] for row in rows]
        steps.extend(row['observed_step'] for row in rows if row['observed_step'] is not None)
        frames = {frame['step']: frame for frame in self.spatial.frames(steps)}
        entries = []
        for row in rows:
            entries.append({
                'step': row['action_step'], 'action': row['action'], 'intent': row['intent'],
                'arguments': json.loads(row['arguments_json']),
                'origin': row['candidate_origin_id'],
                'result': json.loads(row['result_json']) if row['result_json'] else None,
                'review': json.loads(row['review_json']) if row['review_json'] else {},
                'start': frames[row['action_step']], 'end': frames.get(row['observed_step']),
            })
        return entries

    def _summary(self, entry):
        start, end = entry['start'], entry['end']
        span = f"{entry['step']}→{end['step']}" if end else f"{entry['step']}→待观察"
        action = ACTION_LABELS.get(entry['action'], entry['action'])
        direction = {'left': '左', 'right': '右'}.get(entry['arguments'].get('direction'), '')
        parts = [f"{span} {action}{direction}：{compact(entry['intent'])}"]
        if end:
            parts.append(f"{start['place_id']}→{end['place_id']}")
        result = entry['result']
        if result is None:
            parts.append('执行结果未返回')
        elif result.get('startup_failed'):
            parts.append('启动失败，未执行接近')
        else:
            parts.append(RESULT_LABELS.get(result['reason'], compact(result['reason'], 24)))
            motion = result.get('motion', {})
            displacement = motion.get('displacement_m')
            path_length = motion.get('path_length_m')
            if entry['action'] == 'navigate' and finite_number(path_length):
                parts.append(f'实际行程{path_length:.2f}m')
                if (path_length >= 4. and finite_number(displacement)
                        and path_length >= 3 * max(displacement, .1)):
                    parts.append('行程明显大于净位移，有绕行或折返，须核对是否真正穿过入口')
            if finite_number(displacement):
                parts.append(f'执行净位移{displacement:.2f}m')
            elif end and start['observer_position'] is not None and end['observer_position'] is not None:
                distance = math.dist(start['observer_position'], end['observer_position'])
                parts.append(f'观测位移{distance:.2f}m')
        review = entry['review'].get('action')
        target = entry['review'].get('target')
        if review:
            parts.append('模型复核：' + REVIEW_LABELS[review['outcome']] + '，' + compact(review['evidence']))
        if target:
            parts.append('目标复核：' + TARGET_LABELS[target['status']] + '，' + compact(target['evidence']))
        if not review and not target:
            parts.append('尚未视觉复核')
        return '；'.join(parts)

    def _near_view(self, first, second):
        if first is None or second is None:
            return False
        a, b = first['observer_position'], second['observer_position']
        headings = [first['heading_deg'], second['heading_deg']]
        return (
            a is not None and b is not None and all(finite_number(value) for value in headings)
            and math.dist(a, b) <= self.settings.place_radius_m
            and angle_distance(*headings) <= self.settings.same_view_angle_deg
        )

    def _same_attempt(self, first, second):
        same_intent = (
            first['origin'] is not None and first['origin'] == second['origin']
        ) or first['intent'] == second['intent']
        return same_intent and self._near_view(first['start'], second['start'])

    def _repetitions(self, entries):
        groups = []
        last_navigation = None
        for entry in entries:
            target = entry['review'].get('target') or {}
            # 接近后的局部转向也可能否定线索或确认到达，解除对应导航的旧提醒。
            if entry['action'] != 'navigate':
                if last_navigation and target.get('status') in ('arrived', 'rejected'):
                    groups = [group for group in groups
                              if not self._same_attempt(group[0], last_navigation)]
                continue

            last_navigation = entry
            result = entry['result']
            if result is None or result.get('startup_failed') or entry['end'] is None:
                continue
            matched = next((group for group in groups if self._same_attempt(group[0], entry)), None)
            review = entry['review'].get('action') or {}
            resolved = review.get('outcome') == 'progress' or target.get('status') in ('arrived', 'rejected')
            if resolved:
                if matched is not None:
                    groups.remove(matched)
                continue
            if matched is None:
                groups.append([entry])
            else:
                matched.append(entry)
        hints = []
        for group in sorted(groups, key=lambda items: items[-1]['step'], reverse=True):
            if len(group) < 2:
                continue
            no_information = sum(
                (entry['review'].get('action') or {}).get('outcome') == 'no_new_information'
                for entry in group
            )
            failures = sum(entry['result']['reason'] in ('blocked', 'no_progress', 'path_repeated', 'timeout', 'navigation_error')
                           for entry in group)
            if not no_information and not failures:
                continue
            last = group[-1]
            evidence = f'{no_information}次模型报告无新信息，{failures}次执行受阻/超时/错误'
            hints.append(
                f"{last['start']['place_id']}附近相近朝向，重复尝试“{compact(last['intent'], 40)}”"
                f"{len(group)}次（步骤{group[0]['step']}…{last['step']}）；{evidence}。"
                '优先复核原因或选择不同观察位置；不是禁止返回。'
            )
            if len(hints) >= self.settings.max_trajectory_hints:
                break
        return hints

    def _returns(self, entries):
        completed = [entry for entry in entries
                     if entry['action'] == 'navigate' and entry['end'] is not None
                     and entry['start']['observer_position'] is not None
                     and entry['end']['observer_position'] is not None
                     and not (entry['result'] or {}).get('startup_failed')]
        if not completed:
            return []
        # 途中经过的位置也属于真实轨迹；P 编号的聚类边界不能决定是否重访。
        with self.spatial.connect() as db:
            steps = [row['step'] for row in db.execute(
                'SELECT step FROM semantic_frames WHERE step < ? ORDER BY step',
                (completed[-1]['step'],),
            )]
        frames = self.spatial.frames(steps)
        hints = []
        for entry in completed:
            start, end = entry['start'], entry['end']
            if math.dist(start['observer_position'], end['observer_position']) < self.settings.place_radius_m:
                continue
            matches = [(frame, math.dist(frame['observer_position'], end['observer_position']))
                       for frame in frames if frame['step'] < entry['step']
                       and frame['observer_position'] is not None]
            matches = [(frame, distance) for frame, distance in matches
                       if distance <= self.settings.place_radius_m]
            if not matches:
                continue
            previous, distance = min(matches, key=lambda item: (item[1], -item[0]['step']))
            hints.append(
                f"导航后 O{end['step']} 回到 O{previous['step']} 附近，位置相距{distance:.2f}m；"
                '这里曾经过，不代表已经搜完；核对是否在往返旧路线，不禁止正常途经。'
            )
        return hints[-self.settings.max_trajectory_hints:]

    def _local_paths(self, entries, current):
        """从完整历史召回附近实测路径，用本轮停车朝向重新计算方向。"""
        position = current['observer_position']
        heading = current['heading_deg']
        if position is None or not finite_number(heading):
            return None

        nearby = []
        for entry in entries:
            result = entry['result']
            if (entry['action'] != 'navigate' or not result or result.get('startup_failed')
                    or entry['end'] is None or entry['end']['step'] > current['step']):
                continue
            points = (result.get('motion') or {}).get('sampled_positions')
            if not isinstance(points, (list, tuple)):
                continue
            matches = [(math.dist(position, point), index)
                       for index, point in enumerate(points) if valid_position(point)]
            if not matches:
                continue
            # 同一段多次经过当前位置时，优先引用最后一次经过。
            distance, index = min(matches, key=lambda item: (item[0], -item[1]))
            if distance > self.settings.place_radius_m:
                continue
            directions = []
            for order, stride in (('toward_start', -1), ('toward_end', 1)):
                direction = self._path_direction(points, index, stride, position, heading)
                if direction is not None:
                    directions.append({'order': order, **direction})
            if not directions:
                continue
            segment = {
                'action_step': entry['step'],
                'start_observation': entry['start']['observation_id'],
                'end_observation': entry['end']['observation_id'],
                'stop_heading_deg': entry['end']['heading_deg'],
                'origin_candidate_id': entry['origin'],
                'intent': compact(entry['intent']),
                'distance_to_path_m': round(distance, 2),
                'directions': directions,
            }
            review = entry['review'].get('action')
            if review:
                segment['visual_review'] = {
                    'outcome': review['outcome'], 'evidence': compact(review['evidence']),
                }
            nearby.append(segment)

        nearby.sort(key=lambda item: (item['distance_to_path_m'], -item['action_step']))
        if not nearby:
            return None
        return {
            'observation_id': current['observation_id'],
            'heading_deg': heading,
            'segments': nearby[:self.settings.max_trajectory_hints],
        }

    def _path_direction(self, points, index, stride, position, heading):
        """只取邻接点的局部方向，采样缺失或大跨度时不拼接路线。"""
        previous = points[index]
        for next_index in range(index + stride, len(points) if stride > 0 else -1, stride):
            point = points[next_index]
            if not valid_position(point) or math.dist(previous, point) > 1.:
                break
            previous = point
            distance = math.dist(position, point)
            if distance < max(.5, self.settings.place_radius_m):
                continue
            bearing = math.degrees(math.atan2(point[1] - position[1], point[0] - position[0]))
            relative = (bearing - heading + 180) % 360 - 180
            labels = ('前方', '左前方', '左侧', '左后方', '后方', '右后方', '右侧', '右前方')
            return {
                'relative_bearing_deg': round(relative, 1),
                'direction': labels[int((relative + 22.5) % 360 // 45)],
                'sample_distance_m': round(distance, 2),
            }
        return None

    def navigation_frames(self, before_step):
        """停车后回看上次导航的两张途中原图；不在行进中调用视觉模型。"""
        with self.spatial.connect() as db:
            action = db.execute(
                """SELECT action_step, observed_step, result_json FROM trajectory_actions
                   WHERE action='navigate' AND observed_step <= ?
                   ORDER BY action_step DESC LIMIT 1""", (before_step,),
            ).fetchone()
            if action is None or not action['result_json']:
                return []
            if json.loads(action['result_json']).get('startup_failed'):
                return []
            steps = [row['step'] for row in db.execute(
                'SELECT step FROM semantic_frames WHERE step BETWEEN ? AND ? ORDER BY step',
                (action['action_step'], action['observed_step']),
            )]
        frames = [frame for frame in self.spatial.frames(steps)
                  if frame['observer_position'] is not None]
        distances = [0.]
        for first, second in zip(frames, frames[1:]):
            distances.append(distances[-1] + math.dist(
                first['observer_position'], second['observer_position'],
            ))
        if len(frames) < 3 or distances[-1] < 1.:
            return []
        interior = [(frame, distance) for frame, distance in zip(frames, distances)
                    if action['action_step'] < frame['step'] < action['observed_step']]
        if not interior:
            return []
        selected = {}
        for fraction in (1 / 3, 2 / 3):
            frame, distance = min(interior, key=lambda item: abs(item[1] - distances[-1] * fraction))
            selected[frame['step']] = frame
        return list(selected.values())

    def query(self, before_step):
        entries = self._read(before_step)
        context = {
            'recent': [self._summary(entry) for entry in entries[-self.settings.max_trajectory_actions:]],
            'repeated_attempts': self._repetitions(entries),
            'returned_places': self._returns(entries),
            'scope': '本次任务；坐标仅辅助同坐标系内的位置比较，不能推断房间连通。'
                     '模型复核不是传感器事实；重复提示不禁止正常返回，有新证据仍可继续。',
        }

        frames = self.spatial.frames([before_step])
        if frames:
            local_paths = self._local_paths(entries, frames[0])
            if local_paths:
                context['local_paths'] = local_paths
        return context
