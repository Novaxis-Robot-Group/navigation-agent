"""单次 run 的 SQLite 空间记忆：观察地点、语义帧和物体目击。"""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
import math
import sqlite3

from navigation_agent.agent.config import MemorySettings, ObservationSettings


def angle_distance(a, b):
    return abs((a - b + 180) % 360 - 180)


def encode(value):
    return json.dumps(value, ensure_ascii=False)


class SpatialMemory:
    def __init__(self, path, goal, *, settings: MemorySettings, observation: ObservationSettings):
        self.settings = settings
        self.same_view_angle_deg = settings.same_view_angle_deg
        self.path = path
        self.goal = goal
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS places (
                    id TEXT PRIMARY KEY,
                    position_json TEXT,
                    label TEXT NOT NULL DEFAULT '',
                    label_frame INTEGER,
                    last_step INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS semantic_frames (
                    step INTEGER PRIMARY KEY,
                    place_id TEXT NOT NULL REFERENCES places(id),
                    observed_at TEXT NOT NULL,
                    image TEXT NOT NULL,
                    heading_deg REAL,
                    observation_json TEXT NOT NULL,
                    scene_json TEXT,
                    report_status TEXT NOT NULL DEFAULT 'accepted',
                    previous_action_step INTEGER,
                    review_json TEXT
                );
                CREATE TABLE IF NOT EXISTS objects (
                    id TEXT PRIMARY KEY,
                    frame_step INTEGER NOT NULL REFERENCES semantic_frames(step),
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    bearing_deg REAL
                );
                CREATE INDEX IF NOT EXISTS objects_name ON objects(name);
                CREATE INDEX IF NOT EXISTS frames_place ON semantic_frames(place_id, step);
            ''')
            # 旧任务数据库可继续读取；来源标记不改写历史观察内容。
            columns = {row['name'] for row in db.execute('PRAGMA table_info(semantic_frames)')}
            if 'report_status' not in columns:
                db.execute("ALTER TABLE semantic_frames ADD COLUMN report_status TEXT NOT NULL DEFAULT 'accepted'")

    @contextmanager
    def connect(self):
        # 每次读写后关闭连接，Windows 收尾重命名 run 目录时不会占用文件。
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA foreign_keys = ON')
            with db:
                yield db

    def observe(self, observation, image, step):
        """匹配观察地点、保存原始帧并查询参考图。"""
        pose = observation.get('pose', {})
        located = observation.get('pose_valid', True) and all(
            isinstance(pose.get(axis), (int, float)) and math.isfinite(pose[axis])
            for axis in ('x', 'y')
        )
        position = [float(pose['x']), float(pose['y'])] if located else None
        yaw = observation.get('yaw', pose.get('yaw'))
        heading = math.degrees(yaw) if isinstance(yaw, (int, float)) and math.isfinite(yaw) else None
        with self.connect() as db:
            places = db.execute('SELECT id, position_json FROM places ORDER BY rowid').fetchall()
            nearby = []
            for row in places:
                anchor = json.loads(row['position_json'])
                if position is not None and anchor is not None:
                    distance = math.dist(position, anchor)
                    if distance <= self.settings.place_radius_m:
                        nearby.append((distance, row['id'], anchor))
            if nearby:
                _, place_id, anchor = min(nearby, key=lambda item: item[0])
            else:
                place_id, anchor = f'P{len(places) + 1}', position
            matching = []
            if heading is not None:
                frames = db.execute(
                    'SELECT image, heading_deg FROM semantic_frames WHERE place_id = ? ORDER BY step',
                    (place_id,),
                ).fetchall()
                matching = [
                    frame for frame in frames if frame['heading_deg'] is not None
                    and angle_distance(frame['heading_deg'], heading) <= self.same_view_angle_deg
                ]
            reference = None
            reference_reason = None
            if matching:
                reference = min(matching, key=lambda frame: angle_distance(frame['heading_deg'], heading))['image']
                reference_reason = 'similar_view'
            else:
                previous = db.execute('SELECT image FROM semantic_frames ORDER BY step DESC LIMIT 1').fetchone()
                if previous:
                    reference, reference_reason = previous['image'], 'previous_action_start'
            db.execute(
                '''INSERT INTO places(id, position_json, last_step) VALUES (?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET last_step = excluded.last_step''',
                (place_id, encode(anchor), step),
            )
            db.execute(
                '''INSERT INTO semantic_frames
                   (step, place_id, observed_at, image, heading_deg, observation_json)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (
                    step, place_id, datetime.now(timezone.utc).isoformat(),
                    image, heading, encode(observation),
                ),
            )
            count = db.execute('SELECT count(*) FROM places').fetchone()[0]
        return {
            'id': place_id, 'position': anchor, 'observer_position': position,
            'step': step, 'observation_id': f'O{step}', 'heading_deg': heading, 'image': image,
            'reference_image': reference, 'reference_reason': reference_reason,
            'same_view_previous_observations': len(matching), 'place_count': count,
        }

    def describe(self, place_id, step, decision, *, action_accepted=True, report_status=None, previous_action_step=None):
        scene = decision.scene.model_dump()
        objects = scene.pop('objects')
        review = decision.previous_action_review if action_accepted else None
        with self.connect() as db:
            # 没有新场景描述的纠错只更新复核，不抹去同帧已保存的视觉线索。
            if decision.scene.model_fields_set:
                db.execute(
                    'UPDATE semantic_frames SET scene_json = ?, report_status = ? WHERE step = ?',
                    (encode(scene), report_status or ('accepted' if action_accepted else 'rejected_action'), step),
                )
                if scene['place_label']:
                    db.execute(
                        'UPDATE places SET label = ?, label_frame = ? WHERE id = ?',
                        (scene['place_label'], step, place_id),
                    )
                db.execute('DELETE FROM objects WHERE frame_step = ?', (step,))
                for index, item in enumerate(objects, 1):
                    # 同名物体保留独立目击；同帧纠错替换报告，不重复追加物体。
                    db.execute(
                        '''INSERT INTO objects(id, frame_step, name, description, bearing_deg)
                           VALUES (?, ?, ?, ?, ?)''',
                        (f'O{step}_{index}', step, item['name'], item['description'], item['bearing_deg']),
                    )
            if review is not None and step > 1:
                db.execute(
                    'UPDATE semantic_frames SET previous_action_step = ?, review_json = ? WHERE step = ?',
                    (previous_action_step if previous_action_step is not None else step - 1, encode(review.model_dump()), step),
                )

    def frames(self, steps):
        """按固定步数读取观察证据，供环视与目标跟进共同引用。"""
        selected = list(dict.fromkeys(steps))
        if not selected:
            return []
        with self.connect() as db:
            placeholders = ','.join('?' for _ in selected)
            rows = db.execute(
                f'SELECT * FROM semantic_frames WHERE step IN ({placeholders})', selected,
            ).fetchall()
            objects = db.execute(
                f'SELECT * FROM objects WHERE frame_step IN ({placeholders})', selected,
            ).fetchall()
        records = {}
        for row in rows:
            observation = json.loads(row['observation_json'])
            pose = observation.get('pose', {})
            position = [pose.get('x'), pose.get('y')]
            if not observation.get('pose_valid', True) or not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in position
            ):
                position = None
            scene = json.loads(row['scene_json']) if row['scene_json'] else None
            if scene is not None:
                scene['objects'] = [
                    {'name': item['name'], 'description': item['description'], 'bearing_deg': item['bearing_deg']}
                    for item in objects if item['frame_step'] == row['step']
                ]
            records[row['step']] = {
                'step': row['step'], 'observation_id': f'O{row["step"]}',
                'place_id': row['place_id'], 'image': row['image'],
                'heading_deg': row['heading_deg'], 'observer_position': position,
                'scene': scene,
                'source': 'model_visual_report', 'action_status': row['report_status'],
            }
        return [records[step] for step in selected if step in records]

    def query(self, place):
        """按当前目标和位置读取少量历史证据，不再组合其他记忆系统。"""
        step = place['step']
        with self.connect() as db:
            frames = db.execute(
                '''SELECT f.*, p.position_json FROM semantic_frames f
                   JOIN places p ON p.id = f.place_id
                   WHERE f.step < ? AND f.scene_json IS NOT NULL ORDER BY f.step DESC''',
                (step,),
            ).fetchall()
            sightings = db.execute(
                '''SELECT o.*, f.place_id, f.image, f.heading_deg, f.observation_json, f.report_status, p.position_json
                   FROM objects o JOIN semantic_frames f ON f.step = o.frame_step
                   JOIN places p ON p.id = f.place_id
                   WHERE f.step < ? ORDER BY f.step DESC''',
                (step,),
            ).fetchall()
            places = db.execute('SELECT * FROM places ORDER BY last_step DESC').fetchall()

        def distance(row):
            position = json.loads(row['position_json'])
            if position is None or place['position'] is None:
                return None
            return math.dist(position, place['position'])

        def rank(row, relevant=False):
            separation = distance(row)
            nearby = separation is not None and separation <= self.settings.nearby_radius_m
            return (relevant, row['place_id'] == place['id'], nearby)

        def goal_match(text):
            # MVP 只做名称包含匹配，不把它伪装成向量语义检索。
            return bool(text) and (text in self.goal or self.goal in text)

        def observer_position(row):
            observation = json.loads(row['observation_json'])
            pose = observation.get('pose', {})
            if not observation.get('pose_valid', True):
                return None
            values = [pose.get('x'), pose.get('y')]
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                return None
            return values

        goal_frames = {item['frame_step'] for item in sightings if goal_match(item['name'])}
        # 已报告目标的帧不依赖中英文物体名称的字符串匹配。
        goal_frames.update(row['step'] for row in frames
                           if json.loads(row['scene_json']).get('target_seen') == 'yes')
        frames = sorted(
            frames,
            key=lambda row: rank(
                row,
                row['step'] in goal_frames,
            ),
            reverse=True,
        )
        selected_frames = []
        selected_views = set()
        for frame in frames:
            # 优先保留不同地点和朝向的画面，避免同一视野挤满上下文。
            heading = frame['heading_deg']
            view_key = (frame['place_id'], round(heading / self.settings.heading_bin_deg) if heading is not None else None)
            if view_key in selected_views:
                continue
            selected_views.add(view_key)
            selected_frames.append({
                'step': frame['step'], 'observation_id': f'O{frame["step"]}', 'place_id': frame['place_id'],
                'image': frame['image'], 'heading_deg': heading,
                'observer_position': observer_position(frame),
                'scene': json.loads(frame['scene_json']),
                'source': 'model_visual_report', 'action_status': frame['report_status'],
            })
            if len(selected_frames) == self.settings.max_semantic_frames:
                break

        sightings = sorted(sightings, key=lambda row: rank(row, row['frame_step'] in goal_frames), reverse=True)
        objects = []
        seen_names = set()
        for item in sightings:
            key = (item['place_id'], item['name'])
            if key in seen_names:
                continue
            seen_names.add(key)
            objects.append({
                'id': item['id'], 'name': item['name'], 'description': item['description'],
                'seen_from_place': item['place_id'],
                'seen_from_position': observer_position(item),
                'observer_heading_deg': item['heading_deg'], 'bearing_deg': item['bearing_deg'],
                'frame_step': item['frame_step'], 'image': item['image'],
                'source': 'model_visual_report', 'action_status': item['report_status'],
            })
            if len(objects) == self.settings.max_objects:
                break

        reviews = []
        for frame in sorted(frames, key=lambda row: row['step'], reverse=True):
            if frame['review_json']:
                reviews.append({
                    'action_step': frame['previous_action_step'], 'observed_step': frame['step'],
                    'image': frame['image'], **json.loads(frame['review_json']),
                    'source': 'model_visual_review',
                })
            if len(reviews) == self.settings.max_reviews:
                break

        return {
            'scope': 'current_run',
            'current_place_id': place['id'],
            'reference_image': place['reference_image'],
            'reference_reason': place['reference_reason'],
            'same_view_previous_observations': place['same_view_previous_observations'],
            'places': [
                {
                    'id': row['id'], 'position': json.loads(row['position_json']),
                    'label_hypothesis': row['label'], 'label_frame': row['label_frame'],
                }
                for row in places[:self.settings.max_places]
            ],
            'objects': objects,
            'semantic_frames': selected_frames,
            'recent_reviews': reviews,
            'limits': '地点是观察点，标签是模型推测；物体坐标仅为观察者位置。同名不代表同一物体。'
            '历史目击不代表当前可见；未看到不等于不存在。相邻地点可能隔墙，不能据坐标直接导航。',
        }
