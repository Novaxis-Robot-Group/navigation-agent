"""有证据来源的区域与已走连接；位置接近不自动代表区域相同。"""

from collections import deque
import json
import math

from navigation_agent.agent.memory.spatial import angle_distance, encode
from navigation_agent.agent.schemas import SpatialUpdate


CLOSED_ROUTES = ('resolved', 'deferred', 'traversed', 'blocked_from_this_viewpoint')


class TopologyMemory:
    def __init__(self, spatial):
        self.spatial = spatial
        self.current_region = 'R1'
        with spatial.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS regions (
                    id TEXT PRIMARY KEY, label TEXT, source_step INTEGER, status TEXT
                );
                CREATE TABLE IF NOT EXISTS region_evidence (
                    step INTEGER PRIMARY KEY, region_id TEXT, report_json TEXT
                );
                CREATE TABLE IF NOT EXISTS region_connections (
                    action_step INTEGER PRIMARY KEY, source TEXT, destination TEXT,
                    start_step INTEGER, end_step INTEGER, evidence TEXT
                );
                CREATE TABLE IF NOT EXISTS exploration_routes (
                    id TEXT PRIMARY KEY, region_id TEXT, candidate_json TEXT, source_step INTEGER
                );
            """)
            db.execute("INSERT OR IGNORE INTO regions VALUES ('R1', '', NULL, 'hypothesis')")
            latest = db.execute('SELECT region_id FROM region_evidence ORDER BY step DESC LIMIT 1').fetchone()
            if latest is not None:
                self.current_region = latest['region_id']

    @staticmethod
    def _pending_navigation(db, view, state):
        """实际移动且房间归属尚未确认的导航，才允许停车转向后继续复核。"""
        navigation = state.last_navigation
        if navigation is None:
            return False
        result = navigation.get('result')
        if result is None and (state.last_action or {}).get('action') == 'navigate':
            result = state.last_action['result']
        result = result or {}
        displacement = result.get('motion', {}).get('displacement_m')
        consumed = db.execute('SELECT 1 FROM region_connections WHERE action_step=?',
                              (navigation['step'],)).fetchone()
        if not consumed:
            # 确认仍在原房间也算复核结束；不能借旧位移把后续原地转头登记成跨房间。
            reports = db.execute(
                'SELECT report_json FROM region_evidence WHERE step>? AND step<?',
                (navigation['step'], view['step']),
            )
            consumed = any(json.loads(row['report_json']).get('region_supported') for row in reports)
        return (
            isinstance(displacement, (int, float)) and displacement >= .25
            and not consumed and not result.get('startup_failed')
            and navigation.get('observer_position') is not None
            and view['observer_position'] is not None
            and math.dist(navigation['observer_position'], view['observer_position']) >= .25
        )

    def comparison_references(self, view, regions):
        """每个候选房间保留固定锚点，再补最接近当前朝向的已确认视图。"""
        frames = []
        with self.spatial.connect() as db:
            for region in regions[:3]:
                anchor = region.get('reference_step')
                if anchor is None:
                    continue
                rows = db.execute("""
                    SELECT e.step, e.report_json, f.heading_deg
                    FROM region_evidence e JOIN semantic_frames f ON f.step=e.step
                    WHERE e.region_id=? AND e.step<? AND f.report_status='accepted'
                          AND f.scene_json IS NOT NULL
                """, (region['id'], view['step'])).fetchall()
                candidates = [row for row in rows if row['step'] != anchor
                              and json.loads(row['report_json']).get('region_supported')]
                heading = view.get('heading_deg')
                extra = min(candidates, key=lambda row: (
                    angle_distance(heading, row['heading_deg'])
                    if heading is not None and row['heading_deg'] is not None else 360,
                    -row['step'],
                ), default=None)
                steps = [anchor, *([extra['step']] if extra is not None else [])]
                frames.extend({**frame, 'region_id': region['id']} for frame in self.spatial.frames(steps))
        return frames

    def region_comparison_needed(self, view, state):
        with self.spatial.connect() as db:
            return self._pending_navigation(db, view, state)

    def update(self, view, state, decision, *, region_match=None):
        """区域身份采用停车对照；没有对照时采用原决策，连接仍须真实运动支持。"""
        update = decision.spatial_update
        if region_match is not None:
            matched = region_match['region_id']
            uncertain = matched == 'uncertain'
            # 定位与搜索分工：原决策归属不一致时，不能把它的“已查完”写入另一个房间。
            search_matches = update is not None and update.region_id == matched and not uncertain
            update = SpatialUpdate(
                region_id=self.current_region if uncertain else matched,
                label=region_match['label'],
                transition=('uncertain' if uncertain else
                            'same' if matched == self.current_region else 'entered'),
                evidence=region_match['evidence'],
                search_status=update.search_status if search_matches else None,
                remaining=update.remaining if search_matches else [],
            )
        review = decision.previous_action_review
        navigation = state.last_navigation
        # 一份明确的跨区视觉复核已足够表达新区域假设，不要求模型重复填写同义字段。
        # 实际运动仍由下方独立校验；显式 uncertain 不能被这个补全覆盖。
        if (update is None and review is not None and review.outcome == 'progress'
                and review.view_change == 'new_area'):
            update = SpatialUpdate(region_id='new', transition='entered', evidence=review.evidence)
        source = self.current_region
        destination = source
        with self.spatial.connect() as db:
            rows = db.execute('SELECT id FROM regions').fetchall()
            known = {row['id'] for row in rows}
            actual_navigation = self._pending_navigation(db, view, state)
            crossed = actual_navigation and update is not None and update.transition == 'entered'
            label_supported = False
            if update is not None:
                requested = update.region_id
                if crossed:
                    if requested == 'new':
                        destination = f'R{len(known) + 1}'
                        db.execute('INSERT INTO regions VALUES (?, ?, ?, ?)',
                                   (destination, update.label, view['step'], 'model_supported'))
                    elif requested in known:
                        destination = requested
                # 同一标签不是合并依据；没有完整证据只保存待复核报告。
                label_supported = (
                    update.transition == 'same' and requested == source
                    or crossed and (destination != source or requested == source)
                )
                if update.label and label_supported:
                    # 房间身份以首次地标为参照；后续视角描述留在证据中，不能把旧房间改名成新景象。
                    db.execute("""UPDATE regions SET label=?, source_step=?
                                  WHERE id=? AND (label IS NULL OR label='')""",
                               (update.label, view['step'], destination))
            if crossed and destination != source:
                db.execute('INSERT OR IGNORE INTO region_connections VALUES (?, ?, ?, ?, ?, ?)',
                           (navigation['step'], source, destination, navigation['step'], view['step'],
                            update.evidence))
            # 每次接受的停车观察都关联当时区域；空报告表示沿用假设，不伪造视觉确认。
            report = update.model_dump() if update else {}
            if update is not None:
                report['region_supported'] = bool(label_supported)
                report['source'] = ('stopped_image_comparison' if region_match is not None
                                    else 'decision_visual_report')
            db.execute('INSERT OR REPLACE INTO region_evidence VALUES (?, ?, ?)',
                       (view['step'], destination, encode(report)))
            self.current_region = destination
            crossed_candidate = next((item for item in state.candidates
                                      if navigation and item['id'] == navigation.get('candidate_id')), None)
            crossed_origin = (crossed_candidate.get('origin_candidate_id', crossed_candidate['id'])
                              if crossed_candidate is not None else None)
            for candidate in state.candidates:
                previous_route = db.execute('SELECT * FROM exploration_routes WHERE id=?',
                                            (candidate['id'],)).fetchone()
                seen_here = candidate['last_seen_step'] == view['step']
                if not seen_here and previous_route is None:
                    continue
                route_region = destination if seen_here else previous_route['region_id']
                source_step = view['step'] if seen_here else previous_route['source_step']
                route = dict(candidate)
                route['exploration_status'] = (
                    'unresolved' if route.get('no_new_information_count', 0) >= 2
                    else 'approached' if route.get('total_attempt_count', 0)
                    else 'seen'
                )
                origin = route.get('origin_candidate_id', route['id'])
                closed = next((goal for goal in reversed(state.subgoal_history)
                               if origin in goal.get('inspection_candidate_origins', [goal['origin_candidate_id']])
                               and goal['status'] in ('completed', 'abandoned')), None)
                previous = json.loads(previous_route['candidate_json']) if previous_route else {}
                resolution = None
                if closed is not None:
                    resolution = {'resolution': closed['evidence'], 'resolved_step': closed['ended_step'],
                                  'resolution_status': closed['status']}
                elif previous.get('resolution'):
                    # 近期目的历史被淘汰后，已解决或暂缓的路线仍保留原因。
                    resolution = {key: previous.get(key) for key in
                                  ('resolution', 'resolved_step', 'resolution_status')}
                # 有新证据并实际重试后重新打开，不永久排除旧入口。
                last_attempt = (route.get('last_attempt') or {}).get('step', 0)
                if resolution and last_attempt <= (resolution['resolved_step'] or 0):
                    route.update(resolution)
                    route['exploration_status'] = (
                        'deferred' if resolution['resolution_status'] == 'abandoned' else 'resolved'
                    )
                # 实际跨区后，入口的去向已知；房间是否查完由区域检查状态另行记录。
                if previous.get('destination_region'):
                    route_region, source_step = previous_route['region_id'], previous_route['source_step']
                    route.update({key: previous[key] for key in ('destination_region', 'traversed_step', 'destination_step')
                                  if key in previous})
                if (crossed and destination != source and route['kind'] == 'open_passage'
                        and origin == crossed_origin and not previous.get('destination_region')):
                    route_region, source_step = source, navigation['step']
                    route.update(destination_region=destination, traversed_step=navigation['step'],
                                 destination_step=view['step'])
                if route.get('destination_region'):
                    route['exploration_status'] = 'traversed'
                if route['status'] == 'avoid_from_here':
                    route['exploration_status'] = 'blocked_from_this_viewpoint'
                db.execute('INSERT OR REPLACE INTO exploration_routes VALUES (?, ?, ?, ?)',
                           (route['id'], route_region, encode(route), source_step))


    def _revisit_candidates(self, view, observations):
        """邻近的已看位置召回房间线索，不据此合并房间或添加连接。"""
        if view is None or view.get('observer_position') is None:
            return []
        matches = []
        for frame in observations:
            if (frame['step'] >= view['step']
                    or not frame['scene_json'] or frame['report_status'] != 'accepted'):
                continue
            observation = json.loads(frame['observation_json'])
            pose = observation.get('pose', {})
            position = [pose.get('x'), pose.get('y')]
            if (not observation.get('pose_valid', True)
                    or not all(isinstance(value, (int, float)) and math.isfinite(value)
                               for value in position)):
                continue
            distance = math.dist(position, view['observer_position'])
            if distance > self.spatial.settings.nearby_radius_m:
                continue
            headings = [frame['heading_deg'], view.get('heading_deg')]
            similar_heading = (
                all(isinstance(value, (int, float)) and math.isfinite(value) for value in headings)
                and angle_distance(*headings) <= self.spatial.settings.same_view_angle_deg
            )
            scene = json.loads(frame['scene_json'])
            description = scene.get('place_label') or '；'.join(scene.get('landmarks', [])[:3])
            description = description or scene.get('visible_ground', '')
            matches.append({
                'region_id': frame['region_id'], 'observation_step': frame['step'],
                'distance_m': round(distance, 3),
                'similar_heading': similar_heading, 'description': description[:120],
            })
        matches.sort(key=lambda item: (item['distance_m'], not item['similar_heading'], -item['observation_step']))
        selected = {}
        for match in matches:
            selected.setdefault(match['region_id'], match)
        return list(selected.values())[:2]

    @staticmethod
    def _route_summary(route):
        """模型只读路线结论；完整候选和复核留在执行状态及数据库。"""
        fields = (
            'id', 'origin_candidate_id', 'region_id', 'source_step', 'kind',
            'description', 'expected_new_view', 'exploration_status',
            'resolution', 'resolved_step', 'resolution_status', 'destination_region', 'traversed_step', 'destination_step',
            'total_attempt_count', 'no_new_information_count',
        )
        return {key: route[key] for key in fields if key in route}

    def _routes(self, db):
        """同一入口合并多次目击；只在实际新尝试后重新打开已结束的问题。"""
        groups = {}
        for row in db.execute('SELECT * FROM exploration_routes ORDER BY source_step DESC'):
            route = dict(region_id=row['region_id'], source_step=row['source_step'],
                         **json.loads(row['candidate_json']))
            # 临时观察点留在原始记录，实际前往后才成为持续待查问题。
            if route['kind'] == 'viewpoint' and not route.get('total_attempt_count', 0):
                continue
            origin = route.get('origin_candidate_id', route['id'])
            groups.setdefault(origin, []).append(route)
        routes = []
        for group in groups.values():
            route = max(group, key=lambda item: (
                (item.get('last_attempt') or {}).get('step', 0), item['source_step'],
            ))
            closed = [item for item in group
                      if item['exploration_status'] in ('resolved', 'deferred')]
            if closed:
                resolution = max(closed, key=lambda item: item.get('resolved_step') or 0)
                if (route.get('last_attempt') or {}).get('step', 0) <= (resolution.get('resolved_step') or 0):
                    route.update({key: resolution[key] for key in
                                  ('exploration_status', 'resolution', 'resolved_step', 'resolution_status')
                                  if key in resolution})
            traversed = [item for item in group if item.get('destination_region')]
            if traversed:
                # 同一来源的首次已走连接保持稳定，后续反向或继续前进不抹掉旧入口。
                passage = min(traversed, key=lambda item: item.get('traversed_step', 0))
                route.update({key: passage[key] for key in
                              ('region_id', 'source_step', 'destination_region', 'traversed_step', 'destination_step')
                              if key in passage})
                if route['exploration_status'] != 'blocked_from_this_viewpoint':
                    route['exploration_status'] = 'traversed'
            routes.append(route)
        return routes

    def closed_route_origins(self):
        """已走通或已结束的问题不再作为待查项；单纯受阻不全局排除入口。"""
        with self.spatial.connect() as db:
            return {route.get('origin_candidate_id', route['id']) for route in self._routes(db)
                    if route.get('destination_region') or route['exploration_status'] in ('resolved', 'deferred')}

    def query(self, view=None, *, focus_region=None):
        with self.spatial.connect() as db:
            regions = [dict(row) for row in db.execute('SELECT * FROM regions')]
            connections = [dict(row) for row in db.execute('SELECT * FROM region_connections ORDER BY action_step')]
            observations = [dict(row) for row in db.execute(
                """SELECT e.region_id, e.step, e.report_json, f.place_id, f.heading_deg,
                          f.observation_json, f.scene_json, f.report_status
                   FROM region_evidence e LEFT JOIN semantic_frames f ON f.step=e.step
                   ORDER BY e.step DESC"""
            )]
            routes = self._routes(db)
        # 本轮 VLM 对照只改变召回焦点；动作接受前不改写已存的房间归属。
        known_regions = {region['id'] for region in regions}
        focus_region = focus_region if focus_region in known_regions else self.current_region
        revisits = self._revisit_candidates(view, observations)
        # 图只建议沿已走连接返回；每一段仍需要当前画面确认。
        adjacency = {}
        for edge in connections:
            adjacency.setdefault(edge['source'], set()).add(edge['destination'])
            adjacency.setdefault(edge['destination'], set()).add(edge['source'])
        paths = {focus_region: [focus_region]}
        queue = deque([focus_region])
        while queue:
            region = queue.popleft()
            for neighbor in sorted(adjacency.get(region, ())):
                if neighbor not in paths:
                    paths[neighbor] = paths[region] + [neighbor]
                    queue.append(neighbor)

        # 先保留当前区和可能重访的旧区，避免近期新区域把旧摘要挤掉。
        relevant = [focus_region, *(item['region_id'] for item in revisits),
                    self.current_region, *sorted(adjacency.get(focus_region, ()))]
        unfinished = [route for route in routes
                      if route['exploration_status'] not in CLOSED_ROUTES]
        for route in unfinished[:4]:
            relevant.extend(paths.get(route['region_id'], []))
        relevant = list(dict.fromkeys(relevant))[:8]
        by_id = {region['id']: region for region in regions}
        summaries = []
        for region_id in relevant:
            region = dict(by_id[region_id])
            visits = int(region_id == 'R1') + sum(edge['destination'] == region_id for edge in connections)
            evidence = next((frame for frame in observations if frame['region_id'] == region_id), None)
            region_routes = [route for route in routes
                             if route['region_id'] == region_id or route.get('destination_region') == region_id]
            region.update(
                visits=visits,
                last_observed_step=evidence['step'] if evidence else None,
                route_counts={
                    'resolved': sum(route['exploration_status'] == 'resolved' for route in region_routes),
                    'pending': sum(route['exploration_status'] not in CLOSED_ROUTES
                                   for route in region_routes),
                    'deferred': sum(route['exploration_status'] == 'deferred' for route in region_routes),
                    'traversed': sum(route['exploration_status'] == 'traversed' for route in region_routes),
                    'blocked': sum(route['exploration_status'] == 'blocked_from_this_viewpoint'
                                   for route in region_routes),
                },
            )
            reports = []
            for frame in observations:
                if frame['region_id'] != region_id:
                    continue
                report = json.loads(frame['report_json'])
                if report.get('region_supported') and report.get('search_status'):
                    reports.append({'status': report['search_status'], 'step': frame['step'],
                                    'evidence': report['evidence'], 'remaining': report.get('remaining', [])})
            # 仅路过不清空检查结果；重新局部检查也保留上次查完的依据。
            search = dict(next((report for report in reports if report['status'] != 'visited'),
                               reports[0] if reports else {'status': 'visited' if evidence else 'unknown'}))
            if reports:
                search['last_activity'] = reports[0]['status']
            previous_check = next((report for report in reports if report['status'] == 'checked'
                                   and report['step'] < search.get('step', 0)), None)
            if previous_check and search['status'] != 'checked':
                search['previous_check'] = previous_check
            region['search'] = search
            # 固定首份已接受的区域证据，避免后续误认把历史参照一起改掉。
            reference = next((frame for frame in reversed(observations)
                              if frame['region_id'] == region_id and frame['report_status'] == 'accepted'
                              and json.loads(frame['report_json']).get('region_supported')), None)
            region['reference_step'] = reference['step'] if reference else None
            summaries.append(region)
        # 优先当前及可能重访区域的入口，不能让其他邻居的近期路线挤掉旧入口。
        region_priority = {region: index for index, region in enumerate(relevant)}
        ranked = sorted(routes, key=lambda route: (
            min(region_priority.get(route['region_id'], len(relevant)),
                region_priority.get(route.get('destination_region'), len(relevant))),
            -route['source_step'],
        ))
        reviewed = [route for route in ranked
                    if route['exploration_status'] in CLOSED_ROUTES]
        unfinished = [route for route in ranked
                      if route['exploration_status'] not in CLOSED_ROUTES]
        # 从未尝试的入口单独给出短提示，避免它被反复接近的新命名入口挤掉。
        untried = [route for route in unfinished
                   if route['kind'] == 'open_passage' and not route.get('total_attempt_count', 0)
                   and route['region_id'] in paths]
        frontier = min(untried, key=lambda route: (
            len(paths[route['region_id']]), route['source_step'],
        ), default=None)
        next_frontier = None
        if frontier is not None:
            next_frontier = {key: frontier[key] for key in ('id', 'region_id', 'source_step', 'description')}
            next_frontier['return_path'] = paths[frontier['region_id']]
        recent_edges = connections[-5:]
        recent_path = ([recent_edges[0]['source'], *[edge['destination'] for edge in recent_edges]]
                       if recent_edges else [self.current_region])
        return {
            'current_region': self.current_region,
            'focus_region': focus_region,
            'recent_region_path': recent_path,
            'next_frontier': next_frontier,
            'revisit_candidates': revisits,
            'regions': summaries,
            'connections': [edge for edge in connections
                            if edge['source'] in relevant and edge['destination'] in relevant][-12:],
            'routes': [dict(self._route_summary(route), return_path=paths.get(route['region_id']))
                       for route in unfinished[:8]],
            'reviewed_routes': [self._route_summary(route) for route in reviewed[:6]],
            'limits': '区域归属和重访线索须当前图确认；同位置或同名不合并房间。'
                      'traversed 入口已走通，destination_region 是已知去向；可正常通行，但不再是未知入口。已解决只指局部问题；暂缓路线保留放弃原因，有新证据可重开。未看到不等于不存在，待查数为0也不证明整间房已搜完。'
                      'checked 是本任务中已检查当前可达范围的模型判断；允许途经，不能当作永久排除。'
                      '返回路径仅含已走连接；反向通行仍须当前图确认。',
        }
