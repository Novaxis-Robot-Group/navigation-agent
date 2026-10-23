"""按技能的观察引用读取记忆，统一提供本轮文字证据。"""


def build_context(request, observation, spatial, view, search, settings, *, trajectory=None, topology=None):
    state = search.state
    closed_origins = topology.closed_route_origins() if topology is not None else set()
    execution = search.context(excluded_route_origins=closed_origins)
    compact_execution_memory(execution, closed_origins)
    # 方向覆盖只作证据摘要；不作为导航前置条件。
    for name in ('look_count', 'observed_headings_deg', 'unobserved_look_directions',
                 'coverage_complete', 'largest_heading_gap_deg', 'direction', 'anchor'):
        execution['scan'].pop(name, None)
    session = execution['exploration_session']
    target = execution.get('target_tracking')
    evidence = [] if state.recovery_required else spatial.frames(
        [frame['step'] for frame in session['views']]
    )
    frames = {frame['step']: frame for frame in evidence}
    session['views'] = [
        {**reference,
         'scene': frames.get(reference['step'], {}).get('scene'),
         'source': frames.get(reference['step'], {}).get('source'),
         'action_status': frames.get(reference['step'], {}).get('action_status')}
        for reference in session['views']
    ]
    memory = None
    if not state.recovery_required:
        memory = spatial.query(view)
        if target:
            target['evidence_frames'] = spatial.frames(state.target.evidence_steps())
            memory['semantic_frames'] = target['evidence_frames']
            # 目标阶段保留目标帧的物体报告，避免无关历史抢占注意力。
            steps = set(state.target.evidence_steps())
            memory['objects'] = [item for item in memory['objects'] if item['frame_step'] in steps]
    if memory is not None:
        # 已附方向图只出现一次；其他历史图片仅在关联目标或待查候选时保留。
        attached_steps = {frame['step'] for frame in session['views']}
        if execution.get('last_navigation'):
            attached_steps.add(execution['last_navigation']['step'])
        candidate_steps = {item['last_seen_step'] for item in execution['unfinished_elsewhere']}
        goal_steps = {item['frame_step'] for item in memory['objects']
                      if item['name'] in request.goal or request.goal in item['name']}
        if not target:
            memory['semantic_frames'] = [
                frame for frame in memory['semantic_frames']
                if frame['step'] not in attached_steps
                and (frame['step'] in candidate_steps or frame['step'] in goal_steps
                     or (frame.get('scene') or {}).get('target_seen') == 'yes')
            ]
    trajectory_context = None
    if trajectory is not None and not state.recovery_required:
        trajectory_context = trajectory.query(view['step'])
        # 模型使用统一经历，运行日志仍保留完整 scan 动作和空间复核记录。
        session.pop('actions', None)
        if memory is not None:
            memory.pop('recent_reviews', None)
    # 请求参数仍供工具调用；模型上下文只展示后端实际启用的分段时限。
    time_limit_enabled = settings.prompt.navigation_time_limit_enabled
    navigation_seconds = request.navigation_seconds if time_limit_enabled else None
    return {
        'exploration_memory': exploration_context(topology, view, execution) if topology is not None else None,
        'trajectory_memory': trajectory_context,
        'goal': request.goal,
        'navigation_settings': {
            'time_limit_enabled': time_limit_enabled,
            'seconds': navigation_seconds,
        },
        'observation_step': view['step'],
        'observation_id': f'O{view["step"]}',
        'observation': observation,
        'spatial_memory': memory,
        'execution': {**execution, 'navigation_seconds': navigation_seconds},
    }


def exploration_context(topology, view, execution, *, comparison=None):
    """已在执行上下文展开的候选，拓扑只补区域、结论和返回线索。"""
    focus = comparison.get('region_id') if comparison else None
    memory = topology.query(view, focus_region=focus)
    if comparison is not None:
        memory['visual_match'] = comparison
    # 决策前只能知道上次区域，不能把待复核的归属标成当前定位事实。
    memory['previous_region'] = memory.pop('current_region')
    represented = {candidate['id'] for key in ('local_candidates', 'unfinished_elsewhere')
                   for candidate in execution[key]}
    for key in ('routes', 'reviewed_routes'):
        for route in memory[key]:
            if route['id'] in represented:
                for name in ('origin_candidate_id', 'kind', 'description', 'expected_new_view',
                             'total_attempt_count', 'no_new_information_count'):
                    route.pop(name, None)
    return memory

def compact_execution_memory(execution, closed_origins):
    """执行状态保留完整记录；决策只读未结束问题及简短结果。"""
    session = execution.get('exploration_session') or {}
    actions = [
        execution.get('last_action'), execution.get('last_navigation'),
        (session.get('previous_result') or {}).get('last_action'),
        *session.get('actions', []),
    ]
    for action in actions:
        motion = ((action or {}).get('result') or {}).get('motion') or {}
        motion.pop('sampled_positions', None)

    candidates = {}
    for candidate in execution['unfinished_elsewhere']:
        origin = candidate.get('origin_candidate_id', candidate['id'])
        if origin not in closed_origins:
            candidates.setdefault(origin, candidate)
    fields = ('id', 'origin_candidate_id', 'place_id', 'kind', 'description',
              'expected_new_view', 'last_seen_step', 'status', 'total_attempt_count',
              'no_new_information_count', 'failed_attempts', 'last_attempt')
    execution['unfinished_elsewhere'] = [
        {key: candidate[key] for key in fields if key in candidate}
        for candidate in candidates.values()
    ]
    # 问题结论由拓扑长期保存；近期状态只补充尚未纳入拓扑的切换记录。
    execution['recent_subgoals'] = [
        {key: goal[key] for key in ('origin_candidate_id', 'question', 'status', 'evidence', 'ended_step')
         if key in goal}
        for goal in execution.get('recent_subgoals', [])
        if goal['origin_candidate_id'] not in closed_origins
    ]
