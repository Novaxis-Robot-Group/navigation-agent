"""校验探索约束、判断运动进展，不直接控制机器人。"""

import re


def candidate_is_current(candidate, current_view_id):
    """候选来源适用于全部动作；历史提及不能刷新目击位置和图片。"""
    references = set(re.findall(
        r'(?<![A-Za-z0-9_])(?:O[1-9]\d*|S[1-9]\d*V[1-9]\d*)(?![A-Za-z0-9_])',
        candidate.evidence,
    ))
    if candidate.evidence_source == 'history' or references - {current_view_id}:
        return False
    return candidate.evidence_source == 'current' or candidate.id.startswith('new_')


def historical_navigation_error():
    return (
        '导航候选缺少当前图证据或引用历史图；'
        '旧 C 候选重新可见时填写 evidence_source=current，并只描述当前终点与通路。'
        '历史入口仍保留；若已转回复核但未见入口，就结束这次检查，'
        '选择 action_options 中未看方向、当前可见的观察点，或已允许的 recover。'
        '只有尚可复核的方向才可 orient，不能反复转回寻找同一入口。'
    )


def validate_action(state, decision):
    if (state.target.active and decision.action == 'navigate'
            and decision.target_review is not None
            and decision.target_review.status == 'temporarily_lost'
            and any(c.kind == 'target' and c.id == decision.candidate_id for c in decision.candidates)):
        raise ValueError('目标暂时不可见时不能直接接近历史目标；先确认或选择当前可见的观察位置')
    target = state.target.active
    review = decision.target_review
    lost = target and (review.status == 'temporarily_lost' if review else target['phase'] == 'temporarily_lost')
    deferring = state.target.defers_local_followup(decision)
    if lost and decision.action == 'navigate' and not deferring:
        candidate = next((c for c in decision.candidates if c.id == decision.candidate_id), None)
        if candidate is None or candidate.kind != 'viewpoint':
            raise ValueError('目标暂时不可见：先 look 定向找回，或选择当前可见的 viewpoint 短距离改善视角；不能把另一个门洞当成原目标路线。')
    if lost and target.get('reacquire_actions', 0) >= state.target.reacquire_limit and not deferring:
        raise ValueError('局部找回预算已用尽；不能继续同方式查找或仅因此结束整个任务。'
                         '若当前仍未见目标，保持 target_review=temporarily_lost，'
                         '用 subgoal_review=abandoned 说明暂缓局部跟进，再按当前图换观察位置或恢复探索。')
    options = state.action_options()
    selected = next((item for item in decision.candidates
                     if item.id == decision.candidate_id), None)
    if decision.action == 'navigate':
        if state.scan.phase == 'aligning':
            raise ValueError('转回后必须先获取新观察，不能在原图上直接导航')
        current_view_id = state.scan.current_view_id
        if decision.view_id is not None and decision.view_id != current_view_id:
            raise ValueError(historical_navigation_error())
        if (selected and selected.kind == 'open_passage'
                and re.search(r'\bdoor\s*frame\b', decision.instruction, re.IGNORECASE)
                and not re.search(r'\b(doorway|opening|passage|open floor)\b',
                                  decision.instruction, re.IGNORECASE)):
            raise ValueError('open_passage 的终点应是当前可通行的开口或门内可见空地，'
                             '不能只写 door frame（门框实体）。若只看见窗框或门框而未见可通行地面，'
                             '先 look 确认；按当前 VLN 指令指南描述终点。')
        if selected and not candidate_is_current(selected, current_view_id):
            raise ValueError(historical_navigation_error())
    pose_available = state.current['observer_position'] is not None and state.current['heading_deg'] is not None
    if decision.action == 'orient':
        if not pose_available:
            raise ValueError('缺少当前位置或朝向，不能转回历史观察方向')
        nearby_ids = {
            view['id'] for view in state.scan.choices(state.candidates, state.yaw, state.same_view_angle_deg)
        }
        if decision.view_id not in nearby_ids:
            raise ValueError('orient 必须选择当前 action_options.orient_view_ids；旧位置图片不能用于原地转回')
    if decision.action in ('look', 'orient'):
        state.scan.validate_reinspection(decision)
    if decision.action == 'give_up' and pose_available and state.target.active is None:
        choices = state.scan.useful_choices(state.candidates, state.yaw, state.same_view_angle_deg)
        if choices:
            raise ValueError('探索 session 中仍有未复核候选方向；先 orient 转回确认，不能因环视结束放弃')
        if not state.must_change_position and (
                state.scan.unobserved_look_directions() or options['unseen_via_orient']):
            raise ValueError('当前位置仍有未看的方向，不能只因左右各看过一次就放弃。'
                             '先选未看 look 方向；若左右都重复，按 unseen_via_orient 转回指定 view_id，'
                             '再沿 then_look 继续看未查方向。发现出口即可前往，不必补整圈。')
    if decision.action == 'look' and decision.direction not in options['look_directions']:
        raise ValueError('look 必须选择当前 action_options.look_directions；根据待确认的问题选择方向，重复方向必须提供复核依据')
    if decision.action == 'recover' and not options['recover_allowed']:
        raise ValueError('当前无需脱困或原地脱困预算已用完；选择可见空地观察点或give_up')


def validate_candidate(state, selected):
    if selected['kind'] != 'target' and not selected.get('expected_new_view', '').strip():
        raise ValueError('探索候选必须填写 expected_new_view，说明接近后想看清的问题或改善的视野，不要求保证发现新区域')
    if state.must_change_position and selected['kind'] != 'viewpoint':
        raise ValueError(
            '原地受阻后必须先换位置：使用recover或选择当前可见的空地viewpoint，不要直接冲向门洞'
        )
    if selected.get('no_new_information_count', 0) >= 2:
        raise ValueError('同一目的连续两次没有新信息，位置变化不能解除限制。'
                         '先 look/orient 获取不同方向的证据；或用 subgoal_review=abandoned 明确放弃原路线。'
                         '只有新的 progress 视觉复核才能继续同一目的。')
    if selected['failed_attempts'] >= state.policy.max_candidate_failures:
        raise ValueError(
            f"{selected['id']} 从当前观察点已失败{state.policy.max_candidate_failures}次，禁止原地换措辞重试。请选择其他方向或观察点、look 或 give_up"
        )


def classify_result(result, policy):
    # 区分接近失败和卡住：失败不等于入口不可通行
    if result['reason'] == 'agent_interrupt':
        return False, False
    displacement = result.get('motion', {}).get('displacement_m')
    failed = result['reason'] in ('blocked', 'no_progress', 'path_repeated', 'timeout', 'navigation_error') or (
        displacement is not None and displacement < policy.failed_displacement_m
    )
    stuck = result['reason'] in ('blocked', 'no_progress') or (
        displacement is not None
        and displacement < policy.stuck_displacement_m
        and result.get('motion', {}).get('blocked_samples', 0) > 0
    )
    return failed, stuck
