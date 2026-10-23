"""在页面运行参数与 dev 的不可变执行配置之间转换，不读取环境或密钥。"""

from navigation_agent.agent.config import AgentSettings
from navigation_agent.app.go2.motion_settings import MotionSettings


def motion_defaults(runtime):
    agent, robot = runtime.agent, runtime.backend.robot
    values = MotionSettings().model_dump()
    if robot.backend == 'go2':
        for name in values['look']:
            key = name + '_rad_s' if name in ('look_angular_min', 'look_angular_max') else name
            if name != 'navigation_seconds':
                values['look'][name] = getattr(robot, key)
        values['protection'] = runtime.backend.protection.model_dump()
    values['look']['navigation_seconds'] = agent.execution.navigation_seconds
    values['agent'] = {
        **runtime.backend.ui.task_defaults.model_dump(), 'decision_timeout_s': agent.llm.request_timeout_s,
        'decision_attempts': agent.llm.format_retries + 1, 'max_tokens': agent.llm.max_tokens,
        'navigation_error_limit': agent.execution.max_consecutive_navigation_errors,
    }
    values['prompt']['instruction_guide'] = agent.prompt.instruction_guide
    values['exploration'] = {
        'history_candidates': agent.execution.max_unfinished_candidates,
        'view_angle_deg': agent.observation.same_view_angle_deg,
        'scan_reset_m': agent.exploration.scan_reset_distance_m,
        'recovery_attempts': agent.exploration.max_recovery_attempts,
        'candidate_failures': agent.exploration.max_candidate_failures,
        'failed_displacement_m': agent.exploration.failed_displacement_m,
        'stuck_displacement_m': agent.exploration.stuck_displacement_m,
    }
    values['memory'] = {
        'place_radius_m': agent.memory.place_radius_m, 'nearby_radius_m': agent.memory.nearby_radius_m,
        'view_angle_deg': agent.memory.same_view_angle_deg, 'heading_bucket_deg': agent.memory.heading_bin_deg,
        'frames': agent.memory.max_semantic_frames, 'objects': agent.memory.max_objects,
        'reviews': agent.memory.max_reviews, 'places': agent.memory.max_places,
    }
    return MotionSettings.model_validate(values)


def apply_agent_settings(base, motion):
    """生成单任务快照，保留 dev 的 LLM 地址、密钥名称、日志路径和任务硬限制。"""
    if motion is None:
        return base
    data = base.model_dump()
    operator, exploration, memory = motion.agent, motion.exploration, motion.memory
    data['task_defaults'].update(max_steps=operator.max_steps, max_seconds=operator.max_seconds)
    data['llm'].update(request_timeout_s=operator.decision_timeout_s,
                       format_retries=operator.decision_attempts - 1, max_tokens=operator.max_tokens)
    data['prompt']['instruction_guide'] = motion.prompt.instruction_guide
    data['execution'].update(navigation_seconds=motion.look.navigation_seconds,
        max_consecutive_navigation_errors=operator.navigation_error_limit,
        max_unfinished_candidates=exploration.history_candidates)
    data['exploration'].update(max_recovery_attempts=exploration.recovery_attempts, max_candidate_failures=exploration.candidate_failures,
        scan_reset_distance_m=exploration.scan_reset_m, failed_displacement_m=exploration.failed_displacement_m,
        stuck_displacement_m=exploration.stuck_displacement_m)
    data['observation']['same_view_angle_deg'] = exploration.view_angle_deg
    data['memory'].update(place_radius_m=memory.place_radius_m, nearby_radius_m=memory.nearby_radius_m,
        same_view_angle_deg=memory.view_angle_deg, heading_bin_deg=memory.heading_bucket_deg,
        max_semantic_frames=memory.frames, max_objects=memory.objects,
        max_reviews=memory.reviews, max_places=memory.places)
    return AgentSettings.model_validate(data)
