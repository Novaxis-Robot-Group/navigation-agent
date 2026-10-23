"""共享找物上下文与执行记录；不替模型选择导航技能。"""

from navigation_agent.agent.tools.registry import ToolCall
from navigation_agent.agent.skills.object_search.state import ExecutionState


class ObjectSearchContext:
    def __init__(self, settings, look_angle_deg, *, scene_validator=None):
        policy = settings.prompt.scene.navigation_policy if settings.prompt.scene else None
        self.state = ExecutionState(
            look_angle_deg=look_angle_deg, exploration=settings.exploration,
            observation=settings.observation, execution=settings.execution,
            navigation_policy=policy, scene_validator=scene_validator,
        )
        self.harness = settings.harness
        self.navigation_errors = 0

    def observe(self, view):
        self.state.observe(view)

    def context(self, *, excluded_route_origins=()):
        return self.state.context(excluded_route_origins=excluded_route_origins)

    def validate_scene(self, decision):
        self.state.validate_scene(decision)

    def validate(self, decision):
        return self.state.resolve(decision)

    def accept(self, decision):
        return self.state.accept(decision)

    def prepare_call(self, decision, candidate_id, seconds):
        action = decision.action
        event = {'action': action}
        max_seconds = None
        max_displacement = None
        if action in ('look', 'orient'):
            return self.state.scan.prepare_call(decision)
        if action == 'recover':
            arguments = {}
        elif action == 'navigate':
            candidate = next(item for item in self.state.candidates if item['id'] == candidate_id)
            if candidate['kind'] == 'viewpoint':
                max_displacement = self.state.policy.viewpoint_max_displacement_m or None
                # 同一问题已获得进展时允许更长观察段，不修改运动保护。
                review = decision.previous_action_review
                if (max_displacement and decision.subgoal_review is not None
                        and decision.subgoal_review.status == 'continue'
                        and review is not None and review.outcome == 'progress'):
                    max_displacement *= self.harness.continued_viewpoint_scale
                event['max_displacement_m'] = max_displacement
            if self.state.target.active and self.state.target.active['phase'] == 'temporarily_lost':
                seconds = min(seconds, 3.0)
                event['reacquiring_target'] = True
                max_seconds = seconds
            arguments = {'instruction': decision.instruction, 'seconds': seconds}
            if max_displacement is not None:
                arguments.update(max_displacement_m=max_displacement,
                                 origin=self.state.current.get('observer_position'))
            event.update(instruction=decision.instruction, candidate_id=candidate_id,
                         place_id=self.state.current['id'])
        else:
            raise ValueError('任务结束信号不是运动工具')
        return ToolCall(action, arguments, max_seconds=max_seconds), event

    def before_execute(self, decision, candidate_id):
        if decision.action == 'navigate':
            self.state.navigation_started(
                candidate_id,
                continue_inspection=(decision.subgoal_review is not None
                                     and decision.subgoal_review.status == 'continue'),
            )

    def record_result(self, decision, candidate_id, result, step):
        state = self.state
        if decision.action == 'navigate':
            result['candidate_id'] = candidate_id
            self.navigation_errors = (
                self.navigation_errors + 1 if result['reason'] == 'navigation_error' else 0
            )
        state.scan.record_result(
            decision.action, result, candidate_id=candidate_id, view_id=decision.view_id,
        )
        # 启动失败不算已执行的接近；后续仍保留完整失败日志。
        if not result.get('startup_failed'):
            state.record_result(candidate_id, result)
            state.target.record_reacquire_action()
        state.last_action = {
            'step': step, 'action': decision.action,
            'instruction': decision.instruction, 'direction': decision.direction,
            'candidate_id': candidate_id,
            'result': {key: result[key] for key in ('reason', 'detail', 'motion', 'arrival_verified') if key in result},
        }
        if decision.action == 'navigate':
            state.last_navigation['result'] = dict(state.last_action['result'])
