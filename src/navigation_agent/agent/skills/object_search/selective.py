"""将选择性调度接到现有找物状态；场景校验仍由部署注入。"""

from navigation_agent.agent.core.scheduling import SelectiveController
from navigation_agent.agent.tools.navigation import validate_instruction
from navigation_agent.agent.tools.registry import ToolCall
from navigation_agent.agent.skills.object_search.context import ObjectSearchContext


class SelectiveSearch(ObjectSearchContext):
    def __init__(self, settings, goal, *, event, look_angle_deg=70., scene_validator=None):
        super().__init__(settings, look_angle_deg, scene_validator=scene_validator)
        self.time_limit_enabled = settings.prompt.navigation_time_limit_enabled
        self.selective = SelectiveController(
            settings, goal, event, scene_guarded=self.state.scene_validator is not None)

    def context(self, **kwargs):
        context = super().context(**kwargs)
        context['selective'] = self.selective.context()
        return context

    def accept(self, decision):
        if self.selective.owns(decision):
            return None
        return super().accept(decision)

    def validate(self, decision):
        if decision.resume_instruction is not None:
            if decision.action not in ('look', 'orient'):
                raise ValueError('仅允许 look/orient 成功后通过 resume_instruction 交还任务')
            validate_instruction(decision.resume_instruction)
        return super().validate(decision)

    def before_execute(self, decision, candidate_id):
        if self.selective.owns(decision):
            self.state.last_navigation = dict(
                step=self.state.step, candidate_id=None, intent=decision.instruction,
                image=self.state.current.get('image'),
                heading_deg=self.state.current.get('heading_deg'),
                observer_position=self.state.current.get('observer_position'))
            return
        return super().before_execute(decision, candidate_id)

    def prepare_call(self, decision, candidate_id, seconds):
        if self.selective.owns(decision) and decision.action == 'navigate':
            return ToolCall('navigate', dict(instruction=decision.instruction, seconds=seconds)), dict(
                action='navigate', instruction=decision.instruction,
                execution_mode='DIRECT', time_limit_enabled=self.time_limit_enabled)
        return super().prepare_call(decision, candidate_id, seconds)

    def record_result(self, decision, candidate_id, result, step):
        if self.selective.owns(decision):
            self.state.scan.record_result(decision.action, result)
            self.state.last_action = dict(
                step=step, action=decision.action, instruction=decision.instruction,
                candidate_id=None, result=dict(result), execution_mode='DIRECT')
            self.state.last_navigation['result'] = dict(result)
        else:
            super().record_result(decision, candidate_id, result, step)
        self.selective.result(decision, result, step)
