"""向模型公开技能说明，并执行模型明确选择的技能；不推断或替换选择。"""

from navigation_agent.agent.skills.search_navigation import SearchNavigationSkill
from navigation_agent.agent.skills.approach_target import ApproachTargetSkill


class SkillRegistry:
    def __init__(self):
        skills = (SearchNavigationSkill(), ApproachTargetSkill())
        self.definitions = {skill.name: skill for skill in skills}

    def context(self):
        # 目前只有两个短技能，每轮完整提供说明，无需增加一次加载技能的模型请求。
        return [
            {'name': skill.name, 'description': skill.description,
             'instructions': skill.instructions, 'actions': list(skill.actions)}
            for skill in self.definitions.values()
        ]

    def validate(self, decision, *, detector_available):
        if decision.skill is None:
            if decision.action == 'navigate':
                raise ValueError(
                    'navigate 须明确选择 skill：search_navigation（寻找未知目标）或 '
                    'approach_target（接近/核实具体目标线索）；与 action 在同一回复填写。'
                )
            return
        if decision.skill not in self.definitions:
            raise ValueError('未知 skill；从 available_skills 中选择：' + ', '.join(self.definitions))
        skill = self.definitions[decision.skill]
        if decision.action not in skill.actions:
            raise ValueError(f'{skill.name} 不支持动作 {decision.action}')
        skill.validate(decision, detector_available=detector_available)

    async def prepare(self, decision, watch):
        if decision.skill == 'selective_direct':
            enabled = bool(watch.settings.server_url and decision.target_labels)
            await watch.configure(enabled, decision.target_labels or [])
            return enabled
        if decision.skill is None:
            await watch.configure(False, [])
            return False
        return await self.definitions[decision.skill].prepare(decision, watch)
