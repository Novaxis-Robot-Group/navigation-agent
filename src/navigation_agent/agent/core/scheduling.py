"""默认直接执行；仅在停止、候选中断或真实静止事件后调用 Planner。"""

from collections import deque
import math

from navigation_agent.agent.model.completion import verify_completion
from navigation_agent.agent.model.decision import decide as plan
from navigation_agent.agent.schemas import AgentError, Decision
from navigation_agent.agent.tools.navigation import validate_instruction


class SelectiveController:
    def __init__(self, settings, goal, event, *, scene_guarded=False):
        self.settings = settings.selective
        self.goal = self.instruction = goal
        self.event = event
        self.scene_guarded = scene_guarded
        self.mode = 'STAGING' if scene_guarded else 'DIRECT'
        self.trigger = 'scene_validation_required' if scene_guarded else 'episode_start'
        self.interventions = 0
        self.diagnosis = None
        self.labels = []
        self.internal_decision = None
        self.motion = deque()
        self.navigation_id = None
        self.last_stamp = None
        self.stages = []

    def context(self):
        return dict(mode=self.mode, trigger=self.trigger, global_task=self.goal,
                    execution_instruction=self.instruction,
                    interventions=self.interventions, diagnosis=self.diagnosis,
                    stage_history=self.stages[-6:])

    def transition(self, mode, reason):
        old = self.mode
        if mode == 'STAGING' and old != 'STAGING':
            self.interventions += 1
            if self.interventions > self.settings.max_interventions:
                raise AgentError('选择性干预预算耗尽，未宣布成功')
        self.mode, self.trigger = mode, reason
        self.event('selective_transition', previous=old, **self.context())

    def owns(self, decision):
        return decision is self.internal_decision

    async def decide(self, client, model_context, images, *args, **kwargs):
        self.internal_decision = None
        if self.mode == 'VERIFY':
            try:
                report, usage = await verify_completion(
                    client, model_context, images, kwargs['settings'], evidence_mode=True)
            except Exception as exc:
                self.event('completion_check', confirmed=False, error=type(exc).__name__)
                raise AgentError('选择性完成核验不可用，未宣布成功：' + type(exc).__name__) from None
            self.diagnosis = usage.get('criteria')
            self.event('completion_check', **report.model_dump(), usage=usage, source='selective_stop')
            if report.confirmed:
                self.internal_decision = Decision(
                    action='finish', reason='LightNav 停止后分项核验通过', evidence=report.evidence)
                return self.internal_decision, usage
            self.transition('STAGING', 'stop_not_verified')

            # 本帧已核验过，Planner 再申请 finish 时只返回同一诊断。
            async def same_frame_report():
                return report
            kwargs['confirm_completion'] = same_frame_report

        if self.mode == 'DIRECT':
            self.internal_decision = Decision(
                action='navigate', skill='selective_direct', instruction=self.instruction,
                target_labels=self.labels, reason='无待处理事件，交由 LightNav 连续执行')
            self.event('selective_direct', **self.context())
            return self.internal_decision, {'planner_calls': 0}

        model_context['execution']['selective'] = self.context()
        self.event('selective_planner_request', **self.context())
        decision, usage = await plan(client, model_context, images, *args, **kwargs)
        if decision.resume_instruction is not None:
            if decision.action not in ('look', 'orient'):
                raise AgentError('仅允许成功的 look/orient 之后交还剩余任务')
            validate_instruction(decision.resume_instruction)
        if decision.target_labels is not None:
            self.labels = list(decision.target_labels)
        return decision, usage

    def result(self, decision, result, step):
        reason = result.get('reason')
        if decision.action == 'navigate':
            if reason == 'model_stop':
                self.transition('VERIFY', reason)
            elif reason in ('agent_interrupt', 'selective_no_motion', 'local_observation_distance',
                            'blocked', 'no_progress', 'turn_timeout'):
                self.transition('STAGING', result.get('detail') or reason)
            elif reason in ('navigation_error', 'upstream_cancelled', 'timeout'):
                raise AgentError('导航执行不可用：' + reason)
            else:
                self.transition('STAGING', reason or 'unknown_navigation_result')
        elif (decision.action in ('look', 'orient') and decision.resume_instruction
              and reason in ('turned', 'oriented')):
            self.stages.append(dict(step=step, action=decision.action,
                                    observation_reason=decision.reason,
                                    resumed_instruction=decision.resume_instruction,
                                    source='planner_proposed_handoff_after_actual_turn'))
            self.instruction = decision.resume_instruction
            self.transition('STAGING' if self.scene_guarded else 'DIRECT',
                            'scene_validation_required' if self.scene_guarded else 'explicit_handoff_after_turn')

    def observe_motion(self, observation, tools):
        """只对新鲜、不同时间戳的真实样本判断静止，不把网络等待算成卡住。"""
        identity = tools.navigation_id
        if identity is None:
            return
        if identity != self.navigation_id:
            self.navigation_id = identity
            self.motion.clear()
            self.last_stamp = None
        stamp = observation.get('camera_stamp_ns')
        pose = observation.get('pose') or {}
        values = [pose.get('x'), pose.get('y'), observation.get('yaw', pose.get('yaw'))]
        if (not observation.get('pose_valid', True) or stamp is None or stamp == self.last_stamp
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)):
            return
        self.last_stamp = stamp
        now = stamp / 1e9
        self.motion.append((now, *values))
        while len(self.motion) > 2 and now - self.motion[1][0] >= self.settings.stationary_window_s:
            self.motion.popleft()
        if len(self.motion) < 3 or now - self.motion[0][0] < self.settings.stationary_window_s:
            return
        origin = self.motion[0]
        distance = max(math.dist(origin[1:3], sample[1:3]) for sample in self.motion)
        turns = sum(abs((b[3] - a[3] + math.pi) % (2*math.pi) - math.pi)
                    for a, b in zip(self.motion, list(self.motion)[1:]))
        if distance < self.settings.stationary_distance_m and math.degrees(turns) < self.settings.stationary_turn_degrees:
            if tools.request_navigation_interrupt(identity, 'selective_no_motion'):
                self.event('selective_motion_trigger', navigation_id=identity,
                           duration_s=now-origin[0], max_displacement_m=distance,
                           absolute_turn_degrees=math.degrees(turns))
                self.motion.clear()
