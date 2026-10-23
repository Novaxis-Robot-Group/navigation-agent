"""任务循环：协调观察、决策、工具与任务状态。"""

import asyncio
import copy
import itertools
import time

from navigation_agent.agent.config import AgentSettings
from navigation_agent.agent.schemas import Decision, AgentError
from navigation_agent.agent.tools import NavigationTools
from navigation_agent.agent.model.decision import decide, DecisionRetryExhausted
from navigation_agent.agent.model.completion import verify_completion
from navigation_agent.agent.model.observation import compare_regions, compare_routes
from navigation_agent.agent.context.prompts import read_prompt, navigation_prompt, target_confirmation_prompt, decision_reminder
from navigation_agent.agent.context.images import image_manifest
from navigation_agent.agent.context.builder import build_context, exploration_context
from navigation_agent.agent.core.runner import task_scope
from navigation_agent.agent.tools.target_watch import TargetWatch
from navigation_agent.agent.tools.capture import capture_during_navigation
from navigation_agent.agent.context.prompts import target_watch_prompt, algorithm_prompt


async def execute(record, request, tools: NavigationTools, *, settings: AgentSettings, api_key: str,
                  model=None, store=None, search=None, scene_validator=None):
    fixed_prompt = None
    if not settings.prompt.reload_each_step:
        fixed_prompt = read_prompt(settings.prompt)
    async with task_scope(record, request, tools, settings, api_key,
                          model=model, store=store, search=search, scene_validator=scene_validator) as scope:
        store, search, spatial = scope.store, scope.search, scope.spatial
        trajectory = scope.trajectory
        execution = search.state
        sequence = itertools.count(1)
        region_check_navigation = None
        region_check_attempts = 0
        replanning_used = False
        decision_recovery = None
        task_anchor = None

        def capture_passive(jpeg, observation, action_step):
            observed_step = next(sequence)
            image_name = f'{observed_step:03d}.jpg'
            store.save_image(record, image_name, jpeg)
            observed = spatial.observe(observation, 'images/' + image_name, observed_step)
            store.event(record, 'passive_observation', step=observed_step, action_step=action_step,
                        image=observed['image'], observation=observation)
            if hasattr(search, 'selective'):
                search.selective.observe_motion(observation, tools)
            return observed

        def capture_detection(jpeg, observation, action_step, boxed):
            view = capture_passive(jpeg, observation, action_step)
            if boxed:
                name = f'detection-{view["step"]:03d}.jpg'
                store.save_image(record, name, boxed)
                view = {**view, 'boxed_image': 'images/' + name}
            return view

        watch = scope.watch = TargetWatch(
            settings.target_watch, scope.detector, tools,
            event=lambda kind, **data: store.event(record, kind, **data),
            capture=capture_detection,
        )
        def validate_decision(decision):
            scope.skills.validate(decision, detector_available=bool(settings.target_watch.server_url))
            watch.validate(decision)
            return search.validate(decision)

        # 决策预算与观察编号分离，被动图像不会消耗动作轮数。
        for decision_step in range(1, request.max_steps + 1):
            step = next(sequence)
            # 停车取图，记录本轮观察
            record.update(step=decision_step, phase='observing')
            store.progress(record, 'observing')
            observation_started = time.monotonic()
            jpeg, observation = await tools.observe()
            store.event(record, 'stopped_observation_latency', step=step,
                        seconds=time.monotonic() - observation_started)
            image_name = f'{step:03d}.jpg'
            record['latest_image'] = image_name
            store.save_image(record, image_name, jpeg)
            view = spatial.observe(observation, 'images/' + image_name, step)
            if task_anchor is None:
                task_anchor = dict(view)
            trajectory.observe(step)
            search.observe(view)
            model_context = build_context(
                request, observation, spatial, view, search, settings, trajectory=trajectory, topology=scope.topology,
            )
            if settings.harness.evidence_completion:
                model_context['task_contract'] = {
                    'original_instruction': request.goal,
                    'initial_observation': {key: task_anchor.get(key) for key in
                                            ('step', 'image', 'heading_deg')},
                    'rule': '原始目标始终有效。方向和计数依原文对应阶段的观察语境；不要默认全部关系都属于起始图。局部观察点不是最终目标。'
                            '分别核对目标实例、用户要求的接近条件和当前朝向；无法确认时继续获取证据。',
                }
            model_context['available_skills'] = scope.skills.context()
            model_context['available_tools'] = scope.tools.context()
            model_context['previous_skill'] = record.get('active_skill')
            if decision_recovery is not None:
                model_context['decision_recovery'] = {
                    'reason': decision_recovery,
                    'instruction': '上轮未执行动作。请按当前新图和已有记忆重新选择下一步；'
                                   '局部路线没走完或动作被拒，不代表整个任务无路可走。'
                                   '检查其他当前可见入口、观察方向或已有 recover 选项，'
                                   '不重复照抄被拒动作，不虚构新证据；确实无路可走可说明原因结束。',
                }
            if settings.target_watch.server_url:
                model_context['target_watch'] = watch.context()
            context_execution = model_context['execution']
            target = context_execution.get('target_tracking') or {}
            regions = model_context['exploration_memory']['regions'][:3]
            store.save_execution(record, execution)
            store.event(
                record,
                'observation',
                step=step,
                image='images/' + image_name,
                observation=observation,
            )

            # 只在停车后独立比较原图；同次导航最多四次，允许转头后补足墙角视野。
            navigation_step = (execution.last_navigation or {}).get('step')
            if navigation_step != region_check_navigation:
                region_check_navigation, region_check_attempts = navigation_step, 0
            has_region_reference = any(region.get('reference_step') is not None for region in regions)
            if (settings.memory.compare_regions_when_stopped and has_region_reference and not target
                    and not execution.recovery_required and region_check_attempts < 4
                    and (execution.last_action or {}).get('action') in ('navigate', 'look', 'orient')
                    and scope.topology.region_comparison_needed(view, execution)):
                region_check_attempts += 1
                check_started = time.monotonic()
                try:
                    async with asyncio.timeout(min(12., settings.llm.request_timeout_s)):
                        comparison_frames = scope.topology.comparison_references(view, regions)
                        crossing_frames = [execution.last_navigation, *trajectory.navigation_frames(step)]
                        comparison, comparison_usage = await compare_regions(
                            scope.model, jpeg, store.load_images(record, comparison_frames), settings.llm,
                            current_view=view, navigation_frames=store.load_images(record, crossing_frames),
                        )
                    model_context['exploration_memory'] = exploration_context(
                        scope.topology, view, context_execution, comparison=comparison.model_dump(),
                    )
                    store.event(record, 'region_comparison', step=step, image=view['image'],
                                reference_images=[frame['image'] for frame in comparison_frames],
                                navigation_images=[frame['image'] for frame in crossing_frames],
                                report=comparison.model_dump(), usage=comparison_usage,
                                seconds=time.monotonic() - check_started)
                except Exception as exc:
                    # 对照不可用时保留普通停车决策；取消继续由外层传播。
                    store.event(record, 'region_comparison_error', step=step,
                                error=type(exc).__name__, seconds=time.monotonic() - check_started)

            # 先完成房间对照，再召回该房间的入口原图和检查结论，避免仍展示出发房间。
            regions = model_context['exploration_memory']['regions'][:3]
            region_steps = {region['reference_step']: region['id'] for region in regions
                            if region.get('reference_step') is not None}
            region_frames = [{**frame, 'region_id': region_steps[frame['step']]}
                             for frame in spatial.frames(list(region_steps))]
            # 待查入口与已走入口分别召回；图像对照只给证据，不代替主模型选路。
            topology_context = model_context['exploration_memory']
            route_references = []
            if topology_context.get('next_frontier') and not target:
                route_references.append(topology_context['next_frontier'])
            known_passages = [route for route in topology_context['reviewed_routes']
                              if route.get('destination_region')
                              and topology_context['focus_region'] in (
                                  route['region_id'], route['destination_region'])][:3]
            if not target:
                route_references.extend(known_passages)
            route_frames = []
            for route in route_references:
                steps = [route['source_step']]
                if route.get('destination_step') is not None:
                    steps.append(route['destination_step'])
                for frame in spatial.frames(steps):
                    route_frames.append({**frame, 'route_id': route['id'],
                                         'region_id': route['region_id'],
                                         'destination_region': route.get('destination_region'),
                                         'at_destination': frame['step'] == route.get('destination_step')})
            known_frames = [frame for frame in route_frames
                            if frame.get('destination_region') and not frame.get('at_destination')]
            if (settings.memory.compare_regions_when_stopped and known_frames and not target
                    and not execution.recovery_required):
                check_started = time.monotonic()
                try:
                    async with asyncio.timeout(min(12., settings.llm.request_timeout_s)):
                        route_match, match_usage = await compare_routes(
                            scope.model, jpeg, store.load_images(record, known_frames), settings.llm,
                        )
                    topology_context['visible_known_routes'] = route_match.model_dump()['matches']
                    matched_ids = {item.route_id for item in route_match.matches}
                    route_frames.sort(key=lambda frame: (
                        frame['route_id'] not in matched_ids, bool(frame.get('at_destination')),
                    ))
                    store.event(record, 'route_comparison', step=step, image=view['image'],
                                reference_images=[frame['image'] for frame in known_frames],
                                report=route_match.model_dump(), usage=match_usage,
                                seconds=time.monotonic() - check_started)
                except Exception as exc:
                    store.event(record, 'route_comparison_error', step=step,
                                error=type(exc).__name__, seconds=time.monotonic() - check_started)

            manifest = image_manifest(
                store.run_directory(record), view,
                context_execution['exploration_session']['views'], execution.last_navigation,
                memory_frames=[*((model_context['spatial_memory'] or {}).get('semantic_frames', [])),
                               *([task_anchor] if settings.harness.evidence_completion else [])],
                target_frames=target.get('evidence_frames', []),
                recent_views=execution.recent_views if target else (),
                region_frames=region_frames, route_frames=route_frames,
                navigation_frames=trajectory.navigation_frames(step),
            )
            if watch.pending:
                pending = watch.pending
                references = [{'image': pending['image'], 'uses': [{
                    'purpose': '检测触发原图，仅为历史候选证据，当前可见性以停车图 A 为准',
                }]}]
                if pending.get('boxed_image'):
                    references.append({'image': pending['boxed_image'], 'uses': [{
                        'purpose': '同一触发图的检测框，框位置见 target_watch.pending.bbox；不是当前图',
                    }]})
                manifest = references + manifest
            if execution.recovery_required:
                manifest = []

            # 使用已组装的空间证据与动作经历；强制脱困时不调用模型。
            record['phase'] = 'deciding'
            store.event(
                record, 'decision_input', step=step,
                image='images/' + image_name,
                model_context=model_context,
                reference_image=view['reference_image'],
                images=manifest,
            )
            if execution.recovery_required:
                # 卡住时强制退回，不叫模型
                decision = Decision(
                    action='recover',
                    reason='程序检测到受阻或原地空转，先沿实际路径退回再观察',
                )
                usage = {}
            else:
                prompt_text = fixed_prompt if fixed_prompt is not None else read_prompt(settings.prompt)
                prompt_text = navigation_prompt(
                    prompt_text, seconds=request.navigation_seconds,
                    guide=settings.prompt.instruction_guide + '\n' + algorithm_prompt(settings.prompt, tools.look_angle_deg),
                    time_limit_enabled=settings.prompt.navigation_time_limit_enabled,
                )
                if target:
                    prompt_text += target_confirmation_prompt()
                prompt_text += decision_reminder(model_context)
                if settings.target_watch.server_url:
                    prompt_text += target_watch_prompt(watch.context())
                store.save_prompt(record, step, prompt_text)
                images = store.load_images(record, manifest)
                def remember_rejected_observation(proposal):
                    # 只保留明确填写的视觉报告，不登记候选执行、不复核未执行动作。
                    if proposal.scene.model_fields_set:
                        spatial.describe(view['id'], step, proposal, action_accepted=False)
                        store.event(
                            record, 'observation_report', step=step,
                            observation_id=f'O{step}', source='model_visual_report',
                            action_status='rejected_action', scene=proposal.scene.model_dump(),
                        )

                async def confirm_completion():
                    started = time.monotonic()
                    try:
                        report, check_usage = await verify_completion(
                            scope.model, model_context, images, settings.llm,
                            evidence_mode=settings.harness.evidence_completion,
                        )
                    except Exception as exc:
                        store.event(record, 'completion_check', step=step, confirmed=False,
                                    error=type(exc).__name__, seconds=time.monotonic() - started)
                        raise AgentError('独立完成核验无法完成，未宣布成功：' + type(exc).__name__) from None
                    store.event(record, 'completion_check', step=step, **report.model_dump(),
                                seconds=time.monotonic() - started, usage=check_usage)
                    return report

                model_started = time.monotonic()
                try:
                    choose = search.selective.decide if hasattr(search, 'selective') else decide
                    decision, usage = await choose(
                        scope.model,
                        model_context,
                        images,
                        diagnostic=lambda **data: store.event(record, 'decision_error', step=step, **data),
                        validate=validate_decision, scene_validator=search.validate_scene,
                        rejected_observation=remember_rejected_observation,
                        confirm_completion=confirm_completion if settings.execution.verify_completion else None,
                        settings=settings.llm, api_key=api_key, prompt_text=prompt_text,
                        max_error_output_chars=settings.storage.max_error_output_chars,
                    )
                except DecisionRetryExhausted as exc:
                    if replanning_used or decision_step >= request.max_steps:
                        raise
                    replanning_used = True
                    decision_recovery = str(exc)
                    store.event(record, 'decision_replan', step=step, reason=decision_recovery,
                                source='rejected_action')
                    continue
                store.event(record, 'decision_latency', step=step,
                            seconds=time.monotonic() - model_started)

            # 局部探索后首次放弃也只重看一次；与纠错共用额度，不制造无限重试。
            if (decision.action == 'give_up' and execution.last_navigation is not None
                    and not replanning_used and decision_step < request.max_steps):
                replanning_used = True
                decision_recovery = decision.reason
                store.event(record, 'decision_replan', step=step, reason=decision_recovery,
                            source='give_up')
                continue
            decision_recovery = None
            await watch.review(decision.detection_review)
            # 只执行模型明确选中的技能；公共观察工具可直接调用，不隐式继承上轮技能。
            previous_skill = record.get('active_skill')
            record['active_skill'] = decision.skill
            store.event(record, 'skill_selected', step=step, name=decision.skill,
                        previous=previous_skill, action=decision.action, reason=decision.reason,
                        source=('selective_controller' if hasattr(search, 'selective') and search.selective.owns(decision)
                                else 'runtime_recovery' if execution.recovery_required else 'model'))
            if decision.skill is not None and decision.skill != previous_skill:
                store.progress(record, 'skill', name=decision.skill)

            # 执行前确认控制权，并提交通过校验的动作
            tools.require_control()
            previous_subgoal = copy.deepcopy(execution.subgoal)
            selected_candidate = search.accept(decision)
            scope.topology.update(
                view, execution, decision,
                region_match=model_context['exploration_memory'].get('visual_match'),
            )
            if (execution.last_action or {}).get('action') in ('look', 'orient'):
                prior_scene = record.get('last_decision', {}).get('scene', {})
                store.event(record, 'observation_gain', step=step,
                            action_step=execution.last_action['step'],
                            target_seen=decision.scene.target_seen,
                            target_visibility_changed=prior_scene.get('target_seen') != decision.scene.target_seen,
                            current_candidates=[
                                item['id'] for item in execution.candidates if item['last_seen_step'] == step
                            ],
                            review=(decision.previous_action_review.model_dump()
                                    if decision.previous_action_review else None),
                            source='model_visual_report')
            scene = decision.scene
            has_description = bool(scene.model_fields_set) or decision.previous_action_review is not None
            if manifest and has_description:
                spatial.describe(view['id'], step, decision,
                                 previous_action_step=(execution.last_action or {}).get('step'))
            trajectory.review(
                step,
                decision.previous_action_review.model_dump() if decision.previous_action_review else None,
                decision.target_review.model_dump() if decision.target_review else None,
            )
            store.save_execution(record, execution)
            if hasattr(search, 'selective'):
                record['scheduling'] = search.selective.context()
            record['last_decision'] = decision.model_dump(exclude_none=True)
            store.event(record, 'decision', step=step, decision=record['last_decision'], usage=usage)

            if decision.action in ('finish', 'give_up'):
                await watch.configure(False, [])
                execution._end_subgoal(decision.action, decision.evidence or decision.reason)
                if execution.subgoal != previous_subgoal:
                    store.event(record, 'subgoal_changed', step=step, subgoal=execution.subgoal,
                                recent=execution.subgoal_history)
                outcome = execution.scan.finish(decision.action)
                record['exploration_result'] = outcome
                store.event(record, 'exploration_session_finished', **outcome)
                store.save_execution(record, execution)
            if decision.action == 'finish':  # 当前图已看到目标
                record.update(
                    status='completed',
                    summary=decision.evidence,
                    success_basis='model_visual_judgment',
                )
                return
            if decision.action == 'give_up':
                record.update(status='failed', summary=decision.reason)
                return

            record['phase'] = 'acting'
            call, action_input = search.prepare_call(decision, selected_candidate, request.navigation_seconds)
            store.event(record, 'action_input', step=step, **action_input)
            search.before_execute(decision, selected_candidate)
            if execution.subgoal != previous_subgoal:
                store.event(record, 'subgoal_changed', step=step, subgoal=execution.subgoal,
                            recent=execution.subgoal_history)
            candidate = next((item for item in execution.candidates
                              if item['id'] == selected_candidate), None)
            intent = candidate['description'] if candidate else decision.reason
            trajectory.start(step, call.name, intent, call.arguments, candidate)
            search_navigation = await scope.skills.prepare(decision, watch)
            action_started = time.monotonic()
            if call.name == 'navigate':
                result = await capture_during_navigation(
                    lambda: watch.execute(
                        lambda: scope.tools.execute(call), step,
                        search_navigation=search_navigation,
                    ), tools=tools, capture=lambda jpeg, observation: capture_passive(jpeg, observation, step),
                    interval_s=settings.harness.capture_interval_s,
                    diagnostic=lambda **data: store.event(record, 'capture_error', step=step, **data),
                )
            else:
                result = await scope.tools.execute(call)
            store.event(record, 'action_latency', step=step, action=call.name,
                        seconds=time.monotonic() - action_started)
            trajectory.result(step, result)
            search.record_result(decision, selected_candidate, result, step)
            store.save_execution(record, execution)
            if hasattr(search, 'selective'):
                record['scheduling'] = search.selective.context()
            record['last_result'] = result
            store.event(record, 'action_result', step=step, result=result)
            if result.get('startup_failed'):
                raise AgentError(f"导航服务启动失败（尝试 {len(result['attempts'])} 次），未获得有效推理结果")
            if search.navigation_errors >= settings.execution.max_consecutive_navigation_errors:
                raise AgentError(f'{settings.execution.max_consecutive_navigation_errors} consecutive navigation service errors')
        record.update(status='budget_exhausted', summary='Maximum decision steps reached')
