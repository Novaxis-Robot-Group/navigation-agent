"""解析模型决策并调用业务校验，反馈拒绝原因与本轮动作条件。"""

import base64
import json
from pydantic import ValidationError
from navigation_agent.agent.config import LlmSettings
from navigation_agent.agent.schemas import Decision, AgentError
from navigation_agent.agent.tools.navigation import validate_instruction


class DecisionRetryExhausted(AgentError):
    """本轮动作纠错耗尽，由任务循环决定是否重新观察一次。"""


def missing_action_fields(content, model_context):
    """纠错时一次列明动作依赖的漏项，不创建候选或补写视觉结论。"""
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    try:
        payload = Decision.normalize_output_fields(payload)
    except ValueError:
        # 无效空字段仍交由原校验拒绝，其余漏项可继续一起报告。
        pass
    action = payload.get('action')
    if not isinstance(action, str):
        return []
    required = ['reason']
    required.extend({
        'look': ['direction'], 'orient': ['view_id', 'observation_intent'],
        'navigate': ['instruction', 'candidates', 'candidate_id'], 'finish': ['evidence'],
    }.get(action, []))
    if action == 'navigate' and model_context.get('available_skills'):
        required.append('skill')
        if payload.get('skill') == 'search_navigation' and model_context.get('target_watch', {}).get('available'):
            if 'target_labels' not in payload:
                required.append('target_labels')
    if (model_context.get('execution', {}).get('target_tracking')
            and action in ('navigate', 'finish')):
        required.append('target_review')
    missing = [[field] for field in required if not payload.get(field)]
    if action == 'navigate' and not payload.get('candidates'):
        # 整个列表漏填时，也一次说明新建候选每项的必填字段。
        missing.extend(['candidates', '*', field]
                       for field in ('id', 'kind', 'description', 'evidence', 'expected_new_view'))
    if action == 'navigate' and isinstance(payload.get('candidates'), list):
        for index, candidate in enumerate(payload['candidates']):
            if not isinstance(candidate, dict):
                continue
            fields = ['id', 'kind', 'description', 'evidence']
            if candidate.get('kind') in ('open_passage', 'viewpoint'):
                fields.append('expected_new_view')
            missing.extend(['candidates', index, field] for field in fields if not candidate.get(field))
    return missing


def repair_field_schema(errors):
    """从实际数据模型提取错误字段结构，纠错不维护另一份手写协议。"""
    schema = Decision.model_json_schema()
    fields = {error['field'][0] for error in errors if error.get('field')}
    properties = {name: definition for name, definition in schema['properties'].items()
                  if name in fields}
    definitions = {}

    def include_references(value):
        if isinstance(value, dict):
            reference = value.get('$ref', '')
            if reference.startswith('#/$defs/'):
                name = reference.removeprefix('#/$defs/')
                if name not in definitions:
                    definitions[name] = schema['$defs'][name]
                    include_references(definitions[name])
            for item in value.values():
                include_references(item)
        elif isinstance(value, list):
            for item in value:
                include_references(item)

    include_references(properties)
    result = {'type': 'object', 'properties': properties}
    if definitions:
        result['$defs'] = definitions
    return result


async def decide(
    client,
    model_context,
    images,
    diagnostic=None,
    *, settings: LlmSettings, api_key: str, prompt_text: str, max_error_output_chars: int,
    validate=None, scene_validator=None, rejected_observation=None, confirm_completion=None,
):
    # 组装当前观察和探索上下文
    body = {
        'model': settings.model.strip(),
        'messages': [
            {'role': 'system', 'content': prompt_text},
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': json.dumps(
                            model_context,
                            ensure_ascii=False,
                            separators=(',', ':'),
                        ),
                    },
                ],
            },
        ],
        'stream': False,
        'enable_thinking': settings.enable_thinking,
        'response_format': {'type': 'json_object'},
        'max_tokens': settings.max_tokens,
    }

    for frame in images:
        body['messages'][1]['content'].extend([
            {'type': 'text', 'text': json.dumps(
                {key: value for key, value in frame.items() if key != 'jpeg'}, ensure_ascii=False,
            )},
            {'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + base64.b64encode(frame['jpeg']).decode(),
            }},
        ])

    # 格式纠错与一次完成否定反馈分别计数；均使用同一原图，不伪造动作。
    retries_left = settings.format_retries
    completion_evidence = None
    for attempt in range(1, settings.format_retries + 3):
        data = await client.complete(body)
        content, finish_reason = None, None
        decision = None
        checking_exploration = False
        checking_scene = False
        checking_completion = False
        new_completion_feedback = False
        try:
            choice = data['choices'][0]
            finish_reason = choice.get('finish_reason')
            content = choice['message']['content']
            if finish_reason != 'stop':
                raise ValueError(
                    '模型输出达到本次输出 token 上限，回答未完整生成'
                    if finish_reason == 'length'
                    else f'模型响应未正常结束（finish_reason={finish_reason}）'
                )

            # 先校验动作格式，再检查探索约束
            decision = Decision.model_validate_json(
                content, context={'observation_step': model_context.get('observation_step')},
            )
            if decision.action == 'navigate':
                validate_instruction(decision.instruction)
            checking_scene = True
            if scene_validator is not None:
                scene_validator(decision)
            checking_scene = False

            if validate is not None:
                checking_exploration = True
                validate(decision)
                checking_exploration = False
            if decision.action == 'finish' and confirm_completion is not None:
                checking_completion = True
                if completion_evidence is not None:
                    raise ValueError('同一画面已未通过完成核验，不能仅修改措辞再次 finish；'
                                     '根据核验意见选择观察或继续探索：' + completion_evidence)
                confirmation = await confirm_completion()
                if not confirmation.confirmed:
                    completion_evidence = confirmation.evidence
                    new_completion_feedback = True
                    raise ValueError('独立完成核验未确认完成：' + confirmation.evidence
                                     + '。这不证明目标不存在；重新选择有依据的观察或探索动作。')
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            if isinstance(exc, ValidationError):
                errors = [
                    {'field': list(error['loc']), 'type': error['type'], 'message': error['msg']}
                    for error in exc.errors(include_input=False, include_context=False)
                ]
            else:
                errors = [
                    {
                        'type': type(exc).__name__,
                        'message': (
                            str(exc) if type(exc) is ValueError else 'Invalid response structure'
                        ),
                    }
                ]
            reported_fields = {tuple(error.get('field', [])) for error in errors}
            for field in missing_action_fields(content, model_context):
                if tuple(field) not in reported_fields:
                    message = ('* 表示候选列表中的每一项；须依据当前图填写该字段'
                               if '*' in field else '当前动作缺少此字段；须依据同一组当前图片补齐')
                    errors.append({'field': field, 'type': 'missing_action_field',
                                   'message': message})
            raw = content if isinstance(content, str) else repr(content)

            # 原始输出和具体校验信息均去掉 Key，再用于日志、纠错和最终报错。
            key = api_key
            if key:
                raw = raw.replace(key, '[REDACTED]')
                errors = json.loads(json.dumps(errors, ensure_ascii=False).replace(key, '[REDACTED]'))
            if checking_completion:
                category = 'completion_verification'
            elif checking_scene:
                category = 'scene_rule'
            elif checking_exploration:
                category = 'exploration_rule'
            else:
                category = 'output_format'

            options = model_context.get('execution', {}).get('action_options', {})
            if key:
                options = json.loads(json.dumps(options, ensure_ascii=False).replace(key, '[REDACTED]'))
            if decision is not None and rejected_observation is not None and not checking_completion:
                # 被拒报告进入记忆前同样脱敏；保留字段是否实际填写的信息。
                report = decision.model_dump_json(exclude_unset=True)
                if key:
                    report = report.replace(key, '[REDACTED]')
                try:
                    observed = Decision.model_validate_json(report)
                except ValidationError:
                    observed = None
                if observed is not None:
                    rejected_observation(observed)
            if diagnostic:
                diagnostic(
                    attempt=attempt,
                    finish_reason=finish_reason,
                    category=category,
                    raw_output=raw[:max_error_output_chars],
                    truncated=len(raw) > max_error_output_chars,
                    validation_errors=errors,
                    usage=data.get('usage', {}) if isinstance(data, dict) else {},
                    action_options=options,
                )
            if not new_completion_feedback and retries_left == 0:
                if checking_completion:
                    raise AgentError(errors[0]['message']) from None
                if checking_scene:
                    raise AgentError(f'Fixed scene policy rejected navigation: {errors[0]["message"]}') from None
                if checking_exploration:
                    raise DecisionRetryExhausted(f'No valid exploration action after {attempt} attempts: {errors[0]["message"]}') from None
                details = []
                for error in errors[:3]:
                    field = '.'.join(str(part) for part in error.get('field', []))
                    message = error.get('message', error['type'])
                    details.append(f'{field}: {message}' if field else message)
                detail = '; '.join(details)[:600]
                raise DecisionRetryExhausted(f'模型回答未通过校验（共 {attempt} 次）：{detail}') from None
            if not new_completion_feedback:
                retries_left -= 1
            completion_advice = (
                '本帧完成核验未通过：不要保留未获支持的目标断言，不再 finish；'
                '根据核验指出的缺口选择观察或继续探索；未确认到达不等于目标身份错误，保留仍有效的历史证据。'
                if completion_evidence is not None else
                'finish 提供当前完成 evidence；已在 target_review 的 arrived 复核中写明则无需重复，不因缺字段改为导航或放弃。'
            )
            # 保留上次视觉描述，动作被拒绝不等于观察到的线索无效。
            body['messages'].append({
                'role': 'assistant', 'content': raw[:max_error_output_chars],
            })
            body['messages'].append(
                {
                    'role': 'user',
                    'content': '上次动作未执行。请根据同一组图片纠正动作，保留仍有效的 scene/candidates，不要虚构执行结果。'
                    'action 为 look/orient/navigate/recover/finish/give_up；reason 必填。'
                    'navigate 同轮选择 available_skills 中的 skill；搜索时为 search_navigation，'
                    '接近或核实明确/疑似目标时为 approach_target；不单独调用监测开关。'
                    'reason、evidence、scene 与记忆描述用中文，仅 instruction 用英文。'
                    'target_review 仅含当前 observation_step、status、evidence；status 只允许 '
                    'visible/arrived/temporarily_lost/rejected。疑似目标尚未确认身份时省略 target_review，'
                    '用 scene.target_seen=uncertain 与 viewpoint 候选核实，不写 tracking。'
                    '局部找回预算用尽时以暂失目标复核加 subgoal_review=abandoned 暂缓跟进，继续探索，不仅因此 give_up。'
                    'look 选择当前 look_directions 并说明待确认问题；'
                    '目标局部重复观察或 orient 时给 observation_intent 的参考图、具体问题和新依据。'
                    '不能把已看方向说成未看过；orient 只能选当前可用的 view_id，空列表表示不能转回，转回后须取新图。'
                    'navigate 给当前可见 candidates、candidate_id、英文 instruction；'
                    'candidates 每项必含 id、kind、description、evidence，旧 C 候选重新可见须填 evidence_source=current；顶层 evidence 不能代替候选 evidence；'
                    '同一候选沿用上下文的 C 编号，新发现候选用 new_1 至 new_4，不得杜撰新的 C 编号；'
                    'open_passage/viewpoint 候选另给 expected_new_view，'
                    '并检查本次终点及路径的场景规则，不得引用历史图直接导航。'
                    + completion_advice
                    + 'new_area 是视觉复核结论，不授权运动。可直接前往当前可见候选；有进展时延续目的，不强制环视。'
                    '当前动作选项：' + json.dumps(options, ensure_ascii=False)
                    + '。具体校验信息：' + json.dumps(errors, ensure_ascii=False)
                    + '。以下只列出出错字段的真实结构（不是可照抄的视觉内容）；'
                    '同时满足上方按动作列出的漏项，保留其余有效字段：'
                    + json.dumps(repair_field_schema(errors), ensure_ascii=False),
                }
            )
            continue
        return decision, data.get('usage', {})
