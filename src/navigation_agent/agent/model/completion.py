"""结合当前画面与本轮记忆独立核验完成，不直接采信主决策结论。"""

import base64
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class CompletionReport(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    evidence: str = Field(min_length=1, max_length=700)
    confirmed: bool = Field(strict=True)


class Criterion(BaseModel):
    """身份、到达和最终朝向分别提供视觉证据。"""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    status: Literal['yes', 'no', 'uncertain']
    evidence: str = Field(min_length=1, max_length=500)


class EvidenceCompletion(BaseModel):
    model_config = ConfigDict(extra='forbid')
    identity: Criterion
    arrival: Criterion
    facing: Criterion

    def report(self):
        criteria = (self.identity, self.arrival, self.facing)
        return CompletionReport(
            confirmed=all(item.status == 'yes' for item in criteria),
            evidence='；'.join(f'{name}: {item.evidence}' for name, item in zip(
                ('目标实例', '到达', '朝向'), criteria))[:700],
        )


async def verify_completion(client, model_context, images, settings, *, evidence_mode=False):
    # 复用本轮已召回的记忆和原图，不把待核验的 finish 结论作为证据。
    context = {'operation': 'verify_completion'}
    for name in ('goal', 'observation_step', 'observation_id', 'observation',
                 'spatial_memory', 'trajectory_memory', 'exploration_memory', 'task_contract'):
        if name in model_context:
            context[name] = model_context[name]
    execution = model_context.get('execution') or {}
    context['execution'] = {
        name: execution[name]
        for name in ('target_tracking', 'last_action', 'last_navigation')
        if name in execution
    }
    content = [{'type': 'text', 'text': json.dumps(context, ensure_ascii=False)}]
    # 沿用主决策的去重顺序和观察编号；当前图仍放在最后。
    for frame in images:
        content.extend([
            {'type': 'text', 'text': json.dumps(
                {key: value for key, value in frame.items() if key != 'jpeg'}, ensure_ascii=False,
            )},
            {'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + base64.b64encode(frame['jpeg']).decode(),
            }},
        ])
    body = {
        'model': settings.model, 'enable_thinking': False, 'stream': False,
        'max_tokens': min(settings.max_tokens, 900 if evidence_mode else 450),
        'response_format': {'type': 'json_object'},
        'messages': [
            {'role': 'system', 'content': (
                '这些是机器人第一人称相机图，图片标注区分历史与当前观察；当前图像 A 是本轮停车图。'
                '相机所在位置就是机器人位置，不需要机器人自身出现在画面。'
                '结合当前图、历史原图和记忆独立判断任务是否完成；记忆中的模型判断可能有误，须与图像及真实动作核对。'
                '先核实目标身份：检查类别、形状及用户指定特征，颜色相近不代表类别相同。'
                '历史图可提供清晰的身份依据；接近后当前图只见局部时，结合前后图像、位姿变化和动作结果判断是否为同一目标，'
                '不要仅因目标过近而要求当前图重新展示完整外形；有矛盾或无法关联同一目标时不确认。'
                '再依据用户完整目标核实完成条件，不只匹配单个关键词；'
                '只要求找到或辨认时，当前可见部分结合历史证据足以确认目标即可，不增加接近要求；要求走到旁边、靠近或特定位置时，还须有对应到达证据，并与接近过程一致，远处看见不能证明到达。'
                '历史目击、机器人位移或 VLN STOP 单独都不能证明已到达；拍摄位置不是目标坐标。'
                '目标完全不可见或证据不足时返回 false，不推测遮挡处内容。'
                '只返回 JSON：{"evidence":"中文说明身份、当前完成依据或具体缺少的证据","confirmed":false}。'
                '先证据后结论；未确认到达不等于否定此前的目标身份。'
            )},
            {'role': 'user', 'content': content},
        ],
    }
    if evidence_mode:
        # 三项证据模式只声明一种 JSON 协议，成功判据保持不变。
        body['messages'][0]['content'] = body['messages'][0]['content'].replace(
            '目标完全不可见或证据不足时返回 false，不推测遮挡处内容。',
            '目标完全不可见或证据不足时相关项为 uncertain，不推测遮挡处内容。',
        ).replace(
            '只返回 JSON：{"evidence":"中文说明身份、当前完成依据或具体缺少的证据","confirmed":false}。', '')
        body['messages'][0]['content'] += (
            '\n本次逐项核验，不输出总体成功结论。先确认是否是原始指令指定的同一实例，'
            '逐项检查方向、计数方向、序数、极值和参照关系；同类物体或颜色相同不够。'
            '初始左右关系以任务起始视角为准，不能按转向后的左右重新选择目标。'
            '再检查到达：目标仅在远处可见、只因移动了一段距离或局部物体变大，都不能证明已到达。'
            '检查当前目标与机器人可通行地面的接近关系；无法区分目标实例或距离则 uncertain。'
            '最后检查相机当前朝向目标，目标只在历史图、画外或机器人背后不算满足。'
            '只要求找到而无需接近的任务，到达项可 yes，但证据须说明原任务不要求接近。'
            '目标首次目击原图是待验证锚点，不保证正确，不可用历史文字断言代替本次视觉检查。'
            '只返回 JSON，identity、arrival、facing 三项均含 status=yes/no/uncertain 和 evidence。'
        )
    response = await client.complete(body)
    choice = response['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('独立完成核验未完整返回')
    if evidence_mode:
        detailed = EvidenceCompletion.model_validate_json(choice['message']['content'])
        report = detailed.report()
        return report, {**response.get('usage', {}), 'criteria': detailed.model_dump()}
    return CompletionReport.model_validate_json(choice['message']['content']), response.get('usage', {})
