"""目标接近与核实技能：已有线索时直接行动，不让检测反复中断跟进。"""


class ApproachTargetSkill:
    name = 'approach_target'
    description = '已有明确或疑似目标物体，接近、核实身份，或在暂时遮挡后局部找回。'
    actions = ('navigate', 'look', 'orient', 'recover', 'finish', 'give_up')
    instructions = (
        '依据当前画面和可靠目击选择直接导航或定向观察，省略 target_labels。'
        '此技能在动作前关闭目标监测并取消在途检测，不要求先完全确认身份。'
        '仅疑似时用 viewpoint 核实；已确认才登记 target 候选。'
        '目标暂失时按连续画面局部找回，不能盲走向历史位置。'
        '核实用户要求的类别、颜色和关系；找到与走到分别按原任务验收。'
        '排除候选或决定暂缓局部核实后，可重新选择 search_navigation。'
    )

    def validate(self, decision, *, detector_available):
        # 模型可能沿用搜索类别；它是闲置参数，不能阻断明确选择的接近动作。
        return None

    async def prepare(self, decision, watch):
        await watch.configure(False, [])
        return False
