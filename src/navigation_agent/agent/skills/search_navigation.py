"""搜索导航技能：模型选择搜索目的，工具管理随导航启停的目标监测。"""


class SearchNavigationSkill:
    name = 'search_navigation'
    description = '尚无具体目标物体线索，前往有价值入口或观察点，沿途寻找目标。'
    actions = ('navigate', 'look', 'orient', 'recover', 'finish', 'give_up')
    instructions = (
        '先查看明显区域，有足够依据就导航，不必转满一圈。'
        'navigate 时在同一决策填写 target_labels：从任务提取简单英文物体类别，'
        '不带颜色、关系或导航指令。类别由模型选择，不按场景硬编码。'
        '无法用简单物体类别表达时填 []；未配置或不支持检测时照常导航。'
        '此技能在搜索导航中自动开启监测，命中立即停车并返回检测证据。'
        '检测停车后先复核；已有明确或疑似目标物体时改选 approach_target。'
        '只看到可能的房间或入口仍属于搜索；原地观察与恢复不运行检测。'
    )

    def validate(self, decision, *, detector_available):
        if decision.action == 'navigate' and detector_available and decision.target_labels is None:
            raise ValueError(
                'search_navigation 的 navigate 须同轮填写 target_labels（简单英文物体类别）；'
                '目标不适合类别检测时填 []，不要另调用监测开关。'
            )

    async def prepare(self, decision, watch):
        enabled = (decision.action == 'navigate' and bool(watch.settings.server_url)
                   and bool(decision.target_labels))
        await watch.configure(enabled, decision.target_labels if enabled else [])
        return enabled
