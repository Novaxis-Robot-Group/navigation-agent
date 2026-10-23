"""按配置启用场景导航校验；依据模型视觉报告，不是实体围栏。"""


def validate_navigation_scene(decision, policy=None):
    if policy is None:
        return
    if policy != 'go2_office':
        raise ValueError('未知的场景校验策略')
    if decision.action != 'navigate':
        return
    check = decision.navigation_check
    if check is None:
        raise ValueError('navigate 缺少 navigation_check：须检查本次终点、到该终点的路径及其上的玻璃/关闭门，不包含终点之外的未知路线')
    if check.target_area != 'allowed':
        raise ValueError('本次终点位于有人使用的工位禁区或允许性不明；选择禁区外可见观察位置，普通地毯不属于禁区')
    if check.path_area != 'allowed':
        raise ValueError('到本次终点的路径涉及人员工位禁区或边界不明；换合规路径或近处观察位置')
    if check.passage != 'clear':
        raise ValueError('到本次终点的路径存在玻璃/关闭门或开口无法确认；先选障碍前可接近的观察位置，不穿越未知入口')
    # 狭窄程度单独报告，不能把“看得到地面”等同于适合钻入。
    if check.space == 'uncertain':
        raise ValueError('navigation_check.space 未确认；须依据当前图检查本次终点和接近路径是否狭窄')
    if check.space == 'narrow':
        selected = next((item for item in decision.candidates if item.id == decision.candidate_id), None)
        if (selected is None or selected.kind != 'target' or decision.scene.target_seen != 'yes'):
            raise ValueError('探索时不能钻入狭窄空间；只有当前图明确看到目标位于其中，才考虑进入接近')


def scene_validator(scene):
    if scene is None or scene.navigation_policy is None:
        return None
    from functools import partial
    return partial(validate_navigation_scene, policy=scene.navigation_policy)
