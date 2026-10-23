"""读取和渲染提示词，保留热读和固定快照语义。"""

import json

def read_prompt(settings):
    parts = [settings.path.read_text(encoding='utf-8')]
    if settings.scene is not None:
        parts.append(settings.scene.path.read_text(encoding='utf-8'))
    return '\n\n'.join(parts)


def algorithm_prompt(settings, look_angle_deg):
    """固定算法规则独立于用户指南，使用当前设备实际转角。"""
    if settings.algorithm_path is None:
        return ''
    return settings.algorithm_path.read_text(encoding='utf-8').replace(
        '{{look_angle_deg}}', f'{look_angle_deg:g}')


def navigation_prompt(prompt, *, seconds, guide, time_limit_enabled=True):
    """将所选 VLN 指南和实际执行时限注入通用 Prompt。"""
    time_limit = f'最多 {seconds:g} 秒' if time_limit_enabled else '不设固定上限'
    # 先替换执行参数，再插入原样指南，避免解释指南中的占位符。
    return (prompt.replace('{{navigation_time_limit}}', time_limit)
            .replace('{{instruction_guide}}', guide))


def target_confirmation_prompt():
    """目标跟进只补充本阶段必填项，不把疑似目击写成已确认身份。"""
    return """
当前有待验证目标线索，历史报告可能误识别。先核对本轮任务所需类别、颜色和形状，
再判断远近；同色的其他物体不是目标，应 rejected。当前图 A 才证明现在可见。
近处物体可能只露部分部件，不要求完整轮廓，但部件须足以支持身份判断。
navigate/finish 前填最外层 target_review：
{"observation_step":当前观察编号,"status":"visible|arrived|temporarily_lost|rejected","evidence":"当前图的具体依据"}
visible：当前仍可见，导航须登记当前候选。
arrived：当前已满足原任务，finish 可复用本条 evidence；远处可见不等于走到旁边。
temporarily_lost：保留线索，定向确认或选当前可见 viewpoint，不能盲走向旧目标。
rejected：有证据否定原识别后继续探索；一帧未见不足以否定。
走偏、继续检查无收益或局部预算用尽仍未见目标时，保持 temporarily_lost，并用 subgoal_review=abandoned 暂缓这次跟进；保留历史线索，继续查看当前未查部分或有价值入口，不能仅因此 give_up。
不要因历史把它称为目标就相信身份正确；重复检查没有进展时换观察位置。
"""


def decision_reminder(context):
    """把当前任务与本帧必填项放在系统提示末尾，避免历史线索或示例替代任务。"""
    execution = context['execution']
    target = execution.get('target_tracking')
    required = (
        '已有目标跟进。navigate/finish 须填写顶层 target_review，observation_step='
        + str(context['observation_step'])
        if target else '尚无目标跟进记录，通常省略 target_review；若填写只能复核当前帧的明确目击。'
    )
    required += ('根据用户完整目标和当前证据判断完成条件，不用单个关键词代替任务理解。'
                 '只要求找到或辨认时，确认当前目标身份可以 finish；'
                 '要求走到旁边、靠近或特定位置时，须有相应到达证据，否则继续 approach_target。'
                 'target_review 的 arrived 表示已满足原任务，不统一解释为必须靠近。')
    if target:
        motion = target.get('camera_motion_since_sighting')
        if motion:
            required += ('相机相对最后目击已平移' + str(motion['translation_m']) + '米、转向'
                         + str(motion['heading_change_deg']) + '度；历史图的左/右不能直接当作当前左/右，'
                         '该运动量不表示目标当前位置或方向。')
        required += ('先核对 last_seen_anchor 中目标与门框、墙角、家具的相对关系，'
                     '再沿近期连续图片检查它如何被遮挡。不同门洞不能因同为白色就合并。'
                     '当前未见目标时进入局部找回：优先向最后目击一侧 look；'
                     '也可选择当前可见 viewpoint 短距离改变遮挡，禁止继续穿新的 open_passage。'
                     '新房间、新门或机器人位移不代表更接近目标。'
                     'look 也应填写当前 target_review；只有当前重新看见才填 visible，不能用历史图。'
                     '局部找回剩余动作数：' + str(target.get('reacquire_actions_remaining', 4))
                     + '；这是上限，不是必须完成的次数。已走偏或继续检查无收益可提前暂缓；保持 target_review=temporarily_lost，'
                     '同时填 subgoal_review={status:abandoned,evidence:具体暂缓原因}，保留最后可靠目击并恢复探索。'
                     '选择当前未查方向、可见观察点或有价值入口；不得把未再看到说成目标不存在。')
    last_action = execution.get('last_action') or {}
    navigation = execution.get('last_navigation') or {}
    if last_action.get('action') == 'navigate':
        required += (
            '本轮先填写 previous_action_review={outcome:progress/no_new_information/uncertain,evidence:当前视觉依据}。上一段要解决的问题是'
            + navigation.get('intent', '上次所选目的') + '；工具结束原因='
            + str((last_action.get('result') or {}).get('reason', 'unknown'))
            + '。对照导航起点、途中图和当前停车图，说明实际到了哪里、原问题获得了什么答案。'
            '只发生位移、STOP、换了视角或又看见旧房间不能写 progress；无新证据填 no_new_information，'
            '看不清则 uncertain 并定向核实。走偏后放弃旧局部终点，再按当前画面选目的；'
            '原问题已解决就 completed，不为完成旧措辞继续穿门。'
            '先辨认本次来路；转向后它可能出现在画面左、右或前方，不因此变成未知入口。'
        )
    subgoal = execution.get('active_subgoal')
    if subgoal:
        required += ('当前持续目的：' + subgoal['description'] + '；待回答问题：' + subgoal['question']
                     + '。填写 subgoal_review（status=continue/completed/abandoned、evidence），'
                     '已经看清该问题就 completed；原问题不值得继续就 abandoned，不要用新局部终点替换原问题。')
    if (context.get('trajectory_memory') or {}).get('local_paths'):
        required += (
            'trajectory_memory.local_paths 是附近实际走过的路径，已按本轮位置和朝向换算：'
            'relative_bearing_deg 为当前身体正前方 0 度、左正右负；toward_start/toward_end 表示沿历史点序朝起点/终点的一侧。'
            '先结合当前图辨认来路与其他旧路，再对照拓扑的已查结论和待查入口选路；'
            'action_step 对应同一段导航及拓扑 connections 的记录。'
            '路径方向只指附近实测点，不证明当前入口身份、通行条件或整条路线可直达；没记录也不等于可走。'
        )
        if not target:
            required += ('无新线索时优先当前图支持的未查入口或观察点；'
                         '若走旧路，在 reason 写明通行、补查或核实的具体目的。')
    topology = context.get('exploration_memory') or {}
    regions = topology.get('regions', [])
    if regions:
        previous = topology['previous_region']
        matched = (topology.get('visual_match') or {}).get('region_id')
        # 停车对照指向旧房间时，突出该房间的历史结论；不提前修改拓扑归属。
        current = next((region for region in regions if region['id'] == matched), None)
        if current is None:
            current = next((region for region in regions if region['id'] == previous), regions[0])
        revisiting = matched == current['id'] and matched != previous
        search = current.get('search', {})
        references = [f"{region['id']}=O{region['reference_step']}" for region in regions[:3]
                      if region.get('reference_step') is not None]
        required += (
            '房间历史参照索引：' + '、'.join(references) + '。只按本轮实际附带的原图核对，未附图的索引仅作线索。结合导航起点、途中原图和当前图判断是否已穿门；上次区域 '
            + previous + ' 只是上一步假设，转头不换房间，跨门或返回须复核家具与入口关系。'
            '本轮重点召回区域 ' + current['id'] + ' 的历史检查状态=' + search.get('status', 'unknown')
            + '；历史具体未查项=' + json.dumps(search.get('remaining', []), ensure_ascii=False)
            + '。只有当前图与该区域相符才沿用结论；若已进入其他区域，不继承这个状态。'
            '主要地面已查且只剩隔壁未知时，当前区填 checked、remaining=[]；'
            '然后选未查入口。traversed 入口已有明确的房间连接，可通行但不能当作未查入口或换名重试。'
        )
        incoming = next((edge for edge in reversed(topology.get('connections', []))
                         if edge['destination'] == current['id']), None)
        if incoming:
            required += (
                '该区历史已走来路：' + incoming['source'] + '→' + incoming['destination']
                + '，起点O' + str(incoming['start_step']) + '、停车O' + str(incoming['end_step'])
                + '。结合本轮附带的原图核对返回方向；上次选的是空地观察点也可能已穿门，'
                '不能因此把来路重新当成未知入口。需要返回时说明具体未查部分、新线索或通行目的。'
            )
        if search.get('previous_check'):
            required += ('该区域此前已查记录：' + json.dumps(search['previous_check'], ensure_ascii=False)
                         + '；若这次确有新疑点，只补查该问题，不从头搜索。')
        if current.get('visits', 0) > 1 or revisiting:
            required += '若当前图证实在该区：此前已访问' + str(current['visits']) + '次。'
            if search.get('status', 'unknown') in ('visited', 'unknown'):
                required += '此前仅路过或缺乏检查结论，不能视为已查完；先查明显区域并选择有价值入口。'
            else:
                required += ('保留已有检查结论，按具体未查项补查、核实新线索或通行；'
                             '不能因回来换了视角就从头重搜整间房。')
    frontier = topology.get('next_frontier')
    if frontier and not target:
        required += (
            '近期房间假设经过：' + '→'.join(topology.get('recent_region_path', [])) + '。'
            '仍未尝试的入口：' + frontier['id'] + '，位于' + frontier['region_id']
            + '，见于O' + str(frontier['source_step']) + '（' + frontier['description'] + '）。'
            '这是可返回的待查项；若当前已有更有价值的新入口或具体线索，先检查当前候选，不必立刻折返。'
            '当前缺少有价值的待查项时再按路径找回；已走通入口用于通行，不因换名重新变成未知入口。'
            'return_path 以 focus_region（本轮对照后的召回焦点）为起点；它仍是视觉假设，按当前图确认后继续。'
            '找回旧入口先观察，不能用旧图直接导航。'
        )
    route_matches = topology.get('visible_known_routes')
    if route_matches:
        required += (
            '当前画面中独立对照出的已走入口：' + json.dumps(route_matches, ensure_ascii=False) + '。'
            '结合原图确认这些对应；相同入口沿用该 C 编号，去向见 reviewed_routes，不能重新称为未知入口。'
            '已走入口可以通行，但须说明去做哪个未完成问题或追随哪条新线索；'
            '没有具体用途时优先当前有价值的新入口或 next_frontier。'
            '未匹配的入口仍须你判断，未匹配不证明它是新入口。'
        )
    comparison = topology.get('visual_match')
    if comparison:
        required += (
            '本轮停车原图对照：' + json.dumps(comparison, ensure_ascii=False) + '。'
            '该对照没有读取旧候选文字，将与实际位移共同用于本轮拓扑归属。'
            'spatial_update 的归属按此对照填写；uncertain 时沿用 previous_region 作为占位、transition=uncertain，'
            '这不确认房间身份，不把 uncertain 当作 region_id，也不凭旧任务猜房间。'
            '你负责报告该房间的检查范围并选择动作。先确认房间，再选入口，'
            '不因原目的尚未完成而沿用上次区域；new 时按当前画面检查明显区域并选择有价值入口。'
            '对照仍是模型视觉判断，不是位置真值；看不清时优先转向获取房间内部特征。'
        )
    options = execution['action_options']
    if options['recover_allowed']:
        required += ('当前允许 recover：若已看过周围却仍被家具遮挡、没有可见路线，'
                     '沿设备记录的实际来路退回后重新观察，不继续在旧朝向间反复 orient。')
    unseen = [direction for direction, item in options['look_options'].items()
              if item['repeat_of'] is None]
    orient_options = [
        item for item in options.get('orient_options', []) if item.get('allowed', True)
    ]
    return (
        '\n\n本轮任务：' + json.dumps(context['goal'], ensure_ascii=False)
        + '\n当前停车图：' + context['observation_id'] + '。\n' + required
        + '\n先判断当前问题是否已有答案，再选明确目标、有价值入口或具体待查部分；不重开已解决的问题。'
        + '\n以下是可选动作，不是执行顺序；已有有价值候选时，不为补齐未看方向额外转头。'
        + '\n当前可直接 look 的未看方向：' + json.dumps(unseen)
        + '；可转回方向及用途：' + json.dumps(orient_options, ensure_ascii=False)
        + '；经旧视角继续查看后方：'
        + json.dumps(options.get('unseen_via_orient', []), ensure_ascii=False)
        + '。无出口时沿未看方向继续，不反复转回中心；发现入口即可走。'
        + '\n进入房间选入口或门内可见空地，不以房间深处家具代替入口。'
        + 'instruction 按本轮 VLN 指令指南描述这个终点。'
        + '\n仅输出 JSON；复核与 evidence 均放最外层，不复制输入状态。'
    )


def target_watch_prompt(state):
    """仅提供检测能力与候选复核约定；监测开关由模型所选技能组织。"""
    review = (
        '当前没有检测停车候选，省略 detection_review。'
        if not state.get('pending') else """
检测命中已停车。对照触发原图、检测框和当前图 A，填写顶层
detection_review={candidate_id:target_watch.pending.candidate_id,verdict:confirmed|rejected|uncertain,evidence:中文依据}。
confirmed 须满足原任务全部要求且当前图支持身份，scene.target_seen=yes；
只有历史命中、遮挡或看不清时填 uncertain。confirmed/uncertain 后选 approach_target 核实或接近；
确实排除候选后可选 search_navigation 继续搜索。单帧未见不等于否定目标。
"""
    )
    return (
        '\n目标检测服务已配置。search_navigation 的 navigate 须同轮填写 target_labels；'
        '技能自动启停监测，不存在单独的 watch_target 决策动作。'
        '类别只用简单英文物体名，颜色和关系由你复核。'
        'target_watch 是工具返回状态，不是要复制或修改的参数。'
        '服务不支持、超时或未检出时继续观察导航，不能据此排除目标。\n'
        + review
    )
