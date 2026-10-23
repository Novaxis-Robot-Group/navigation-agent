"""生成模型与日志共用的图片清单。"""

from navigation_agent.agent.memory.spatial import angle_distance

def image_manifest(directory, view, session_views, last_navigation, *,
                   memory_frames=(), target_frames=(), recent_views=(), region_frames=(),
                   navigation_frames=(), route_frames=()):
    """同一份图片清单用于请求和日志，重复原图保留全部用途。"""
    images = {}

    def add(path, purpose, *, view_id=None, heading_deg=None, observer_position=None):
        if not path or not (directory / path).is_file():
            return
        item = images.setdefault(path, {'image': path, 'uses': []})
        item['uses'].append({
            'purpose': purpose, 'view_id': view_id, 'heading_deg': heading_deg,
            'observer_position': observer_position,
        })

    add(view.get('image'), '当前图像 A', view_id=f'O{view["step"]}', heading_deg=view.get('heading_deg'),
        observer_position=view.get('observer_position'))
    for frame in target_frames:
        add(frame.get('image'), '目标历史证据，仅用于身份对照，不证明当前可见',
            view_id=f'O{frame["step"]}', heading_deg=frame.get('heading_deg'), observer_position=frame.get('observer_position'))
    for frame in recent_views:
        if frame['step'] != view['step']:
            add(frame.get('image'), '近期连续停车画面，按观察编号比较同一入口和目标遮挡变化',
                view_id=f'O{frame["step"]}', heading_deg=frame.get('heading_deg'),
                observer_position=frame.get('observer_position'))
    if last_navigation:
        add(last_navigation.get('image'), '真实上次导航起始图',
            view_id=f'O{last_navigation["step"]}', heading_deg=last_navigation.get('heading_deg'),
            observer_position=last_navigation.get('observer_position'))
    for frame in navigation_frames[:2]:
        add(frame.get('image'), '上次导航途中原图，停车后按观察编号回看是否穿门；不是当前图',
            view_id=f'O{frame["step"]}', heading_deg=frame.get('heading_deg'),
            observer_position=frame.get('observer_position'))
    # 原地观察时留一张附近方向原图，避免只能凭旧文字猜测转回后的视野。
    historical = [frame for frame in session_views
                  if frame['step'] != view['step']]
    heading = view.get('heading_deg')
    historical.sort(key=lambda frame: (
        angle_distance(frame['heading_deg'], heading)
        if heading is not None and frame.get('heading_deg') is not None else 0,
        frame['step'],
    ), reverse=True)
    first_region_image = region_frames[0].get('image') if region_frames else None
    needs_direction_slot = any(frame.get('image') not in images
                               and frame.get('image') != first_region_image for frame in historical)
    selected_regions = region_frames[:1 if needs_direction_slot else 2]
    for frame in selected_regions:
        add(frame.get('image'), f"区域 {frame['region_id']} 的历史参照，比较家具与入口关系，不证明当前位置",
            view_id=f'O{frame["step"]}', heading_deg=frame.get('heading_deg'),
            observer_position=frame.get('observer_position'))
    remaining = max(0, 2 - len(selected_regions))
    attached = [frame for frame in historical if frame.get('image') in images]
    additional = [frame for frame in historical if frame.get('image') not in images][:remaining]
    for frame in [*attached, *additional]:
        add(frame.get('image'), '当前位置附近的历史方向，仅用于比较和转回确认',
            view_id=frame['id'], heading_deg=frame['heading_deg'],
            observer_position=frame.get('observer_position'))
    for frame in route_frames[:3]:
        destination = frame.get('destination_region')
        purpose = (
            f"已走通入口 {frame['route_id']} 的原图：{frame['region_id']} 通向 {destination}，只作入口身份对照"
            if destination else
            f"尚未尝试入口 {frame['route_id']} 的原图，位于 {frame['region_id']}；需当前图重新确认后接近"
        )
        if destination and frame.get('at_destination'):
            purpose = (f"通过入口 {frame['route_id']} 后在 {destination} 的停车图；"
                       f"来路连接 {frame['region_id']}，帮助识别返回方向，不证明图中所有门都是来路")
        add(frame.get('image'), purpose, view_id=f'O{frame["step"]}',
            heading_deg=frame.get('heading_deg'), observer_position=frame.get('observer_position'))
    for frame in memory_frames:
        add(frame.get('image'), '检索到的历史观察证据，不证明当前可见',
            view_id=f'O{frame["step"]}', heading_deg=frame.get('heading_deg'), observer_position=frame.get('observer_position'))
    # 历史先给，当前帧最后给，避免末尾旧图覆盖本轮视觉判断；同图仍只发送一次。
    current = images.pop(view.get('image'), None)
    return [*images.values(), *([current] if current else [])]
