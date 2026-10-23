"""只读检查 Go2 协议、观测和检测服务，不申请控制权。"""

import json

import aiohttp

from navigation_agent.app.deployments.check import check_detector
from navigation_agent.app.deployments.factory import create_backend
from navigation_agent.app.go2.connection import missing_capabilities
from navigation_agent.contracts.agent_backend import ToolError


async def check(settings):
    if settings.backend.robot.backend != 'go2':
        raise ValueError('只读本体检查需要 GO2 配置')
    report = {
        'robot_url': settings.backend.robot.base_url,
        'model_url_from_robot': settings.backend.vln.server_url,
        'llm_configured': bool(settings.api_key and settings.agent.llm.model.strip()),
        'robot_service': False,
        'observation': False,
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3), trust_env=False) as http:
        try:
            async with http.get(settings.backend.robot.base_url.rstrip('/') + '/api/health') as response:
                response.raise_for_status()
                state = await response.json()
            protocol = state.get('agent', {}).get('protocol_version')
            report.update(
                robot_service=True,
                protocol=protocol,
                robot=state.get('robot'),
                control_source=state.get('control_source'),
                configured_model_url=state.get('vln', {}).get('server_url'),
            )
            missing = missing_capabilities(state)
            if (settings.agent.harness.preserve_same_instruction_history
                    and state.get('agent', {}).get('capabilities', {}).get('navigation_history_continuation') is not True):
                missing.append('navigation_history_continuation')
            report['missing_capabilities'] = missing
            if protocol != 1:
                report['error'] = '客户端缺少 Agent 协议版本 1；请启动本仓库的 Go2 镜像。'
            elif missing:
                report['error'] = '本体客户端需要更新，缺少能力：' + ', '.join(missing)
            else:
                # 使用真实适配对象的只读采图；不进入会申请控制的连接上下文。
                backend = create_backend(settings.backend)
                report['streaming_observation'] = backend.supports_streaming_observation
                if not backend.supports_streaming_observation:
                    raise ToolError('Go2 适配层未实现途中采图')
                backend.robot.channel.session = http
                backend.robot.channel.odom_frame = state.get('odom', {}).get('frame_id')
                _, observation = await backend.snapshot_observation()
                report['observation'] = observation['pose_valid']
                report['camera_stamp_ns'] = observation['camera_stamp_ns']
        except (aiohttp.ClientError, TimeoutError, ValueError, ToolError) as exc:
            report['error'] = (
                f'本体连接或适配检查失败（{type(exc).__name__}: {exc}）；'
                '请检查 robot_url、狗端客户端和 SSH 隧道。'
            )

        # 检测由 Agent 主机调用；模型地址则由狗端使用，不能在此代测。
        detector_url = settings.agent.target_watch.server_url
        if detector_url:
            report['target_detector_url'] = detector_url
            report['target_detector'] = await check_detector(http, detector_url)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    ready = (
        report['robot_service']
        and report.get('protocol') == 1
        and not report.get('missing_capabilities')
        and report.get('streaming_observation', False)
        and report['observation']
        and report.get('target_detector', True)
    )
    # 不发模型输入或机器人命令；模型推理和实际运动仍需分别验证。
    return 0 if ready else 1
