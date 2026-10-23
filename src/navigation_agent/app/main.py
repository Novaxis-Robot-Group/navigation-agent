"""Agent 服务入口；先校验配置，再启动业务服务或开发重载器。"""

import argparse
import os
import uvicorn

from navigation_agent.app.deployments.settings import DEFAULT_AGENT_CONFIG, DEFAULT_BACKEND_CONFIG, PROJECT_ROOT, load_settings, from_launch_environment


def create_app(settings=None):
    from navigation_agent.app.agent import api
    from navigation_agent.app.deployments.runtime import Deployment

    api.configure(settings or from_launch_environment())
    api.install_deployment(Deployment(api))
    return api.app


def main(*, app_factory='navigation_agent.app.main:create_app'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-config', default=DEFAULT_AGENT_CONFIG)
    parser.add_argument('--backend-config', default=DEFAULT_BACKEND_CONFIG)
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--check-config', action='store_true')
    parser.add_argument('--check', action='store_true', help='只读检查连接；Go2 另检查协议与观测')
    args = parser.parse_args()
    try:
        settings = load_settings(args.agent_config, args.backend_config)
    except (ValueError, OSError, KeyError) as exc:
        # Pydantic 默认错误包含输入值；仅显示字段与错误类别。
        if hasattr(exc, 'errors'):
            parser.error(str([(item['loc'], item['type']) for item in exc.errors(include_input=False)]))
        parser.error(str(exc))
    from navigation_agent.app.go2.motion_settings import load_saved
    if settings.backend.robot.backend == 'go2':
        try:
            from navigation_agent.app.deployments.runtime_overrides import apply_agent_settings, motion_defaults
            saved = load_saved(
                settings.backend.robot.motion_settings_file, settings.backend.robot.base_url,
                prompt_defaults=motion_defaults(settings).prompt,
            )
            apply_agent_settings(settings.agent, saved)
        except (OSError, ValueError) as exc:
            # 指明实际原因；参数校验仅输出字段和错误类别，不回显保存值。
            if hasattr(exc, 'errors'):
                details = [(item['loc'], item['type']) for item in exc.errors(include_input=False)]
                parser.error('保存的运行参数无效：' + str(details))
            parser.error('无法加载保存的运行参数：' + str(exc))
    if args.check_config:
        import json
        print(json.dumps(settings.snapshot(), ensure_ascii=False, indent=2))
        return
    if args.check:
        import asyncio
        if settings.backend.robot.backend == 'go2':
            from navigation_agent.app.go2.check import check
        else:
            from navigation_agent.app.deployments.check import check
        raise SystemExit(asyncio.run(check(settings)))
    # 子进程只继承配置文件选择；不使用环境变量传递业务参数或密钥副本。
    os.environ['NAV_AGENT_CONFIG_PATH'] = str(settings.agent_file)
    os.environ['NAV_BACKEND_CONFIG_PATH'] = str(settings.backend_file)
    service = settings.backend.service
    if args.reload:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from deploy.agent.devserver import run_reload
        run_reload(app_factory, PROJECT_ROOT / 'src', service.host, service.port)
    else:
        uvicorn.run(create_app(settings), host=service.host, port=service.port, access_log=False)


if __name__ == '__main__':
    main()
