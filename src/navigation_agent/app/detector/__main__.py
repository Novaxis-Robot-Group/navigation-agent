"""python -m navigation_agent.app.detector --weights /path/to/yolov8l-worldv2.pt"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18060)
    parser.add_argument('--inference-timeout', type=float, default=30., help='单次推理挂起超过此秒数时退出，交给容器重启')
    args = parser.parse_args()
    if args.inference_timeout <= 0:
        parser.error('推理时限必须大于零')
    if not Path(args.weights).is_file():
        parser.error('模型文件不存在；请先下载权重，再启动服务')
    import uvicorn
    from navigation_agent.app.detector.engine import YoloEngine
    from navigation_agent.app.detector.server import create_app
    app = create_app(lambda: YoloEngine(args.weights, args.device), inference_timeout_s=args.inference_timeout)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == '__main__':
    main()
