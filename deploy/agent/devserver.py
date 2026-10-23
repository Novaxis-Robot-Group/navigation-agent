"""协作式热重载：等旧任务清理完成，再替换工作进程。"""

import asyncio
from contextlib import suppress
from functools import partial
import logging
import multiprocessing

import uvicorn
from uvicorn._subprocess import get_subprocess
from uvicorn.supervisors.watchfilesreload import WatchFilesReload


def serve_worker(sockets, *, config, stop):
    async def serve():
        server = uvicorn.Server(config)

        async def shutdown_request():
            while not stop.is_set():
                await asyncio.sleep(0.1)
            server.should_exit = True

        monitor = asyncio.create_task(shutdown_request())
        try:
            await server.serve(sockets=sockets)
        finally:
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor

    # 使用 Uvicorn 选择的事件循环，保持子进程 socket 语义。
    with asyncio.Runner(loop_factory=config.get_loop_factory()) as runner:
        runner.run(serve())


class CooperativeReload(WatchFilesReload):
    def __init__(self, config, sockets):
        self.stop_worker = multiprocessing.get_context('spawn').Event()
        super().__init__(
            config, partial(serve_worker, config=config, stop=self.stop_worker), sockets
        )

    def stop_process(self):
        # 先请求旧进程退出，超时后才强制结束
        self.stop_worker.set()
        self.process.join(timeout=10)
        if self.process.is_alive():
            logging.getLogger('uvicorn.error').error(
                'Worker shutdown timed out; terminating dev worker'
            )
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            raise RuntimeError('Old worker did not exit; refusing to start another')

    def restart(self):
        # 旧进程退出后再启动新进程，避免同时占用控制权
        self.stop_process()
        self.stop_worker.clear()
        self.process = get_subprocess(config=self.config, target=self.target, sockets=self.sockets)
        self.process.start()

    def shutdown(self):
        self.stop_process()
        for sock in self.sockets:
            sock.close()
        logging.getLogger('uvicorn.error').info('Development reloader stopped')


def run_reload(app_import, directory, host, port):
    config = uvicorn.Config(
        app_import,
        factory=True,
        host=host,
        port=port,
        access_log=False,
        reload=True,
        reload_dirs=[
            str(directory / name) for name in ('navigation_agent',) if (directory / name).is_dir()
        ],
        reload_excludes=[str(p) for p in (directory / '.venv', directory / 'tests') if p.is_dir()],
        timeout_graceful_shutdown=5,
    )

    # 先绑定端口，重复启动时不再创建监视器和工作进程。
    sock = config.bind_socket()
    try:
        CooperativeReload(config, [sock]).run()
    finally:
        sock.close()
