"""共享任务录像编码器；原始采图时间另存，不修改导航输入。"""

import asyncio
import json
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def encoder_command(path, fps):
    # 输入每帧对应固定的视频时长；真实采图时间另存，回放按映射定位。
    return [
        'gst-launch-1.0', '-q', 'fdsrc', 'fd=0', '!', f'image/jpeg,framerate={fps}/1',
        '!', 'jpegparse', '!', 'jpegdec', '!', 'videoconvert', '!', 'video/x-raw,format=I420',
        '!', 'x264enc', 'speed-preset=ultrafast', 'tune=zerolatency', 'bitrate=2000',
        '!', 'h264parse', '!', 'mp4mux', 'faststart=true', '!', 'filesink', 'location=' + str(path),
    ]


class TaskRecording:
    def __init__(self, directory, task, fps, *, views=('first', 'third')):
        self.directory = directory
        self.fps = fps
        self.views = tuple(views)
        self.frames = queue.Queue(maxsize=8)
        self.done = threading.Event()
        self.processes = []
        self.error = None
        self.dropped = 0
        self.count = 0
        self.metadata = {
            'task_id': task['id'], 'started_at': task['started_at'], 'fps': fps,
            'status': 'recording', 'views': list(self.views), 'frames': 0,
        }
        self.directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / 'recording.json', self.metadata)
        self.thread = threading.Thread(target=self._encode, name='task-video', daemon=True)
        self.thread.start()

    def offer(self, *frames):
        if len(frames) != len(self.views):
            raise ValueError('录像帧数与视角数不匹配')
        if self.done.is_set() or self.error:
            return
        try:
            self.frames.put_nowait(frames)
        except queue.Full:
            self.dropped += 1

    def _encode(self):
        try:
            if shutil.which('gst-launch-1.0') is None:
                raise RuntimeError('未安装 GStreamer：请按 docs/zh/replay.md 安装录像依赖')
            for view in self.views:
                process = subprocess.Popen(encoder_command(self.directory / (view + '.mp4'), self.fps),
                                           stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
                self.processes.append(process)
            with (self.directory / 'frames.jsonl').open('w', encoding='utf-8') as stream:
                while not self.done.is_set() or not self.frames.empty():
                    try:
                        frames = self.frames.get(timeout=.1)
                        first = frames[0]
                    except queue.Empty:
                        continue
                    for process, frame in zip(self.processes, frames):
                        process.stdin.write(frame.jpeg)
                        process.stdin.flush()
                    stream.write(json.dumps({'stamp_ns': first.stamp_ns, 'video_s': self.count / self.fps,
                                             'pose': list(first.pose)}) + '\n')
                    stream.flush()
                    self.count += 1
            for process in self.processes:
                process.stdin.close()
            for process in self.processes:
                if process.wait(timeout=15) != 0:
                    raise RuntimeError('H.264 编码失败，请检查 GStreamer 的 x264enc、jpegparse 和 mp4mux 插件')
        except Exception as exc:
            self.error = str(exc)
        finally:
            for process in self.processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if process.stdin and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
            if self.count == 0 and not self.error:
                self.error = '任务期间没有取得有效录像画面'
            self.metadata.update(status='failed' if self.error else 'ready', frames=self.count,
                                 dropped_frames=self.dropped, error=self.error,
                                 video_duration_s=self.count / self.fps)
            if self.error:
                for view in self.views:
                    (self.directory / (view + '.mp4')).unlink(missing_ok=True)
            try:
                write_json(self.directory / 'recording.json', self.metadata)
            except OSError:
                self.metadata.update(status='failed', error='无法保存录像状态，请检查磁盘空间与目录权限')

    async def close(self, status='ready'):
        self.done.set()
        await asyncio.to_thread(self.thread.join, 20)
        if self.thread.is_alive():
            self.error = '录像编码超时，已停止编码进程'
            for process in self.processes:
                if process.poll() is None:
                    process.kill()
            await asyncio.to_thread(self.thread.join, 3)
        if status != 'ready' and self.metadata['status'] == 'ready':
            self.metadata['status'] = status
            write_json(self.directory / 'recording.json', self.metadata)
        return self.metadata
