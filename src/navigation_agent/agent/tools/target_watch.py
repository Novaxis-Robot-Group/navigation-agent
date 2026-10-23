"""目标监测会话：只读采样、候选复核和短期抑制，不直接执行运动。"""

import asyncio
import base64
import math
import time
from uuid import uuid4

import aiohttp

from navigation_agent.contracts.agent_backend import ToolError
from navigation_agent.contracts.detection import (
    Capabilities, FrameResponse, SessionResponse, WatchArguments,
)


class DetectorClient:
    def __init__(self, settings):
        self.settings = settings
        self.http = None

    async def __aenter__(self):
        self.http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.settings.request_timeout_s),
        )
        return self

    async def __aexit__(self, *args):
        await self.http.close()

    async def request(self, method, path, payload=None):
        async with self.http.request(
            method, self.settings.server_url.rstrip('/') + path, json=payload,
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def capabilities(self):
        return Capabilities.model_validate(await self.request('GET', '/capabilities'))

    async def open(self, labels):
        result = await self.request('POST', '/sessions', {
            'labels': labels, 'confidence': self.settings.confidence,
        })
        return SessionResponse.model_validate(result).session_id

    async def close(self, session):
        await self.request('DELETE', '/sessions/' + session)

    async def detect(self, session, jpeg, stamp):
        identity = uuid4().hex
        result = FrameResponse.model_validate(await self.request(
            'POST', '/sessions/' + session + '/frames',
            {'request_id': identity, 'camera_stamp_ns': stamp,
             'jpeg_base64': base64.b64encode(jpeg).decode('ascii')},
        ))
        if result.request_id != identity or result.camera_stamp_ns != stamp:
            raise ValueError('检测响应不属于当前图片')
        return result


class TargetWatch:
    def __init__(self, settings, client, tools, *, event, capture):
        self.settings, self.client, self.tools = settings, client, tools
        self.event, self.capture = event, capture
        self.enabled = False
        self.labels = []
        self.status = 'disabled'
        self.session = None
        self.pending = None
        self.last_review = None
        self.last_tool_result = None
        self.ignored = {}
        self.task = None
        self.navigation = None
        self.generation = 0
        self.hit_time = None
        self.consistent_hit = None

    def context(self):
        return {
            'available': bool(self.settings.server_url), 'enabled': self.enabled,
            'labels': self.labels[:], 'status': self.status,
            'pending': dict(self.pending) if self.pending else None,
            'last_review': self.last_review, 'last_tool_result': self.last_tool_result,
        }

    async def configure(self, enabled, labels):
        arguments = WatchArguments(enabled=enabled, labels=labels)
        if enabled and not self.tools.supports_streaming_observation:
            await self.pause()
            self.enabled, self.labels, self.status = False, arguments.labels, 'unsupported'
            self.event('target_watch_degraded', reason='backend_unsupported',
                       stage='capability', retrying=False)
            return self.context()
        if self.enabled == enabled and self.labels == arguments.labels:
            return self.context()
        await self.pause()
        self.enabled, self.labels = enabled, arguments.labels
        self.pending = None
        self.status = 'armed' if enabled else 'disabled'
        self.last_tool_result = {
            'tool': 'watch_target', 'configured': True, 'enabled': enabled,
            'labels': self.labels[:],
            'message': '搜索导航技能已启用途中监测'
                       if enabled else '已关闭目标监测',
        }
        # 创建远程会话放在后台导航中，开关工具不等待模型加载。
        self.event('target_watch_configured', **self.context())
        return self.context()

    def validate(self, decision):
        review = decision.detection_review
        if review and (not self.pending or review.candidate_id != self.pending['candidate_id']):
            raise ValueError('detection_review 必须引用当前 target_watch.pending.candidate_id')
        if self.pending and decision.action != 'recover' and review is None:
            raise ValueError('先填写 detection_review：candidate_id、confirmed/rejected/uncertain 和当前视觉依据')
        if review and review.verdict == 'confirmed' and decision.scene.target_seen != 'yes':
            raise ValueError('确认检测目标需要当前图 scene.target_seen=yes；仅历史命中应填 uncertain')

    async def review(self, review):
        if review is None:
            return
        self.last_review = review.model_dump()
        self.event('target_detection_review', **self.last_review)
        if review.verdict == 'rejected':
            self.ignored[self.pending['continuity_id']] = {
                **self.pending, 'last_seen': time.monotonic(),
            }
        self.pending = None
        # 待核实也是跟进具体线索；之后由模型所选技能决定是否恢复搜索。
        if review.verdict in ('confirmed', 'uncertain'):
            await self.configure(False, [])

    @staticmethod
    def _observer_pose(observation):
        pose = observation.get('pose') or {}
        values = (pose.get('x'), pose.get('y'), observation.get('yaw', pose.get('yaw')))
        if not observation.get('pose_valid', True) or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in values
        ):
            return None
        return values

    @staticmethod
    def _box_overlap(first, second):
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0., x2 - x1) * max(0., y2 - y1)
        area = ((first[2] - first[0]) * (first[3] - first[1])
                + (second[2] - second[0]) * (second[3] - second[1]))
        return intersection / (area - intersection)

    def _suppressed(self, detections, observation):
        """短暂漏检保留否决；仅在相近视角、框唯一对应时衔接换号。"""
        now = time.monotonic()
        pose = self._observer_pose(observation)
        suppressed = set()
        live = {item.continuity_id: item for item in detections}
        for identity, rejected in list(self.ignored.items()):
            if now - rejected['last_seen'] > 2.0:
                del self.ignored[identity]
                continue
            match = live.get(identity)
            if match is None:
                previous = rejected.get('observer_pose')
                nearby = (pose is not None and previous is not None
                          and math.dist(pose[:2], previous[:2]) <= .3
                          and abs((pose[2] - previous[2] + math.pi) % (2 * math.pi) - math.pi)
                          <= math.radians(15))
                matches = [
                    item for item in detections
                    if nearby and item.label == rejected['label']
                    and self._box_overlap(item.bbox, rejected['bbox']) >= .6
                ]
                # 多个重叠框有歧义时允许重新停车，不把其他同类目标一起忽略。
                match = matches[0] if len(matches) == 1 else None
            if match is None:
                continue
            suppressed.add(match.continuity_id)
            del self.ignored[identity]
            self.ignored[match.continuity_id] = {
                **rejected, **match.model_dump(), 'observer_pose': pose, 'last_seen': now,
            }
            if match.continuity_id != identity:
                self.event('target_detection_suppressed', continuity_id=match.continuity_id,
                           previous_continuity_id=identity, reason='same_rejected_view')
        return suppressed

    async def pause(self):
        """转头、接近或结束时使旧结果失效；不跨视角复用误检抑制。"""
        self.generation += 1
        self.navigation = None
        self.consistent_hit = None
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        if self.session is not None:
            try:
                await self.client.close(self.session)
            except Exception:
                pass
            self.session = None
        self.ignored.clear()

    async def execute(self, execute_action, action_step, *, search_navigation):
        """按所选技能随导航启停检测；不调用规划模型或判断导航路线。"""
        if not search_navigation:
            await self.pause()
            return await execute_action()

        action = asyncio.create_task(execute_action())
        try:
            await self.start(action, action_step)
            return await action
        except BaseException:
            # 取消或异常须结束检测会话，不能向下一段导航发送迟到结果。
            await self.pause()
            raise
        finally:
            if not action.done():
                action.cancel()
                await asyncio.gather(action, return_exceptions=True)
            self.navigation_ended()

    async def start(self, action, action_step):
        if not self.enabled:
            return
        self.navigation = (action, action_step)
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._sample(self.generation))

    def navigation_ended(self):
        # 中断请求被接受不等于停车；动作返回后单独记录闭环耗时。
        if self.hit_time is not None:
            self.event('target_detection_navigation_returned',
                       seconds_since_capture=time.time() - self.hit_time)
            self.hit_time = None
        self.consistent_hit = None
        self.navigation = None

    async def _retry_connection(self, error, failures, *, stage='detector'):
        """重连只属于检测工具；关闭技能会取消等待，不影响当前导航。"""
        if self.status != 'degraded':
            self.event('target_watch_degraded', reason=type(error).__name__, stage=stage, retrying=True)
        self.status = 'degraded'
        expired = isinstance(error, aiohttp.ClientResponseError) and error.status == 404
        if stage == 'detector' and self.session is not None and (expired or failures >= 3):
            session, self.session = self.session, None
            try:
                await self.client.close(session)
            except Exception:
                pass
        await asyncio.sleep(min(5., self.settings.frame_interval_s * 2 ** min(failures, 6)))

    async def _sample(self, generation):
        previous_stamp = None
        failures = 0
        try:
            while self.enabled and self.generation == generation:
                if self.session is None:
                    try:
                        capabilities = await self.client.capabilities()
                        if not capabilities.ready:
                            raise ConnectionError('检测服务尚未就绪')
                        if capabilities.vocabulary == 'fixed' and any(
                            label not in capabilities.labels for label in self.labels
                        ):
                            self.status = 'unsupported'
                            self.event('target_watch_degraded', reason=self.status,
                                       stage='capability', retrying=False)
                            return
                        self.session = await self.client.open(self.labels)
                    except Exception as exc:
                        failures += 1
                        await self._retry_connection(exc, failures)
                        continue
                started = time.monotonic()
                navigation = self.navigation
                # 停车复核期间仅维持身份，不发送停车；转头前 pause 会关闭此采样。
                if navigation is not None and self.tools.navigation_id is None:
                    await asyncio.sleep(.05)
                    continue
                identity = self.tools.navigation_id if navigation else None
                try:
                    jpeg, observation = await asyncio.wait_for(
                        self.tools.snapshot_observation(), self.settings.request_timeout_s,
                    )
                except (TimeoutError, OSError, ToolError, aiohttp.ClientError) as exc:
                    failures += 1
                    await self._retry_connection(exc, failures, stage='observation')
                    continue
                # snapshot 的完成与取消可能同时发生，切换后不能再发检测请求。
                if self.generation != generation or not self.enabled:
                    return
                stamp = observation.get('camera_stamp_ns')
                if stamp and stamp != previous_stamp:
                    previous_stamp = stamp
                    age = (time.time_ns() - stamp) / 1e9
                    if 0 <= age <= self.settings.max_frame_age_s:
                        try:
                            result = await self.client.detect(self.session, jpeg, stamp)
                            failures = 0
                            self.status = 'watching'
                        except Exception as exc:
                            failures += 1
                            await self._retry_connection(exc, failures)
                            continue
                        age = (time.time_ns() - stamp) / 1e9
                        self.event('target_detection_frame', camera_stamp_ns=stamp,
                                   age_s=age, inference_ms=result.inference_ms,
                                   detections=len(result.detections), navigation_id=identity)
                        suppressed = (self._suppressed(result.detections, observation)
                                      if 0 <= age <= self.settings.max_frame_age_s else set())
                        fresh = (0 <= age <= self.settings.max_frame_age_s
                                 and self.generation == generation
                                 and self.navigation is navigation and navigation is not None
                                 and not navigation[0].done()
                                 and identity is not None and self.tools.navigation_id == identity)
                        candidates = [item for item in result.detections
                                      if item.label in self.labels
                                      and item.confidence >= self.settings.confidence
                                      and item.continuity_id not in suppressed]
                        if not fresh or not candidates:
                            self.consistent_hit = None
                        if fresh and candidates and self.pending is None:
                            item = max(candidates, key=lambda item: item.confidence)
                            previous = self.consistent_hit
                            same = (previous is not None and previous['identity'] == identity
                                    and previous['label'] == item.label
                                    and self._box_overlap(previous['bbox'], item.bbox) >= .3
                                    and time.monotonic() - previous['time'] <= 2.)
                            hits = previous['hits'] + 1 if same else 1
                            self.consistent_hit = {'identity': identity, 'label': item.label,
                                                   'bbox': item.bbox, 'hits': hits,
                                                   'time': time.monotonic()}
                            if hits < self.settings.min_consistent_frames:
                                self.event('target_detection_pending', label=item.label, hits=hits)
                                continue
                            accepted = self.tools.request_navigation_interrupt(identity, 'target_detected')
                            self.event('target_detection_interrupt', accepted=accepted,
                                       navigation_id=identity, age_s=age, **item.model_dump())
                            if accepted:
                                boxed = (base64.b64decode(result.boxed_jpeg_base64, validate=True)
                                         if result.boxed_jpeg_base64 else None)
                                view = self.capture(jpeg, observation, navigation[1], boxed)
                                self.pending = {
                                    **item.model_dump(), 'candidate_id': uuid4().hex,
                                    'image': view['image'], 'step': view['step'],
                                    'boxed_image': view.get('boxed_image'),
                                    'camera_stamp_ns': stamp,
                                    'observer_pose': self._observer_pose(observation),
                                }
                                self.hit_time = stamp / 1e9
                                self.event('target_detection_hit', **self.pending)
                await asyncio.sleep(max(0., self.settings.frame_interval_s - (time.monotonic() - started)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.status = 'degraded'
            self.event('target_watch_degraded', reason=type(exc).__name__,
                       stage='sampler', retrying=False)
