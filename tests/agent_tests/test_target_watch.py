"""检测中断须绑定实时导航；工具配置和误检不改变运动记忆。"""

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch

from navigation_agent.agent.tools.target_watch import DetectorClient, TargetWatch
from tests.config_support import legacy_search_run_task as run_task
from navigation_agent.agent.schemas import Decision, TaskRequest
from navigation_agent.app.agent.demo import DemoBackend, DemoModel
from navigation_agent.agent.tools import NavigationTools
from navigation_agent.contracts.detection import Capabilities, Detection, DetectionReview, FrameResponse
from tests.config_support import agent_settings


class FakeDetector:
    def __init__(self):
        self.calls = 0
        self.delay = 0
        self.fail = False
        self.closed = []
        self.track = 'track-1'
        self.bbox = (.1, .1, .8, .8)
        self.detected = True
        self.vocabulary = 'open'

    async def capabilities(self):
        return Capabilities(ready=True, model='fake', vocabulary=self.vocabulary, labels=['chair'])

    async def open(self, labels):
        return 'session'

    async def close(self, session):
        self.closed.append(session)

    async def detect(self, session, jpeg, stamp):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError('detector unavailable')
        return FrameResponse(
            request_id='test', camera_stamp_ns=stamp, width=480, height=270, inference_ms=1,
            detections=[Detection(label='sofa', confidence=.9, bbox=self.bbox,
                                  continuity_id=self.track)] if self.detected else [],
        )


class FakeTools:
    navigation_id = None
    supports_streaming_observation = True

    def __init__(self):
        self.interrupts = []
        self.stale = False
        self.pose = {'x': 0., 'y': 0., 'yaw': 0.}
        self.stopped = asyncio.Event()

    async def snapshot_observation(self):
        return b'jpeg', {'camera_stamp_ns': time.time_ns() - (2_000_000_000 if self.stale else 0),
                         'pose': self.pose}

    def request_navigation_interrupt(self, identity, reason):
        if identity != self.navigation_id:
            return False
        self.interrupts.append((identity, reason))
        self.stopped.set()
        return True

    async def navigate(self, seconds=.06):
        self.navigation_id = 'nav-1'
        try:
            try:
                await asyncio.wait_for(self.stopped.wait(), seconds)
                return {'reason': 'agent_interrupt'}
            except TimeoutError:
                return {'reason': 'model_stop'}
        finally:
            self.navigation_id = None


class WatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.events, self.frames = [], []
        self.tools, self.client = FakeTools(), FakeDetector()
        self.settings = agent_settings()
        self.settings = self.settings.model_copy(update={
            'target_watch': self.settings.target_watch.model_copy(update={
                'server_url': 'http://localhost:8060', 'frame_interval_s': .001,
            }),
        })

        def capture(jpeg, observation, step, boxed):
            self.frames.append(jpeg)
            return {'image': 'images/001.jpg', 'step': 1}

        self.watch = TargetWatch(
            self.settings.target_watch, self.client, self.tools, capture=capture,
            event=lambda kind, **data: self.events.append((kind, data)),
        )
        await self.watch.configure(True, ['sofa'])

    async def asyncTearDown(self):
        await self.watch.pause()

    async def navigate(self, search=True, seconds=.06):
        self.tools.stopped.clear()
        return await self.watch.execute(lambda: self.tools.navigate(seconds), 3,
                                          search_navigation=search)

    async def test_two_consistent_frames_are_required(self):
        self.watch.settings = self.watch.settings.model_copy(update={'min_consistent_frames': 2})
        await self.navigate()
        self.assertEqual(self.client.calls, 2)
        self.assertEqual(len(self.tools.interrupts), 1)
        self.assertTrue(any(kind == 'target_detection_pending' for kind, _ in self.events))

    async def test_disjoint_boxes_never_accumulate_two_hits(self):
        self.watch.settings = self.watch.settings.model_copy(update={'min_consistent_frames': 2})
        detect = self.client.detect
        async def alternating(*args):
            self.client.bbox = ((.1, .1, .2, .2) if self.client.calls % 2 else (.7, .7, .9, .9))
            return await detect(*args)
        self.client.detect = alternating
        await self.navigate(seconds=.02)
        self.assertGreater(self.client.calls, 2)
        self.assertFalse(self.tools.interrupts)

    async def test_hit_stops_without_model_and_preserves_evidence(self):
        result = await self.navigate()
        self.assertEqual(result['reason'], 'agent_interrupt')
        self.assertEqual(self.tools.interrupts, [('nav-1', 'target_detected')])
        self.assertEqual(len(self.frames), 1)
        self.assertIsNotNone(self.watch.pending)

    async def test_approach_does_not_start_detector(self):
        self.assertEqual((await self.navigate(False))['reason'], 'model_stop')
        self.assertEqual(self.client.calls, 0)

    async def test_old_capture_is_not_sent(self):
        self.tools.stale = True
        await self.navigate()
        self.assertEqual(self.client.calls, 0)
        self.assertFalse(self.tools.interrupts)

    async def test_slow_response_cannot_stop(self):
        self.watch.settings = self.watch.settings.model_copy(update={'max_frame_age_s': .005})
        self.client.delay = .02
        await self.navigate()
        self.assertFalse(self.tools.interrupts)

    async def test_navigation_identity_change_invalidates_result(self):
        self.client.delay = .02
        async def replace_identity():
            await asyncio.sleep(.01)
            self.tools.navigation_id = 'new-nav'
        changing = asyncio.create_task(replace_identity())
        await self.navigate(seconds=.03)
        await changing
        self.assertFalse(self.tools.interrupts)

    async def test_rejection_survives_tracker_id_change_in_same_view(self):
        await self.navigate()
        identity = self.watch.pending['candidate_id']
        await self.watch.review(DetectionReview(candidate_id=identity, verdict='rejected', evidence='这是椅子'))
        await self.navigate(seconds=.02)
        self.assertEqual(len(self.tools.interrupts), 1)
        self.client.track = 'new-segment'
        await self.navigate()
        self.assertEqual(len(self.tools.interrupts), 1)
        self.assertTrue(any(kind == 'target_detection_suppressed' for kind, _ in self.events))

        self.client.track = 'other-object'
        self.client.bbox = (.8, .8, .99, .99)
        await self.navigate()
        self.assertEqual(len(self.tools.interrupts), 2)

    async def test_short_miss_keeps_rejection_but_long_loss_expires_it(self):
        await self.navigate()
        await self.watch.review(DetectionReview(candidate_id=self.watch.pending['candidate_id'],
                                               verdict='rejected', evidence='类别错误'))
        self.client.detected = False
        await asyncio.sleep(.01)
        self.client.detected = True
        await self.navigate(seconds=.02)
        self.assertEqual(len(self.tools.interrupts), 1)

        self.client.detected = False
        for rejected in self.watch.ignored.values():
            rejected['last_seen'] -= 3.
        await asyncio.sleep(.01)
        self.client.detected = True
        await self.navigate()
        self.assertEqual(len(self.tools.interrupts), 2)

    async def test_rejection_does_not_match_across_large_pose_change_or_missing_pose(self):
        await self.navigate()
        await self.watch.review(DetectionReview(candidate_id=self.watch.pending['candidate_id'],
                                               verdict='rejected', evidence='不是目标'))
        item = Detection(label='sofa', confidence=.9, bbox=self.client.bbox, continuity_id='new')
        for pose in ({'x': 1., 'y': 0., 'yaw': 0.},
                     {'x': 0., 'y': 0., 'yaw': 1.}, {}):
            with self.subTest(pose=pose):
                self.assertEqual(self.watch._suppressed([item], {'pose': pose}), set())

    async def test_ambiguous_overlapping_boxes_are_not_suppressed(self):
        await self.navigate()
        await self.watch.review(DetectionReview(candidate_id=self.watch.pending['candidate_id'],
                                               verdict='rejected', evidence='不是目标'))
        items = [Detection(label='sofa', confidence=.9, bbox=self.client.bbox, continuity_id=identity)
                 for identity in ('new-a', 'new-b')]
        self.assertEqual(self.watch._suppressed(items, {'pose': self.tools.pose}), set())

    async def test_uncertain_exits_search_until_explicitly_reenabled(self):
        await self.navigate()
        await self.watch.review(DetectionReview(candidate_id=self.watch.pending['candidate_id'],
                                               verdict='uncertain', evidence='需靠近核实'))
        self.assertFalse(self.watch.ignored)
        self.assertFalse(self.watch.enabled)
        self.assertIsNone(self.watch.task)
        self.assertEqual(self.client.closed, ['session'])
        calls = self.client.calls
        self.assertEqual((await self.navigate())['reason'], 'model_stop')
        self.assertEqual(self.client.calls, calls)
        self.assertEqual(len(self.tools.interrupts), 1)

        await self.watch.configure(True, ['sofa'])
        self.assertEqual((await self.navigate())['reason'], 'agent_interrupt')
        self.assertEqual(len(self.tools.interrupts), 2)

    async def test_switch_to_approach_stops_existing_sampler_before_motion(self):
        await self.navigate()
        calls = self.client.calls
        self.assertEqual((await self.navigate(False))['reason'], 'model_stop')
        self.assertEqual(self.client.calls, calls)
        self.assertIsNone(self.watch.task)
        self.assertIsNone(self.watch.session)

    async def test_confirmation_disables_background(self):
        await self.navigate()
        await self.watch.review(DetectionReview(candidate_id=self.watch.pending['candidate_id'],
                                               verdict='confirmed', evidence='当前棕色沙发明确可见'))
        self.assertFalse(self.watch.enabled)
        self.assertIsNone(self.watch.task)
        self.assertEqual(self.client.closed, ['session'])

    async def test_rotation_pause_cancels_request_and_resets_identity(self):
        self.client.delay = .2
        task = asyncio.create_task(self.tools.navigate(.3))
        await self.watch.start(task, 1)
        await asyncio.sleep(.01)
        await self.watch.pause()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(self.tools.interrupts)
        self.assertIsNone(self.watch.task)
        self.assertFalse(self.watch.ignored)

    async def test_navigation_cancel_closes_detector_and_waits_for_motion(self):
        self.client.delay = 1.
        task = asyncio.create_task(self.navigate(seconds=10.))
        while self.client.calls == 0:
            await asyncio.sleep(.001)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(self.tools.navigation_id)
        self.assertIsNone(self.watch.task)
        self.assertIsNone(self.watch.session)
        self.assertIsNone(self.watch.navigation)
        self.assertFalse(self.tools.interrupts)

    async def test_navigation_failure_closes_detector_without_late_interrupt(self):
        self.client.delay = 1.
        async def fail():
            self.tools.navigation_id = 'failed'
            try:
                while self.client.calls == 0:
                    await asyncio.sleep(.001)
                raise RuntimeError('navigation failure')
            finally:
                self.tools.navigation_id = None
        with self.assertRaisesRegex(RuntimeError, 'navigation failure'):
            await self.watch.execute(fail, 3, search_navigation=True)
        self.assertIsNone(self.watch.task)
        self.assertIsNone(self.watch.navigation)
        self.assertIsNone(self.watch.session)
        self.assertFalse(self.tools.interrupts)

    async def test_identical_configuration_keeps_state(self):
        self.watch.ignored['x'] = {'last_seen': time.monotonic()}
        generation = self.watch.generation
        await self.watch.configure(True, ['SOFA'])
        self.assertEqual(self.watch.generation, generation)
        self.assertIn('x', self.watch.ignored)

    async def test_fixed_vocabulary_falls_back(self):
        self.client.vocabulary = 'fixed'
        result = await self.navigate()
        self.assertEqual(result['reason'], 'model_stop')
        self.assertEqual(self.watch.status, 'unsupported')
        self.assertEqual(self.client.calls, 0)

    async def test_detector_failure_does_not_cancel_navigation(self):
        self.client.fail = True
        result = await self.navigate()
        self.assertEqual(result['reason'], 'model_stop')
        self.assertFalse(self.tools.interrupts)
        self.assertEqual(self.watch.status, 'degraded')

    async def test_service_restart_reopens_expired_session_during_navigation(self):
        import aiohttp
        opened = []
        async def open_session(labels):
            identity = 'session-' + str(len(opened) + 1)
            opened.append(identity)
            return identity
        original = self.client.detect
        async def detect(session, jpeg, stamp):
            if session == 'session-1':
                raise aiohttp.ClientResponseError(None, (), status=404)
            return await original(session, jpeg, stamp)
        self.client.open, self.client.detect = open_session, detect
        result = await self.navigate(seconds=.2)
        self.assertEqual(result['reason'], 'agent_interrupt')
        self.assertEqual(opened, ['session-1', 'session-2'])
        self.assertIn('session-1', self.client.closed)
        self.assertEqual(len(self.tools.interrupts), 1)

    async def test_service_outage_recovers_and_disabling_cancels_retries(self):
        attempts = 0
        original = self.client.capabilities
        async def capabilities():
            nonlocal attempts
            attempts += 1
            if attempts <= 4:
                raise ConnectionError('service restarting')
            return await original()
        self.client.capabilities = capabilities
        self.assertEqual((await self.navigate(seconds=.3))['reason'], 'agent_interrupt')
        self.assertEqual(attempts, 5)
        await self.watch.configure(False, [])
        attempts = 0
        await self.watch.configure(True, ['sofa'])
        navigation = asyncio.create_task(self.tools.navigate(.1))
        await self.watch.start(navigation, 4)
        await asyncio.sleep(.01)
        await self.watch.configure(False, [])
        captured = attempts
        await asyncio.sleep(.03)
        self.assertEqual(attempts, captured)
        self.assertIsNone(self.watch.task)
        await navigation

    async def test_transient_capture_failure_recovers_in_the_same_navigation(self):
        capture = self.tools.snapshot_observation
        attempts = 0
        async def fail_once():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError('camera busy')
            return await capture()
        self.tools.snapshot_observation = fail_once
        result = await self.navigate(seconds=.2)
        self.assertEqual(result['reason'], 'agent_interrupt')
        self.assertGreaterEqual(attempts, 2)
        errors = [data for kind, data in self.events if kind == 'target_watch_degraded']
        self.assertEqual(errors[0]['stage'], 'observation')
        self.assertTrue(errors[0]['retrying'])
        self.assertEqual(self.client.closed, [])

    async def test_sampler_programming_error_is_reported_as_stopped(self):
        async def broken():
            raise AttributeError('snapshot missing')
        self.tools.snapshot_observation = broken
        self.assertEqual((await self.navigate())['reason'], 'model_stop')
        errors = [data for kind, data in self.events if kind == 'target_watch_degraded']
        self.assertEqual(errors[-1]['stage'], 'sampler')
        self.assertFalse(errors[-1]['retrying'])
        self.assertEqual(self.client.calls, 0)

    async def test_backend_without_streaming_is_rejected_before_sampling(self):
        self.tools.supports_streaming_observation = False
        context = await self.watch.configure(True, ['sofa'])
        self.assertFalse(context['enabled'])
        self.assertEqual(context['status'], 'unsupported')
        await self.navigate()
        self.assertEqual(self.client.calls, 0)

    async def test_cannot_review_other_candidate(self):
        await self.navigate()
        decision = Decision(action='look', direction='left', reason='核实',
                            detection_review={'candidate_id':'other', 'verdict':'rejected', 'evidence':'错误'})
        with self.assertRaises(ValueError):
            self.watch.validate(decision)
        self.assertIsNotNone(self.watch.pending)

    async def test_response_request_identity_is_verified(self):
        client = DetectorClient(self.settings.target_watch)
        async def request(*args):
            return FrameResponse(request_id='wrong', camera_stamp_ns=1, width=480,
                                 height=270, inference_ms=1).model_dump(mode='json')
        client.request = request
        with self.assertRaises(ValueError):
            await client.detect('s', b'jpeg', 1)


class InspectionLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_detection_then_uncertain_approach_has_no_second_interrupt(self):
        await self.run_inspection_loop(detected=True)

    async def test_vlm_clue_without_detector_hit_also_disables_search_monitor(self):
        await self.run_inspection_loop(detected=False)

    async def test_confirmed_target_via_passage_has_no_background_interruption(self):
        await self.run_inspection_loop(detected=True, confirmed=True)

    async def run_inspection_loop(self, *, detected, confirmed=False):
        detector = FakeDetector()
        detector.detected = detected

        class DetectorScope:
            async def __aenter__(self):
                return detector

            async def __aexit__(self, *args):
                pass

        class Backend(DemoBackend):
            supports_streaming_observation = True
            active_navigation_id = None

            def __init__(self):
                super().__init__()
                self.stopped = asyncio.Event()
                self.interrupts = []
                self.samples_per_navigation = []

            async def snapshot_observation(self):
                jpeg, observation = await super().observe()
                observation['camera_stamp_ns'] = time.time_ns()
                return jpeg, observation

            def request_navigation_interrupt(self, identity, reason):
                if identity != self.active_navigation_id:
                    return False
                self.interrupts.append(reason)
                self.stopped.set()
                return True

            async def navigate(self, instruction, *, seconds):
                self.actions.append(('navigate', instruction))
                self.active_navigation_id = str(len(self.actions))
                self.stopped.clear()
                calls = detector.calls
                try:
                    try:
                        await asyncio.wait_for(self.stopped.wait(), .08)
                        return {'reason': 'agent_interrupt', 'detail': 'target_detected'}
                    except TimeoutError:
                        return {'reason': 'model_stop'}
                finally:
                    self.samples_per_navigation.append(detector.calls - calls)
                    self.active_navigation_id = None

        class Model(DemoModel):
            def __init__(self):
                super().__init__()
                self.step = 0

            async def complete(self, body):
                assert backend.active_navigation_id is None, '导航期间不应调用千问'
                context = json.loads(body['messages'][1]['content'][0]['text'])
                if context.get('operation') != 'verify_completion':
                    self.step += 1
                    if self.step in (1, 2):
                        response = {
                            'action': 'navigate', 'candidate_id': 'new_1',
                            'skill': 'search_navigation' if self.step == 1 else 'approach_target',
                            'view_id': context['observation_id'],
                            'instruction': 'Go through the doorway towards the white fixture.',
                            'reason': '搜索' if self.step == 1 else '靠近疑似目标核实',
                            'scene': {'target_seen': 'no' if self.step == 1 else ('yes' if confirmed else 'uncertain')},
                            'candidates': [{'id': 'new_1', 'kind': 'open_passage',
                                            'description': '前方门洞', 'evidence': '当前图可见门内空地',
                                            'expected_new_view': '看清门内物体'}],
                        }
                        if self.step == 1:
                            response['target_labels'] = ['sofa']
                        if self.step == 2:
                            pending = context['target_watch']['pending']
                            if detected:
                                assert pending is not None
                                response['detection_review'] = {
                                    'candidate_id': pending['candidate_id'], 'verdict': 'confirmed' if confirmed else 'uncertain',
                                    'evidence': '疑似目标被部分遮挡，需要靠近核实',
                                }
                            response['previous_action_review'] = {
                                'outcome': 'progress', 'evidence': '发现可跟进的物体线索',
                            }
                    else:
                        response = {'action': 'finish', 'reason': '核实完成',
                                    'scene': {'target_seen': 'yes'}, 'evidence': '当前目标清晰可见'}
                    self.responses = [response]
                return await super().complete(body)

        with tempfile.TemporaryDirectory() as directory:
            settings = agent_settings()
            settings = settings.model_copy(update={
                'storage': settings.storage.model_copy(update={'runs_dir': Path(directory)}),
                'memory': settings.memory.model_copy(update={'compare_regions_when_stopped': False}),
                'target_watch': settings.target_watch.model_copy(update={
                    'server_url': 'http://localhost:8060', 'frame_interval_s': .001,
                }),
            })
            record = {'id': 'inspection-loop', 'goal': '找到沙发', 'status': 'running',
                      'started_at': datetime.now(timezone.utc).isoformat()}
            request = TaskRequest(goal='找到沙发', max_steps=3, max_seconds=10, navigation_seconds=5.)
            (Path(directory) / record['id'] / 'images').mkdir(parents=True)
            backend, model = Backend(), Model()
            with patch('navigation_agent.agent.core.runner.DetectorClient', return_value=DetectorScope()):
                await run_task(record, request, NavigationTools(backend),
                               settings=settings, api_key='', model=model)
            self.assertEqual(record['status'], 'completed', record)
            self.assertEqual(len(model.requests), 3)
            self.assertEqual(len(backend.actions), 2)
            self.assertEqual(backend.interrupts, ['target_detected'] if detected else [])
            self.assertGreater(backend.samples_per_navigation[0], 0)
            self.assertEqual(backend.samples_per_navigation[1], 0)
            contexts = [json.loads(body['messages'][1]['content'][0]['text']) for body in model.requests]
            self.assertTrue(all({item['name'] for item in context['available_skills']}
                                == {'search_navigation', 'approach_target'} for context in contexts))
            self.assertEqual(contexts[1]['previous_skill'], 'search_navigation')
            self.assertEqual(contexts[2]['previous_skill'], 'approach_target')
            self.assertEqual(contexts[2]['execution']['last_action']['action'], 'navigate')
            context = contexts[-1]
            self.assertFalse(context['target_watch']['enabled'])
            if detected:
                self.assertEqual(context['target_watch']['last_review']['verdict'], 'confirmed' if confirmed else 'uncertain')
