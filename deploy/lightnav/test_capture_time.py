"""在 LightNav 服务环境验证时间补丁，不加载权重。"""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lightnav.inference.policies import NavigationPolicy
from lightnav.processing import VLNQwen3VLProcessor
from lightnav.serving.tracking_service import BatchedTrackingService


class CaptureTimeTests(unittest.TestCase):
    def policy(self, window=4):
        engine = SimpleNamespace(
            bundle=SimpleNamespace(video_size=(4, 4), slowfast_tiers=None),
            reset_episode_state=lambda: None,
        )
        return NavigationPolicy(engine, num_history_frames=window)

    def test_variable_intervals_keep_frame_ids_and_reset(self):
        policy = self.policy()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for stamp in (1_000_000_000, 1_600_000_000, 2_300_000_000):
            policy.observe(frame, capture_stamp_ns=stamp)
        self.assertEqual(policy._history_frame_ids, [0, 1, 2])
        self.assertEqual(policy._history_capture_times, [0, .6, 1.3])
        policy.reset()
        self.assertEqual(policy._history_capture_times, [])
        self.assertIsNone(policy._capture_origin_ns)
        policy.observe(frame)
        self.assertEqual(policy._history_capture_times, [])

    def test_bad_or_mixed_timestamps_are_rejected_without_appending(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for bad in (0, -1, True, 1.2, "1"):
            with self.assertRaises(ValueError):
                self.policy().observe(frame, capture_stamp_ns=bad)
        policy = self.policy()
        policy.observe(frame, capture_stamp_ns=100)
        for stamp in (100, 99, None):
            with self.assertRaises(ValueError):
                policy.observe(frame, capture_stamp_ns=stamp)
        self.assertEqual(policy._history_frame_ids, [0])
        legacy = self.policy()
        legacy.observe(frame)
        with self.assertRaises(ValueError):
            legacy.observe(frame, capture_stamp_ns=100)

    def test_ring_trims_times_with_images(self):
        policy = self.policy(2)
        for stamp in (100, 200, 300):
            policy.observe(np.zeros((4, 4, 3), dtype=np.uint8), capture_stamp_ns=stamp)
        self.assertEqual(policy._history_frame_ids, [1, 2])
        self.assertEqual(policy._history_capture_times, [1e-7, 2e-7])

    def test_relative_time_uses_global_capture_reference_and_odd_padding(self):
        processor = VLNQwen3VLProcessor.__new__(VLNQwen3VLProcessor)
        with patch.dict(os.environ, VLN_TIMESTAMP_RELATIVE="1"):
            self.assertEqual(processor._calculate_timestamps(
                [0, 1], 4, times_s=[0, .6], reference_s=1.3,
            ), [1.0])
            self.assertEqual(processor._calculate_timestamps(
                [2], 4, times_s=[1.3], reference_s=1.3,
            ), [0.0])
            self.assertEqual(processor._calculate_timestamps([0, 1], 4), [.125])
        with patch.dict(os.environ, VLN_TIMESTAMP_RELATIVE="0"):
            self.assertEqual(processor._calculate_timestamps(
                [0, 1], 4, times_s=[0, .6], reference_s=1.3,
            ), [.3])

    def test_sample_maps_sparse_ids_without_changing_cache_keys(self):
        session = SimpleNamespace(
            _get_video_tensor=lambda: "video", _history_frame_ids=[7, 8, 9],
            _history_capture_times=[3.0, 3.8, 4.4], instruction="Go to the door.",
        )
        service = BatchedTrackingService.__new__(BatchedTrackingService)
        service.serve_task = "vln"
        service.bundle = None
        sample = {"video_segments": [{"frame_indices": [7, 9]}, {"frame_indices": [8, 9]}]}
        with patch("lightnav.serving.tracking_service.build_vln_traj_sample", return_value=sample):
            result = service._build_sample_for_session(session)
        self.assertEqual(result["video_capture_times_s"], [[3.0, 4.4], [3.8, 4.4]])
        self.assertEqual(result["video_segments"][0]["frame_indices"], [7, 9])


if __name__ == "__main__":
    unittest.main()
