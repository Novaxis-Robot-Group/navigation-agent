import base64
import json
import threading

import cv2
import numpy as np
import pytest
from vln_client.vln_client import (
    CameraFrame,
    VlnClient,
    normalize_server_url,
)


class Logger:
    def info(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass


def test_accepts_next_camera_frame_only_after_request_completes():
    client = VlnClient.__new__(VlnClient)
    client._lock = threading.Lock()
    client._paused = False
    client._generation = 1
    client._action_token = 1
    client._instruction = "follow the person"
    client._session = 1
    client._connected = True
    client._in_flight = 0
    client._pending_frame = None
    frame = {
        "stamp_sec": 1,
        "stamp_nanosec": 2,
        "width": 2,
        "height": 1,
        "step": 6,
        "rgb": b"\x00" * 6,
    }

    assert client.offer_frame(**frame) is True
    assert client.offer_frame(**frame) is False
    assert client._take_frame(1) is not None
    assert client._in_flight == 1
    assert client.offer_frame(**frame) is False

    client._set_runtime(1, in_flight=0)
    assert client.offer_frame(**frame) is True


def test_accepts_and_resizes_compressed_camera_frame():
    source = np.zeros((90, 160, 3), dtype=np.uint8)
    source[:, :, 1] = 255
    success, encoded = cv2.imencode(".jpg", source)
    assert success
    frame = CameraFrame(
        stamp_sec=1,
        stamp_nanosec=2,
        width=0,
        height=0,
        step=0,
        rgb=b"",
        jpeg=encoded.tobytes(),
    )

    payload = base64.b64decode(VlnClient._encode_image(frame))
    decoded = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    assert decoded.shape[:2] == (270, 480)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.8.204.70:8050", "ws://10.8.204.70:8050"),
        ("ws://robot.local:8050/path", "ws://robot.local:8050/path"),
        ("wss://example.com/trackvla", "wss://example.com/trackvla"),
        ("", ""),
    ],
)
def test_normalizes_server_url(value, expected):
    assert normalize_server_url(value) == expected


@pytest.mark.parametrize(
    "value",
    ["http://example.com", "ws://", "ws://user@example.com", "ws://host:0"],
)
def test_rejects_invalid_server_url(value):
    with pytest.raises(ValueError, match="VLN server URL"):
        normalize_server_url(value)


def test_cannot_start_without_server_url():
    client = VlnClient(logger=Logger())
    try:
        with pytest.raises(ValueError, match="server URL is not set"):
            client.start("follow the person")
    finally:
        client.close()


def test_switches_server_url_without_stopping_active_instruction():
    client = VlnClient(logger=Logger(), server_url="ws://old-host:8050")
    try:
        episode = client.start("follow the person")

        assert client.set_server_url("new-host:8051") is True
        snapshot = client.snapshot()

        assert snapshot.server_url == "ws://new-host:8051"
        assert snapshot.state == "RUNNING"
        assert snapshot.connected is False
        assert snapshot.episode == episode
        assert client.set_server_url("ws://new-host:8051") is False
    finally:
        client.close()


def test_parses_strict_inference_response():
    payload = VlnClient._parse_inference(
        json.dumps(
            {
                "data": {
                    "rc": 0,
                    "seq": 7,
                    "actions": [1, 0.2, -0.1],
                    "stop": False,
                    "visible": True,
                    "pointing": {
                        "apos_state": "rot_left",
                        "opos_state": "point",
                        "apos_px": None,
                        "opos_px": [98.6, 23.9],
                    },
                }
            }
        )
    )
    assert payload.sequence == 7
    assert payload.waypoints == [(1.0, 0.2, -0.1)]
    assert payload.stop is False
    assert payload.visible is True
    assert payload.apos_state == "rot_left"
    assert payload.opos_state == "point"
    assert payload.apos_px is None
    assert payload.opos_px == (98.6, 23.9)


def test_accepts_unknown_semantic_state_strings():
    payload = VlnClient._parse_inference(
        json.dumps(
            {
                "data": {
                    "rc": 0,
                    "seq": 8,
                    "actions": [],
                    "pointing": {
                        "apos_state": "future_special",
                        "opos_state": None,
                        "apos_px": [255, 135],
                        "opos_px": None,
                    },
                }
            }
        )
    )
    assert payload.apos_state == "future_special"
    assert payload.opos_state is None
    assert payload.apos_px == (255.0, 135.0)
    assert payload.opos_px is None


def test_rejects_response_without_data_sequence():
    with pytest.raises(RuntimeError, match="data.seq"):
        VlnClient._parse_inference(
            json.dumps(
                {
                    "seq": 7,
                    "data": {
                        "rc": 0,
                        "actions": [1, 0, 0],
                    },
                }
            )
        )


def test_rejects_response_without_pointing_object():
    with pytest.raises(RuntimeError, match="data.pointing"):
        VlnClient._parse_inference(
            json.dumps(
                {
                    "data": {
                        "rc": 0,
                        "seq": 7,
                        "actions": [1, 0, 0],
                    }
                }
            )
        )


def test_rejects_failed_inference():
    with pytest.raises(RuntimeError, match="next failed"):
        VlnClient._parse_inference(
            json.dumps({"data": {"rc": 1, "message": "failed"}})
        )


def test_history_resume_preserves_episode_but_discards_old_action_results():
    from unittest.mock import patch
    from vln_client.vln_client import InferenceResult
    with patch('threading.Thread.start'):
        client = VlnClient(logger=Logger(), server_url='ws://offline')
    client.start('Go to the door.', generation=1)
    session = client._session
    client._connected = True
    client.offer_compressed_frame(stamp_sec=1, stamp_nanosec=0, jpeg=b'old')
    frame = client._take_frame(session)
    episode = client.snapshot().episode
    client.pause(1)
    assert not client.offer_compressed_frame(stamp_sec=2, stamp_nanosec=0, jpeg=b'paused')
    assert client.resume('Go to the door.', 2)
    assert client.snapshot().episode == episode
    old = InferenceResult(episode=episode, sequence=1, frame=frame, waypoints=[(.1, 0., 0.)],
                          stop=False, visible=None, apos_state=None, opos_state=None,
                          apos_px=None, opos_px=None, latency_ms=1.,
                          generation=frame.generation, action_token=frame.action_token)
    client._queue_result(session, old)
    assert client.take_results() == []
    with pytest.raises(ValueError, match='stale'):
        client.pause(1)
    with pytest.raises(ValueError, match='stale'):
        client.finish(1)
    client._set_runtime(session, in_flight=0)
    assert client.offer_compressed_frame(stamp_sec=3, stamp_nanosec=0, jpeg=b'new')
    new_frame = client._take_frame(session)
    from dataclasses import replace
    current = replace(old, frame=new_frame, generation=2, action_token=new_frame.action_token, stop=True)
    client._queue_result(session, current)
    assert client.take_results() == [current]
    client.pause(2)
    assert not client.resume('Go to the door.', 3)
    assert client.snapshot().episode == episode + 1


def test_changed_instruction_and_disconnection_invalidate_history():
    from unittest.mock import patch
    with patch('threading.Thread.start'):
        client = VlnClient(logger=Logger(), server_url='ws://offline')
    client.start('Go to the door.', generation=1)
    client._connected = True
    client.pause(1)
    assert not client.resume('Go to the chair.', 2)
    client._connected = True
    client.pause(2)
    client._set_runtime(client._session, state='ERROR', connected=False, error='disconnected')
    assert not client.resume('Go to the chair.', 3)
