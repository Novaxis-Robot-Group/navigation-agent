import json

import pytest
from vln_mpc.mpc_node import (
    MPC_CONFIG_NAMES,
    OBJNAV_MODE,
    TRACK_MODE,
    parse_vln_response,
    scale_command,
    should_complete_task,
    validate_mpc_config,
)


def response(*, stop):
    return json.dumps(
        {
            "episode": 3,
            "seq": 17,
            "capture_stamp_ns": 123456789,
            "frame_id": "base_link",
            "waypoints": [[1.0, 0.2, -0.1]],
            "stop": stop,
            "visible": True,
        }
    )


def test_parse_vln_response_keeps_stop():
    episode, sequence, stamp_ns, stop, waypoints = parse_vln_response(
        response(stop=True), "base_link"
    )
    assert (episode, sequence, stamp_ns) == (3, 17, 123456789)
    assert stop is True
    assert waypoints == [(1.0, 0.2, -0.1)]


@pytest.mark.parametrize("stop", [False, None])
def test_objnav_continues_without_true_stop(stop):
    assert not should_complete_task(OBJNAV_MODE, stop)


def test_objnav_completes_on_true_stop():
    assert should_complete_task(OBJNAV_MODE, True)


def test_track_ignores_true_stop():
    assert not should_complete_task(TRACK_MODE, True)


def test_mpc_config_validation_and_output_scaling():
    config = {name: 1.0 for name in MPC_CONFIG_NAMES}
    assert validate_mpc_config(config) == ""
    assert scale_command((0.8, -0.4), 1.5, 0.5) == pytest.approx((1.2, -0.2))

    config["q_yaw"] = 0.0
    assert validate_mpc_config(config) == ""
    config["w_max"] = 0.0
    assert "limits" in validate_mpc_config(config)


@pytest.mark.parametrize('case, expected', [
    ('fresh', ''), ('received_old', 'VLN trajectory is stale'),
    ('capture_old', 'VLN capture timestamp is stale or invalid'),
    ('capture_future', 'VLN capture timestamp is stale or invalid'),
])
def test_trajectory_timeout_checks_receipt_and_capture_age(case, expected):
    import time
    from types import SimpleNamespace
    from vln_mpc.mpc_node import MpcNode
    now = time.monotonic()
    path = SimpleNamespace(received_s=now, stamp_ns=10_000_000_000, trajectory=[(0, 0, 0)])
    if case == 'received_old':
        path.received_s -= 4
    elif case == 'capture_old':
        path.stamp_ns = 1
    elif case == 'capture_future':
        path.stamp_ns = 20_000_000_000
    node = SimpleNamespace(
        _enabled=True, _vln_state='RUNNING', _odom=SimpleNamespace(received_s=now),
        odom_timeout_s=.5, path_timeout_s=3., capture_timeout_s=5., _path=path,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)),
    )
    assert MpcNode._block_reason(node) == expected


@pytest.mark.parametrize('fault', ['disabled', 'old_capture', 'stale_capture', 'future_capture',
                                   'other_episode', 'old_sequence', 'already_complete', 'fresh'])
def test_stop_response_respects_current_navigation_and_capture_age(fault):
    from types import SimpleNamespace
    from vln_mpc.mpc_node import MpcNode
    finished = []
    node = SimpleNamespace(
        _enabled=fault != 'disabled', _vln_state='RUNNING', base_frame='base_link',
        _goal_reached=fault == 'already_complete', _session_started_ns=8_000_000_000,
        _session_episode=2 if fault == 'other_episode' else None,
        _session_sequence=17 if fault == 'old_sequence' else 0,
        capture_timeout_s=5., _path=None, _task_mode=OBJNAV_MODE,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)),
        _complete_task=lambda episode, seq: finished.append((episode, seq)),
    )
    payload = json.loads(response(stop=True))
    payload['capture_stamp_ns'] = {'old_capture': 7_000_000_000, 'stale_capture': 1,
                                   'future_capture': 11_000_000_000}.get(fault, 9_000_000_000)
    if fault == 'stale_capture':
        node._session_started_ns = 0
    MpcNode._on_vln_response(node, SimpleNamespace(data=json.dumps(payload)))
    assert finished == ([(3, 17)] if fault == 'fresh' else [])


def test_enabling_new_navigation_clears_old_trajectory_and_session():
    from types import SimpleNamespace
    from vln_mpc.mpc_node import MpcNode
    cleared = []
    node = SimpleNamespace(_session_episode=3, _session_sequence=42,
                           _goal_reached=True, _last_error='old error',
                           get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100)),
                           _clear_trajectory=lambda: cleared.append(True), _publish_status=lambda: None)
    MpcNode._on_enable(node, SimpleNamespace(data=True))
    assert node._session_started_ns == 100
    assert node._session_episode is None and node._session_sequence == 0
    assert not node._goal_reached and node._last_error == ''
    assert cleared == [True]


def test_paused_vln_clears_trajectory_without_protocol_error():
    from types import SimpleNamespace
    from vln_mpc.mpc_node import MpcNode
    cleared = []
    node = SimpleNamespace(_vln_state='RUNNING', _last_error='',
                           _clear_trajectory=lambda: cleared.append(True))
    MpcNode._on_vln_status(node, SimpleNamespace(data='PAUSED'))
    assert node._vln_state == 'PAUSED'
    assert node._last_error == ''
    assert cleared == [True]
