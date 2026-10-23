"""Unit tests for the robot diagnostics parsing in the web node."""

from types import SimpleNamespace

from vln_web.web_node import VlnWebNode


def _status(level=0, message="online"):
    return SimpleNamespace(level=level, message=message)


def test_parse_robot_diagnostics_includes_motor_temperatures():
    values = {
        "adapter": "go2",
        "connected": "True",
        "robot_id": "go2",
        "mode": "WALK",
        "battery": "73",
        "imu": "OK",
        "motor": "OK",
        "motor_temps": "[32,33,34,35,36,37,38,39,40,41,42,43]",
        "motor_temp_max": "43",
        "control_source": "disabled",
        "policy": "",
        "vln_policy": "",
    }
    data = VlnWebNode._parse_robot_diagnostics(_status(), values)
    assert data["motor_temps"] == [
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    ]
    assert data["motor_temp_max"] == 43
    assert data["connected"] is True
    assert data["battery"] == 73.0
    assert data["mode"] == "WALK"


def test_parse_robot_diagnostics_tolerates_invalid_motor_temperatures():
    values = {
        "adapter": "go2",
        "motor_temps": "[32,",
        "motor_temp_max": "not-a-number",
    }
    data = VlnWebNode._parse_robot_diagnostics(_status(), values)
    assert data["motor_temps"] is None
    assert data["motor_temp_max"] is None


def test_parse_robot_diagnostics_rejects_non_integer_motor_temperatures():
    values = {
        "adapter": "go2",
        "motor_temps": "[32,33.5]",
        "motor_temp_max": "33",
    }
    data = VlnWebNode._parse_robot_diagnostics(_status(), values)
    assert data["motor_temps"] is None
    assert data["motor_temp_max"] == 33


def test_parse_robot_diagnostics_defaults_missing_motor_temperatures():
    data = VlnWebNode._parse_robot_diagnostics(_status(), {"adapter": "go2"})
    assert data["motor_temps"] is None
    assert data["motor_temp_max"] is None
    assert data["mode"] == "UNKNOWN"


def test_parse_robot_diagnostics_preserves_execution_and_safety_errors():
    data = VlnWebNode._parse_robot_diagnostics(_status(2, 'Move failed'),
        {'sdk_error': 'Move failed', 'safety_error': 'avoidance failed'})
    assert data['sdk_error'] == 'Move failed'
    assert data['safety_error'] == 'avoidance failed'
    assert data['level'] == 2


def test_stale_stop_does_not_complete_current_navigation():
    import json
    messages, completed = [], []
    node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)),
        get_logger=lambda: SimpleNamespace(warning=messages.append),
        _complete_objnav=lambda: completed.append(True),
        web_server=SimpleNamespace(vln_running=lambda: True),
    )
    payload = {'episode': 1, 'seq': 1, 'capture_stamp_ns': 1, 'frame_id': 'base_link',
               'waypoints': [], 'stop': True}
    VlnWebNode._on_vln_response(node, SimpleNamespace(data=json.dumps(payload)))
    assert not completed
    assert messages == ['discarding stale VLN response']
