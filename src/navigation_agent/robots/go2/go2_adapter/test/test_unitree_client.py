from types import SimpleNamespace
import threading
import time

from go2_adapter.unitree_client import (
    UnitreeClient,
    battery_from_message,
    normalized_mode,
    state_from_message,
)


class _FakeObstacles:
    def __init__(self):
        self.calls = []
        self.enabled = True
        self.read_code = self.write_code = 0

    def SwitchGet(self):
        self.calls.append(('get_avoidance',))
        return self.read_code, self.enabled

    def SwitchSet(self, enabled):
        self.calls.append(('set_avoidance', enabled))
        if self.write_code == 0:
            self.enabled = enabled
        return self.write_code

    def UseRemoteCommandFromApi(self, enabled):
        self.calls.append(("api", enabled))
        return 0

    def Move(self, linear, lateral, angular):
        self.calls.append(("move", linear, lateral, angular))
        return 0


def _control_client():
    client = UnitreeClient.__new__(UnitreeClient)
    client._api_enabled = False
    client._obstacles = _FakeObstacles()
    client.events = []
    client.errors = []
    client.sent = []
    client._event_callback = client.events.append
    client._error_callback = client.errors.append
    client._sent_callback = client.sent.append
    return client


def test_state_from_message_converts_wxyz_to_xyzw():
    message = SimpleNamespace(
        position=[1.0, 2.0, 0.3],
        velocity=[0.4, 0.1, 0.0],
        yaw_speed=-0.2,
        body_height=0.31,
        mode=3,
        gait_type=1,
        error_code=0,
        imu_state=SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0]),
    )
    state = state_from_message(
        message, received_ns=123, received_s=4.5
    )
    assert state.received_ns == 123
    assert state.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert state.position == (1.0, 2.0, 0.3)
    assert state.velocity == (0.4, 0.1, 0.0)


def test_normalized_modes_match_shared_interface():
    assert normalized_mode(0) == "STAND"
    assert normalized_mode(1) == "WALK"
    assert normalized_mode(3) == "WALK"
    assert normalized_mode(7) == "DAMPING"
    assert normalized_mode(10) == "SIT"
    assert normalized_mode(99) == "UNKNOWN"
    assert (
        normalized_mode(0, error_code=1001, body_height=0.07)
        == "DAMPING"
    )
    assert normalized_mode(0, error_code=100, body_height=0.31) == "WALK"


def test_battery_from_message():
    message = SimpleNamespace(
        bms_state=SimpleNamespace(
            soc=73,
            status=1,
            current=0,
            cycle=0,
            cell_vol=[3900] * 15,
        ),
        motor_state=[SimpleNamespace(mode=1, temperature=32)] * 12,
        power_v=29.4,
    )
    battery = battery_from_message(message, received_s=1.0)
    assert battery.percentage == 73.0
    assert battery.voltage == 29.4
    assert battery.motors_ok is True
    assert battery.motor_temps == (32,) * 12


def test_battery_without_motor_states_reports_no_temperatures():
    message = SimpleNamespace(
        bms_state=SimpleNamespace(
            soc=73,
            status=1,
            current=0,
            cycle=0,
            cell_vol=[3900] * 15,
        ),
        motor_state=[],
        power_v=29.4,
    )
    battery = battery_from_message(message, received_s=1.0)
    assert battery.motor_temps is None
    assert battery.motors_ok is None


def test_unpopulated_bms_does_not_report_false_zero_percentage():
    message = SimpleNamespace(
        bms_state=SimpleNamespace(
            soc=0,
            status=0,
            current=0,
            cycle=0,
            cell_vol=[0] * 15,
        ),
        motor_state=[SimpleNamespace(mode=1, temperature=32)] * 12,
        power_v=28.6,
    )
    battery = battery_from_message(message, received_s=1.0)
    assert battery.percentage is None
    assert battery.motor_temps == (32,) * 12


def test_api_control_is_acquired_lazily_and_released_after_zero():
    client = _control_client()
    assert client._enable_api_control() is True
    assert client._api_enabled is True
    client._release_api_control()
    assert client._api_enabled is False
    assert client._obstacles.calls == [
        ('get_avoidance',),
        ("api", True),
        ("move", 0.0, 0.0, 0.0),
        ("api", False),
    ]
    assert client.sent == [(0.0, 0.0, 0.0)]
    assert client.errors == []


def test_disabled_avoidance_is_enabled_and_read_back_before_control():
    client = _control_client()
    client._obstacles.enabled = False
    assert client._enable_api_control()
    assert client._obstacles.calls == [('get_avoidance',), ('set_avoidance', True),
                                       ('get_avoidance',), ('api', True)]


def test_unknown_or_failed_avoidance_does_not_acquire_control():
    for failure in ('read', 'write', 'readback'):
        client = _control_client()
        client._obstacles.enabled = False
        if failure == 'read':
            client._obstacles.read_code = 42
        elif failure == 'write':
            client._obstacles.write_code = 42
        else:
            client._obstacles.SwitchSet = lambda enabled: 0
        assert not client._enable_api_control()
        assert not client._api_enabled
        assert not any(c[0] in ('move', 'api') for c in client._obstacles.calls)
        assert client.errors
        assert client.events[-1]['success'] is False


def test_slow_acquisition_never_replays_expired_movement():
    client = _control_client()
    client._condition = threading.Condition()
    client._actions = []
    client._closed = False
    client._command_max_age_s = .01
    client._latest_move = (.3, 0., 0.)
    client._latest_move_s = time.monotonic()
    acquire = client._enable_api_control
    acquired = threading.Event()
    def slow_acquire():
        time.sleep(.03)
        result = acquire()
        acquired.set()
        return result
    client._enable_api_control = slow_acquire
    worker = threading.Thread(target=client._command_loop)
    worker.start()
    try:
        assert acquired.wait(1)
        time.sleep(.03)
        with client._condition:
            client._closed = True
            client._condition.notify_all()
    finally:
        worker.join(1)
    assert not worker.is_alive()
    assert not any(c[0] == 'move' for c in client._obstacles.calls)


def test_release_future_waits_for_sdk_and_propagates_failure():
    from collections import deque
    import pytest
    for failed in (False, True):
        client = _control_client()
        client._api_enabled = True
        client._condition = threading.Condition()
        client._actions = deque()
        client._closed = False
        client._latest_move = None
        started, finish = threading.Event(), threading.Event()
        def release(enabled):
            assert enabled is False
            started.set()
            assert finish.wait(1)
            return 42 if failed else 0
        client._obstacles.UseRemoteCommandFromApi = release
        worker = threading.Thread(target=client._command_loop)
        worker.start()
        try:
            completion = client.release_control()
            assert started.wait(1)
            assert not completion.done()
            finish.set()
            if failed:
                with pytest.raises(RuntimeError, match='release failed: 42'):
                    completion.result(timeout=1)
                assert client._api_enabled
            else:
                assert completion.result(timeout=1) is None
                assert not client._api_enabled
        finally:
            finish.set()
            with client._condition:
                client._closed = True
                client._condition.notify_all()
            worker.join(1)
            assert not worker.is_alive()


def test_zero_exception_still_attempts_sdk_release_and_reports_failure():
    import pytest
    client = _control_client()
    client._api_enabled = True
    def failed_move(*args):
        raise RuntimeError('transport error')
    client._obstacles.Move = failed_move
    with pytest.raises(RuntimeError, match='zero command failed'):
        client._release_api_control()
    assert ('api', False) in client._obstacles.calls
    assert not client._api_enabled
    assert not client.events[-1]['success']
