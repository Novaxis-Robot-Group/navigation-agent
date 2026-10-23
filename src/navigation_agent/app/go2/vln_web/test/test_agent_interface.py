"""本体 Agent 协议测试：使用替代状态与服务，不连接硬件。"""
import asyncio
import base64
import json
import math
import queue
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from nav_msgs.msg import Odometry
from vln_web.web_server import WebServer
from vln_web.web_node import VlnWebNode


class Socket:
    closed = False

    def __init__(self):
        self.messages = []

    async def send_json(self, value):
        self.messages.append(value)

    async def close(self):
        self.closed = True


def server():
    return WebServer(
        host='127.0.0.1', port=0,
        web_dir=Path(__file__).parents[1] / 'web',
        image_topic='camera/color/image_raw',
        manual_linear_limit=1., manual_angular_limit=1.,
        manual_linear_accel=1., manual_angular_accel=1.,
        commands=queue.Queue(), logger=SimpleNamespace(info=lambda _: None),
    )


def odom(stamp=1_000_000_000, x=1., yaw=0.):
    return {'stamp_ns': stamp, 'frame_id': 'odom', 'pose_valid': True,
            'pose': {'x': x, 'y': 2., 'z': .3, 'yaw': yaw},
            'twist_linear': [0., 0., 0.], 'twist_angular': [0., 0., 0.]}


def observe(s, query=None):
    response = asyncio.run(s._observation_handler(SimpleNamespace(query=query or {})))
    return json.loads(response.text)


def frame(s, stamp=1_000_000_000, jpeg=b'\xff\xd8test'):
    s.update_frame(jpeg, time.monotonic(), stamp_ns=stamp,
                   frame_id='camera_color_optical_frame', capture_age_s=0.)


def test_atomic_observation_matches_capture_pose_not_latest_pose():
    s = server()
    s.update_odom(odom())
    frame(s)
    s.update_odom(odom(1_100_000_000, x=2.))
    value = observe(s)
    assert base64.b64decode(value['jpeg_base64']) == b'\xff\xd8test'
    assert value['pose']['x'] == 1.
    assert value['odom']['pose']['x'] == 2.
    assert value['pose_alignment'] == 'nearest_odom'
    with pytest.raises(web.HTTPConflict):
        observe(s, {'after_sequence': str(value['sequence'])})
    frame(s, 1_100_000_000, b'next')
    assert observe(s, {'after_sequence': str(value['sequence'])})['pose']['x'] == 2.


@pytest.mark.parametrize('fault', ['camera_old', 'capture_old', 'capture_future', 'no_odom', 'odom_old', 'mismatch', 'invalid_pose'])
def test_observation_rejects_invalid_or_stale_inputs(fault):
    s = server()
    s.update_odom(odom())
    frame(s)
    if fault == 'camera_old':
        s._frame_received_s -= 2
    elif fault == 'capture_old':
        s._frame_capture_age_s = 2
    elif fault == 'capture_future':
        s._frame_capture_age_s = -2
    elif fault == 'no_odom':
        s._odom_history.clear()
    elif fault == 'odom_old':
        s._odom_history[-1]['received_s'] -= 2
    elif fault == 'mismatch':
        s._frame_stamp_ns += 500_000_000
    else:
        s._odom_history[-1]['pose_valid'] = False
    with pytest.raises(web.HTTPServiceUnavailable):
        observe(s)


def test_odom_clock_reset_invalidates_old_frame():
    s = server()
    s.update_odom(odom())
    frame(s)
    s.update_odom(odom(100))
    with pytest.raises(web.HTTPServiceUnavailable):
        observe(s)


def test_ros_odom_exports_pose_and_rejects_zero_quaternion():
    s = server()
    node = SimpleNamespace(web_server=s)
    msg = Odometry()
    msg.header.frame_id = 'odom'
    msg.header.stamp.sec = 1
    msg.pose.pose.position.x = 2.
    msg.pose.pose.orientation.z = math.sin(.6)
    msg.pose.pose.orientation.w = math.cos(.6)
    VlnWebNode._on_odom(node, msg)
    assert s._odom['pose']['yaw'] == pytest.approx(1.2)
    assert s._odom['pose']['x'] == 2.
    msg.pose.pose.orientation.z = 0.
    msg.pose.pose.orientation.w = 0.
    VlnWebNode._on_odom(node, msg)
    assert not s._odom['pose_valid']


def test_model_stop_survives_reset_and_idle_until_next_navigation():
    s = server()
    s.reset_vln('go to chair', 'objnav')
    s.update_vln_response({'stop': True, 'episode': 4, 'last_sequence': 7}, {}, time.monotonic(), capture_stamp_ns=time.time_ns())
    s.reset_vln('', 'objnav')
    s.update_vln_status('IDLE')
    assert s._snapshot()['agent']['terminal']['reason'] == 'model_stop'
    assert s._snapshot()['vln']['stop'] is None
    s.reset_vln('go to table', 'objnav')
    assert s._snapshot()['agent']['terminal'] is None


def test_expired_owner_releases_once_and_blocks_takeover_until_ack():
    async def run():
        s = server()
        owner, other = Socket(), Socket()
        s._agent_deadlines[owner] = time.monotonic() - 1
        s._websockets[owner] = 'owner'
        s._auto_controller = owner
        await s._expire_agent_leases()
        assert owner.closed and s._auto_controller is None
        assert s._commands.get_nowait() == ('agent_release', {'manual': False, 'automatic': True}, 'owner')
        await s._handle_browser_message(other, 'other', {'type': 'acquire_control'})
        assert not other.messages[-1]['ok']
        assert s._commands.empty()
        s._finish_agent_release('owner', True, 'disabled')
        assert not s._release_pending
    asyncio.run(run())


def test_agent_stop_cannot_stop_another_owner():
    async def run():
        s = server()
        owner, other = Socket(), Socket()
        s._auto_controller = owner
        s._agent_deadlines[other] = time.monotonic() + 10
        await s._handle_browser_message(other, 'other', {'type': 'stop'})
        assert s._auto_controller is owner and s._commands.empty()
    asyncio.run(run())


def test_heartbeat_and_rejected_unsafe_agent_commands():
    async def run():
        s = server()
        agent = Socket()
        s._agent_deadlines[agent] = time.monotonic() + .1
        await s._handle_browser_message(agent, 'agent', {'type': 'agent_heartbeat'})
        assert s._agent_deadlines[agent] - time.monotonic() > 2
        for payload in ({'type': 'robot_action', 'action': 'stand'},
                        {'type': 'set_vln', 'enabled': True, 'mode': 'track'}):
            await s._handle_browser_message(agent, 'agent', payload)
            assert not agent.messages[-1]['ok']
        assert s._commands.empty()
    asyncio.run(run())


def test_release_waits_for_robot_service_and_failure_keeps_barrier():
    s = server()
    events, futures = [], []
    class Service:
        def service_is_ready(self):
            return True
        def call_async(self, request):
            events.append(request.data)
            f = Future()
            futures.append(f)
            return f
    s.agent_release_finished = lambda cid, ok, message: s._finish_agent_release(cid, ok, message)
    node = SimpleNamespace(
        _mpc_request_generation=0, _mpc_desired_enabled=True,
        mpc_enable_pub=SimpleNamespace(publish=lambda msg: events.append(msg.data)),
        _request_vln=lambda *args: events.append(args),
        auto_control_client=Service(), web_server=s,
    )
    s._release_pending = True
    VlnWebNode._request_agent_release(node, {'automatic': True, 'manual': False}, 'agent')
    assert s._release_pending and not node._mpc_desired_enabled
    futures[0].set_result(SimpleNamespace(success=False, message='failed'))
    assert s._release_pending


def test_http_and_websocket_routes_work_without_hardware():
    async def run():
        s = server()
        s.update_odom(odom())
        frame(s)
        async with TestClient(TestServer(s._create_app())) as client:
            response = await client.get('/api/observation')
            assert response.status == 200
            assert (await response.json())['pose_valid']
            ws = await client.ws_connect('/agent/ws')
            assert (await ws.receive_json())['data']['agent']['protocol_version'] == 1
            assert (await ws.receive_json())['type'] == 'control_state'
            await ws.send_json({'type': 'agent_heartbeat'})
            assert (await ws.receive_json())['type'] == 'agent_heartbeat'
            await ws.close()
    asyncio.run(run())


def test_model_completion_uses_release_barrier_for_agent():
    async def run():
        s = server()
        owner = Socket()
        s._agent_deadlines[owner] = time.monotonic() + 3
        s._auto_controller = owner
        s._websockets[owner] = 'owner'
        s.reset_vln('go to chair', 'objnav')
        s.update_vln_response({'stop': True, 'episode': 1, 'last_sequence': 1}, {'body_waypoints': []}, time.monotonic(), capture_stamp_ns=time.time_ns())
        await s._complete_navigation(1)
        assert s._release_pending and s._auto_controller is None
        assert s._commands.get_nowait()[0] == 'agent_release'
        assert s._snapshot()['agent']['terminal']['reason'] == 'model_stop'
    asyncio.run(run())


def test_agent_must_release_manual_before_navigation():
    async def run():
        s = server()
        owner = Socket()
        s._agent_deadlines[owner] = time.monotonic() + 3
        s._controller = owner
        await s._handle_browser_message(owner, 'owner', {
            'type': 'set_vln', 'enabled': True, 'mode': 'objnav', 'instruction': 'go'})
        assert not owner.messages[-1]['ok'] and s._commands.empty()
    asyncio.run(run())


def test_dead_idle_connection_does_not_break_lease_watchdog():
    async def run():
        s = server()
        class Dead(Socket):
            async def send_json(self, value):
                raise ConnectionError('disconnected')
        first, second = Dead(), Socket()
        s._agent_deadlines = {first: time.monotonic()-1, second: time.monotonic()-1}
        s._auto_controller = second
        await s._expire_agent_leases()
        assert s._release_pending and second.closed
    asyncio.run(run())



def test_disconnect_cleanup_never_acknowledges_closed_connection():
    async def run():
        s = server()
        class Closed(Socket):
            closed = True
            async def send_json(self, value):
                raise AssertionError('must not send on a closed transport')
        socket = Closed()
        await s._release_agent(socket, 'owner', 'disconnected', notify=False)
        s._auto_controller = socket
        await s._release_agent(socket, 'owner', 'disconnected', notify=False)
        assert s._commands.get_nowait()[0] == 'agent_release'
        assert s._release_pending
    asyncio.run(run())


def test_body_limit_command_only_when_idle_and_valid():
    s=server(); ws=Socket()
    config={'linear_limit':.2,'angular_limit':.5}
    asyncio.run(s._handle_browser_message(ws,'viewer',{'type':'set_body_limits','config':config}))
    assert s._commands.get_nowait()==('set_body_limits',config,'viewer')
    s._controller=Socket()
    asyncio.run(s._handle_browser_message(ws,'viewer',{'type':'set_body_limits','config':config}))
    assert ws.messages[-1]['ok'] is False
    assert s._commands.empty()
    s._controller=None
    asyncio.run(s._handle_browser_message(ws,'viewer',{'type':'set_body_limits','config':{'linear_limit':True,'angular_limit':.5}}))
    assert ws.messages[-1]['ok'] is False
    assert s._commands.empty()


def test_duplicate_release_waits_for_original_sdk_confirmation():
    async def run():
        s = server()
        owner = Socket()
        s._auto_controller = owner
        s._websockets[owner] = 'owner'
        s._agent_deadlines[owner] = time.monotonic() + 3
        await s._release_agent(owner, 'owner', 'model_stop')
        assert s._release_pending
        assert s._commands.get_nowait()[0] == 'agent_release'
        for _ in range(2):
            await s._release_agent(owner, 'owner', 'agent_release')
        assert s._commands.empty()
        assert not any(item.get('type') == 'command_result' for item in owner.messages)
        replies = []
        s.send_client = lambda client_id, value: replies.append((client_id, value))
        s._finish_agent_release('owner', True, 'SDK released')
        assert not s._release_pending
        assert replies == [('owner', {'type': 'command_result', 'command': 'agent_release',
                                     'ok': True, 'message': 'SDK released'})]
    asyncio.run(run())


def test_navigation_rejects_previous_capture_episode_and_reordered_results():
    s = server()
    def result(episode, sequence, stamp, stop=True):
        return s.update_vln_response({'episode': episode, 'last_sequence': sequence, 'stop': stop},
                                     {}, time.monotonic(), capture_stamp_ns=stamp)
    s.reset_vln('go to chair', 'objnav', started_stamp_ns=100)
    assert not result(1, 10, 99)
    assert s._terminal is None
    assert result(2, 1, 101, False)
    assert not result(1, 11, 102)
    assert not result(2, 1, 102)
    assert s._terminal is None
    assert result(2, 2, 103)
    assert s._terminal['episode'] == 2
    s.reset_vln('go to table', 'objnav', started_stamp_ns=200)
    assert not result(2, 3, 199)
    assert s._terminal is None
    s.reset_vln('', 'objnav')
    assert not result(3, 1, 201)


def test_history_pause_keeps_sdk_release_barrier_and_rejects_stale_generation():
    async def run():
        s, owner = server(), Socket()
        s._history_capable = True
        s._auto_controller = owner
        s._websockets[owner] = 'owner'
        s._agent_deadlines[owner] = time.monotonic() + 3
        s.reset_vln('Go to the door.', 'objnav', started_stamp_ns=100)
        await s._handle_browser_message(owner, 'owner', {'type': 'pause_vln', 'generation': 0})
        assert s._commands.empty()
        assert not owner.messages[-1]['ok']
        await s._handle_browser_message(owner, 'owner', {'type': 'pause_vln', 'generation': 1})
        assert s._release_pending
        kind, payload, client_id = s._commands.get_nowait()
        assert kind == 'agent_release'
        assert payload == {'manual': False, 'automatic': True, 'preserve_history': True, 'generation': 1}
        assert s._history_owner is owner
        assert s._navigation_started_ns == 0
        assert s._terminal['reason'] == 'paused'
    asyncio.run(run())


def test_same_episode_stop_is_bound_to_new_action_generation():
    s = server()
    s._history_mode = True
    s.reset_vln('Go to the door.', 'objnav', started_stamp_ns=100)
    def result(generation, stamp, seq, stop):
        return s.update_vln_response({'generation': generation, 'episode': 4,
                                     'last_sequence': seq, 'stop': stop},
                                    {}, time.monotonic(), capture_stamp_ns=stamp)
    assert result(1, 101, 1, False)
    s.reset_vln('Go to the door.', 'objnav', started_stamp_ns=200)
    assert not result(1, 201, 2, True)
    assert s._terminal is None
    assert result(2, 202, 3, True)
    assert s._terminal['generation'] == 2
    assert s._terminal['reason'] == 'model_stop'


def test_queued_stop_cannot_release_next_navigation():
    async def run():
        s, owner = server(), Socket()
        s._auto_controller = owner
        s._websockets[owner] = 'owner'
        s._agent_deadlines[owner] = time.monotonic() + 3
        s.reset_vln('Go to the door.', 'objnav', started_stamp_ns=100)
        s.update_vln_response({'stop': True, 'episode': 1, 'last_sequence': 1},
                              {}, time.monotonic(), capture_stamp_ns=101)
        s.reset_vln('Go to the chair.', 'objnav', started_stamp_ns=200)
        await s._complete_navigation(1)
        assert s._auto_controller is owner
        assert not s._release_pending
        assert s._commands.empty()
    asyncio.run(run())


def test_web_restart_adopts_client_generation_without_rewinding_live_action():
    s = server()
    s.update_history_capability(True, 8)
    assert s.history_generation() == 8
    s.reset_vln('Go to the door.', 'objnav', started_stamp_ns=100)
    s.update_history_capability(True, 3)
    assert s.history_generation() == 9


def test_history_status_from_previous_action_does_not_fail_new_action():
    s = server()
    s._history_mode = True
    s.reset_vln('Go to the door.', 'objnav', started_stamp_ns=100)
    s.reset_vln('Go to the chair.', 'objnav', started_stamp_ns=200)
    s.update_vln_status('ERROR', generation=1)
    s.update_vln_status('ERROR')
    assert s._terminal is None
    s.update_vln_status('ERROR', generation=2)
    assert s._terminal == {'generation': 2, 'reason': 'navigation_error'}
    assert s._vln['generation'] == 2
