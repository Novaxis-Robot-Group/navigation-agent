import threading
from types import SimpleNamespace
from rclpy.parameter import Parameter
from go2_adapter.adapter_node import Go2AdapterNode


def test_runtime_limits_update_both_values_and_reject_while_moving():
    node=SimpleNamespace(linear_limit=.3,angular_limit=.6,_lock=threading.Lock(),_source='disabled',_zero_frames_remaining=0)
    params=[Parameter('linear_limit',value=.2),Parameter('angular_limit',value=.5)]
    assert Go2AdapterNode._set_motion_limits(node,params).successful
    assert (node.linear_limit,node.angular_limit)==(.2,.5)
    node._source='auto'
    assert not Go2AdapterNode._set_motion_limits(node,[Parameter('linear_limit',value=.4)]).successful
    assert node.linear_limit==.2
    node._source='disabled';node._zero_frames_remaining=1
    assert not Go2AdapterNode._set_motion_limits(node,params).successful


def test_invalid_limits_never_partially_apply():
    node=SimpleNamespace(linear_limit=.3,angular_limit=.6,_lock=threading.Lock(),_source='disabled',_zero_frames_remaining=0)
    for bad in (float('nan'),0.,-1.,4.,True):
        params=[Parameter('linear_limit',value=.2),Parameter('angular_limit',value=bad)]
        assert not Go2AdapterNode._set_motion_limits(node,params).successful
        assert (node.linear_limit,node.angular_limit)==(.3,.6)


def test_disable_service_does_not_acknowledge_queued_release():
    from concurrent.futures import Future
    for failed in (False, True):
        completion = Future()
        entered = threading.Event()
        def disable(source):
            assert source == 'disabled'
            entered.set()
            return completion
        node = SimpleNamespace(_set_source=disable, sdk_timeout_s=.1)
        response = SimpleNamespace(success=None, message='')
        worker = threading.Thread(target=Go2AdapterNode._set_control_enabled,
                                  args=(node, 'auto', SimpleNamespace(data=False), response))
        worker.start()
        assert entered.wait(1)
        assert response.success is None
        if failed:
            completion.set_exception(RuntimeError('SDK release failed'))
        else:
            completion.set_result(None)
        worker.join(1)
        assert not worker.is_alive()
        assert response.success is not failed
        if failed:
            assert 'SDK release failed' in response.message


def test_state_frames_cannot_erase_sdk_errors():
    import queue
    node = SimpleNamespace(_incoming=queue.Queue(), _lock=threading.Lock(),
                           _last_sdk_error='Move failed', _publish_odom=lambda state: None,
                           _publish_sent_twist=lambda command: None)
    node._incoming.put(('state', object()))
    Go2AdapterNode._process_incoming(node)
    assert node._last_sdk_error == 'Move failed'
    node._incoming.put(('sent', (0., 0., 0.)))
    Go2AdapterNode._process_incoming(node)
    assert node._last_sdk_error == ''


def test_enable_is_blocked_until_release_completes_successfully():
    from concurrent.futures import Future
    import pytest
    completion = Future()
    node = SimpleNamespace(_lock=threading.Lock(), _release_future=completion,
                           _source='disabled', _zero_frames_remaining=0, final_zero_frames=3,
                           _publish_source=lambda: None, _publish_diagnostics=lambda: None)
    with pytest.raises(RuntimeError, match='pending'):
        Go2AdapterNode._set_source(node, 'auto')
    assert node._source == 'disabled'
    completion.set_exception(RuntimeError('SDK failed'))
    with pytest.raises(RuntimeError, match='SDK failed'):
        Go2AdapterNode._set_source(node, 'auto')
    assert node._source == 'disabled'
    node._release_future = Future()
    node._release_future.set_result(None)
    Go2AdapterNode._set_source(node, 'auto')
    assert node._source == 'auto'
