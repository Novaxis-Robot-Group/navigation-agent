import math

from vln_mpc.geometry import (
    TimedPose,
    pose_at_stamp,
    project_body_to_odom,
    project_local_to_odom,
    project_odom_to_local,
)


def test_pose_is_interpolated_at_image_stamp_with_wrapped_yaw():
    history = [
        TimedPose(1_000_000_000, 0.0, 2.0, math.pi - 0.1),
        TimedPose(1_200_000_000, 2.0, 4.0, -math.pi + 0.1),
    ]
    matched = pose_at_stamp(history, 1_100_000_000, 0.3)
    assert matched is not None
    pose, gap_s = matched
    assert abs(pose.x - 1.0) < 1e-9
    assert abs(pose.y - 3.0) < 1e-9
    assert abs(abs(pose.yaw) - math.pi) < 1e-6
    assert abs(gap_s - 0.1) < 1e-9


def test_pose_match_rejects_distant_odom():
    history = [TimedPose(1_000_000_000, 0.0, 0.0, 0.0)]
    assert pose_at_stamp(history, 1_500_000_000, 0.3) is None


def test_pose_match_does_not_interpolate_across_large_odom_gap():
    history = [
        TimedPose(1_000_000_000, 1.0, 2.0, 0.1),
        TimedPose(3_000_000_000, 9.0, 8.0, 0.9),
    ]
    matched = pose_at_stamp(history, 1_100_000_000, 0.3)
    assert matched == (history[0], 0.1)


def test_body_waypoint_is_projected_into_capture_odom_frame():
    pose = TimedPose(0, 2.0, 3.0, math.pi / 2)
    path = project_body_to_odom([(1.0, 0.0, 0.2)], pose)
    assert abs(path[0][0] - 2.0) < 1e-9
    assert abs(path[0][1] - 4.0) < 1e-9
    assert abs(path[0][2] - (math.pi / 2 + 0.2)) < 1e-9


def test_odom_trajectory_round_trips_through_current_body_frame():
    origin = TimedPose(0, 2.0, 3.0, math.pi / 2)
    odom = [(2.0, 4.0, math.pi / 2 + 0.2)]
    local = project_odom_to_local(odom, origin)
    assert abs(local[0][0] - 1.0) < 1e-9
    assert abs(local[0][1]) < 1e-9
    assert abs(local[0][2] - 0.2) < 1e-9
    restored = project_local_to_odom(local, origin)
    for actual, expected in zip(restored[0], odom[0]):
        assert abs(actual - expected) < 1e-9
