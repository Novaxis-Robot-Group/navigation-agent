import math

import numpy as np
from vln_mpc.mpc import (
    MPCController,
    _reference_velocity_guess,
    _rollout_unicycle,
    build_pose_aligned_reference,
)


def test_reference_starts_after_nearest_weighted_pose_and_unwraps_yaw():
    trajectory = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, -math.pi + 0.1],
            [2.0, 0.0, -math.pi + 0.2],
        ]
    )
    reference = build_pose_aligned_reference(
        trajectory,
        robot_pose=np.asarray([1.1, 0.0, math.pi - 0.1]),
        horizon=3,
        weights=(10.0, 10.0, 1.0),
    )
    np.testing.assert_allclose(
        reference[:, :2],
        [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
    )
    np.testing.assert_allclose(
        reference[:, 2],
        [math.pi + 0.2, math.pi + 0.2, math.pi + 0.2],
    )


def test_reference_weights_yaw_when_selecting_nearest_index():
    trajectory = np.asarray([[1.0, 0.0, -1.0], [1.0, 0.0, 0.2]])
    reference = build_pose_aligned_reference(
        trajectory,
        robot_pose=np.zeros(3),
        horizon=2,
        weights=(10.0, 10.0, 1.0),
    )
    np.testing.assert_allclose(reference, np.tile(trajectory[-1:], (2, 1)))


def test_reference_velocity_is_used_for_rollout_initialization():
    pose = np.zeros(3)
    reference = np.asarray([[0.1, 0.0, 0.1], [0.2, 0.0, 0.2], [0.3, 0.0, 0.3]])
    controls = _reference_velocity_guess(
        pose,
        reference,
        dt_s=0.1,
        v_max=2.0,
        w_max=2.0,
    )

    np.testing.assert_allclose(controls, np.ones((3, 2)))
    states = _rollout_unicycle(pose, controls, 0.1)
    np.testing.assert_allclose(states[1], [0.1, 0.0, 0.1])


def test_unified_tracker_can_rotate_without_translation():
    controller = MPCController(
        horizon=5,
        dt_s=0.1,
        w_max=1.0,
        a_max_v=1.0,
        a_max_w=2.0,
        q_weights=(10.0, 10.0, 2.0),
        r_weights=(0.05, 0.2),
    )
    reference = np.column_stack(
        (
            np.zeros(5),
            np.zeros(5),
            np.linspace(-0.1, -0.5, 5),
        )
    )

    command, _ = controller.solve(np.zeros(3), reference, (0.0, 0.0), v_max=1.0)

    assert abs(command[0]) < 1e-4
    assert command[1] < -0.1
