"""Pose-aligned unicycle MPC."""

from __future__ import annotations

import math

import casadi as ca
import numpy as np


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def build_pose_aligned_reference(
    trajectory: np.ndarray,
    robot_pose: np.ndarray,
    *,
    horizon: int,
    weights: tuple[float, float, float],
) -> np.ndarray:
    """Select the waypoints following the nearest weighted pose."""
    points = np.asarray(trajectory, dtype=np.float64).copy()
    pose = np.asarray(robot_pose, dtype=np.float64)
    cost_weights = np.asarray(weights, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("trajectory must be a non-empty Nx3 array")
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError("robot_pose must be a finite 3-vector")
    if not np.all(np.isfinite(points)):
        raise ValueError("trajectory must contain only finite values")
    if (
        horizon <= 0
        or cost_weights.shape != (3,)
        or not np.all(np.isfinite(cost_weights))
        or np.any(cost_weights <= 0.0)
    ):
        raise ValueError("horizon and pose weights must be positive")

    errors = points - pose
    errors[:, 2] = [_wrap_angle(value) for value in errors[:, 2]]
    nearest_index = int(np.argmin(np.sum(errors * errors * cost_weights, axis=1)))
    indices = np.minimum(
        nearest_index + 1 + np.arange(horizon, dtype=np.int64),
        len(points) - 1,
    )
    reference = points[indices].copy()
    previous_yaw = float(pose[2])
    for waypoint in reference:
        waypoint[2] = previous_yaw + _wrap_angle(waypoint[2] - previous_yaw)
        previous_yaw = float(waypoint[2])
    return reference


def _rollout_unicycle(
    pose: np.ndarray,
    controls: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    states = np.empty((len(controls) + 1, 3), dtype=np.float64)
    states[0] = np.asarray(pose, dtype=np.float64)
    for index, (linear_velocity, angular_velocity) in enumerate(controls):
        x, y, yaw = states[index]
        states[index + 1] = (
            x + dt_s * linear_velocity * math.cos(yaw),
            y + dt_s * linear_velocity * math.sin(yaw),
            yaw + dt_s * angular_velocity,
        )
    return states


def _reference_velocity_guess(
    pose: np.ndarray,
    reference: np.ndarray,
    dt_s: float,
    v_max: float,
    w_max: float,
) -> np.ndarray:
    points = np.vstack((np.asarray(pose, dtype=np.float64), reference.copy()))
    previous_yaw = float(points[0, 2])
    for point in points[1:]:
        point[2] = previous_yaw + _wrap_angle(point[2] - previous_yaw)
        previous_yaw = float(point[2])
    deltas = np.diff(points, axis=0)
    return np.column_stack(
        (
            np.clip(np.hypot(deltas[:, 0], deltas[:, 1]) / dt_s, 0.0, v_max),
            np.clip(deltas[:, 2] / dt_s, -w_max, w_max),
        )
    )


class MPCController:
    """CasADi/IPOPT trajectory tracker for a velocity-driven unicycle."""

    def __init__(
        self,
        *,
        horizon: int,
        dt_s: float,
        w_max: float,
        a_max_v: float,
        a_max_w: float,
        q_weights: tuple[float, float, float],
        r_weights: tuple[float, float],
    ) -> None:
        self.horizon = int(horizon)
        self.dt_s = float(dt_s)
        self.w_max = float(w_max)
        self.q_weights = tuple(float(value) for value in q_weights)
        self.r_weights = tuple(float(value) for value in r_weights)
        if (
            self.horizon <= 0
            or self.dt_s <= 0.0
            or min(self.w_max, a_max_v, a_max_w) <= 0.0
        ):
            raise ValueError("invalid MPC horizon, timing, or limit")

        opti = ca.Opti()
        controls = opti.variable(self.horizon, 2)
        states = opti.variable(self.horizon + 1, 3)
        initial_state = opti.parameter(3)
        reference = opti.parameter(self.horizon, 3)
        previous_control = opti.parameter(2)
        v_max = opti.parameter()

        opti.subject_to(states[0, :] == initial_state.T)
        for index in range(self.horizon):
            yaw = states[index, 2]
            derivative = ca.horzcat(
                controls[index, 0] * ca.cos(yaw),
                controls[index, 0] * ca.sin(yaw),
                controls[index, 1],
            )
            opti.subject_to(
                states[index + 1, :] == states[index, :] + self.dt_s * derivative
            )

        q_matrix = ca.diag(ca.DM(self.q_weights))
        r_matrix = ca.diag(ca.DM(self.r_weights))
        objective = 0
        for index in range(self.horizon):
            error = states[index + 1, :] - reference[index, :]
            objective += ca.mtimes([error, q_matrix, error.T])
            objective += ca.mtimes([controls[index, :], r_matrix, controls[index, :].T])
        opti.minimize(objective)
        opti.subject_to(controls[:, 0] >= 0.0)
        opti.subject_to(controls[:, 0] <= v_max)
        opti.subject_to(opti.bounded(-self.w_max, controls[:, 1], self.w_max))

        dv_max = float(a_max_v) * self.dt_s
        dw_max = float(a_max_w) * self.dt_s
        opti.subject_to(
            opti.bounded(-dv_max, controls[0, 0] - previous_control[0], dv_max)
        )
        opti.subject_to(
            opti.bounded(-dw_max, controls[0, 1] - previous_control[1], dw_max)
        )
        for index in range(self.horizon - 1):
            opti.subject_to(
                opti.bounded(
                    -dv_max,
                    controls[index + 1, 0] - controls[index, 0],
                    dv_max,
                )
            )
            opti.subject_to(
                opti.bounded(
                    -dw_max,
                    controls[index + 1, 1] - controls[index, 1],
                    dw_max,
                )
            )

        opti.solver(
            "ipopt",
            {
                "ipopt.max_iter": 100,
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.acceptable_tol": 1e-8,
                "ipopt.acceptable_obj_change_tol": 1e-6,
            },
        )
        self.opti = opti
        self.controls = controls
        self.states = states
        self.initial_state = initial_state
        self.reference = reference
        self.previous_control = previous_control
        self.v_max = v_max

    def solve(
        self,
        pose: np.ndarray,
        reference: np.ndarray,
        previous: tuple[float, float],
        v_max: float,
    ) -> tuple[tuple[float, float], np.ndarray]:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.shape != (self.horizon, 3):
            raise ValueError(f"reference must have shape ({self.horizon}, 3)")
        if not math.isfinite(v_max) or v_max <= 0.0:
            raise ValueError("v_max must be positive")
        self.opti.set_value(self.initial_state, np.asarray(pose, dtype=np.float64))
        self.opti.set_value(self.reference, reference)
        self.opti.set_value(self.previous_control, previous)
        self.opti.set_value(self.v_max, float(v_max))
        controls_guess = _reference_velocity_guess(
            pose,
            reference,
            self.dt_s,
            v_max,
            self.w_max,
        )
        self.opti.set_initial(self.controls, controls_guess)
        self.opti.set_initial(
            self.states,
            _rollout_unicycle(pose, controls_guess, self.dt_s),
        )
        solution = self.opti.solve()
        solved_controls = np.asarray(solution.value(self.controls), dtype=np.float64)
        solved_states = np.asarray(solution.value(self.states), dtype=np.float64)
        command = (
            float(np.clip(solved_controls[0, 0], 0.0, v_max)),
            float(np.clip(solved_controls[0, 1], -self.w_max, self.w_max)),
        )
        return command, solved_states.copy()
