"""Odom interpolation and planar frame transformations."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TimedPose:
    stamp_ns: int
    x: float
    y: float
    yaw: float


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_at_stamp(
    history: Sequence[TimedPose], stamp_ns: int, max_gap_s: float
) -> tuple[TimedPose, float] | None:
    """Interpolate an odom pose at a sensor timestamp."""
    if not history or max_gap_s < 0.0:
        return None
    stamps = [sample.stamp_ns for sample in history]
    index = bisect.bisect_left(stamps, stamp_ns)
    if index <= 0:
        nearest = history[0]
        gap_s = abs(nearest.stamp_ns - stamp_ns) / 1e9
        return (nearest, gap_s) if gap_s <= max_gap_s else None
    if index >= len(history):
        nearest = history[-1]
        gap_s = abs(nearest.stamp_ns - stamp_ns) / 1e9
        return (nearest, gap_s) if gap_s <= max_gap_s else None

    left = history[index - 1]
    right = history[index]
    left_gap_s = (stamp_ns - left.stamp_ns) / 1e9
    right_gap_s = (right.stamp_ns - stamp_ns) / 1e9
    gap_s = min(left_gap_s, right_gap_s)
    if left_gap_s > max_gap_s and right_gap_s > max_gap_s:
        return None
    if right_gap_s > max_gap_s:
        return left, left_gap_s
    if left_gap_s > max_gap_s:
        return right, right_gap_s
    span_ns = right.stamp_ns - left.stamp_ns
    if span_ns <= 0:
        return left, gap_s
    ratio = (stamp_ns - left.stamp_ns) / span_ns
    yaw = left.yaw + ratio * wrap_angle(right.yaw - left.yaw)
    return (
        TimedPose(
            stamp_ns=stamp_ns,
            x=left.x + ratio * (right.x - left.x),
            y=left.y + ratio * (right.y - left.y),
            yaw=wrap_angle(yaw),
        ),
        gap_s,
    )


def project_body_to_odom(
    waypoints: Sequence[tuple[float, float, float]], capture_pose: TimedPose
) -> list[tuple[float, float, float]]:
    """Project x-forward/y-left/yaw-CCW waypoints into odom."""
    cosine = math.cos(capture_pose.yaw)
    sine = math.sin(capture_pose.yaw)
    return [
        (
            capture_pose.x + cosine * forward - sine * lateral,
            capture_pose.y + sine * forward + cosine * lateral,
            wrap_angle(capture_pose.yaw + yaw),
        )
        for forward, lateral, yaw in waypoints
    ]


def project_odom_to_local(
    trajectory: Sequence[tuple[float, float, float]],
    origin: TimedPose,
) -> list[tuple[float, float, float]]:
    """Express an odom trajectory in a fixed frame at the current body pose."""
    cosine = math.cos(origin.yaw)
    sine = math.sin(origin.yaw)
    return [
        (
            cosine * (x - origin.x) + sine * (y - origin.y),
            -sine * (x - origin.x) + cosine * (y - origin.y),
            yaw - origin.yaw,
        )
        for x, y, yaw in trajectory
    ]


def project_local_to_odom(
    trajectory: Sequence[tuple[float, float, float]],
    origin: TimedPose,
) -> list[tuple[float, float, float]]:
    """Transform a trajectory from a fixed current-body frame into odom."""
    cosine = math.cos(origin.yaw)
    sine = math.sin(origin.yaw)
    return [
        (
            origin.x + cosine * x - sine * y,
            origin.y + sine * x + cosine * y,
            origin.yaw + yaw,
        )
        for x, y, yaw in trajectory
    ]
