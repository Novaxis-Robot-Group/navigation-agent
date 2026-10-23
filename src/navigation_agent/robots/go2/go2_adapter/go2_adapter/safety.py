"""Pure command-selection helpers for the Go2 adapter."""

from __future__ import annotations

import math
from typing import Optional

Twist2D = tuple[float, float]


def select_command(
    *,
    source: str,
    robot_mode: str,
    manual_command: Twist2D,
    manual_age_s: float,
    auto_command: Twist2D,
    auto_age_s: float,
    watchdog_s: float,
    linear_limit: float = 0.3,
    angular_limit: float = 0.6,
) -> Optional[Twist2D]:
    """Return the active finite command when the Go2 is ready to walk."""
    if robot_mode not in {"STAND", "WALK"}:
        return None
    if source == "manual" and manual_age_s <= watchdog_s:
        command = manual_command
    elif source == "auto" and auto_age_s <= watchdog_s:
        command = auto_command
    else:
        return None
    if not all(math.isfinite(value) for value in command):
        return None
    if not all(math.isfinite(value) and value > 0 for value in (linear_limit, angular_limit)):
        return None
    return (max(-linear_limit, min(linear_limit, command[0])),
            max(-angular_limit, min(angular_limit, command[1])))
