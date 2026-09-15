#!/usr/bin/env python3
"""Send one bounded velocity pulse to the running DimOS G1 simulation."""

import argparse
import time
from typing import Any, Callable

from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3


PULSE_HEARTBEAT_SECONDS = 0.1


def bounded_float(name: str, value: float, limit: float) -> float:
    if abs(value) > limit:
        raise ValueError(f"{name} must be between {-limit} and {limit}, got {value}")
    return value


def broadcast_velocity_pulse(
    transport: Any,
    command: Any,
    stop: Any,
    *,
    duration: float,
    heartbeat_seconds: float = PULSE_HEARTBEAT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Refresh a bounded command until its deadline, then fail closed to zero."""

    if duration <= 0.0 or heartbeat_seconds <= 0.0:
        raise ValueError("duration and heartbeat_seconds must be positive")
    deadline = clock() + duration
    try:
        while True:
            now = clock()
            if now >= deadline:
                break
            transport.broadcast(None, command)
            sleeper(min(heartbeat_seconds, max(0.0, deadline - clock())))
    finally:
        # Repeat the zero command to make the stop robust to a single UDP loss.
        for _ in range(3):
            transport.broadcast(None, stop)
            sleeper(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a short, bounded velocity command to the G1 MuJoCo simulation."
    )
    parser.add_argument("--x", type=float, default=0.0, help="forward velocity in m/s")
    parser.add_argument("--y", type=float, default=0.0, help="left velocity in m/s")
    parser.add_argument("--yaw", type=float, default=0.0, help="yaw velocity in rad/s")
    parser.add_argument("--duration", type=float, default=1.0, help="pulse duration in seconds")
    args = parser.parse_args()

    x = bounded_float("x", args.x, 0.5)
    y = bounded_float("y", args.y, 0.5)
    yaw = bounded_float("yaw", args.yaw, 1.0)
    if not 0.05 <= args.duration <= 5.0:
        raise ValueError(f"duration must be between 0.05 and 5.0, got {args.duration}")

    transport = LCMTransport("/cmd_vel", Twist)
    command = Twist(
        linear=Vector3(x, y, 0.0),
        angular=Vector3(0.0, 0.0, yaw),
    )
    stop = Twist(
        linear=Vector3(0.0, 0.0, 0.0),
        angular=Vector3(0.0, 0.0, 0.0),
    )

    try:
        print(f"sent velocity x={x} y={y} yaw={yaw} for {args.duration}s")
        broadcast_velocity_pulse(
            transport,
            command,
            stop,
            duration=args.duration,
        )
    finally:
        print("sent stop")


if __name__ == "__main__":
    main()
