#!/usr/bin/env python3
"""Start the DimOS CLI with Luxi's local MuJoCo worker wrapper selected."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

from dimos.robot.unitree import mujoco_connection
from dimos.robot.cli.dimos import cli_main

from harness.integrations.dimos.navigation_compat import install_navigation_compat
from harness.integrations.dimos.local_blueprints import register_luxi_blueprints
from harness.integrations.mcp.mcp_client_compat import install_mcp_client_timeout_compat
from harness.integrations.dimos.object_navigation_compat import install_object_navigation_compat
from harness.integrations.qwen.qwen_vl_compat import install_qwen_vl_compat
from harness.integrations.dimos.spatial_memory_compat import install_spatial_memory_location_compat
from harness.evaluation.blind_evaluation import reject_forbidden_blind_cli_args


PROJECT_ROOT = Path(__file__).resolve().parent.parent
mujoco_connection.LAUNCHER_PATH = (
    PROJECT_ROOT / "harness/robots/g1/mujoco/mujoco_observer_launcher.py"
)
install_navigation_compat()
install_mcp_client_timeout_compat()
install_object_navigation_compat()
install_qwen_vl_compat()
install_spatial_memory_location_compat()

# Register import paths only. The CLI applies its global options before it
# resolves and imports the selected blueprint, so --viewer/--rerun-open are
# honored by the nested upstream blueprint factories.
register_luxi_blueprints()


def _argv_with_operator_scene_start(
    argv: Sequence[str],
    environment: Mapping[str, str],
) -> list[str]:
    """Make upstream's initial x/y agree with the selected scene payload.

    The worker wrapper also applies the complete x/y/yaw pose, but pinned
    DimOS initializes its controller from ``mujoco_start_pos`` before entering
    that wrapper. Feeding the same x/y through the native option prevents the
    controller and observer from starting with different world origins.
    """

    from harness.robots.g1.mujoco.operator_scenes import (
        OPERATOR_SCENE_PAYLOAD_ENV,
        load_operator_scene_payload,
    )

    result = list(argv)
    payload_value = environment.get(OPERATOR_SCENE_PAYLOAD_ENV, "").strip()
    if not payload_value:
        return result
    if "--mujoco-start-pos" in result:
        raise ValueError(
            "selected operator scene conflicts with explicit --mujoco-start-pos"
        )
    try:
        run_index = result.index("run")
    except ValueError:
        return result
    payload = load_operator_scene_payload(Path(payload_value))
    start = payload.get("robot_start")
    if start is None:
        return result
    x, y, _yaw = (float(value) for value in start)
    result[run_index:run_index] = ["--mujoco-start-pos", f"{x},{y}"]
    return result


if __name__ == "__main__":
    sys.argv[:] = _argv_with_operator_scene_start(sys.argv, os.environ)
    reject_forbidden_blind_cli_args(sys.argv, os.environ)
    raise SystemExit(cli_main())
