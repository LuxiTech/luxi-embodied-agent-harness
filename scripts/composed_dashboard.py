#!/usr/bin/env python3
"""Open the existing Dashboard with an explicit MuJoCo candidate task catalog."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.runtime.task_state import ComposedTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, action="append", default=[])
    parser.add_argument("--locations", type=Path, help="Trusted name -> [x,y,yaw] references; never scene truth")
    args, ui_args = parser.parse_known_args()
    from harness.runtime.composition_goals import pose_value
    locations = {k: pose_value(v) for k, v in json.loads(args.locations.read_text()).items()} if args.locations else {}
    tasks = {}
    for path in args.task:
        task = ComposedTask.from_payload(json.loads(path.read_text()))
        if task.release or task.entity_id not in (None, "water_bottle"):
            parser.error("候选只支持位置任务和 water_bottle 附着运输；live 释放尚未验收")
        if task.task_key in tasks:
            parser.error("task_key 必须唯一")
        tasks[task.task_key] = task
    os.environ["LUXI_COMPOSED_DEV"] = "1"
    os.environ["LUXI_SIM_ATTACHMENT_DEV"] = "1"
    from harness.app.server import main as dashboard_main
    from harness.runtime.composition import create_agent_runtime_service
    from harness.runtime.composed_backend import mujoco_candidate_backend

    def compose(*a, **kw):
        if kw.get("backend") != "mujoco":
            raise ValueError("组合候选 Dashboard 当前仅支持 MuJoCo G1")
        def backend(skills):
            value = mujoco_candidate_backend(skills)
            value.references = locations
            value.location_catalog_path = ROOT / "config/composed/home_complex-locations.json"
            return value
        return create_agent_runtime_service(*a, **kw, composed_backend=backend,
                                            composed_tasks=tasks)

    with tempfile.TemporaryDirectory(prefix="luxi-composed-") as directory:
        os.environ["LUXI_COMPOSED_SHM_MANIFEST"] = str(Path(directory)/"shm.json")
        with patch("harness.runtime.composition.create_agent_runtime_service", side_effect=compose):
            return dashboard_main(ui_args)


if __name__ == "__main__":
    raise SystemExit(main())
