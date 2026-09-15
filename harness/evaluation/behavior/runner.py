"""Run an isolated composition experiment, never an official submission."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import time

from harness.runtime.session_store import LuxiSessionStore
from .development import DevelopmentTask, PROVENANCE, SYSTEM_PROMPT, build_development_runtime


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("oracle-dev", "challenge"), default="oracle-dev")
    parser.add_argument("--backend", choices=("fixture", "omnigibson"), default="fixture")
    parser.add_argument("--provider", choices=("scripted-fixture", "compatible"), default="scripted-fixture")
    parser.add_argument("--task", type=Path, default=Path(__file__).resolve().parents[3] / "config/behavior/cupboard-dev-task.json")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/behavior-development"))
    parser.add_argument("--model", help="Model name for the existing compatible provider")
    parser.add_argument("--fail-first-pick", action="store_true")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--max-tools", type=int, default=64)
    parser.add_argument("--max-action-steps", type=int, default=2000)
    parser.add_argument("--max-sim-steps", type=int, default=20000)
    args = parser.parse_args(argv)
    if args.mode != "oracle-dev":
        parser.error("this runner uses privileged development skills; challenge execution is unavailable")
    if args.doctor:
        installed = importlib.util.find_spec("omnigibson") is not None
        print(json.dumps({**PROVENANCE, "omnigibson_installed": installed,
                          "fixture_available": True, "expected_upstream_tag": "v3.9.2",
                          "live_backend_validated": False}, indent=2))
        return 0 if args.backend == "fixture" or installed else 2
    if min(args.max_steps, args.max_tools, args.max_action_steps, args.max_sim_steps) < 1:
        parser.error("budgets must be positive")
    task = DevelopmentTask.parse(json.loads(args.task.read_text()))
    if args.provider == "scripted-fixture" and args.backend != "fixture":
        parser.error("scripted-fixture provider is only for the non-physical fixture backend")
    if args.fail_first_pick and args.backend != "fixture":
        parser.error("fault injection is supported only by the fixture backend")
    if args.provider == "scripted-fixture":
        from .fixtures import ScriptedFixtureProvider
        model = ScriptedFixtureProvider()
    else:
        if not args.model or not os.environ.get("BEHAVIOR_MODEL_API_KEY"):
            parser.error("compatible provider requires --model and BEHAVIOR_MODEL_API_KEY")
        from openai import OpenAI
        from harness.runtime.providers import OpenAICompatibleModelProvider
        client = OpenAI(api_key=os.environ["BEHAVIOR_MODEL_API_KEY"],
                        base_url=os.environ.get("BEHAVIOR_MODEL_BASE_URL"), max_retries=0, timeout=50.0)
        model = OpenAICompatibleModelProvider(client, model=args.model, system_prompt=SYSTEM_PROMPT)
    env = None
    if args.backend == "fixture":
        from .fixtures import FixtureBackend
        backend = FixtureBackend(fail_first_pick=args.fail_first_pick)
    else:
        if not args.env_config:
            parser.error("omnigibson requires an existing compatible --env-config and --task with exact scene references")
        if importlib.util.find_spec("omnigibson") is None:
            parser.error("OmniGibson is not installed in this interpreter; run --doctor")
        import omnigibson as og
        from .omnigibson_backend import OmniGibsonDevelopmentBackend
        config = json.loads(args.env_config.read_text())
        config.setdefault("env", {})["automatic_reset"] = False
        env = og.Environment(configs=config)
        try:
            env.reset()
            backend = OmniGibsonDevelopmentBackend(env, mode=args.mode, max_sim_steps=args.max_sim_steps)
        except BaseException:
            env.close()
            raise
    try:
        run_dir = args.output / f"{time.time_ns()}"
        run_dir.mkdir(parents=True, exist_ok=False)
        store = LuxiSessionStore(run_dir / "events.sqlite3")
        loop, scope, skills = build_development_runtime(
            backend=backend, task=task, model=model, events=store, mode=args.mode,
            max_steps=args.max_steps, max_tools=args.max_tools,
            max_action_steps=args.max_action_steps, max_episode_steps=args.max_sim_steps,
        )
        loop.safety.stop(backend.robot_id, "development_startup")
        result = loop.run(task.instruction, scope=scope)
        report = {**PROVENANCE, "backend_kind": backend.kind,
                  "provider": args.provider,
                  "model_planning_tested": args.provider != "scripted-fixture" and result.planning_steps > 0,
                  "physical_execution_tested": args.backend == "omnigibson" and skills.action_steps > 0,
                  "result": asdict(result), "subgoals": {key: [asdict(c) for c in values]
                   for key, values in skills.subgoals.items()}, "events": str(run_dir / "events.sqlite3")}
        # Fixture results are explicitly non-physical; do not reuse loop prose as evidence.
        if args.backend == "fixture":
            report["result"]["response"] = "Contract fixture result only; no robot physics validated."
        (run_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({"report": str(run_dir / "result.json"), "completed": result.completed,
                          "task_status": result.task_status, "planning_steps": result.planning_steps,
                          "tool_calls": result.tool_calls, **PROVENANCE}, ensure_ascii=False, indent=2))
        return 0 if result.completed else 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
