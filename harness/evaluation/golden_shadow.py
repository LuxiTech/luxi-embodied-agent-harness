"""Collect real-Qwen, zero-actuation Golden Case decision traces."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable, Mapping

from harness.integrations.qwen.config import (
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    _default_client_factory,
)
from harness.runtime.tool_catalog import (
    G1_AGENT_CONTRACT,
    ISAAC_AGENT_CONTRACT,
    _tools_for_backend,
)
from harness.runtime.task_policy import (
    _forced_tool_for_instruction,
)
from harness.runtime.composition import descriptor_from_tool_spec


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    backend: str
    instruction: str
    observation: Mapping[str, Any]


CASES = (
    GoldenCase(
        "observe_environment",
        "mujoco",
        "请使用视觉分析检查当前第一人称画面并描述主要物体，不要移动机器人。",
        {
            "backend": "mujoco",
            "camera_available": True,
            "odometry_available": True,
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "metrics": {"risk": "normal", "forward_open_distance_m": 2.4},
        },
    ),
    GoldenCase(
        "status_only",
        "mujoco",
        "检查 DimOS 和机器人当前状态，只报告结果，不要移动。",
        {
            "backend": "mujoco",
            "camera_available": True,
            "odometry_available": True,
            "metrics": {"risk": "normal"},
            "simulation": {"process_alive": True, "mcp": "ready"},
        },
    ),
    GoldenCase(
        "explicit_distance",
        "isaac-g1",
        "向前走0.6米",
        {
            "backend": "isaac-g1",
            "camera_available": True,
            "odometry_available": True,
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "metrics": {"risk": "normal", "forward_open_distance_m": 2.0},
        },
    ),
    GoldenCase(
        "person_follow",
        "isaac-g1",
        "跟随前方的人物10秒，然后停车。",
        {
            "backend": "isaac-g1",
            "camera_available": True,
            "odometry_available": True,
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "metrics": {"risk": "normal", "forward_open_distance_m": 3.0},
        },
    ),
    GoldenCase(
        "visual_search",
        "isaac-g1",
        "寻找红色杯子并接近它，完成后停车。",
        {
            "backend": "isaac-g1",
            "camera_available": True,
            "odometry_available": True,
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "metrics": {"risk": "normal", "forward_open_distance_m": 2.5},
        },
    ),
)


def _contract(backend: str) -> str:
    return ISAAC_AGENT_CONTRACT if backend == "isaac-g1" else G1_AGENT_CONTRACT


def _payload(case: GoldenCase) -> dict[str, Any]:
    return {
        "instruction": case.instruction,
        "initial_observation": dict(case.observation),
        "note": "运行时已自动完成本轮首次 observe_environment。",
    }


def _messages(case: GoldenCase) -> list[dict[str, Any]]:
    encoded = json.dumps(
        _payload(case),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"role": "system", "content": _contract(case.backend)},
        {"role": "user", "content": encoded},
    ]


def _unified_tools(case: GoldenCase) -> list[Mapping[str, Any]]:
    return [
        descriptor_from_tool_spec(spec, backend=case.backend).model_tool()
        for spec in _tools_for_backend(case.backend)
    ]


def _tool_choice(case: GoldenCase, tools: Iterable[Mapping[str, Any]]) -> Any:
    allowed = {
        str(tool.get("function", {}).get("name", ""))
        for tool in tools
        if isinstance(tool.get("function"), Mapping)
    }
    forced = _forced_tool_for_instruction(case.instruction)
    if forced not in allowed:
        return "auto"
    return {"type": "function", "function": {"name": forced}}


def _decision(message: Any, case: GoldenCase) -> dict[str, Any]:
    calls = list(getattr(message, "tool_calls", None) or [])
    decisions: list[dict[str, Any]] = []
    valid = True
    for call in calls:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"_invalid_json": str(call.function.arguments)}
            valid = False
        if not isinstance(arguments, dict):
            valid = False
        decisions.append({"name": str(call.function.name), "arguments": arguments})
    content = str(getattr(message, "content", "") or "")
    signature_value: Any = decisions if decisions else [{"name": "__answer__", "arguments": {}}]
    signature = json.dumps(
        signature_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    semantic_value = json.loads(json.dumps(signature_value, ensure_ascii=False))
    if case.case_id == "observe_environment":
        for decision in semantic_value:
            if decision.get("name") == "analyze_scene":
                decision.get("arguments", {}).pop("question", None)
    semantic_signature = json.dumps(
        semantic_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "decisions": decisions,
        "signature": signature,
        "semantic_signature": semantic_signature,
        "arguments_valid": valid,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content_excerpt": content[:300],
    }


def _complete(client: Any, case: GoldenCase, *, unified: bool, model: str) -> dict[str, Any]:
    tools = _unified_tools(case) if unified else _tools_for_backend(case.backend)
    tool_choice = _tool_choice(case, tools)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _messages(case),
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": 0.1,
    }
    if isinstance(tool_choice, Mapping) and model.casefold().startswith("qwen3"):
        kwargs["extra_body"] = {"enable_thinking": False}
    started = time.monotonic()
    completion = client.chat.completions.create(**kwargs)
    elapsed = time.monotonic() - started
    result = _decision(completion.choices[0].message, case)
    usage = getattr(completion, "usage", None)
    result.update(
        {
            "elapsed_s": round(elapsed, 3),
            "finish_reason": str(completion.choices[0].finish_reason),
            "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else {},
            "physical_execution": False,
        }
    )
    return result


def _dominant_rate(signatures: list[str]) -> float:
    if not signatures:
        return 0.0
    return max(signatures.count(value) for value in set(signatures)) / len(signatures)


def collect(
    *,
    api_key: str,
    model: str,
    base_url: str,
    repetitions: int,
    output_root: Path,
    cases: Iterable[GoldenCase] = CASES,
) -> tuple[Path, dict[str, Any]]:
    key = api_key.strip()
    if not key or any(character.isspace() for character in key):
        raise RuntimeError("Qwen API key is missing or invalid")
    legacy_client = _default_client_factory(key, base_url)
    unified_client = _default_client_factory(key, base_url)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    traces: list[dict[str, Any]] = []
    selected = tuple(cases)
    for repetition in range(1, repetitions + 1):
        for case in selected:
            legacy = _complete(legacy_client, case, unified=False, model=model)
            unified = _complete(unified_client, case, unified=True, model=model)
            traces.append(
                {
                    "case_id": case.case_id,
                    "backend": case.backend,
                    "instruction": case.instruction,
                    "repetition": repetition,
                    "legacy": legacy,
                    "unified": unified,
                    "equivalent": legacy["semantic_signature"]
                    == unified["semantic_signature"],
                }
            )
    case_reports: list[dict[str, Any]] = []
    for case in selected:
        values = [trace for trace in traces if trace["case_id"] == case.case_id]
        legacy_signatures = [trace["legacy"]["semantic_signature"] for trace in values]
        unified_signatures = [trace["unified"]["semantic_signature"] for trace in values]
        case_reports.append(
            {
                "case_id": case.case_id,
                "backend": case.backend,
                "pair_match_rate": sum(trace["equivalent"] for trace in values) / len(values),
                "legacy_dominant_rate": _dominant_rate(legacy_signatures),
                "unified_dominant_rate": _dominant_rate(unified_signatures),
                "arguments_valid": all(
                    trace[side]["arguments_valid"]
                    for trace in values
                    for side in ("legacy", "unified")
                ),
                "legacy_signatures": sorted(set(legacy_signatures)),
                "unified_signatures": sorted(set(unified_signatures)),
            }
        )
    stable = all(
        item["pair_match_rate"] >= 0.8
        and item["legacy_dominant_rate"] >= 0.8
        and item["unified_dominant_rate"] >= 0.8
        and item["arguments_valid"]
        for item in case_reports
    )
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "model": model,
        "base_url": base_url,
        "repetitions": repetitions,
        "case_count": len(selected),
        "request_count": len(traces) * 2,
        "physical_execution": False,
        "thresholds": {"pair_match_rate": 0.8, "dominant_rate": 0.8},
        "stable": stable,
        "case_reports": case_reports,
        "latency": {
            side: {
                "median_s": round(
                    statistics.median(trace[side]["elapsed_s"] for trace in traces), 3
                ),
                "max_s": max(trace[side]["elapsed_s"] for trace in traces),
            }
            for side in ("legacy", "unified")
        },
    }
    (output / "traces.jsonl").write_text(
        "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces),
        encoding="utf-8",
    )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--model", default=os.environ.get("LUXI_QWEN_MODEL", DEFAULT_QWEN_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("LUXI_QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL))
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--output-root", type=Path, default=Path("runs/qwen-shadow-golden"))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"environment variable {args.api_key_env!r} is required")
    if not 1 <= args.repetitions <= 5:
        raise SystemExit("--repetitions must be between 1 and 5")
    selected_cases = CASES
    if args.case_ids:
        requested = frozenset(args.case_ids)
        selected_cases = tuple(case for case in CASES if case.case_id in requested)
        missing = requested - {case.case_id for case in selected_cases}
        if missing:
            raise SystemExit(f"unknown Golden Case ids: {sorted(missing)}")
    output, report = collect(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url.rstrip("/"),
        repetitions=args.repetitions,
        output_root=args.output_root,
        cases=selected_cases,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "stable": report["stable"],
                "request_count": report["request_count"],
                "case_reports": report["case_reports"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["stable"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
