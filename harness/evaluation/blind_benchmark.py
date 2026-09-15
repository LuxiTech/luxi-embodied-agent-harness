"""Run one real, information-isolated blind-scene closed-loop trial."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from harness.evaluation.blind_evaluation import (
    BLIND_SCENE_IDS,
    prepare_blind_run,
    require_blind_harness,
    score_blind_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTRUCTION = (
    "这是未知场景。请只根据本轮在线第一视角、RGB-D 和地图找到门，"
    "走到门前约 0.9 米停下，并在停车完成后用新的第一视角 RGB-D 帧确认距离；"
    "如果没有完成最终验证，不要报告成功。"
)


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def summarize_tool_timings(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair real tool call/result events in order; leave unmatched events out."""

    pending: dict[str, deque[float]] = defaultdict(deque)
    timings: list[dict[str, Any]] = []
    for event in events:
        if event.get("source") != "tool":
            continue
        data = event.get("data")
        tool = str(data.get("tool", "")) if isinstance(data, dict) else ""
        event_time = _timestamp(event.get("timestamp"))
        if not tool or event_time is None:
            continue
        if event.get("kind") == "call":
            pending[tool].append(event_time)
        elif event.get("kind") == "result" and pending[tool]:
            started = pending[tool].popleft()
            timings.append(
                {
                    "tool": tool,
                    "duration_ms": round(max(0.0, event_time - started) * 1_000.0, 1),
                }
            )
    return timings


def _target_discovery_wall_time(
    vision_requests: Iterable[dict[str, Any]],
) -> float | None:
    """Return the first completed semantic localization, never a failed request."""

    candidates = [
        value
        for request in vision_requests
        if request.get("localization_stage") == "localized"
        and not request.get("error")
        for value in [_timestamp(request.get("finished_at"))]
        if value is not None
    ]
    return min(candidates) if candidates else None


def _wait_for_scorer_sample_after(
    telemetry_path: Path,
    after_wall_time: Any,
    *,
    timeout_seconds: float = 2.0,
) -> bool | None:
    """Wait briefly for independent truth at or after the verification frame."""

    try:
        boundary = float(after_wall_time)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(boundary):
        return None
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        try:
            lines = telemetry_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in reversed(lines):
            try:
                record = json.loads(line)
                wall_time = float(record.get("wall_time"))
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                continue
            if (
                record.get("source") == "scorer_ground_truth"
                and math.isfinite(wall_time)
                and wall_time >= boundary
            ):
                return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _scorer_evidence_boundary(
    task_result: dict[str, Any],
) -> tuple[str | None, float | None]:
    """Choose the strongest available agent-side boundary for scorer truth."""

    for field in (
        "verification_frame_timestamp",
        "stationary_confirmed_at",
        "stop_command_completed_at",
    ):
        try:
            value = float(task_result.get(field))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            return field, value
    return None, None


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _json_request(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(base_url + path, data=data, headers=headers)
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback only
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{path} did not return a JSON object")
    return parsed


def _bytes_request(
    base_url: str,
    path: str,
    *,
    timeout: float = 3.0,
) -> bytes:
    request = Request(base_url + path, headers={"Accept": "image/jpeg"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback only
        return response.read()


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _write_private_jpeg(path: Path, payload: bytes) -> None:
    if len(payload) < 4 or not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise ValueError("first-person evidence is not a complete JPEG")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_bytes(payload)
    path.chmod(0o600)


def _capture_first_person_frame(
    base_url: str,
    destination: Path,
    *,
    timeout_seconds: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            _write_private_jpeg(
                destination,
                _bytes_request(base_url, "/api/camera.jpg"),
            )
            return
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
            last_error = error
            time.sleep(0.1)
    raise RuntimeError(f"could not capture first-person evidence: {last_error}")


def _collect_events(base_url: str, cursor: int) -> tuple[list[dict[str, Any]], int]:
    payload = _json_request(base_url, f"/api/events?after={cursor}")
    events = payload.get("events")
    records = [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []
    return records, int(payload.get("cursor", cursor))


def _wait_for_ready(
    process: subprocess.Popen[str],
    base_url: str,
    depth_path: Path,
    *,
    launched_at_wall: float,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    measurements: dict[str, Any] = {
        "first_rgb_frame_ready_s": None,
        "first_depth_frame_ready_s": None,
        "first_costmap_ready_s": None,
    }
    last_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"blind UI exited before readiness (code {process.returncode})")
        try:
            state = _json_request(base_url, "/api/state")
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(poll_seconds)
            continue
        last_state = state
        elapsed = max(0.0, time.time() - launched_at_wall)
        world = state.get("world") if isinstance(state.get("world"), dict) else {}
        costmap = state.get("costmap") if isinstance(state.get("costmap"), dict) else {}
        simulation = (
            state.get("simulation")
            if isinstance(state.get("simulation"), dict)
            else {}
        )
        if world.get("camera_available") and measurements["first_rgb_frame_ready_s"] is None:
            measurements["first_rgb_frame_ready_s"] = round(elapsed, 3)
        if depth_path.is_file() and time.time() - depth_path.stat().st_mtime <= 3.0:
            if measurements["first_depth_frame_ready_s"] is None:
                measurements["first_depth_frame_ready_s"] = round(elapsed, 3)
        live_costmap = bool(
            costmap.get("available")
            and costmap.get("source") == "live"
            and float(costmap.get("age_seconds", math.inf)) <= 3.0
        )
        if live_costmap and measurements["first_costmap_ready_s"] is None:
            measurements["first_costmap_ready_s"] = round(elapsed, 3)
        risk = (
            world.get("metrics", {}).get("risk")
            if isinstance(world.get("metrics"), dict)
            else "unknown"
        )
        if (
            simulation.get("mcp")
            and simulation.get("command_center")
            and measurements["first_rgb_frame_ready_s"] is not None
            and measurements["first_depth_frame_ready_s"] is not None
            and live_costmap
            and risk == "clear"
        ):
            measurements["costmap_age_at_submit_s"] = round(
                float(costmap.get("age_seconds", 0.0)), 3
            )
            measurements["costmap_timestamp_at_submit"] = costmap.get("timestamp")
            measurements["costmap_source"] = costmap.get("source")
            return state, measurements
        time.sleep(poll_seconds)
    raise TimeoutError(
        "blind runtime did not produce fresh RGB, depth, costmap and clear risk "
        f"within {timeout_seconds:.0f}s; last_state={json.dumps(last_state, default=str)[:1000]}"
    )


def run_blind_trial(
    *,
    scene_id: str,
    seed: int,
    instruction: str = DEFAULT_INSTRUCTION,
    port: int = 8787,
    readiness_timeout_seconds: float = 180.0,
    task_timeout_seconds: float = 240.0,
    poll_seconds: float = 0.25,
    include_person: bool = False,
    scorer_video: bool = False,
    cuda_device: str | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute exactly one isolated trial and return its scorer report."""

    launch_environment = dict(os.environ if environment is None else environment)
    require_blind_harness(launch_environment)
    for required_port in (port, 9990, 7779):
        if not _port_is_free(required_port):
            raise RuntimeError(
                f"port {required_port} is already in use; refusing to disturb another project"
            )
    runtime_root = Path(
        launch_environment.get(
            "DIMOS_RUNTIME_DIR",
            str(Path.home() / "work/Asset/dimos/runtime"),
        )
    ).expanduser().resolve()
    asset_root = Path(
        launch_environment.get(
            "DIMOS_ASSET_ROOT",
            str(Path.home() / "work/Asset/dimos"),
        )
    ).expanduser().resolve()
    prepared = prepare_blind_run(
        runtime_root,
        scene_id=scene_id,
        seed=seed,
        include_person=include_person,
        scorer_video=scorer_video,
    )
    paths = prepared.runtime_paths()
    launch_environment.update(prepared.agent_environment())
    launch_environment["LUXI_MUJOCO_HEADLESS"] = "1"
    launch_environment["LUXI_THIRD_PERSON_ENABLED"] = "0"
    launch_environment.pop("LUXI_THIRD_PERSON_FRAME", None)
    if cuda_device is not None:
        launch_environment["CUDA_VISIBLE_DEVICES"] = cuda_device

    scorer_directory = paths.run_directory / "scorer"
    scorer_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    scorer_directory.chmod(0o700)
    log_path = scorer_directory / "ui.log"
    command = [
        sys.executable,
        "-m",
        "harness.app.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--asset-root",
        str(asset_root),
        "--no-browser",
    ]
    launched_at_wall = time.time()
    base_url = f"http://127.0.0.1:{port}"
    process: subprocess.Popen[str] | None = None
    last_state: dict[str, Any] = {}
    owned_simulation = False
    events: list[dict[str, Any]] = []
    cursor = 0
    timed_out = False
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log_path.chmod(0o600)
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=launch_environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            last_state, measurements = _wait_for_ready(
                process,
                base_url,
                paths.head_depth_path,
                launched_at_wall=launched_at_wall,
                timeout_seconds=readiness_timeout_seconds,
                poll_seconds=poll_seconds,
            )
            simulation = last_state.get("simulation", {})
            owned_simulation = bool(
                isinstance(simulation, dict) and simulation.get("owned_by_ui")
            )
            agent_status = last_state.get("agent", {})
            if not isinstance(agent_status, dict) or agent_status.get("provider") != "qwen":
                raise RuntimeError("blind benchmark requires the configured Qwen agent provider")
            evaluation_status = last_state.get("evaluation", {})
            measurements.update(
                {
                    "agent_provider": agent_status.get("provider"),
                    "information_isolation": (
                        evaluation_status.get("information_isolation")
                        if isinstance(evaluation_status, dict)
                        else None
                    ),
                    "third_person_enabled": bool(
                        isinstance(simulation, dict)
                        and simulation.get("third_person_enabled")
                    ),
                    "scorer_video_enabled": bool(scorer_video),
                }
            )
            _capture_first_person_frame(
                base_url,
                scorer_directory / "initial-first-person.jpg",
            )
            measurements["initial_first_person_frame"] = (
                "scorer/initial-first-person.jpg"
            )

            submitted_at_wall = time.time()
            accepted = _json_request(
                base_url,
                "/api/commands",
                payload={"instruction": instruction},
            )
            if not accepted.get("ok"):
                raise RuntimeError(str(accepted.get("message") or "agent rejected instruction"))

            task_deadline = time.monotonic() + task_timeout_seconds
            first_motion_wall: float | None = None
            task_result: dict[str, Any] | None = None
            while time.monotonic() < task_deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"blind UI exited during task (code {process.returncode})"
                    )
                state = _json_request(base_url, "/api/state")
                last_state = state
                new_events, cursor = _collect_events(base_url, cursor)
                events.extend(new_events)
                world = state.get("world", {})
                command_values = world.get("command") if isinstance(world, dict) else None
                if (
                    first_motion_wall is None
                    and isinstance(command_values, list)
                    and len(command_values) >= 6
                    and max(
                        abs(float(command_values[0])),
                        abs(float(command_values[1])),
                        abs(float(command_values[5])),
                    )
                    >= 0.01
                ):
                    first_motion_wall = time.time()
                agent = state.get("agent", {})
                if isinstance(agent, dict) and not agent.get("busy"):
                    candidate = agent.get("last_task_result")
                    if isinstance(candidate, dict) and candidate:
                        task_result = dict(candidate)
                        break
                time.sleep(poll_seconds)
            if task_result is None:
                timed_out = True
                try:
                    _json_request(base_url, "/api/stop", payload={})
                except (HTTPError, URLError, TimeoutError):
                    pass
                task_result = {
                    "task_status": "benchmark_timeout",
                    "completed": False,
                    "planner_goal_reached": False,
                    "elapsed_s": round(time.time() - submitted_at_wall, 3),
                }

            new_events, cursor = _collect_events(base_url, cursor)
            events.extend(new_events)
            _capture_first_person_frame(
                base_url,
                scorer_directory / "final-first-person.jpg",
            )
            measurements["final_first_person_frame"] = (
                "scorer/final-first-person.jpg"
            )
            vision_requests = task_result.get("vision_requests")
            detection_wall: float | None = None
            if isinstance(vision_requests, list):
                detection_wall = _target_discovery_wall_time(
                    request
                    for request in vision_requests
                    if isinstance(request, dict)
                )
            scorer_boundary_field, scorer_boundary_time = _scorer_evidence_boundary(
                task_result
            )
            scorer_sample_ready = _wait_for_scorer_sample_after(
                paths.scorer_telemetry_path,
                scorer_boundary_time,
            )
            measurements.update(
                {
                    "instruction_submitted_at": datetime.fromtimestamp(
                        submitted_at_wall
                    ).astimezone().isoformat(timespec="milliseconds"),
                    "target_discovery_s": (
                        round(max(0.0, detection_wall - submitted_at_wall), 3)
                        if detection_wall is not None
                        else None
                    ),
                    "first_motion_command_s": (
                        round(max(0.0, first_motion_wall - submitted_at_wall), 3)
                        if first_motion_wall is not None
                        else None
                    ),
                    "perception_to_first_motion_s": (
                        round(max(0.0, first_motion_wall - detection_wall), 3)
                        if first_motion_wall is not None and detection_wall is not None
                        else None
                    ),
                    "tool_timings": summarize_tool_timings(events),
                    "timed_out": timed_out,
                    "scorer_evidence_boundary": scorer_boundary_field,
                    "scorer_post_evidence_sample_ready": scorer_sample_ready,
                    "scorer_post_verification_sample_ready": (
                        scorer_sample_ready
                        if scorer_boundary_field == "verification_frame_timestamp"
                        else None
                    ),
                }
            )
            task_result.update(measurements)
            _write_private_json(scorer_directory / "agent-result.json", task_result)
            _write_private_json(
                scorer_directory / "events.json",
                {"schema_version": 1, "events": events},
            )
            report = score_blind_run(runtime_root, prepared.run_token, task_result)
            return report
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if owned_simulation and (not _port_is_free(9990) or not _port_is_free(7779)):
            subprocess.run(
                [str(PROJECT_ROOT / "scripts/dimos.sh"), "stop"],
                cwd=PROJECT_ROOT,
                env=launch_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Luxi blind-scene smoke trial")
    parser.add_argument("--scene", required=True, choices=BLIND_SCENE_IDS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--readiness-timeout", type=float, default=180.0)
    parser.add_argument("--task-timeout", type=float, default=240.0)
    parser.add_argument("--with-person", action="store_true")
    parser.add_argument("--scorer-video", action="store_true")
    parser.add_argument("--cuda-device")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = run_blind_trial(
        scene_id=args.scene,
        seed=args.seed,
        instruction=args.instruction,
        port=args.port,
        readiness_timeout_seconds=args.readiness_timeout,
        task_timeout_seconds=args.task_timeout,
        include_person=args.with_person,
        scorer_video=args.scorer_video,
        cuda_device=args.cuda_device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
