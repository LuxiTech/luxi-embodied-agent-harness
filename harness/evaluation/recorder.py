#!/usr/bin/env python3
"""Record a synchronized G1 map replay and first-person camera video."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
import json
import gzip
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from harness.app.server import (
    CAMERA_BYTES,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DEFAULT_ASSET_ROOT,
    PoseSnapshot,
    RectangleObstacle,
    SharedMemoryProbe,
    load_table_obstacles,
)
from harness.evaluation.blind_evaluation import open_prepared_blind_run


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAP_PANEL = (24, 72, 610, 624)
CAMERA_PANEL = (654, 72, 602, 339)


@lru_cache(maxsize=16)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def read_camera_rgb(probe: SharedMemoryProbe) -> np.ndarray | None:
    """Return a copied RGB frame, tolerating a producer update during the read."""

    path = probe.camera_path()
    if path is None:
        return None
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    if len(payload) != CAMERA_BYTES:
        return None
    return np.frombuffer(payload, dtype=np.uint8).reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 3).copy()


def append_trail_point(
    trail: list[dict[str, float]], pose: PoseSnapshot, *, min_distance: float = 0.015
) -> None:
    if trail:
        previous = trail[-1]
        if math.hypot(pose.x - previous["x"], pose.y - previous["y"]) < min_distance:
            return
    trail.append({"x": pose.x, "y": pose.y, "t": pose.timestamp})


def _fit_rgb(image: np.ndarray, width: int, height: int) -> Image.Image:
    source = Image.fromarray(image, mode="RGB")
    scale = min(width / source.width, height / source.height)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.BILINEAR,
    )
    result = Image.new("RGB", (width, height), (4, 10, 9))
    result.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return result


def _draw_map(
    canvas: Image.Image,
    pose: PoseSnapshot,
    trail: Sequence[dict[str, float]],
    obstacles: Sequence[RectangleObstacle],
    costmap: dict[str, Any] | None = None,
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    x0, y0, width, height = MAP_PANEL
    draw.rounded_rectangle(
        (x0, y0, x0 + width, y0 + height),
        radius=16,
        fill=(7, 18, 16, 255),
        outline=(39, 74, 66, 255),
        width=2,
    )

    origin = trail[0] if trail else {"x": pose.x, "y": pose.y}
    center_x = float(origin["x"])
    center_y = float(origin["y"])
    points = [*trail, {"x": pose.x, "y": pose.y}]
    max_extent = max(
        [
            4.0,
            *(abs(float(point["x"]) - center_x) + 0.6 for point in points),
            *(abs(float(point["y"]) - center_y) + 0.6 for point in points),
        ]
    )
    radius = min(12.0, max_extent)
    scale = min(width, height) / (2.25 * radius)

    def to_screen(x: float, y: float) -> tuple[float, float]:
        return (
            x0 + width / 2 + (x - center_x) * scale,
            y0 + height / 2 - (y - center_y) * scale,
        )

    grid_start_x = math.floor(center_x - radius)
    grid_end_x = math.ceil(center_x + radius)
    grid_start_y = math.floor(center_y - radius)
    grid_end_y = math.ceil(center_y + radius)
    for world_x in range(grid_start_x, grid_end_x + 1):
        screen_x, _ = to_screen(world_x, center_y)
        if x0 <= screen_x <= x0 + width:
            draw.line((screen_x, y0, screen_x, y0 + height), fill=(113, 241, 200, 18), width=1)
    for world_y in range(grid_start_y, grid_end_y + 1):
        _, screen_y = to_screen(center_x, world_y)
        if y0 <= screen_y <= y0 + height:
            draw.line((x0, screen_y, x0 + width, screen_y), fill=(113, 241, 200, 18), width=1)

    for obstacle in obstacles:
        left, top = to_screen(obstacle.x_min, obstacle.y_max)
        right, bottom = to_screen(obstacle.x_max, obstacle.y_min)
        if right < x0 or left > x0 + width or bottom < y0 or top > y0 + height:
            continue
        draw.rectangle(
            (left, top, right, bottom),
            fill=(78, 104, 112, 86),
            outline=(111, 151, 158, 150),
            width=2,
        )

    if costmap is not None:
        width_cells = int(costmap["width"])
        resolution = float(costmap["resolution"])
        origin_data = costmap["origin"]
        origin_x = float(origin_data["x"])
        origin_y = float(origin_data["y"])
        origin_yaw = float(origin_data.get("yaw", 0.0))
        cosine_origin = math.cos(origin_yaw)
        sine_origin = math.sin(origin_yaw)
        raw = costmap["raw"]
        occupied_pixels: list[tuple[int, int]] = []
        for cell_index, value in enumerate(raw):
            if not 50 <= value <= 100:
                continue
            cell_x = (cell_index % width_cells + 0.5) * resolution
            cell_y = (cell_index // width_cells + 0.5) * resolution
            world_x = origin_x + cell_x * cosine_origin - cell_y * sine_origin
            world_y = origin_y + cell_x * sine_origin + cell_y * cosine_origin
            screen_x, screen_y = to_screen(world_x, world_y)
            if x0 <= screen_x <= x0 + width and y0 <= screen_y <= y0 + height:
                occupied_pixels.append((round(screen_x), round(screen_y)))
        if occupied_pixels:
            draw.point(occupied_pixels, fill=(143, 178, 184, 190))

    if len(trail) > 1:
        path = [to_screen(float(point["x"]), float(point["y"])) for point in trail]
        draw.line(path, fill=(113, 241, 200, 190), width=4, joint="curve")
        for point in path[:: max(1, len(path) // 20)]:
            draw.ellipse(
                (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2),
                fill=(184, 255, 231, 210),
            )

    robot_x, robot_y = to_screen(pose.x, pose.y)
    direction = -pose.yaw + math.pi / 2
    local_points = [(0.0, -13.0), (9.0, 10.0), (0.0, 6.0), (-9.0, 10.0)]
    robot_points: list[tuple[float, float]] = []
    cosine = math.cos(direction)
    sine = math.sin(direction)
    for local_x, local_y in local_points:
        robot_points.append(
            (
                robot_x + local_x * cosine - local_y * sine,
                robot_y + local_x * sine + local_y * cosine,
            )
        )
    draw.polygon(robot_points, fill=(113, 241, 200, 255))
    draw.ellipse((robot_x - 15, robot_y - 15, robot_x + 15, robot_y + 15), outline=(113, 241, 200, 90), width=2)


def render_replay_frame(
    camera_rgb: np.ndarray,
    pose: PoseSnapshot,
    trail: Sequence[dict[str, float]],
    obstacles: Sequence[RectangleObstacle],
    *,
    costmap: dict[str, Any] | None = None,
    elapsed: float,
    command: Sequence[float] | None = None,
    fps: float = 10.0,
) -> np.ndarray:
    """Render one RGB composite frame for the saved replay."""

    canvas = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (3, 9, 8))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((24, 20), "LUXIAGENT / G1 MAPPING REPLAY", font=_font(25, True), fill=(205, 255, 239))
    draw.text(
        (FRAME_WIDTH - 24, 27),
        f"T+{elapsed:06.1f}s  /  {fps:.0f} FPS",
        font=_font(14, True),
        fill=(113, 241, 200),
        anchor="ra",
    )

    _draw_map(canvas, pose, trail, obstacles, costmap)
    draw.text((42, 88), "MAP / ACCUMULATED TRAJECTORY", font=_font(14, True), fill=(184, 216, 207))

    camera_x, camera_y, camera_width, camera_height = CAMERA_PANEL
    camera = _fit_rgb(camera_rgb, camera_width, camera_height)
    canvas.paste(camera, (camera_x, camera_y))
    draw.rounded_rectangle(
        (camera_x, camera_y, camera_x + camera_width, camera_y + camera_height),
        radius=12,
        outline=(39, 74, 66, 255),
        width=2,
    )
    draw.rectangle(
        (camera_x, camera_y, camera_x + camera_width, camera_y + 34),
        fill=(3, 10, 9, 185),
    )
    draw.text(
        (camera_x + 14, camera_y + 9),
        "FIRST-PERSON RGB / 640x360",
        font=_font(13, True),
        fill=(205, 255, 239),
    )

    telemetry_top = 438
    draw.rounded_rectangle(
        (camera_x, telemetry_top, camera_x + camera_width, FRAME_HEIGHT - 24),
        radius=16,
        fill=(7, 18, 16, 255),
        outline=(39, 74, 66, 255),
        width=2,
    )
    draw.text((camera_x + 18, telemetry_top + 18), "ODOMETRY", font=_font(14, True), fill=(113, 241, 200))
    draw.text(
        (camera_x + 18, telemetry_top + 53),
        f"X   {pose.x:+7.3f} m\nY   {pose.y:+7.3f} m\nYAW {math.degrees(pose.yaw):+7.2f} deg",
        font=_font(21),
        spacing=13,
        fill=(220, 236, 231),
    )
    distance = 0.0
    for left, right in zip(trail, trail[1:]):
        distance += math.hypot(float(right["x"]) - float(left["x"]), float(right["y"]) - float(left["y"]))
    draw.text((camera_x + 320, telemetry_top + 18), "REPLAY", font=_font(14, True), fill=(113, 241, 200))
    draw.text(
        (camera_x + 320, telemetry_top + 53),
        f"PATH     {distance:6.2f} m\nSAMPLES  {len(trail):6d}\nSTATUS   RECORDING",
        font=_font(18),
        spacing=15,
        fill=(220, 236, 231),
    )
    if command is not None and len(command) >= 6:
        draw.text(
            (camera_x + 18, FRAME_HEIGHT - 54),
            f"CMD  x={command[0]:+.2f}  y={command[1]:+.2f}  yaw={command[5]:+.2f}",
            font=_font(13),
            fill=(139, 170, 162),
        )

    return np.asarray(canvas, dtype=np.uint8)


def _open_writer(path: Path, fps: float) -> tuple[cv2.VideoWriter, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (FRAME_WIDTH, FRAME_HEIGHT),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError("OpenCV could not open an H.264 or MPEG-4 video writer")


def _wait_for_sources(
    probe: SharedMemoryProbe, timeout: float
) -> tuple[np.ndarray, PoseSnapshot]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        camera = read_camera_rgb(probe)
        pose = probe.pose()
        if camera is not None and pose is not None:
            return camera, pose
        time.sleep(0.25)
    raise TimeoutError(
        "Timed out waiting for G1 camera and odometry shared memory. "
        "Start ./scripts/dimos.sh g1-tools first."
    )


def _read_live_costmap(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict) or payload.get("source") != "live":
            return None
        width = int(payload["width"])
        height = int(payload["height"])
        resolution = float(payload["resolution"])
        origin = payload["origin"]
        raw = base64.b64decode(payload["data"], validate=True)
        if width <= 0 or height <= 0 or len(raw) != width * height:
            return None
        if not math.isfinite(resolution) or resolution <= 0 or not isinstance(origin, dict):
            return None
        return {
            "width": width,
            "height": height,
            "resolution": resolution,
            "origin": dict(origin),
            "timestamp": payload.get("timestamp"),
            "raw": raw,
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def resolve_recording_inputs(
    *,
    environment: dict[str, str],
    manifest_path: Path | None,
    costmap_path: Path | None,
    blind_run_token: str | None,
    asset_root: Path,
    legacy_shm_scan: bool,
) -> tuple[Path | None, Path | None]:
    """Resolve exact sensor/map bindings without silently scanning shared memory."""

    token = (blind_run_token or environment.get("LUXI_BLIND_RUN_TOKEN", "")).strip()
    blind = environment.get("LUXI_BLIND_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if token:
        runtime_root = Path(
            environment.get("DIMOS_RUNTIME_DIR", str(asset_root / "runtime"))
        ).expanduser()
        paths = open_prepared_blind_run(runtime_root, token).runtime_paths()
        if manifest_path is not None and manifest_path.expanduser().resolve() != paths.shm_manifest_path:
            raise RuntimeError("blind recording must use this run's exact SHM manifest")
        if costmap_path is not None and costmap_path.expanduser().resolve() != paths.costmap_path:
            raise RuntimeError("blind recording must use this run's live costmap")
        return paths.shm_manifest_path, paths.costmap_path
    if blind:
        raise RuntimeError("blind recording requires LUXI_BLIND_RUN_TOKEN or --blind-run-token")
    if manifest_path is not None:
        return manifest_path.expanduser().resolve(), (
            costmap_path.expanduser().resolve() if costmap_path is not None else None
        )
    if legacy_shm_scan:
        return None, costmap_path.expanduser().resolve() if costmap_path is not None else None
    raise RuntimeError(
        "recording requires --shm-manifest; legacy psm_* scanning is disabled by default "
        "and requires explicit --legacy-shm-scan"
    )


def record_replay(
    output: Path,
    *,
    duration: float,
    fps: float,
    wait_timeout: float,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    manifest_path: Path | None = None,
    costmap_path: Path | None = None,
    blind_run_token: str | None = None,
    legacy_shm_scan: bool = False,
    legacy_office_overlay: bool = False,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    effective_environment = dict(os.environ if environment is None else environment)
    blind_requested = bool(
        (blind_run_token or effective_environment.get("LUXI_BLIND_RUN_TOKEN", "")).strip()
        or effective_environment.get("LUXI_BLIND_MODE", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if blind_requested and legacy_office_overlay:
        raise RuntimeError("blind recording forbids the legacy office1 overlay")
    resolved_manifest, resolved_costmap = resolve_recording_inputs(
        environment=effective_environment,
        manifest_path=manifest_path,
        costmap_path=costmap_path,
        blind_run_token=blind_run_token,
        asset_root=asset_root,
        legacy_shm_scan=legacy_shm_scan,
    )
    probe = SharedMemoryProbe(manifest_path=resolved_manifest)
    last_camera, last_pose = _wait_for_sources(probe, wait_timeout)
    obstacles = load_table_obstacles(asset_root) if legacy_office_overlay else []
    last_costmap = _read_live_costmap(resolved_costmap)
    trail: list[dict[str, float]] = []
    append_trail_point(trail, last_pose)

    metadata_path = output.with_suffix(".jsonl")
    writer, codec = _open_writer(output, fps)
    started = time.monotonic()
    frame_count = max(1, round(duration * fps))
    frame_interval = 1.0 / fps

    try:
        with metadata_path.open("w", encoding="utf-8") as metadata:
            for frame_index in range(frame_count):
                target = started + frame_index * frame_interval
                remaining = target - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

                camera = read_camera_rgb(probe)
                pose = probe.pose()
                command = probe.command()
                costmap = _read_live_costmap(resolved_costmap)
                if camera is not None:
                    last_camera = camera
                if pose is not None:
                    last_pose = pose
                    append_trail_point(trail, pose)
                if costmap is not None:
                    last_costmap = costmap

                elapsed = frame_index / fps
                frame = render_replay_frame(
                    last_camera,
                    last_pose,
                    trail,
                    obstacles,
                    costmap=last_costmap,
                    elapsed=elapsed,
                    command=command,
                    fps=fps,
                )
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                metadata.write(
                    json.dumps(
                        {
                            "frame": frame_index,
                            "elapsed": elapsed,
                            "pose": asdict(last_pose),
                            "command": command,
                            "camera_available": camera is not None,
                            "costmap_timestamp": (
                                last_costmap.get("timestamp") if last_costmap else None
                            ),
                            "person_distance_m": None,
                            "table_distance_m": None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        writer.release()

    return {
        "output": str(output.resolve()),
        "metadata": str(metadata_path.resolve()),
        "codec": codec,
        "frames": frame_count,
        "fps": fps,
        "duration": frame_count / fps,
        "trail_points": len(trail),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a synchronized G1 trajectory-map replay and first-person MP4."
    )
    parser.add_argument("--duration", type=float, default=30.0, help="recording length in seconds")
    parser.add_argument("--fps", type=float, default=10.0, help="output frames per second")
    parser.add_argument("--wait-timeout", type=float, default=120.0)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shm-manifest", type=Path)
    parser.add_argument("--costmap", type=Path)
    parser.add_argument("--blind-run-token")
    parser.add_argument(
        "--legacy-shm-scan",
        action="store_true",
        help="explicitly allow legacy /dev/shm/psm_* discovery outside blind evaluation",
    )
    parser.add_argument(
        "--legacy-office-overlay",
        action="store_true",
        help="explicitly draw the old office1 table overlay outside blind evaluation",
    )
    args = parser.parse_args()

    if not 1.0 <= args.duration <= 3_600.0:
        parser.error("--duration must be between 1 and 3600 seconds")
    if not 1.0 <= args.fps <= 30.0:
        parser.error("--fps must be between 1 and 30")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output or Path("outputs/luxi-replays") / f"g1-map-fpv-{timestamp}.mp4"

    result = record_replay(
        output,
        duration=args.duration,
        fps=args.fps,
        wait_timeout=args.wait_timeout,
        asset_root=args.asset_root,
        manifest_path=args.shm_manifest,
        costmap_path=args.costmap,
        blind_run_token=args.blind_run_token,
        legacy_shm_scan=args.legacy_shm_scan,
        legacy_office_overlay=args.legacy_office_overlay,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
