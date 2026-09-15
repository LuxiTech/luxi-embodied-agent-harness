"""Live DimOS module that feeds robot-independent navigation metrics."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime
import os
from pathlib import Path as FilePath
import time
from typing import Any

from dimos_lcm.std_msgs import Bool
from pydantic import Field

from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Path import Path

from harness.evaluation.navigation_evaluation import (
    BenchmarkReportWriter,
    EpisodeResult,
    EpisodeSpec,
    compute_episode_result,
    distance_xy,
    summarize_results,
)


def standard_navigation_episodes() -> list[EpisodeSpec]:
    """A deterministic 0.75 m square around the initial simulation pose."""

    timeout = 35.0
    return [
        EpisodeSpec("east_075m", (0.75, 0.0, 0.0), timeout),
        EpisodeSpec("north_075m", (0.75, 0.75, 0.0), timeout),
        EpisodeSpec("west_075m", (0.0, 0.75, 0.0), timeout),
        EpisodeSpec("return_to_origin", (0.0, 0.0, 0.0), timeout),
    ]


def _default_output_root() -> FilePath:
    runtime = os.environ.get(
        "DIMOS_RUNTIME_DIR",
        str(FilePath.home() / "work/Asset/dimos/runtime"),
    )
    return FilePath(runtime) / "evaluations"


class NavigationBenchmarkConfig(ModuleConfig):
    episodes: list[EpisodeSpec] = Field(default_factory=standard_navigation_episodes)
    goal_tolerance_m: float = Field(default=0.25, gt=0.0)
    initial_odom_timeout_s: float = Field(default=300.0, gt=0.0)
    episode_dwell_s: float = Field(default=1.0, ge=0.0)
    min_trace_step_m: float = Field(default=0.005, ge=0.0)
    run_name: str = "g1-navigation-sim"
    output_root: FilePath = Field(default_factory=_default_output_root)


class NavigationBenchmark(Module):
    """Run explicit goal episodes and persist JSONL/CSV/summary reports."""

    config: NavigationBenchmarkConfig
    odom: In[PoseStamped]
    path: In[Path]
    goal_reached: In[Bool]
    collision: In[Bool]
    goal_request: Out[PoseStamped]
    stop_movement: Out[Bool]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_odom: PoseStamped | None = None
        self._odom_ready = asyncio.Event()
        self._episode_finished = asyncio.Event()
        self._benchmark_task: asyncio.Task[None] | None = None
        self._active = False
        self._episode_success = False
        self._episode_started_at = 0.0
        self._goal_xy = (0.0, 0.0)
        self._positions: list[tuple[float, float]] = []
        self._planning_latency_s: float | None = None
        self._path_messages = 0
        self._collision_stream_observed = False
        self._collision_active = False
        self._collision_count = 0

    async def main(self) -> AsyncGenerator[None, None]:
        self._benchmark_task = asyncio.create_task(self._run_benchmark())
        yield
        if self._benchmark_task is not None and not self._benchmark_task.done():
            self._benchmark_task.cancel()
            try:
                await self._benchmark_task
            except asyncio.CancelledError:
                pass
        self._benchmark_task = None

    async def handle_odom(self, message: PoseStamped) -> None:
        self._latest_odom = message
        self._odom_ready.set()
        if not self._active:
            return
        position = (message.x, message.y)
        if (
            not self._positions
            or distance_xy(self._positions[-1], position) >= self.config.min_trace_step_m
        ):
            self._positions.append(position)
        if distance_xy(position, self._goal_xy) <= self.config.goal_tolerance_m:
            self._episode_success = True
            self._episode_finished.set()

    async def handle_path(self, message: Path) -> None:
        if not self._active or not message:
            return
        self._path_messages += 1
        if self._planning_latency_s is None:
            self._planning_latency_s = time.monotonic() - self._episode_started_at

    async def handle_goal_reached(self, message: Bool) -> None:
        if self._active and message.data:
            self._episode_success = True
            self._episode_finished.set()

    async def handle_collision(self, message: Bool) -> None:
        self._collision_stream_observed = True
        active = bool(message.data)
        if self._active and active and not self._collision_active:
            self._collision_count += 1
        self._collision_active = active

    async def _run_benchmark(self) -> None:
        try:
            await asyncio.wait_for(
                self._odom_ready.wait(), timeout=self.config.initial_odom_timeout_s
            )
        except asyncio.TimeoutError:
            return
        assert self._latest_odom is not None
        origin = self._latest_odom
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        writer = BenchmarkReportWriter(
            self.config.output_root / f"{stamp}-{self.config.run_name}"
        )
        results: list[EpisodeResult] = []
        for spec in self.config.episodes:
            result = await self._run_episode(spec, origin)
            writer.append(result)
            results.append(result)
            self.stop_movement.publish(Bool(data=True))
            if self.config.episode_dwell_s:
                await asyncio.sleep(self.config.episode_dwell_s)
        writer.write_summary(summarize_results(results))

    async def _run_episode(
        self, spec: EpisodeSpec, origin: PoseStamped
    ) -> EpisodeResult:
        assert self._latest_odom is not None
        goal = PoseStamped(
            frame_id=origin.frame_id,
            position=Vector3(
                origin.x + spec.goal_offset_m[0],
                origin.y + spec.goal_offset_m[1],
                origin.z + spec.goal_offset_m[2],
            ),
            orientation=origin.orientation,
        )
        self._goal_xy = (goal.x, goal.y)
        self._positions = [(self._latest_odom.x, self._latest_odom.y)]
        self._planning_latency_s = None
        self._path_messages = 0
        self._collision_count = 0
        self._episode_success = False
        self._episode_finished.clear()
        self._episode_started_at = time.monotonic()
        self._active = True
        self.goal_request.publish(goal)
        try:
            await asyncio.wait_for(self._episode_finished.wait(), timeout=spec.timeout_s)
        except asyncio.TimeoutError:
            pass
        finally:
            self._active = False
        elapsed = time.monotonic() - self._episode_started_at
        if self._latest_odom is not None:
            final_position = (self._latest_odom.x, self._latest_odom.y)
            if self._positions[-1] != final_position:
                self._positions.append(final_position)
        return compute_episode_result(
            spec=spec,
            success=self._episode_success,
            elapsed_s=elapsed,
            positions=self._positions,
            goal_xy=self._goal_xy,
            planning_latency_s=self._planning_latency_s,
            path_messages=self._path_messages,
            collision_count=(
                self._collision_count if self._collision_stream_observed else None
            ),
        )
