"""Robot-independent navigation metrics and portable report files.

The formulas are intentionally independent from DimOS so reports remain
readable and testable even when a simulator is not installed.  The live runner
that feeds these values is defined in :mod:`harness.integrations.dimos.tool_blueprints`.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from itertools import pairwise
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class EpisodeSpec:
    """One goal relative to the benchmark's initial odometry pose."""

    name: str
    goal_offset_m: tuple[float, float, float]
    timeout_s: float = 45.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")


@dataclass(frozen=True)
class EpisodeResult:
    """Metrics captured for one closed-loop navigation episode."""

    name: str
    seed: int
    success: bool
    timed_out: bool
    elapsed_s: float
    path_length_m: float
    optimal_distance_m: float
    spl: float
    planning_latency_s: float | None
    replans: int
    collision_count: int | None
    final_error_m: float
    odom_samples: int


@dataclass(frozen=True)
class BenchmarkSummary:
    episodes: int
    successes: int
    success_rate: float
    timeout_rate: float
    mean_elapsed_s: float
    mean_path_length_m: float
    mean_spl: float
    mean_planning_latency_s: float | None
    collision_episode_rate: float | None
    total_collisions: int | None


def distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def path_length_xy(positions: list[tuple[float, float]]) -> float:
    return sum(distance_xy(start, end) for start, end in pairwise(positions))


def spl(success: bool, optimal_distance_m: float, path_length_m: float) -> float:
    """Compute Success weighted by Path Length for one episode."""

    if not success:
        return 0.0
    if optimal_distance_m <= 0.0:
        return 1.0
    return optimal_distance_m / max(optimal_distance_m, path_length_m)


def compute_episode_result(
    *,
    spec: EpisodeSpec,
    success: bool,
    elapsed_s: float,
    positions: list[tuple[float, float]],
    goal_xy: tuple[float, float],
    planning_latency_s: float | None,
    path_messages: int,
    collision_count: int | None,
) -> EpisodeResult:
    if not positions:
        raise ValueError("positions must contain at least the episode start pose")
    travelled = path_length_xy(positions)
    optimal = distance_xy(positions[0], goal_xy)
    return EpisodeResult(
        name=spec.name,
        seed=spec.seed,
        success=success,
        timed_out=not success and elapsed_s >= spec.timeout_s,
        elapsed_s=max(0.0, elapsed_s),
        path_length_m=travelled,
        optimal_distance_m=optimal,
        spl=spl(success, optimal, travelled),
        planning_latency_s=planning_latency_s,
        replans=max(0, path_messages - 1),
        collision_count=collision_count,
        final_error_m=distance_xy(positions[-1], goal_xy),
        odom_samples=len(positions),
    )


def summarize_results(results: list[EpisodeResult]) -> BenchmarkSummary:
    if not results:
        raise ValueError("results must not be empty")
    count = len(results)
    successes = sum(result.success for result in results)
    latencies = [
        result.planning_latency_s
        for result in results
        if result.planning_latency_s is not None
    ]
    collision_known = all(result.collision_count is not None for result in results)
    collision_counts = [int(result.collision_count or 0) for result in results]
    return BenchmarkSummary(
        episodes=count,
        successes=successes,
        success_rate=successes / count,
        timeout_rate=sum(result.timed_out for result in results) / count,
        mean_elapsed_s=sum(result.elapsed_s for result in results) / count,
        mean_path_length_m=sum(result.path_length_m for result in results) / count,
        mean_spl=sum(result.spl for result in results) / count,
        mean_planning_latency_s=(sum(latencies) / len(latencies) if latencies else None),
        collision_episode_rate=(
            sum(value > 0 for value in collision_counts) / count
            if collision_known
            else None
        ),
        total_collisions=sum(collision_counts) if collision_known else None,
    )


class BenchmarkReportWriter:
    """Persist one run as JSONL, CSV and one aggregate JSON document."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._results: list[EpisodeResult] = []

    @property
    def results(self) -> list[EpisodeResult]:
        return list(self._results)

    def append(self, result: EpisodeResult) -> None:
        record = asdict(result)
        with (self.output_dir / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        csv_path = self.output_dir / "episodes.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            if write_header:
                writer.writeheader()
            writer.writerow(record)
        self._results.append(result)

    def write_summary(self, summary: BenchmarkSummary) -> None:
        (self.output_dir / "summary.json").write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
