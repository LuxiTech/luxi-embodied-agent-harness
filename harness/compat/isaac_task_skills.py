"""Compatibility imports for the former Isaac-specific module name.

New code must import :mod:`harness.skills.task_primitives`.  These primitives never
depended on Isaac APIs and now belong to the shared task layer.
"""

from harness.skills.task_primitives import (  # noqa: F401
    CsrtTargetTracker,
    FollowControlConfig,
    FrontierExplorationResult,
    FrontierSelectionConfig,
    KnownFreeFrontierGoal,
    PersonFollowResult,
    TrackingMeasurement,
    bbox_is_continuous,
    compute_follow_twist,
    select_known_free_frontier,
)

__all__ = [
    "CsrtTargetTracker",
    "FollowControlConfig",
    "FrontierExplorationResult",
    "FrontierSelectionConfig",
    "KnownFreeFrontierGoal",
    "PersonFollowResult",
    "TrackingMeasurement",
    "bbox_is_continuous",
    "compute_follow_twist",
    "select_known_free_frontier",
]
