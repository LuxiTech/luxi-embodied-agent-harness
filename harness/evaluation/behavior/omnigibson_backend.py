"""Optional v3.9.2 Starter adapter; privileged, local development only.

Owns env.step synchronously. Do not connect this backend to the official policy
server: it requires the simulator object graph and is not an RGB-D policy.
"""

from __future__ import annotations

import math
import time
from typing import Any

from .development import Condition, EpisodeEnded, PrimitiveFailure


class OmniGibsonDevelopmentBackend:
    kind = "omnigibson-starter-oracle"

    def __init__(self, env: Any, *, mode: str, max_sim_steps: int = 20000,
                 primitives: Any = None, primitive_set: Any = None,
                 state_types: Any = None, primitive_errors: tuple[type[Exception], ...] = ()) -> None:
        if mode != "oracle-dev":
            raise ValueError("OmniGibson Starter backend is privileged development only")
        if len(env.robots) != 1:
            raise ValueError("development adapter requires exactly one robot")
        if max_sim_steps < 1:
            raise ValueError("max_sim_steps must be positive")
        self.env, self.robot = env, env.robots[0]
        self.robot_id = str(self.robot.name)
        self.revision = 0
        self.max_sim_steps = max_sim_steps
        self.ended = False
        if primitives is None:
            # Heavy imports stay entirely outside the core harness.
            from omnigibson import object_states
            from omnigibson.action_primitives.action_primitive_set_base import (
                ActionPrimitiveError, ActionPrimitiveErrorGroup,
            )
            from omnigibson.action_primitives.starter_semantic_action_primitives import (
                StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet,
            )
            primitives = StarterSemanticActionPrimitives(env, self.robot, enable_head_tracking=False)
            primitive_set = StarterSemanticActionPrimitiveSet
            state_types = object_states
            primitive_errors = (ActionPrimitiveError, ActionPrimitiveErrorGroup)
        self.primitives, self.primitive_set = primitives, primitive_set
        self.state_types, self.primitive_errors = state_types, primitive_errors

    def _object(self, reference: str):
        obj = self.env.scene.objects_by_name.get(reference)
        if obj is None or obj is self.robot:
            raise ValueError(f"unknown development object reference: {reference}")
        return obj

    def observe(self):
        return {
            "revision": self.revision,
            "objects": [{"object_ref": name, "category": str(getattr(obj, "category", ""))}
                        for name, obj in sorted(self.env.scene.objects_by_name.items()) if obj is not self.robot],
            "held_object": getattr(self.primitives._get_obj_in_hand(), "name", None),
            "source": "simulator-object-registry (privileged)",
        }

    def condition(self, condition: Condition):
        try:
            obj = self._object(condition.subject)
            if condition.predicate == "holding":
                return self.primitives._get_obj_in_hand() is obj
            state_name = {"open": "Open", "toggled_on": "ToggledOn",
                          "inside": "Inside", "ontop": "OnTop"}[condition.predicate]
            state_type = getattr(self.state_types, state_name)
            state = obj.states.get(state_type)
            if state is None:
                return None
            args = () if condition.target is None else (self._object(condition.target),)
            return bool(state.get_value(*args))
        except (KeyError, ValueError, AttributeError):
            return None

    def primitive(self, name: str, object_ref: str):
        obj = self._object(object_ref)
        primitive = getattr(self.primitive_set, name)
        try:
            # Upstream defaults to five attempts; recovery belongs to our Agent.
            yield from self.primitives.apply_ref(primitive, obj, attempts=1)
        except self.primitive_errors as exc:
            raise PrimitiveFailure(str(exc)) from exc

    def step(self, action):
        if self.ended or self.revision >= self.max_sim_steps:
            raise EpisodeEnded("development episode ended or simulation-step budget exhausted")
        if tuple(action.shape) != (self.robot.action_dim,):
            raise ValueError("primitive action does not match robot.action_dim")
        if not all(math.isfinite(float(value)) for value in action):
            raise ValueError("primitive action contains non-finite values")
        result = self.env.step(action)
        self.revision += 1
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError("expected v3.9.2 Gymnasium five-value env.step result")
        self.ended = bool(result[2] or result[3])
        if self.ended:
            raise EpisodeEnded("environment terminated; no automatic reset or replay")

    def hold_step(self):
        # Controller-aware no-op: zero velocity, hold current position targets.
        self.step(self.primitives._empty_action(follow_arm_targets=False))

    def motion_sample(self):
        velocity = self.robot.get_linear_velocity()
        angular = self.robot.get_angular_velocity()
        return {"timestamp_monotonic": time.monotonic(),
                "planar_speed_mps": math.hypot(float(velocity[0]), float(velocity[1])),
                "yaw_rate_rps": float(angular[2])}
