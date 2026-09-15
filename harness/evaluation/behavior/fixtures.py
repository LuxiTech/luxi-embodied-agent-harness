"""Deterministic contract fixtures. These do not simulate robot physics or an LLM."""

from __future__ import annotations

import json
import time

from harness.runtime.contracts import ModelReply, ToolDecision
from .development import Condition, PrimitiveFailure


class FixtureBackend:
    kind = "contract-fixture-no-physics"
    robot_id = "fixture-robot"

    def __init__(self, *, fail_first_pick: bool = False):
        self.revision = 0
        self.open = False
        self.held = None
        self.inside = False
        self.fail_first_pick = fail_first_pick
        self.actions = []

    def observe(self):
        return {"objects": [{"object_ref": "cup"}, {"object_ref": "cabinet"}],
                "held_object": self.held, "cabinet_open": self.open,
                "cup_inside_cabinet": self.inside, "source": self.kind}

    def condition(self, c: Condition):
        if c.predicate == "open" and c.subject == "cabinet":
            return self.open
        if c.predicate == "holding" and c.subject == "cup":
            return self.held == "cup"
        if c.predicate == "inside" and (c.subject, c.target) == ("cup", "cabinet"):
            return self.inside
        return None

    def primitive(self, name, object_ref):
        if object_ref not in {"cup", "cabinet"}:
            raise PrimitiveFailure("fixture target not found")
        yield (name, object_ref)

    def step(self, action):
        self.revision += 1
        self.actions.append(action)
        name, obj = action
        if name == "OPEN" and obj == "cabinet":
            self.open = True
        elif name == "CLOSE" and obj == "cabinet":
            self.open = False
        elif name == "GRASP" and obj == "cup":
            if self.fail_first_pick:
                self.fail_first_pick = False
                raise PrimitiveFailure("injected empty grasp; inspect and retry")
            self.held, self.inside = "cup", False
        elif name == "PLACE_INSIDE" and obj == "cabinet":
            if self.held != "cup" or not self.open:
                raise PrimitiveFailure("place requires held cup and open cabinet")
            self.held, self.inside = None, True
        elif name != "NAVIGATE_TO":
            raise PrimitiveFailure("unsupported fixture operation")

    def hold_step(self):
        self.revision += 1

    def motion_sample(self):
        return {"timestamp_monotonic": time.monotonic(), "planar_speed_mps": 0.0, "yaw_rate_rps": 0.0}


class ScriptedFixtureProvider:
    """Feedback-aware scripted test driver, never presented as model planning."""

    def __init__(self):
        self.goals = [
            ("open-cabinet", {"predicate": "open", "subject": "cabinet"}, "behavior_open", "cabinet"),
            ("hold-cup", {"predicate": "holding", "subject": "cup"}, "behavior_pick", "cup"),
            ("put-cup", {"predicate": "inside", "subject": "cup", "target": "cabinet"}, "behavior_place_inside", "cabinet"),
            ("close-cabinet", {"predicate": "open", "subject": "cabinet", "expected": False}, "behavior_close", "cabinet"),
        ]
        self.index = 0
        self.phase = "observe"
        self.recovery = False

    def complete(self, request, cancel):
        cancel.raise_if_cancelled()
        latest = next((json.loads(m["content"]) for m in reversed(request.messages) if m["role"] == "tool"), {})
        if self.phase == "observe":
            self.phase = "declare"
            return self.call("behavior_observe")
        if self.index >= len(self.goals):
            return self.call("behavior_check_task")
        subgoal, condition, skill, obj = self.goals[self.index]
        if self.phase == "declare":
            self.phase = "check"
            return self.call("behavior_set_subgoal", subgoal_id=subgoal, conditions=[condition])
        if self.phase == "check":
            self.phase = "decide"
            return self.call("behavior_check_subgoal", subgoal_id=subgoal)
        if latest.get("verdict") == "passed":
            self.index += 1
            self.phase = "declare"
            self.recovery = False
            return self.complete(request, cancel)
        if latest.get("status") == "primitive_failed":
            self.recovery = True
        if latest.get("status") in {"primitive_executed", "primitive_failed"}:
            self.phase = "check"
            return self.complete(request, cancel)
        self.phase = "decide"
        args = {"subgoal_id": subgoal, "object_ref": obj}
        if self.recovery:
            args["recovery_reason"] = "Fresh check confirms empty hand; make one new grasp attempt."
        return self.call(skill, **args)

    @staticmethod
    def call(name, **arguments):
        return ModelReply(tool_calls=(ToolDecision(name, arguments),))
