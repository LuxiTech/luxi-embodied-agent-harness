"""Lightweight registration for Luxi-owned DimOS blueprints.

Keep this module free of blueprint imports. DimOS must apply CLI global
configuration (especially viewer selection) before resolving these strings;
eagerly importing ``tool_blueprints`` bakes the upstream defaults into the
composed blueprints too early.
"""

from __future__ import annotations

import importlib


LOCAL_BLUEPRINTS = {
    "luxi-g1-tools-sim": "harness.integrations.dimos.tool_blueprints:luxi_g1_tools_sim",
    "luxi-g1-isaac-tools-sim": (
        "harness.integrations.dimos.tool_blueprints:luxi_g1_isaac_tools_sim"
    ),
    "luxi-g1-navigation-benchmark": (
        "harness.integrations.dimos.tool_blueprints:luxi_g1_navigation_benchmark"
    ),
}


def register_luxi_blueprints() -> dict[str, str]:
    registry = importlib.import_module("dimos.robot.all_blueprints")
    registry.all_blueprints.update(LOCAL_BLUEPRINTS)
    return dict(LOCAL_BLUEPRINTS)
