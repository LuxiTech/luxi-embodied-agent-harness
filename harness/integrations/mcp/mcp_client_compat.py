"""Project-local timeout compatibility for long but bounded MCP skills."""

from __future__ import annotations

import os
import sys
from typing import Any


FETCH_OBJECT_MCP_TIMEOUT = 240


def _configured_timeout(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv if argv is None else argv)
    if any(
        arguments[index : index + 3] == ["mcp", "call", "fetch_object"]
        for index in range(max(0, len(arguments) - 2))
    ):
        try:
            value = int(
                os.environ.get(
                    "LUXI_FETCH_OBJECT_MCP_TIMEOUT",
                    str(FETCH_OBJECT_MCP_TIMEOUT),
                )
            )
        except ValueError:
            value = FETCH_OBJECT_MCP_TIMEOUT
        return max(180, min(300, value))
    try:
        value = int(os.environ.get("LUXI_MCP_TIMEOUT", "120"))
    except ValueError:
        value = 120
    return max(30, min(300, value))


def install_mcp_client_timeout_compat() -> bool:
    """Raise only the pinned adapter's default 30-second request timeout."""

    from dimos.agents.mcp import mcp_adapter

    current = mcp_adapter.McpAdapter
    if getattr(current, "_luxi_timeout_compatible", False):
        return True
    original_init = current.__init__

    def compatible_init(
        self: Any,
        url: str | None = None,
        timeout: int = mcp_adapter.DEFAULT_TIMEOUT,
    ) -> None:
        if timeout == mcp_adapter.DEFAULT_TIMEOUT:
            timeout = _configured_timeout()
        original_init(self, url=url, timeout=timeout)

    current.__init__ = compatible_init
    current._luxi_timeout_compatible = True
    return True
