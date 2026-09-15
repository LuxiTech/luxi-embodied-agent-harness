#!/usr/bin/env python3
"""Thin client for UI-owned long MCP jobs.

There is deliberately no detached fallback: without the operator UI there is
no stable lifecycle owner, cancellation path, or idle confirmation.
"""

from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


UI_BASE_URL = os.environ.get("LUXI_UI_URL", "http://127.0.0.1:8787").rstrip("/")


def request(method: str, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    message = Request(
        f"{UI_BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(message, timeout=5.0) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8")).get("error")
        except Exception:  # noqa: BLE001 - retain HTTP status as evidence
            detail = None
        raise SystemExit(str(detail or f"Luxi UI returned HTTP {error.code}")) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise SystemExit(
            "Luxi UI is unavailable; refusing an unsupervised detached MCP job"
        ) from error
    if not isinstance(result, dict):
        raise SystemExit("Luxi UI returned an invalid job response")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument("tool")
    start.add_argument("--json-args", required=True)
    for name in ("status", "cancel"):
        item = commands.add_parser(name)
        item.add_argument("job_id")
    args = parser.parse_args()

    if args.command == "start":
        try:
            arguments = json.loads(args.json_args)
        except json.JSONDecodeError as error:
            raise SystemExit(f"--json-args must be valid JSON: {error}") from error
        if not isinstance(arguments, dict):
            raise SystemExit("--json-args must contain a JSON object")
        result = request(
            "POST",
            "/api/mcp-jobs/start",
            {"tool": args.tool, "arguments": arguments},
        )
    elif args.command == "status":
        result = request("GET", f"/api/mcp-jobs/{args.job_id}")
    else:
        result = request(
            "POST",
            f"/api/mcp-jobs/{args.job_id}/cancel",
            {},
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
