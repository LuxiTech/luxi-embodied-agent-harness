"""One-shot Qwen request worker used to enforce a total wall-clock deadline."""

from __future__ import annotations

import json
import sys
from typing import Any


def _reply(payload: dict[str, Any], *, exit_code: int) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.flush()
    raise SystemExit(exit_code)


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        api_key = str(payload["api_key"])
        base_url = str(payload["base_url"])
        request = payload["request"]
        timeout_seconds = float(payload["http_timeout_seconds"])
        if not isinstance(request, dict):
            raise TypeError("request must be an object")

        import httpx
        from openai import OpenAI

        http_client = httpx.Client(
            trust_env=False,
            timeout=httpx.Timeout(timeout_seconds, connect=min(15.0, timeout_seconds)),
        )
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client,
                max_retries=0,
            )
            completion = client.chat.completions.create(**request)
            content = completion.choices[0].message.content or ""
            request_id = getattr(completion, "id", None)
        finally:
            http_client.close()
    except BaseException as exc:  # noqa: BLE001 - serialize failure across process
        error = str(exc)
        configured_key = locals().get("api_key")
        if configured_key:
            error = error.replace(str(configured_key), "[redacted]")
        _reply(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": error[:400],
            },
            exit_code=1,
        )
    _reply(
        {
            "ok": True,
            "content": content,
            "request_id": str(request_id) if request_id else None,
        },
        exit_code=0,
    )


if __name__ == "__main__":
    main()
