"""Persistent dialogue context plus fresh, sanitized robot observations."""
from __future__ import annotations

import base64
import json
from typing import Any

from .session_store import SessionContextProvider
from harness.skills.tool_results import _blind_safe_payload


class RobotContextProvider(SessionContextProvider):
    def __init__(self, store, *, observe, prepare_turn=lambda: None,
                 blind_mode=False, camera=None, vision_enabled=False, **budgets):
        super().__init__(store, **budgets)
        self.observe = observe
        self.prepare_turn = prepare_turn
        self.blind_mode = blind_mode
        self.camera = camera
        self.vision_enabled = vision_enabled

    def build_context(self, *, session_id, turn_id, instruction):
        self.prepare_turn()
        messages = self.history(session_id)
        try:
            observation = self.observe()
        except Exception as exc:
            observation = {"available": False, "error": str(exc)[:1000]}
        payload = {"instruction": instruction, "initial_observation": observation,
                   "note": "运行时已自动完成本轮首次 observe_environment。历史观察不代表当前物理状态。"}
        if self.blind_mode:
            payload = _blind_safe_payload(payload)
        text = json.dumps(payload, ensure_ascii=False, default=str)
        content: Any = text
        if self.vision_enabled and self.camera is not None:
            frame = self.camera()
            if isinstance(frame, bytes) and 128 <= len(frame) <= 5_000_000 and frame.startswith(b"\xff\xd8"):
                content = [{"type": "text", "text": text},
                           {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii")}}]
        messages.append({"role": "user", "content": content})
        return messages


def persistent_messages(messages):
    """Do not retain raw images in the event log or resend old camera frames."""
    result = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = [part for part in content if isinstance(part, dict) and part.get("type") == "text"]
        result.append(item)
    return result
