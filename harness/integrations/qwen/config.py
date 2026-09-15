"""Extracted shared implementation; independent of Agent entry points."""

from __future__ import annotations
from typing import Any


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DEFAULT_QWEN_MODEL = "qwen3.7-max"

def _default_client_factory(api_key: str, base_url: str) -> Any:
    # The workstation may export a socks:// proxy intended for browser tools.
    # HTTPX rejects that URL form, while DashScope is directly reachable.  Keep
    # this one client isolated instead of mutating process-wide proxy settings.
    import httpx
    from openai import OpenAI

    http_client = httpx.Client(
        trust_env=False,
        timeout=httpx.Timeout(75.0, connect=15.0),
    )
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        max_retries=0,
    )


from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path


@dataclass(frozen=True)
class QwenModelConfig:
    model: str = DEFAULT_QWEN_MODEL
    base_url: str = DEFAULT_QWEN_BASE_URL
    key_file: Path | None = None
    vision_enabled: bool = False

    @classmethod
    def from_environment(cls):
        path = os.environ.get("LUXI_QWEN_API_KEY_FILE", "").strip()
        return cls(
            model=os.environ.get("LUXI_QWEN_MODEL", os.environ.get("OPENAI_MODEL", DEFAULT_QWEN_MODEL)),
            base_url=os.environ.get("LUXI_QWEN_BASE_URL", os.environ.get("OPENAI_BASE_URL", DEFAULT_QWEN_BASE_URL)).rstrip("/"),
            key_file=Path(path).expanduser() if path else None,
            vision_enabled=os.environ.get("LUXI_QWEN_VISION", "0").lower() in {"1", "true", "yes", "on"},
        )

    def configuration_error(self) -> str | None:
        if not self.base_url.startswith("https://"):
            return "LUXI_QWEN_BASE_URL 必须使用 HTTPS"
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            if self.key_file is None or not self.key_file.is_file():
                return "未配置可读的模型 API key 文件"
            try:
                if self.key_file.stat().st_mode & 0o077:
                    return "模型 API key 文件权限过宽，请设置为 600"
                if not self.key_file.read_text(encoding="utf-8").strip():
                    return "模型 API key 文件为空"
            except (OSError, UnicodeError):
                return "无法读取模型 API key 文件"
        if importlib.util.find_spec("openai") is None or importlib.util.find_spec("httpx") is None:
            return "当前环境缺少 openai 或 httpx"
        return None

    def create_client(self):
        error = self.configuration_error()
        if error:
            raise RuntimeError(error)
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            try:
                key = self.key_file.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise RuntimeError("无法读取模型 API key 文件") from exc
        if not key.startswith("sk-") or any(c.isspace() for c in key):
            raise RuntimeError("模型 API key 格式无效")
        return _default_client_factory(key, self.base_url)
