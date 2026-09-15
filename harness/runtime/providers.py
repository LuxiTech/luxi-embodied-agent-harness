"""Pure model adapters; no tool execution or robot access is allowed here."""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Mapping

from .contracts import CancellationToken, ModelReply, ModelRequest, ToolDecision


class OpenAICompatibleModelProvider:
    def __init__(
        self,
        client: Any,
        *,
        model: str,
        system_prompt: str | Callable[[ModelRequest], str],
        temperature: float = 0.1,
        extra_body: Mapping[str, Any] | Callable[[ModelRequest], Mapping[str, Any]] | None = None,
        tool_choice: Callable[[ModelRequest], str | Mapping[str, Any]] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.tool_choice = tool_choice

    def complete(self, request: ModelRequest, cancel: CancellationToken) -> ModelReply:
        cancel.raise_if_cancelled()
        tool_choice = self.tool_choice(request) if self.tool_choice else "auto"
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt(request) if callable(self.system_prompt) else self.system_prompt},
                *request.messages,
            ],
            "tools": list(request.tools),
            "tool_choice": tool_choice,
            "temperature": self.temperature,
        }
        extra_body = dict(self.extra_body(request) if callable(self.extra_body) else self.extra_body)
        if extra_body:
            kwargs["extra_body"] = extra_body
        try:
            completion = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Provider exceptions may echo credentials or request bodies.
            status = getattr(exc, 'status_code', None)
            code = f" HTTP {status}" if isinstance(status, int) else ""
            raise RuntimeError(f"Model completion failed: {type(exc).__name__}{code}") from None
        cancel.raise_if_cancelled()
        choice = completion.choices[0]
        message = choice.message
        decisions: list[ToolDecision] = []
        for call in list(getattr(message, "tool_calls", None) or []):
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"model returned invalid tool arguments: {exc}") from exc
            if not isinstance(arguments, dict):
                raise ValueError("model tool arguments must be an object")
            decisions.append(
                ToolDecision(str(call.function.name), arguments, str(call.id))
            )
        usage = getattr(completion, "usage", None)
        usage_data = usage.model_dump() if usage and hasattr(usage, "model_dump") else {}
        return ModelReply(
            content=str(message.content or ""),
            tool_calls=tuple(decisions),
            finish_reason=str(getattr(choice, "finish_reason", "stop")),
            usage=usage_data,
        )


class LazyOpenAICompatibleModelProvider:
    """Create one isolated client lazily for the model instance."""

    def __init__(
        self,
        client_factory: Callable[[], Any],
        **provider_options: Any,
    ) -> None:
        self.client_factory = client_factory
        self.provider_options = provider_options
        self._provider: OpenAICompatibleModelProvider | None = None
        self._lock = threading.Lock()

    def complete(self, request: ModelRequest, cancel: CancellationToken) -> ModelReply:
        with self._lock:
            if self._provider is None:
                self._provider = OpenAICompatibleModelProvider(
                    self.client_factory(), **self.provider_options
                )
            provider = self._provider
        return provider.complete(request, cancel)

    def close(self):
        with self._lock:
            provider, self._provider = self._provider, None
        if provider is not None:
            close = getattr(provider.client, "close", None)
            if callable(close):
                close()


class ObservedModelProvider:
    def __init__(
        self,
        provider: Any,
        observer: Callable[[ModelRequest, ModelReply], None],
    ) -> None:
        self.provider = provider
        self.observer = observer

    def complete(self, request: ModelRequest, cancel: CancellationToken) -> ModelReply:
        reply = self.provider.complete(request, cancel)
        self.observer(request, reply)
        return reply
