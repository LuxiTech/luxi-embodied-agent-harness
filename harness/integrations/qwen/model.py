"""Qwen completion options; no session, planning loop, tools or robot access."""
from collections.abc import Mapping
from harness.runtime.providers import LazyOpenAICompatibleModelProvider


def create_qwen_model_provider(config, *, system_prompt, tool_choice):
    def extra_body(request):
        options = {'top_k': 1}
        if config.model.casefold().startswith('qwen3') and isinstance(tool_choice(request), Mapping):
            options['enable_thinking'] = False
        return options
    return LazyOpenAICompatibleModelProvider(config.create_client, model=config.model,
        system_prompt=system_prompt, temperature=0.1, tool_choice=tool_choice,
        extra_body=extra_body)
