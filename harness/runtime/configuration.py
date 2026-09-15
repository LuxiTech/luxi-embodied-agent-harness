"""Single-entry deployment configuration; retired selectors fail explicitly."""

RETIRED_SETTINGS = frozenset({
    'LUXI_AGENT_PROVIDER', 'LUXI_G1_AGENT_PROVIDER',
    'LUXI_AGENT_LOOP_MODE', 'LUXI_G1_AGENT_LOOP_MODE',
    'LUXI_QWEN_PHYSICAL_PIPELINE_TOOLS', 'LUXI_ISAAC_PHYSICAL_TOOLS',
    'LUXI_QWEN_READONLY_PIPELINE',
})


def validate_runtime_environment(environment):
    retired = sorted(name for name in RETIRED_SETTINGS if environment.get(name, '').strip())
    if retired:
        raise ValueError('已删除旧入口配置：' + ', '.join(retired) +
                         '。请移除这些变量；唯一入口为 Harness，模型配置使用 LUXI_MODEL_PROVIDER=qwen，物理工具使用 LUXI_PHYSICAL_PIPELINE_TOOLS。')
    if environment.get('LUXI_MODEL_PROVIDER', 'qwen').strip().lower() != 'qwen':
        raise ValueError('当前仅安装 qwen 模型适配器；model_provider 不用于选择 Agent runtime')
