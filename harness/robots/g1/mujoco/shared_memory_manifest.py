"""Exact worker-owned channel binding for explicit candidate sessions."""
import json
import os
from pathlib import Path
import re


def write_manifest(path, names):
    channels = {}
    for channel in ('odom', 'cmd', 'video'):
        name = names.get(channel) if isinstance(names, dict) else None
        if not isinstance(name, str) or re.fullmatch(r'psm_[A-Za-z0-9_]+', name) is None:
            raise ValueError(f'invalid shared-memory channel {channel}')
        channels[channel] = name
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump({'schema_version': 1, 'channels': channels}, handle)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
