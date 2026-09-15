"""Exec a tool process that cannot outlive its local runtime owner on Linux."""

from __future__ import annotations

import ctypes
import os
import signal
import sys


PR_SET_PDEATHSIG = 1


def main() -> None:
    arguments = sys.argv[1:]
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        raise SystemExit("parent_death_exec requires a command")
    original_parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Close the race where the parent exited immediately before prctl.
    if os.getppid() != original_parent:
        os.kill(os.getpid(), signal.SIGKILL)
    os.execvpe(arguments[0], arguments, os.environ)


if __name__ == "__main__":
    main()
