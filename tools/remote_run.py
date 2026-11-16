#!/usr/bin/env python3
"""Run a long-lived command on the remote host, streaming its output.

The remote platform used in the write-up reaps background processes when the SSH
session closes, so long jobs (model loads, generations, the web UI) must own a
persistent session. `nohup` is not enough. This is that session.

    python3 tools/remote_run.py 'bash /workspace/run.sh'
"""
import shlex
import subprocess
import sys

import pexpect

from remote_config import ConfigError, load_config


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sleep infinity"
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 3

    remote_cmd = "bash -lc " + shlex.quote(cmd)
    args = cfg.ssh_base_args() + [
        "-tt",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=6",
        remote_cmd,
    ]

    if not cfg.uses_password:
        return subprocess.call(args)

    password = cfg.read_password()
    child = pexpect.spawn(
        args[0], args[1:], encoding="utf-8", timeout=None, maxread=200000,
        dimensions=(240, 400),
    )
    idx = child.expect(
        [r"[Pp]assword:", r"Permission denied", pexpect.EOF, pexpect.TIMEOUT],
        timeout=25,
    )
    if idx == 0:
        child.sendline(password)
    elif idx == 1:
        print("AUTH_FAILED", flush=True)
        return 2
    elif idx == 2:
        print(child.before or "", flush=True)
        return 0
    else:
        print("TIMEOUT_WAITING_FOR_PASSWORD_PROMPT", flush=True)
        return 3

    while True:
        try:
            data = child.read_nonblocking(size=8192, timeout=1)
            if data:
                sys.stdout.write(data)
                sys.stdout.flush()
        except pexpect.TIMEOUT:
            continue
        except pexpect.EOF:
            break
    child.close()
    return child.exitstatus or 0


if __name__ == "__main__":
    raise SystemExit(main())
