#!/usr/bin/env python3
"""Run a single command on the remote host and print its output.

Reads MOSS_HOST / MOSS_PORT / MOSS_USER / MOSS_PASS_FILE; see remote_config.py.

    python3 tools/remote_exec.py 'hostname; hy-smi'
"""
import re
import shlex
import sys

import pexpect

from remote_config import ConfigError, load_config

TIMEOUT = 900


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "hostname"
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 3

    remote_cmd = "bash -lc " + shlex.quote(cmd)
    args = cfg.ssh_base_args() + (["-tt"] if cfg.uses_password else []) + [remote_cmd]

    if not cfg.uses_password:
        # Key-based auth: hand the terminal straight to ssh.
        import subprocess
        return subprocess.call(args)

    password = cfg.read_password()
    child = pexpect.spawn(
        args[0], args[1:], encoding="utf-8", timeout=60, maxread=400000,
        dimensions=(240, 400),
    )
    try:
        idx = child.expect(
            [r"[Pp]assword:", r"Permission denied", pexpect.EOF, pexpect.TIMEOUT],
            timeout=25,
        )
        if idx == 0:
            child.sendline(password)
            child.expect(pexpect.EOF, timeout=TIMEOUT)
        elif idx == 1:
            print("AUTH_FAILED")
            return 2
        elif idx == 2:
            pass
        else:
            print("TIMEOUT_WAITING_FOR_PASSWORD_PROMPT")
            return 3
        out = child.before or ""
        out = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", out)
        print(out.replace("\r\n", "\n").replace("\r", "\n"))
        return child.exitstatus or 0
    finally:
        child.close()


if __name__ == "__main__":
    raise SystemExit(main())
