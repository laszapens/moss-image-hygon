#!/usr/bin/env python3
"""Upload one local file to the remote host (base64 over the SSH channel).

Avoids depending on scp/rsync in minimal container images.

    python3 tools/remote_put.py tools/ui_server.py /workspace/ui_server.py
"""
import base64
import os
import shlex
import subprocess
import sys

import pexpect

from remote_config import ConfigError, load_config


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 64
    src, dst = sys.argv[1], sys.argv[2]
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 3

    payload = base64.b64encode(open(src, "rb").read()).decode()
    remote = (
        f"mkdir -p {shlex.quote(os.path.dirname(dst) or '.')} && "
        f"echo {shlex.quote(payload)} | base64 -d > {shlex.quote(dst)} && "
        f"wc -c {shlex.quote(dst)}"
    )
    args = cfg.ssh_base_args() + ["-tt", "bash -lc " + shlex.quote(remote)]

    if not cfg.uses_password:
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
            child.expect(pexpect.EOF, timeout=300)
        elif idx == 1:
            print("AUTH_FAILED")
            return 2
        print((child.before or "").replace("\r\n", "\n").replace("\r", "\n"))
        return child.exitstatus or 0
    finally:
        child.close()


if __name__ == "__main__":
    raise SystemExit(main())
