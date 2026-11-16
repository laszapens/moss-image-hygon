#!/usr/bin/env python3
"""Long-lived remote session plus local port forwarding.

Used to reach the web UI (port 8081) that runs on the remote box. SSH forward
failure is fatal on purpose: a silently missing tunnel looks like "the UI is
broken" when in fact nothing is listening locally.

    python3 tools/remote_tunnel.py 'cd /workspace && exec python3 ui_server.py'

Override the forwarded ports with MOSS_FORWARDS="8081:8081,8002:8002".
"""
import shlex
import subprocess
import sys

import pexpect

from remote_config import ConfigError, load_config

DEFAULT_FORWARDS = "8081:8081"


def parse_forwards(spec: str) -> list[tuple[str, str]]:
    pairs = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        local, _, remote = chunk.partition(":")
        pairs.append((local.strip(), (remote or local).strip()))
    return pairs


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sleep infinity"
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 3

    forwards = parse_forwards(
        __import__("os").environ.get("MOSS_FORWARDS", DEFAULT_FORWARDS)
    )
    remote_cmd = "bash -lc " + shlex.quote(cmd)
    args = cfg.ssh_base_args() + [
        "-tt",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=6",
        "-o", "ExitOnForwardFailure=yes",
    ]
    for local, remote in forwards:
        args += ["-L", f"127.0.0.1:{local}:127.0.0.1:{remote}"]
    args += [remote_cmd]

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

    print("[tunnel] " + ", ".join(f"{l}->{r}" for l, r in forwards), flush=True)
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
