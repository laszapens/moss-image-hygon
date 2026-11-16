"""Shared connection config for the remote helpers.

Nothing host-specific lives in the repository. Supply the target machine via
environment variables (or ~/.ssh/config) so that cloning this repo never hands
out someone else's rented instance:

    MOSS_HOST      SSH hostname or Host alias            (required)
    MOSS_PORT      SSH port                              (default 22)
    MOSS_USER      SSH username                          (default: current $USER)
    MOSS_PASS_FILE file containing the password, mode 600 (optional)

If MOSS_PASS_FILE is set the helpers drive `ssh` through pexpect and type the
password. If it is not set they exec `ssh` directly, which lets your own
~/.ssh/config, agent, and key-based auth work normally — the preferred setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    """Raised when required connection settings are missing."""


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    port: str
    user: str
    pass_file: str | None

    @property
    def uses_password(self) -> bool:
        return bool(self.pass_file)

    def read_password(self) -> str:
        if not self.pass_file:
            raise ConfigError("MOSS_PASS_FILE is not set")
        path = Path(self.pass_file).expanduser()
        if not path.is_file():
            raise ConfigError(f"password file not found: {path}")
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise ConfigError(
                f"password file {path} is group/world accessible (mode {oct(mode)}); "
                f"run: chmod 600 {path}"
            )
        return path.read_text(encoding="utf-8").strip()

    def ssh_base_args(self) -> list[str]:
        return [
            "ssh",
            "-p", self.port,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=20",
            f"{self.user}@{self.host}",
        ]


def load_config() -> RemoteConfig:
    host = os.environ.get("MOSS_HOST", "").strip()
    if not host:
        raise ConfigError(
            "MOSS_HOST is not set. Export it, or add a Host entry to ~/.ssh/config.\n"
            "See .env.example for the full list of variables."
        )
    return RemoteConfig(
        host=host,
        port=os.environ.get("MOSS_PORT", "22").strip(),
        user=os.environ.get("MOSS_USER", os.environ.get("USER", "")).strip(),
        pass_file=(os.environ.get("MOSS_PASS_FILE") or None),
    )
