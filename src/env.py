"""Read settings and API keys from the environment or the project's .env file.

Keys are never hard-coded and never written anywhere by this module. A real
environment variable takes precedence over the .env file.
"""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def read_env_file(path=ENV_FILE):
    """KEY=VALUE pairs from a .env file (comments, blank lines and quotes handled)."""
    values = {}
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        values[key] = value.strip().strip("'\"")
    return values


def get_setting(name, env_file=ENV_FILE):
    """The value of `name` from the environment, else from .env; None if unset or blank."""
    value = os.environ.get(name) or read_env_file(env_file).get(name)
    return value.strip() if value and value.strip() else None
