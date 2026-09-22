"""Environment isolation contract shared by Python and Bash launchers.

This is an inventory of names, not configuration values or defaults.
"""

from pathlib import Path

LOAD_CLIENT_ENV_VARS = (
    (Path(__file__).resolve().parents[3] / "config/load_client_env.txt").read_text().splitlines()
)
