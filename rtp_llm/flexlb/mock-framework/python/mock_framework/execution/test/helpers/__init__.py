"""Test-only paths and replay fixture helpers."""
from pathlib import Path

import mock_framework

FRAMEWORK = Path(mock_framework.__file__).resolve().parents[2]
