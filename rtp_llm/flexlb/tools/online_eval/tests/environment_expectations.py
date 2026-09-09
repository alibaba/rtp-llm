"""Independent expected environment data for configuration regression tests."""

import json
from pathlib import Path
from types import SimpleNamespace

_DATA = Path(__file__).with_name("fixtures") / "environment_expectations.json"


def environment(name, context):
    profile = context if isinstance(context, str) else context.profile
    return SimpleNamespace(**configuration(name, profile))


def configuration(name, profile):
    return json.loads(_DATA.read_text())[f"{name}/{profile}"]
