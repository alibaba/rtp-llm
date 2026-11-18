"""Worker evidence for cache deselection, captured after ordinary selection."""

import json
import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--rtp-cache-deselect-file", default=None)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_collection_modifyitems(config, items):
    yield
    path = config.getoption("--rtp-cache-deselect-file")
    if not path:
        return
    cached = set(Path(path).read_text().splitlines())
    selected = [item.nodeid for item in items]
    deselected = [item for item in items if item.nodeid in cached]
    report = config.getoption("xmlpath")
    if report:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "local")
        evidence = Path(f"{report}.collection.{worker}.json")
        evidence.parent.mkdir(parents=True, exist_ok=True)
        temporary = evidence.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"selected": selected, "cached": [item.nodeid for item in deselected]}
            )
        )
        temporary.replace(evidence)
    items[:] = [item for item in items if item.nodeid not in cached]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
