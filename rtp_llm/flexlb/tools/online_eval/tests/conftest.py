"""Propagate pytest's configured import roots to real child interpreters."""
import os


def pytest_configure(config):
    paths = [str(path) for path in config.getini("pythonpath")]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
