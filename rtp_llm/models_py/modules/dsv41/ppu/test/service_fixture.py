"""Health checks, wait deadlines and JSON output for native service tests."""

import json
import time
import urllib.error
import urllib.request


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def healthy(url):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + "/health", timeout=3) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def wait_for(predicate, process, deadline, description):
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"native server exited during {description}: {process.returncode}"
            )
        if predicate():
            return
        time.sleep(1)
    raise TimeoutError(description)
