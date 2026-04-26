"""Shared constants, exceptions, and HTTP/utility helpers."""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


PROJECT_ID = "2654816"
PIPELINE_ID = "1346"
BRANCH_REF = "main-internal"
CI_STATUS_URL = "https://get-tasend-back-twkvcdsbpj.cn-hangzhou-vpc.fcapp.run"
CI_TRIGGER_URL = "https://triggerid-to-mq-wjrdhcgbie.cn-hangzhou-vpc.fcapp.run"
GITHUB_API = "https://api.github.com"


class GateError(RuntimeError):
    """Expected operational error with a process exit code."""

    def __init__(self, message, exit_code=1):
        # type: (str, int) -> None
        super(GateError, self).__init__(message)
        self.exit_code = exit_code


def log(message):
    # type: (str) -> None
    print(message, flush=True)


def http_json(
    url,        # type: str
    headers=None,   # type: Optional[Dict[str, str]]
    payload=None,   # type: Optional[Dict[str, Any]]
    context="",     # type: str
    method=None,    # type: Optional[str]
):
    # type: (...) -> Tuple[int, Any, str]
    data = None
    request_headers = headers or {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers = dict({"Content-Type": "application/json"}, **request_headers)

    request = urllib.request.Request(url, data=data, headers=request_headers)
    if method is not None:
        request.method = method
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
            status = response.getcode()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except urllib.error.URLError as exc:
        raise GateError("::error::Network error during %s: %s" % (context, exc), 2)

    if not body:
        parsed = None  # type: Any
    else:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
    return status, parsed, body


def short_sha(sha):
    # type: (str) -> str
    return sha[:8] if sha else "unknown"


def write_output(name, value, output_file=""):
    # type: (str, str, str) -> None
    if output_file:
        with open(output_file, "a") as handle:
            handle.write("%s=%s\n" % (name, value))
    else:
        log("%s=%s" % (name, value))


def is_true(value):
    # type: (Any) -> bool
    if isinstance(value, bool):
        return value
    return str(value or "").lower() == "true"
