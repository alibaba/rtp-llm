"""HTTP control requests, bounded waits and socket probes."""

from __future__ import annotations

import json
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Optional


def http_get_json(url: str, timeout: float = 10.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def http_post_json(
    url: str, body: dict, timeout: float = 10.0
) -> tuple[int, Optional[dict]]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(payload)
            except (ValueError, json.JSONDecodeError):
                return resp.status, None
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            return exc.code, json.loads(payload)
        except (ValueError, json.JSONDecodeError):
            return exc.code, None
    except Exception as exc:
        return 0, {"error": repr(exc)}


def wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    last_error: Optional[Exception] = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    print(f"[harness] timeout waiting for {host}:{port}: {last_error}", file=sys.stderr)
    return False


def wait_for(
    predicate: Callable[[], bool],
    timeout_s: float,
    interval_s: float = 0.5,
    desc: str = "condition",
) -> bool:
    """Poll *predicate* until it returns True or *timeout_s* elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval_s)
    return False


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return False
        except OSError:
            return True


def port_accepting(port: int, host: str = "127.0.0.1") -> bool:
    """Post-start liveness probe; SO_REUSEADDR bind probes can miss listeners on macOS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


# JVM servers bind all interfaces; a pre-start loopback-only probe misses
# listeners on other interfaces. Post-start connection probes use loopback.
PROBE_BIND_HOST = "0.0.0.0"


def master_url(host, port, endpoint):
    """One definition of Master control routes; transport failures remain explicit."""
    routes = {"info": "master/info", "inflight": "inflight_status"}
    return f"http://{host}:{port}/rtp_llm/{routes[endpoint]}"
