"""Two Bazel remote shards rendezvous before running the native K3 PD smoke.

Shard 0 owns Prefill and requests; shard 1 owns Decode. The HTTP broker carries
only coordination metadata. Model traffic flows directly between the workers.
"""

import copy
import ipaddress
import json
import math
import os
import re
import signal
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROLES = ("prefill", "decode")
MAX_JSON_BYTES = 1024 * 1024
FAILURE_LOG_TAIL_BYTES = 64 * 1024
IDENTITY_KEYS = {
    "hostname",
    "gpu_uuids",
    "external_revision",
    "internal_revision",
    "checkpoint_identity",
    "draft_identity",
    "guard_sha256",
}


def _object(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Invalid {label} fields")


def _digest(value, length=40):
    if not isinstance(value, str) or not re.fullmatch(
        "[0-9a-f]{" + str(length) + "}", value
    ):
        raise ValueError("Invalid source or artifact digest")


def _hostname(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,253}", value):
        raise ValueError("Invalid hostname")


def _ip(value):
    if not isinstance(value, str):
        raise ValueError("Invalid worker IPv4 address")
    address = ipaddress.IPv4Address(value)
    if (
        address.is_loopback
        or address.is_unspecified
        or address.is_multicast
        or address.is_link_local
        or address.is_reserved
    ):
        raise ValueError("Worker needs a routable IPv4 address")
    return str(address)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Non-finite JSON number")


def decode_json(data):
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("Coordination JSON exceeds size limit")
    return json.loads(
        data, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant
    )


def validate_registration(value):
    _object(
        value,
        {"hostname", "ip", "external_revision", "internal_revision"},
        "registration",
    )
    _hostname(value["hostname"])
    _ip(value["ip"])
    for key in ("external_revision", "internal_revision"):
        _digest(value[key])


def _checkpoint(value):
    _object(value, {"config_sha256", "index_sha256", "shards"}, "checkpoint")
    _digest(value["config_sha256"], 64)
    if value["index_sha256"] is not None:
        _digest(value["index_sha256"], 64)
    shards = value["shards"]
    if not isinstance(shards, list) or not 1 <= len(shards) <= 4096:
        raise ValueError("Invalid checkpoint shard list")
    names = set()
    for shard in shards:
        if (
            not isinstance(shard, list)
            or len(shard) != 2
            or not isinstance(shard[0], str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,255}", shard[0])
            or type(shard[1]) is not int
            or shard[1] <= 0
            or shard[0] in names
        ):
            raise ValueError("Invalid or duplicate checkpoint shard")
        names.add(shard[0])


def validate_identity(value):
    _object(value, IDENTITY_KEYS, "prepared identity")
    _hostname(value["hostname"])
    for key in ("external_revision", "internal_revision"):
        _digest(value[key])
    _digest(value["guard_sha256"], 64)
    uuids = value["gpu_uuids"]
    if (
        not isinstance(uuids, list)
        or len(uuids) != 8
        or any(
            not isinstance(item, str)
            or not re.fullmatch(r"GPU-[A-Za-z0-9-]{1,128}", item)
            for item in uuids
        )
        or len(set(uuids)) != 8
    ):
        raise ValueError("Each role must identify eight distinct physical GPUs")
    _checkpoint(value["checkpoint_identity"])
    _checkpoint(value["draft_identity"])


def _outcome(value):
    _object(value, {"passed", "errors"}, "outcome")
    errors = value["errors"]
    if (
        type(value["passed"]) is not bool
        or not isinstance(errors, list)
        or len(errors) > 16
        or any(
            not isinstance(error, str) or not 1 <= len(error) <= 2048
            for error in errors
        )
        or value["passed"] == bool(errors)
    ):
        raise ValueError("Invalid outcome")


def validate_snapshot(state, run_id):
    _object(state, {"run_id", "deadline_unix_s", "roles", "error"}, "broker state")
    deadline = state["deadline_unix_s"]
    if (
        state["run_id"] != run_id
        or type(deadline) not in (int, float)
        or not math.isfinite(deadline)
        or deadline <= 0
    ):
        raise ValueError("Broker run identity or deadline mismatch")
    if state["error"] is not None and (
        not isinstance(state["error"], str) or not 1 <= len(state["error"]) <= 2048
    ):
        raise ValueError("Invalid broker error")
    roles = state["roles"]
    if not isinstance(roles, dict) or not set(roles) <= set(ROLES):
        raise ValueError("Invalid broker roles")
    for value in roles.values():
        _object(
            value,
            {"registration", "prepared", "ready", "result", "cleanup"},
            "role state",
        )
        validate_registration(value["registration"])
        if value["prepared"] is not None:
            validate_identity(value["prepared"])
        if type(value["ready"]) is not bool or (
            value["ready"] and value["prepared"] is None
        ):
            raise ValueError("Invalid role readiness")
        for key in ("result", "cleanup"):
            if value[key] is not None:
                _outcome(value[key])
    return state


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        # The run token must never follow a redirect to another endpoint.
        return None


class Broker:
    def __init__(self, url, token, run_id, timeout=5):
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Broker URL must be an HTTP(S) origin")
        if not re.fullmatch(r"k3-pd-[A-Za-z0-9]{1,64}", run_id):
            raise ValueError("Invalid broker run ID")
        if (
            not isinstance(token, str)
            or not 1 <= len(token) <= 4096
            or any(ord(character) < 33 or ord(character) > 126 for character in token)
        ):
            raise ValueError("Invalid broker token")
        self.url = url.rstrip("/") + "/v1/runs/" + run_id
        self.run_id, self.token, self.timeout = run_id, token, timeout
        self.deadline = None
        self.lock = threading.Lock()
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )

    def request(self, operation=None, role=None, data=None):
        body = None
        if operation is not None:
            if (
                operation
                not in (
                    "register",
                    "heartbeat",
                    "prepared",
                    "ready",
                    "result",
                    "cleanup",
                )
                or role not in ROLES
            ):
                raise ValueError("Invalid broker operation")
            body = json.dumps(
                {"op": operation, "role": role, "data": data}, allow_nan=False
            ).encode()
            if len(body) > MAX_JSON_BYTES:
                raise ValueError("Broker request exceeds size limit")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST" if body is not None else "GET",
        )
        try:
            with self.lock, self.opener.open(request, timeout=self.timeout) as response:
                if (
                    response.status != 200
                    or response.headers.get_content_type() != "application/json"
                ):
                    raise RuntimeError("Broker returned an invalid HTTP response")
                state = validate_snapshot(
                    decode_json(response.read(MAX_JSON_BYTES + 1)), self.run_id
                )
                if self.deadline is None:
                    self.deadline = state["deadline_unix_s"]
                elif state["deadline_unix_s"] != self.deadline:
                    raise ValueError("Broker changed the fixed run deadline")
                return state
        except (urllib.error.URLError, OSError) as error:
            # Do not echo URLs, headers, or response bodies containing credentials.
            raise RuntimeError("Broker connection failed") from error


def check_pair(state, config, prepared=False):
    if set(state["roles"]) != set(ROLES):
        return False
    registrations = [state["roles"][role]["registration"] for role in ROLES]
    if any(
        registrations[0][key] == registrations[1][key] for key in ("hostname", "ip")
    ):
        raise ValueError("PD shards must use distinct physical hosts and addresses")
    for registration in registrations:
        for key in ("external_revision", "internal_revision"):
            if registration[key] != config[key]:
                raise ValueError("PD shard source pins differ")
    if not prepared:
        return True
    identities = [state["roles"][role]["prepared"] for role in ROLES]
    if any(identity is None for identity in identities):
        return False
    for registration, identity in zip(registrations, identities):
        for key in ("hostname", "external_revision", "internal_revision"):
            if identity[key] != registration[key]:
                raise ValueError("Prepared worker identity differs from registration")
    if set(identities[0]["gpu_uuids"]) & set(identities[1]["gpu_uuids"]):
        raise ValueError("PD shards overlap physical GPUs")
    for key in ("checkpoint_identity", "draft_identity", "guard_sha256"):
        if identities[0][key] != identities[1][key]:
            raise ValueError("PD shards disagree on " + key)
    return True


def _failure(state):
    if state["error"]:
        return "Broker: " + state["error"]
    for role, record in state["roles"].items():
        for key in ("result", "cleanup"):
            if record[key] is not None and not record[key]["passed"]:
                return role + " " + key + ": " + "; ".join(record[key]["errors"])
    return None


class Coordination:
    def __init__(self, broker, role, disconnected, shard_status=None, interval=2):
        self.broker, self.role, self.disconnected = broker, role, disconnected
        self.shard_status, self.interval = shard_status, interval
        self.stop = threading.Event()
        self.error = None
        self.thread = None

    def touch_status(self):
        if self.shard_status is not None:
            self.shard_status.touch()

    def start(self):
        self.touch_status()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            try:
                state = self.broker.request("heartbeat", self.role, {})
                self.touch_status()
                reason = _failure(state)
                if reason or time.time() >= state["deadline_unix_s"]:
                    raise RuntimeError(reason or "Run deadline exceeded")
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"[:2048]
                self.disconnected.set()
                return

    def check(self):
        if self.disconnected.is_set():
            raise RuntimeError(self.error or "Shard interrupted")
        if time.time() >= self.broker.deadline:
            raise TimeoutError("Run deadline exceeded")

    def wait(self, predicate, timeout=None, tick=None, cleanup=False):
        deadline = time.monotonic() + (timeout if timeout is not None else 86400)
        while True:
            if not cleanup:
                self.check()
            state = self.broker.request()
            if not cleanup and _failure(state):
                raise RuntimeError(_failure(state))
            if predicate(state):
                return state
            if time.monotonic() >= deadline:
                raise TimeoutError("Peer coordination timed out")
            if tick is not None:
                tick()
            self.stop.wait(self.interval)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=self.broker.timeout + 1)


def _error(error):
    return f"{type(error).__name__}: {error}"[:2048]


def _print_failure_logs(task, role):
    if task is None:
        return
    for name in ("service.log", "requests.log"):
        path = task / name
        try:
            with path.open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - FAILURE_LOG_TAIL_BYTES))
                tail = stream.read(FAILURE_LOG_TAIL_BYTES).decode("utf-8", "replace")
        except FileNotFoundError:
            continue
        except OSError as error:
            tail = f"<unavailable: {type(error).__name__}>"
        token = os.environ.get("K3_PD_BROKER_TOKEN")
        if token:
            tail = tail.replace(token, "<redacted>")
        try:
            print(
                f"\nK3 {role} failure log: {path} "
                f"(last {FAILURE_LOG_TAIL_BYTES} bytes)\n{tail}",
                flush=True,
            )
        except (OSError, UnicodeError, ValueError):
            # Diagnostics must not replace the original test failure.
            pass


def run_shard(
    config,
    registration,
    broker,
    role,
    output,
    worker_factory,
    checker,
    shard_status=None,
    interval=2,
):
    """Run one native role; all owned processes are closed on every exit path."""
    disconnected = threading.Event()
    coordination = Coordination(broker, role, disconnected, shard_status, interval)
    worker = worker_factory(disconnected)
    errors, cleanup_errors = [], []
    registered = False
    result_sent = False
    paired_result = None
    try:
        validate_registration(registration)
        state = broker.request("register", role, registration)
        registered = True
        if not 0 < broker.deadline - time.time() <= 86400:
            raise ValueError("Invalid or expired run deadline")
        coordination.start()
        state = coordination.wait(lambda value: check_pair(value, config))
        paired = copy.deepcopy(config)
        for name in ROLES:
            paired[name]["ip"] = state["roles"][name]["registration"]["ip"]
        paired[role]["artifact_root"] = str(output)
        coordination.check()
        identity = worker.prepare_native(paired, role, broker.run_id)
        identity = {key: identity[key] for key in IDENTITY_KEYS}
        # Normalize tuples produced by checkpoint_identity to the wire schema.
        identity = decode_json(json.dumps(identity).encode())
        validate_identity(identity)
        broker.request("prepared", role, identity)
        coordination.wait(lambda value: check_pair(value, config, prepared=True))
        coordination.check()
        worker.start()
        deadline = time.monotonic() + config.get("startup_timeout_s", 3600)
        while not worker.status()["healthy"]:
            coordination.check()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"PD service startup timed out: role={role}; "
                    f"readiness={worker.task / 'readiness.json'}; "
                    f"service_log={worker.task / 'service.log'}"
                )
            disconnected.wait(interval)
        broker.request("ready", role, True)
        coordination.wait(
            lambda value: all(value["roles"][name]["ready"] for name in ROLES),
            tick=worker.status,
        )
        if role == "prefill":
            coordination.check()
            timeout = min(
                config.get("request_timeout_s", 7200),
                max(1, int(broker.deadline - time.time())),
            )
            result = worker.requests({"timeout_s": timeout})
            coordination.check()
            if result.get("requests_passed") is not True:
                raise RuntimeError("Full-model request runner did not pass")
            verdict = checker(worker.task / "requests" / "accuracy.json")
            (output / "result-check.json").write_text(
                json.dumps(verdict, indent=2) + "\n"
            )
            if verdict.get("passed") is not True:
                raise RuntimeError(
                    "Full-model request result check failed; "
                    f"report={output / 'result-check.json'}: "
                    + "; ".join(str(error) for error in verdict.get("errors", [])[:5])
                )
            coordination.check()
        else:
            coordination.wait(
                lambda value: value["roles"]["prefill"]["result"] is not None,
                tick=worker.status,
            )
        broker.request("result", role, {"passed": True, "errors": []})
        result_sent = True
        paired_result = coordination.wait(
            lambda value: all(
                value["roles"][name]["result"] is not None for name in ROLES
            )
        )
    except Exception as error:
        errors.append(_error(error))
        disconnected.set()
        if registered and not result_sent:
            try:
                broker.request("result", role, {"passed": False, "errors": errors})
            except Exception as report_error:
                errors.append("Report: " + _error(report_error))
    finally:
        try:
            worker.close()
        except Exception as error:
            cleanup_errors.append(_error(error))
        if registered:
            try:
                broker.request(
                    "cleanup",
                    role,
                    {"passed": not cleanup_errors, "errors": cleanup_errors},
                )
                paired_result = coordination.wait(
                    lambda value: all(
                        value["roles"].get(name, {}).get("cleanup") is not None
                        for name in ROLES
                    ),
                    timeout=120,
                    cleanup=True,
                )
                if _failure(paired_result):
                    errors.append(_failure(paired_result)[:2048])
            except Exception as error:
                errors.append("Cleanup coordination: " + _error(error))
        coordination.close()
    if errors or cleanup_errors:
        _print_failure_logs(worker.task, role)
    summary = {
        "passed": not errors and not cleanup_errors,
        "role": role,
        "run_id": broker.run_id,
        "errors": errors,
        "cleanup_errors": cleanup_errors,
        "paired_state": paired_result,
        "validation_scope": "fixed-full93-text-request-contracts",
        "kernel_paths_verified": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def resolve_input(value, environment):
    if not value:
        raise ValueError("Missing native PD input path")
    path = Path(value)
    if path.is_absolute():
        candidates = [path]
    else:
        if ".." in path.parts:
            raise ValueError("Input path cannot escape runfiles")
        candidates = []
        for key in ("RUNFILES_DIR", "TEST_SRCDIR"):
            if environment.get(key):
                root = Path(environment[key])
                candidates.extend(
                    [
                        root / path,
                        root / environment.get("TEST_WORKSPACE", "rtp_llm") / path,
                    ]
                )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError("Native PD input is absent from runfiles")


def advertise_ip(broker_url, override=None):
    if override:
        return _ip(override)
    endpoint = urllib.parse.urlsplit(broker_url)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect(
            (
                endpoint.hostname,
                endpoint.port or (443 if endpoint.scheme == "https" else 80),
            )
        )
        return _ip(route.getsockname()[0])


def main():
    from result_check import check_result
    from worker import Worker

    environment = os.environ
    if environment.get("TEST_TOTAL_SHARDS") != "2" or environment.get(
        "TEST_SHARD_INDEX"
    ) not in ("0", "1"):
        raise ValueError("Native remote PD smoke requires exactly two Bazel shards")
    role = ROLES[int(environment["TEST_SHARD_INDEX"])]
    status = Path(environment["TEST_SHARD_STATUS_FILE"])
    status.touch()
    config = decode_json(
        resolve_input(environment.get("K3_PD_CONFIG"), environment).read_bytes()
    )
    for key in ("external_revision", "internal_revision"):
        _digest(config[key])
    broker = Broker(
        environment["K3_PD_BROKER_URL"],
        environment["K3_PD_BROKER_TOKEN"],
        environment["K3_PD_RUN_ID"],
    )
    registration = {
        "hostname": socket.gethostname(),
        "ip": advertise_ip(
            environment["K3_PD_BROKER_URL"], environment.get("K3_PD_ADVERTISE_IP")
        ),
        "external_revision": config["external_revision"],
        "internal_revision": config["internal_revision"],
    }
    output = Path(environment["TEST_UNDECLARED_OUTPUTS_DIR"]) / "kimi-k3-pd"
    output.mkdir(exist_ok=False)

    def interrupted(signum, frame):
        raise RuntimeError(f"Remote PD shard received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    summary = run_shard(
        config,
        registration,
        broker,
        role,
        output,
        Worker,
        check_result,
        status,
    )
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "role": role,
                "errors": summary["errors"],
                "cleanup_errors": summary["cleanup_errors"],
            }
        )
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
