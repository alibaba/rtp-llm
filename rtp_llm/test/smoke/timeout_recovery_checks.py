"""Check timeout and recovery evidence from one running model server's logs.

This deliberately does not infer I/O completion from a successful HTTP request,
or infer an SDK timeout from its configured timeout value. Frozen records must
be paired with the production recovery worker's proof for the same execution.
"""

import json
import math
import re


_TIMEOUT_PATTERNS = {
    "local_business_queue": re.compile(
        r"(?:load|store) expired in business queue|store business queue timeout"
    ),
    "local_workloop_queue": re.compile(r"transfer expired in WorkLoop queue"),
    "local_worker_queue": re.compile(r"transfer expired in TE worker queue"),
    "local_resource_wait": re.compile(
        r"staging admission timed out|transfer resource wait timed out"
    ),
    "local_pre_submit": re.compile(
        r"transfer (?:deadline exceeded before (?:TE submission|dispatch|multi-rank submission)"
        r"|expired before (?:WorkLoop admission|stage submission))"
    ),
    "kvcm_pre_io": re.compile(r"storage read deadline exceeded before I/O"),
    "kvcm_payload_pre_io": re.compile(r"KVCM READ payload expired before I/O:"),
    # This upper-layer deadline shares the KVCM task budget. It is separate
    # evidence and must not stand in for a backend queue or RPC timeout.
    "kvcm_match_to_ready": re.compile(
        r"block tree load failed,[^\r\n]+error=match-to-ready deadline exceeded"
    ),
    "kvcm_task_deadline": re.compile(
        r"KVCM task deadline exceeded: [^\r\n]+|KVCM task expired before rank submission"
    ),
    "kvcm_sdk_timeout": re.compile(
        r"kvcm client loadKvCaches fail, ec \[(?:100|119)\]"
        r"|local file sdk timeout abort: op=get\b"
    ),
    "kvcm_rpc_timeout": re.compile(r"KVCM broadcast timed out:"),
}
_FROZEN_PREFIXES = {
    "transfer frozen:": "transfer",
    "KVCM READ frozen:": "kvcm_read",
}
_RELEASE_PREFIX = "FrozenTask recovery released"
_EXECUTION_BACKEND = {"transfer": "local", "kvcm_read": "kvcm"}


class TimeoutRecoveryCheckError(AssertionError):
    def __init__(self, report):
        self.report = report
        super().__init__("; ".join(report["errors"]))


def _parts(body):
    """Split top-level fields in FrozenTask::debugString, respecting quoted text.

    Nested descriptors/errors can contain text such as ``all_safe=true``. They
    must never stand in for fields of the actual completion-query object.
    """
    parts = []
    start = 0
    stack = []
    quoted = escaped = False
    for index, char in enumerate(body):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or (stack.pop(), char) not in (("[", "]"), ("{", "}")):
                raise ValueError("unbalanced record")
        elif char == "," and not stack:
            parts.append(body[start:index])
            start = index + 1
    if quoted or escaped or stack:
        raise ValueError("truncated record")
    parts.append(body[start:])
    return parts


def _fields(body):
    result = {}
    for part in _parts(body):
        key, separator, value = part.strip().partition("=")
        if not separator or not key or key in result:
            raise ValueError("invalid or duplicate record field")
        result[key] = value
    return result


def _value(fields, name):
    value = fields[name]
    if value.startswith('"'):
        return json.loads(value)
    if value == "true":
        return True
    if value == "false":
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def _objects(value):
    if not value.startswith("[") or not value.endswith("]"):
        raise ValueError("invalid object list")
    for item in _parts(value[1:-1]) if value[1:-1] else []:
        if not item.startswith("{") or not item.endswith("}"):
            raise ValueError("invalid object")
        yield _fields(item[1:-1])


def _record(line):
    start = line.index("FrozenTask{") + len("FrozenTask{")
    # debugString closes at the end of the log message, before the newline.
    body = line[start:].strip()
    if not body.endswith("}"):
        raise ValueError("truncated FrozenTask")
    fields = _fields(body[:-1])
    execution_type = _value(fields, "execution_type")
    if execution_type not in _EXECUTION_BACKEND:
        raise ValueError("unknown execution_type")
    trace = _value(fields, "trace")
    if execution_type == "transfer":
        if type(trace) is not int or trace <= 0:
            raise ValueError("transfer trace must be positive")
    elif not isinstance(trace, str) or not trace:
        raise ValueError("KVCM trace must be nonempty")
    count = _value(fields, "target_block_count")
    if type(count) is not int or count <= 0:
        raise ValueError("target_block_count must be positive")
    result = {
        "execution_type": execution_type,
        "trace": trace,
        "target_block_count": count,
        "state": _value(fields, "state"),
        "failure_reason": _value(fields, "failure_reason"),
    }
    result["operation"] = _value(fields, "operation")
    result["directions"] = []
    result["target_pools"] = []
    for descriptor in _objects(fields.get("descriptors", "[]")):
        if "source_tier" not in descriptor or "target_tier" not in descriptor:
            continue
        direction = [descriptor["source_tier"], descriptor["target_tier"]]
        if direction not in result["directions"]:
            result["directions"].append(direction)
        for target in _objects(descriptor.get("targets", "[]")):
            pool = _value(target, "pool")
            if pool not in result["target_pools"]:
                result["target_pools"].append(pool)
    for name in ("frozen_at_us", "released_at_us", "expected_rank_count"):
        if name in fields:
            result[name] = _value(fields, name)
    # Local MultiRankBlockTransferEngine currently leaves failure_reason empty;
    # its original per-rank gRPC status is the authoritative timeout evidence.
    # Never search nested quoted error strings for a fabricated grpc_code=4.
    deadline_ranks = []
    snapshots = fields.get("rank_snapshots", "unknown")
    if snapshots != "unknown":
        if not snapshots.startswith("[") or not snapshots.endswith("]"):
            raise ValueError("invalid rank snapshots")
        for snapshot in _parts(snapshots[1:-1]) if snapshots[1:-1] else []:
            if not snapshot.startswith("{") or not snapshot.endswith("}"):
                raise ValueError("invalid rank snapshot")
            rank = _fields(snapshot[1:-1])
            code = _value(rank, "grpc_code")
            if type(code) is int and code == 4:
                deadline_ranks.append(_value(rank, "rank"))
    result["rpc_deadline_ranks"] = deadline_ranks
    if result["state"] == "RELEASED":
        query_count = _value(fields, "recovery_query_count")
        if type(query_count) is not int or query_count <= 0:
            raise ValueError("release has no recovery query")
        query = fields["last_recovery_query"]
        if not query.startswith("{") or not query.endswith("}"):
            raise ValueError("release has no completion query proof")
        proof = _fields(query[1:-1])
        if _value(proof, "all_safe") is not True:
            raise ValueError("release query does not prove all_safe=true")
        proof_trace = _value(proof, "trace")
        if type(proof_trace) is not type(trace) or proof_trace != trace:
            raise ValueError("release query trace does not match execution")
        result["recovery_query_count"] = query_count
        result["all_safe"] = True
    return result


def analyze_timeout_recovery(
    text, *, backend, require_frozen=False, required_evidence=()
):
    """Return a JSON-serializable report; incomplete recovery is a failed report.

    ``required_evidence`` lists categories that must all be observed. With no
    explicit categories, any real timeout of the selected backend is required.
    Pass the entire log window from one server lifetime, including freeze lines.
    Repeated copies of identical log records are harmless.
    """
    if backend not in ("local", "kvcm"):
        raise ValueError("backend must be local or kvcm")
    known = set(_TIMEOUT_PATTERNS) | {"local_rpc_timeout"}
    if set(required_evidence) - known:
        raise ValueError("unknown timeout evidence category")
    if any(not name.startswith(backend + "_") for name in required_evidence):
        raise ValueError("required evidence belongs to another backend")
    evidence = {name: [] for name in known}
    errors = []
    frozen = {}
    released = {}
    for number, line in enumerate(text.splitlines(), 1):
        for name, pattern in _TIMEOUT_PATTERNS.items():
            if pattern.search(line):
                evidence[name].append(number)
        expected_type = next(
            (kind for prefix, kind in _FROZEN_PREFIXES.items() if prefix in line),
            None,
        )
        is_release = _RELEASE_PREFIX in line
        if expected_type is None and not is_release:
            continue
        try:
            record = _record(line)
            state = "RELEASED" if is_release else "FROZEN"
            if record["state"] != state:
                raise ValueError(f"expected state={state}")
            if expected_type and record["execution_type"] != expected_type:
                raise ValueError("freeze prefix disagrees with execution_type")
            key = (record["execution_type"], record["trace"])
            target = released if is_release else frozen
            if key in target and target[key] != record:
                raise ValueError("conflicting duplicate execution record")
            target[key] = record
            if record["rpc_deadline_ranks"]:
                kind = _EXECUTION_BACKEND[record["execution_type"]]
                evidence[kind + "_rpc_timeout"].append(number)
            if record["failure_reason"] in ("rpc_timeout", "sdk_timeout"):
                kind = _EXECUTION_BACKEND[record["execution_type"]]
                category = kind + "_" + record["failure_reason"]
                if category in evidence:
                    evidence[category].append(number)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"line {number}: invalid recovery evidence: {error}")

    for key, record in released.items():
        original = frozen.get(key)
        if original is None:
            errors.append(f"release has no matching frozen execution: {key}")
            continue
        for field in (
            "target_block_count",
            "failure_reason",
            "frozen_at_us",
            "expected_rank_count",
            "operation",
            "directions",
            "target_pools",
        ):
            if original.get(field) != record.get(field):
                errors.append(f"release changed {field} for {key}")
    pending = [record for key, record in frozen.items() if key not in released]
    if pending:
        errors.append(f"{len(pending)} frozen execution(s) have no verified release")
    matching_frozen = [
        record
        for record in frozen.values()
        if _EXECUTION_BACKEND[record["execution_type"]] == backend
    ]
    if require_frozen and not matching_frozen:
        errors.append(f"no {backend} frozen execution was observed")
    observed = {name: sorted(set(lines)) for name, lines in evidence.items() if lines}
    if not any(name.startswith(backend + "_") for name in observed):
        errors.append(f"no {backend} timeout was observed")
    for name in required_evidence:
        if name not in observed:
            errors.append(f"required timeout evidence was not observed: {name}")
    return {
        "passed": not errors,
        "backend": backend,
        "errors": errors,
        "evidence": dict(sorted(observed.items())),
        "frozen": list(frozen.values()),
        "released": list(released.values()),
        "pending": pending,
    }


def d2h_timeout_records(report):
    """只接受同一 eviction execution 的 D2H 方向与真实 rank deadline。"""
    return [
        item
        for item in report["frozen"]
        if item["execution_type"] == "transfer"
        and item["operation"] == "evict"
        and item["directions"] == [["DEVICE", "HOST"]]
        and item["rpc_deadline_ranks"]
        and item.get("expected_rank_count") == 2
        and item["target_pools"]
    ]


def frozen_host_metric_samples(text, *, after_ms):
    """读取真实 kmonitor gauge；缺样本、旧样本均不能视为零。"""
    samples = {}
    pattern = re.compile(
        r"\brtp_llm_kv_cache_frozen_ref_blocks\s+(\d+)\s+(\S+)([^\r\n]*)"
    )
    for match in pattern.finditer(text):
        timestamp = int(match[1])
        tags = dict(re.findall(r"\b(\w+)=([^\s,]+)", match[3]))
        if timestamp < after_ms or tags.get("tier") != "HOST":
            continue
        pool = tags.get("pool_name")
        if not pool:
            continue
        # kmonitor gauge 的布局为 avg_sum_max_min_count。
        values = [float(value) for value in match[2].split("_")]
        if len(values) != 5 or not all(math.isfinite(value) for value in values):
            raise ValueError("invalid frozen-block gauge")
        if values[4] <= 0:
            raise ValueError("empty frozen-block gauge")
        sample = {"timestamp_ms": timestamp, "maximum": values[2]}
        previous = samples.get(pool)
        if previous is None or timestamp > previous["timestamp_ms"]:
            samples[pool] = sample
        elif timestamp == previous["timestamp_ms"]:
            previous["maximum"] = max(previous["maximum"], sample["maximum"])
    return samples


def check_timeout_recovery(text, **kwargs):
    report = analyze_timeout_recovery(text, **kwargs)
    if not report["passed"]:
        raise TimeoutRecoveryCheckError(report)
    return report
