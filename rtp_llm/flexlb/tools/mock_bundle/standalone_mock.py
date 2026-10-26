"""Run multiple logical engines in one Whale Mock pod, without a Master JVM."""

import ipaddress
import json
import os
import shlex
import socket
import sys
from pathlib import Path

from deployment import resolve

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "online_eval" / "src"))
from adapters.mock_launch import MockLaunchSpec


def integer(env, key, default, minimum=1, maximum=None):
    raw = env.get(key, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{key} must be an integer") from error
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{key} is out of range")
    return value


def boolean(env, key, default):
    raw = env.get(key, default).lower()
    if raw not in ("0", "1", "false", "true"):
        raise ValueError(f"{key} must be 0, 1, false or true")
    return raw in ("1", "true")


def json_object(raw, name):
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def command(env):
    if env.get("RTP_LLM_MOCK_BUNDLE") != "1":
        raise ValueError("RTP_LLM_MOCK_BUNDLE=1 is required for test-only Mock")
    plan = resolve(env, "mock")
    role = env.get("ROLE_TYPE")
    if role is not None and (plan.mode != "per_engine" or role not in ("PREFILL", "DECODE")):
        raise ValueError("ROLE_TYPE is PREFILL or DECODE and only applies to per_engine")
    prefill = integer(env, "MOCK_PREFILL_COUNT", int(role != "DECODE"), minimum=0)
    decode = integer(env, "MOCK_DECODE_COUNT", int(role != "PREFILL"), minimum=0)
    if role and (prefill, decode) != ((1, 0) if role == "PREFILL" else (0, 1)):
        raise ValueError("ROLE_TYPE conflicts with Mock engine counts")
    if prefill + decode == 0:
        raise ValueError("At least one Mock engine is required")
    remote_decode = plan.remote_decode
    if remote_decode and prefill + decode != 1:
        raise ValueError("per_engine requires exactly one engine per Pod")
    http_port = integer(env, "START_PORT", 29129, minimum=1, maximum=65534)
    port = integer(env, "MOCK_GRPC_BASE_PORT", http_port + 1, minimum=2, maximum=65535)
    if "START_PORT" in env and port != http_port + 1:
        raise ValueError("MOCK_GRPC_BASE_PORT must equal START_PORT + 1")
    if port + prefill + decode - 1 > 65535:
        raise ValueError("Mock engine ports exceed 65535")
    host = env.get("POD_IP") or socket.gethostbyname(socket.gethostname())
    address = ipaddress.ip_address(host)
    if address.is_loopback or address.is_unspecified or address.version != 4:
        raise ValueError("POD_IP must be a routable IPv4 pod address")
    numeric = dict(n_prefill=prefill, n_decode=decode, base_grpc_port=port)
    for flag, key, default in (
        ("--mock.event_loop_threads", "MOCK_EVENT_LOOP_THREADS", 4),
        ("--mock.completion_threads", "MOCK_COMPLETION_THREADS", 4),
    ):
        numeric[flag.removeprefix("--mock.")] = integer(env, key, default)
    from runtime_configs import calibration_input, engine_input

    engine = engine_input(env)
    performance = calibration_input(env)
    master = json_object(env.get("FLEXLB_CONFIG", "{}"), "FLEXLB_CONFIG")
    fetch = boolean(env, "FETCH_OUTPUT_STREAM", "1")
    kmonitor = boolean(env, "MOCK_KMONITOR_ENABLED", "true")
    compact = boolean(env, "MOCK_CACHE_STATUS_COMPACT_ENCODING", "false")
    runtime = Path(env.get("MOCK_BUNDLE_RUN_DIR", "/home/admin/ai-whale/mock"))
    java = str(Path(env.get("JAVA_HOME", "/opt/taobao/java")) / "bin/java")
    jar = Path(env.get("MOCK_BUNDLE_JAR_DIR", str(ROOT / "jars"))) / "mock.jar"
    args = MockLaunchSpec(
        java=java,
        jar=jar,
        jvm_options=tuple(shlex.split(env.get("MOCK_JAVA_OPTS", "-Xms512m -Xmx2g"))),
        whale=True,
        whale_bundle=not remote_decode,
        kmonitor=kmonitor,
        host=host,
        unique_engine_ips=False,
        auto_fetch=not fetch,
        cache_status_compact_encoding=compact,
        endpoint_file=runtime / "endpoints.json",
        discovery_file=None if remote_decode else runtime / "discovery.json",
        calibration=runtime / "calibration.json",
        engine_config=runtime / "engine-config.json",
        **numeric,
    ).argv()
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "calibration.json").write_text(json.dumps(performance))
    (runtime / "engine-config.json").write_text(json.dumps(engine))
    (runtime / "master-config.json").write_text(json.dumps(master))
    plan.record(
        runtime,
        prefill=prefill,
        decode=decode,
        host=host,
        grpc_base_port=port,
        vip_grpc_endpoints=[
            {"role": "prefill" if i < prefill else "decode", "ip": host, "port": port + i} for i in range(prefill + decode)
        ],
    )
    return args


def run():
    args = command(os.environ)
    # Replace the launcher so Java receives container termination signals directly.
    os.execv(args[0], args)


if __name__ == "__main__":
    run()
