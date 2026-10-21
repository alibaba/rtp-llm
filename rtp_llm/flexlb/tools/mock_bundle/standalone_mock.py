"""Run multiple logical engines in one Whale Mock pod, without a Master JVM."""

import ipaddress
import json
import os
from pathlib import Path
import shlex
import socket

from deployment import resolve


ROOT = Path(__file__).resolve().parent


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
    block_size = integer(env, "MOCK_BLOCK_SIZE", 1024)
    numeric = {
        "--n-prefill": prefill,
        "--n-decode": decode,
        "--base-grpc-port": port,
        "--block-size": block_size,
    }
    for role in ("PREFILL", "DECODE"):
        key = f"MOCK_{role}_BLOCK_SIZE"
        if key in env:
            numeric[f"--{role.lower()}-block-size"] = integer(env, key, block_size)
    for flag, key, default in (
        ("--prefill-kv-pool-blocks", "MOCK_PREFILL_KV_POOL_BLOCKS", 1024),
        ("--decode-kv-pool-blocks", "MOCK_DECODE_KV_POOL_BLOCKS", 1024),
        ("--decode-max-concurrency", "MOCK_DECODE_MAX_CONCURRENCY", 128),
        ("--event-loop-threads", "MOCK_EVENT_LOOP_THREADS", 4),
        ("--completion-threads", "MOCK_COMPLETION_THREADS", 4),
    ):
        numeric[flag] = integer(env, key, default)
    performance = json_object(
        env.get("MOCK_PERFORMANCE_CONFIG_JSON", (ROOT / "performance.json").read_text()),
        "MOCK_PERFORMANCE_CONFIG_JSON",
    )
    if "block_size" in performance and performance["block_size"] != block_size:
        raise ValueError("Performance block_size must match MOCK_BLOCK_SIZE")
    performance["block_size"] = block_size
    master = json_object(env.get("FLEXLB_CONFIG", "{}"), "FLEXLB_CONFIG")
    fetch = boolean(env, "FETCH_OUTPUT_STREAM", "1")
    kmonitor = boolean(env, "MOCK_KMONITOR_ENABLED", "true")
    compact = boolean(env, "MOCK_CACHE_STATUS_COMPACT_ENCODING", "false")
    runtime = Path(env.get("MOCK_BUNDLE_RUN_DIR", "/home/admin/ai-whale/mock"))
    java = str(Path(env.get("JAVA_HOME", "/opt/taobao/java")) / "bin/java")
    jar = Path(env.get("MOCK_BUNDLE_JAR_DIR", str(ROOT / "jars"))) / "mock.jar"
    args = [java, *shlex.split(env.get("MOCK_JAVA_OPTS", "-Xms512m -Xmx2g")),
            "-jar", str(jar), "--whale", "true", "--whale-bundle", str(not remote_decode).lower(),
            "--kmonitor", str(kmonitor).lower(), "--host", host,
            "--unique-engine-ips", "false", "--auto-fetch", str(not fetch).lower(),
            "--cache-status-compact-encoding", str(compact).lower()]
    for flag, value in numeric.items():
        args.extend([flag, str(value)])
    for flag, name in (("--endpoint-file", "endpoints.json"),
                       ("--discovery-file", "discovery.json"),
                       ("--performance", "performance.json"),
                       ("--master-config", "master-config.json")):
        if remote_decode and flag == "--discovery-file":
            continue
        args.extend([flag, str(runtime / name)])
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "performance.json").write_text(json.dumps(performance))
    (runtime / "master-config.json").write_text(json.dumps(master))
    plan.record(runtime, prefill=prefill, decode=decode, host=host, grpc_base_port=port,
                vip_grpc_endpoints=[
                    {"role": "prefill" if i < prefill else "decode", "ip": host, "port": port + i}
                    for i in range(prefill + decode)
                ])
    return args


def run():
    args = command(os.environ)
    # Replace the launcher so Java receives container termination signals directly.
    os.execv(args[0], args)


if __name__ == "__main__":
    run()
