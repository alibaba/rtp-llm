"""Run multiple logical engines in one Whale Mock pod, without a Master JVM."""

import ipaddress
import json
import os
from pathlib import Path
import shlex
import socket

from master_compat import mock_formula_config


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
    prefill = integer(env, "MOCK_PREFILL_COUNT", 1, minimum=0)
    decode = integer(env, "MOCK_DECODE_COUNT", 1, minimum=0)
    if prefill + decode == 0:
        raise ValueError("At least one Mock engine is required")
    port = integer(env, "MOCK_GRPC_BASE_PORT", 29130, minimum=2, maximum=65535)
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
    raw = env.get("FLEXLB_CONFIG", "{}")
    json_object(raw, "FLEXLB_CONFIG")
    master = json.loads(mock_formula_config(raw, False))
    fetch = boolean(env, "FETCH_OUTPUT_STREAM", "1")
    kmonitor = boolean(env, "MOCK_KMONITOR_ENABLED", "true")
    runtime = Path(env.get("MOCK_BUNDLE_RUN_DIR", "/home/admin/ai-whale/mock"))
    java = str(Path(env.get("JAVA_HOME", "/opt/taobao/java")) / "bin/java")
    jar = Path(env.get("MOCK_BUNDLE_JAR_DIR", str(ROOT / "jars"))) / "mock.jar"
    args = [java, *shlex.split(env.get("MOCK_JAVA_OPTS", "-Xms512m -Xmx2g")),
            "-jar", str(jar), "--whale", "true", "--whale-bundle", "true",
            "--kmonitor", str(kmonitor).lower(), "--host", host,
            "--unique-engine-ips", "false", "--auto-fetch", str(not fetch).lower()]
    for flag, value in numeric.items():
        args.extend([flag, str(value)])
    for flag, name in (("--endpoint-file", "endpoints.json"),
                       ("--discovery-file", "discovery.json"),
                       ("--performance", "performance.json"),
                       ("--master-config", "master-config.json")):
        args.extend([flag, str(runtime / name)])
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "performance.json").write_text(json.dumps(performance))
    (runtime / "master-config.json").write_text(json.dumps(master))
    return args


def run():
    args = command(os.environ)
    # Replace the launcher so Java receives container termination signals directly.
    os.execv(args[0], args)


if __name__ == "__main__":
    run()
