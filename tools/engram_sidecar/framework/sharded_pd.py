"""Four-host experiment: two independent CP4 P instances and one DPEP8 D.

Each role group uses two real Engram shards on distinct hosts. This launcher
does not schedule containers or elect owners. It renders explicit, reviewable
RTP settings and refuses malformed topology before loading CUDA.
"""
import argparse
import hashlib
import ipaddress
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tools.engram_sidecar.framework import full_model_http

NODES = ("p0", "p1", "d0", "d1")


def validate_topology(config):
    if set(config) != {"schema", "nodes"} or config["schema"] != 1:
        raise ValueError("expected schema=1 and nodes")
    if set(config["nodes"]) != set(NODES):
        raise ValueError("require p0, p1, d0, d1")
    hosts, devices = set(), set()
    for name in NODES:
        node = config["nodes"][name]
        if set(node) != {"ip", "port", "gpu_uuids"}:
            raise ValueError("node requires ip, port and gpu_uuids only")
        ipaddress.IPv4Address(node["ip"])
        if node["ip"] in hosts:
            raise ValueError("each node must be on a different host")
        hosts.add(node["ip"])
        if type(node["port"]) is not int or not 1024 <= node["port"] <= 65000:
            raise ValueError("invalid base port")
        uuids = node["gpu_uuids"]
        if not isinstance(uuids, list) or len(uuids) != 4:
            raise ValueError("exactly four GPU UUIDs per host required")
        for uuid in uuids:
            if not isinstance(uuid, str) or not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid):
                raise ValueError("invalid GPU UUID")
            if uuid in devices:
                raise ValueError("GPU repeated in topology")
            devices.add(uuid)
    if config["nodes"]["d0"]["port"] != config["nodes"]["d1"]["port"]:
        raise ValueError("decode gang requires the same base port on both hosts")
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def launch_environment(config, node_name, args, inherited=None):
    digest = validate_topology(config)
    if node_name not in NODES:
        raise ValueError("unknown node")
    nodes = config["nodes"]
    node = nodes[node_name]
    is_prefill = node_name.startswith("p")
    env = full_model_http.launch_environment(args, inherited=inherited)
    # Clear stale gang settings even for single-host prefill. Never inherit a
    # previous experiment's rank or CP layout from the shell.
    env.pop("GANG_CONFIG_STRING", None)
    env.pop("GANG_CONFIG_PATH", None)
    strategy = "mega" if is_prefill else getattr(args, "decode_moe_strategy", "mega")
    env.update({
        "ROLE_TYPE": "PREFILL" if is_prefill else "DECODE",
        "CUDA_VISIBLE_DEVICES": ",".join(node["gpu_uuids"]),
        "START_PORT": str(node["port"]),
        "WORLD_SIZE": "4" if is_prefill else "8",
        "LOCAL_WORLD_SIZE": "4",
        "WORLD_RANK": "4" if node_name == "d1" else "0",
        "TP_SIZE": "4" if is_prefill else "1",
        "DP_SIZE": "1" if is_prefill else "8",
        "EP_SIZE": "4" if is_prefill else "8",
        "PREFILL_CP_SIZE": "4",
        "PREFILL_CP_KV_CACHE_SHARDED": "1",
        "SEQ_SIZE_PER_BLOCK": "256",
        "KERNEL_SEQ_SIZE_PER_BLOCK": "128",
        "CP_ROTATE_METHOD": "ALL_GATHER" if is_prefill else "PREFILL_CP",
        # Parallel topology does not choose the MoE kernel implementation.
        "DSV4_MOE_STRATEGY": strategy,
        # The engine initializes communication before Python model selection.
        # Keep both layers aligned so Mega does not initialize unused DeepEP.
        "MOE_STRATEGY": "mega_moe" if strategy == "mega" else "auto",
        "DSV41_CED": "1",
        "DSV41_SWA_BOUNDED_REPLAY": "1",
        "CACHE_STORE_RDMA_MODE": "1",
        "LOAD_CACHE_TIMEOUT_MS": "120000",
        "CONCURRENCY_LIMIT": "8",
        "RTP_ENGRAM_TOPOLOGY_SHA256": digest,
    })
    if is_prefill:
        env["ENABLE_CUDA_GRAPH"] = "0"
        env["DECODE_CAPTURE_CONFIG"] = ""
        if args.storage_mode == "external-egm":
            env["RTP_ENGRAM_GRAPH_MODE"] = "eager"
    else:
        env["GANG_CONFIG_STRING"] = ";".join(
            f"name:decode_part{i},ip:{nodes[f'd{i}']['ip']},port:{node['port']}"
            for i in range(2)
        )
    peer = nodes["d0"] if is_prefill else nodes["p0"]
    env["REMOTE_RPC_SERVER_IP"] = peer["ip"]
    env["REMOTE_SERVER_PORT"] = str(peer["port"])
    # Each P frontend is bound to its local P backend for unambiguous coverage.
    # Decode is one gang; only its coordinator is advertised for discovery.
    p_names = [node_name] if is_prefill else ["p0", "p1"]
    def endpoint(names):
        return {"type": "Vipserver", "protocol": "http", "path": "/",
                "address": ",".join(f"{nodes[n]['ip']}:{nodes[n]['port']}" for n in names)}
    env["MODEL_SERVICE_CONFIG"] = json.dumps({
        "service_id": "engram_sharded_pd", "use_local": True,
        "role_endpoints": [{"group": "default", "prefill_endpoint": endpoint(p_names),
                            "decode_endpoint": endpoint(["d0"])}],
    })
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--node", choices=NODES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--storage-mode", choices=full_model_http.STORAGE_MODES, default="external-egm")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-allocation-bytes", type=int, default=128 * 1024**3)
    parser.add_argument("--budget-bytes", type=int, default=256 * 1024**3)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--decode-moe-strategy", choices=("mega", "deepep"), default="mega")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    topology = json.loads(args.topology.read_text())
    validate_topology(topology)
    args.port = topology["nodes"][args.node]["port"]
    env = launch_environment(topology, args.node, args)
    keys = ("ROLE_TYPE", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "WORLD_RANK", "TP_SIZE",
            "DP_SIZE", "EP_SIZE", "PREFILL_CP_SIZE", "PREFILL_CP_KV_CACHE_SHARDED",
            "SEQ_SIZE_PER_BLOCK", "CACHE_STORE_RDMA_MODE", "ENABLE_CUDA_GRAPH", "DSV4_MOE_STRATEGY",
            "RTP_ENGRAM_TOPOLOGY_SHA256")
    print(json.dumps({k: env[k] for k in keys}, sort_keys=True), flush=True)
    if args.dry_run:
        return 0
    available = set(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=15).split())
    if not set(topology["nodes"][args.node]["gpu_uuids"]).issubset(available):
        raise RuntimeError("topology does not match this host's GPUs")
    # Native/service imports initialize relative log paths. Match the standard
    # launcher instead of relying on a writable container default directory.
    os.chdir(full_model_http.REPO)
    # Fresh exec below loads this same environment again. Check the loader in a
    # CUDA-hidden child first, before mapping Engram or loading the backbone.
    native_check = (
        "import rtp_llm.ops as ops; ops.ensure_engine_ops_loaded(); "
        "import librtp_compute_ops; "
        "from rtp_llm.async_decoder_engine.rpc_engine import LanguageCppEngine; "
        "from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model; "
        "from tools.engram_sidecar.framework.native_runtime import assert_single_compute_library; "
        "print('NATIVE_COMPUTE_SINGLETON_OK', assert_single_compute_library())"
    )
    subprocess.run([sys.executable, "-c", native_check],
                   env={**env, "CUDA_VISIBLE_DEVICES": ""}, check=True, timeout=180)
    full_model_http.checkpoint_preflight(args.checkpoint, args.storage_mode)
    if args.storage_mode == "external-egm":
        args.world_size = int(env["WORLD_SIZE"])
        full_model_http.cohort_preflight(args)
    os.execve(sys.executable, [sys.executable, "-m", "rtp_llm.start_server"], env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
