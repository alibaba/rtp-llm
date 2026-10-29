"""Launch one BF16-compute PD endpoint; FP8 operands are explicit options.

Run the declared server binary inside one Bazel remote action. The paired
worker owns GPU allocation, output directories, and process cleanup.
"""

import argparse
import csv
import json
import math
import os
import re
import socket
import stat
import subprocess
import sys
from pathlib import Path


def launch_config(args):
    fp8_gemm = bool(getattr(args, "fp8_gemm", False))
    fp8_kv_cache = bool(getattr(args, "fp8_kv_cache", False))
    warm_up = bool(getattr(args, "warm_up", False))
    model_warm_up = bool(getattr(args, "model_warm_up", False))
    if model_warm_up and not warm_up:
        raise ValueError("model warmup requires --warm-up")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    draft = Path(args.draft_checkpoint).resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    text = config.get("text_config", config)
    if text.get("num_hidden_layers") != 93:
        raise ValueError("Bazel native smoke requires the complete 93-layer target")
    orthogonal_smoke = bool(getattr(args, "orthogonal_smoke", False))
    tp_size = int(getattr(args, "tp_size", 8))
    dp_size = int(getattr(args, "dp_size", 1))
    ep_size = int(getattr(args, "ep_size", tp_size * dp_size))
    world_size = tp_size * dp_size
    source_tp_size = int(getattr(args, "prefill_source_tp_size", 8))
    if tp_size < 1 or dp_size < 1 or ep_size != world_size or source_tp_size < 1:
        raise ValueError("PD topology requires positive TP/DP and EP=TP×DP")
    if orthogonal_smoke and args.role == "PREFILL" and dp_size != 1:
        raise ValueError("Orthogonal Prefill profile requires DP1")
    if (
        orthogonal_smoke
        and args.role == "DECODE"
        and (source_tp_size % tp_size and tp_size % source_tp_size)
    ):
        raise ValueError("PageRR PD attention TP sizes must divide one another")
    for port in (args.start_port, args.peer_port):
        if port < 1024 or port + world_size * 9 > 65535:
            raise ValueError("Invalid service port range")
    reserve_runtime_mem_mb = getattr(args, "reserve_runtime_mem_mb", 14336)
    if reserve_runtime_mem_mb <= 0:
        raise ValueError("Runtime memory reserve must be positive")
    kv_cache_mem_mb = getattr(args, "kv_cache_mem_mb", None)
    if kv_cache_mem_mb is None and getattr(args, "orthogonal_smoke", False):
        if args.role == "PREFILL":
            # A bounded Device pool permits Host demotion while retaining room
            # for the long-KV seed and two 64K requests.
            kv_cache_mem_mb = 4096
        else:
            # At 30000 MiB, the reserve watermark postponed one of 64 PD
            # arrivals until another Decode stream finished.
            kv_cache_mem_mb = 34000
    if kv_cache_mem_mb is not None and kv_cache_mem_mb <= 0:
        raise ValueError("Explicit KV cache capacity must be positive")
    socket.inet_aton(args.peer_ip)
    route = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        route.connect((args.peer_ip, args.peer_port))
        local_ip = route.getsockname()[0]
    finally:
        route.close()
    no_proxy = f"localhost,127.0.0.1,{local_ip},{args.peer_ip}"
    # Match the execution settings in prefill-merge.env / decode-merge.env.
    # The smoke owns topology, cache capacity, paths and request boundaries.
    packer_impl = (
        os.environ.get("MEGA_MOE_INPUT_PACKER_IMPL", "optimized").strip().lower()
    )
    if packer_impl not in ("legacy", "optimized", "fast_finite"):
        raise ValueError(
            "MEGA_MOE_INPUT_PACKER_IMPL must be legacy|optimized|fast_finite"
        )
    environment = {
        "MODEL_TYPE": "kimi_k3",
        "CHECKPOINT_PATH": str(checkpoint),
        "TOKENIZER_PATH": str(checkpoint),
        "LOAD_METHOD": "fastsafetensors",
        "LOAD_PYTHON_MODEL": "1",
        "ACT_TYPE": "BF16",
        "HACK_LAYER_NUM": "0",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "0",
        "CHECK_NAN": "0",
        "DEEP_EP_NUM_SM": "24",
        "DEEPGEMM_JIT_COMPILER": "auto",
        "DG_JIT_CPP_STANDARD": "20",
        "DG_JIT_PRINT_COMPILER_COMMAND": "1",
        "DG_JIT_PRINT_LOAD_TIME": "1",
        "FLASHINFER_CUDA_ARCH_LIST": "10.3",
        "MEGA_MOE_INPUT_PACKER": "fused",
        "KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD": "1" if args.role == "PREFILL" else "0",
        "RTP_LLM_FP8_AG_GEMM": str(int(fp8_gemm)),
        "RTP_LLM_FP8_GEMM_RS": str(int(fp8_gemm)),
        "USE_ALL_GATHER": "0",
        "USE_DEEPEP_INTERNODE": "0",
        "USE_DEEPEP_LOW_LATENCY": "0",
        "USE_DEEPEP_MOE": "0",
        "GRAMMAR_COMPILE_CONCURRENCY": "2",
        "GRAMMAR_COMPILE_QUEUE_SIZE": "16",
        "GRAMMAR_COMPILE_TIMEOUT_MS": "30000",
        **(
            {
                "RTP_LLM_STREAM_ASYNC": "1",
                "RTP_LLM_DROP_BROAD_SYNC": "1",
                "RTP_LLM_DEVICE_INPUT": "1",
                "RTP_LLM_MTP_ASYNC_PREPARE": "1",
                "DISABLE_PDL": "0",
            }
            if args.role == "DECODE"
            else {
                "PREFILL_CHUNK_SIZE": "65536",
                "PREFILL_CACHE_HIT_METRIC_ENABLE": "1",
                "RESERVE_BLOCK_RATIO": "0",
            }
        ),
        "SP_TYPE": "mtp",
        "SP_MODEL_TYPE": "kimi_k3_mtp",
        "SP_CHECKPOINT_PATH": str(draft),
        "SP_ACT_TYPE": "BF16",
        "FT_DISABLE_CUSTOM_AR": "1",
        "GEN_NUM_PER_CIRCLE": "3",
        "ENABLE_LINEAR_REPLAY": os.environ.get("ENABLE_LINEAR_REPLAY", "0"),
        "RTP_LLM_FP8_COLLECTIVE_MAX_TOKENS": "65536",
        "QUANTIZATION": "FP8_PER_BLOCK" if fp8_gemm else "",
        "SP_QUANTIZATION": "",
        "FP8_KV_CACHE": str(int(fp8_kv_cache)),
        "FP8_MLA": str(int(fp8_kv_cache)),
        "MEGA_MOE_INPUT_PACKER_IMPL": packer_impl,
        "START_PORT": str(args.start_port),
        "LOCAL_WORLD_SIZE": str(world_size),
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(world_size)),
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        # Keep the merge environment's allocator profile explicit.
        # Torch versions differ in alias precedence; inherited conflicts must
        # not change weight-conversion or KV-pool allocation behavior.
        "PYTORCH_ALLOC_CONF": "expandable_segments:True,large_segment_size_mb:128",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,large_segment_size_mb:128",
        "FRONTEND_SERVER_COUNT": "1",
        "THINK_START_TAG": "<|open|>think<|sep|>",
        "THINK_END_TAG": "<|close|>think<|sep|><|open|>response<|sep|>",
        "REMOTE_RPC_SERVER_IP": f"{args.peer_ip}:{args.peer_port + 1}",
        # RPCPool's 5-second keepalive ACK deadline requires a shorter backup
        # poll interval while PD streams wait for model execution.
        "GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS": "500",
        "GRPC_CONFIG_JSON": json.dumps(
            {
                "client_config": {
                    "grpc.max_receive_message_length": 1073741824,
                    "grpc.max_metadata_size": 1073741824,
                },
                "server_config": {
                    "grpc.max_receive_message_length": 1073741824,
                    "grpc.max_metadata_size": 1073741824,
                    "grpc.max_concurrent_streams": 100000,
                    "grpc.max_connection_idle_ms": 600000,
                    "grpc.http2.min_recv_ping_interval_without_data_ms": 1000,
                    "grpc.http2.max_ping_strikes": 1000,
                },
                "max_server_pollers": 4,
            }
        ),
        "MODEL_SERVICE_CONFIG": json.dumps(
            {
                "service_id": "kimi-k3-smoke",
                "use_local": True,
                "role_endpoints": [
                    {
                        "group": "smoke",
                        **{
                            f"{role}_endpoint": {
                                "type": role,
                                "address": address,
                                "protocol": "http",
                                "path": "",
                            }
                            for role, address in (
                                (
                                    args.role.lower(),
                                    f"{local_ip}:{args.start_port}",
                                ),
                                (
                                    "decode" if args.role == "PREFILL" else "prefill",
                                    f"{args.peer_ip}:{args.peer_port}",
                                ),
                            )
                        },
                    }
                ],
            }
        ),
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
    }
    if orthogonal_smoke:
        environment["RTP_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB"] = "6.0"
        if args.role == "DECODE":
            # NCCL graph registration hangs on the SM103 DCP collective path
            # at the 64-request capture bucket; the fixed feat smoke uses 0.
            environment["NCCL_GRAPH_REGISTER"] = "0"
            environment["NCCL_MAX_CTAS"] = "8"
    decode_capture_batches = (
        [1, 2, 4, 8, 16, 32]
        if orthogonal_smoke and dp_size > 1
        else [1, 2, 4, 8, 16, 32, 64] if orthogonal_smoke else [1, 2, 3, 4, 7, 8, 9, 16]
    )
    options = {
        "role_type": args.role,
        "tp_size": tp_size,
        "ep_size": ep_size,
        "dp_size": dp_size,
        "ffn_sp_size": tp_size,
        "world_size": world_size,
        "local_world_size": world_size,
        "world_rank": 0,
        "prefill_cp_kv_cache_sharded": int(orthogonal_smoke),
        "prefill_cp_size": source_tp_size if orthogonal_smoke else 1,
        "remote_server_port": args.peer_port,
        "use_local": 1,
        "max_seq_len": 2097152 if orthogonal_smoke else 262144,
        # MTP update capture uses this request capacity independently of the
        # Decode runner. It must cover every configured proposal/Verify bucket.
        "max_context_batch_size": (
            64
            if orthogonal_smoke and args.role == "PREFILL"
            else max(decode_capture_batches)
        ),
        "max_batch_tokens_size": 262144 if orthogonal_smoke else 65536,
        **({"max_batch_tokens_without_cache": 65536} if orthogonal_smoke else {}),
        "concurrency_limit": 64 if orthogonal_smoke else 16,
        "seq_size_per_block": 4096,
        "kernel_seq_size_per_block": 128 if fp8_kv_cache else 64,
        "linear_step": 1,
        "ssm_state_dtype": "fp32",
        "fp8_kv_cache": int(fp8_kv_cache),
        # In the orthogonal PD profile, Decode receives the Prefill state
        # through transfer and must not reuse its own previous prefix.
        "reuse_cache": int(not orthogonal_smoke or args.role == "PREFILL"),
        "enable_device_cache": 1,
        "enable_memory_cache": int(orthogonal_smoke and args.role == "PREFILL"),
        "moe_strategy": getattr(args, "moe_strategy", None)
        or ("mega_moe_k3" if args.role == "PREFILL" else "mega_moe_k3_se"),
        "enable_cuda_graph": int(args.role == "DECODE"),
        "cache_store_rdma_mode": 1,
        "cache_store_rdma_connect_timeout_ms": 30000,
        # Formal smoke must expose the first failure, including PD retries.
        "prefill_retry_times": 0,
        "decode_retry_times": 0,
        "load_cache_timeout_ms": 7200000,
        "load_method": "fastsafetensors",
        "warm_up": int(warm_up),
        "model_warm_up": int(model_warm_up),
        "reserver_runtime_mem_mb": reserve_runtime_mem_mb,
    }
    if orthogonal_smoke and args.role == "PREFILL":
        options["memory_cache_size_mb"] = getattr(args, "memory_cache_size_mb", 32768)
    if kv_cache_mem_mb is not None:
        options["kv_cache_mem_mb"] = kv_cache_mem_mb
    if args.role == "DECODE":
        options["decode_capture_config"] = ",".join(map(str, decode_capture_batches))
    command = [str(Path(args.server).resolve(strict=True))]
    for key, value in options.items():
        command.extend(["--" + key, str(value)])
    return environment, command


def require_bazel_destination(path):
    """Allow only owned directories within this action's output or temporary root."""
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ValueError("Bazel destination must be an absolute owned directory")
    resolved = candidate.resolve(strict=True)
    for variable in ("TEST_UNDECLARED_OUTPUTS_DIR", "TEST_TMPDIR"):
        value = os.environ.get(variable)
        if not value:
            continue
        root = Path(value)
        if not root.is_absolute() or root.is_symlink():
            continue
        root = root.resolve(strict=True)
        if root == Path(root.anchor) or not resolved.is_relative_to(root):
            continue
        for directory in (resolved, *resolved.parents):
            info = directory.lstat()
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o022
            ):
                raise ValueError(
                    "Bazel destination ancestors must be owned and private to the runtime user"
                )
            if directory == root:
                return
    raise ValueError(
        "Bazel destination is outside this test's output and temporary roots"
    )


def require_bazel_temporary(path, prefix):
    """Validate short IPC roots allocated by this worker, without allowing arbitrary /tmp."""
    candidate = Path(path)
    info = candidate.lstat()
    if (
        not candidate.is_absolute()
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("Bazel temporary directory must be an owned private directory")
    if candidate.parent == Path("/tmp") and re.fullmatch(
        rf"{re.escape(prefix)}-{os.geteuid()}-[a-z0-9_]+", candidate.name
    ):
        return
    require_bazel_destination(candidate)


def cpu_tp_socket_environment(run, socket_dir=None):
    directory = Path(run) / "uds"
    if socket_dir is not None:
        directory = Path(socket_dir)
        info = directory.lstat()
        if (
            not directory.is_absolute()
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise ValueError(
                "CPU TP socket directory must be an owned private directory"
            )
        directory = directory.resolve(strict=True)
    # Supported PD profiles use single-digit DP ranks, with this same suffix size.
    socket_path = directory / "rtp_llm_tp_k3_dp0_0.sock"
    if len(os.fsencode(socket_path)) >= 108:
        raise ValueError("CPU TP Unix socket path must be shorter than 108 bytes")
    return {
        "RTP_LLM_CPU_TP_BROADCASTER_DIR": str(directory),
        "RTP_LLM_CPU_TP_BROADCASTER_ID": "k3",
    }


def runtime_tmp_environment(run, tmp_dir=None):
    directory = Path(run) / "tmp"
    if tmp_dir is not None:
        directory = Path(tmp_dir)
        info = directory.lstat()
        if (
            not directory.is_absolute()
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise ValueError("Runtime tmp directory must be an owned private directory")
        directory = directory.resolve(strict=True)
    # PyTorch SymmetricMemory uses TMPDIR/symm_mem-<pid> for FD exchange.
    # Reserve the full signed 32-bit PID width before loading any model weights.
    socket_path = directory / "symm_mem-2147483647"
    if len(os.fsencode(socket_path)) >= 108:
        raise ValueError(
            "SymmetricMemory Unix socket path must be shorter than 108 bytes"
        )
    return {"TMPDIR": str(directory)}


def require_rdma_device(run):
    """Reject missing container devices before loading the full checkpoint.

    This proves local device visibility only; real PD transfer is a separate gate.
    """
    evidence = Path(run) / "rdma-preflight.txt"
    try:
        result = subprocess.run(
            ["ibv_devinfo"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        evidence.write_text(f"RDMA device probe failed: {exc}\n")
        raise RuntimeError(
            "Cannot verify RDMA devices in this runtime container"
        ) from exc
    evidence.write_text(result.stdout)
    if result.returncode or not re.search(
        r"\bstate:\s+PORT_ACTIVE\s*\(4\)", result.stdout
    ):
        raise RuntimeError("No active RDMA port in this runtime container")


def validate_rdma_hcas(value, devices, links):
    names = value.split(",")
    if (
        not names
        or len(set(names)) != len(names)
        or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names)
    ):
        raise ValueError("RDMA HCAs must be unique, non-empty device names")
    available = {line.split()[0] for line in devices.splitlines() if line.split()}
    for name in names:
        if name not in available:
            raise ValueError(f"RDMA HCA absent from ibv_devices: {name}")
        pattern = rf"(?<![A-Za-z0-9_.-]){re.escape(name)}/"
        if not any(
            re.search(pattern, line)
            and re.search(r"\bstate\s+ACTIVE\b", line)
            and re.search(r"\bphysical_state\s+LINK_UP\b", line)
            for line in links.splitlines()
        ):
            raise ValueError(f"RDMA HCA is not ACTIVE/LINK_UP: {name}")
    return ",".join(names)


def rdma_hca_environment(run, value):
    if value is None:
        return {}
    devices = subprocess.check_output(["ibv_devices"], text=True, timeout=30)
    links = subprocess.check_output(["rdma", "link", "show"], text=True, timeout=30)
    selected = validate_rdma_hcas(value, devices, links)
    (Path(run) / "rdma-hcas.json").write_text(
        json.dumps(
            {
                "devices": devices,
                "links": links,
                "ACCL_USE_NICS": selected,
                "note": "Local HCA validation only; real RDMA transfer must still pass.",
            },
            indent=2,
        )
    )
    return {"ACCL_USE_NICS": selected}


def require_gpu_capacity(
    run, allow_shared_accuracy=False, min_free_gib=250, gpu_count=8
):
    """Record a final selected-GPU capacity snapshot; shared runs prove correctness only.

    Host-side fleet selection must also record owners: container PID namespaces
    can hide the owners of nvidia-smi's host PIDs.
    """
    if not math.isfinite(min_free_gib) or min_free_gib <= 0:
        raise ValueError("GPU free-memory requirement must be finite and positive")

    def query(fields):
        return subprocess.check_output(
            ["nvidia-smi", fields, "--format=csv,noheader,nounits"],
            text=True,
            timeout=30,
        )

    gpu_text = query("--query-gpu=index,uuid,memory.free")
    process_text = query(
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"
    )
    evidence = {
        "allow_shared_accuracy": allow_shared_accuracy,
        "performance_validated": False,
        "min_free_gib": min_free_gib,
        "gpus": gpu_text,
        "compute_processes": process_text,
        "owner_evidence": "See host-side fleet selection and process monitor; host PIDs may be hidden here.",
    }
    (Path(run) / "gpu-preflight.json").write_text(json.dumps(evidence, indent=2) + "\n")
    rows = list(csv.reader(gpu_text.splitlines(), skipinitialspace=True))
    selected = {int(row[0]): float(row[2]) for row in rows if len(row) == 3}
    if any(
        i not in selected
        or not math.isfinite(selected[i])
        or selected[i] < min_free_gib * 1024
        for i in range(gpu_count)
    ):
        raise RuntimeError(
            "Insufficient free GPU memory for the selected profile; reselect hosts"
        )
    selected_uuids = {row[1].strip() for row in rows if int(row[0]) in range(gpu_count)}
    occupied = [
        row
        for row in csv.reader(process_text.splitlines(), skipinitialspace=True)
        if len(row) >= 2
        and row[0].strip() in selected_uuids
        and row[1].strip().isdigit()
    ]
    if occupied and not allow_shared_accuracy:
        raise RuntimeError(
            "GPUs occupied; reselect hosts or explicitly allow shared accuracy validation"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=["PREFILL", "DECODE"])
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, help="Defaults to TP×DP")
    parser.add_argument(
        "--prefill-source-tp-size",
        type=int,
        default=8,
        help="Physical Prefill attention TP / PageRR shard count for Decode",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--draft-checkpoint", required=True)
    parser.add_argument("--peer-ip", required=True)
    parser.add_argument("--start-port", required=True, type=int)
    parser.add_argument("--peer-port", required=True, type=int)
    parser.add_argument("--server", required=True)
    parser.add_argument(
        "--allow-checkpoint-root",
        required=True,
        help="Explicitly approved checkpoint mount root for a Bazel native action",
    )
    parser.add_argument(
        "--expected-draft-shards",
        type=int,
        help="Optional exact shard count pinned by this run's MTP configuration",
    )
    parser.add_argument(
        "--rdma-hcas", help="Explicit comma-separated RDMA HCA allowlist"
    )
    parser.add_argument(
        "--reserve-runtime-mem-mb",
        type=int,
        default=14336,
        help="Per-rank runtime memory reserve in MiB",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="New directory within this Bazel action's outputs",
    )
    parser.add_argument(
        "--cpu-tp-socket-dir",
        help="Existing owned 0700 directory for CPU TP IPC; caller owns cleanup (default: run-dir/uds)",
    )
    parser.add_argument(
        "--runtime-tmp-dir",
        help="Existing owned 0700 short directory for runtime temporary files; caller owns cleanup (default: run-dir/tmp)",
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument(
        "--fp8-gemm", action="store_true", help="Enable FP8 projection GEMM"
    )
    parser.add_argument(
        "--fp8-kv-cache",
        action="store_true",
        help="Enable ordinary E4M3 MLA operands and KV cache via the existing FP8_KV_CACHE setting",
    )
    parser.add_argument(
        "--warm-up", action="store_true", help="Enable startup request warmup"
    )
    parser.add_argument(
        "--model-warm-up",
        action="store_true",
        help="Enable model kernel warmup; requires --warm-up",
    )
    parser.add_argument(
        "--moe-strategy",
        choices=["mega_moe", "mega_moe_se", "mega_moe_k3", "mega_moe_k3_se"],
        default=None,
        help="Select the routed/shared MoE executor",
    )
    parser.add_argument(
        "--orthogonal-smoke",
        action="store_true",
        help="Enable the complete 64K PageRR, Host cache, DCP and Graph smoke profile",
    )
    parser.add_argument(
        "--memory-cache-size-mb",
        type=int,
        default=32768,
        help="Per-rank Prefill Host cache capacity in the orthogonal profile",
    )
    parser.add_argument(
        "--kv-cache-mem-mb",
        type=int,
        help="Explicit per-rank Device KV cache capacity for bounded smoke profiles",
    )
    parser.add_argument(
        "--allow-shared-accuracy",
        action="store_true",
        help="Allow correctness-only coexistence after host-side isolation checks; never for performance",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=250,
        help="Required free memory on each selected GPU (default: 250 GiB)",
    )
    args = parser.parse_args()
    if args.expected_draft_shards is not None and args.expected_draft_shards < 1:
        parser.error("--expected-draft-shards requires a positive count")
    if args.ep_size is None:
        args.ep_size = args.tp_size * args.dp_size
    environment, command = launch_config(args)
    if args.print_config:
        print(json.dumps({"environment": environment, "command": command}, indent=2))
        return
    if os.geteuid() == 0:
        raise ValueError(
            "Run as the workspace owner inside the verified runtime container"
        )
    run = Path(args.run_dir).resolve()
    require_bazel_destination(run.parent)
    if args.cpu_tp_socket_dir is not None:
        require_bazel_temporary(args.cpu_tp_socket_dir, "k3pd")
    if args.runtime_tmp_dir is not None:
        require_bazel_temporary(args.runtime_tmp_dir, "k3tmp")
    environment.update(cpu_tp_socket_environment(run, args.cpu_tp_socket_dir))
    environment.update(runtime_tmp_environment(run, args.runtime_tmp_dir))
    run.mkdir(exist_ok=False)
    if args.cpu_tp_socket_dir is None:
        (run / "uds").mkdir(mode=0o700)
    require_rdma_device(run)
    environment.update(rdma_hca_environment(run, args.rdma_hcas))
    # Reject potentially inherited experimental settings instead of trusting
    # a shell left over from a DCP/FP8/synthetic-acceptance experiment.
    forbidden = ("CP_ROTATE_METHOD", "QUANTIZATION", "SP_QUANTIZATION")
    for name in forbidden:
        if os.environ.get(name):
            raise ValueError(f"Unset inherited {name} before precision validation")
    inherited = os.environ.copy()
    inherited.update(environment)
    jit_cache_root = os.environ.get("RTP_LLM_JIT_CACHE_ROOT")
    if jit_cache_root:
        jit_cache_root = Path(jit_cache_root).resolve()
        require_bazel_destination(jit_cache_root.parent)
        jit_cache_root.mkdir(exist_ok=True)
        require_bazel_destination(jit_cache_root)
    if args.runtime_tmp_dir is None:
        (run / "tmp").mkdir(mode=0o700)
    logs = run / "logs"
    logs.mkdir()
    inherited["LOG_PATH"] = str(logs)
    environment["LOG_PATH"] = str(logs)
    # An explicit local root opts these components out of remote snapshots.
    # Without it, leave their paths unset for the shared JIT manager to assign.
    if jit_cache_root:
        for key, subdir in {
            "TRITON_CACHE_DIR": "triton",
            "DG_JIT_CACHE_DIR": "deep-gemm",
            "FLASHINFER_WORKSPACE_BASE": "flashinfer",
        }.items():
            if inherited.get(key, "").strip():
                continue
            directory = jit_cache_root / subdir
            directory.mkdir(exist_ok=True)
            inherited[key] = str(directory)
            environment[key] = str(directory)
    for label, checkpoint in (
        ("target", args.checkpoint),
        ("draft", args.draft_checkpoint),
    ):
        guard_command = [
            sys.executable,
            str(Path(__file__).with_name("checkpoint_guard.py")),
            "metadata",
            "--checkpoint",
            checkpoint,
            "--load-method",
            "fastsafetensors",
            "--allow-checkpoint-root",
            args.allow_checkpoint_root,
        ]
        if label == "draft" and args.expected_draft_shards is not None:
            guard_command.extend(["--expected-shards", str(args.expected_draft_shards)])
        report = run / f"{label}-preflight.txt"
        with report.open("w") as output:
            try:
                subprocess.run(
                    guard_command,
                    env=inherited,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"{args.role} {label} checkpoint metadata check failed: "
                    f"checkpoint={checkpoint}; report={report}"
                ) from exc
    # This is an additional final check, not a replacement for fleet selection.
    require_gpu_capacity(
        run, args.allow_shared_accuracy, args.min_free_gib, args.tp_size * args.dp_size
    )
    ports = []
    try:
        for port in range(
            args.start_port, args.start_port + args.tp_size * args.dp_size * 9
        ):
            sock = socket.socket()
            ports.append(sock)
            sock.bind(("0.0.0.0", port))
    finally:
        for sock in ports:
            sock.close()
    (run / "launch.json").write_text(
        json.dumps(
            {
                "profile": (
                    f"{'fp8' if args.fp8_kv_cache or args.fp8_gemm else 'bf16'}-"
                    "full93-"
                    f"dp{args.dp_size}-tp{args.tp_size}-ep{args.ep_size}-sp-mtp3-rdma"
                ),
                "full_model_acceptance_eligible": True,
                "environment": environment,
                "command": command,
                "pid": os.getpid(),
                "allow_shared_accuracy": args.allow_shared_accuracy,
                "approved_checkpoint_root": args.allow_checkpoint_root,
                "execution_mode": "bazel-native",
                "performance_validated": False,
            },
            indent=2,
        )
        + "\n"
    )
    os.execve(command[0], command, inherited)


if __name__ == "__main__":
    main()
