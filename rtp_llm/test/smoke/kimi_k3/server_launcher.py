"""Launch one BF16-compute PD endpoint; K3.1 defaults to FP8 operands.

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

from rtp_llm.test.smoke.kimi_k3.mtp_contract import native_mla_draft


def smoke_topology(args):
    dp = getattr(args, "dp_size", None)
    if dp is None:
        dp = 2 if getattr(args, "model_type", "kimi_k3") == "kimi_k31" and args.role == "DECODE" else 1
    dp = int(dp)
    if dp < 1:
        raise ValueError("PD topology requires positive TP/DP")
    tp = getattr(args, "tp_size", None)
    tp = int(8 // dp if tp is None else tp)
    ep = getattr(args, "ep_size", None)
    return tp, dp, int(tp * dp if ep is None else ep)


def launch_config(args):
    from rtp_llm.models.kimi_k3.nvfp4_config import parse_nvfp4_flag

    model_type = getattr(args, "model_type", "kimi_k3")
    nvfp4 = {
        key.upper(): str(int(model_type == "kimi_k31" and parse_nvfp4_flag(
            key.upper(), getattr(args, key, None), default=True
        ))) for key in ("kda_nvfp4", "dsa_nvfp4")
    }
    fp8_gemm = getattr(args, "fp8_gemm", None)
    fp8_kv_cache = getattr(args, "fp8_kv_cache", None)
    fp8_gemm = model_type == "kimi_k31" if fp8_gemm is None else bool(fp8_gemm)
    fp8_kv_cache = model_type == "kimi_k31" if fp8_kv_cache is None else bool(fp8_kv_cache)
    warm_up = bool(getattr(args, "warm_up", False))
    model_warm_up = bool(getattr(args, "model_warm_up", False))
    enable_cuda_graph = getattr(args, "enable_cuda_graph", None)
    if enable_cuda_graph is not None and type(enable_cuda_graph) is not bool:
        raise ValueError("enable_cuda_graph must be a boolean")
    if model_warm_up and not warm_up:
        raise ValueError("model warmup requires --warm-up")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    target_only = bool(getattr(args, "target_only", False))
    draft = None if target_only else Path(args.draft_checkpoint).resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    text = config.get("text_config", config)
    expected_layers = int(getattr(args, "expected_target_layers", 93))
    if model_type == "kimi_k3" and (
        expected_layers != 93 or text.get("num_hidden_layers") != 93
    ):
        raise ValueError("Smoke target must be a complete 93-layer checkpoint")
    if expected_layers not in (4, 93) or text.get("num_hidden_layers") != expected_layers:
        raise ValueError("Smoke target must match its explicit four/93-layer contract")
    if model_type == "kimi_k31":
        if config.get("model_type") != "kimi_k31":
            raise ValueError("K3.1 requires a kimi_k31 checkpoint")
        if not target_only:
            native_mla_draft(draft, config)
    orthogonal_smoke = bool(getattr(args, "orthogonal_smoke", False))
    tp_size, dp_size, ep_size = smoke_topology(args)
    world_size = tp_size * dp_size
    prefill_capacity = getattr(args, "prefill_token_capacity", None)
    if prefill_capacity is not None and (
        model_type != "kimi_k31" or args.role != "PREFILL"
        or not 1024 <= prefill_capacity <= 65536 or prefill_capacity % 1024
    ):
        raise ValueError("K3.1 Prefill fixture token capacity must be page-aligned in [1024,65536]")
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
    reserve_runtime_mem_mb = getattr(args, "reserve_runtime_mem_mb", None)
    if reserve_runtime_mem_mb is None:
        # Auto-size KV after weight loading, leaving room for the 64K Q fixture.
        reserve_runtime_mem_mb = (
            28672 if model_type == "kimi_k31" and orthogonal_smoke
            and args.role == "PREFILL" else 14336
        )
    if reserve_runtime_mem_mb <= 0:
        raise ValueError("Runtime memory reserve must be positive")
    kv_cache_mem_mb = getattr(args, "kv_cache_mem_mb", None)
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
        "MODEL_TYPE": model_type,
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
        "SP_TYPE": "none" if target_only else "mtp",
        "SP_MODEL_TYPE": "" if target_only else model_type + "_mtp",
        "SP_CHECKPOINT_PATH": "" if target_only else str(draft),
        "SP_ACT_TYPE": "BF16",
        "FT_DISABLE_CUSTOM_AR": "1",
        "GEN_NUM_PER_CIRCLE": "0" if target_only else "3",
        "ENABLE_LINEAR_REPLAY": (
            "1" if args.role == "DECODE" else os.environ.get("ENABLE_LINEAR_REPLAY", "0")
        ),
        "QUANTIZATION": "FP8_PER_BLOCK" if fp8_gemm else "",
        **nvfp4,
        "RTP_LLM_FP4_GEMM_BACKEND": "cutlass",
        "SP_QUANTIZATION": "",
        "FP8_KV_CACHE": str(int(fp8_kv_cache)),
        "FP8_MLA": "0" if model_type == "kimi_k31" else str(int(fp8_kv_cache)),
        "MEGA_MOE_INPUT_PACKER_IMPL": packer_impl,
        "START_PORT": str(args.start_port),
        "LOCAL_WORLD_SIZE": str(world_size),
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(world_size)),
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        # Keep the merge environment's allocator profile explicit.
        # Torch versions differ in alias precedence; inherited conflicts must
        # not change weight-conversion or KV-pool allocation behavior.
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
    if model_type == "kimi_k31":
        environment["RTP_MTP_ACCEPTANCE_STATS"] = "1"
        if args.role == "DECODE":
            environment["DSA_DCP_TOPK"] = os.environ.get("DSA_DCP_TOPK", "mega_topk")
            environment["DSA_DCP_COMBINE"] = os.environ.get("DSA_DCP_COMBINE", "mega_a2a")
            environment["DSA_DCP_PROLOGUE"] = os.environ.get("DSA_DCP_PROLOGUE", "multi_stream")
    if getattr(args, "execution_evidence", False):
        environment["KIMI_K3_SMOKE_EVIDENCE"] = "1"
    decode_capture_batches = (
        [1, 2, 4, 8, 16, 32]
        if orthogonal_smoke and dp_size > 1
        else [1, 2, 4, 8, 16, 32, 64] if orthogonal_smoke else [1, 2, 3, 4, 7, 8, 9, 16]
    )
    capture = getattr(args, "decode_capture_batches", None)
    if capture is not None:
        decode_capture_batches = [int(value) for value in capture.split(",")]
        if (args.role != "DECODE" or not decode_capture_batches
                or decode_capture_batches != sorted(set(decode_capture_batches))
                or decode_capture_batches[0] < 1):
            raise ValueError("Decode capture batches must be sorted unique positive sizes")
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
        # K3.1's longest smoke history is655360, plus generation. Preserve
        # every case while budgeting BF16 MLA-draft Graph buckets for
        # that declared service range. The checkpoint is never rewritten.
        "max_seq_len": 2097152 if orthogonal_smoke else 262144,
        # MTP update capture uses this request capacity independently of the
        # Decode runner. It must cover every configured proposal/Verify bucket.
        "max_context_batch_size": (
            16
            if orthogonal_smoke and args.role == "PREFILL"
            else max(decode_capture_batches)
        ),
        "max_batch_tokens_size": 262144 if orthogonal_smoke else 65536,
        **({"max_batch_tokens_without_cache": 65536} if orthogonal_smoke else {}),
        "concurrency_limit": 64 if orthogonal_smoke and args.role == "DECODE" else 16,
        "seq_size_per_block": 1024 if model_type == "kimi_k31" else 4096,
        "kernel_seq_size_per_block": 128 if model_type == "kimi_k31" or fp8_kv_cache else 64,
        "reuse_cache_tail_blocks": 2147483647,
        "ssm_state_dtype": "fp32",
        "fp8_kv_cache": int(fp8_kv_cache),
        # In the orthogonal PD profile, Decode receives the Prefill state
        # through transfer and must not reuse its own previous prefix.
        "reuse_cache": int(not orthogonal_smoke or args.role == "PREFILL"),
        "enable_device_cache": 1,
        "enable_memory_cache": int(orthogonal_smoke and args.role == "PREFILL"),
        "moe_strategy": getattr(args, "moe_strategy", None)
        or ("mega_moe" if args.role == "PREFILL" else "mega_moe_se"),
        "enable_cuda_graph": int(args.role == "DECODE" if enable_cuda_graph is None else enable_cuda_graph),
        "cache_store_rdma_mode": int(getattr(args, "cache_store_rdma_mode", True)),
        "cache_store_rdma_connect_timeout_ms": 30000,
        # Formal smoke must expose the first failure, including PD retries.
        "prefill_retry_times": 0,
        "decode_retry_times": 0,
        "load_cache_timeout_ms": 7200000,
        "load_method": "fastsafetensors",
        "warm_up": int(warm_up),
        "model_warm_up": int(model_warm_up),
        "reserver_runtime_mem_mb": reserve_runtime_mem_mb,
        # Match the deployed service and retain the normal staged drain defaults.
        "shutdown_timeout": 120,
    }
    last_n = getattr(args, "last_n_checkpoints", None)
    last_n_page_tokens = getattr(args, "last_n_page_tokens", None)
    if last_n_page_tokens is not None and (last_n is None or last_n_page_tokens not in (64, 128)):
        raise ValueError("Last-N page size requires an explicit Last-N smoke and 64 or 128 tokens")
    if last_n is not None:
        if (type(last_n) is not int or not 1 <= last_n <= 5
                or model_type != "kimi_k31" or not orthogonal_smoke
                or tp_size != 8 or dp_size != 1 or source_tp_size != 8):
            raise ValueError("K3.1 Last-N smoke requires N=1..5, TP8/EP8 and CP8")
        options["seq_size_per_block"] = options["kernel_seq_size_per_block"] = last_n_page_tokens or 128
        options["linear_pool_blocks"] = (getattr(args, 'linear_pool_blocks', None)
                                         or (96 if args.role == 'PREFILL' else 512))
        tier = getattr(args, 'last_n_tier_probe', None)
        if tier is not None:
            if tier not in ('DEVICE', 'HOST', 'DISK') or last_n != 5:
                raise ValueError('isolated Last-5 probe requires DEVICE, HOST or DISK')
            if args.role == 'PREFILL':
                options['enable_device_cache'] = int(tier == 'DEVICE')
                options['enable_memory_cache'] = int(tier == 'HOST')
        # Decode keeps its existing rolling-state policy. It needs the same
        # physical page geometry to receive P's final state and MLA pages.
        if args.role == "PREFILL":
            options["reuse_cache_tail_blocks"] = (2147483647 if getattr(args, 'last_n_calibration', False) else last_n)
            disk_paths = getattr(args, "disk_cache_paths", None)
            if disk_paths and tier in (None, 'DISK'):
                if len(disk_paths.split(",")) != 8:
                    raise ValueError("Last-N SSD smoke needs one task-owned path per rank")
                options.update(enable_disk_cache=1, disk_cache_paths=disk_paths,
                               disk_cache_size_mb=getattr(args, "disk_cache_size_mb", 65536),
                               disk_cache_buffered_io=1)
    if orthogonal_smoke and args.role == "PREFILL":
        options["memory_cache_size_mb"] = getattr(args, "memory_cache_size_mb", 32768)
    if kv_cache_mem_mb is not None:
        options["kv_cache_mem_mb"] = kv_cache_mem_mb
    if model_type == "kimi_k31":
        options["cp_force_single_prefill"] = 0
        if orthogonal_smoke and not options["cache_store_rdma_mode"]:
            # A single decode owner can load one RPC per cache group/layer
            # for 32 concurrent requests from the same Prefill peer.
            options["cache_store_tcp_anet_rpc_queue_num"] = 8192
        if orthogonal_smoke and args.role == "PREFILL":
            # Retain the full inventory's short-input Decode63/64 groups.
            # Existing64K token admission limits still bound the rectangle.
            # max_context_batch_size controls the synthetic max-history
            # startup warmup, not FIFO's real request concurrency. Keep the
            # shared16-row warmup;64 x2M virtual histories exceed DSA budgets.
            options["concurrency_limit"] = 64
        elif orthogonal_smoke and args.role == "DECODE":
            # Each independent DP owner needs32 requests for globalBS64.
            # LINEAR tail/replay pools use this per-owner concurrency too.
            options["concurrency_limit"] = 64 // dp_size
    if getattr(args, "test_block_num", None) is not None:
        if args.test_block_num < 2:
            raise ValueError("test cache pool must include reserved block zero and a real block")
        options["test_block_num"] = args.test_block_num
    linear_blocks = getattr(args, "linear_pool_blocks", None)
    if linear_blocks is not None:
        if type(linear_blocks) is not int or not 2 <= linear_blocks <= 2**31 - 1:
            raise ValueError("linear pool blocks must be an integer in [2, INT_MAX]")
        options["linear_pool_blocks"] = linear_blocks
    transfer_workers = getattr(args, "block_tree_transfer_worker_count", None)
    if transfer_workers is not None:
        if type(transfer_workers) is not int or not 1 <= transfer_workers <= 64:
            raise ValueError("BlockTree transfer workers must be an integer in [1, 64]")
        options["block_tree_transfer_worker_count"] = transfer_workers
    if prefill_capacity is not None:
        # Explicit fixture admission/workspace capacity, not an in-model Q loop.
        # The caller must prove each complete physical query fits this capacity.
        environment["PREFILL_CHUNK_SIZE"] = str(prefill_capacity)
        options["max_batch_tokens_without_cache"] = prefill_capacity
        options["max_batch_tokens_size"] = prefill_capacity
    max_seq_len = getattr(args, "max_seq_len", None)
    if max_seq_len is not None:
        if max_seq_len < 1024:
            raise ValueError("max sequence length must be at least one physical page")
        options["max_seq_len"] = max_seq_len
    capacity = getattr(args, "request_capacity", None)
    if capacity is not None:
        if not 1 <= capacity <= 64 or (args.role == "DECODE" and max(decode_capture_batches) > capacity):
            raise ValueError("request capacity must cover all Decode graph buckets and remain in [1,64]")
        options["concurrency_limit"] = capacity
        options["max_context_batch_size"] = capacity
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
    parser.add_argument("--tp-size", type=int, help="Defaults to 8 / DP")
    parser.add_argument("--dp-size", type=int, help="Defaults to 2 for K3.1 Decode, otherwise 1")
    parser.add_argument("--ep-size", type=int, help="Defaults to TP×DP")
    parser.add_argument(
        "--prefill-source-tp-size",
        type=int,
        default=8,
        help="Physical Prefill attention TP / PageRR shard count for Decode",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", choices=("kimi_k3", "kimi_k31"), default="kimi_k3")
    parser.add_argument("--expected-target-layers", type=int, choices=(4, 93), default=93)
    parser.add_argument("--draft-checkpoint")
    parser.add_argument("--target-only", action="store_true",
                        help="Launch a true SP_TYPE=none baseline without draft weights")
    parser.add_argument("--max-seq-len", type=int,
                        help="Explicit bounded sequence capacity for acceptance runs")
    parser.add_argument("--request-capacity", type=int,
                        help="Explicit per-owner request capacity for bounded acceptance runs")
    parser.add_argument("--decode-capture-batches",
                        help="Explicit comma-separated Decode graph buckets for bounded acceptance runs")
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
        help="Per-rank runtime reserve in MiB (K3.1 PD Prefill 28672, other roles 14336)",
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
    parser.add_argument("--kda-nvfp4", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dsa-nvfp4", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--fp8-gemm", action=argparse.BooleanOptionalAction, default=None,
        help="FP8 projection GEMM (default: enabled for K3.1)"
    )
    parser.add_argument(
        "--fp8-kv-cache",
        action=argparse.BooleanOptionalAction, default=None,
        help="FP8 KV cache (default: enabled for K3.1); K3.1 retains BF16 sparse attention compute",
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
        choices=["mega_moe", "mega_moe_se"],
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
    parser.add_argument("--execution-evidence", action="store_true",
                        help="Record optional CPU execution evidence;disable for performance")
    parser.add_argument("--test-block-num", type=int,
                        help="Explicit bounded cache-pool size for reproducible tier-pressure smoke")
    parser.add_argument("--linear-pool-blocks", type=int,
                        help="Independent physical LINEAR pool size, including reserved block zero")
    parser.add_argument("--prefill-token-capacity", type=int,
                        help="K3.1 fixture-only page-aligned token/workspace capacity;default65536 unchanged")
    args = parser.parse_args()
    if not args.target_only and not args.draft_checkpoint:
        parser.error("MTP requires --draft-checkpoint; synthetic fallback is disabled")
    if args.expected_draft_shards is not None and args.expected_draft_shards < 1:
        parser.error("--expected-draft-shards requires a positive count")
    args.tp_size, args.dp_size, args.ep_size = smoke_topology(args)
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
    if (args.model_type == "kimi_k31" and args.role == "PREFILL"
            and args.orthogonal_smoke and args.execution_evidence):
        from rtp_llm.test.smoke.kimi_k3.k31_chunk_fixture import fixture_environment
        fixture = fixture_environment(run, inherited)
        environment.update(fixture); inherited.update(fixture)
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
        if label == "draft" and args.target_only:
            continue
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
                    f"{'fp8' if environment['FP8_KV_CACHE'] == '1' or environment['QUANTIZATION'] else 'bf16'}-"
                    f"real{args.expected_target_layers}-"
                    f"dp{args.dp_size}-tp{args.tp_size}-ep{args.ep_size}-"
                    f"{'target-only' if args.target_only else 'native-mla-mtp3'}-rdma"
                ),
                "draft_contract": (
                    None if args.target_only or args.model_type != "kimi_k31"
                    else native_mla_draft(args.draft_checkpoint,
                                          json.loads((Path(args.checkpoint) / "config.json").read_text()))
                ),
                "full_model_acceptance_eligible": args.expected_target_layers == 93,
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
