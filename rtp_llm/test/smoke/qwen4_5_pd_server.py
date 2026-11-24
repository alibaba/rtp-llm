"""Pure PD-disaggregated Bazel server launcher for Qwen 4.5-Flash (``qwen4_5``).

Launches dedicated ``ROLE_TYPE=PREFILL`` and ``ROLE_TYPE=DECODE`` RTP-LLM instances
via ``MagaServerManager`` + ``ServiceRoute`` with mandatory RDMA CacheStore transport
and Prefill Device + Host Memory cache enabled (no unified / PDFUSION mode).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ipaddress
import logging
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM
from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.maga_server_manager import MagaServerManager

DEFAULT_3FS_CKPT = "/mnt/hf3fs/3fs/models/Qwen4.5-Flash-1109-bf16-re-mxfp4-ple-fp8"


def _parse_gpu_list(raw: str) -> List[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("GPU list must be non-empty")
    return gpus


def _hca_usable_link_layers(name: str) -> set[str]:
    ports = Path("/sys/class/infiniband") / name / "ports"
    if not ports.is_dir():
        return set()
    layers: set[str] = set()
    for gid_path in ports.glob("*/gids/*"):
        try:
            raw_gid = gid_path.read_text().strip()
            addr = ipaddress.IPv6Address(raw_gid)
            if addr.is_unspecified or addr == ipaddress.IPv6Address("fe80::"):
                continue
            port_dir = gid_path.parent.parent
            link_layer = (port_dir / "link_layer").read_text().strip()
            if link_layer == "Ethernet":
                ndev = (port_dir / "gid_attrs/ndevs" / gid_path.name).read_text().strip()
                if not ndev or not (Path("/sys/class/net") / ndev).exists():
                    continue
            layers.add(link_layer)
        except (OSError, ValueError):
            continue
    return layers


def select_single_layer_rdma_hcas(explicit: Optional[str] = None) -> Tuple[str, str]:
    """Validate or auto-select RDMA HCAs belonging to a single active link layer.

    InfiniBand and Ethernet (RoCE) HCAs must not be mixed in ``ACCL_USE_NICS``.
    """
    soft_memlock, hard_memlock = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft_memlock != resource.RLIM_INFINITY and soft_memlock < 64 * 1024 * 1024:
        raise RuntimeError(
            f"RDMA requires RLIMIT_MEMLOCK >= 64 MiB (unlimited recommended), "
            f"got soft={soft_memlock}, hard={hard_memlock}. "
            f"Recreate container with --ulimit memlock=-1:-1 --cap-add=IPC_LOCK "
            f"or run on an RDMA-enabled container (e.g. L20-dev-113 lhc_GPU)."
        )

    requested = (
        explicit
        or os.environ.get("QWEN4_5_RDMA_HCAS", "")
        or os.environ.get("ACCL_USE_NICS", "")
    ).strip()

    try:
        links_out = subprocess.check_output(["rdma", "link", "show"], text=True, timeout=15)
        devices_out = subprocess.check_output(["ibv_devices"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"RDMA preflight command failed in runtime container: {exc}") from exc

    active_candidates = list(
        dict.fromkeys(
            word.split("/")[0]
            for line in links_out.splitlines()
            if "ACTIVE" in line and "LINK_UP" in line
            for word in line.split()
            if word.startswith("mlx5_") and "/" in word
        )
    )
    active_set = set(active_candidates)

    if requested:
        names = list(dict.fromkeys(item.strip() for item in requested.split(",") if item.strip()))
        if not names:
            raise ValueError("Explicit RDMA HCA list must be non-empty")
        layers: set[str] = set()
        for name in names:
            if name not in devices_out:
                raise RuntimeError(f"RDMA HCA {name!r} absent from ibv_devices")
            if name not in active_set:
                raise RuntimeError(f"RDMA HCA {name!r} is not ACTIVE/LINK_UP")
            hca_layers = _hca_usable_link_layers(name)
            if not hca_layers:
                raise RuntimeError(f"RDMA HCA {name!r} lacks usable GID/netdev")
            layers.update(hca_layers)
        if len(layers) != 1:
            raise ValueError(
                f"Mixed RDMA link layers in {names}: {sorted(layers)}; keep single link layer"
            )
        return ",".join(names), next(iter(layers))

    by_layer: Dict[str, List[str]] = {}
    for name in active_candidates:
        if name not in devices_out:
            continue
        for layer in _hca_usable_link_layers(name):
            by_layer.setdefault(layer, []).append(name)

    for preferred_layer in ("Ethernet", "InfiniBand"):
        if by_layer.get(preferred_layer):
            return ",".join(by_layer[preferred_layer]), preferred_layer

    if by_layer:
        layer, hcas = next(iter(by_layer.items()))
        return ",".join(hcas), layer

    raise RuntimeError(
        f"No active usable RDMA HCA found (active_candidates={active_candidates})"
    )


def wait_tcp_port_ready(host: str, port: int, timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                logging.info("TCP ready: %s (%s:%d)", label, host, port)
                return
        except OSError as exc:
            last_err = exc
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {label} ({host}:{port}): {last_err}")


def build_pure_pd_managers(
    *,
    service_id: str,
    prefill_gpus: Sequence[str],
    decode_gpus: Sequence[str],
    prefill_port: int,
    decode_port: int,
    tp_size: int,
    ep_size: int,
    seq_size_per_block: int,
    max_seq_len: int,
    reserver_runtime_mem_mb: int,
    load_workers: int,
    rdma_hcas: str,
    memory_cache_size_mb: int,
    kv_cache_mem_mb: Optional[int] = None,
    rdma_connect_timeout_ms: int = 30000,
    max_context_batch_size: int = 8,
    concurrency_limit: int = 16,
    enable_decode_cuda_graph: bool = True,
    force_host_demotion: bool = False,
    role_name_suffix: str = "",
    sp_type: str = "",
    sp_model_type: str = "",
    sp_checkpoint_path: str = "",
    sp_act_type: str = "BF16",
    gen_num_per_cycle: int = 1,
    extra_common_env: Optional[Dict[str, str]] = None,
) -> Tuple[MagaServerManager, MagaServerManager]:
    """Create paired pure-PD Prefill and Decode MagaServerManager instances."""
    role_size = len(prefill_gpus)
    if len(decode_gpus) != role_size:
        raise ValueError(
            f"Prefill ({role_size}) and Decode ({len(decode_gpus)}) GPU counts must match"
        )
    if set(prefill_gpus) & set(decode_gpus):
        raise ValueError(
            f"Prefill GPUs {list(prefill_gpus)} and Decode GPUs {list(decode_gpus)} must be disjoint"
        )
    if tp_size != role_size:
        raise ValueError(f"tp_size ({tp_size}) must match role GPU count ({role_size})")

    route = ServiceRoute(
        service_id=service_id,
        use_local=True,
        role_endpoints=[
            GroupEndPoint(
                group="default",
                prefill_endpoint=EndPoint(
                    type="Vipserver",
                    address=f"127.0.0.1:{prefill_port}",
                    protocol="http",
                    path="/",
                ),
                decode_endpoint=EndPoint(
                    type="Vipserver",
                    address=f"127.0.0.1:{decode_port}",
                    protocol="http",
                    path="/",
                ),
            )
        ],
    )

    common_env: Dict[str, str] = {
        "LOAD_PYTHON_MODEL": "1",
        "WORLD_SIZE": str(role_size),
        "REMOTE_RPC_SERVER_IP": "127.0.0.1",
        "QWEN4_5_LOAD_WORKERS": str(load_workers),
        "ACCL_USE_NICS": rdma_hcas,
    }
    if extra_common_env:
        common_env.update(extra_common_env)

    common_args = (
        f"--tp_size {tp_size} "
        f"--ep_size {ep_size} "
        f"--dp_size 1 "
        f"--world_size {role_size} "
        f"--act_type BF16 "
        f"--seq_size_per_block {seq_size_per_block} "
        f"--max_seq_len {max_seq_len} "
        f"--max_context_batch_size {max_context_batch_size} "
        f"--concurrency_limit {concurrency_limit} "
        f"--cache_store_rdma_mode 1 "
        f"--cache_store_rdma_connect_timeout_ms {rdma_connect_timeout_ms} "
        f"--use_local 1 "
        f"--load_cache_timeout_ms 120000 "
        f"--reserver_runtime_mem_mb {reserver_runtime_mem_mb} "
        f"--warm_up 0"
    )
    if kv_cache_mem_mb is not None:
        common_args += f" --kv_cache_mem_mb {kv_cache_mem_mb}"
    sp_type_clean = (sp_type or "").strip()
    if sp_type_clean and sp_type_clean.lower() != "none":
        common_args += (
            f" --sp_type {sp_type_clean}"
            f" --sp_model_type {sp_model_type}"
            f" --sp_act_type {sp_act_type}"
            f" --gen_num_per_cycle {int(gen_num_per_cycle)}"
        )
        if sp_checkpoint_path:
            common_args += f" --sp_checkpoint_path {sp_checkpoint_path}"

    prefill_args = (
        f"{common_args} "
        f"--role_type PREFILL "
        f"--enable_cuda_graph 0 "
        f"--reuse_cache 1 "
        f"--enable_device_cache 1 "
        f"--enable_memory_cache 1 "
        f"--memory_cache_size_mb {memory_cache_size_mb}"
    )
    if force_host_demotion:
        prefill_args += (
            " --block_tree_device_evict_low_watermark_ratio 1e-12"
            " --block_tree_device_evict_high_watermark_ratio 2e-12"
        )

    decode_cg_flag = 1 if enable_decode_cuda_graph else 0
    decode_args = (
        f"{common_args} "
        f"--role_type DECODE "
        f"--enable_cuda_graph {decode_cg_flag} "
        f"--reuse_cache 0 "
        f"--enable_device_cache 1 "
        f"--enable_memory_cache 0"
    )

    suffix = f"_{role_name_suffix}" if role_name_suffix else ""
    route_json = route.model_dump_json()

    prefill_mgr = MagaServerManager(
        env_args={
            **common_env,
            "REMOTE_SERVER_PORT": str(decode_port),
            "MODEL_SERVICE_CONFIG": route_json,
        },
        device_ids=list(prefill_gpus),
        port=str(prefill_port),
        role_name=f"qwen4_5_prefill{suffix}",
        smoke_args_str=prefill_args,
    )
    decode_mgr = MagaServerManager(
        env_args={
            **common_env,
            "REMOTE_SERVER_PORT": str(prefill_port),
            "MODEL_SERVICE_CONFIG": route_json,
        },
        device_ids=list(decode_gpus),
        port=str(decode_port),
        role_name=f"qwen4_5_decode{suffix}",
        smoke_args_str=decode_args,
    )
    return prefill_mgr, decode_mgr


def start_pure_pd_and_wait_rpc(
    prefill_mgr: MagaServerManager,
    decode_mgr: MagaServerManager,
    *,
    ckpt_path: str,
    role_size: int,
    startup_timeout_s: int,
) -> None:
    """Start Prefill + Decode servers in parallel and wait for /health + model RPC ports."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        futs = [
            executor.submit(
                mgr.start_server,
                model_path=ckpt_path,
                model_type="qwen4_5",
                timeout=startup_timeout_s,
            )
            for mgr in (prefill_mgr, decode_mgr)
        ]
        for fut in futs:
            if not fut.result():
                raise RuntimeError("Pure-PD server startup failed /health check")

    for role_name, base_port in (("prefill", prefill_mgr.port), ("decode", decode_mgr.port)):
        for rank in range(role_size):
            wait_tcp_port_ready(
                "127.0.0.1",
                int(base_port) + rank * MIN_WORKER_INFO_PORT_NUM + 1,
                timeout_s=120.0,
                label=f"{role_name} rank {rank} model rpc",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch pure PD-disaggregated Qwen 4.5-Flash RTP-LLM servers (PREFILL + DECODE)."
    )
    parser.add_argument(
        "--ckpt-path",
        default=DEFAULT_3FS_CKPT,
        help="Path to Qwen 4.5-Flash checkpoint directory (read-only).",
    )
    parser.add_argument(
        "--prefill-gpus",
        default="0,1,2,3",
        help="Comma-separated GPU IDs assigned to ROLE_TYPE=PREFILL.",
    )
    parser.add_argument(
        "--decode-gpus",
        default="4,5,6,7",
        help="Comma-separated GPU IDs assigned to ROLE_TYPE=DECODE.",
    )
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--ep-size", type=int, default=None)
    parser.add_argument("--prefill-port", type=int, default=28000)
    parser.add_argument("--decode-port", type=int, default=29000)
    parser.add_argument("--seq-size-per-block", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-context-batch-size", type=int, default=8)
    parser.add_argument("--concurrency-limit", type=int, default=16)
    parser.add_argument(
        "--enable-decode-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=(os.environ.get("QWEN4_5_ENABLE_DECODE_CUDA_GRAPH", "1").strip() == "1"),
    )
    parser.add_argument("--reserver-runtime-mem-mb", type=int, default=16384)
    parser.add_argument("--memory-cache-size-mb", type=int, default=32768)
    parser.add_argument("--kv-cache-mem-mb", type=int, default=None)
    parser.add_argument("--rdma-hcas", default=None)
    parser.add_argument("--rdma-connect-timeout-ms", type=int, default=30000)
    parser.add_argument("--load-workers", type=int, default=8)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--sp-type", default="")
    parser.add_argument("--sp-model-type", default="qwen4_5_dspark")
    parser.add_argument("--sp-checkpoint-path", default="")
    parser.add_argument("--sp-act-type", default="BF16")
    parser.add_argument("--gen-num-per-cycle", type=int, default=8)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    if not Path(args.ckpt_path).is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.ckpt_path}")

    prefill_gpus = _parse_gpu_list(args.prefill_gpus)
    decode_gpus = _parse_gpu_list(args.decode_gpus)
    role_size = len(prefill_gpus)
    tp_size = args.tp_size if args.tp_size is not None else role_size
    ep_size = args.ep_size if args.ep_size is not None else tp_size

    rdma_hcas, link_layer = select_single_layer_rdma_hcas(args.rdma_hcas)
    logging.info("Selected RDMA HCAs (%s): %s", link_layer, rdma_hcas)

    prefill_mgr, decode_mgr = build_pure_pd_managers(
        service_id="qwen4_5_pd_server",
        prefill_gpus=prefill_gpus,
        decode_gpus=decode_gpus,
        prefill_port=args.prefill_port,
        decode_port=args.decode_port,
        tp_size=tp_size,
        ep_size=ep_size,
        seq_size_per_block=args.seq_size_per_block,
        max_seq_len=args.max_seq_len,
        reserver_runtime_mem_mb=args.reserver_runtime_mem_mb,
        load_workers=args.load_workers,
        rdma_hcas=rdma_hcas,
        memory_cache_size_mb=args.memory_cache_size_mb,
        kv_cache_mem_mb=args.kv_cache_mem_mb,
        rdma_connect_timeout_ms=args.rdma_connect_timeout_ms,
        max_context_batch_size=args.max_context_batch_size,
        concurrency_limit=args.concurrency_limit,
        enable_decode_cuda_graph=args.enable_decode_cuda_graph,
        sp_type=args.sp_type,
        sp_model_type=args.sp_model_type,
        sp_checkpoint_path=args.sp_checkpoint_path,
        sp_act_type=args.sp_act_type,
        gen_num_per_cycle=args.gen_num_per_cycle,
    )

    stop_requested = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        logging.info("Received signal %d; shutting down pure-PD servers...", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        start_pure_pd_and_wait_rpc(
            prefill_mgr,
            decode_mgr,
            ckpt_path=args.ckpt_path,
            role_size=role_size,
            startup_timeout_s=args.startup_timeout,
        )
        print(
            f"\n[Qwen 4.5-Flash Pure PD Ready]\n"
            f"  RDMA HCAs    : {rdma_hcas} ({link_layer}, timeout={args.rdma_connect_timeout_ms}ms)\n"
            f"  Prefill Cache: device=1 memory=1 ({args.memory_cache_size_mb} MiB), "
            f"max_context_batch_size={args.max_context_batch_size}\n"
            f"  Prefill HTTP : http://127.0.0.1:{args.prefill_port} (GPUs={prefill_gpus})\n"
            f"  Decode  HTTP : http://127.0.0.1:{args.decode_port} (GPUs={decode_gpus})\n"
            f"  Sample curl  :\n"
            f"    curl -s http://127.0.0.1:{args.prefill_port}/ -H \"Content-Type: application/json\" "
            "-d '{\"prompt\": \"中国的首都是\", \"generate_config\": {\"max_new_tokens\": 24, \"top_k\": 1, \"temperature\": 0.0, \"can_use_pd_separation\": true}}'\n",
            flush=True,
        )
        while not stop_requested:
            time.sleep(1.0)
    finally:
        decode_mgr.stop_server()
        prefill_mgr.stop_server()

    return 0


if __name__ == "__main__":
    sys.exit(main())
