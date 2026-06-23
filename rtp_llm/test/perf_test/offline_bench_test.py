"""Offline throughput benchmark — standalone entry point.

Currently supports Qwen3.5 series models only.

Flow:
  1. Parse args, estimate capacity
  2. Start engine server
  3. Run benchmark, collect metrics
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

from rtp_llm.test.perf_test.dataset import extract_arg
from rtp_llm.test.perf_test.offline_runner import (
    OfflineBenchConfig,
    OfflineRunner,
)
from rtp_llm.test.perf_test.perf_config import resolve_perf_engine_paths
from rtp_llm.test.perf_test.server import EngineServer
from rtp_llm.test.utils.coredump_util import summarize_and_cleanup_coredumps


_SUPPORTED_ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}


# ---------------------------------------------------------------------------
# Capacity Estimation (Qwen3.5 hybrid attention)
# ---------------------------------------------------------------------------


@dataclass
class EstimatedCapacity:
    gpu_total_bytes: int = 0
    total_weight_bytes: int = 0
    weight_per_gpu_bytes: int = 0
    kv_cache_mem_bytes: int = 0
    kv_bytes_per_token: int = 0
    per_request_overhead: int = 0
    avg_tokens_per_request: float = 0.0
    kv_capacity_per_rank: int = 0
    kv_capacity: int = 0
    concurrency_limit: int = 0
    overload_ratio: float = 1.0
    full_attn_layers: int = 0
    linear_layers: int = 0
    tp_size: int = 1
    dp_size: int = 1

    def print_summary(self):
        lines = [
            "================== Capacity Estimation ==================",
            f"GPU Total:             {self.gpu_total_bytes / 1024**3:.2f} GiB",
            f"Total Weight (disk):   {self.total_weight_bytes / 1024**3:.2f} GiB",
            f"Weight per GPU:        {self.weight_per_gpu_bytes / 1024**3:.2f} GiB (÷ tp_size={self.tp_size})",
            f"KV Cache per GPU:      {self.kv_cache_mem_bytes / 1024**3:.2f} GiB",
            f"Full Attn Layers:      {self.full_attn_layers}",
            f"Linear Layers:         {self.linear_layers}",
            f"KV Bytes/Token:        {self.kv_bytes_per_token:,} (full attn only, per GPU)",
            f"Per-Request Overhead:  {self.per_request_overhead:,} bytes (linear state, per GPU)",
            f"Avg Tokens/Request:    {self.avg_tokens_per_request:.0f}",
            f"Parallelism:           tp_size={self.tp_size}, dp_size={self.dp_size}",
            f"KV Capacity per rank:  {self.kv_capacity_per_rank} concurrent requests",
            f"KV Capacity (total):   {self.kv_capacity} (× dp_size={self.dp_size})",
            f"Overload Ratio:        {self.overload_ratio}x",
            f"Concurrency Limit:     {self.concurrency_limit}",
            "=========================================================",
        ]
        for line in lines:
            logging.info(line)


def _validate_model_config(checkpoint_path: str) -> Dict[str, Any]:
    """Load and validate model config. Raises if not a supported Qwen3.5 model."""
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"config.json not found at {checkpoint_path}")

    with open(config_path) as f:
        raw_cfg = json.load(f)

    architectures = set(raw_cfg.get("architectures", []))
    if not architectures & _SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported model architecture: {architectures}. "
            f"Only Qwen3.5 series supported: {_SUPPORTED_ARCHITECTURES}"
        )

    text_cfg = raw_cfg.get("text_config", raw_cfg)
    required_fields = [
        "num_hidden_layers", "num_key_value_heads", "head_dim",
        "full_attention_interval", "linear_num_key_heads",
        "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
    ]
    missing = [f for f in required_fields if f not in text_cfg]
    if missing:
        raise ValueError(f"Model config missing required fields: {missing}")

    return raw_cfg


def estimate_capacity(
    checkpoint_path: str,
    bench_config: OfflineBenchConfig,
    tp_size: int = 1,
    dp_size: int = 1,
    overload_ratio: float = 1.0,
) -> EstimatedCapacity:
    """Estimate concurrency for Qwen3.5 hybrid attention models.

    Parallelism model (tp_size × dp_size GPUs total):
      - TP: weights and KV heads split across tp_size GPUs within each DP rank.
      - DP: each DP rank is an independent inference instance on its own set of
        tp_size GPUs, handling separate requests with its own KV cache.
      - Total capacity = per-rank capacity × dp_size.

    KV memory model (per GPU):
      - Full attention layers: per-token KV growth (2 × kv_heads/tp × head_dim × dtype)
      - Linear attention layers: fixed per-request state (SSM + conv, heads/tp)
      - per_rank_capacity = kv_mem_per_gpu / (per_token × avg_tokens + per_request_overhead)
    """
    import torch

    raw_cfg = _validate_model_config(checkpoint_path)
    text_cfg = raw_cfg.get("text_config", raw_cfg)

    est = EstimatedCapacity(overload_ratio=overload_ratio, tp_size=tp_size, dp_size=dp_size)
    est.gpu_total_bytes = torch.cuda.get_device_properties(0).total_memory

    # Weight size (disk total → per-GPU = total / tp_size)
    total_weight = 0
    for root, _dirs, files in os.walk(checkpoint_path):
        for fname in files:
            if fname.endswith((".safetensors", ".bin", ".pt")):
                total_weight += os.path.getsize(os.path.join(root, fname))
    est.total_weight_bytes = total_weight
    est.weight_per_gpu_bytes = total_weight // max(1, tp_size)

    # KV cache memory budget (per GPU)
    pytorch_overhead = int(600 * 1024 * 1024)
    runtime_reserve = max(2 * 1024**3, int(est.gpu_total_bytes * 0.05))
    device_available = est.gpu_total_bytes - est.weight_per_gpu_bytes - pytorch_overhead
    est.kv_cache_mem_bytes = max(0, device_available - runtime_reserve)

    # Layer counts
    num_layers = text_cfg["num_hidden_layers"]
    interval = text_cfg["full_attention_interval"]
    est.full_attn_layers = num_layers // interval
    est.linear_layers = num_layers - est.full_attn_layers

    # Per-token cost (full attention layers only, per GPU after TP split)
    kv_dtype_str = text_cfg.get("dtype") or text_cfg.get("torch_dtype") or "bfloat16"
    dtype_map = {"bfloat16": 2, "float16": 2, "float32": 4, "float8_e4m3fn": 1, "int8": 1}
    dtype = dtype_map.get(str(kv_dtype_str).lower(), 2)
    num_kv_heads = text_cfg["num_key_value_heads"]
    head_dim = text_cfg["head_dim"]
    kv_heads_per_tp = max(1, num_kv_heads // tp_size)
    est.kv_bytes_per_token = 2 * kv_heads_per_tp * head_dim * dtype * est.full_attn_layers

    # Per-request overhead (linear attention SSM + conv state, per GPU after TP split)
    num_k_heads = text_cfg["linear_num_key_heads"]
    num_v_heads = text_cfg.get("linear_num_value_heads", num_k_heads)
    local_k_heads = max(1, num_k_heads // tp_size)
    local_v_heads = max(1, num_v_heads // tp_size)
    head_k_dim = text_cfg["linear_key_head_dim"]
    head_v_dim = text_cfg["linear_value_head_dim"]
    conv_kernel = text_cfg["linear_conv_kernel_dim"]

    ssm_state = local_v_heads * head_k_dim * head_v_dim * dtype
    qkv_size = head_k_dim * local_k_heads * 2 + head_v_dim * local_v_heads
    conv_state = max(0, conv_kernel - 1) * qkv_size * dtype
    est.per_request_overhead = (ssm_state + conv_state) * est.linear_layers

    # Solve for capacity
    est.avg_tokens_per_request = (
        (bench_config.input_len_min + bench_config.input_len_max) / 2
        + (bench_config.output_len_min + bench_config.output_len_max) / 2
    )
    denom = est.kv_bytes_per_token * est.avg_tokens_per_request + est.per_request_overhead
    if denom > 0:
        est.kv_capacity_per_rank = int(est.kv_cache_mem_bytes / denom)
    est.kv_capacity = est.kv_capacity_per_rank * dp_size
    est.concurrency_limit = max(1, int(est.kv_capacity * overload_ratio))
    return est


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------


def parse_offline_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="RTP-LLM offline throughput benchmark (Qwen3.5 series). "
        "Unrecognized arguments are forwarded to the engine server.",
    )

    bench = parser.add_argument_group("benchmark")
    bench.add_argument("--duration", type=int, default=300,
                       help="Duration in seconds to dispatch requests")
    bench.add_argument("--drain_timeout", type=int, default=0,
                       help="Max seconds to wait for in-flight requests after dispatch ends. "
                            "0 = wait forever (default). "
                            "When exceeded, remaining requests are cancelled and report is generated.")
    bench.add_argument("--input_len_min", type=int, default=512)
    bench.add_argument("--input_len_max", type=int, default=2048)
    bench.add_argument("--output_len_min", type=int, default=64)
    bench.add_argument("--output_len_max", type=int, default=512)
    bench.add_argument("--prefix_groups", type=int, default=1)
    bench.add_argument("--prefix_len", type=int, default=0)
    bench.add_argument("--dump_workload", type=str, default="")
    bench.add_argument("--result_dir", type=str,
                       default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "./offline_bench_results"))
    bench.add_argument("--overload_ratio", type=float, default=1.0,
                       help="Multiply KV capacity estimate by this ratio")
    bench.add_argument("--auto_concurrency", action="store_true", default=False,
                       help="Auto-estimate concurrency_limit from GPU/model")
    bench.add_argument("--max_seq_len", type=int, default=32768,
                       help="Max sequence length (context + output). Defaults to 32768.")
    bench.add_argument("--concurrency_limit", type=int, default=256,
                       help="Engine concurrency limit (ignored if --auto_concurrency)")
    bench.add_argument("--profile", action="store_true", default=False,
                       help="Enable GPU timeline profiling during steady state")
    bench.add_argument("--profile_steps", type=int, default=50,
                       help="Number of steps to profile (default 50)")
    bench.add_argument("--dp_size", type=int, default=1)
    bench.add_argument("--partial", type=int, default=1)
    bench.add_argument("--checkpoint_path", type=str, default="",
                       help="Model checkpoint path")
    bench.add_argument("--tokenizer_path", type=str, default="",
                       help="Tokenizer path (defaults to checkpoint_path)")
    bench.add_argument("--tp_size", type=int, default=1)

    args, remaining = parser.parse_known_args()
    return args, remaining


def main() -> str:
    from rtp_llm.config.log_config import setup_logging

    setup_logging()

    args, remaining = parse_offline_args()
    os.makedirs(args.result_dir, exist_ok=True)

    checkpoint_path = args.checkpoint_path or os.environ.get("CHECKPOINT_PATH", "")
    tokenizer_path = args.tokenizer_path or checkpoint_path or os.environ.get("TOKENIZER_PATH", "")
    tp_size = args.tp_size or 1

    if checkpoint_path and not extract_arg(remaining, "checkpoint_path"):
        remaining.extend(["--checkpoint_path", checkpoint_path])
    if tokenizer_path and not extract_arg(remaining, "tokenizer_path"):
        remaining.extend(["--tokenizer_path", tokenizer_path])
    if args.tp_size and not extract_arg(remaining, "tp_size"):
        remaining.extend(["--tp_size", str(tp_size)])

    # Resolve Hub IDs (e.g. "Qwen/Qwen3.5-35B-A3B") to local paths
    remaining = resolve_perf_engine_paths(remaining)
    checkpoint_path = extract_arg(remaining, "checkpoint_path") or checkpoint_path
    tokenizer_path = extract_arg(remaining, "tokenizer_path") or tokenizer_path

    EngineServer.propagate_engine_env(remaining)

    dump_workload = args.dump_workload
    if dump_workload and not os.path.isabs(dump_workload):
        dump_workload = os.path.join(args.result_dir, dump_workload)

    config = OfflineBenchConfig(
        input_len_min=args.input_len_min,
        input_len_max=args.input_len_max,
        output_len_min=args.output_len_min,
        output_len_max=args.output_len_max,
        prefix_groups=args.prefix_groups,
        prefix_len=args.prefix_len,
        duration_s=args.duration,
        drain_timeout_s=args.drain_timeout,
        concurrency_limit=args.concurrency_limit,
        dump_workload=dump_workload,
    )

    # Capacity estimation
    dp_size = args.dp_size or 1
    if args.auto_concurrency and checkpoint_path:
        est = estimate_capacity(
            checkpoint_path, config, tp_size=tp_size, dp_size=dp_size,
            overload_ratio=args.overload_ratio,
        )
        est.print_summary()
        args.concurrency_limit = est.concurrency_limit
        config.concurrency_limit = est.concurrency_limit

    max_seq_len = args.max_seq_len

    # Start engine and run benchmark
    server = EngineServer(args, remaining)
    server_started = False
    try:
        server.start(max_seq_len=max_seq_len, max_concurrency=args.concurrency_limit)
        server_started = True

        runner = OfflineRunner(
            port=server.port,
            config=config,
            tokenizer_path=tokenizer_path,
            result_dir=args.result_dir,
            profile=args.profile,
            profile_steps=args.profile_steps,
        )

        runner.run()
    finally:
        try:
            if server_started:
                server.stop()
        finally:
            summarize_and_cleanup_coredumps(args.result_dir)

    return args.result_dir


if __name__ == "__main__":
    main()
