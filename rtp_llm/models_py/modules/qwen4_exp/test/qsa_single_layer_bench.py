"""Single-layer QSA prefix-reuse prefill bench: DtoH sync points + layer wall.

Runs the production single-layer forward on one H20-class GPU --
``Qwen4ExpAttention.forward`` + ``SparseGqaFmhaImpl`` + the real fused main KV
writer -- with a page-aligned prefix-reuse prefill input shaped like the
timeline's full3 workload: one request with a ~4900-token prefix (snapped to
the 128-token page grid) plus ~3600 new tokens, 12 q heads / 1 kv head /
D=256, 128-token pages, indexer ratio 4 with a 2048-token budget (selected
width 2051).

The measured layer covers both stages the QSA prep-bubble plan targets:
the prep segment (indexer projection, side-cache plan/writer, paged score,
top-k) and the fmha-prep segment (incremental-prefill input planning, main KV
write, sparse paged attention).  Reports, per forward:

* ``Memcpy DtoH`` count and byte distribution from a torch.profiler chrome
  trace, plus ``cudaStreamSynchronize`` / ``cudaMemcpyAsync`` call counts;
* per-layer wall from CUDA events and host perf counters;
* the top DtoH sources by their innermost ``cpu_op`` (correlation-linked), so
  P1's validation demotion can be verified by a before/after diff of this
  table.

``RTP_LLM_QSA_STRICT_VALIDATION=1`` restores the legacy host-synchronous
validation and is reported in the header; compare both modes on the same
commit to attribute the DtoH delta to the validation chain.

Run directly (GPU host, repo on PYTHONPATH ahead of the installed wheel):

    python qsa_single_layer_bench.py [--iters 20] [--warmup 5]
        [--profile-iters 10] [--prefix 4900] [--new-tokens 3600]
"""

import argparse
import json
import os
import statistics
import tempfile
import time
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile

# Bootstrap native .so search paths (rtp_llm/libs) before importing the pybind
# modules directly; in wheel-based dev containers rtp_llm.ops.__init__ is what
# appends the libs dir to sys.path.
import rtp_llm.ops  # noqa: F401

from librtp_compute_ops import LayerKVCache, PyAttentionInputs, get_typemeta
from libth_transformer_config import AttentionConfigs, RopeStyle

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc import qwen3_next, qwen4_exp
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import SparseGqaFmhaImpl
from rtp_llm.utils.model_weight import W

HIDDEN = 2048
Q_HEADS, KV_HEADS, HEAD_DIM = 12, 1, 256
PAGE = 128
INDEXER_HEADS, INDEXER_HEAD_DIM = 16, 128
RATIO, BUDGET = 4, 2048
Q_WIDTH = Q_HEADS * HEAD_DIM
QKV_WIDTH = Q_WIDTH + 2 * KV_HEADS * HEAD_DIM


def _attn_configs(max_seq_len: int) -> AttentionConfigs:
    """Production-shaped pybind attention config (12 q / 1 kv / D=256 / 128)."""
    configs = AttentionConfigs()
    configs.head_num, configs.kv_head_num, configs.size_per_head = (
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
    )
    configs.kernel_tokens_per_block, configs.max_seq_len = PAGE, max_seq_len
    configs.need_rope_kv_cache, configs.use_logn_attn = True, False
    configs.is_sparse, configs.use_sparse_gqa_fmha, configs.use_mla = True, True, False
    rope = configs.rope_config
    rope.style = RopeStyle.Mrope
    rope.dim, rope.base, rope.scale = 64, 1_000_000.0, 1.0
    rope.index_factor = 3
    # MRoPE sections must sum to dim // 2 = 32 rotary pairs; interleaved slots
    # cap H at len(range(1, 32, 3)) = 11 and W at len(range(2, 32, 3)) = 10.
    rope.mrope_dim1, rope.mrope_dim2, rope.mrope_dim3 = 11, 11, 10
    rope.mrope_interleaved, rope.indexer_is_neox_style = True, True
    return configs


def _attention(attn_configs: AttentionConfigs, device: torch.device):
    """Build the real Qwen4ExpAttention with minimal projection modules.

    The Qwen3Next base constructor needs the full weight factory; patch it out
    (same pattern as qwen4_exp_layer_test) and attach plain BF16 projections.
    The indexer, rope config and forward orchestration are production code.
    """

    def _fake_base_init(module, *args, **kwargs):
        torch.nn.Module.__init__(module)

    weights = {
        W.qwen4_indexer_qk_proj_w: torch.randn(
            (INDEXER_HEADS + 1) * INDEXER_HEAD_DIM,
            HIDDEN,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.02,
        W.qwen4_indexer_q_ln_gamma: torch.ones(
            INDEXER_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        * 0.1,
        W.qwen4_indexer_k_ln_gamma: torch.ones(
            INDEXER_HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        * 0.1,
    }
    model_config = SimpleNamespace(
        enable_qwen4_qsa=True,
        is_mtp=False,
        _qwen4_indexer_head_dim=INDEXER_HEAD_DIM,
        _qwen4_indexer_compress_ratio=RATIO,
        _qwen4_indexer_budget=BUDGET,
        _qwen4_indexer_kv_heads=1,
        attn_config=SimpleNamespace(
            indexer_head_num=INDEXER_HEADS,
            rope_config=attn_configs.rope_config,
        ),
    )
    with patch.object(
        qwen3_next.Qwen3NextAttention, "__init__", _fake_base_init
    ):
        attention = qwen4_exp.Qwen4ExpAttention(
            attn_configs,
            SimpleNamespace(),
            weights,
            1e-6,
            model_config,
            layer_idx=0,
        )
    attention.qkv_proj = torch.nn.Linear(
        HIDDEN, QKV_WIDTH, bias=False, device=device, dtype=torch.bfloat16
    )
    attention.gate = torch.nn.Linear(
        HIDDEN, Q_WIDTH, bias=False, device=device, dtype=torch.bfloat16
    )
    attention.o_proj = torch.nn.Linear(
        Q_WIDTH, HIDDEN, bias=False, device=device, dtype=torch.bfloat16
    )
    attention.qk_fuse_norm = None
    attention.tp_size = 1
    return attention


class _Fixture:
    """Pinned-host metadata plus device mirrors for one prefix-reuse request."""

    def __init__(self, device: torch.device, prefix: int, new_tokens: int):
        torch.manual_seed(20261102)
        self.prefix, self.new_tokens = prefix, new_tokens
        total = prefix + new_tokens
        blocks = (total + PAGE - 1) // PAGE  # main/state pages (128-token both)
        kv_entries_per_block = PAGE // RATIO  # compressed entries per KV page
        pool_blocks = blocks + 1  # block id zero stays reserved

        input_lengths = torch.tensor([new_tokens], dtype=torch.int32)
        prefix_lengths = torch.tensor([prefix], dtype=torch.int32)
        cu_host = torch.tensor([0, new_tokens], dtype=torch.int32)
        cu_kv_host = torch.tensor([0, total], dtype=torch.int32)
        # One shared logical->physical table per pool family. With 128-token
        # pages the main, indexer-KV and indexer-state tables all have the
        # same width; ids 1..blocks stay in range for every pool.
        table_host = torch.arange(1, blocks + 1, dtype=torch.int32).unsqueeze(0)
        positions = torch.arange(prefix, total, device=device, dtype=torch.int32)
        combo_position_ids = (
            positions.unsqueeze(1).expand(-1, 3).reshape(-1).contiguous()
        )

        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        inputs.is_target_verify = False
        inputs.is_cuda_graph = False
        inputs.is_exact_cuda_graph_batch = False
        inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
        # Pinned host mirrors (bare names, OpDefs.h convention).
        inputs.input_lengths = input_lengths.pin_memory()
        inputs.prefix_lengths = prefix_lengths.pin_memory()
        inputs.sequence_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
        inputs.cu_seqlens = cu_host.pin_memory()
        inputs.kv_cache_kernel_block_id = table_host.pin_memory()
        inputs.kv_cache_block_id = table_host.pin_memory()
        # Device counterparts (_device suffix). cu_kv_seqlens is device-only
        # (no host mirror, OpDefs.h).
        inputs.cu_seqlens_device = inputs.cu_seqlens.to(device, non_blocking=True)
        inputs.cu_kv_seqlens_device = cu_kv_host.pin_memory().to(
            device, non_blocking=True
        )
        inputs.kv_cache_kernel_block_id_device = (
            table_host.to(device, non_blocking=True).contiguous()
        )
        inputs.kv_cache_block_id_device = (
            table_host.to(device, non_blocking=True).contiguous()
        )
        inputs.combo_position_ids = combo_position_ids
        inputs.context_total_kv_length = total
        inputs.total_tokens = new_tokens
        self.inputs = inputs

        self.main_cache = LayerKVCache(
            (torch.randn(pool_blocks, 2, KV_HEADS, PAGE, HEAD_DIM, device=device) * 0.1)
            .to(torch.bfloat16)
            .contiguous(),
            PAGE,
            layer_id=0,
            tag="default",
        )
        self.kv_cache = LayerKVCache(
            torch.zeros(
                pool_blocks, kv_entries_per_block * INDEXER_HEAD_DIM * 2,
                dtype=torch.uint8, device=device,
            ),
            PAGE,
            layer_id=0,
            tag=INDEXER_KV_TAG,
        )
        self.state_cache = LayerKVCache(
            torch.zeros(
                pool_blocks, 2 * RATIO * INDEXER_HEAD_DIM,
                dtype=torch.float32, device=device,
            ),
            PAGE,
            layer_id=0,
            tag=INDEXER_STATE_TAG,
        )
        self.runtime = Qwen4ExpQSARuntimeContext(
            main_cache=self.main_cache,
            main_inputs=inputs,
            indexer_kv_cache=self.kv_cache,
            indexer_kv_inputs=inputs,
            indexer_state_cache=self.state_cache,
            indexer_state_inputs=inputs,
            is_mtp_draft=False,
        )
        self.hidden_states = torch.randn(
            new_tokens, HIDDEN, device=device, dtype=torch.bfloat16
        )

    def forward(self, attention, fmha):
        return attention(
            self.hidden_states,
            fmha,
            self.main_cache,
            self.inputs,
            qsa_runtime=self.runtime,
        )


def _measure_wall(fixture, attention, fmha, warmup, iters):
    for _ in range(warmup):
        fixture.forward(attention, fmha)
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    walls = []
    torch.cuda.synchronize()
    for start, end in events:
        clock = time.perf_counter()
        start.record()
        fixture.forward(attention, fmha)
        end.record()
        walls.append((time.perf_counter() - clock) * 1000.0)
    torch.cuda.synchronize()
    event_ms = [start.elapsed_time(end) for start, end in events]
    return event_ms, walls


def _parse_trace(path):
    """Summarize DtoH memcpys, runtime sync APIs and cpu_op attribution."""
    with open(path) as handle:
        events = json.load(handle).get("traceEvents", [])
    correlation_to_op = {}
    runtime_counts = Counter()
    memcpy_events = []
    busy_us = 0
    for event in events:
        category = event.get("cat")
        name = event.get("name", "")
        if category == "cpu_op":
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                correlation_to_op.setdefault(correlation, name)
        elif category == "cuda_runtime":
            if any(
                api in name
                for api in (
                    "cudaStreamSynchronize",
                    "cudaDeviceSynchronize",
                    "cudaMemcpyAsync",
                    "cudaEventSynchronize",
                    "cudaStreamWaitEvent",
                    "cudaEventRecord",
                )
            ):
                runtime_counts[name] += 1
        elif category == "gpu_memcpy" and "DtoH" in name:
            memcpy_events.append(event)
        elif category in ("kernel",) and event.get("ph") == "X":
            busy_us += event.get("dur", 0) or 0

    by_source = Counter()
    bytes_by_source = Counter()
    byte_buckets = Counter()
    total_bytes = 0
    for event in memcpy_events:
        correlation = event.get("args", {}).get("correlation")
        source = correlation_to_op.get(correlation, "<unlinked>")
        byte_count = int(event.get("args", {}).get("bytes") or 0)
        by_source[source] += 1
        bytes_by_source[source] += byte_count
        total_bytes += byte_count
        bucket = 8
        while bucket < byte_count:
            bucket *= 8
        byte_buckets[f"<= {bucket} B"] += 1
    return {
        "memcpy_events": memcpy_events,
        "by_source": by_source,
        "bytes_by_source": bytes_by_source,
        "byte_buckets": byte_buckets,
        "total_bytes": total_bytes,
        "runtime_counts": runtime_counts,
        "busy_us": busy_us,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument("--prefix", type=int, default=4900)
    parser.add_argument("--new-tokens", type=int, default=3600)
    parser.add_argument("--keep-trace", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("qsa_single_layer_bench requires a CUDA GPU (H20 target)")
    device = torch.device("cuda")
    torch.cuda.init()
    # Prefix reuse requires a page-aligned prefix (the production contract).
    if args.prefix % PAGE:
        aligned = args.prefix - args.prefix % PAGE
        print(
            f"note: prefix {args.prefix} is not {PAGE}-token page aligned; "
            f"snapping down to {aligned}"
        )
        args.prefix = aligned
    total = args.prefix + args.new_tokens

    attn_configs = _attn_configs(max_seq_len=total)
    attention = _attention(attn_configs, device)
    fixture = _Fixture(device, args.prefix, args.new_tokens)
    fmha = SparseGqaFmhaImpl(attn_configs, fixture.inputs)

    output = fixture.forward(attention, fmha)
    if tuple(output.shape) != (args.new_tokens, HIDDEN):
        raise SystemExit(f"unexpected output shape {tuple(output.shape)}")

    event_ms, host_ms = _measure_wall(
        fixture, attention, fmha, args.warmup, args.iters
    )

    trace_path = os.path.join(
        tempfile.mkdtemp(prefix="qsa_single_layer_bench_"), "trace.json"
    )
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(args.profile_iters):
            fixture.forward(attention, fmha)
    prof.export_chrome_trace(trace_path)
    summary = _parse_trace(trace_path)

    forwards = args.profile_iters
    gpu_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    strict = os.environ.get("RTP_LLM_QSA_STRICT_VALIDATION", "")
    print("=" * 72)
    print(f"device={gpu_name} (sm{capability[0]}{capability[1]}) torch={torch.__version__}")
    print(
        f"geometry: 1 request, prefix={args.prefix} + new={args.new_tokens} tokens,"
        f" {Q_HEADS}q/{KV_HEADS}kv D={HEAD_DIM} page={PAGE},"
        f" indexer ratio={RATIO} budget={BUDGET} (selected {BUDGET + RATIO - 1})"
    )
    print(
        f"iters: warmup={args.warmup} wall={args.iters} profile={args.profile_iters}"
        f" | RTP_LLM_QSA_STRICT_VALIDATION={strict or '(off)'}"
    )
    print("-" * 72)
    print(
        f"per-forward DtoH memcpys: {len(summary['memcpy_events']) / forwards:.1f}"
        f" ({summary['total_bytes'] / forwards:.0f} B/forward)"
    )
    for api, count in sorted(summary["runtime_counts"].items()):
        print(f"  {api}: {count / forwards:.1f}/forward")
    if summary["byte_buckets"]:
        buckets = ", ".join(
            f"{name}: {count}" for name, count in summary["byte_buckets"].items()
        )
        print(f"  DtoH byte buckets: {buckets}")
    print("-" * 72)
    print("top DtoH sources (innermost cpu_op via correlation):")
    for source, count in summary["by_source"].most_common(12):
        per_forward = count / forwards
        bytes_forward = summary["bytes_by_source"][source] / forwards
        print(f"  {per_forward:8.2f}/forward  {bytes_forward:10.0f} B/forward  {source}")
    print("-" * 72)
    print(
        "layer wall (CUDA events): "
        f"mean={statistics.mean(event_ms):.3f} ms  p50={statistics.median(event_ms):.3f} ms"
        f"  min={min(event_ms):.3f} ms  max={max(event_ms):.3f} ms"
    )
    print(
        "layer wall (host clock): "
        f"mean={statistics.mean(host_ms):.3f} ms  p50={statistics.median(host_ms):.3f} ms"
    )
    profile_busy_ms = summary["busy_us"] / 1000.0 / forwards
    mean_event_ms = statistics.mean(event_ms)
    print(
        f"profiler busy (kernels only): {profile_busy_ms:.3f} ms/forward;"
        f" event-wall idle share ~{(1 - profile_busy_ms / mean_event_ms) * 100:.1f}%"
    )
    if args.keep_trace:
        print(f"trace kept at: {trace_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
