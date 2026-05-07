"""Repro/regression harness for ``PyFA3PagedDraftPrefillAttnOp`` under
heterogeneous Q lengths and CUDA graph capture/replay.

Background
----------
``PyFA3PagedAttnOp`` (cuda_impl/py_fa3_paged.py) replaces
the legacy ``PyFlashinferPrefillPagedAttnOp`` driving the MTP draft-model
prefill step.  The legacy impl hit FlashInfer's ``plan/run`` aliasing bug
under CUDA graph capture (captured kernel reads stale ``plan_info``).
FA3 exposes a single fused launch, so per-call scalars are read from device
buffers at runtime.  The op stashes references to:

* ``cu_seqlens_q_buf`` (own GPU buffer, refreshed per replay from H2D
  transfer of ``input_lengths`` cumsum),
* ``cache_seqlens_buf`` (own GPU buffer, refreshed per replay from
  ``prefix_lengths + input_lengths``),
* ``page_table_ref`` → ``kv_cache_kernel_block_id_device`` (stable CG
  buffer maintained by C++ ``CudaGraphRunner::prepareInputs``).

Unlike target verify, draft prefill has variable Q lengths per request
(each batch occupies up to ``num_tokens_per_bs=4`` rows but the actual
draft-token count varies).  Q layout is *compact* (only sum(input_lengths)
real rows; trailing rows are stale).  The CG-captured kernel tile
allocation must therefore tolerate cu_seqlens_q changing between replays.

Coverage
--------
* ``test_homogeneous_q_eager_passes`` — baseline, all batches Q=4, eager
  (no CG): op output matches per-request reference.
* ``test_heterogeneous_q_eager_passes`` — eager + heterogeneous Q lengths
  (1, 4, 2, 3): op output matches per-request reference.
* ``test_heterogeneous_q_cg_replay_matches_eager`` — capture once at full
  Q=4-each, replay with mixed Q (1, 4, 2, 3); CG output must equal the
  eager output for the same inputs.
* ``test_multi_replay_varying_q_patterns`` — capture once, then 5 replays
  cycling through different mixed-Q patterns; each replay must match
  eager.
* ``test_cg_replay_reads_input_lengths_at_runtime`` — sanity probe that
  the captured FA3 kernel reads cu_seqlens_q at replay (changing Q lengths
  produces different output).
* ``test_padded_batch_capture_smaller_replay`` — capture at bs=8, replay
  at actual_bs=4 with padding rows zeroed; only active rows compared.

Run:
    bazelisk test //rtp_llm/models_py/modules/factory/attention/cuda_impl/test:test_fa3_draft_prefill_heterogeneous \
        --config=cuda12_9 --config=sm9x --test_output=all
"""

import logging
import math
import unittest
from types import SimpleNamespace
from typing import Any, List, Tuple

import torch
from base_attention_test import BaseAttentionTest, compare_tensors
from flash_attn_interface import flash_attn_with_kvcache

from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, get_typemeta

logging.basicConfig(level=logging.INFO, format="%(message)s")


# Match production smoke (Qwen3.5-27B per-TP-rank, MTP gen_num_per_cycle=3).
HEAD_NUM = 8
KV_HEAD_NUM = 1
HEAD_DIM = 256
PAGE_SIZE = 64
NUM_TOKENS_PER_BS = 4  # = gen_num_per_cycle + 1
MAX_SEQ_LEN = 28000


def _build_paged_kv_cache(
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> LayerKVCache:
    storage = torch.randn(
        num_pages, 2, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )
    cache = LayerKVCache()
    cache.kv_cache_base = storage
    return cache


def _build_block_table(
    cache_seqlens: List[int],
    page_size: int,
    max_blocks: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = len(cache_seqlens)
    table = torch.zeros((bs, max_blocks), dtype=torch.int32, device="cpu")
    block_offset = 1
    for i, cs in enumerate(cache_seqlens):
        n = math.ceil(cs / page_size)
        if n == 0:
            continue
        table[i, :n] = torch.arange(block_offset, block_offset + n, dtype=torch.int32)
        # Pad with last valid page id so capture-time over-reads stay in-bounds.
        table[i, n:].fill_(block_offset + n - 1)
        block_offset += n
    return table, table.to(device)


def _build_attn_inputs(
    *,
    cache_seqlens: List[int],
    input_lengths: List[int],
    block_id_host: torch.Tensor,
    block_id_device: torch.Tensor,
    is_cuda_graph: bool,
    dtype: torch.dtype,
    device: torch.device,
    zero_device_lengths: bool = False,
) -> Any:
    """Construct an attn_inputs mock for the draft-prefill path.

    Production C++ runner ships ``is_target_verify=False`` and
    ``prefill_cuda_graph_copy_params`` set for draft prefill.  The op
    only consults ``is_cuda_graph`` and the host-pinned per-batch tensors,
    so duck typing is fine.
    """
    prefix = [cs - il for cs, il in zip(cache_seqlens, input_lengths)]
    inputs = SimpleNamespace()
    inputs.is_prefill = True
    inputs.is_target_verify = False
    inputs.is_cuda_graph = is_cuda_graph

    inputs.prefix_lengths = torch.tensor(prefix, dtype=torch.int32, device="cpu")
    inputs.sequence_lengths = torch.tensor(
        cache_seqlens, dtype=torch.int32, device="cpu"
    )
    inputs.input_lengths = torch.tensor(input_lengths, dtype=torch.int32, device="cpu")

    if zero_device_lengths:
        inputs.prefix_lengths_d = torch.zeros(
            len(prefix), dtype=torch.int32, device=device
        )
        inputs.input_lengths_d = torch.zeros(
            len(input_lengths), dtype=torch.int32, device=device
        )
    else:
        inputs.prefix_lengths_d = inputs.prefix_lengths.to(device)
        inputs.input_lengths_d = inputs.input_lengths.to(device)

    inputs.kv_cache_kernel_block_id_host = block_id_host
    inputs.kv_cache_kernel_block_id_device = block_id_device
    inputs.kv_cache_block_id_host = block_id_host
    inputs.kv_cache_block_id_device = block_id_device

    cu = [0]
    for il in input_lengths:
        cu.append(cu[-1] + il)
    inputs.cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    inputs.cu_kv_seqlens = inputs.cu_seqlens.clone()

    inputs.dtype = get_typemeta(torch.zeros([1], dtype=dtype))
    return inputs


def _reference_per_request(
    q: torch.Tensor,
    cache_seqlens: List[int],
    input_lengths: List[int],
    page_table_h: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    page_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Per-request FA3 attention computed independently — ground truth."""
    k_cache = paged_kv_cache[:, 0].permute(0, 2, 1, 3).contiguous()
    v_cache = paged_kv_cache[:, 1].permute(0, 2, 1, 3).contiguous()
    outputs = []
    q_off = 0
    for i, (cs, il) in enumerate(zip(cache_seqlens, input_lengths)):
        if il == 0:
            continue
        q_seq = q[q_off : q_off + il].unsqueeze(0)
        n_pages = math.ceil(cs / page_size)
        page_table = page_table_h[i : i + 1, :n_pages].to(q.device)
        cache_seqlen_t = torch.tensor([cs], dtype=torch.int32, device=q.device)
        out_seq = flash_attn_with_kvcache(
            q=q_seq,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlen_t,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=True,
            pack_gqa=True,
        )
        if isinstance(out_seq, tuple):
            out_seq = out_seq[0]
        outputs.append(out_seq.squeeze(0))
        q_off += il
    return torch.cat(outputs, dim=0)


class TestFA3DraftPrefillHeterogeneous(BaseAttentionTest):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        cap_major, _ = torch.cuda.get_device_capability(0)
        if cap_major < 9:
            self.skipTest("FA3 only supported on Hopper (sm9x) or newer")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_attn_configs(self):
        from rtp_llm.ops import AttentionConfigs, RopeConfig, RopeStyle

        attn_configs = AttentionConfigs()
        attn_configs.head_num = HEAD_NUM
        attn_configs.kv_head_num = KV_HEAD_NUM
        attn_configs.size_per_head = HEAD_DIM
        attn_configs.tokens_per_block = PAGE_SIZE
        attn_configs.kernel_tokens_per_block = PAGE_SIZE
        attn_configs.use_mla = False
        attn_configs.dtype = torch.bfloat16
        attn_configs.kv_cache_dtype = KvCacheDataType.BASE
        attn_configs.max_seq_len = MAX_SEQ_LEN
        rope = RopeConfig()
        rope.style = RopeStyle.No
        attn_configs.rope_config = rope
        return attn_configs

    def _make_op_eager(self, attn_inputs: Any):
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_fa3_paged import (
            PyFA3PagedAttnOp as PyFA3PagedDraftPrefillAttnOp,
        )

        attn_inputs.is_cuda_graph = False
        op = PyFA3PagedDraftPrefillAttnOp(self._make_attn_configs(), attn_inputs)
        op.prepare(attn_inputs)
        return op

    def _make_op_cg(self, capture_attn_inputs: Any):
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_fa3_paged import (
            PyFA3PagedAttnOp as PyFA3PagedDraftPrefillAttnOp,
        )

        capture_attn_inputs.is_cuda_graph = True
        op = PyFA3PagedDraftPrefillAttnOp(
            self._make_attn_configs(), capture_attn_inputs
        )
        op.prepare(capture_attn_inputs)
        return op

    def _build_total_kv_cache(self, cache_seqlens: List[int]) -> LayerKVCache:
        total_pages = sum(math.ceil(cs / PAGE_SIZE) for cs in cache_seqlens) + 4
        return _build_paged_kv_cache(
            num_pages=total_pages,
            page_size=PAGE_SIZE,
            num_kv_heads=KV_HEAD_NUM,
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def _max_q_buf_size(self, captured_bs: int) -> int:
        # Max allocation matches what the wrapper would size at capture
        # time: max_bs * num_tokens_per_bs.  Each replay copies a compact
        # `sum(input_lengths)` prefix into the head of this buffer.
        return captured_bs * NUM_TOKENS_PER_BS

    # ------------------------------------------------------------------
    # Cases
    # ------------------------------------------------------------------
    def test_homogeneous_q_eager_passes(self):
        """Baseline: all 4 batches with Q=4 (full draft window) eager."""
        input_lengths = [NUM_TOKENS_PER_BS] * 4
        cache_seqlens = [5000, 5000, 5000, 5000]
        kv_cache = self._build_total_kv_cache(cache_seqlens)
        max_blocks = max(math.ceil(cs / PAGE_SIZE) for cs in cache_seqlens) + 4
        block_h, block_d = _build_block_table(
            cache_seqlens, PAGE_SIZE, max_blocks, self.device
        )

        attn_inputs = _build_attn_inputs(
            cache_seqlens=cache_seqlens,
            input_lengths=input_lengths,
            block_id_host=block_h,
            block_id_device=block_d,
            is_cuda_graph=False,
            dtype=torch.bfloat16,
            device=self.device,
        )
        op = self._make_op_eager(attn_inputs)
        torch.manual_seed(123)
        q = torch.randn(
            sum(input_lengths),
            HEAD_NUM,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        out = op.forward(q, kv_cache)
        ref = _reference_per_request(
            q=q,
            cache_seqlens=cache_seqlens,
            input_lengths=input_lengths,
            page_table_h=block_h,
            paged_kv_cache=kv_cache.kv_cache_base,
            page_size=PAGE_SIZE,
            softmax_scale=HEAD_DIM**-0.5,
        )
        compare_tensors(
            out, ref, rtol=5e-3, atol=5e-3, name="[homogeneous-Q4-eager] op vs ref"
        )

    def test_heterogeneous_q_eager_passes(self):
        """Eager, mixed Q lengths per request — cu_seqlens_q built from
        actual input_lengths cumsum."""
        input_lengths = [1, 4, 2, 3]  # mixed draft token counts
        cache_seqlens = [4000, 14000, 24000, 8000]
        kv_cache = self._build_total_kv_cache(cache_seqlens)
        max_blocks = max(math.ceil(cs / PAGE_SIZE) for cs in cache_seqlens) + 4
        block_h, block_d = _build_block_table(
            cache_seqlens, PAGE_SIZE, max_blocks, self.device
        )

        attn_inputs = _build_attn_inputs(
            cache_seqlens=cache_seqlens,
            input_lengths=input_lengths,
            block_id_host=block_h,
            block_id_device=block_d,
            is_cuda_graph=False,
            dtype=torch.bfloat16,
            device=self.device,
        )
        op = self._make_op_eager(attn_inputs)
        torch.manual_seed(7)
        q = torch.randn(
            sum(input_lengths),
            HEAD_NUM,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        out = op.forward(q, kv_cache)
        ref = _reference_per_request(
            q=q,
            cache_seqlens=cache_seqlens,
            input_lengths=input_lengths,
            page_table_h=block_h,
            paged_kv_cache=kv_cache.kv_cache_base,
            page_size=PAGE_SIZE,
            softmax_scale=HEAD_DIM**-0.5,
        )
        compare_tensors(
            out, ref, rtol=5e-3, atol=5e-3, name="[heterogeneous-Q-eager] op vs ref"
        )

    def test_heterogeneous_q_cg_replay_matches_eager(self):
        """Capture at full Q=4-each, replay with mixed Q.  CG output must
        match eager output for the same inputs.

        Mirrors production behavior: capture buffers sized for max draft
        tokens per request, but actual replay can have fewer tokens per
        request when the speculator is partially cancelled.
        """
        captured_bs = 4
        capture_q_per_bs = NUM_TOKENS_PER_BS
        # Replay-time mixed Q lengths (sum < captured_bs * num_tokens_per_bs).
        replay_input_lengths = [1, 4, 2, 3]
        replay_cache_seqlens = [4000, 14000, 24000, 8000]
        kv_cache = self._build_total_kv_cache(replay_cache_seqlens)

        max_blocks = math.ceil(MAX_SEQ_LEN / PAGE_SIZE) + 4
        block_h = torch.zeros(
            (captured_bs, max_blocks), dtype=torch.int32, device="cpu"
        )
        block_d = torch.zeros(
            (captured_bs, max_blocks), dtype=torch.int32, device=self.device
        )
        replay_block_h, _ = _build_block_table(
            replay_cache_seqlens, PAGE_SIZE, max_blocks, self.device
        )

        # Q buffer sized for max captured tokens (compact layout in production —
        # only the leading sum(input_lengths) rows are meaningful).
        q_buf_size = self._max_q_buf_size(captured_bs)
        q_buf = torch.zeros(
            q_buf_size, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )

        # Capture-time inputs: all batches at full Q=4, zero device lengths
        # (production CudaGraphRunner::initCaptureAttentionInputs does this).
        capture_input_lengths = [capture_q_per_bs] * captured_bs
        capture_cache_seqlens = [MAX_SEQ_LEN] * captured_bs
        attn_inputs_capture = _build_attn_inputs(
            cache_seqlens=capture_cache_seqlens,
            input_lengths=capture_input_lengths,
            block_id_host=block_h,
            block_id_device=block_d,
            is_cuda_graph=True,
            dtype=torch.bfloat16,
            device=self.device,
            zero_device_lengths=True,
        )
        cg_op = self._make_op_cg(attn_inputs_capture)

        q_buf.zero_()
        for _ in range(2):
            _ = cg_op.forward(q_buf, kv_cache)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        out_cg_buf = torch.empty_like(q_buf)
        with torch.cuda.graph(graph):
            captured = cg_op.forward(q_buf, kv_cache)
            out_cg_buf.copy_(captured)

        # Refresh stable buffers in place (mirror C++ runner).
        block_d.zero_()
        block_d.copy_(replay_block_h.to(self.device))
        block_h.zero_()
        block_h.copy_(replay_block_h)
        replay_inputs = _build_attn_inputs(
            cache_seqlens=replay_cache_seqlens,
            input_lengths=replay_input_lengths,
            block_id_host=block_h,
            block_id_device=block_d,
            is_cuda_graph=True,
            dtype=torch.bfloat16,
            device=self.device,
        )
        # The host-pinned input/prefix tensors are the C++ stable buffers in
        # production; here we just swap them on the namespace to mimic the
        # refresh.  Since CG only stashes _refresh_cg_buffers reads them
        # again on prepare(), this is enough to drive the next replay.
        attn_inputs_capture.prefix_lengths = replay_inputs.prefix_lengths
        attn_inputs_capture.input_lengths = replay_inputs.input_lengths
        attn_inputs_capture.cu_seqlens = replay_inputs.cu_seqlens
        cg_op.prepare(attn_inputs_capture)

        torch.manual_seed(42)
        q_active = torch.randn(
            sum(replay_input_lengths),
            HEAD_NUM,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_buf.zero_()
        q_buf[: sum(replay_input_lengths)].copy_(q_active)

        graph.replay()
        torch.cuda.synchronize()
        out_cg = out_cg_buf[: sum(replay_input_lengths)].clone()

        # Eager reference for the same inputs.
        eager_inputs = _build_attn_inputs(
            cache_seqlens=replay_cache_seqlens,
            input_lengths=replay_input_lengths,
            block_id_host=replay_block_h,
            block_id_device=replay_block_h.to(self.device),
            is_cuda_graph=False,
            dtype=torch.bfloat16,
            device=self.device,
        )
        eager_op = self._make_op_eager(eager_inputs)
        out_eager = eager_op.forward(q_active, kv_cache).clone()

        compare_tensors(
            out_cg,
            out_eager,
            rtol=5e-3,
            atol=5e-3,
            name="[heterogeneous-Q-CG vs eager]",
        )

    def test_multi_replay_varying_q_patterns(self):
        """Capture once, replay 5 times with cycling mixed-Q patterns.
        Each replay's output must match a fresh eager re-run."""
        captured_bs = 4
        capture_input_lengths = [NUM_TOKENS_PER_BS] * captured_bs
        capture_cache_seqlens = [MAX_SEQ_LEN] * captured_bs

        replay_patterns = [
            ([1, 4, 2, 3], [4000, 14000, 24000, 8000]),
            ([4, 4, 4, 4], [22000, 22000, 22000, 22000]),
            ([2, 3, 1, 4], [1000, 27000, 1000, 27000]),
            ([3, 2, 4, 1], [8000, 8000, 8000, 8000]),
            ([1, 1, 1, 1], [4000, 14000, 24000, 8000]),
        ]
        kv_cache = self._build_total_kv_cache([max(p[1]) for p in replay_patterns])

        max_blocks = math.ceil(MAX_SEQ_LEN / PAGE_SIZE) + 4
        block_h = torch.zeros(
            (captured_bs, max_blocks), dtype=torch.int32, device="cpu"
        )
        block_d = torch.zeros(
            (captured_bs, max_blocks), dtype=torch.int32, device=self.device
        )

        q_buf_size = self._max_q_buf_size(captured_bs)
        q_buf = torch.zeros(
            q_buf_size, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )

        attn_inputs_capture = _build_attn_inputs(
            cache_seqlens=capture_cache_seqlens,
            input_lengths=capture_input_lengths,
            block_id_host=block_h,
            block_id_device=block_d,
            is_cuda_graph=True,
            dtype=torch.bfloat16,
            device=self.device,
            zero_device_lengths=True,
        )
        cg_op = self._make_op_cg(attn_inputs_capture)

        q_buf.zero_()
        for _ in range(2):
            _ = cg_op.forward(q_buf, kv_cache)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        out_cg_buf = torch.empty_like(q_buf)
        with torch.cuda.graph(graph):
            captured = cg_op.forward(q_buf, kv_cache)
            out_cg_buf.copy_(captured)

        torch.manual_seed(7)
        for it, (input_lengths, cache_seqlens) in enumerate(replay_patterns):
            replay_block_h, _ = _build_block_table(
                cache_seqlens, PAGE_SIZE, max_blocks, self.device
            )
            block_h.copy_(replay_block_h)
            block_d.copy_(replay_block_h.to(self.device))

            replay_inputs = _build_attn_inputs(
                cache_seqlens=cache_seqlens,
                input_lengths=input_lengths,
                block_id_host=block_h,
                block_id_device=block_d,
                is_cuda_graph=True,
                dtype=torch.bfloat16,
                device=self.device,
            )
            attn_inputs_capture.prefix_lengths = replay_inputs.prefix_lengths
            attn_inputs_capture.input_lengths = replay_inputs.input_lengths
            attn_inputs_capture.cu_seqlens = replay_inputs.cu_seqlens
            cg_op.prepare(attn_inputs_capture)

            total_q = sum(input_lengths)
            q_active = torch.randn(
                total_q, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=self.device
            )
            q_buf.zero_()
            q_buf[:total_q].copy_(q_active)
            graph.replay()
            torch.cuda.synchronize()
            out_cg = out_cg_buf[:total_q].clone()

            eager_inputs = _build_attn_inputs(
                cache_seqlens=cache_seqlens,
                input_lengths=input_lengths,
                block_id_host=block_h,
                block_id_device=block_d,
                is_cuda_graph=False,
                dtype=torch.bfloat16,
                device=self.device,
            )
            eager_op = self._make_op_eager(eager_inputs)
            out_eager = eager_op.forward(q_active, kv_cache).clone()
            compare_tensors(
                out_cg,
                out_eager,
                rtol=5e-3,
                atol=5e-3,
                name=f"[multi-replay iter {it} Q={input_lengths} cs={cache_seqlens}]",
            )

    def test_cg_replay_reads_input_lengths_at_runtime(self):
        """Sanity probe: prove FA3 captured kernel honors live cu_seqlens_q
        / cache_seqlens (NOT baked at capture).  Two replays with distinct
        Q-length patterns must produce different outputs.
        """
        captured_bs = 4
        capture_input_lengths = [NUM_TOKENS_PER_BS] * captured_bs
        capture_cache_seqlens = [MAX_SEQ_LEN] * captured_bs

        probe_a = ([1, 4, 2, 3], [4000, 14000, 24000, 8000])
        probe_b = ([4, 1, 3, 2], [24000, 4000, 8000, 14000])

        kv_cache = self._build_total_kv_cache(probe_a[1] + probe_b[1])
        max_blocks = math.ceil(MAX_SEQ_LEN / PAGE_SIZE) + 4
        block_h = torch.zeros(
            (captured_bs, max_blocks), dtype=torch.int32, device="cpu"
        )
        block_d = torch.zeros(
            (captured_bs, max_blocks), dtype=torch.int32, device=self.device
        )

        q_buf_size = self._max_q_buf_size(captured_bs)
        q_buf = torch.zeros(
            q_buf_size, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )

        attn_inputs_capture = _build_attn_inputs(
            cache_seqlens=capture_cache_seqlens,
            input_lengths=capture_input_lengths,
            block_id_host=block_h,
            block_id_device=block_d,
            is_cuda_graph=True,
            dtype=torch.bfloat16,
            device=self.device,
            zero_device_lengths=True,
        )
        cg_op = self._make_op_cg(attn_inputs_capture)

        q_buf.zero_()
        for _ in range(2):
            _ = cg_op.forward(q_buf, kv_cache)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        out_cg_buf = torch.empty_like(q_buf)
        with torch.cuda.graph(graph):
            captured = cg_op.forward(q_buf, kv_cache)
            out_cg_buf.copy_(captured)

        torch.manual_seed(123)
        # Use IDENTICAL random Q tokens for both probes; only the per-batch
        # split of those tokens (cu_seqlens_q) and cache_seqlens differ.
        # Same total Q size for both probes (10 tokens each).
        assert sum(probe_a[0]) == sum(probe_b[0]) == 10
        q_active = torch.randn(
            10, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )

        def replay_with(input_lengths, cache_seqlens):
            probe_block_h, _ = _build_block_table(
                cache_seqlens, PAGE_SIZE, max_blocks, self.device
            )
            block_h.copy_(probe_block_h)
            block_d.copy_(probe_block_h.to(self.device))
            probe_inputs = _build_attn_inputs(
                cache_seqlens=cache_seqlens,
                input_lengths=input_lengths,
                block_id_host=block_h,
                block_id_device=block_d,
                is_cuda_graph=True,
                dtype=torch.bfloat16,
                device=self.device,
            )
            attn_inputs_capture.prefix_lengths = probe_inputs.prefix_lengths
            attn_inputs_capture.input_lengths = probe_inputs.input_lengths
            attn_inputs_capture.cu_seqlens = probe_inputs.cu_seqlens
            cg_op.prepare(attn_inputs_capture)

            q_buf.zero_()
            q_buf[:10].copy_(q_active)
            graph.replay()
            torch.cuda.synchronize()
            return out_cg_buf[:10].clone()

        out_a = replay_with(*probe_a)
        out_b = replay_with(*probe_b)

        diff = (out_a - out_b).abs().max().item()
        self.assertGreater(
            diff,
            1e-3,
            f"FA3 captured kernel returned IDENTICAL output for two distinct "
            f"Q-length / cache_seqlens patterns (max diff = {diff}); proves "
            f"the kernel baked per-batch scheduling at capture time.",
        )

    def test_padded_batch_capture_smaller_replay(self):
        """Capture at bs=8, replay at actual_bs=4 with padding rows zeroed.

        Production CudaGraphRunner::prepareInputs zero-fills padding rows of
        prefix/input lengths and zeros padding rows of the page_table.  The
        captured FA3 kernel processes captured_bs=8 batches; for padding
        rows cache_seqlens=0 and FA3 emits no work.  Active rows must match
        per-request reference.
        """
        captured_bs = 8
        actual_bs = 4
        active_input_lengths = [1, 4, 2, 3]
        active_cache_seqlens = [4000, 14000, 24000, 8000]

        capture_input_lengths = [NUM_TOKENS_PER_BS] * captured_bs
        capture_cache_seqlens = [MAX_SEQ_LEN] * captured_bs
        replay_input_lengths = active_input_lengths + [0] * (captured_bs - actual_bs)
        replay_cache_seqlens = active_cache_seqlens + [0] * (captured_bs - actual_bs)

        kv_cache = self._build_total_kv_cache(active_cache_seqlens)
        max_blocks = math.ceil(MAX_SEQ_LEN / PAGE_SIZE) + 4
        block_h = torch.zeros(
            (captured_bs, max_blocks), dtype=torch.int32, device="cpu"
        )
        block_d = torch.zeros(
            (captured_bs, max_blocks), dtype=torch.int32, device=self.device
        )

        q_buf_size = self._max_q_buf_size(captured_bs)
        q_buf = torch.zeros(
            q_buf_size, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )

        attn_inputs_capture = _build_attn_inputs(
            cache_seqlens=capture_cache_seqlens,
            input_lengths=capture_input_lengths,
            block_id_host=block_h,
            block_id_device=block_d,
            is_cuda_graph=True,
            dtype=torch.bfloat16,
            device=self.device,
            zero_device_lengths=True,
        )
        cg_op = self._make_op_cg(attn_inputs_capture)

        q_buf.zero_()
        for _ in range(2):
            _ = cg_op.forward(q_buf, kv_cache)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        out_cg_buf = torch.empty_like(q_buf)
        with torch.cuda.graph(graph):
            captured = cg_op.forward(q_buf, kv_cache)
            out_cg_buf.copy_(captured)

        # Replay refresh: page_table for active rows, lengths for active +
        # zeroed padding.
        active_block_h, _ = _build_block_table(
            active_cache_seqlens, PAGE_SIZE, max_blocks, self.device
        )
        block_h.zero_()
        block_h[:actual_bs, :].copy_(active_block_h)
        block_d.zero_()
        block_d[:actual_bs, :].copy_(active_block_h.to(self.device))

        replay_inputs = _build_attn_inputs(
            cache_seqlens=replay_cache_seqlens,
            input_lengths=replay_input_lengths,
            block_id_host=block_h,
            block_id_device=block_d,
            is_cuda_graph=True,
            dtype=torch.bfloat16,
            device=self.device,
        )
        attn_inputs_capture.prefix_lengths = replay_inputs.prefix_lengths
        attn_inputs_capture.input_lengths = replay_inputs.input_lengths
        attn_inputs_capture.cu_seqlens = replay_inputs.cu_seqlens
        cg_op.prepare(attn_inputs_capture)

        torch.manual_seed(7)
        total_active = sum(active_input_lengths)
        q_active = torch.randn(
            total_active, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        # Padding rows of q_buf intentionally garbage (large magnitude) — if
        # FA3 leaks q from padding rows into active output under PackGQA it
        # will fail loud here.
        garbage = (
            torch.randn(
                q_buf_size - total_active,
                HEAD_NUM,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 100.0
        )
        q_buf[:total_active].copy_(q_active)
        q_buf[total_active:].copy_(garbage)
        graph.replay()
        torch.cuda.synchronize()
        out_cg = out_cg_buf[:total_active].clone()

        ref = _reference_per_request(
            q=q_active,
            cache_seqlens=active_cache_seqlens,
            input_lengths=active_input_lengths,
            page_table_h=active_block_h,
            paged_kv_cache=kv_cache.kv_cache_base,
            page_size=PAGE_SIZE,
            softmax_scale=HEAD_DIM**-0.5,
        )
        compare_tensors(
            out_cg,
            ref,
            rtol=5e-3,
            atol=5e-3,
            name=f"[padded captured_bs={captured_bs} actual_bs={actual_bs}] "
            f"CG active rows vs per-request reference",
        )


if __name__ == "__main__":
    unittest.main()
