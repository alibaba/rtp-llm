"""Repro harness for ``PyFA3PagedTargetVerifyAttnOp`` heterogeneous-batch bug.

Background
----------
``PyFA3PagedTargetVerifyAttnOp`` (cuda_impl/py_fa3_target_verify.py) was
designed to replace the FlashInfer plan/run target-verify path that suffered
from a CG buffer-aliasing bug.  In **homogeneous** stress runs (160/160 req,
identical prompts) the FA3 op is clean: 0 repetition, 0 garbled.

In **heterogeneous mixed batches** (different real prompts in the same
batch — long + short prompt mixed, e.g. 4k / 14k / 24k token), the FA3 op
under CUDA graph still produces garbled / cross-request KV-polluted output.
Three-way ablation isolated the bug to the target-verify CG path:
disabling it (``DISABLE_SP_TARGET_VERIFY_CUDA_GRAPH=1``) takes the failure
rate from 6/40 down to 0/40 on the same mixed-prompt smoke stress.

What this harness does
----------------------
1. Build a paged KV cache populated with random data.
2. Construct attn_inputs mocks for capture-time (max-padded) and replay-time
   (actual heterogeneous lengths).
3. Run the op via:
     - eager (no CG)               — single source of truth.
     - CUDA-graph capture+replay   — the suspect path.
4. Cross-check eager against a per-request FA3 reference (bs=1 calls).
5. ``test_heterogeneous_multi_replay_reproduces_bug`` cycles through 5
   different replay patterns to surface state-leakage between replays.

A ``SimpleNamespace`` mock is used in place of ``PyAttentionInputs`` because
``prefix_lengths_d`` and ``input_lengths_d`` are ``def_readonly`` in the
pybind11 binding (populated from C++ in production, not assignable from
Python).  The op only reads attributes so duck typing is sufficient.

Status (2026-04-27): all 6 tests PASS — the isolated harness does not
trigger the production bug.  The probe test
``test_replay_reads_cache_seqlens_at_runtime`` PROVES that FA3's captured
kernel honors live ``cache_seqlens`` at replay (two distinct probes give
distinct outputs).  Combined with the other CG-vs-eager passes under:
  * heterogeneous batches,
  * multi-replay with cycling seqlens patterns,
  * captured_bs > current_batch_size with stale q garbage in padding rows
    + the production padding cu_seqlens collapse,
  * production-fidelity capture state (zero device lengths, zero page table),

these tests collectively rule out FA3's per-launch attention params as the
bug source.  The production bug must therefore live OUTSIDE
``flash_attn_with_kvcache`` itself — most likely candidates:
  1. ``KVCacheWriteOp`` captured-graph interaction (writes K/V to wrong page
     under some replay-time state),
  2. ``fmha_params.fill_params`` returning fresh tensors at replay despite
     ``forbid_realloc=True`` (captured op stashes stale data_ptrs),
  3. Wrapper-level ordering between RoPE / KV write / attention under CG.
  4. Or something at the scheduler / frontend level (wrong batch routing).

Test cases (all pass today, kept as a regression harness):
* ``test_homogeneous_batch_passes``                          — baseline.
* ``test_heterogeneous_batch_reproduces_bug``                — single capture+replay.
* ``test_heterogeneous_multi_replay_reproduces_bug``         — cycling seqlens.
* ``test_padded_batch_capture_smaller_replay_reproduces_bug``— captured_bs > actual_bs.
* ``test_replay_reads_cache_seqlens_at_runtime``             — cache_seqlens probe.
* ``test_heterogeneous_eager_only_passes``                   — eager negative control.

Run:
    bazelisk test //rtp_llm/models_py/modules/factory/attention/cuda_impl/test:test_fa3_target_verify_heterogeneous
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


# Match the production smoke setup: Qwen3.5-27B per-TP-rank attention shape
HEAD_NUM = 8  # tp_q_head_num after TP=2
KV_HEAD_NUM = 1  # tp_k_head_num after TP=2
HEAD_DIM = 256
PAGE_SIZE = 64  # = --kernel_seq_size_per_block 64
NUM_DRAFT_TOKENS = 4  # = --gen_num_per_cycle 3 + 1 base
MAX_SEQ_LEN = 28000  # mirrors production --max_seq_len bucket


def _build_paged_kv_cache(
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> LayerKVCache:
    """5-D HND paged KV cache: [num_pages, 2, kv_heads, page_size, head_dim]."""
    storage = torch.randn(
        num_pages,
        2,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=dtype,
        device=device,
    )
    cache = LayerKVCache()
    cache.kv_cache_base = storage
    return cache


def _build_attn_inputs(
    *,
    cache_seqlens: List[int],
    input_lengths: List[int],
    block_id_host: torch.Tensor,
    block_id_device: torch.Tensor,
    is_cuda_graph: bool,
    dtype: torch.dtype,
    decode_cu_seqlens_d: torch.Tensor,
    device: torch.device,
    zero_device_lengths: bool = False,
) -> Any:
    """Construct an attn_inputs mock mirroring what
    ``CudaGraphRunner::prepareInputs`` ships to the Python attn impl for the
    target-verify path.

    * ``prefix_lengths`` = cache_seqlens - input_lengths (the historical KV
      length prior to the new draft tokens).
    * ``sequence_lengths`` = cache_seqlens (full visible KV).
    * ``input_lengths`` = per-request draft token count.
    * ``decode_cu_seqlens_d`` is provided externally so capture and replay
      share the same stable buffer (the C++ runner refreshes it in place).
    * ``zero_device_lengths``: when True, set ``prefix_lengths_d`` and
      ``input_lengths_d`` to ZEROS regardless of cache_seqlens.  This is
      what production ``CudaGraphRunner::initCaptureAttentionInputs`` does
      at capture time (cuda_graph_runner.cc:578-579 explicitly overwrite
      both device tensors to ``torch::zeros(...)`` after they were filled
      with sane values).  At capture, ``_cache_seqlens_buf = prefix_d +
      input_d`` therefore evaluates to all zeros — meaning the FA3 kernel
      is captured with zero cache_seqlens, doing no actual attention work,
      and any host-side scheduler / launch-param decisions FA3 makes are
      taken under the assumption "no KV to read".  At replay these get
      reused with real seqlens.

    Uses SimpleNamespace because PyAttentionInputs has prefix_lengths_d /
    input_lengths_d declared as ``def_readonly`` in the pybind11 binding.
    """
    prefix = [cs - il for cs, il in zip(cache_seqlens, input_lengths)]
    inputs = SimpleNamespace()
    inputs.is_prefill = True
    inputs.is_target_verify = True
    inputs.is_cuda_graph = is_cuda_graph

    inputs.prefix_lengths = torch.tensor(prefix, dtype=torch.int32, device="cpu")
    inputs.sequence_lengths = torch.tensor(
        cache_seqlens, dtype=torch.int32, device="cpu"
    )
    inputs.input_lengths = torch.tensor(input_lengths, dtype=torch.int32, device="cpu")

    if zero_device_lengths:
        # Match production initCaptureAttentionInputs: device lengths zeroed.
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

    # Stable per-graph_instance cu_seqlens_q buffer.  In production this is
    # initialised by C++ to ``arange(0, (max_bs+1)*num_tokens_per_bs, step=
    # num_tokens_per_bs)`` and refreshed in place each replay.
    inputs.decode_cu_seqlens_d = decode_cu_seqlens_d

    inputs.dtype = get_typemeta(torch.zeros([1], dtype=dtype))
    return inputs


def _build_block_table(
    cache_seqlens: List[int],
    page_size: int,
    max_blocks: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequential block IDs per request — disjoint, never overlapping, so any
    cross-request leakage shows up as wrong KV values.

    Every column up to ``max_blocks`` gets a valid (in-range) page id so the
    kernel can't OOB even when capture-time ``cache_seqlens`` reads the full
    width.  Pages past the request's actual cache_seqlen point at the
    request's last real page (FA3 still won't read them at replay time
    because ``cache_seqlens_buf`` cuts off the iteration).
    """
    bs = len(cache_seqlens)
    table = torch.zeros((bs, max_blocks), dtype=torch.int32, device="cpu")
    block_offset = 1  # leave page 0 as "scratch / safe" — production runner
    # treats stale zeros there as a no-op for padding rows.
    for i, cs in enumerate(cache_seqlens):
        n = math.ceil(cs / page_size)
        if n == 0:
            table[i, :].fill_(0)
            continue
        table[i, :n] = torch.arange(block_offset, block_offset + n, dtype=torch.int32)
        # Pad with the last valid page id — keeps reads in-bounds even if
        # the kernel walks the full width during capture-time warmup.
        table[i, n:].fill_(block_offset + n - 1)
        block_offset += n
    return table, table.to(device)


def _reference_per_request(
    q: torch.Tensor,
    cache_seqlens: List[int],
    input_lengths: List[int],
    page_table_h: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    page_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Per-request FA3 attention computed independently — ground truth.

    Each request gets its own ``flash_attn_with_kvcache`` call with bs=1
    (no batching at all), so cross-request schedule choices in batched FA3
    cannot affect the result.  Concatenated back into the same packed-varlen
    layout as the op's output.
    """
    # paged: [num_pages, 2, kv_heads, page_size, head_dim]
    k_cache = paged_kv_cache[:, 0].permute(0, 2, 1, 3).contiguous()
    v_cache = paged_kv_cache[:, 1].permute(0, 2, 1, 3).contiguous()
    outputs = []
    q_off = 0
    for i, (cs, il) in enumerate(zip(cache_seqlens, input_lengths)):
        q_seq = q[q_off : q_off + il].unsqueeze(0)  # [1, il, H, D]
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
        outputs.append(out_seq.squeeze(0))  # [il, H, D]
        q_off += il
    return torch.cat(outputs, dim=0)


class TestFA3TargetVerifyHeterogeneous(BaseAttentionTest):
    """Reproduce PyFA3PagedTargetVerifyAttnOp's heterogeneous-batch bug."""

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

    def _build_total_kv_cache(self, cache_seqlens: List[int]) -> LayerKVCache:
        # Size for the actual replay seqlens — block_table only assigns real
        # disjoint page ids for these.  Capture-time over-reads land on the
        # last valid page (see _build_block_table), which still resides in
        # this cache.
        total_pages = (
            sum(math.ceil(cs / PAGE_SIZE) for cs in cache_seqlens) + 4
        )  # +4 scratch / safety
        return _build_paged_kv_cache(
            num_pages=total_pages,
            page_size=PAGE_SIZE,
            num_kv_heads=KV_HEAD_NUM,
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def _make_op_eager(self, attn_inputs: Any):
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_fa3_target_verify import (
            PyFA3PagedTargetVerifyAttnOp,
        )

        attn_inputs.is_cuda_graph = False
        op = PyFA3PagedTargetVerifyAttnOp(self._make_attn_configs(), attn_inputs)
        op.prepare(attn_inputs)
        return op

    def _make_op_cg(self, capture_attn_inputs: Any):
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_fa3_target_verify import (
            PyFA3PagedTargetVerifyAttnOp,
        )

        capture_attn_inputs.is_cuda_graph = True
        op = PyFA3PagedTargetVerifyAttnOp(
            self._make_attn_configs(), capture_attn_inputs
        )
        op.prepare(capture_attn_inputs)
        return op

    def _build_q_buffer(self, total_q: int) -> torch.Tensor:
        """Caller-stable Q tensor — same data_ptr across captures + replays."""
        return torch.zeros(
            total_q,
            HEAD_NUM,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # The shared scenario runner
    # ------------------------------------------------------------------
    def _run_scenario(
        self,
        cache_seqlens: List[int],
        capture_cache_seqlens: List[int],
        scenario: str,
    ):
        """Run op in eager and CG modes; compare outputs to a per-request
        reference and to each other.

        ``capture_cache_seqlens`` is what would land on the op's first
        ``prepare()`` (mimicking the C++ ``initCaptureAttentionInputs`` which
        fills with ``max_seq_len - num_tokens_per_bs`` for all batches).
        ``cache_seqlens`` is the live replay-time data.
        """
        bs = len(cache_seqlens)
        assert len(capture_cache_seqlens) == bs

        input_lengths = [NUM_DRAFT_TOKENS] * bs
        total_q = sum(input_lengths)

        # KV cache holds enough pages for the *replay* requests.
        kv_cache = self._build_total_kv_cache(cache_seqlens)

        # Stable per-graph_instance cu_seqlens_q (uniform stride =
        # num_tokens_per_bs, what C++ initialises for target verify).
        decode_cu_seqlens_d = torch.arange(
            0,
            (bs + 1) * NUM_DRAFT_TOKENS,
            NUM_DRAFT_TOKENS,
            dtype=torch.int32,
            device=self.device,
        )

        # Block tables — sized for the *capture* seqlen worst case
        # (production C++ allocates page_table for max_seq_len at capture).
        # For replay seqlens, only the leading n columns hold real disjoint
        # page ids; trailing columns are pinned to the last valid page so
        # capture-time over-reads stay in-bounds.
        max_blocks = (
            max(
                max(math.ceil(cs / PAGE_SIZE) for cs in cache_seqlens),
                max(math.ceil(cs / PAGE_SIZE) for cs in capture_cache_seqlens),
            )
            + 4
        )
        block_id_host, block_id_device = _build_block_table(
            cache_seqlens, PAGE_SIZE, max_blocks, self.device
        )

        # Stable Q buffer (capture-time data is just zeros; replay copies in
        # the real Q values).  Single buffer reused by both modes so addresses
        # are identical.
        q_buf = self._build_q_buffer(total_q)
        torch.manual_seed(123)
        q_replay = torch.randn_like(q_buf)

        # ---------- 1) Eager reference path ----------
        attn_inputs_eager = _build_attn_inputs(
            cache_seqlens=cache_seqlens,
            input_lengths=input_lengths,
            block_id_host=block_id_host,
            block_id_device=block_id_device,
            is_cuda_graph=False,
            dtype=torch.bfloat16,
            decode_cu_seqlens_d=decode_cu_seqlens_d,
            device=self.device,
        )
        eager_op = self._make_op_eager(attn_inputs_eager)
        out_eager = eager_op.forward(q_replay, kv_cache).clone()

        # ---------- 2) Per-request reference (single source of truth) ----------
        ref_out = _reference_per_request(
            q=q_replay,
            cache_seqlens=cache_seqlens,
            input_lengths=input_lengths,
            page_table_h=block_id_host,
            paged_kv_cache=kv_cache.kv_cache_base,
            page_size=PAGE_SIZE,
            softmax_scale=HEAD_DIM**-0.5,
        )

        # Sanity: eager output must match per-request reference.
        compare_tensors(
            out_eager,
            ref_out,
            rtol=5e-3,
            atol=5e-3,
            name=f"[{scenario}] eager vs per-request reference",
        )

        # ---------- 3) CG path ----------
        # Production initCaptureAttentionInputs (cuda_graph_runner.cc:493+)
        # initialises capture buffers as follows:
        #   - prefix_lengths_d / input_lengths_d : torch::zeros (line 578-579)
        #   - kv_cache_kernel_block_id_device    : torch::zeros (line 514-515)
        #     and prepareInputs clears it again (line 126) before each replay
        #   - q_buf at capture is implicitly zero (no real input flowed yet)
        # That means the captured FA3 launch does ZERO actual attention
        # work — every per-host-side scheduler decision FA3 makes (num_splits,
        # tile counts, scheduler_metadata) is taken under "no KV to read".
        # The repro fidelity that was missing in earlier versions of this
        # UT was capture-time non-zero values; mirror production now.
        capture_block_h = torch.zeros_like(block_id_host)
        capture_block_d = torch.zeros_like(block_id_device)
        attn_inputs_capture = _build_attn_inputs(
            cache_seqlens=capture_cache_seqlens,  # CPU-only padding values
            input_lengths=input_lengths,
            block_id_host=capture_block_h,
            block_id_device=capture_block_d,
            is_cuda_graph=True,
            dtype=torch.bfloat16,
            decode_cu_seqlens_d=decode_cu_seqlens_d,
            device=self.device,
            zero_device_lengths=True,  # match production
        )
        cg_op = self._make_op_cg(attn_inputs_capture)

        # Warmup with capture-time (all-zero) inputs.  q_buf still holds
        # whatever data was zero'd at allocation; do NOT copy q_replay yet.
        q_buf.zero_()
        for _ in range(2):
            _ = cg_op.forward(q_buf, kv_cache)
        torch.cuda.synchronize()

        # Capture: forward once inside graph context — still on zero inputs.
        graph = torch.cuda.CUDAGraph()
        out_cg_buf = torch.empty_like(q_buf)
        with torch.cuda.graph(graph):
            captured = cg_op.forward(q_buf, kv_cache)
            out_cg_buf.copy_(captured)

        # Replay refresh: production C++ does
        #   1) clearTensorAsync(kv_cache_kernel_block_id_device)
        #   2) copySmallerIntoLarger(live, captured)  <-- live block_ids in
        #   3) optimizedCopyAsync(live, captured)     <-- live prefix/input_d
        #   4) attn_pyobj.prepare_cuda_graph()        <-- refresh seqlens_buf
        # Mirror that here against the SAME capture buffers (data_ptrs
        # preserved across capture/replay).
        capture_block_d.zero_()
        capture_block_d.copy_(
            block_id_device
        )  # block_id_device built from real replay seqlens
        attn_inputs_replay = _build_attn_inputs(
            cache_seqlens=cache_seqlens,
            input_lengths=input_lengths,
            block_id_host=block_id_host,
            block_id_device=block_id_device,
            is_cuda_graph=True,
            dtype=torch.bfloat16,
            decode_cu_seqlens_d=decode_cu_seqlens_d,
            device=self.device,
        )
        attn_inputs_capture.prefix_lengths_d.copy_(attn_inputs_replay.prefix_lengths_d)
        attn_inputs_capture.input_lengths_d.copy_(attn_inputs_replay.input_lengths_d)
        cg_op.prepare(attn_inputs_capture)  # refresh _cache_seqlens_buf

        q_buf.copy_(q_replay)
        graph.replay()
        torch.cuda.synchronize()
        out_cg = out_cg_buf.clone()

        # ---------- 4) Compare CG vs eager ----------
        compare_tensors(
            out_cg,
            out_eager,
            rtol=5e-3,
            atol=5e-3,
            name=f"[{scenario}] CG-replay vs eager",
        )

    # ------------------------------------------------------------------
    # Cases
    # ------------------------------------------------------------------
    def test_homogeneous_batch_passes(self):
        """Baseline: all 4 requests share a single ~5k-token cache. Both
        eager and CG should match per-request reference."""
        bs = 4
        cache_seqlens = [5000] * bs
        # Capture-time uses the C++ default: prefix=max_seq_len-num_tokens_per_bs
        capture = [MAX_SEQ_LEN] * bs
        self._run_scenario(cache_seqlens, capture, "homogeneous-5000")

    def test_heterogeneous_batch_reproduces_bug(self):
        """Reproduction target: 4 requests with cache lengths spanning 6×.

        Production smoke (5 iter × 8 concurrent × mixed taobao+cooking pool)
        reliably triggers garbled output / cross-request KV-pollution in this
        regime.  Until the bug is fixed, this assertion will fail at the
        ``CG-replay vs eager`` compare; the eager and per-request reference
        paths still agree (sanity holds — only the CG path corrupts).
        """
        bs = 4
        cache_seqlens = [4000, 14000, 24000, 8000]  # 6× spread
        capture = [MAX_SEQ_LEN] * bs  # what C++ ships at capture
        self._run_scenario(cache_seqlens, capture, "heterogeneous-mixed")

    def test_heterogeneous_multi_replay_reproduces_bug(self):
        """Stress variant: capture once, then replay N times with each
        replay using a different per-request cache_seqlens pattern.

        Production decode runs target verify many times per generation step
        across many layers, with the request mix changing constantly.  If
        any replay-time state leaks (e.g., FA3 internal scheduler caches a
        per-launch decision that doesn't refresh, or split-K reduction reads
        from the wrong scratch slot between replays), the cumulative drift
        is more likely to surface here than in a single capture+replay.

        Each replay's output is independently cross-checked against an
        eager re-run with the same inputs, so cross-replay state leakage
        manifests as growing eager/CG divergence over the loop.
        """
        bs = 4
        capture = [MAX_SEQ_LEN] * bs
        # Cycle through 5 different heterogeneous patterns; the spread,
        # ordering and absolute values all change between replays.
        replay_patterns = [
            [4000, 14000, 24000, 8000],
            [24000, 4000, 8000, 14000],  # shuffled
            [22000, 22000, 22000, 22000],  # uniform short of capture max
            [1000, 27000, 1000, 27000],  # extreme mix
            [8000, 8000, 8000, 8000],  # uniform mid
        ]
        input_lengths = [NUM_DRAFT_TOKENS] * bs
        total_q = sum(input_lengths)

        # KV cache sized to accommodate the largest pattern.
        kv_cache = self._build_total_kv_cache([max(p) for p in replay_patterns])

        decode_cu_seqlens_d = torch.arange(
            0,
            (bs + 1) * NUM_DRAFT_TOKENS,
            NUM_DRAFT_TOKENS,
            dtype=torch.int32,
            device=self.device,
        )

        max_blocks = (
            max(
                max(math.ceil(cs / PAGE_SIZE) for p in replay_patterns for cs in p),
                math.ceil(MAX_SEQ_LEN / PAGE_SIZE),
            )
            + 4
        )
        # Stable buffers used across capture + every replay (data_ptrs
        # preserved).  At capture, these are all zero (matches production
        # initCaptureAttentionInputs).  Each replay refreshes them in place.
        block_id_host = torch.zeros((bs, max_blocks), dtype=torch.int32, device="cpu")
        block_id_device = torch.zeros(
            (bs, max_blocks), dtype=torch.int32, device=self.device
        )

        q_buf = self._build_q_buffer(total_q)

        # Build capture inputs the production way: zeroed device lengths +
        # zeroed page table (so FA3 captures zero-work scheduling).
        attn_inputs_capture = _build_attn_inputs(
            cache_seqlens=capture,
            input_lengths=input_lengths,
            block_id_host=block_id_host,
            block_id_device=block_id_device,
            is_cuda_graph=True,
            dtype=torch.bfloat16,
            decode_cu_seqlens_d=decode_cu_seqlens_d,
            device=self.device,
            zero_device_lengths=True,
        )
        cg_op = self._make_op_cg(attn_inputs_capture)

        torch.manual_seed(7)
        q_per_replay = [torch.randn_like(q_buf) for _ in replay_patterns]

        # Warmup with capture-time (zero) inputs.
        q_buf.zero_()
        for _ in range(2):
            _ = cg_op.forward(q_buf, kv_cache)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        out_cg_buf = torch.empty_like(q_buf)
        with torch.cuda.graph(graph):
            captured = cg_op.forward(q_buf, kv_cache)
            out_cg_buf.copy_(captured)

        for it, replay_seqlens in enumerate(replay_patterns):
            # Refresh block_id buffers in place to reflect this iter's pages.
            new_block_h, _ = _build_block_table(
                replay_seqlens, PAGE_SIZE, max_blocks, self.device
            )
            block_id_host.copy_(new_block_h)
            block_id_device.copy_(new_block_h.to(self.device))

            # Refresh per-request lengths in the SAME storage the op stashed.
            replay_inputs = _build_attn_inputs(
                cache_seqlens=replay_seqlens,
                input_lengths=input_lengths,
                block_id_host=block_id_host,
                block_id_device=block_id_device,
                is_cuda_graph=True,
                dtype=torch.bfloat16,
                decode_cu_seqlens_d=decode_cu_seqlens_d,
                device=self.device,
            )
            attn_inputs_capture.prefix_lengths_d.copy_(replay_inputs.prefix_lengths_d)
            attn_inputs_capture.input_lengths_d.copy_(replay_inputs.input_lengths_d)
            cg_op.prepare(attn_inputs_capture)

            # CG replay
            q_buf.copy_(q_per_replay[it])
            graph.replay()
            torch.cuda.synchronize()
            out_cg = out_cg_buf.clone()

            # Eager reference for the same inputs (uses a fresh op instance
            # each iter, so it cannot share state with the CG path).
            eager_inputs = _build_attn_inputs(
                cache_seqlens=replay_seqlens,
                input_lengths=input_lengths,
                block_id_host=block_id_host,
                block_id_device=block_id_device,
                is_cuda_graph=False,
                dtype=torch.bfloat16,
                decode_cu_seqlens_d=decode_cu_seqlens_d,
                device=self.device,
            )
            eager_op = self._make_op_eager(eager_inputs)
            out_eager = eager_op.forward(q_per_replay[it], kv_cache).clone()

            compare_tensors(
                out_cg,
                out_eager,
                rtol=5e-3,
                atol=5e-3,
                name=f"[multi-replay iter {it} seqlens={replay_seqlens}] CG vs eager",
            )

    def test_padded_batch_capture_smaller_replay_reproduces_bug(self):
        """Repro of the production scenario where the **graph instance bs is
        LARGER than the actual current_batch_size** at replay.

        Production runs N capture_range graph_instances (e.g., bs=1,2,4,8).
        Each request batch picks the smallest captured bs >= actual.  When
        actual_bs < captured_bs (very common when concurrency dips below the
        max capture size), prepareInputs:
          - zero-fills padding rows of prefix_lengths_d / input_lengths_d
            (cuda_graph_runner.cc:206-213, 250-263, 295-318);
          - clears full kv_cache_kernel_block_id_device, then
            copySmallerIntoLarger fills only the leading actual_bs rows.
        The captured FA3 kernel still processes the full captured_bs rows
        (cu_seqlens_q has captured_bs+1 entries).  Padding rows have
        cache_seqlens=0 / page_table=0 / q=stale.

        If FA3 leaks state across rows under PackGQA tile scheduling, or
        confuses the active rows' work allocation with the padding rows
        (e.g., scheduler decisions made at capture-time off the all-zero
        cache_seqlens), the bug should manifest as wrong output on rows
        0..actual_bs.

        This is the closest in-isolation analogue to the production failure.
        """
        captured_bs = 8  # max graph instance bs
        actual_bs = 4  # active requests
        # Heterogeneous mix on the active requests — mirrors taobao+cooking.
        active_seqlens = [4000, 14000, 24000, 8000]
        # Padding rows: prefix=0 + input=0 = cache_seqlen=0 (production behavior).
        replay_cache_seqlens = active_seqlens + [0] * (captured_bs - actual_bs)
        capture_cache_seqlens = [MAX_SEQ_LEN] * captured_bs

        input_lengths_active = [NUM_DRAFT_TOKENS] * actual_bs
        # Padding rows have input_length=0 (prepareInputs zeros these).
        input_lengths_full = input_lengths_active + [0] * (captured_bs - actual_bs)
        total_q = captured_bs * NUM_DRAFT_TOKENS  # captured graph processes all rows

        # KV cache sized for the active requests; padding rows read page 0.
        kv_cache = self._build_total_kv_cache(active_seqlens)

        # decode_cu_seqlens_d at capture: arange(0, (captured_bs+1)*4, 4)
        # NOTE: at REPLAY production overwrites entries (current_batch_size+1
        # .. captured_bs+1) with last_offset = current_batch_size * num_tokens_per_bs
        # (cuda_graph_runner.cc:210-213).  We replicate that below — without it,
        # the captured cu_seqlens_q would still say padding rows have 4 q-tokens
        # each.
        decode_cu_seqlens_d = torch.arange(
            0,
            (captured_bs + 1) * NUM_DRAFT_TOKENS,
            NUM_DRAFT_TOKENS,
            dtype=torch.int32,
            device=self.device,
        )

        # Block tables sized for capture worst case across captured_bs rows.
        max_blocks = math.ceil(MAX_SEQ_LEN / PAGE_SIZE) + 4
        # Stable buffers (data_ptrs preserved across capture/replay).
        block_id_host = torch.zeros(
            (captured_bs, max_blocks), dtype=torch.int32, device="cpu"
        )
        block_id_device = torch.zeros(
            (captured_bs, max_blocks), dtype=torch.int32, device=self.device
        )

        q_buf = torch.zeros(
            total_q, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )

        # Capture-time inputs: production behavior — zero device lengths,
        # zero page table, zero q.
        attn_inputs_capture = _build_attn_inputs(
            cache_seqlens=capture_cache_seqlens,
            input_lengths=[NUM_DRAFT_TOKENS] * captured_bs,
            block_id_host=block_id_host,
            block_id_device=block_id_device,
            is_cuda_graph=True,
            dtype=torch.bfloat16,
            decode_cu_seqlens_d=decode_cu_seqlens_d,
            device=self.device,
            zero_device_lengths=True,
        )
        cg_op = self._make_op_cg(attn_inputs_capture)

        # Warmup at capture-time (zero) state.
        q_buf.zero_()
        for _ in range(2):
            _ = cg_op.forward(q_buf, kv_cache)
        torch.cuda.synchronize()

        # Capture.
        graph = torch.cuda.CUDAGraph()
        out_cg_buf = torch.empty_like(q_buf)
        with torch.cuda.graph(graph):
            captured = cg_op.forward(q_buf, kv_cache)
            out_cg_buf.copy_(captured)

        # Replay refresh: fill leading actual_bs rows of page_table with live
        # block IDs (production: copySmallerIntoLarger after clearTensorAsync).
        active_block_h, _ = _build_block_table(
            active_seqlens, PAGE_SIZE, max_blocks, self.device
        )
        # Mirror production: clear the captured page_table, then copy leading rows.
        block_id_device.zero_()
        block_id_device[:actual_bs, :].copy_(active_block_h.to(self.device))
        block_id_host.zero_()
        block_id_host[:actual_bs, :].copy_(active_block_h)

        # Refresh prefix/input lengths in the SAME storage the op stashed at
        # capture (data_ptrs preserved).
        replay_inputs = _build_attn_inputs(
            cache_seqlens=replay_cache_seqlens,
            input_lengths=input_lengths_full,
            block_id_host=block_id_host,
            block_id_device=block_id_device,
            is_cuda_graph=True,
            dtype=torch.bfloat16,
            decode_cu_seqlens_d=decode_cu_seqlens_d,
            device=self.device,
        )
        attn_inputs_capture.prefix_lengths_d.copy_(replay_inputs.prefix_lengths_d)
        attn_inputs_capture.input_lengths_d.copy_(replay_inputs.input_lengths_d)

        # Mirror cuda_graph_runner.cc:210-213: collapse padding cu_seqlens to
        # last_offset so padding rows have q[last_offset:last_offset] = empty.
        last_offset = actual_bs * NUM_DRAFT_TOKENS
        decode_cu_seqlens_d[actual_bs + 1 :].fill_(last_offset)

        cg_op.prepare(attn_inputs_capture)

        # Replay-time q: leading slots = real, padding slots = STALE garbage.
        # Production can have anything in padding rows of q (no explicit zero).
        # If FA3 leaks tile work from padding into active rows under PackGQA,
        # the stale garbage pollutes active output.
        torch.manual_seed(123)
        active_q = torch.randn(
            actual_bs * NUM_DRAFT_TOKENS,
            HEAD_NUM,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        # Stale garbage with very different magnitude — easy to detect leakage.
        stale_garbage = (
            torch.randn(
                (captured_bs - actual_bs) * NUM_DRAFT_TOKENS,
                HEAD_NUM,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 100.0
        )
        q_replay = torch.cat([active_q, stale_garbage], dim=0)
        q_buf.copy_(q_replay)

        graph.replay()
        torch.cuda.synchronize()
        out_cg = out_cg_buf.clone()

        # Reference: per-request FA3 on ACTIVE requests only.
        ref = _reference_per_request(
            q=active_q,
            cache_seqlens=active_seqlens,
            input_lengths=input_lengths_active,
            page_table_h=active_block_h,
            paged_kv_cache=kv_cache.kv_cache_base,
            page_size=PAGE_SIZE,
            softmax_scale=HEAD_DIM**-0.5,
        )

        # Compare leading rows of CG output against per-request reference.
        compare_tensors(
            out_cg[: actual_bs * NUM_DRAFT_TOKENS],
            ref,
            rtol=5e-3,
            atol=5e-3,
            name=f"[padded captured_bs={captured_bs} actual_bs={actual_bs}] "
            f"CG vs per-request reference (active rows only)",
        )

    def test_replay_reads_cache_seqlens_at_runtime(self):
        """Sanity probe: confirm FA3 actually reads cache_seqlens at replay
        time (not captured-as-immediate).

        Captures with zero cache_seqlens → no work.  Then before each replay,
        sets cache_seqlens to two DIFFERENT values and verifies output
        actually changes accordingly (i.e., the captured kernel honours the
        live buffer).  If it ever returns the same output for two different
        cache_seqlens, that's proof FA3 baked something at capture time —
        the smoking gun for the production bug.
        """
        bs = 4
        capture_seqlens = [MAX_SEQ_LEN] * bs
        input_lengths = [NUM_DRAFT_TOKENS] * bs
        total_q = sum(input_lengths)

        # Two distinct probe seqlens patterns.  Pick widely different so the
        # output difference dominates any near-zero noise.
        probe_a = [4000, 14000, 24000, 8000]
        probe_b = [24000, 4000, 8000, 14000]  # shuffled

        kv_cache = self._build_total_kv_cache([max(probe_a + probe_b)] * bs)
        decode_cu_seqlens_d = torch.arange(
            0,
            (bs + 1) * NUM_DRAFT_TOKENS,
            NUM_DRAFT_TOKENS,
            dtype=torch.int32,
            device=self.device,
        )
        max_blocks = math.ceil(MAX_SEQ_LEN / PAGE_SIZE) + 4
        block_id_host = torch.zeros((bs, max_blocks), dtype=torch.int32, device="cpu")
        block_id_device = torch.zeros(
            (bs, max_blocks), dtype=torch.int32, device=self.device
        )
        q_buf = self._build_q_buffer(total_q)

        # Capture in the production-fidelity zero state.
        attn_inputs_capture = _build_attn_inputs(
            cache_seqlens=capture_seqlens,
            input_lengths=input_lengths,
            block_id_host=block_id_host,
            block_id_device=block_id_device,
            is_cuda_graph=True,
            dtype=torch.bfloat16,
            decode_cu_seqlens_d=decode_cu_seqlens_d,
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

        torch.manual_seed(42)
        q_replay = torch.randn_like(q_buf)

        def replay_with(probe_seqlens: List[int]) -> torch.Tensor:
            probe_block_h, _ = _build_block_table(
                probe_seqlens, PAGE_SIZE, max_blocks, self.device
            )
            block_id_device.zero_()
            block_id_device.copy_(probe_block_h.to(self.device))
            block_id_host.zero_()
            block_id_host.copy_(probe_block_h)

            probe_inputs = _build_attn_inputs(
                cache_seqlens=probe_seqlens,
                input_lengths=input_lengths,
                block_id_host=block_id_host,
                block_id_device=block_id_device,
                is_cuda_graph=True,
                dtype=torch.bfloat16,
                decode_cu_seqlens_d=decode_cu_seqlens_d,
                device=self.device,
            )
            attn_inputs_capture.prefix_lengths_d.copy_(probe_inputs.prefix_lengths_d)
            attn_inputs_capture.input_lengths_d.copy_(probe_inputs.input_lengths_d)
            cg_op.prepare(attn_inputs_capture)

            q_buf.copy_(q_replay)
            graph.replay()
            torch.cuda.synchronize()
            return out_cg_buf.clone()

        out_a = replay_with(probe_a)
        out_b = replay_with(probe_b)

        diff = (out_a - out_b).abs().max().item()
        self.assertGreater(
            diff,
            1e-3,
            f"FA3 captured kernel returned IDENTICAL output for two distinct "
            f"cache_seqlens patterns (max diff = {diff}); this proves the "
            f"kernel baked cache_seqlens at capture time and ignores the "
            f"refreshed live buffer at replay — the production bug.",
        )

    def test_heterogeneous_eager_only_passes(self):
        """Negative control: eager (non-CG) heterogeneous works fine, proving
        the bug is CG-specific.  This lifts the CG path entirely from the
        comparison."""
        bs = 4
        cache_seqlens = [4000, 14000, 24000, 8000]
        input_lengths = [NUM_DRAFT_TOKENS] * bs

        kv_cache = self._build_total_kv_cache(cache_seqlens)
        max_blocks = max(math.ceil(cs / PAGE_SIZE) for cs in cache_seqlens) + 4
        block_id_host, block_id_device = _build_block_table(
            cache_seqlens, PAGE_SIZE, max_blocks, self.device
        )
        decode_cu_seqlens_d = torch.arange(
            0,
            (bs + 1) * NUM_DRAFT_TOKENS,
            NUM_DRAFT_TOKENS,
            dtype=torch.int32,
            device=self.device,
        )
        attn_inputs = _build_attn_inputs(
            cache_seqlens=cache_seqlens,
            input_lengths=input_lengths,
            block_id_host=block_id_host,
            block_id_device=block_id_device,
            is_cuda_graph=False,
            dtype=torch.bfloat16,
            decode_cu_seqlens_d=decode_cu_seqlens_d,
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
            page_table_h=block_id_host,
            paged_kv_cache=kv_cache.kv_cache_base,
            page_size=PAGE_SIZE,
            softmax_scale=HEAD_DIM**-0.5,
        )
        compare_tensors(
            out,
            ref,
            rtol=5e-3,
            atol=5e-3,
            name="[heterogeneous-eager-only] op vs per-request reference",
        )


if __name__ == "__main__":
    unittest.main()
