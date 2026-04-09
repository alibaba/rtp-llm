"""
Shared test utilities for context-parallel all-gather prefill attention tests.

Provides:
  - Pure-Python zigzag helpers (no C++ dependency)
  - Reference single-GPU causal attention via FlashInfer
  - Builder functions for PyAttentionInputs / PyContextParallelParams / KVCache
  - A generic single-GPU correctness driver that mocks ``all_gather``
"""

import contextlib
import logging
import math
import unittest
from typing import List, Tuple
from unittest.mock import patch

import torch
from flashinfer.prefill import single_prefill_with_kv_cache

from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    LayerKVCache,
    PyAttentionInputs,
    PyContextParallelParams,
    fill_mla_params,
    get_typemeta,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ---------------------------------------------------------------------------
# Zigzag helpers
# ---------------------------------------------------------------------------


def zigzag_positions_for_rank(full_len: int, cp_size: int, rank: int) -> List[int]:
    """Original-sequence positions assigned to *rank* under zigzag.

    Example (cp_size=2, full_len=16):
      rank 0 -> [0,1,2,3, 12,13,14,15]
      rank 1 -> [4,5,6,7,  8, 9,10,11]
    """
    chunk_len = full_len // cp_size
    half = chunk_len // 2
    first = list(range(rank * half, (rank + 1) * half))
    second = list(range(full_len - (rank + 1) * half, full_len - rank * half))
    return first + second


def build_restore_indices(cp_chunk_lengths: List[int], cp_size: int) -> torch.Tensor:
    """``prefill_qkv_restore_indice``: original-pos -> all-gathered-idx."""
    batch_size = len(cp_chunk_lengths)
    ag_positions: List[int] = []
    for rank in range(cp_size):
        for b in range(batch_size):
            chunk_len = cp_chunk_lengths[b]
            full_len = chunk_len * cp_size
            positions = zigzag_positions_for_rank(full_len, cp_size, rank)
            seq_offset = sum(cp_chunk_lengths[:b]) * cp_size
            ag_positions.extend([p + seq_offset for p in positions])

    total = len(ag_positions)
    restore = [0] * total
    for ag_idx, orig_pos in enumerate(ag_positions):
        restore[orig_pos] = ag_idx
    return torch.tensor(restore, dtype=torch.int32)


def build_padding_mask(cp_chunk_lengths: List[int], cp_size: int) -> torch.Tensor:
    return torch.ones(sum(cp_chunk_lengths) * cp_size, dtype=torch.int32)


# ---------------------------------------------------------------------------
# Reference attention (single-GPU, FlashInfer)
# ---------------------------------------------------------------------------


def reference_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: List[int],
) -> torch.Tensor:
    """Per-sequence causal attention.  q, k, v are ragged with *equal* lengths."""
    outputs = []
    for i in range(len(cu_seqlens) - 1):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        outputs.append(
            single_prefill_with_kv_cache(
                q[s:e], k[s:e], v[s:e], causal=True, kv_layout="NHD"
            )
        )
    return torch.cat(outputs, dim=0)


def reference_prefill_with_prefix(
    new_q: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    new_lengths: List[int],
    prefix_lengths: List[int],
) -> torch.Tensor:
    """Reference for chunked prefill: Q is *new* tokens only, KV is prefix + new.

    FlashInfer handles q_len < kv_len with causal masking correctly:
        q[i] attends to k[j] iff j <= i + (kv_len - q_len).
    """
    outputs = []
    q_off, pk_off, nk_off = 0, 0, 0
    for new_len, pfx_len in zip(new_lengths, prefix_lengths):
        q_i = new_q[q_off : q_off + new_len]
        k_i = torch.cat(
            [prefix_k[pk_off : pk_off + pfx_len], new_k[nk_off : nk_off + new_len]],
            dim=0,
        )
        v_i = torch.cat(
            [prefix_v[pk_off : pk_off + pfx_len], new_v[nk_off : nk_off + new_len]],
            dim=0,
        )
        outputs.append(
            single_prefill_with_kv_cache(q_i, k_i, v_i, causal=True, kv_layout="NHD")
        )
        q_off += new_len
        pk_off += pfx_len
        nk_off += new_len
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Config / input builders
# ---------------------------------------------------------------------------


def make_configs(
    head_num: int = 8,
    kv_head_num: int = 2,
    head_dim: int = 64,
    tokens_per_block: int = 16,
    cp_size: int = 2,
    cp_rank: int = 0,
) -> Tuple[AttentionConfigs, ParallelismConfig]:
    attn_cfg = AttentionConfigs()
    attn_cfg.head_num = head_num
    attn_cfg.kv_head_num = kv_head_num
    attn_cfg.size_per_head = head_dim
    attn_cfg.tokens_per_block = tokens_per_block
    attn_cfg.kernel_tokens_per_block = tokens_per_block
    attn_cfg.use_mla = False

    par_cfg = ParallelismConfig()
    par_cfg.tp_size = cp_size
    par_cfg.tp_rank = cp_rank
    return attn_cfg, par_cfg


def build_cp_attn_inputs(
    sequence_lengths: List[int],
    cp_chunk_lengths: List[int],
    cp_size: int,
    tokens_per_block: int,
    prefix_lengths: List[int] | None = None,
    device: torch.device = torch.device("cuda"),
) -> PyAttentionInputs:
    """Build ``PyAttentionInputs`` with properly populated CP info.

    ``prefix_lengths``: per-batch prefix cache lengths (default all-zero).
    """
    batch_size = len(cp_chunk_lengths)
    if prefix_lengths is None:
        prefix_lengths = [0] * batch_size

    inp = PyAttentionInputs()
    inp.is_prefill = True

    inp.input_lengths = torch.tensor(
        cp_chunk_lengths, dtype=torch.int32, device="cpu"
    ).pin_memory()
    inp.sequence_lengths = torch.tensor(
        sequence_lengths, dtype=torch.int32, device="cpu"
    ).pin_memory()
    inp.prefix_lengths = torch.tensor(prefix_lengths, dtype=torch.int32, device="cpu")

    cu = [0]
    for cl in cp_chunk_lengths:
        cu.append(cu[-1] + cl)
    inp.cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)

    max_blocks = max(math.ceil(sl / tokens_per_block) for sl in sequence_lengths)
    block_ids = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
    offset = 0
    for i, sl in enumerate(sequence_lengths):
        nb = math.ceil(sl / tokens_per_block)
        block_ids[i, :nb] = torch.arange(offset, offset + nb, dtype=torch.int32)
        offset += nb
    inp.kv_cache_block_id_host = block_ids
    inp.kv_cache_kernel_block_id_host = block_ids
    inp.kv_cache_block_id_device = block_ids.to(device)
    inp.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))

    # new_lengths = sequence_lengths - prefix_lengths (total new tokens per batch)
    new_lengths = [sl - pl for sl, pl in zip(sequence_lengths, prefix_lengths)]

    cp_info = PyContextParallelParams()
    cp_info.prefill_cp_chunk_lengths = torch.tensor(cp_chunk_lengths, dtype=torch.int32)
    cp_info.prefill_cp_padding_lengths = torch.zeros(batch_size, dtype=torch.int32)
    cp_info.prefill_qkv_padding_mask = build_padding_mask(cp_chunk_lengths, cp_size).to(
        device
    )
    cp_info.prefill_qkv_restore_indice = build_restore_indices(
        cp_chunk_lengths, cp_size
    ).to(device)
    cp_info.prefill_actual_input_lengths_cpu = torch.tensor(
        new_lengths, dtype=torch.int32
    )
    cp_info.prefill_shuffle_indices = torch.tensor([], dtype=torch.int32)
    inp.context_parallel_info = cp_info
    return inp


def make_kv_cache(
    total_blocks: int,
    kv_head_num: int,
    tokens_per_block: int,
    head_dim: int,
    device: torch.device = torch.device("cuda"),
) -> LayerKVCache:
    kv = LayerKVCache()
    kv.kv_cache_base = torch.zeros(
        total_blocks,
        2,
        kv_head_num,
        tokens_per_block,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    return kv


def extract_kv_from_paged_cache(
    kv_cache: LayerKVCache,
    sequence_lengths: List[int],
    tokens_per_block: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read K and V back from a paged KV cache (HND layout).

    Returns (K, V) each as [total_tokens, H_kv, D] in NHD layout.
    Assumes blocks are allocated sequentially per sequence.
    """
    cache = kv_cache.kv_cache_base  # [blocks, 2, H_kv, page_size, D]
    all_k: List[torch.Tensor] = []
    all_v: List[torch.Tensor] = []
    block_offset = 0
    for seq_len in sequence_lengths:
        n_blocks = math.ceil(seq_len / tokens_per_block)
        remaining = seq_len
        for b in range(n_blocks):
            n_tokens = min(tokens_per_block, remaining)
            # HND [H, tokens, D] -> NHD [tokens, H, D]
            all_k.append(cache[block_offset + b, 0, :, :n_tokens, :].permute(1, 0, 2))
            all_v.append(cache[block_offset + b, 1, :, :n_tokens, :].permute(1, 0, 2))
            remaining -= n_tokens
        block_offset += n_blocks
    return torch.cat(all_k, dim=0), torch.cat(all_v, dim=0)


def fill_prefix_into_kv_cache(
    kv_cache: LayerKVCache,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    prefix_lengths: List[int],
    sequence_lengths: List[int],
    tokens_per_block: int,
):
    """Write prefix K/V into the paged KV cache (HND layout).

    ``prefix_k``, ``prefix_v`` are ragged [total_prefix_tokens, H_kv, D] in NHD.
    """
    cache = kv_cache.kv_cache_base  # [blocks, 2, H_kv, page_size, D]
    block_offset = 0
    token_offset = 0
    for pfx_len, seq_len in zip(prefix_lengths, sequence_lengths):
        n_blocks_seq = math.ceil(seq_len / tokens_per_block)
        n_prefix_blocks = pfx_len // tokens_per_block
        for b in range(n_prefix_blocks):
            s = token_offset + b * tokens_per_block
            e = s + tokens_per_block
            # NHD -> HND
            cache[block_offset + b, 0] = prefix_k[s:e].permute(1, 0, 2)
            cache[block_offset + b, 1] = prefix_v[s:e].permute(1, 0, 2)
        block_offset += n_blocks_seq
        token_offset += pfx_len


# ---------------------------------------------------------------------------
# Zigzag split helpers
# ---------------------------------------------------------------------------


def compute_rank_positions(lengths: List[int], cp_size: int) -> List[List[int]]:
    """For each rank, return original-sequence positions (global, across batch)."""
    all_rank_positions: List[List[int]] = [[] for _ in range(cp_size)]
    seq_offset = 0
    for ln in lengths:
        for r in range(cp_size):
            positions = zigzag_positions_for_rank(ln, cp_size, r)
            all_rank_positions[r].extend([p + seq_offset for p in positions])
        seq_offset += ln
    return all_rank_positions


# ---------------------------------------------------------------------------
# Generic correctness driver
# ---------------------------------------------------------------------------


class CPAttnTestBase(unittest.TestCase):
    """Base class with the correctness drivers.  Subclasses set ``OP_CLASS``
    and ``AG_MODULE`` (the module path where ``all_gather`` is imported)."""

    OP_CLASS = None  # override in subclass
    AG_MODULE: str = ""  # override in subclass

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    # ---- helpers ----

    def _extra_patches(self, stack: contextlib.ExitStack):
        """Override to add extra mock patches (e.g. user-buffers)."""

    def _assert_close(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ):
        af, ef = actual.float(), expected.float()
        diff = (af - ef).abs()
        max_diff, mean_diff = diff.max().item(), diff.mean().item()
        logging.info(
            f"  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
            f"(rtol={rtol}, atol={atol})"
        )
        self.assertTrue(
            torch.allclose(af, ef, rtol=rtol, atol=atol),
            f"Output mismatch: max_diff={max_diff}, mean_diff={mean_diff}",
        )

    # ---- no-prefix test driver ----

    def run_no_prefix(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        cp_size: int = 2,
        cp_rank: int = 0,
        head_num: int = 8,
        kv_head_num: int = 2,
        head_dim: int = 64,
        tokens_per_block: int = 16,
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ):
        """Test CP attention **without** prefix cache."""
        assert all(sl % cp_size == 0 for sl in sequence_lengths)
        cp_chunk_lengths = [sl // cp_size for sl in sequence_lengths]
        assert all(cl % 2 == 0 for cl in cp_chunk_lengths)

        attn_cfg, par_cfg = make_configs(
            head_num=head_num,
            kv_head_num=kv_head_num,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )

        total_tokens = sum(sequence_lengths)
        q_full = torch.randn(
            total_tokens, head_num, head_dim, dtype=torch.bfloat16, device=self.device
        )
        k_full = torch.randn(
            total_tokens,
            kv_head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        v_full = torch.randn(
            total_tokens,
            kv_head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        cu_full = [0]
        for sl in sequence_lengths:
            cu_full.append(cu_full[-1] + sl)
        ref_output = reference_causal_attention(q_full, k_full, v_full, cu_full)

        all_rank_pos = compute_rank_positions(sequence_lengths, cp_size)
        all_local_k = [
            k_full[torch.tensor(p, device=self.device)].reshape(
                -1, kv_head_num * head_dim
            )
            for p in all_rank_pos
        ]
        all_local_v = [
            v_full[torch.tensor(p, device=self.device)].reshape(
                -1, kv_head_num * head_dim
            )
            for p in all_rank_pos
        ]

        rank_idx = torch.tensor(all_rank_pos[cp_rank], device=self.device)
        qkv = torch.cat(
            [
                q_full[rank_idx].reshape(-1, head_num * head_dim),
                k_full[rank_idx].reshape(-1, kv_head_num * head_dim),
                v_full[rank_idx].reshape(-1, kv_head_num * head_dim),
            ],
            dim=-1,
        )

        attn_inputs = build_cp_attn_inputs(
            sequence_lengths,
            cp_chunk_lengths,
            cp_size,
            tokens_per_block,
            device=self.device,
        )
        total_blocks = sum(math.ceil(s / tokens_per_block) for s in sequence_lengths)
        kv_cache = make_kv_cache(
            total_blocks, kv_head_num, tokens_per_block, head_dim, device=self.device
        )

        call_idx = [0]

        def mock_ag(tensor, group=None):
            data = all_local_k if call_idx[0] % 2 == 0 else all_local_v
            call_idx[0] += 1
            return torch.cat(data, dim=0)

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch(f"{self.AG_MODULE}.all_gather", side_effect=mock_ag)
            )
            self._extra_patches(stack)

            op = self.OP_CLASS(attn_cfg, attn_inputs, par_cfg)
            params = op.prepare(attn_inputs)
            output = op.forward(qkv, kv_cache, params)

        self._assert_close(output, ref_output[rank_idx], rtol=rtol, atol=atol)

        cache_k, cache_v = extract_kv_from_paged_cache(
            kv_cache, sequence_lengths, tokens_per_block
        )
        self.assertTrue(
            torch.equal(cache_k, k_full),
            f"KV cache K mismatch: max_diff="
            f"{(cache_k.float() - k_full.float()).abs().max().item()}",
        )
        self.assertTrue(
            torch.equal(cache_v, v_full),
            f"KV cache V mismatch: max_diff="
            f"{(cache_v.float() - v_full.float()).abs().max().item()}",
        )

    # ---- prefix-cache test driver ----

    def run_with_prefix(
        self,
        batch_size: int,
        new_lengths: List[int],
        prefix_lengths: List[int],
        cp_size: int = 2,
        cp_rank: int = 0,
        head_num: int = 8,
        kv_head_num: int = 2,
        head_dim: int = 64,
        tokens_per_block: int = 16,
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ):
        """Test CP attention **with** prefix cache.

        Constraints:
          - ``new_lengths[i] % cp_size == 0``
          - ``(new_lengths[i] // cp_size) % 2 == 0``
          - ``prefix_lengths[i] % tokens_per_block == 0``
        """
        sequence_lengths = [p + n for p, n in zip(prefix_lengths, new_lengths)]
        cp_chunk_lengths = [n // cp_size for n in new_lengths]
        assert all(cl % 2 == 0 for cl in cp_chunk_lengths)
        assert all(pl % tokens_per_block == 0 for pl in prefix_lengths)

        attn_cfg, par_cfg = make_configs(
            head_num=head_num,
            kv_head_num=kv_head_num,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )

        total_prefix = sum(prefix_lengths)
        total_new = sum(new_lengths)

        prefix_k = torch.randn(
            total_prefix,
            kv_head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        prefix_v = torch.randn(
            total_prefix,
            kv_head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        new_q = torch.randn(
            total_new, head_num, head_dim, dtype=torch.bfloat16, device=self.device
        )
        new_k = torch.randn(
            total_new, kv_head_num, head_dim, dtype=torch.bfloat16, device=self.device
        )
        new_v = torch.randn(
            total_new, kv_head_num, head_dim, dtype=torch.bfloat16, device=self.device
        )

        ref_output = reference_prefill_with_prefix(
            new_q,
            prefix_k,
            prefix_v,
            new_k,
            new_v,
            new_lengths,
            prefix_lengths,
        )

        # Zigzag split on NEW tokens only
        all_rank_pos = compute_rank_positions(new_lengths, cp_size)
        all_local_k = [
            new_k[torch.tensor(p, device=self.device)].reshape(
                -1, kv_head_num * head_dim
            )
            for p in all_rank_pos
        ]
        all_local_v = [
            new_v[torch.tensor(p, device=self.device)].reshape(
                -1, kv_head_num * head_dim
            )
            for p in all_rank_pos
        ]

        rank_idx = torch.tensor(all_rank_pos[cp_rank], device=self.device)
        qkv = torch.cat(
            [
                new_q[rank_idx].reshape(-1, head_num * head_dim),
                new_k[rank_idx].reshape(-1, kv_head_num * head_dim),
                new_v[rank_idx].reshape(-1, kv_head_num * head_dim),
            ],
            dim=-1,
        )

        attn_inputs = build_cp_attn_inputs(
            sequence_lengths,
            cp_chunk_lengths,
            cp_size,
            tokens_per_block,
            prefix_lengths=prefix_lengths,
            device=self.device,
        )
        total_blocks = sum(math.ceil(s / tokens_per_block) for s in sequence_lengths)
        kv_cache = make_kv_cache(
            total_blocks, kv_head_num, tokens_per_block, head_dim, device=self.device
        )
        fill_prefix_into_kv_cache(
            kv_cache,
            prefix_k,
            prefix_v,
            prefix_lengths,
            sequence_lengths,
            tokens_per_block,
        )

        call_idx = [0]

        def mock_ag(tensor, group=None):
            data = all_local_k if call_idx[0] % 2 == 0 else all_local_v
            call_idx[0] += 1
            return torch.cat(data, dim=0)

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch(f"{self.AG_MODULE}.all_gather", side_effect=mock_ag)
            )
            self._extra_patches(stack)

            op = self.OP_CLASS(attn_cfg, attn_inputs, par_cfg)
            params = op.prepare(attn_inputs)
            output = op.forward(qkv, kv_cache, params)

        self._assert_close(output, ref_output[rank_idx], rtol=rtol, atol=atol)

        cache_k, cache_v = extract_kv_from_paged_cache(
            kv_cache, sequence_lengths, tokens_per_block
        )
        expected_k_parts: List[torch.Tensor] = []
        expected_v_parts: List[torch.Tensor] = []
        pk_off, nk_off = 0, 0
        for pfx_len, new_len in zip(prefix_lengths, new_lengths):
            expected_k_parts.append(prefix_k[pk_off : pk_off + pfx_len])
            expected_k_parts.append(new_k[nk_off : nk_off + new_len])
            expected_v_parts.append(prefix_v[pk_off : pk_off + pfx_len])
            expected_v_parts.append(new_v[nk_off : nk_off + new_len])
            pk_off += pfx_len
            nk_off += new_len
        expected_cache_k = torch.cat(expected_k_parts, dim=0)
        expected_cache_v = torch.cat(expected_v_parts, dim=0)
        self.assertTrue(
            torch.equal(cache_k, expected_cache_k),
            f"KV cache K mismatch: max_diff="
            f"{(cache_k.float() - expected_cache_k.float()).abs().max().item()}",
        )
        self.assertTrue(
            torch.equal(cache_v, expected_cache_v),
            f"KV cache V mismatch: max_diff="
            f"{(cache_v.float() - expected_cache_v.float()).abs().max().item()}",
        )
