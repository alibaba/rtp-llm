"""RTP correctness test for the TP1 Mega HCA attention sublayer.

Mirrors ``test_mega_csa_rtp_eager``: one real AttentionFP8 layer
(``compress_ratio == 128``, no indexer) with deterministic random weights, real
pybind KVCache pools, and the original ``Block.forward_decode`` attention
branch as the reference. HCA has no indexer/TopK/MQA stage; the dense
compressed index comes from the per-step metadata.
"""

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV, HCA_STATE, SWA_KV
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    dequantize_slots_to_bf16,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
    build_decode_metadata_fp8,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_runtime import MegaCSARuntime
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
    DIM,
    HC,
    HEAD_DIM,
    MAIN_HEADS,
    Q_LORA_RANK,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_adapter import MegaHCAAdapter
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_weights import (
    HCA_APE_ROWS,
    HCA_COMPRESS_RATIO,
    HCA_STATE_WIDTH,
)
from rtp_llm.models_py.modules.dsv4.hc import build_hc_unit
from rtp_llm.ops.compute_ops import CacheGroupType, KVCache, KVCacheRegionName
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.utils.model_weight import W

_TEST_MAX_SEQ_LEN = 4096
_TOKENS_PER_BLOCK = 256
# DSV4CacheConfigHelper.cc: HCA_KV entries = kernel_tokens_per_block / 128.
_HCA_ENTRIES_PER_BLOCK = _TOKENS_PER_BLOCK // HCA_COMPRESS_RATIO
_SWA_ENTRIES_PER_BLOCK = 128
# computeStateRing(128, kHcaOverlap=0, gen=0) = 128 — NOT the CSA overlap=1.
_HCA_STATE_ENTRIES_PER_BLOCK = 128
_KV_ENTRY_BYTES = 584
_KV_BLOCK_ALIGNMENT_BYTES = 576
_REGION_COUNT = 8
_INDEX_TOPK = 1024
_O_GROUPS = 16
_O_LORA_RANK = 1024
_ROPE_DIM = 64


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _random_bf16(
    shape: tuple[int, ...], device: torch.device, *, scale: float = 0.02
) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=torch.bfloat16).mul_(scale)


def _random_fp8(
    shape: tuple[int, ...], device: torch.device, *, scale: float = 0.02
) -> torch.Tensor:
    return _random_bf16(shape, device, scale=scale).to(torch.float8_e4m3fn)


def _ue8m0_ones(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.full(shape, 127, dtype=torch.uint8, device=device).view(
        torch.float8_e8m0fnu
    )


def _make_layer_weights(device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260818)
    weights = {
        W.v4_attn_wq_a_w: _random_fp8((Q_LORA_RANK, DIM), device),
        W.v4_attn_wq_a_s: _ue8m0_ones((Q_LORA_RANK // 128, DIM // 128), device),
        W.v4_attn_wkv_w: _random_fp8((HEAD_DIM, DIM), device),
        W.v4_attn_wkv_s: _ue8m0_ones((HEAD_DIM // 128, DIM // 128), device),
        W.v4_attn_wq_b_w: _random_fp8(
            (MAIN_HEADS * HEAD_DIM, Q_LORA_RANK), device, scale=0.03
        ),
        W.v4_attn_wq_b_s: _ue8m0_ones(
            (MAIN_HEADS * HEAD_DIM // 128, Q_LORA_RANK // 128), device
        ),
        W.v4_compressor_wkv: _random_bf16((HCA_STATE_WIDTH, DIM), device),
        W.v4_compressor_wgate: _random_bf16((HCA_STATE_WIDTH, DIM), device),
        W.v4_attn_q_norm: torch.rand(Q_LORA_RANK, device=device).add_(0.5).bfloat16(),
        W.v4_attn_kv_norm: torch.rand(HEAD_DIM, device=device).add_(0.5).bfloat16(),
        W.v4_compressor_norm: torch.rand(HEAD_DIM, device=device).add_(0.5).bfloat16(),
        W.v4_compressor_ape: torch.randn(
            HCA_APE_ROWS, HCA_STATE_WIDTH, device=device
        ).mul_(0.02),
        W.v4_hc_attn_fn: torch.randn(24, HC * DIM, device=device).mul_(0.01),
        W.v4_hc_attn_base: torch.randn(24, device=device).mul_(0.1),
        W.v4_hc_attn_scale: torch.rand(3, device=device).add_(0.5),
        W.v4_attn_norm: torch.rand(DIM, device=device).add_(0.5).bfloat16(),
        W.v4_attn_sink: torch.randn(MAIN_HEADS, device=device),
    }

    o_group_input = MAIN_HEADS * HEAD_DIM // _O_GROUPS
    weights.update(
        {
            W.v4_attn_wo_a_w: _random_fp8(
                (_O_GROUPS * _O_LORA_RANK, o_group_input), device, scale=0.01
            ),
            W.v4_attn_wo_a_s: _ue8m0_ones(
                (_O_GROUPS * _O_LORA_RANK // 128, o_group_input // 128), device
            ),
            W.v4_attn_wo_b_w: _random_fp8(
                (DIM, _O_GROUPS * _O_LORA_RANK), device, scale=0.01
            ),
            W.v4_attn_wo_b_s: _ue8m0_ones(
                (DIM // 128, _O_GROUPS * _O_LORA_RANK // 128), device
            ),
        }
    )
    return weights


class _AttentionBlock(torch.nn.Module):
    def __init__(
        self, attention: AttentionFP8, layer_weights: dict[str, torch.Tensor]
    ) -> None:
        super().__init__()
        self.attn = attention
        self.attn_norm = RMSNorm(layer_weights[W.v4_attn_norm], 1.0e-6)
        self.attn_hc = build_hc_unit(
            layer_weights[W.v4_hc_attn_fn],
            layer_weights[W.v4_hc_attn_base],
            layer_weights[W.v4_hc_attn_scale],
            dim=DIM,
            hc_mult=HC,
            hc_sinkhorn_iters=20,
            norm_eps=1.0e-6,
            hc_eps=1.0e-6,
            layer_id=0,
            name="attn",
        )


@dataclass
class _Pools:
    kv_cache: KVCache
    tensors: dict[int, torch.Tensor]
    block_tables: dict[int, torch.Tensor]
    entries_per_block: dict[int, int]
    tokens_per_block: dict[int, int]
    max_seq_len: int

    def reset(self) -> None:
        self.tensors[HCA_KV].zero_()
        self.tensors[SWA_KV].zero_()
        state = self.tensors[HCA_STATE].view(
            self.tensors[HCA_STATE].shape[0],
            _HCA_STATE_ENTRIES_PER_BLOCK,
            2 * HCA_STATE_WIDTH,
        )
        state[..., :HCA_STATE_WIDTH].zero_()
        state[..., HCA_STATE_WIDTH:].fill_(float("-inf"))

    def packed_view(
        self, attn_type: int, entries_per_block: int, entry_bytes: int
    ) -> torch.Tensor:
        raw = self.tensors[attn_type]
        return raw.as_strided(
            (int(raw.shape[0]), entries_per_block, entry_bytes),
            (int(raw.stride(0)), entry_bytes, 1),
        )


def _make_pools(
    device: torch.device,
    batch_size: int,
    max_seq_len: int = _TEST_MAX_SEQ_LEN,
) -> _Pools:
    compressed_pages_per_request = max_seq_len // _TOKENS_PER_BLOCK
    compressed_blocks = 1 + batch_size * compressed_pages_per_request
    fixed_blocks = 1 + batch_size
    hca_stride = _align_up(
        _HCA_ENTRIES_PER_BLOCK * _KV_ENTRY_BYTES,
        _KV_BLOCK_ALIGNMENT_BYTES,
    )
    swa_stride = _align_up(
        _SWA_ENTRIES_PER_BLOCK * _KV_ENTRY_BYTES,
        _KV_BLOCK_ALIGNMENT_BYTES,
    )
    tensors = {
        HCA_KV: torch.zeros(
            compressed_blocks, hca_stride, dtype=torch.uint8, device=device
        ),
        HCA_STATE: torch.empty(
            fixed_blocks,
            _HCA_STATE_ENTRIES_PER_BLOCK * 2 * HCA_STATE_WIDTH,
            dtype=torch.float32,
            device=device,
        ),
        SWA_KV: torch.zeros(fixed_blocks, swa_stride, dtype=torch.uint8, device=device),
    }

    compressed_tables = torch.arange(
        1,
        1 + batch_size * compressed_pages_per_request,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, compressed_pages_per_request)
    fixed_tables = torch.arange(
        1, 1 + batch_size, dtype=torch.int32, device=device
    ).view(batch_size, 1)
    block_tables = {
        HCA_KV: compressed_tables,
        HCA_STATE: fixed_tables,
        SWA_KV: fixed_tables.clone(),
    }
    entries = {
        HCA_KV: _HCA_ENTRIES_PER_BLOCK,
        HCA_STATE: _HCA_STATE_ENTRIES_PER_BLOCK,
        SWA_KV: _SWA_ENTRIES_PER_BLOCK,
    }
    tokens = {attn_type: _TOKENS_PER_BLOCK for attn_type in entries}

    kv_cache = KVCache()
    kv_cache.seq_size_per_block = _TOKENS_PER_BLOCK
    kv_cache.kernel_seq_size_per_block = _TOKENS_PER_BLOCK
    kv_cache.layer_group_types = [CacheGroupType.FULL]
    kv_cache.group_region_names = [
        KVCacheRegionName.CSA_KV,
        KVCacheRegionName.HCA_KV,
        KVCacheRegionName.INDEXER_KV,
        KVCacheRegionName.INDEXER_STATE,
        KVCacheRegionName.CSA_STATE,
        KVCacheRegionName.HCA_STATE,
        KVCacheRegionName.SWA_KV,
    ]
    kv_cache.group_seq_size_per_block = [_TOKENS_PER_BLOCK] * 7
    region_to_group = [-1] * _REGION_COUNT
    region_to_group[HCA_KV] = 1
    region_to_group[HCA_STATE] = 5
    region_to_group[SWA_KV] = 6
    kv_cache.layer_region_to_group_id = [region_to_group]
    empty = torch.empty(0, dtype=torch.uint8, device=device)
    by_region = [empty] * _REGION_COUNT
    for attn_type, tensor in tensors.items():
        by_region[attn_type] = tensor
    kv_cache.kv_cache_base_by_layer_region = [by_region]
    kv_cache.kv_cache_base_by_layer = [tensors[SWA_KV]]

    pools = _Pools(kv_cache, tensors, block_tables, entries, tokens, max_seq_len)
    pools.reset()
    return pools


def _fill_random_state(pools: _Pools, device: torch.device, seed: int) -> None:
    """Populate the HCA state ring with plausible prior-token compressor rows."""
    generator = torch.Generator(device=device).manual_seed(seed)
    state = pools.tensors[HCA_STATE].view(
        pools.tensors[HCA_STATE].shape[0],
        _HCA_STATE_ENTRIES_PER_BLOCK,
        2 * HCA_STATE_WIDTH,
    )
    kv = torch.randn(
        tuple(state.shape[:2]) + (HCA_STATE_WIDTH,),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).mul_(0.05)
    gate = torch.randn(
        tuple(state.shape[:2]) + (HCA_STATE_WIDTH,),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).mul_(0.5)
    state[..., :HCA_STATE_WIDTH] = kv
    state[..., HCA_STATE_WIDTH:] = gate


def _fill_random_context(pools: _Pools, device: torch.device, seed: int) -> None:
    """Seed valid FP8 packed HCA/SWA cache entries for long-context reads."""
    generator = torch.Generator(device=device).manual_seed(seed)
    batch_size = int(pools.block_tables[HCA_KV].shape[0])
    compressed_count = batch_size * (pools.max_seq_len // HCA_COMPRESS_RATIO)
    compressed_slots = torch.arange(
        _HCA_ENTRIES_PER_BLOCK,
        _HCA_ENTRIES_PER_BLOCK + compressed_count,
        dtype=torch.int64,
        device=device,
    )
    swa_slots = torch.arange(
        _SWA_ENTRIES_PER_BLOCK,
        _SWA_ENTRIES_PER_BLOCK + batch_size * _SWA_ENTRIES_PER_BLOCK,
        dtype=torch.int64,
        device=device,
    )
    compressed = torch.randn(
        compressed_count,
        HEAD_DIM,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.05)
    swa = torch.randn(
        batch_size * _SWA_ENTRIES_PER_BLOCK,
        HEAD_DIM,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.05)
    quantize_and_insert_k_cache(
        compressed,
        pools.packed_view(HCA_KV, _HCA_ENTRIES_PER_BLOCK, _KV_ENTRY_BYTES),
        compressed_slots,
    )
    quantize_and_insert_k_cache(
        swa,
        pools.packed_view(SWA_KV, _SWA_ENTRIES_PER_BLOCK, _KV_ENTRY_BYTES),
        swa_slots,
    )


class MegaHCARTPEagerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        capability = torch.cuda.get_device_capability()
        if capability not in ((10, 0), (10, 3)):
            raise unittest.SkipTest(
                f"Mega HCA requires sm_100a/sm_103a, got {capability}"
            )

        cls.device = torch.device("cuda", torch.cuda.current_device())
        os.environ["DSV4_HC_IMPL"] = "tilelang"
        test_tmpdir = os.environ.get("TEST_TMPDIR")
        if test_tmpdir:
            os.environ["TILELANG_CACHE_DIR"] = os.path.join(
                test_tmpdir, "tilelang_cache"
            )
        weights = _make_layer_weights(cls.device)
        attention = AttentionFP8(
            layer_id=0,
            dim=DIM,
            n_heads=MAIN_HEADS,
            q_lora_rank=Q_LORA_RANK,
            head_dim=HEAD_DIM,
            rope_head_dim=_ROPE_DIM,
            o_lora_rank=_O_LORA_RANK,
            o_groups=_O_GROUPS,
            window_size=128,
            compress_ratio=HCA_COMPRESS_RATIO,
            compress_rope_theta=160000.0,
            rope_theta=10000.0,
            rope_factor=16.0,
            beta_fast=32,
            beta_slow=1,
            original_seq_len=65536,
            max_batch_size=128,
            max_seq_len=65536,
            index_n_heads=64,
            index_head_dim=128,
            index_topk=_INDEX_TOPK,
            norm_eps=1.0e-6,
            layer_weights=weights,
            tp_size=1,
            tp_rank=0,
        )
        attention.reset_rope_cache(cls.device)
        assert attention.indexer is None, "HCA layer must not build an indexer"
        cls.block = _AttentionBlock(attention, weights)
        cls.runtime = MegaCSARuntime()
        cls.adapter = MegaHCAAdapter(cls.block, weights, cls.runtime)
        cls.mega_pools = _make_pools(cls.device, batch_size=1)
        cls.reference_pools = _make_pools(cls.device, batch_size=1)

    def _metadata(self, position: int, pools: _Pools):
        batch_size = int(pools.block_tables[HCA_KV].shape[0])
        return build_decode_metadata_fp8(
            attention_inputs=torch.full(
                (batch_size,), position, dtype=torch.int32, device=self.device
            ),
            q_len=1,
            window_size=128,
            head_dim=HEAD_DIM,
            max_seq_len=pools.max_seq_len,
            compress_ratios=[HCA_COMPRESS_RATIO],
            index_topk=_INDEX_TOPK,
            device=self.device,
            paged_block_tables=pools.block_tables,
            paged_pool_entries_per_block=pools.entries_per_block,
            paged_pool_tokens_per_block=pools.tokens_per_block,
        )

    @torch.inference_mode()
    def _forward_mega(
        self, hidden: torch.Tensor, metadata: object, pools: _Pools
    ) -> torch.Tensor:
        return self.adapter.forward_attention_sublayer(
            self.block,
            hidden,
            metadata,
            kv_cache=pools.kv_cache,
        )

    @torch.inference_mode()
    def _forward_reference(
        self, hidden: torch.Tensor, metadata: object, pools: _Pools
    ) -> torch.Tensor:
        residual = hidden
        x_pre, post, comb = self.block.attn_hc.pre(hidden)
        bsz, q_len, dim = x_pre.shape
        x_pre = self.block.attn_norm(x_pre.reshape(bsz * q_len, dim)).view(
            bsz, q_len, dim
        )
        attn_out = self.block.attn.forward_decode(
            x_pre,
            metadata,
            kv_cache=pools.kv_cache,
        )
        return self.block.attn_hc.post(attn_out, residual, post, comb)

    @torch.inference_mode()
    def _run_mega_step(
        self, position: int, hidden: torch.Tensor, pools: _Pools
    ) -> tuple[torch.Tensor, object]:
        metadata = self._metadata(position, pools)
        self.runtime.begin_decode(metadata)
        output = self._forward_mega(hidden, metadata, pools)
        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        self.assertTrue(torch.isfinite(output).all().item())
        return output, metadata

    @torch.inference_mode()
    def _run_reference_step(
        self, position: int, hidden: torch.Tensor, pools: _Pools
    ) -> tuple[torch.Tensor, object]:
        metadata = self._metadata(position, pools)
        output = self._forward_reference(hidden, metadata, pools)
        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        self.assertTrue(torch.isfinite(output).all().item())
        return output, metadata

    def _assert_written_pools_match(
        self,
        mega_metadata: object,
        reference_metadata: object,
        mega_pools: _Pools,
        reference_pools: _Pools,
        *,
        label: str,
        expect_boundary_write: bool,
    ) -> None:
        cache_specs = [("SWA KV", SWA_KV, _SWA_ENTRIES_PER_BLOCK)]
        if expect_boundary_write:
            cache_specs.append(("HCA KV", HCA_KV, _HCA_ENTRIES_PER_BLOCK))
        for name, attn_type, entries in cache_specs:
            mega_slots = mega_metadata.pool_write_slot_mappings[attn_type]
            reference_slots = reference_metadata.pool_write_slot_mappings[attn_type]
            torch.testing.assert_close(mega_slots, reference_slots, rtol=0.0, atol=0.0)
            self.assertTrue((mega_slots >= 0).all().item(), msg=f"{label} {name}")
            mega_value = dequantize_slots_to_bf16(
                mega_pools.packed_view(attn_type, entries, _KV_ENTRY_BYTES),
                mega_slots.to(torch.int64),
            )
            reference_value = dequantize_slots_to_bf16(
                reference_pools.packed_view(attn_type, entries, _KV_ENTRY_BYTES),
                reference_slots.to(torch.int64),
            )
            value_diff = calc_diff(mega_value.float(), reference_value.float())
            print(f"{label} Mega/reference {name} calc_diff: {value_diff:.6e}")
            self.assertLess(value_diff, 1.0e-3, msg=f"{label} {name}")

        mega_slots = mega_metadata.compressor_state_slot_mappings[HCA_STATE]
        reference_slots = reference_metadata.compressor_state_slot_mappings[HCA_STATE]
        torch.testing.assert_close(mega_slots, reference_slots, rtol=0.0, atol=0.0)
        mega_rows = mega_pools.tensors[HCA_STATE].view(-1, 2 * HCA_STATE_WIDTH)
        reference_rows = reference_pools.tensors[HCA_STATE].view(
            -1, 2 * HCA_STATE_WIDTH
        )
        mega_state = mega_rows[mega_slots.long()]
        reference_state = reference_rows[reference_slots.long()]
        mega_finite = torch.isfinite(mega_state)
        reference_finite = torch.isfinite(reference_state)
        torch.testing.assert_close(mega_finite, reference_finite, rtol=0.0, atol=0.0)
        value_diff = calc_diff(
            mega_state[mega_finite], reference_state[reference_finite]
        )
        print(f"{label} Mega/reference HCA state calc_diff: {value_diff:.6e}")
        self.assertLess(value_diff, 1.0e-4, msg=f"{label} HCA state")

    def test_matches_original_rtp_attention_sublayer(self) -> None:
        self.mega_pools.reset()
        self.reference_pools.reset()
        generator = torch.Generator(device=self.device).manual_seed(2026)
        mega_output = reference_output = None
        mega_metadata = reference_metadata = None

        for position in range(4):
            hidden = torch.randn(
                (1, 1, HC, DIM),
                generator=generator,
                device=self.device,
                dtype=torch.bfloat16,
            ).mul_(0.05)
            reference_output, reference_metadata = self._run_reference_step(
                position, hidden.clone(), self.reference_pools
            )
            mega_output, mega_metadata = self._run_mega_step(
                position, hidden.clone(), self.mega_pools
            )

        assert mega_output is not None and reference_output is not None
        assert mega_metadata is not None and reference_metadata is not None
        output_diff = calc_diff(mega_output.float(), reference_output.float())
        print(f"Mega/reference HCA attention sublayer calc_diff: {output_diff:.6e}")
        self.assertLess(output_diff, 1.0e-3)
        self._assert_written_pools_match(
            mega_metadata,
            reference_metadata,
            self.mega_pools,
            self.reference_pools,
            label="pos 0..3",
            expect_boundary_write=False,
        )

    def test_boundary_compression_writes_hca_kv(self) -> None:
        self.mega_pools.reset()
        self.reference_pools.reset()
        _fill_random_state(self.mega_pools, self.device, seed=1618)
        _fill_random_state(self.reference_pools, self.device, seed=1618)
        generator = torch.Generator(device=self.device).manual_seed(314)
        hidden = torch.randn(
            (1, 1, HC, DIM),
            generator=generator,
            device=self.device,
            dtype=torch.bfloat16,
        ).mul_(0.05)

        boundary = HCA_COMPRESS_RATIO - 1  # first 128-token boundary
        reference_output, reference_metadata = self._run_reference_step(
            boundary, hidden.clone(), self.reference_pools
        )
        mega_output, mega_metadata = self._run_mega_step(
            boundary, hidden.clone(), self.mega_pools
        )

        output_diff = calc_diff(mega_output.float(), reference_output.float())
        print(f"Mega/reference HCA boundary calc_diff: {output_diff:.6e}")
        self.assertLess(output_diff, 1.0e-3)
        self.assertTrue(self.mega_pools.tensors[HCA_KV][1].any().item())
        self._assert_written_pools_match(
            mega_metadata,
            reference_metadata,
            self.mega_pools,
            self.reference_pools,
            label="boundary 127",
            expect_boundary_write=True,
        )

    def test_matches_original_rtp_at_long_context(self) -> None:
        self.mega_pools.reset()
        self.reference_pools.reset()
        _fill_random_context(self.mega_pools, self.device, seed=31415)
        _fill_random_context(self.reference_pools, self.device, seed=31415)
        _fill_random_state(self.mega_pools, self.device, seed=27182)
        _fill_random_state(self.reference_pools, self.device, seed=27182)
        generator = torch.Generator(device=self.device).manual_seed(999)
        hidden = torch.randn(
            (1, 1, HC, DIM),
            generator=generator,
            device=self.device,
            dtype=torch.bfloat16,
        ).mul_(0.05)

        position = self.mega_pools.max_seq_len - 1
        reference_output, _ = self._run_reference_step(
            position, hidden.clone(), self.reference_pools
        )
        mega_output, _ = self._run_mega_step(position, hidden.clone(), self.mega_pools)
        output_diff = calc_diff(mega_output.float(), reference_output.float())
        print(
            "Mega/reference HCA long-context attention sublayer "
            f"calc_diff: {output_diff:.6e}"
        )
        self.assertLess(output_diff, 1.0e-3)

    def test_cuda_graph_capture_and_replay(self) -> None:
        self.mega_pools.reset()
        generator = torch.Generator(device=self.device).manual_seed(47)
        history = [
            torch.randn(
                (1, 1, HC, DIM),
                generator=generator,
                device=self.device,
                dtype=torch.bfloat16,
            ).mul_(0.05)
            for _ in range(4)
        ]

        for position in range(4):
            self._run_mega_step(position, history[position].clone(), self.mega_pools)

        self.mega_pools.reset()
        for position in range(3):
            self._run_mega_step(position, history[position].clone(), self.mega_pools)

        metadata = self._metadata(3, self.mega_pools)
        metadata.is_cuda_graph = True
        self.runtime.begin_decode(metadata)
        torch.cuda.synchronize(self.device)

        graph_input = history[3].clone()
        graph_work = torch.empty_like(graph_input)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_work.copy_(graph_input)
            graph_output = self.adapter.forward_attention_sublayer(
                self.block,
                graph_work,
                metadata,
                kv_cache=self.mega_pools.kv_cache,
            )

        graph.replay()
        torch.cuda.synchronize(self.device)
        first = graph_output.clone()
        graph.replay()
        torch.cuda.synchronize(self.device)
        second = graph_output.clone()
        self.assertTrue(torch.isfinite(first).all().item())
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
