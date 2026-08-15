"""RTP correctness and performance test for the TP1 Mega CSA attention sublayer.

This intentionally builds one real AttentionFP8 layer with deterministic
random weights instead of loading a complete DSV4 checkpoint.  The cache is
the framework's pybind KVCache object populated with production-shaped typed
regions.  The reference is the original attention branch in
``Block.forward_decode`` (mHC pre, RMSNorm, AttentionFP8, mHC post), with an
independent but identically initialized cache.
"""

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4.attn_type import (
    CSA_KV,
    CSA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    dequantize_indexer_k,
    quantize_indexer_k,
)
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
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_adapter import MegaCSAAdapter
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_runtime import MegaCSARuntime
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
    DIM,
    HC,
    HEAD_DIM,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    MAIN_HEADS,
    Q_LORA_RANK,
    ROPE_DIM,
)
from rtp_llm.models_py.modules.dsv4.hc import build_hc_unit
from rtp_llm.ops.compute_ops import CacheGroupType, KVCache, KVCacheRegionName
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.utils.model_weight import W

_MAX_SEQ_LEN = 4096
_TOKENS_PER_BLOCK = 256
_COMPRESSED_ENTRIES_PER_BLOCK = _TOKENS_PER_BLOCK // 4
_SWA_ENTRIES_PER_BLOCK = 128
_STATE_ENTRIES_PER_BLOCK = 8
_KV_ENTRY_BYTES = 584
_INDEXER_ENTRY_BYTES = 132
_KV_BLOCK_ALIGNMENT_BYTES = 576
_REGION_COUNT = 8
_RUN_PERF = os.environ.get("DSV4_MEGA_RUN_PERF", "0") == "1"


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
    torch.manual_seed(20260815)
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
        W.v4_indexer_wq_b_w: _random_fp8(
            (INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK), device, scale=0.03
        ),
        W.v4_indexer_wq_b_s: _ue8m0_ones(
            (INDEX_HEADS * INDEX_HEAD_DIM // 128, Q_LORA_RANK // 128), device
        ),
        W.v4_compressor_wkv: _random_bf16((2 * HEAD_DIM, DIM), device),
        W.v4_compressor_wgate: _random_bf16((2 * HEAD_DIM, DIM), device),
        W.v4_indexer_compressor_wkv: _random_bf16((2 * INDEX_HEAD_DIM, DIM), device),
        W.v4_indexer_compressor_wgate: _random_bf16((2 * INDEX_HEAD_DIM, DIM), device),
        W.v4_indexer_weights_proj_w: _random_bf16((INDEX_HEADS, DIM), device),
        W.v4_attn_q_norm: torch.rand(Q_LORA_RANK, device=device).add_(0.5).bfloat16(),
        W.v4_attn_kv_norm: torch.rand(HEAD_DIM, device=device).add_(0.5).bfloat16(),
        W.v4_indexer_compressor_norm: torch.rand(INDEX_HEAD_DIM, device=device)
        .add_(0.5)
        .bfloat16(),
        W.v4_compressor_norm: torch.rand(HEAD_DIM, device=device).add_(0.5).bfloat16(),
        W.v4_compressor_ape: torch.randn(4, 2 * HEAD_DIM, device=device).mul_(0.02),
        W.v4_indexer_compressor_ape: torch.randn(
            4, 2 * INDEX_HEAD_DIM, device=device
        ).mul_(0.02),
        W.v4_hc_attn_fn: torch.randn(24, HC * DIM, device=device).mul_(0.01),
        W.v4_hc_attn_base: torch.randn(24, device=device).mul_(0.1),
        W.v4_hc_attn_scale: torch.rand(3, device=device).add_(0.5),
        W.v4_attn_norm: torch.rand(DIM, device=device).add_(0.5).bfloat16(),
        W.v4_attn_sink: torch.randn(MAIN_HEADS, device=device),
    }

    o_groups = 8
    o_lora_rank = 1024
    o_group_input = MAIN_HEADS * HEAD_DIM // o_groups
    weights.update(
        {
            W.v4_attn_wo_a_w: _random_fp8(
                (o_groups * o_lora_rank, o_group_input), device, scale=0.01
            ),
            W.v4_attn_wo_a_s: _ue8m0_ones(
                (o_groups * o_lora_rank // 128, o_group_input // 128), device
            ),
            W.v4_attn_wo_b_w: _random_fp8(
                (DIM, o_groups * o_lora_rank), device, scale=0.01
            ),
            W.v4_attn_wo_b_s: _ue8m0_ones(
                (DIM // 128, o_groups * o_lora_rank // 128), device
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

    def reset(self) -> None:
        self.tensors[CSA_KV].zero_()
        self.tensors[INDEXER_KV].zero_()
        self.tensors[SWA_KV].zero_()
        main = self.tensors[CSA_STATE].view(
            self.tensors[CSA_STATE].shape[0], _STATE_ENTRIES_PER_BLOCK, 4 * HEAD_DIM
        )
        indexer = self.tensors[INDEXER_STATE].view(
            self.tensors[INDEXER_STATE].shape[0],
            _STATE_ENTRIES_PER_BLOCK,
            4 * INDEX_HEAD_DIM,
        )
        main[..., : 2 * HEAD_DIM].zero_()
        main[..., 2 * HEAD_DIM :].fill_(float("-inf"))
        indexer[..., : 2 * INDEX_HEAD_DIM].zero_()
        indexer[..., 2 * INDEX_HEAD_DIM :].fill_(float("-inf"))

    def packed_view(
        self, attn_type: int, entries_per_block: int, entry_bytes: int
    ) -> torch.Tensor:
        raw = self.tensors[attn_type]
        return raw.as_strided(
            (int(raw.shape[0]), entries_per_block, entry_bytes),
            (int(raw.stride(0)), entry_bytes, 1),
        )


def _make_pools(device: torch.device, batch_size: int) -> _Pools:
    compressed_pages_per_request = _MAX_SEQ_LEN // _TOKENS_PER_BLOCK
    compressed_blocks = 1 + batch_size * compressed_pages_per_request
    fixed_blocks = 1 + batch_size
    csa_stride = _align_up(
        _COMPRESSED_ENTRIES_PER_BLOCK * _KV_ENTRY_BYTES,
        _KV_BLOCK_ALIGNMENT_BYTES,
    )
    swa_stride = _align_up(
        _SWA_ENTRIES_PER_BLOCK * _KV_ENTRY_BYTES,
        _KV_BLOCK_ALIGNMENT_BYTES,
    )
    tensors = {
        CSA_KV: torch.zeros(
            compressed_blocks, csa_stride, dtype=torch.uint8, device=device
        ),
        INDEXER_KV: torch.zeros(
            compressed_blocks,
            _COMPRESSED_ENTRIES_PER_BLOCK * _INDEXER_ENTRY_BYTES,
            dtype=torch.uint8,
            device=device,
        ),
        INDEXER_STATE: torch.empty(
            fixed_blocks,
            _STATE_ENTRIES_PER_BLOCK * 4 * INDEX_HEAD_DIM,
            dtype=torch.float32,
            device=device,
        ),
        CSA_STATE: torch.empty(
            fixed_blocks,
            _STATE_ENTRIES_PER_BLOCK * 4 * HEAD_DIM,
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
        CSA_KV: compressed_tables,
        INDEXER_KV: compressed_tables.clone(),
        INDEXER_STATE: fixed_tables,
        CSA_STATE: fixed_tables.clone(),
        SWA_KV: fixed_tables.clone(),
    }
    entries = {
        CSA_KV: _COMPRESSED_ENTRIES_PER_BLOCK,
        INDEXER_KV: _COMPRESSED_ENTRIES_PER_BLOCK,
        INDEXER_STATE: _STATE_ENTRIES_PER_BLOCK,
        CSA_STATE: _STATE_ENTRIES_PER_BLOCK,
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
    region_to_group[CSA_KV] = 0
    region_to_group[INDEXER_KV] = 2
    region_to_group[INDEXER_STATE] = 3
    region_to_group[CSA_STATE] = 4
    region_to_group[SWA_KV] = 6
    kv_cache.layer_region_to_group_id = [region_to_group]
    empty = torch.empty(0, dtype=torch.uint8, device=device)
    by_region = [empty] * _REGION_COUNT
    for attn_type, tensor in tensors.items():
        by_region[attn_type] = tensor
    kv_cache.kv_cache_base_by_layer_region = [by_region]
    kv_cache.kv_cache_base_by_layer = [tensors[SWA_KV]]

    pools = _Pools(kv_cache, tensors, block_tables, entries, tokens)
    pools.reset()
    return pools


def _bench_cuda_event(
    fn, *, warmup: int = 10, iterations: int = 30, samples: int = 5
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / iterations)
    timings.sort()
    return timings[len(timings) // 2]


def _fill_random_context(pools: _Pools, device: torch.device, seed: int) -> None:
    generator = torch.Generator(device=device).manual_seed(seed)
    compressed_count = _MAX_SEQ_LEN // 4
    compressed_slots = torch.arange(
        _COMPRESSED_ENTRIES_PER_BLOCK,
        _COMPRESSED_ENTRIES_PER_BLOCK + compressed_count,
        dtype=torch.int64,
        device=device,
    )
    swa_slots = torch.arange(
        _SWA_ENTRIES_PER_BLOCK,
        2 * _SWA_ENTRIES_PER_BLOCK,
        dtype=torch.int64,
        device=device,
    )
    main = torch.randn(
        compressed_count,
        HEAD_DIM,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.05)
    indexer = torch.randn(
        compressed_count,
        INDEX_HEAD_DIM,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.05)
    swa = torch.randn(
        _SWA_ENTRIES_PER_BLOCK,
        HEAD_DIM,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.05)
    quantize_and_insert_k_cache(
        main,
        pools.packed_view(CSA_KV, _COMPRESSED_ENTRIES_PER_BLOCK, _KV_ENTRY_BYTES),
        compressed_slots,
    )
    quantize_indexer_k(
        indexer,
        compressed_slots,
        pools.packed_view(
            INDEXER_KV, _COMPRESSED_ENTRIES_PER_BLOCK, _INDEXER_ENTRY_BYTES
        ),
    )
    quantize_and_insert_k_cache(
        swa,
        pools.packed_view(SWA_KV, _SWA_ENTRIES_PER_BLOCK, _KV_ENTRY_BYTES),
        swa_slots,
    )


class MegaCSARTPEagerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        capability = torch.cuda.get_device_capability()
        if capability not in ((10, 0), (10, 3)):
            raise unittest.SkipTest(
                f"Mega CSA requires sm_100a/sm_103a, got {capability}"
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
            rope_head_dim=ROPE_DIM,
            o_lora_rank=1024,
            o_groups=8,
            window_size=128,
            compress_ratio=4,
            compress_rope_theta=160000.0,
            rope_theta=10000.0,
            rope_factor=16.0,
            beta_fast=32,
            beta_slow=1,
            original_seq_len=65536,
            max_batch_size=128,
            max_seq_len=_MAX_SEQ_LEN,
            index_n_heads=INDEX_HEADS,
            index_head_dim=INDEX_HEAD_DIM,
            index_topk=512,
            norm_eps=1.0e-6,
            layer_weights=weights,
            tp_size=1,
            tp_rank=0,
        )
        attention.reset_rope_cache(cls.device)
        cls.block = _AttentionBlock(attention, weights)
        cls.runtime = MegaCSARuntime()
        cls.adapter = MegaCSAAdapter(cls.block, weights, cls.runtime)
        cls.mega_pools = _make_pools(cls.device, batch_size=1)
        cls.reference_pools = _make_pools(cls.device, batch_size=1)

    def _metadata(self, position: int, pools: _Pools):
        batch_size = int(pools.block_tables[CSA_KV].shape[0])
        return build_decode_metadata_fp8(
            attention_inputs=torch.full(
                (batch_size,), position, dtype=torch.int32, device=self.device
            ),
            q_len=1,
            window_size=128,
            head_dim=HEAD_DIM,
            max_seq_len=_MAX_SEQ_LEN,
            compress_ratios=[4],
            index_topk=512,
            device=self.device,
            paged_block_tables=pools.block_tables,
            paged_pool_entries_per_block=pools.entries_per_block,
            paged_pool_tokens_per_block=pools.tokens_per_block,
        )

    @torch.inference_mode()
    def _forward_mega(
        self, hidden: torch.Tensor, metadata: object, pools: _Pools
    ) -> torch.Tensor:
        output = self.adapter.forward_attention_sublayer(
            self.block,
            hidden,
            metadata,
            kv_cache=pools.kv_cache,
        )
        return output

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
        output = self.block.attn_hc.post(attn_out, residual, post, comb)
        return output

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

    @torch.inference_mode()
    def _run_until_boundary(self, hidden_seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(hidden_seed)
        output = None
        for position in range(4):
            hidden = torch.randn(
                (1, 1, HC, DIM),
                generator=generator,
                device=self.device,
                dtype=torch.bfloat16,
            ).mul_(0.05)
            output, _ = self._run_mega_step(position, hidden, self.mega_pools)
        assert output is not None
        return output.clone()

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

        # Warm every JIT path and persistent workspace before capture.
        for position in range(4):
            self._run_mega_step(position, history[position].clone(), self.mega_pools)

        self.mega_pools.reset()
        for position in range(3):
            self._run_mega_step(position, history[position].clone(), self.mega_pools)

        metadata = self._metadata(3, self.mega_pools)
        metadata.is_cuda_graph = True
        self.runtime.begin_decode(metadata)
        self.runtime.mqa_schedule(
            metadata.compressed_lens[4], _COMPRESSED_ENTRIES_PER_BLOCK
        )
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

    def test_eager_compression_boundary_and_slot_reuse(self) -> None:
        first = self._run_until_boundary(hidden_seed=11)
        self.assertTrue(self.mega_pools.tensors[CSA_KV][1].any().item())
        self.assertTrue(self.mega_pools.tensors[INDEXER_KV][1].any().item())
        self.assertTrue(self.mega_pools.tensors[SWA_KV][1].any().item())

        self.mega_pools.reset()
        second = self._run_until_boundary(hidden_seed=29)
        self.assertTrue(torch.isfinite(second).all().item())
        self.assertFalse(torch.equal(first, second))

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
        print(f"Mega/reference attention sublayer calc_diff: {output_diff:.6e}")
        self.assertLess(output_diff, 1.0e-3)

        torch.testing.assert_close(
            mega_metadata.topk_buffer_compressed,
            reference_metadata.topk_buffer_compressed,
            rtol=0.0,
            atol=0.0,
        )

        main_slot = torch.tensor(
            [_COMPRESSED_ENTRIES_PER_BLOCK], dtype=torch.int64, device=self.device
        )
        swa_slot = torch.tensor(
            [_SWA_ENTRIES_PER_BLOCK + 3], dtype=torch.int64, device=self.device
        )
        mega_main = dequantize_slots_to_bf16(
            self.mega_pools.packed_view(
                CSA_KV, _COMPRESSED_ENTRIES_PER_BLOCK, _KV_ENTRY_BYTES
            ),
            main_slot,
        )
        reference_main = dequantize_slots_to_bf16(
            self.reference_pools.packed_view(
                CSA_KV, _COMPRESSED_ENTRIES_PER_BLOCK, _KV_ENTRY_BYTES
            ),
            main_slot,
        )
        mega_swa = dequantize_slots_to_bf16(
            self.mega_pools.packed_view(
                SWA_KV, _SWA_ENTRIES_PER_BLOCK, _KV_ENTRY_BYTES
            ),
            swa_slot,
        )
        reference_swa = dequantize_slots_to_bf16(
            self.reference_pools.packed_view(
                SWA_KV, _SWA_ENTRIES_PER_BLOCK, _KV_ENTRY_BYTES
            ),
            swa_slot,
        )
        mega_indexer = dequantize_indexer_k(
            self.mega_pools.packed_view(
                INDEXER_KV, _COMPRESSED_ENTRIES_PER_BLOCK, _INDEXER_ENTRY_BYTES
            ),
            main_slot,
        )
        reference_indexer = dequantize_indexer_k(
            self.reference_pools.packed_view(
                INDEXER_KV, _COMPRESSED_ENTRIES_PER_BLOCK, _INDEXER_ENTRY_BYTES
            ),
            main_slot,
        )
        for name, mega_value, reference_value in (
            ("CSA KV", mega_main, reference_main),
            ("Indexer KV", mega_indexer, reference_indexer),
            ("SWA KV", mega_swa, reference_swa),
        ):
            value_diff = calc_diff(mega_value.float(), reference_value.float())
            print(f"Mega/reference {name} calc_diff: {value_diff:.6e}")
            self.assertLess(value_diff, 1.0e-3, msg=name)

        for name, attn_type in (
            ("CSA state", CSA_STATE),
            ("Indexer state", INDEXER_STATE),
        ):
            mega_state = self.mega_pools.tensors[attn_type]
            reference_state = self.reference_pools.tensors[attn_type]
            mega_finite = torch.isfinite(mega_state)
            reference_finite = torch.isfinite(reference_state)
            torch.testing.assert_close(
                mega_finite, reference_finite, rtol=0.0, atol=0.0
            )
            value_diff = calc_diff(
                mega_state[mega_finite], reference_state[reference_finite]
            )
            print(f"Mega/reference {name} calc_diff: {value_diff:.6e}")
            self.assertLess(value_diff, 1.0e-4, msg=name)

    def test_matches_original_rtp_at_nontrivial_topk_context(self) -> None:
        self.mega_pools.reset()
        self.reference_pools.reset()
        _fill_random_context(self.mega_pools, self.device, seed=31415)
        _fill_random_context(self.reference_pools, self.device, seed=31415)
        generator = torch.Generator(device=self.device).manual_seed(27182)
        hidden = torch.randn(
            (1, 1, HC, DIM),
            generator=generator,
            device=self.device,
            dtype=torch.bfloat16,
        ).mul_(0.05)

        reference_output, reference_metadata = self._run_reference_step(
            _MAX_SEQ_LEN - 1, hidden.clone(), self.reference_pools
        )
        mega_output, mega_metadata = self._run_mega_step(
            _MAX_SEQ_LEN - 1, hidden.clone(), self.mega_pools
        )
        output_diff = calc_diff(mega_output.float(), reference_output.float())
        mega_topk = mega_metadata.topk_buffer_compressed.flatten()
        reference_topk = reference_metadata.topk_buffer_compressed.flatten()
        overlap = len(set(mega_topk.tolist()) & set(reference_topk.tolist()))
        print(
            "Mega/reference long-context attention sublayer "
            f"calc_diff: {output_diff:.6e}; TopK overlap: {overlap}/512"
        )
        self.assertLess(output_diff, 1.0e-3)
        self.assertGreaterEqual(overlap, 510)

    @unittest.skipUnless(
        _RUN_PERF,
        "set DSV4_MEGA_RUN_PERF=1 for the single-card performance comparison",
    )
    def test_performance_against_original_rtp_attention_sublayer(self) -> None:
        for batch_size in (1, 8, 16):
            reference_pools = _make_pools(self.device, batch_size)
            mega_pools = _make_pools(self.device, batch_size)
            generator = torch.Generator(device=self.device).manual_seed(
                9000 + batch_size
            )
            hidden = torch.randn(
                (batch_size, 1, HC, DIM),
                generator=generator,
                device=self.device,
                dtype=torch.bfloat16,
            ).mul_(0.05)
            reference_metadata = self._metadata(2047, reference_pools)
            mega_metadata = self._metadata(2047, mega_pools)
            reference_hidden = hidden.clone()
            mega_hidden = hidden.clone()

            reference_fn = lambda: self._forward_reference(
                reference_hidden, reference_metadata, reference_pools
            )
            self.runtime.begin_decode(mega_metadata)
            mega_fn = lambda: self._forward_mega(mega_hidden, mega_metadata, mega_pools)

            # The helper uses median CUDA-event time after all lazy kernels and
            # workspaces are warm. Metadata construction and begin_decode are
            # per-step framework work, shared across layers, so neither belongs
            # in this one-layer attention-sublayer comparison.
            reference_eager_us = _bench_cuda_event(reference_fn)
            mega_eager_us = _bench_cuda_event(mega_fn)

            reference_metadata.is_cuda_graph = True
            reference_fn()
            torch.cuda.synchronize(self.device)
            reference_graph_input = hidden.clone()
            reference_graph_work = torch.empty_like(reference_graph_input)
            reference_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(reference_graph):
                reference_graph_work.copy_(reference_graph_input)
                reference_graph_output = self._forward_reference(
                    reference_graph_work, reference_metadata, reference_pools
                )

            mega_metadata.is_cuda_graph = True
            self.runtime.begin_decode(mega_metadata)
            self.runtime.mqa_schedule(
                mega_metadata.compressed_lens[4], _COMPRESSED_ENTRIES_PER_BLOCK
            )
            mega_fn()
            torch.cuda.synchronize(self.device)
            self.runtime.begin_decode(mega_metadata)
            self.runtime.mqa_schedule(
                mega_metadata.compressed_lens[4], _COMPRESSED_ENTRIES_PER_BLOCK
            )
            mega_graph = torch.cuda.CUDAGraph()
            mega_graph_input = hidden.clone()
            mega_graph_work = torch.empty_like(mega_graph_input)
            with torch.cuda.graph(mega_graph):
                mega_graph_work.copy_(mega_graph_input)
                mega_graph_output = self._forward_mega(
                    mega_graph_work, mega_metadata, mega_pools
                )

            reference_graph_us = _bench_cuda_event(
                reference_graph.replay, warmup=5, iterations=50
            )
            mega_graph_us = _bench_cuda_event(
                mega_graph.replay, warmup=5, iterations=50
            )
            reference_graph.replay()
            mega_graph.replay()
            torch.cuda.synchronize(self.device)
            self.assertTrue(torch.isfinite(reference_graph_output).all().item())
            self.assertTrue(torch.isfinite(mega_graph_output).all().item())

            row = {
                "batch": batch_size,
                "reference_eager_us": reference_eager_us,
                "mega_eager_us": mega_eager_us,
                "reference_graph_us": reference_graph_us,
                "mega_graph_us": mega_graph_us,
            }
            print(
                "Mega/reference performance "
                f"B={batch_size}: eager {mega_eager_us:.2f}/{reference_eager_us:.2f} us "
                f"({(mega_eager_us / reference_eager_us - 1.0) * 100.0:+.1f}%), "
                f"graph {mega_graph_us:.2f}/{reference_graph_us:.2f} us "
                f"({(mega_graph_us / reference_graph_us - 1.0) * 100.0:+.1f}%)"
            )

            self.assertLessEqual(
                mega_eager_us,
                reference_eager_us * 1.05,
                msg=f"eager regression at B={batch_size}: {row}",
            )
            self.assertLessEqual(
                mega_graph_us,
                reference_graph_us * 1.05,
                msg=f"CUDA Graph regression at B={batch_size}: {row}",
            )

            del reference_graph, mega_graph
            del reference_pools, mega_pools, reference_metadata, mega_metadata
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
