"""Host-cache byte preservation and production SparseMLA integration tests."""

import ctypes
import importlib.util
import mmap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

# This component has only torch/triton dependencies. Loading its source directly
# also lets the tests run before rebuilding the engine's native bindings.
_MODULE_PATH = Path(__file__).resolve().parents[1] / "pinned_mla_cache.py"
_SPEC = importlib.util.spec_from_file_location("glm53_pinned_mla_cache", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PinnedMlaWorkingSet = _MODULE.PinnedMlaWorkingSet


class Glm53PinnedMlaCacheTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "requires a CUDA GPU")
        torch.manual_seed(123)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.page = 128
        self.hbm_tokens = 256
        self.host_tokens = 8192
        self.capacity = 2176  # 2051 semantic selections rounded to whole pages.

    def make_cache(self, width=528, layers=2, fetch_rows=0):
        # GLM53 FP8 NoPE: 512 quantized bytes + four FP32 scales.
        host = [
            torch.randint(
                0,
                256,
                (self.host_tokens // self.page, self.page, width),
                dtype=torch.uint8,
                pin_memory=True,
            )
            for _ in range(layers)
        ]
        resident = [
            torch.randint(
                0,
                256,
                ((self.hbm_tokens + self.capacity) // self.page, self.page, width),
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(layers)
        ]
        expected = [
            torch.cat((r.flatten(0, 1)[: self.hbm_tokens].cpu(), h.flatten(0, 1)))
            for h, r in zip(host, resident)
        ]
        generations = torch.zeros(
            (self.hbm_tokens + self.host_tokens) // 256,
            dtype=torch.int64,
            pin_memory=True,
        )
        with patch.dict("os.environ", {"RTP_LLM_DSA_MLA_FETCH_ROWS": str(fetch_rows)}):
            cache = PinnedMlaWorkingSet(
                host,
                self.capacity,
                self.page,
                self.device,
                block_generations=generations,
                allocator_block_size=256,
                hbm_tokens=self.hbm_tokens,
                hbm_cache=resident,
            )
        return cache, expected

    def check_rows(self, cache, expected, ids, physical, *, joined=False):
        valid = ids.cpu().flatten() >= 0
        for layer, ref in enumerate(expected):
            resident = (
                cache.resident[layer] if joined else cache.layer_cache(layer)
            ).flatten(0, 1)
            result = resident[physical.flatten().clamp_min(0).long()].cpu()
            torch.testing.assert_close(
                result[valid], ref[ids.cpu().flatten()[valid].long()], rtol=0, atol=0
            )
        self.assertTrue(torch.all(physical[ids < 0] == -1))

    def test_glm53_2051_selections_eviction_duplicates_and_padding(self):
        for width, rows in ((528, 0), (528, 8), (1024, 0)):
            with self.subTest(width=width, fetch_rows=rows):
                cache, expected = self.make_cache(width=width, fetch_rows=rows)
                for offset in (0, 2048, 4096, 0):
                    ids = (
                        torch.arange(2051, device=self.device, dtype=torch.int32)
                        + self.hbm_tokens
                        + offset
                    ).view(1, -1)
                    ids[0, :3] = torch.tensor([-1, 1, 1], device=self.device)
                    ids[0, 3] = ids[0, 4]
                    self.check_rows(cache, expected, ids, cache.begin(ids))

    def test_current_token_write_through_and_hot_refresh(self):
        cache, expected = self.make_cache()
        ids = torch.tensor([[2, 257, 258, -1]], dtype=torch.int32, device=self.device)
        physical = cache.begin(ids)
        for layer in range(2):
            values = torch.full(
                (3, cache.width), 40 + layer, dtype=torch.uint8, device=self.device
            )
            slots = ids[0, :3].contiguous()
            cache.write(layer, slots, values)
            expected[layer][slots.cpu().long()] = values.cpu()
        self.check_rows(cache, expected, ids, physical)
        torch.cuda.synchronize()
        for layer in range(2):
            torch.testing.assert_close(
                cache.backing[layer].flatten(0, 1)[1:3],
                expected[layer][257:259],
                rtol=0,
                atol=0,
            )
        self.check_rows(cache, expected, ids, cache.begin(ids))

    def test_reallocated_physical_block_refreshes_all_layers(self):
        cache, expected = self.make_cache()
        ids = torch.tensor([[300, 301, 2]], dtype=torch.int32, device=self.device)
        self.check_rows(cache, expected, ids, cache.begin(ids))
        torch.cuda.synchronize()  # Simulate completed external producer write.
        for layer in range(2):
            cache.backing[layer].flatten(0, 1)[
                300 - self.hbm_tokens : 302 - self.hbm_tokens
            ].fill_(17 + layer)
            expected[layer][300:302].fill_(17 + layer)
        cache.generations[300 // 256] += 1
        self.check_rows(cache, expected, ids, cache.begin(ids))

    def test_graph_replay_uses_new_selection(self):
        cache, expected = self.make_cache(layers=1)
        ids = torch.tensor([[128, 129, 2, -1]], dtype=torch.int32, device=self.device)
        self.check_rows(cache, expected, ids, cache.begin(ids))
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            physical = cache.begin(ids)
            cache.layer_cache(0)
        for selection in ([5000, 5001, 5, -1], [7000, 7000, 0, -1], [128, 129, 2, -1]):
            ids.copy_(torch.tensor([selection], dtype=torch.int32, device=self.device))
            graph.replay()
            # The graph already joins the transfer stream. Its internal event
            # must not be waited on again from outside the captured graph.
            self.check_rows(cache, expected, ids, physical, joined=True)

    def test_cuda_registered_arena_read_and_write(self):
        runtime = ctypes.CDLL("libcudart.so.13")
        runtime.cudaHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        runtime.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        width = 528
        size = self.host_tokens * width
        with mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS) as arena:
            host = torch.frombuffer(arena, dtype=torch.uint8).reshape(
                -1, self.page, width
            )
            host.fill_(29)
            self.assertEqual(runtime.cudaHostRegister(host.data_ptr(), size, 0), 0)
            cache = None
            try:
                self.assertTrue(host.is_pinned())
                cache = PinnedMlaWorkingSet(
                    [host], self.capacity, self.page, self.device
                )
                expected = [host.flatten(0, 1).clone()]
                ids = torch.tensor(
                    [[4000, 4001, -1]], dtype=torch.int32, device=self.device
                )
                physical = cache.begin(ids)
                self.check_rows(cache, expected, ids, physical)
                values = torch.full(
                    (2, width), 71, dtype=torch.uint8, device=self.device
                )
                cache.write(0, ids[0, :2], values)
                expected[0][4000:4002] = 71
                self.check_rows(cache, expected, ids, physical)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    host.flatten(0, 1), expected[0], rtol=0, atol=0
                )
            finally:
                torch.cuda.synchronize()
                self.assertEqual(runtime.cudaHostUnregister(host.data_ptr()), 0)
                del cache, host

    def test_rejects_insufficient_working_set(self):
        cache, _ = self.make_cache(layers=1)
        ids = torch.zeros(self.capacity + 1, device=self.device, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "flattened topk"):
            cache.begin(ids)

    def test_bf16_flashmla_attention_matches_hbm(self):
        from flash_mla import flash_mla_sparse_fwd

        baseline = torch.randn(
            (self.hbm_tokens + self.host_tokens, 1, 512),
            dtype=torch.bfloat16,
            device=self.device,
        )
        host = (
            baseline[self.hbm_tokens :].reshape(-1, self.page, 512).cpu().pin_memory()
        )
        resident = torch.empty(
            ((self.hbm_tokens + self.capacity) // self.page, self.page, 512),
            dtype=torch.bfloat16,
            device=self.device,
        )
        resident.flatten(0, 1)[: self.hbm_tokens].copy_(baseline[: self.hbm_tokens, 0])
        cache = PinnedMlaWorkingSet(
            [host],
            self.capacity,
            self.page,
            self.device,
            allocator_block_size=256,
            hbm_tokens=self.hbm_tokens,
            hbm_cache=[resident],
        )
        q = torch.randn((1, 64, 512), dtype=torch.bfloat16, device=self.device)
        for start in (256, 4000, 256):
            ids = torch.arange(
                start, start + 2051, dtype=torch.int32, device=self.device
            ).view(1, 1, -1)
            ids[..., :3] = torch.tensor(
                [2, 2, -1], dtype=torch.int32, device=self.device
            )
            # FlashMLA needs a multiple of 128; these pads carry no tokens.
            ids = torch.nn.functional.pad(ids, (0, 2176 - 2051), value=-1)
            expected = flash_mla_sparse_fwd(q, baseline, ids, 512**-0.5, d_v=512)[0]
            physical = cache.begin(ids)
            kv = cache.layer_cache(0).reshape(-1, 1, 512)
            actual = flash_mla_sparse_fwd(q, kv, physical, 512**-0.5, d_v=512)[0]
            self.assertTrue(torch.isfinite(expected).all())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fp8_production_write_gather_attention_matches_hbm(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaFp8Op,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
            MlaKVCacheWriteOp,
        )
        from rtp_llm.ops import KvCacheDataType
        from rtp_llm.ops.compute_ops import LayerKVCache

        logical_tokens = self.hbm_tokens + self.host_tokens
        full_cache = LayerKVCache()
        full_cache.kv_cache_base = torch.empty(
            (logical_tokens // self.page, self.page, 528),
            dtype=torch.uint8,
            device=self.device,
        )
        values = torch.randn(
            (logical_tokens, 512), dtype=torch.bfloat16, device=self.device
        )
        writer = MlaKVCacheWriteOp(KvCacheDataType.FP8)
        writer.forward(
            values,
            values.new_empty(logical_tokens, 0),
            full_cache,
            SimpleNamespace(
                slot_mapping=torch.arange(
                    logical_tokens, dtype=torch.int64, device=self.device
                )
            ),
        )
        host = (
            full_cache.kv_cache_base[self.hbm_tokens // self.page :].cpu().pin_memory()
        )
        resident = torch.empty(
            ((self.hbm_tokens + self.capacity) // self.page, self.page, 528),
            dtype=torch.uint8,
            device=self.device,
        )
        resident[: self.hbm_tokens // self.page].copy_(
            full_cache.kv_cache_base[: self.hbm_tokens // self.page]
        )
        cache = PinnedMlaWorkingSet(
            [host],
            self.capacity,
            self.page,
            self.device,
            allocator_block_size=256,
            hbm_tokens=self.hbm_tokens,
            hbm_cache=[resident],
        )
        op = SparseMlaFp8Op(
            num_heads=64,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            qk_nope_head_dim=256,
            page_size=self.page,
            softmax_extra_scale=1.0,
            top_k=2051,
            indexer_top_k=512,
            indexer_group_size=4,
        )
        q = torch.randn((1, 64, 512), dtype=torch.bfloat16, device=self.device)
        for start in (256, 4000, 256):
            ids = torch.arange(
                start, start + 2051, dtype=torch.int32, device=self.device
            ).view(1, 1, -1)
            ids[..., :3] = torch.tensor(
                [128, 128, -1], dtype=torch.int32, device=self.device
            )
            physical = cache.begin(ids)
            # Refresh a prefetched current token using the same production
            # quantizer as the all-HBM path, including its four FP32 scales.
            new_value = torch.randn((1, 512), dtype=torch.bfloat16, device=self.device)
            logical_slot = ids[0, 0, 4:5].long()
            writer.forward(
                new_value,
                new_value.new_empty(1, 0),
                full_cache,
                SimpleNamespace(slot_mapping=logical_slot),
            )
            scratch = LayerKVCache()
            scratch.kv_cache_base = torch.empty(
                (1, 1, 528), dtype=torch.uint8, device=self.device
            )
            writer.forward(
                new_value,
                new_value.new_empty(1, 0),
                scratch,
                SimpleNamespace(
                    slot_mapping=torch.zeros(1, dtype=torch.int64, device=self.device)
                ),
            )
            cache.write(0, logical_slot, scratch.kv_cache_base.flatten(0, 1))
            expected = op.forward(
                q, full_cache.kv_cache_base, ids, physical_indices=ids
            )
            actual = op.forward(q, cache.layer_cache(0), ids, physical_indices=physical)
            self.assertTrue(torch.isfinite(expected).all())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hybrid_builder_never_interprets_kda_state_as_tokens(self):
        source, expected = self.make_cache(layers=1)
        hbm = source.resident[0]
        generations = source.generations

        # Three KDA layers followed by one MLA layer, as in GLM53. A KDA
        # get_layer_cache call is an error, even though those states use bytes.
        def get_layer_cache(layer):
            self.assertEqual(layer, 3)
            return SimpleNamespace(kv_cache_base=source.backing[0])

        metadata = [SimpleNamespace(hbm_cache=None) for _ in range(3)]
        metadata.append(
            SimpleNamespace(
                hbm_cache=hbm, block_generations=generations, hbm_tokens=self.hbm_tokens
            )
        )
        kv_cache = SimpleNamespace(
            mla_host_cache_by_layer=metadata,
            get_layer_cache=get_layer_cache,
            seq_size_per_block=256,
            kernel_seq_size_per_block=self.page,
        )
        groups = _MODULE.build_working_sets(kv_cache)
        self.assertEqual(list(groups), [3])
        working, local_layer = groups[3]
        self.assertEqual(local_layer, 0)
        self.assertEqual(working.resident[0].data_ptr(), hbm.data_ptr())
        ids = torch.tensor([[300, 2, -1]], dtype=torch.int32, device=self.device)
        self.check_rows(working, expected, ids, working.begin(ids))

    def test_full_sparse_mla_decode_and_chunked_prefill(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaImpl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
            MlaKVCacheWriteOp,
        )
        from rtp_llm.ops import AttentionConfigs, KvCacheDataType
        from rtp_llm.ops.compute_ops import (
            LayerKVCache,
            PyAttentionInputs,
            get_typemeta,
        )
        from rtp_llm.utils.model_weight import W

        config = AttentionConfigs()
        for name, value in dict(
            head_num=64,
            kv_lora_rank=512,
            rope_head_dim=0,
            nope_head_dim=256,
            kernel_tokens_per_block=128,
            softmax_extra_scale=1.0,
            use_mla=True,
            is_sparse=True,
            indexer_topk=512,
            indexer_compress_ratio=4,
            sparse_attention_topk=2051,
        ).items():
            setattr(config, name, value)
        config.kv_cache_dtype = KvCacheDataType.FP8
        weights = [
            {
                W.mla_kc: torch.randn(
                    64, 256, 512, dtype=torch.bfloat16, device=self.device
                )
                * 0.01,
                W.mla_vc: torch.randn(
                    64, 512, 256, dtype=torch.bfloat16, device=self.device
                )
                * 0.01,
            }
        ]
        logical_tokens = self.hbm_tokens + self.host_tokens
        full = LayerKVCache()
        full.kv_cache_base = torch.empty(
            logical_tokens // 128, 128, 528, dtype=torch.uint8, device=self.device
        )
        values = torch.randn(
            logical_tokens, 512, dtype=torch.bfloat16, device=self.device
        )
        MlaKVCacheWriteOp(KvCacheDataType.FP8).forward(
            values,
            values.new_empty(logical_tokens, 0),
            full,
            SimpleNamespace(
                slot_mapping=torch.arange(
                    logical_tokens, dtype=torch.int64, device=self.device
                )
            ),
        )
        host = LayerKVCache()
        host.kv_cache_base = full.kv_cache_base[2:].cpu().pin_memory()
        resident = torch.empty(
            (self.hbm_tokens + self.capacity) // 128,
            128,
            528,
            dtype=torch.uint8,
            device=self.device,
        )
        resident[:2].copy_(full.kv_cache_base[:2])
        working = PinnedMlaWorkingSet(
            [host.kv_cache_base],
            self.capacity,
            128,
            self.device,
            allocator_block_size=256,
            hbm_tokens=self.hbm_tokens,
            hbm_cache=[resident],
        )

        for rows in (1, 3):
            with self.subTest(rows=rows):
                # Noncontiguous physical pages exercise request-local grouped
                # indices -> raw causal tokens -> allocator addresses.
                a = PyAttentionInputs()
                a.is_prefill = rows > 1
                a.is_cuda_graph = False
                a.input_lengths = torch.tensor(
                    [rows], dtype=torch.int32, device=self.device
                )
                a.prefix_lengths = torch.tensor(
                    [6000] if rows > 1 else [], dtype=torch.int32, device=self.device
                )
                a.sequence_lengths = torch.tensor(
                    [] if rows > 1 else [6000], dtype=torch.int32, device=self.device
                )
                a.sequence_lengths_plus_1_d = a.sequence_lengths + 1
                a.cu_seqlens = torch.tensor(
                    [0, rows], dtype=torch.int32, device=self.device
                )
                a.cu_kv_seqlens = torch.tensor(
                    [0, 6000 + rows], dtype=torch.int32, device=self.device
                )
                a.context_total_kv_length = 6000 + rows if rows > 1 else 0
                a.total_tokens = rows
                a.padding_offset = torch.zeros(
                    rows, dtype=torch.int32, device=self.device
                )
                table = torch.arange(1, 49, dtype=torch.int32).flip(0).reshape(1, -1)
                a.kv_cache_block_id_host = a.kv_cache_kernel_block_id_host = table
                a.kv_cache_block_id_device = a.kv_cache_kernel_block_id_device = (
                    table.to(self.device)
                )
                a.dtype = get_typemeta(torch.empty(0, dtype=torch.bfloat16))
                baseline = SparseMlaImpl(config, a, weights, values.new_empty(8192, 0))
                tiered = SparseMlaImpl(config, a, weights, values.new_empty(8192, 0))
                tiered.pinned_mla_groups = {0: (working, 0)}
                q = torch.randn(rows, 64, 256, dtype=torch.bfloat16, device=self.device)
                kv = torch.randn(rows, 512, dtype=torch.bfloat16, device=self.device)
                k_pe = kv.new_empty(rows, 0)
                for offset in (0, 700, 0):
                    topk = torch.arange(
                        offset, offset + 512, dtype=torch.int32, device=self.device
                    ).repeat(rows, 1)
                    expected = baseline.forward(q, kv, k_pe, full, 0, topk)
                    if rows == 1:
                        tiered.prefetch_kv(0, topk)
                    actual = tiered.forward(
                        q, kv, k_pe, host, 0, topk, kv_prefetched=rows == 1
                    )
                    self.assertTrue(torch.isfinite(expected).all())
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

                # Replay changes the selection and current KV at fixed input
                # addresses. This covers production scratch writes and stream
                # joins as well as the component-only graph test above.
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = tiered.forward(q, kv, k_pe, host, 0, topk)
                for offset in (700, 0):
                    topk.copy_(
                        torch.arange(
                            offset, offset + 512, dtype=torch.int32, device=self.device
                        ).repeat(rows, 1)
                    )
                    kv.normal_()
                    expected = baseline.forward(q, kv, k_pe, full, 0, topk)
                    graph.replay()
                    torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)

    def test_native_metadata_preserves_physical_and_kernel_views(self):
        from rtp_llm.ops.compute_ops import CacheGroupType, KVCache, MlaHostCacheInfo

        source, expected = self.make_cache(layers=1)
        cache = KVCache()
        cache.seq_size_per_block = 256
        cache.kernel_seq_size_per_block = 128
        cache.use_mla = True
        cache.kv_lora_rank = 512
        cache.layer_group_types = [CacheGroupType.LINEAR, CacheGroupType.FULL]
        kda = torch.zeros((4, 3072), dtype=torch.uint8, device=self.device)
        cache.kv_cache_base_by_layer = [kda, source.backing[0].reshape(-1, 256, 528)]
        info = MlaHostCacheInfo()
        info.hbm_cache = source.resident[0]
        info.hbm_tokens = self.hbm_tokens
        info.block_generations = source.generations
        cache.mla_host_cache_by_layer = [MlaHostCacheInfo(), info]
        self.assertEqual(
            cache.get_layer_cache(0).kv_cache_base.data_ptr(), kda.data_ptr()
        )
        layer = cache.get_layer_cache(1)
        self.assertEqual(tuple(layer.kv_cache_base.shape), (64, 128, 528))
        self.assertEqual(
            layer.mla_host_cache.hbm_cache.data_ptr(), info.hbm_cache.data_ptr()
        )
        groups = _MODULE.build_working_sets(cache)
        self.assertEqual(list(groups), [1])
        working, _ = groups[1]
        ids = torch.tensor([[2, 256, 4096, -1]], dtype=torch.int32, device=self.device)
        self.check_rows(working, expected, ids, working.begin(ids))


if __name__ == "__main__":
    unittest.main()
