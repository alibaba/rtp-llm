"""GPU regressions for tiered MLA storage and graph/eager transitions."""

import ctypes
import mmap
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.pinned_mla_cache import (
    PinnedMlaWorkingSet,
)


@triton.jit
def _check_selected_bytes(
    Host,
    Device,
    Ids,
    Slots,
    Errors,
    N: tl.constexpr,
    CAPACITY: tl.constexpr,
    WIDTH: tl.constexpr,
    B: tl.constexpr,
    HBM_TOKENS: tl.constexpr = 0,
):
    for i in range(tl.program_id(0), N, tl.num_programs(0)):
        token = tl.load(Ids + i).to(tl.int64)
        slot = tl.load(Slots + i).to(tl.int64)
        bad = (token < 0) & (slot != -1)
        if (token >= 0) & (token < HBM_TOKENS):
            bad = slot != token
        if token >= HBM_TOKENS:
            valid = (slot >= HBM_TOKENS) & (slot < CAPACITY + HBM_TOKENS)
            x = tl.arange(0, B)
            expected = tl.load(
                Host + (token - HBM_TOKENS) * WIDTH + x, x < WIDTH, other=0
            )
            actual = tl.load(Device + slot * WIDTH + x, valid & (x < WIDTH), other=0)
            bad = ~valid | (tl.sum((actual != expected).to(tl.int32), 0) > 0)
        if bad:
            tl.atomic_add(Errors, 1, sem="relaxed")


class PinnedMlaCacheTest(unittest.TestCase):
    def test_hybrid_metadata_exposes_per_layer_allocator_generations(self):
        from rtp_llm.ops.compute_ops import KVCache

        self.assertEqual(KVCache().block_generations_by_layer, [])

    def test_glm53_bf16_mixed_tiers_prefill_gather_and_recycle(self):
        torch.manual_seed(81)
        page, hbm_blocks, host_blocks, width = 128, 3, 37, 512
        host = torch.randn(
            (host_blocks, page, width), dtype=torch.bfloat16
        ).pin_memory()
        hbm = torch.randn(
            (hbm_blocks + 17, page, width), dtype=torch.bfloat16, device="cuda"
        )
        generations = torch.zeros(
            hbm_blocks + host_blocks, dtype=torch.int64, pin_memory=True
        )
        cache = PinnedMlaWorkingSet(
            [host],
            2176,
            page,
            hbm.device,
            block_generations=generations,
            hbm_tokens=hbm_blocks * page,
            hbm_cache=[hbm],
        )
        reference = torch.cat([hbm[:hbm_blocks], host.cuda()]).flatten(0, 1)
        page_ids = torch.tensor(
            [2, 3, 0, hbm_blocks + host_blocks - 1, 3],
            dtype=torch.int32,
            device="cuda",
        )
        expected_pages = reference.reshape(-1, page, width)[page_ids.long()]
        self.assertTrue(torch.equal(cache.gather_pages(0, page_ids), expected_pages))
        ids = torch.randint(
            0, reference.shape[0], (19, 1, 2051), device="cuda", dtype=torch.int32
        )
        ids[:, :, -3:] = -1
        gathered, physical = cache.prefill_cache(0, ids)
        actual = gathered.flatten(0, 1)[physical.long().clamp_min(0)]
        self.assertTrue(torch.equal(actual[ids >= 0], reference[ids[ids >= 0].long()]))
        self.assertTrue(torch.equal(physical < 0, ids < 0))
        # Decode after prefill must still use the bounded persistent working set.
        decode = ids[0].contiguous()
        slots = cache.begin(decode)
        resident = cache.layer_cache(0).flatten(0, 1)
        self.assertTrue(
            torch.equal(
                resident[slots[decode >= 0].long()],
                reference[decode[decode >= 0].long()],
            )
        )
        self.stream.synchronize()
        host[0].fill_(7)
        generations[hbm_blocks] += 1
        recycled = torch.arange(
            hbm_blocks * page, (hbm_blocks + 1) * page, device="cuda", dtype=torch.int32
        )
        slots = cache.begin(recycled)
        resident = cache.layer_cache(0).flatten(0, 1)
        self.assertTrue(torch.all(resident[slots.long()] == 7).item())

    def test_dense_prefill_plan_preserves_logical_write_and_indexer_mapping(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
            MlaFlashInferPrefillImpl,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        inputs = SimpleNamespace(
            prefix_lengths=torch.tensor([256, 0, 128], dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
            input_lengths=torch.tensor([129, 7, 65], dtype=torch.int32),
            kv_cache_kernel_block_id_host=torch.tensor(
                [[7, 3, 10, 4], [2, 0, 0, 0], [3, 8, 0, 0]], dtype=torch.int32
            ),
        )
        logical = rtp_llm_ops.FlashInferMlaAttnParams()
        logical.fill_params(
            inputs.prefix_lengths,
            inputs.sequence_lengths,
            inputs.input_lengths,
            inputs.kv_cache_kernel_block_id_host,
            128,
            False,
        )
        pages = logical.page_indice_d.clone()
        slots = logical.slot_mapping.clone()
        instance = MlaFlashInferPrefillImpl.__new__(MlaFlashInferPrefillImpl)
        instance.attn_inputs = inputs
        instance.fmha_params = instance.rope_params = logical
        instance.seq_size_per_block = 128
        instance.fmha_impl = Mock()
        instance.absorb_fmha = Mock()
        instance._prepare_pinned_prefill()
        compact = instance._pinned_prefill_params
        self.assertEqual(pages.tolist(), [7, 3, 10, 4, 2, 3, 8])
        self.assertEqual(compact.page_indice_d.tolist(), list(range(7)))
        reconstructed_slots = (
            pages[compact.slot_mapping.long() // 128].long() * 128
            + compact.slot_mapping % 128
        )
        self.assertTrue(torch.equal(reconstructed_slots, slots))
        self.assertTrue(torch.equal(logical.slot_mapping, slots))
        self.assertTrue(torch.equal(logical.page_indice_d, pages))
        self.assertTrue(
            torch.equal(
                pages[compact.reuse_cache_page_indice_d.long()],
                logical.reuse_cache_page_indice_d,
            )
        )
        instance._prepare_pinned_prefill()
        instance.fmha_impl.plan.assert_called_once_with(compact)
        instance.absorb_fmha.plan.assert_called_once_with(compact)

    def test_glm53_prefetch_checks_expanded_not_compressed_topk(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaImpl,
        )

        working = SimpleNamespace(capacity=2176, begin=Mock())
        op = SimpleNamespace(top_k=2051, _convert_topk_indices_to_global=Mock())
        implementation = SimpleNamespace(
            pinned_mla_groups={1: (working, 0)}, fmha_impl=op
        )
        SparseMlaImpl.prefetch_kv(
            implementation, 1, torch.empty((2, 512), device="cuda")
        )
        working.begin.assert_not_called()
        SparseMlaImpl.prefetch_kv(
            implementation, 1, torch.empty((1, 512), device="cuda")
        )
        working.begin.assert_called_once()

    def test_glm53_all_decoder_paths_drain_before_moe(self):
        from rtp_llm.models_py.model_desc.kimi_linear import (
            KimiLinearDecoderLayer,
            KimiLinearMetadata,
        )
        from rtp_llm.ops import HybridAttentionType

        for method in (
            "forward",
            "_forward_hc",
            "_forward_hc_sequence_parallel",
            "forward_hc_deferred",
        ):
            calls = []
            layer = KimiLinearDecoderLayer.__new__(KimiLinearDecoderLayer)
            torch.nn.Module.__init__(layer)
            layer.layer_idx = 1
            layer.layer_type = HybridAttentionType.NONE
            layer.hc_enabled = False
            layer._drain_pinned_mla_before_moe = True
            layer.self_attn = lambda **kwargs: kwargs["hidden_states"]
            layer.input_layernorm = lambda x, *args: (x, x) if args else x
            layer.post_attention_layernorm = layer.input_layernorm
            layer.mlp = lambda x, **kwargs: (calls.append("moe") or x)
            layer.attn_hc = layer.ffn_hc = SimpleNamespace(
                pre=lambda x: (x, x, x),
                post=lambda x, *args: x,
                fused_post_pre=lambda x, *args: (x, x, x, x),
            )
            working = SimpleNamespace(
                started=True, wait_prefetch_complete=lambda: calls.append("drain")
            )
            fmha = SimpleNamespace(pinned_mla_groups={1: (working, 0)})
            meta = KimiLinearMetadata()
            x = torch.zeros((1, 4), device="cuda")
            with patch(
                "rtp_llm.models_py.model_desc.kimi_linear.all_gather_trim",
                side_effect=lambda x, *args: x,
            ), patch(
                "rtp_llm.models_py.model_desc.kimi_linear.reduce_scatter_glm53",
                side_effect=lambda x: x,
            ):
                if method == "forward":
                    layer.forward(x, x, fmha, attn_meta=meta)
                elif method == "forward_hc_deferred":
                    layer.forward_hc_deferred(
                        x, fmha, None, None, meta, None, None, None, None
                    )
                else:
                    if method == "_forward_hc_sequence_parallel":
                        meta.token_shard = SimpleNamespace(logical_tokens=1)
                    getattr(layer, method)(x, fmha, None, None, meta, None)
            self.assertEqual(calls, ["drain", "moe"], method)

    def test_concurrent_admissions_with_sparse_victims(self):
        # Keep the cache full while misses in separate CTAs compete for a few
        # victims at the end of the ring. Include the B64 x 4 x 2048 capacity.
        for capacity in (512, 8192, 524288):
            cache, reference = self.make_cache(capacity=capacity, width=16, layers=2)
            ids = torch.arange(capacity, dtype=torch.int32, device="cuda")
            self.assert_rows(cache, reference, ids, cache.begin(ids))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                physical = cache.begin(ids)
                cache.layer_cache(1)
            victims = min(capacity // 256, 32)
            positions = torch.arange(victims, device="cuda") * 256
            keep = torch.ones(capacity, dtype=torch.bool, device="cuda")
            keep[positions] = False
            # Exercise long-running counters across the int32 boundary too.
            cache.epoch.fill_((1 << 31) - 4)
            timings = []
            for step in range(32):
                old = cache.tags.clone()
                ids[keep] = old[:-victims]
                ids[positions] = (
                    torch.arange(victims, dtype=ids.dtype, device="cuda")
                    + capacity
                    + step * victims
                )
                cache.clock.fill_((1 << 32) * (step % 2))
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
                self.assert_rows(cache, reference, ids, physical, joined=True)
                self.assertTrue(
                    torch.all(cache.mapping[old[-victims:].long()] == -1).item()
                )
                self.assertEqual(physical.unique().numel(), capacity)
            self.assertLess(max(timings), 100.0)

    def setUp(self):
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.stream_context = torch.cuda.stream(self.stream)
        self.stream_context.__enter__()

    def tearDown(self):
        self.stream.synchronize()
        self.stream_context.__exit__(None, None, None)

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device required")

    def make_cache(self, dtype=torch.uint8, width=656, layers=3, capacity=4096):
        torch.manual_seed(412)
        host = [
            torch.randint(0, 128, (capacity // 4, 64, width), dtype=torch.uint8)
            .to(dtype)
            .pin_memory()
            for _ in range(layers)
        ]
        generations = torch.zeros(host[0].shape[0], dtype=torch.int64, pin_memory=True)
        cache = PinnedMlaWorkingSet(
            host, capacity, 64, torch.device("cuda", 0), block_generations=generations
        )
        return cache, [tensor.cuda() for tensor in host]

    def test_prefetch_drain_joins_caller_stream_on_every_graph_replay(self):
        cache, _ = self.make_cache(width=16, layers=4, capacity=4096)
        with self.assertRaisesRegex(RuntimeError, "begin must precede"):
            cache.wait_prefetch_complete()
        producer = torch.cuda.current_stream()
        consumer = torch.cuda.Stream()
        ids = torch.arange(4096, device="cuda", dtype=torch.int32)
        payload = torch.full_like(cache.resident[-1], 17)
        observed = torch.empty_like(payload)

        def run():
            cache.begin(ids)
            # Delay the last producer and publish a replay-dependent payload.
            # Only waiting on ready[0], or waiting on the old TopK stream,
            # must not allow the consumer to read this data early.
            with torch.cuda.stream(cache.transfer_stream):
                torch.cuda._sleep(5_000_000)
                cache.resident[-1].copy_(payload)
                cache.ready[-1].record()
            consumer.wait_stream(producer)
            with torch.cuda.stream(consumer):
                cache.wait_prefetch_complete()
                observed.copy_(cache.resident[-1])
            producer.wait_stream(consumer)

        run()
        producer.synchronize()
        self.assertTrue(torch.equal(observed, payload))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=producer):
            run()
        for value in (31, 47, 59, 71, 83, 97, 109, 127):
            payload.fill_(value)
            graph.replay()
            producer.synchronize()
            self.assertTrue(torch.equal(observed, payload), f"replay payload={value}")

    def test_mtp_graph_clone_reuses_working_set_for_same_kv_arena(self):
        from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel
        from rtp_llm.models_py.model_desc.module_base import GptModelBase

        model = GenericMoeMTPModel.__new__(GenericMoeMTPModel)
        resource_config = SimpleNamespace(enable_layer_micro_batch=0)
        GptModelBase.__init__(
            model,
            SimpleNamespace(num_layers=1, vocab_size=8),
            None,
            None,
            1,
            device_resource_config=resource_config,
        )
        model.moe_config = None
        model.max_generate_batch_size = 1
        model.device_resource_config = resource_config
        model.context_parallel_enabled = False
        model.prefill_mla_cp = False
        model.mla_parallelism = None
        model.multimodal_embedding_injector = None
        for name in (
            "embed_tokens",
            "pre_fc_norm_embedding",
            "pre_fc_norm_hidden",
            "fc",
            "norm",
        ):
            setattr(model, name, torch.nn.Identity())
        model.layers = torch.nn.ModuleList()
        model._mtp_indexer_share_enabled = True
        model._mtp_shared_topk_indices = torch.zeros((1, 4), dtype=torch.int32)
        backing = torch.zeros((1, 64, 656), dtype=torch.uint8)
        kv = SimpleNamespace(
            dsa_mla_resident_tokens=64,
            dsa_mla_hbm_blocks=0,
            block_generations=torch.zeros(1, dtype=torch.int64),
            mla_hbm_cache_by_layer=[torch.empty_like(backing)],
            kv_cache_base_by_layer=[backing],
            kv_scale_base_by_layer=[],
            get_layer_cache=lambda _: SimpleNamespace(kv_cache_base=backing),
        )
        resources = SimpleNamespace(kv_cache=kv)
        groups, other_groups = {0: object()}, {0: object()}
        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.pinned_mla_cache.build_working_sets",
            side_effect=[groups, other_groups],
        ) as build:
            model.initialize(resources)
            clone = model.clone_for_cuda_graph()
            clone.initialize(resources)
            self.assertIs(clone.pinned_mla_groups, model.pinned_mla_groups)
            self.assertEqual(build.call_count, 1)
            kv.mla_hbm_cache_by_layer = [torch.empty_like(backing)]
            clone.initialize(resources)
            self.assertIs(clone.pinned_mla_groups, other_groups)
            self.assertIs(model.pinned_mla_groups, groups)
            self.assertEqual(build.call_count, 2)

    def test_cmp_remains_enabled_and_passes_hybrid_physical_indices(self):
        from rtp_llm.models_py.modules.hybrid.glm5_cmp import (
            Glm5Cmp,
            should_enable_glm5_cmp,
        )

        layer_cmp = SimpleNamespace(
            _disabled_reason=None,
            has_indexer=True,
            ops=object(),
            _unsupported_call_reason=lambda *args: None,
        )
        query = torch.empty((4, 64, 576), device="cuda")
        indices = torch.empty((4, 2048), dtype=torch.int32, device="cuda")
        resident = torch.empty((128, 64, 656), dtype=torch.uint8, device="cuda")
        physical = torch.empty_like(indices)
        working = SimpleNamespace(resident=[resident], physical_indices=physical)
        flashmla = SimpleNamespace(expects_paged_kv=True, forward=Mock())
        implementation = SimpleNamespace(
            fmha_impl=flashmla,
            fmha_params=None,
            weights={},
            pinned_mla_groups={0: (working, 0)},
        )
        kv_cache = SimpleNamespace(get_layer_cache=lambda _: None)
        self.assertTrue(
            should_enable_glm5_cmp(
                [SimpleNamespace(cmp=layer_cmp)], 1, query, implementation, kv_cache
            )
        )
        cmp = object.__new__(Glm5Cmp)
        cmp.layer_idx = 0
        cmp.sparse_mla(
            query, indices, implementation, SimpleNamespace(kv_cache_base=None)
        )
        flashmla.forward.assert_called_once_with(
            query, resident, indices, layer_id=0, physical_indices=physical
        )

    def test_cmp_fused_kv_write_matches_hbm_with_shared_layers_and_graph(self):
        for dtype in (torch.uint8, torch.int8, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                self._check_cmp_fused_kv_write(dtype)

    def _check_cmp_fused_kv_write(self, dtype):
        from rtp_kernel import glm5 as ops

        from rtp_llm.models_py.modules.hybrid.glm5_cmp import Glm5Cmp
        from rtp_llm.utils.model_weight import W

        rows, layers, total, capacity, hbm_tokens = 4, 4, 32768, 8192, 64
        reference = [
            torch.zeros((total // 64, 64, 656), dtype=torch.uint8, device="cuda")
            for _ in range(layers)
        ]
        host = [tensor[1:].cpu().pin_memory().view(dtype) for tensor in reference]
        resident = [
            torch.zeros(
                ((hbm_tokens + capacity) // 64, 64, 656),
                dtype=torch.uint8,
                device="cuda",
            ).view(dtype)
            for _ in reference
        ]
        working = PinnedMlaWorkingSet(
            host,
            capacity,
            64,
            torch.device("cuda", 0),
            hbm_tokens=hbm_tokens,
            hbm_cache=resident,
        )
        ids = torch.arange(capacity, dtype=torch.int32, device="cuda").reshape(
            rows, 2048
        )
        slots = torch.tensor([3, 65, 4097, -1], dtype=torch.int64, device="cuda")
        positions = torch.arange(rows, dtype=torch.int32, device="cuda")
        projected = torch.randn(
            (layers, rows, 2624), dtype=torch.bfloat16, device="cuda"
        )
        q_norm = torch.ones(2048, dtype=torch.bfloat16, device="cuda")
        kv_norm = torch.ones(512, dtype=torch.bfloat16, device="cuda")
        cos_sin = torch.cat(
            (torch.ones((rows, 32)), torch.zeros((rows, 32))), dim=-1
        ).cuda()
        hidden = torch.zeros((rows, 6144), dtype=torch.bfloat16, device="cuda")
        query = torch.zeros((rows, 64, 576), dtype=torch.bfloat16, device="cuda")
        implementation = SimpleNamespace(
            fmha_params=SimpleNamespace(slot_mapping=slots, positions_d=positions),
            weights=[{W.mla_kc: None} for _ in range(layers)],
            _cos_sin_cache=cos_sin,
            pinned_mla_groups={layer: (working, layer) for layer in range(layers)},
            prefetch_kv=lambda layer, topk: working.begin(topk) if layer == 0 else None,
        )
        cmps = []
        for layer in range(layers):
            cmp = object.__new__(Glm5Cmp)
            cmp.layer_idx = layer
            cmp.input_layernorm = SimpleNamespace(weight=q_norm, variance_epsilon=1e-5)
            cmp.self_attn = SimpleNamespace(
                has_indexer=False,
                q_a_layernorm=SimpleNamespace(weight=q_norm, variance_epsilon=1e-5),
                kv_a_layernorm=SimpleNamespace(weight=kv_norm, variance_epsilon=1e-5),
            )
            cmp._qkv_projection = cmp._q_b_proj = (None, None)
            # Isolate cache integration while exercising the real CMP fused
            # RMSNorm/FP8/RoPE/cache kernel and its real CUDA graph dependencies.
            cmp.ops = SimpleNamespace(
                add_norm_quant=lambda *a, **kw: (hidden, hidden, hidden, None),
                qkv_a_proj=lambda *a, layer=layer, **kw: projected[layer],
                qkv_rmsnorm_quant_rope_cached=ops.qkv_rmsnorm_quant_rope_cached,
                q_b_proj=lambda *a, **kw: (query, query),
                absorbed_q_nope_bmm=lambda *a, **kw: None,
            )
            cmps.append(cmp)

        def run():
            for layer, cmp in enumerate(cmps):
                cmp.mla_prologue(
                    hidden,
                    hidden,
                    implementation,
                    SimpleNamespace(kv_cache_base=host[layer]),
                    ids,
                    reuse_topk_indices=True,
                )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            run()
        for step in range(8):
            projected.add_(0.125)
            ids.copy_(
                (
                    (torch.arange(capacity, device="cuda") + step * 2048) % total
                ).reshape_as(ids)
            )
            ids[0, :3] = slots[:3]
            ids[-1, -1] = -1
            for layer in range(layers):
                ops.qkv_rmsnorm_quant_rope_cached(
                    projected[layer],
                    q_norm,
                    kv_norm,
                    cos_sin,
                    positions,
                    slots,
                    cache=reference[layer],
                )
            graph.replay()
            self.assert_rows(
                working, reference, ids, working.physical_indices, joined=True
            )
        self.stream.synchronize()
        for layer in range(layers):
            self.assertTrue(
                torch.equal(host[layer].view(torch.uint8), reference[layer][1:].cpu())
            )

    def test_prefetch_precedes_q_expansion_without_changing_projections(self):
        from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention

        attention = MlaAttention.__new__(MlaAttention)
        torch.nn.Module.__init__(attention)
        attention.q_lora_rank = attention.kv_lora_rank = attention.q_head_dim = 4
        attention.num_heads = attention.qk_rope_head_dim = attention.v_head_dim = 2
        attention.layer_idx = 0
        attention.reuse_topk_indices = False
        attention._fuse_q_a_norm_mode = "none"
        attention._fuse_kv_a_norm = False
        attention.parallelism_config = SimpleNamespace(get_attn_tp_size=lambda: 1)
        attention.fused_qkv_a_proj = torch.nn.Linear(16, 10, bias=False).cuda()
        attention.q_a_layernorm = torch.nn.LayerNorm(4).cuda()
        attention.q_b_proj = torch.nn.Linear(4, 8, bias=False).cuda()
        attention.kv_a_layernorm = torch.nn.LayerNorm(4).cuda()
        attention.o_proj = torch.nn.Identity()
        events = []
        attention.q_b_proj.register_forward_hook(lambda *args: events.append("q_b"))
        attention.kv_a_layernorm.register_forward_hook(
            lambda *args: events.append("kv_norm")
        )
        topk = torch.zeros((3, 1, 1), dtype=torch.int32, device="cuda")

        def indexer(*args, **kwargs):
            events.append("indexer")
            return topk

        def forward(q, kv, rope, cache, layer, selected, **kwargs):
            events.append("attention")
            self.assertIs(selected, topk)
            self.assertEqual(
                kwargs.get("kv_prefetched", False), bool(fmha.pinned_mla_groups)
            )
            return q[:, :, :2] + kv.reshape(-1, 2, 2)

        attention.indexer = indexer
        fmha = SimpleNamespace(
            pinned_mla_groups={},
            is_sparse=lambda: True,
            fmha_params=None,
            attn_inputs=None,
            cp_params=None,
            forward=forward,
            prefetch_kv=lambda *args: events.append("prefetch"),
        )
        hidden = torch.randn((3, 16), device="cuda")
        reference = attention(hidden, fmha)
        self.assertEqual(events, ["q_b", "kv_norm", "indexer", "attention"])
        events.clear()
        fmha.pinned_mla_groups = {0: object()}
        actual = attention(hidden, fmha)
        self.assertEqual(events, ["indexer", "prefetch", "q_b", "kv_norm", "attention"])
        self.assertTrue(torch.equal(actual, reference))

    def assert_rows(self, cache, reference, logical, physical, joined=False):
        if not joined:
            cache.layer_cache(len(reference) - 1)
        valid = logical >= 0
        self.assertTrue(torch.equal(physical < 0, ~valid))
        for layer, original in enumerate(reference):
            resident = cache.resident[layer].view(torch.uint8).flatten(0, 1)
            expected = original.view(torch.uint8).flatten(0, 1)[logical[valid].long()]
            actual = resident[physical[valid].long()]
            self.assertTrue(torch.equal(actual, expected), f"layer={layer}")

    def test_hbm_priority_spill_shared_group_long_reuse(self):
        torch.manual_seed(712)
        hbm_tokens, capacity, total, width = 8192, 8192, 73728, 16
        reference = [
            torch.randint(
                0, 255, (total // 64, 64, width), dtype=torch.uint8, device="cuda"
            )
            for _ in range(3)
        ]
        host = [tensor[hbm_tokens // 64 :].cpu().pin_memory() for tensor in reference]
        resident = [
            torch.empty((256, 64, width), dtype=torch.uint8, device="cuda")
            for _ in reference
        ]
        for target, source in zip(resident, reference):
            target[:128].copy_(source[:128])
        generations = torch.zeros(total // 64, dtype=torch.int64, pin_memory=True)
        cache = PinnedMlaWorkingSet(
            host,
            capacity,
            64,
            torch.device("cuda", 0),
            block_generations=generations,
            hbm_tokens=hbm_tokens,
            hbm_cache=resident,
        )
        ids = torch.arange(capacity, dtype=torch.int32, device="cuda").reshape(4, 2048)
        slots = cache.begin(ids)
        self.assert_rows(cache, reference, ids, slots)
        self.assertTrue(torch.equal(slots, ids))
        self.assertTrue((cache.mapping == -1).all().item())
        # HBM writes do not touch any pin block, including the same compact offset.
        pin_before = host[0].clone()
        write_ids = torch.tensor(
            [17, hbm_tokens + 31], dtype=torch.int32, device="cuda"
        )
        values = torch.full((2, width), 42, dtype=torch.uint8, device="cuda")
        cache.write(0, write_ids[:1], values[:1])
        reference[0].flatten(0, 1)[17].fill_(42)
        self.stream.synchronize()
        self.assertTrue(torch.equal(host[0], pin_before))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            physical = cache.begin(ids)
            for layer in range(3):
                cache.write(layer, write_ids, values)
        steps = 512
        for step in range(steps):
            # All HBM, mixed, and mostly pin, with more unique pin IDs than capacity over time.
            start = 0 if step % 3 == 0 else (step * 1024) % (total - capacity)
            ids.copy_((torch.arange(capacity, device="cuda") + start).reshape_as(ids))
            ids[:, -1] = -1
            if step % 16 == 0:
                self.stream.synchronize()
                block = hbm_tokens // 64 + (step * 7) % (host[0].shape[0])
                generations[block] += 1
                for layer in range(3):
                    host[layer][block - hbm_tokens // 64].fill_(step % 251)
                    reference[layer][block].fill_(step % 251)
            graph.replay()
            for tensor in reference:
                tensor.flatten(0, 1)[write_ids.long()] = values
            self.assert_rows(cache, reference, ids, physical, joined=True)
            direct = (ids >= 0) & (ids < hbm_tokens)
            self.assertTrue(torch.equal(physical[direct], ids[direct]))
        self.stream.synchronize()
        for layer in range(3):
            self.assertTrue(torch.equal(host[layer], reference[layer][128:].cpu()))
        print(
            f"[HYBRID_MLA_REUSE] steps={steps} layers=3 mtp_queries=4 capacity={capacity}",
            flush=True,
        )

    def test_byte_checker_detects_corruption(self):
        cache, _ = self.make_cache(layers=1)
        ids = torch.tensor([1, 1, 50, -1], dtype=torch.int32, device="cuda")
        slots = cache.begin(ids)
        resident = cache.layer_cache(0)
        errors = torch.zeros((), dtype=torch.int32, device="cuda")

        def check():
            _check_selected_bytes[(4,)](
                cache.backing[0].view(torch.uint8),
                resident.view(torch.uint8),
                ids,
                slots,
                errors,
                ids.numel(),
                cache.capacity,
                cache.width,
                1024,
            )

        check()
        self.assertEqual(errors.item(), 0)
        resident.flatten(0, 1)[slots[0].long(), 0] ^= 1
        check()
        self.assertEqual(errors.item(), 2)  # Both duplicate selections must fail.

    def test_eviction_duplicates_padding_and_multiple_requests(self):
        for dtype, width in ((torch.uint8, 656), (torch.bfloat16, 576)):
            cache, reference = self.make_cache(dtype, width)
            for step in range(12):
                # Force eviction while keeping overlapping requests and topk hits.
                ids = (
                    torch.arange(4096, device="cuda") + step * 2048
                ) % cache.logical_tokens
                ids = ids.to(torch.int32).reshape(2, 2048)
                ids[1, :512] = ids[0, :512]
                ids[0, -64:] = -1
                physical = cache.begin(ids)
                self.assert_rows(cache, reference, ids, physical)
            all_padding = torch.full_like(ids, -1)
            self.assert_rows(cache, reference, all_padding, cache.begin(all_padding))

    def test_new_tokens_and_allocator_reuse(self):
        cache, reference = self.make_cache()
        ids = torch.arange(4096, dtype=torch.int32, device="cuda").reshape(2, 2048)
        physical = cache.begin(ids)
        updated_ids = ids.flatten()[::64].contiguous()
        for layer in range(len(reference)):
            values = torch.full(
                (updated_ids.numel(), 656),
                211 + layer,
                dtype=torch.uint8,
                device="cuda",
            )
            cache.write(layer, updated_ids, values)
            reference[layer].flatten(0, 1)[updated_ids.long()] = values
        self.assert_rows(cache, reference, ids, physical)
        cache.invalidate(updated_ids)
        # Simulate an external connector replacing recycled block contents.
        torch.cuda.synchronize()
        for layer in range(len(reference)):
            cache.backing[layer].flatten(0, 1)[updated_ids.cpu().long()] = 19 + layer
            reference[layer].flatten(0, 1)[updated_ids.long()] = 19 + layer
        self.assert_rows(cache, reference, ids, cache.begin(ids))

    def test_eviction_with_only_one_available_slot(self):
        for capacity in (64, 4096, 65536):
            cache, reference = self.make_cache(capacity=capacity, width=16)
            ids = torch.arange(capacity, dtype=torch.int32, device="cuda")
            self.assert_rows(cache, reference, ids, cache.begin(ids))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                cache.clock.zero_()
                physical = cache.begin(ids)
                cache.layer_cache(len(reference) - 1)
            # Exactly one victim, at the end of the protected scan range.
            # Preserve every other slot across all shared-index group layers.
            for step in range(3):
                ids.copy_(cache.tags)
                victim = ids[-1].item()
                ids[-1] = capacity + step
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                print(
                    f"[PINNED_MLA_SINGLE_VICTIM] capacity={capacity} "
                    f"step={step} ms={start.elapsed_time(end):.6f}",
                    flush=True,
                )
                # Wide margin over the sub-ms scan; catches the old ~96 ms walk.
                self.assertLess(start.elapsed_time(end), 10.0)
                self.assert_rows(cache, reference, ids, physical, joined=True)
                self.assertTrue(
                    torch.equal(
                        physical,
                        torch.arange(capacity, dtype=torch.int32, device="cuda"),
                    )
                )
                self.assertEqual(cache.mapping[victim].item(), -1)
                self.assertTrue(torch.equal(cache.tags, ids))

    def test_small_admissions_use_entire_capacity(self):
        cache, reference = self.make_cache(capacity=512, width=16)
        ids = torch.zeros(1, dtype=torch.int32, device="cuda")
        cache.begin(ids)
        cache.layer_cache(len(reference) - 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            cache.begin(ids)
            cache.layer_cache(len(reference) - 1)
        for token in range(1, cache.capacity):
            ids.fill_(token)
            graph.replay()
        all_ids = torch.arange(cache.capacity, dtype=torch.int32, device="cuda")
        slots = cache.mapping[: cache.capacity].clone()
        self.assertTrue(torch.all(slots >= 0).item())
        self.assertEqual(slots.unique().numel(), cache.capacity)
        self.assert_rows(cache, reference, all_ids, slots, joined=True)

    def test_cuda_graph_replay_changes_selection(self):
        cache, reference = self.make_cache()
        ids = torch.arange(4096, dtype=torch.int32, device="cuda").reshape(2, 2048)
        # Warm all kernel specializations on the owning stream.
        for _ in range(3):
            physical = cache.begin(ids)
            self.assert_rows(cache, reference, ids, physical)
        graph = torch.cuda.CUDAGraph()
        # graph() normally substitutes a side stream; this cache owns this stream.
        with torch.cuda.graph(graph, stream=cache.compute_stream):
            physical = cache.begin(ids)
            snapshots = [cache.layer_cache(i).clone() for i in range(3)]
        for step in range(8):
            ids.copy_(
                (torch.arange(4096, device="cuda").reshape(2, 2048) + step * 1536)
                % cache.logical_tokens
            )
            graph.replay()
            for layer, resident in enumerate(snapshots):
                actual = resident.flatten(0, 1)[physical.long()]
                expected = reference[layer].flatten(0, 1)[ids.long()]
        self.assertTrue(torch.equal(actual, expected))

    def test_capacity_guard(self):
        cache, _ = self.make_cache(layers=1)
        with self.assertRaisesRegex(ValueError, "flattened topk"):
            cache.begin(torch.zeros(4097, dtype=torch.int32, device="cuda"))

    def test_registered_arena_gpu_reads_and_writes(self):
        runtime = ctypes.CDLL("libcudart.so.13")
        runtime.cudaHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        runtime.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        size = 16384 * 656
        with mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS) as arena:
            host = torch.frombuffer(arena, dtype=torch.uint8).reshape(256, 64, 656)
            host.fill_(29)
            self.assertEqual(runtime.cudaHostRegister(host.data_ptr(), size, 0), 0)
            cache = None
            try:
                self.assertTrue(host.is_pinned())
                cache = PinnedMlaWorkingSet([host], 2048, 64, torch.device("cuda", 0))
                reference = [host.cuda()]
                for offset in (0, 8192, 0):
                    ids = torch.arange(2048, dtype=torch.int32, device="cuda") + offset
                    self.assert_rows(cache, reference, ids, cache.begin(ids))
                values = torch.full((1, 656), 137, dtype=torch.uint8, device="cuda")
                cache.write(0, ids[:1], values)
                reference[0].flatten(0, 1)[0] = values[0]
                self.assert_rows(cache, reference, ids, cache.begin(ids))
                self.stream.synchronize()
                self.assertTrue(torch.all(host[0, 0] == 137).item())
            finally:
                torch.cuda.synchronize()
                self.assertEqual(runtime.cudaHostUnregister(host.data_ptr()), 0)
                del cache, host

    def test_eager_write_after_capture_and_replay(self):
        cache, reference = self.make_cache(layers=2)
        ids = torch.arange(2048, dtype=torch.int32, device="cuda")
        for _ in range(3):
            cache.begin(ids)
            cache.layer_cache(1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            cache.begin(ids)
            for layer in range(2):
                cache.layer_cache(layer)
            cache.wait_prefetch_complete()
        # The first real request is an eager prefill, before any graph replay.
        # Its write must not wait on an internal captured event.
        for step in range(4):
            if step:
                graph.replay()
            for layer in range(2):
                values = torch.full(
                    (1, 656), 31 + layer + step, dtype=torch.uint8, device="cuda"
                )
                cache.write(layer, ids[:1], values)
                reference[layer].flatten(0, 1)[0] = values[0]
            cache.wait_prefetch_complete()
            for layer in range(2):
                actual = cache.gather_pages(
                    layer, torch.zeros(1, dtype=torch.int32, device="cuda")
                )
                self.assertTrue(torch.equal(actual[0], reference[layer][0]))
            self.stream.synchronize()
            for layer in range(2):
                self.assertTrue(
                    torch.all(cache.backing[layer][0, 0] == 31 + layer + step).item()
                )

    def test_switch_captured_batch_sizes_and_write_current_kv(self):
        cache, reference = self.make_cache(layers=1)
        captures = []
        for rows in (1, 2):
            ids = torch.arange(rows * 2048, dtype=torch.int32, device="cuda").reshape(
                rows, 2048
            )
            values = torch.zeros((rows, 656), dtype=torch.uint8, device="cuda")

            def run():
                physical = cache.begin(ids)
                cache.write(0, ids[:, 0].contiguous(), values)
                return physical, cache.layer_cache(0).clone()

            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                physical, snapshot = run()
            captures.append((graph, ids, values, physical, snapshot))
        # Warmup/capture wrote the current-token rows into the backing store.
        self.stream.synchronize()
        reference[0].copy_(cache.backing[0])
        for step in range(12):
            graph, ids, values, physical, snapshot = captures[step % 2]
            ids.copy_(
                (torch.arange(ids.numel(), device="cuda") + step * 1536).reshape(
                    ids.shape
                )
                % cache.logical_tokens
            )
            values.fill_(step + 17)
            graph.replay()
            reference[0].flatten(0, 1)[ids[:, 0].long()] = values
            self.assertTrue(
                torch.equal(
                    snapshot.flatten(0, 1)[physical.long()],
                    reference[0].flatten(0, 1)[ids.long()],
                )
            )

    def test_paged_fp8_attention_matches_hbm(self):
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        torch.manual_seed(93)
        host = torch.empty((2048, 64, 656), dtype=torch.uint8, pin_memory=True)
        host[..., :512] = (
            torch.randn(2048, 64, 512).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        host[..., 512:528].view(torch.float32).fill_(0.01)
        host[..., 528:] = torch.randn(2048, 64, 64, dtype=torch.bfloat16).view(
            torch.uint8
        )
        baseline = host.cuda()
        hbm_tokens = 65536
        resident = torch.empty((2048, 64, 656), dtype=torch.uint8, device="cuda")
        resident[:1024].copy_(baseline[:1024])
        cache = PinnedMlaWorkingSet(
            [host[1024:]],
            65536,
            64,
            torch.device("cuda", 0),
            hbm_tokens=hbm_tokens,
            hbm_cache=[resident],
        )
        block_table = torch.arange(2048, dtype=torch.int32, device="cuda").unsqueeze(0)

        def attend(q, kv, indices):
            metadata, _ = get_mla_metadata(
                cache_seqlens=None,
                num_q_tokens_per_head_k=q.shape[1],
                topk=2048,
                num_heads_q=128,
                num_heads_k=1,
                is_fp8_kvcache=True,
            )
            return flash_mla_with_kvcache(
                q=q,
                k_cache=kv.unsqueeze(2),
                block_table=block_table,
                head_dim_v=512,
                cache_seqlens=None,
                tile_scheduler_metadata=metadata,
                num_splits=None,
                is_fp8_kvcache=True,
                indices=indices,
                softmax_scale=192**-0.5,
            )[0]

        # Decode/MTP graph sizes, unrelated physical pages, padding and repeated
        # topk tokens must preserve attention even as the working set churns.
        for rows in (1, 4, 8, 32, 8, 32, 1):
            q = torch.randn(1, rows, 128, 576, dtype=torch.bfloat16, device="cuda")
            ids = torch.randint(
                0,
                32768 if rows == 1 else 131072,
                (1, rows, 2048),
                dtype=torch.int32,
                device="cuda",
            )
            ids = ids.sort(dim=-1).values
            ids[..., -8:] = -1
            expected = attend(q, baseline, ids)
            physical = cache.begin(ids)
            actual = attend(q, cache.layer_cache(0), physical)
            self.assertTrue(torch.isfinite(expected).all().item())
            self.assertTrue(
                torch.equal(actual, expected), f"attention differs at rows={rows}"
            )

    def test_recycled_block_refreshes_existing_slots(self):
        cache, reference = self.make_cache()
        ids = torch.arange(2048, dtype=torch.int32, device="cuda").repeat(2, 1)
        physical = cache.begin(ids)
        self.assert_rows(cache, reference, ids, physical)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=cache.compute_stream):
            refreshed = cache.begin(ids)
            snapshots = [cache.layer_cache(i).clone() for i in range(3)]
        for step in range(3):
            self.stream.synchronize()
            cache.generations[3] += 1
            for layer in range(len(reference)):
                cache.backing[layer][3].fill_(73 + layer + step)
                reference[layer][3].fill_(73 + layer + step)
            graph.replay()
            self.assertTrue(torch.equal(physical, refreshed))
            for layer, resident in enumerate(snapshots):
                self.assertTrue(
                    torch.equal(
                        resident.flatten(0, 1)[refreshed.long()],
                        reference[layer].flatten(0, 1)[ids.long()],
                    )
                )

    def test_high_concurrency_random_selection(self):
        # 48 independent 42K contexts, four target-verify rows each. Random
        # histories expose UVA latency hidden by contiguous microbenchmarks.
        batch, context, topk, rows = 48, 42048, 2048, 4
        capacity = batch * rows * topk
        torch.manual_seed(2718)
        host = torch.randint(
            0, 256, (batch * context // 64, 64, 656), dtype=torch.uint8
        ).pin_memory()
        generations = torch.zeros(host.shape[0], dtype=torch.int64, pin_memory=True)
        cache = PinnedMlaWorkingSet(
            [host], capacity, 64, self.stream.device, block_generations=generations
        )
        indices = (
            torch.randint(
                context, (batch, rows, topk), device="cuda", dtype=torch.int32
            )
            + torch.arange(batch, device="cuda", dtype=torch.int32)[:, None, None]
            * context
        )
        indices[:, :, -1] = -1
        slots = cache.begin(indices)
        cache.layer_cache(0)
        self.stream.synchronize()
        valid = indices >= 0
        torch.testing.assert_close(
            cache.resident[0].reshape(-1, 656)[slots[valid].long()].cpu(),
            host.reshape(-1, 656)[indices[valid].cpu().long()],
            rtol=0,
            atol=0,
        )
        # Compare the original and candidate paths with identical selections.
        for snapshot, fetch_rows in (
            (False, 0),
            (True, 0),
            (True, 4),
            (True, 8),
            (True, 16),
        ):
            cache.snapshot_generations = snapshot
            if snapshot and cache.generation_snapshot is None:
                cache.generation_snapshot = torch.empty_like(generations, device="cuda")
            cache.fetch_rows = fetch_rows
            cache.fetch_tiled = fetch_rows > 0
            # Replay after external reuse must fetch the new bytes, including
            # duplicate selections. Updates happen only after stream completion.
            self.stream.synchronize()
            host[0].fill_(fetch_rows + 17)
            generations[0] += 1
            slots = cache.begin(indices)
            cache.layer_cache(0)
            self.stream.synchronize()
            torch.testing.assert_close(
                cache.resident[0].reshape(-1, 656)[slots[valid].long()].cpu(),
                host.reshape(-1, 656)[indices[valid].cpu().long()],
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
