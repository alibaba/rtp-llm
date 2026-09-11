"""CPU capacity checks and L20D_DEV GPU tests; missing GPUs must fail."""

import ctypes
import json
import mmap
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.pinned_mla_cache import (
    PinnedMlaWorkingSet,
    _check_selected_bytes,
)


@triton.jit
def _gather_bf16(KV, Indices, Out, N: tl.constexpr):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    ids = tl.load(Indices + rows, rows < N, other=-1).to(tl.int64)
    valid = (rows < N) & (ids >= 0)
    x = tl.arange(0, 512)
    quantized = tl.load((KV + ids[:, None] * 656 + x[None, :]).to(tl.pointer_type(tl.float8e4nv)),
                        valid[:, None], other=0.0).to(tl.float32)
    scales = tl.load((KV + ids[:, None] * 656 + 512 + (x[None, :] // 128) * 4)
                     .to(tl.pointer_type(tl.float32)), valid[:, None], other=0)
    tl.store(Out + rows[:, None] * 576 + x[None, :],
             (quantized * scales).to(tl.bfloat16), (rows < N)[:, None])
    r = tl.arange(0, 64)
    rope = tl.load((KV + ids[:, None] * 656 + 528 + r[None, :] * 2)
                   .to(tl.pointer_type(tl.bfloat16)), valid[:, None], other=0)
    tl.store(Out + rows[:, None] * 576 + 512 + r[None, :], rope, (rows < N)[:, None])


class PinnedMlaCapacityTest(unittest.TestCase):
    def test_rejects_int32_capacity_overflow_before_allocation(self):
        # Meta tensors exercise huge shapes without allocating host/GPU memory.
        for host_tokens, hbm_tokens, resident_tokens, error in (
            (1 << 31, 0, 64, "logical"),
            (64, 0, 1 << 31, "physical"),
            (64, (1 << 31) - 128, 128, "physical"),
        ):
            with self.subTest(
                host_tokens=host_tokens,
                hbm_tokens=hbm_tokens,
                resident_tokens=resident_tokens,
            ):
                backing = torch.empty((host_tokens // 64, 64, 1), device="meta")
                with self.assertRaisesRegex(
                    ValueError, f"{error} token capacity must fit int32"
                ):
                    PinnedMlaWorkingSet(
                        [backing],
                        resident_tokens,
                        64,
                        torch.device("cuda"),
                        hbm_tokens=hbm_tokens,
                    )

    def test_int32_capacity_limit_passes_size_validation(self):
        max_tokens = torch.iinfo(torch.int32).max
        for host_tokens, hbm_tokens, resident_tokens in (
            (max_tokens, 0, 1),
            (1, max_tokens - 1, 1),
            (1, 0, max_tokens),
        ):
            with self.subTest(
                host_tokens=host_tokens,
                hbm_tokens=hbm_tokens,
                resident_tokens=resident_tokens,
            ):
                backing = torch.empty((host_tokens, 1, 1), device="meta")
                # An in-range shape reaches the backing-storage check.
                with self.assertRaisesRegex(ValueError, "CPU pinned memory"):
                    PinnedMlaWorkingSet(
                        [backing],
                        resident_tokens,
                        1,
                        torch.device("cuda"),
                        hbm_tokens=hbm_tokens,
                    )


class PinnedMlaCacheTest(unittest.TestCase):
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
                ids[positions] = torch.arange(victims, dtype=ids.dtype, device="cuda") + capacity + step * victims
                cache.clock.fill_((1 << 32) * (step % 2))
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
                self.assert_rows(cache, reference, ids, physical, joined=True)
                self.assertTrue(torch.all(cache.mapping[old[-victims:].long()] == -1).item())
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
            raise RuntimeError("L20D_DEV CUDA worker required")
        name = torch.cuda.get_device_name()
        if "L20D" not in name:
            raise RuntimeError(f"Expected L20D_DEV, got {name}")
        print(f"[PINNED_MLA_WORKER] hostname={socket.gethostname()} gpu={name}", flush=True)

    def make_cache(self, dtype=torch.uint8, width=656, layers=3, capacity=4096):
        torch.manual_seed(412)
        host = [
            torch.randint(0, 128, (capacity // 4, 64, width), dtype=torch.uint8)
            .to(dtype).pin_memory()
            for _ in range(layers)
        ]
        generations = torch.zeros(host[0].shape[0], dtype=torch.int64, pin_memory=True)
        cache = PinnedMlaWorkingSet(
            host, capacity, 64, torch.device("cuda", 0), block_generations=generations
        )
        return cache, [tensor.cuda() for tensor in host]

    def test_mtp_graph_clone_reuses_working_set_for_same_kv_arena(self):
        from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel
        from rtp_llm.models_py.model_desc.module_base import GptModelBase

        model = GenericMoeMTPModel.__new__(GenericMoeMTPModel)
        resource_config = SimpleNamespace(enable_layer_micro_batch=0)
        GptModelBase.__init__(
            model, SimpleNamespace(num_layers=1, vocab_size=8), None, None, 1,
            device_resource_config=resource_config,
        )
        model.moe_config = None
        model.max_generate_batch_size = 1
        model.device_resource_config = resource_config
        for name in ("embed_tokens", "pre_fc_norm_embedding", "pre_fc_norm_hidden", "fc", "norm"):
            setattr(model, name, torch.nn.Identity())
        model.layers = torch.nn.ModuleList()
        model._mtp_indexer_share_enabled = True
        model._mtp_shared_topk_indices = torch.zeros((1, 4), dtype=torch.int32)
        backing = torch.zeros((1, 64, 656), dtype=torch.uint8)
        kv = SimpleNamespace(
            dsa_mla_resident_tokens=64, dsa_mla_hbm_blocks=0,
            block_generations=torch.zeros(1, dtype=torch.int64),
            mla_hbm_cache_by_layer=[torch.empty_like(backing)],
            kv_cache_base_by_layer=[backing], kv_scale_base_by_layer=[],
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
        from rtp_llm.models_py.modules.hybrid.glm5_cmp import Glm5Cmp, should_enable_glm5_cmp

        layer_cmp = SimpleNamespace(
            _disabled_reason=None, has_indexer=True, ops=object(),
            _unsupported_call_reason=lambda *args: None,
        )
        query = torch.empty((4, 64, 576), device="cuda")
        indices = torch.empty((4, 2048), dtype=torch.int32, device="cuda")
        resident = torch.empty((128, 64, 656), dtype=torch.uint8, device="cuda")
        physical = torch.empty_like(indices)
        working = SimpleNamespace(resident=[resident], physical_indices=physical)
        flashmla = SimpleNamespace(expects_paged_kv=True, forward=Mock())
        implementation = SimpleNamespace(
            fmha_impl=flashmla, fmha_params=None, weights={},
            pinned_mla_groups={0: (working, 0)},
        )
        kv_cache = SimpleNamespace(get_layer_cache=lambda _: None)
        self.assertTrue(should_enable_glm5_cmp(
            [SimpleNamespace(cmp=layer_cmp)], 1, query, implementation, kv_cache
        ))
        cmp = object.__new__(Glm5Cmp)
        cmp.layer_idx = 0
        cmp.sparse_mla(query, indices, implementation, SimpleNamespace(kv_cache_base=None))
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
        reference = [torch.zeros((total // 64, 64, 656), dtype=torch.uint8, device="cuda")
                     for _ in range(layers)]
        host = [tensor[1:].cpu().pin_memory().view(dtype) for tensor in reference]
        resident = [torch.zeros(((hbm_tokens + capacity) // 64, 64, 656),
                                dtype=torch.uint8, device="cuda").view(dtype) for _ in reference]
        working = PinnedMlaWorkingSet(host, capacity, 64, torch.device("cuda", 0),
                                    hbm_tokens=hbm_tokens, hbm_cache=resident)
        ids = torch.arange(capacity, dtype=torch.int32, device="cuda").reshape(rows, 2048)
        slots = torch.tensor([3, 65, 4097, -1], dtype=torch.int64, device="cuda")
        positions = torch.arange(rows, dtype=torch.int32, device="cuda")
        projected = torch.randn((layers, rows, 2624), dtype=torch.bfloat16, device="cuda")
        q_norm = torch.ones(2048, dtype=torch.bfloat16, device="cuda")
        kv_norm = torch.ones(512, dtype=torch.bfloat16, device="cuda")
        cos_sin = torch.cat((torch.ones((rows, 32)), torch.zeros((rows, 32))), dim=-1).cuda()
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
                cmp.mla_prologue(hidden, hidden, implementation,
                                 SimpleNamespace(kv_cache_base=host[layer]), ids,
                                 reuse_topk_indices=True)

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            run()
        for step in range(8):
            projected.add_(0.125)
            ids.copy_(((torch.arange(capacity, device="cuda") + step * 2048) % total).reshape_as(ids))
            ids[0, :3] = slots[:3]
            ids[-1, -1] = -1
            for layer in range(layers):
                ops.qkv_rmsnorm_quant_rope_cached(
                    projected[layer], q_norm, kv_norm, cos_sin, positions, slots,
                    cache=reference[layer],
                )
            graph.replay()
            self.assert_rows(working, reference, ids, working.physical_indices, joined=True)
        self.stream.synchronize()
        for layer in range(layers):
            self.assertTrue(torch.equal(host[layer].view(torch.uint8), reference[layer][1:].cpu()))

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
        attention.kv_a_layernorm.register_forward_hook(lambda *args: events.append("kv_norm"))
        topk = torch.zeros((3, 1, 1), dtype=torch.int32, device="cuda")

        def indexer(*args, **kwargs):
            events.append("indexer")
            return topk

        def forward(q, kv, rope, cache, layer, selected, **kwargs):
            events.append("attention")
            self.assertIs(selected, topk)
            self.assertEqual(kwargs.get("kv_prefetched", False), bool(fmha.pinned_mla_groups))
            return q[:, :, :2] + kv.reshape(-1, 2, 2)

        attention.indexer = indexer
        fmha = SimpleNamespace(
            pinned_mla_groups={}, is_sparse=lambda: True, fmha_params=None,
            attn_inputs=None, cp_params=None, forward=forward,
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
        reference = [torch.randint(0, 255, (total // 64, 64, width),
                                  dtype=torch.uint8, device="cuda") for _ in range(3)]
        host = [tensor[hbm_tokens // 64:].cpu().pin_memory() for tensor in reference]
        resident = [torch.empty((256, 64, width), dtype=torch.uint8, device="cuda")
                    for _ in reference]
        for target, source in zip(resident, reference):
            target[:128].copy_(source[:128])
        generations = torch.zeros(total // 64, dtype=torch.int64, pin_memory=True)
        cache = PinnedMlaWorkingSet(host, capacity, 64, torch.device("cuda", 0),
                                   block_generations=generations, hbm_tokens=hbm_tokens,
                                   hbm_cache=resident)
        ids = torch.arange(capacity, dtype=torch.int32, device="cuda").reshape(4, 2048)
        slots = cache.begin(ids)
        self.assert_rows(cache, reference, ids, slots)
        self.assertTrue(torch.equal(slots, ids))
        self.assertTrue((cache.mapping == -1).all().item())
        # HBM writes do not touch any pin block, including the same compact offset.
        pin_before = host[0].clone()
        write_ids = torch.tensor([17, hbm_tokens + 31], dtype=torch.int32, device="cuda")
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
        print(f"[HYBRID_MLA_REUSE] steps={steps} layers=3 mtp_queries=4 capacity={capacity}", flush=True)

    def test_byte_checker_detects_corruption(self):
        cache, _ = self.make_cache(layers=1)
        ids = torch.tensor([1, 1, 50, -1], dtype=torch.int32, device="cuda")
        slots = cache.begin(ids)
        resident = cache.layer_cache(0)
        errors = torch.zeros((), dtype=torch.int32, device="cuda")

        def check():
            _check_selected_bytes[(4,)](
                cache.backing[0].view(torch.uint8), resident.view(torch.uint8),
                ids, slots, errors, ids.numel(), cache.capacity, cache.width, 1024,
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
                ids = (torch.arange(4096, device="cuda") + step * 2048) % cache.logical_tokens
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
            values = torch.full((updated_ids.numel(), 656), 211 + layer,
                                dtype=torch.uint8, device="cuda")
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
                print(f"[PINNED_MLA_SINGLE_VICTIM] capacity={capacity} "
                      f"step={step} ms={start.elapsed_time(end):.6f}", flush=True)
                # Wide margin over the sub-ms scan; catches the old ~96 ms walk.
                self.assertLess(start.elapsed_time(end), 10.0)
                self.assert_rows(cache, reference, ids, physical, joined=True)
                self.assertTrue(torch.equal(physical, torch.arange(
                    capacity, dtype=torch.int32, device="cuda")))
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
        slots = cache.mapping[:cache.capacity].clone()
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
            ids.copy_((torch.arange(4096, device="cuda").reshape(2, 2048) + step * 1536)
                      % cache.logical_tokens)
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
        runtime.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
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

    def test_switch_captured_batch_sizes_and_write_current_kv(self):
        cache, reference = self.make_cache(layers=1)
        captures = []
        for rows in (1, 2):
            ids = torch.arange(rows * 2048, dtype=torch.int32, device="cuda").reshape(rows, 2048)
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
            ids.copy_((torch.arange(ids.numel(), device="cuda") + step * 1536)
                      .reshape(ids.shape) % cache.logical_tokens)
            values.fill_(step + 17)
            graph.replay()
            reference[0].flatten(0, 1)[ids[:, 0].long()] = values
            self.assertTrue(torch.equal(
                snapshot.flatten(0, 1)[physical.long()], reference[0].flatten(0, 1)[ids.long()]
            ))

    def test_paged_fp8_attention_matches_hbm(self):
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        torch.manual_seed(93)
        host = torch.empty((2048, 64, 656), dtype=torch.uint8, pin_memory=True)
        host[..., :512] = torch.randn(2048, 64, 512).to(torch.float8_e4m3fn).view(torch.uint8)
        host[..., 512:528].view(torch.float32).fill_(0.01)
        host[..., 528:] = torch.randn(2048, 64, 64, dtype=torch.bfloat16).view(torch.uint8)
        baseline = host.cuda()
        hbm_tokens = 65536
        resident = torch.empty((2048, 64, 656), dtype=torch.uint8, device="cuda")
        resident[:1024].copy_(baseline[:1024])
        cache = PinnedMlaWorkingSet([host[1024:]], 65536, 64, torch.device("cuda", 0),
                                   hbm_tokens=hbm_tokens, hbm_cache=[resident])
        block_table = torch.arange(2048, dtype=torch.int32, device="cuda").unsqueeze(0)

        def attend(q, kv, indices):
            metadata, _ = get_mla_metadata(
                cache_seqlens=None, num_q_tokens_per_head_k=q.shape[1], topk=2048,
                num_heads_q=128, num_heads_k=1, is_fp8_kvcache=True,
            )
            return flash_mla_with_kvcache(
                q=q, k_cache=kv.unsqueeze(2), block_table=block_table,
                head_dim_v=512, cache_seqlens=None, tile_scheduler_metadata=metadata,
                num_splits=None, is_fp8_kvcache=True, indices=indices,
                softmax_scale=192 ** -0.5,
            )[0]

        # Decode/MTP graph sizes, unrelated physical pages, padding and repeated
        # topk tokens must preserve attention even as the working set churns.
        for rows in (1, 4, 8, 32, 8, 32, 1):
            q = torch.randn(1, rows, 128, 576, dtype=torch.bfloat16, device="cuda")
            ids = torch.randint(0, 32768 if rows == 1 else 131072, (1, rows, 2048), dtype=torch.int32, device="cuda")
            ids = ids.sort(dim=-1).values
            ids[..., -8:] = -1
            expected = attend(q, baseline, ids)
            physical = cache.begin(ids)
            actual = attend(q, cache.layer_cache(0), physical)
            self.assertTrue(torch.isfinite(expected).all().item())
            self.assertTrue(torch.equal(actual, expected), f"attention differs at rows={rows}")

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
                self.assertTrue(torch.equal(
                    resident.flatten(0, 1)[refreshed.long()],
                    reference[layer].flatten(0, 1)[ids.long()],
                ))

    def test_high_concurrency_random_selection(self):
        # 48 independent 42K contexts, four target-verify rows each. Random
        # histories expose UVA latency hidden by contiguous microbenchmarks.
        batch, context, topk, rows = 48, 42048, 2048, 4
        capacity = batch * rows * topk
        torch.manual_seed(2718)
        host = torch.randint(0, 256, (batch * context // 64, 64, 656), dtype=torch.uint8).pin_memory()
        generations = torch.zeros(host.shape[0], dtype=torch.int64, pin_memory=True)
        cache = PinnedMlaWorkingSet([host], capacity, 64, self.stream.device,
                                    block_generations=generations)
        indices = (torch.randint(context, (batch, rows, topk), device="cuda", dtype=torch.int32)
                   + torch.arange(batch, device="cuda", dtype=torch.int32)[:, None, None] * context)
        indices[:, :, -1] = -1
        slots = cache.begin(indices)
        cache.layer_cache(0)
        self.stream.synchronize()
        valid = indices >= 0
        torch.testing.assert_close(cache.resident[0].reshape(-1, 656)[slots[valid].long()].cpu(),
                                   host.reshape(-1, 656)[indices[valid].cpu().long()], rtol=0, atol=0)
        # Compare the original and candidate paths with identical selections.
        for snapshot, fetch_rows in ((False, 0), (True, 0), (True, 4), (True, 8), (True, 16)):
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
            torch.testing.assert_close(cache.resident[0].reshape(-1, 656)[slots[valid].long()].cpu(),
                                       host.reshape(-1, 656)[indices[valid].cpu().long()], rtol=0, atol=0)

    def test_report_transfer_cost(self):
        cache, reference = self.make_cache(layers=1, capacity=65536)
        self._report_transfer_cost(cache, reference)
        # Compare a large, non-power-of-two automatic capacity on identical KV
        # and selected IDs, independently of full-model batch scheduling.
        cache = PinnedMlaWorkingSet(
            cache.backing, cache.logical_tokens - 64, 64, cache.device,
            block_generations=cache.generations,
        )
        self._report_transfer_cost(cache, reference)

    def _report_transfer_cost(self, cache, reference):
        ids = torch.arange(65536, dtype=torch.int32, device="cuda").reshape(32, 2048)
        self.assert_rows(cache, reference, ids, cache.begin(ids))
        results = {}
        for cold in (False, True):
            samples = []
            for _ in range(20):
                if cold:
                    cache.invalidate(ids)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                cache.begin(ids)
                cache.layer_cache(0)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            results["cold_ms" if cold else "hot_ms"] = sorted(samples)[len(samples) // 2]
        for cold in (False, True):
            def run():
                if cold:
                    cache.invalidate(ids)
                cache.begin(ids)
                cache.layer_cache(0)
            results["graph_cold_with_invalidate_ms" if cold else "graph_hot_ms"] = self.time_graph(run)
        q = torch.randn(128, 32, 128, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(128, 128, 512, dtype=torch.bfloat16, device="cuda")
        projected = torch.empty(128, 32, 512, dtype=torch.bfloat16, device="cuda")

        def project():
            torch.bmm(q, weight, out=projected)

        def serial():
            cache.begin(ids)
            cache.layer_cache(0)
            project()

        def overlap():
            cache.begin(ids)
            project()
            cache.layer_cache(0)

        results["graph_projection_ms"] = self.time_graph(project)
        results["graph_hot_serial_projection_ms"] = self.time_graph(serial)
        results["graph_hot_overlap_projection_ms"] = self.time_graph(overlap)
        # Isolate repeated UVA generation reads from cache-management cost.
        # A production snapshot would be shared by all layers in one forward.
        host_generations = cache.generations
        cache.generations = host_generations.to(cache.device)
        results["graph_hot_gpu_generations_ms"] = self.time_graph(serial)
        results["graph_generation_snapshot_ms"] = self.time_graph(
            lambda: cache.generations.copy_(host_generations, non_blocking=True)
        )
        cache.generations = host_generations
        results["cold_bytes"] = ids.numel() * cache.width
        results["backing_tokens"] = cache.logical_tokens
        results["resident_tokens"] = cache.capacity
        print("[PINNED_MLA_BENCH] " + json.dumps(results), flush=True)

    def time_graph(self, fn):
        for _ in range(3):
            fn()
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            for _ in range(10):
                fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(5):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 10)
        return sorted(samples)[2]

    def test_report_group_stream_cost(self):
        first, _ = self.make_cache(layers=1, capacity=65536)
        groups = [first] + [
            PinnedMlaWorkingSet(
                first.backing, first.capacity, 64, first.device,
                block_generations=first.generations,
            ) for _ in range(77)
        ]
        ids = torch.arange(65536, dtype=torch.int32, device="cuda").reshape(32, 2048)
        q = torch.randn(128, 32, 128, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(128, 128, 512, dtype=torch.bfloat16, device="cuda")
        projected = torch.empty(128, 32, 512, dtype=torch.bfloat16, device="cuda")

        def run():
            for cache in groups:
                cache.begin(ids)
                torch.bmm(q, weight, out=projected)
                cache.layer_cache(0)

        result = {"groups": len(groups), "separate_streams_ms": self.time_graph(run)}
        self.stream.synchronize()
        transfer = torch.cuda.Stream()
        for cache in groups:
            cache.transfer_stream = transfer
        result["shared_stream_ms"] = self.time_graph(run)
        self.stream.synchronize()
        generation_snapshot = first.generations.to(first.device)
        for cache in groups:
            cache.generations = generation_snapshot
        result["shared_stream_gpu_generations_ms"] = self.time_graph(run)
        print("[PINNED_MLA_GROUP_BENCH] " + json.dumps(result), flush=True)

    def test_flat_vs_paged_decode(self):
        from flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata


        for rows in (1, 8, 32, 128):
            tokens = rows * 2048 * 2
            host = torch.empty((tokens // 64, 64, 656), dtype=torch.uint8, pin_memory=True)
            # Ordinary FP32 scales, matching the real MLA quantization contract.
            host[..., :512] = torch.randint(0, 120, (tokens // 64, 64, 512), dtype=torch.uint8)
            host[..., 512:528].view(torch.float32).uniform_(0.003, 0.02)
            host[..., 528:] = torch.randn(tokens // 64, 64, 64, dtype=torch.bfloat16).view(torch.uint8)
            paged = host.cuda()
            indices = torch.randperm(tokens, device="cuda")[:rows * 2048].to(torch.int32).reshape(rows, 1, 2048)
            flat_indices = torch.arange(rows * 2048, dtype=torch.int32, device="cuda").reshape(rows, 1, 2048)
            flat = torch.empty((rows * 2048, 1, 576), dtype=torch.bfloat16, device="cuda")
            q = torch.randn(rows, 128, 576, dtype=torch.bfloat16, device="cuda")
            block_table = torch.arange(tokens // 64, dtype=torch.int32, device="cuda").unsqueeze(0)
            metadata, _ = get_mla_metadata()

            def run_paged():
                return flash_mla_with_kvcache(
                    q=q.unsqueeze(0), k_cache=paged.unsqueeze(2), block_table=block_table,
                    cache_seqlens=None, head_dim_v=512, tile_scheduler_metadata=metadata,
                    is_fp8_kvcache=True, indices=indices.squeeze(1).unsqueeze(0),
                    softmax_scale=192 ** -0.5,
                )[0].squeeze(0)

            def run_flat():
                return flash_mla_sparse_fwd(q, flat, flat_indices, 192 ** -0.5, d_v=512)[0]

            def gather_flat(source):
                _gather_bf16[(triton.cdiv(rows * 2048, 4),)](source, indices, flat, rows * 2048)
                return run_flat()

            expected = run_paged()
            actual = gather_flat(paged)
            relative_rmse = ((actual.float() - expected.float()).square().mean()
                             / expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
            max_error = (actual.float() - expected.float()).abs().max().item()
            self.assertTrue(torch.isfinite(actual).all().item())
            self.assertLess(relative_rmse, 0.01)
            self.assertTrue(torch.equal(gather_flat(host), actual))
            report = {
                "query_rows": rows, "heads": 128, "topk": 2048,
                "paged_ms": self.time_graph(run_paged),
                "flat_attention_only_ms": self.time_graph(run_flat),
                "hbm_gather_flat_total_ms": self.time_graph(lambda: gather_flat(paged)),
                "pinned_gather_flat_total_ms": self.time_graph(lambda: gather_flat(host)),
                "relative_rmse": relative_rmse, "max_abs_error": max_error,
                "flat_workspace_bytes": flat.numel() * flat.element_size(),
            }
            print("[PINNED_MLA_FLAT_COMPARE] " + json.dumps(report), flush=True)


if __name__ == "__main__":
    unittest.main()
