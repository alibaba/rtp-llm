"""Checkpoint-free contracts for the DSpark group32 projection helper."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41 import _dspark_commit_wkv as commit_wkv
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


def make_linears():
    torch.manual_seed(1901)
    linears = []
    for _ in range(3):
        weights = torch.randn(512, 5120, device="cuda").to(torch.float8_e4m3fn)
        scales = (2.0 ** torch.randint(-5, 3, (16, 160), device="cuda")).to(
            torch.float8_e8m0fnu
        )
        linears.append(V41MXFP8Linear(weights, scales))
    return linears


def exact(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DSparkCommitWKVTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("DeepGEMM MXFP8 requires SM10x")
        cls.linears = make_linears()

    def prepare(self, mode):
        with patch.object(commit_wkv, "sleep_enabled", return_value=False):
            prepared = commit_wkv.prepare_dspark_commit_wkv(self.linears, mode)
        self.assertIsNotNone(prepared)
        return prepared

    def test_shared_stacked_and_scale_layout(self):
        shared, stacked = self.prepare("shared"), self.prepare("stacked")
        self.assertEqual(stacked.stacked.weight_scales.stride(), (1, 1536))
        for i, linear in enumerate(self.linears):
            exact(self, stacked.stacked.weight[i * 512 : (i + 1) * 512], linear.weight)
            exact(
                self,
                stacked.stacked.weight_scales[i * 512 : (i + 1) * 512],
                linear.weight_scales,
            )
        for rows in (1, 6, 65, 192):
            x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
            for factor in (1.0, 1e-12, 0.0):
                source = x * factor
                expected = [linear(source) for linear in self.linears]
                for prepared in (shared, stacked):
                    for actual, wanted in zip(prepared.project(source), expected):
                        exact(self, actual, wanted)
                slices = stacked.project(source)
                self.assertEqual(slices[0].stride(), (1536, 1))

    def test_graph_replay_and_weight_lifetime(self):
        prepared = self.prepare("stacked")
        x = torch.randn(65, 5120, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                prepared.project(x)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = prepared.project(x)
        with torch.cuda.stream(stream):
            for _ in range(3):
                x.normal_()
                graph.replay()
                for actual, linear in zip(output, self.linears):
                    exact(self, actual, linear(x))
            original = self.linears[0].weight
            original_output = output[0].clone()
            address = prepared.stacked.weight.data_ptr()
            try:
                self.linears[0].weight = (-original.float()).to(original.dtype)
                self.assertFalse(prepared.matches(self.linears))
                fresh = self.prepare("stacked")
                self.assertTrue(fresh.matches(self.linears))
                self.assertNotEqual(fresh.stacked.weight.data_ptr(), address)
                graph.replay()
                exact(self, output[0], original_output)
                exact(self, fresh.project(x)[0], self.linears[0](x))
            finally:
                self.linears[0].weight = original
        torch.cuda.current_stream().wait_stream(stream)

    def test_auto_dispatch_boundaries_and_unsupported_shapes(self):
        with patch.object(
            commit_wkv, "sleep_enabled", return_value=False
        ), patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ), patch.dict(
            "os.environ", {}, clear=False
        ):
            import os

            os.environ.pop("DSV4_DSPARK_COMMIT_WKV", None)
            automatic = commit_wkv.prepare_dspark_commit_wkv(self.linears)
        self.assertEqual(automatic.mode, "auto")
        self.assertIsNotNone(automatic.stacked)
        for rows in (1, 3072, 3073):
            x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
            actual = automatic.project(x)
            if rows == 3073:
                self.assertIsNone(actual)
            else:
                for got, linear in zip(actual, self.linears):
                    exact(self, got, linear(x))
        for capability in ((9, 0), (10, 0)):
            with patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ):
                unsupported = self.prepare("auto")
            self.assertIsNone(unsupported.stacked)
            self.assertIsNone(unsupported.project(x[:1]))
        for n, k in ((256, 5120), (512, 256)):
            linears = [
                V41MXFP8Linear(
                    torch.zeros(n, k, device="cuda", dtype=torch.float8_e4m3fn),
                    torch.ones(n // 32, k // 32, device="cuda").to(
                        torch.float8_e8m0fnu
                    ),
                )
                for _ in range(3)
            ]
            with patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 3)
            ):
                unsupported = commit_wkv.prepare_dspark_commit_wkv(linears, "auto")
            self.assertIsNone(unsupported.stacked)

    def test_model_routes_each_precomputed_projection_to_its_layer(self):
        import rtp_llm.models_py.model_desc.deepseek_v41_base_dspark_model as module

        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            prepared = self.prepare("auto")
        # Exercise SM103 auto dispatch even when the consumer test runs on SM100.
        self.assertIsNotNone(prepared.stacked)
        x = torch.randn(6, 5120, device="cuda", dtype=torch.bfloat16)
        rows = torch.arange(6, device="cuda", dtype=torch.int32)
        received = []
        model = SimpleNamespace(
            _commit_wkv_projection=prepared,
            v4=SimpleNamespace(
                layers=[
                    SimpleNamespace(attn=SimpleNamespace(wkv=w)) for w in self.linears
                ]
            ),
            kv_cache=object(),
            _swa_block_table=lambda *args: torch.ones(
                1, 1, device="cuda", dtype=torch.int32
            ),
            _commit_layer_features=lambda index, *args, **kwargs: received.append(
                (index, kwargs["projected_kv"])
            ),
        )
        with patch.object(
            module, "as_tagged_attention_inputs", side_effect=lambda x: x
        ), patch.object(
            module, "cached_swa_region", return_value="decoder_swa_kv"
        ), patch.object(
            module, "require_pool_tokens_per_block", return_value=256
        ), patch.object(
            module, "create_tagged_cache_store_writer", return_value=None
        ), patch.object(
            module._profiler,
            "make_layer_forward_range",
            return_value=lambda _: nullcontext(),
        ), patch.object(
            prepared, "project", wraps=prepared.project
        ) as project:
            module.DeepSeekV4DSparkModel.commit_feature_rows(
                model,
                x,
                torch.zeros_like(rows),
                rows,
                rows.new_tensor([6]),
                SimpleNamespace(attention_inputs=object()),
            )
            project.assert_called_once_with(x)
        self.assertEqual([index for index, _ in received], [0, 1, 2])
        for (_, actual), linear in zip(received, self.linears):
            exact(self, actual, linear(x))
        self.assertEqual(
            len({value.untyped_storage().data_ptr() for _, value in received}), 1
        )

    def test_shared_storage_reload_updates_commit_and_proposal_graphs(self):
        from rtp_llm.models_py.modules.dsv41.utils import merge_v41_qkv_weights
        from rtp_llm.utils.model_weight import W

        owners, linears, proposals = [], [], []
        for _ in range(3):
            weights = {
                W.v4_attn_wq_a_w: torch.randn(128, 5120, device="cuda").to(
                    torch.float8_e4m3fn
                ),
                W.v4_attn_wq_a_s: torch.ones(4, 160, device="cuda").to(
                    torch.float8_e8m0fnu
                ),
                W.v4_attn_wkv_w: torch.randn(512, 5120, device="cuda").to(
                    torch.float8_e4m3fn
                ),
                W.v4_attn_wkv_s: torch.ones(16, 160, device="cuda").to(
                    torch.float8_e8m0fnu
                ),
            }
            proposal = merge_v41_qkv_weights(
                weights,
                W.v4_attn_wq_a_w,
                W.v4_attn_wq_a_s,
                W.v4_attn_wkv_w,
                W.v4_attn_wkv_s,
            )
            owners.append(weights)
            proposals.append(proposal)
            linears.append(
                V41MXFP8Linear(weights[W.v4_attn_wkv_w], weights[W.v4_attn_wkv_s])
            )
        with patch.object(commit_wkv, "sleep_enabled", return_value=False):
            prepared = commit_wkv.prepare_dspark_commit_wkv(
                linears, "stacked", layer_weights=owners, merged_linears=proposals
            )
        buffers = dict(prepared.named_buffers())
        self.assertIs(buffers["stacked.weight"], prepared.stacked.weight)
        self.assertIs(buffers["stacked.weight_scales"], prepared.stacked.weight_scales)
        self.assertEqual(prepared.state_dict(), {})
        for weights, linear in zip(owners, linears):
            self.assertIs(weights[W.v4_attn_wkv_w], linear.weight)
            self.assertEqual(
                linear.weight.untyped_storage().data_ptr(),
                prepared.stacked.weight.data_ptr(),
            )
        pointers = (
            prepared.stacked.weight.data_ptr(),
            prepared.stacked.weight_scales.data_ptr(),
        )
        x = torch.randn(6, 5120, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            prepared.project(x)
            for proposal in proposals:
                proposal(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            committed = prepared.project(x)
            proposed = [proposal(x) for proposal in proposals]
        original = [owner[W.v4_attn_wkv_w].clone() for owner in owners]
        query = [proposal.weight[:128].clone() for proposal in proposals]
        initial_result = [v.clone() for v in prepared.project(x)]
        for factor in (-1.0, 0.5):
            for weights, initial in zip(owners, original):
                weights[W.v4_attn_wkv_w].copy_(
                    (initial.float() * factor).to(initial.dtype)
                )
                weights[W.v4_attn_wkv_s].copy_(
                    torch.full((16, 160), 2.0, device="cuda").to(torch.float8_e8m0fnu)
                )
            prepared.reload(owners)
            graph.replay()
            self.assertEqual(
                pointers,
                (
                    prepared.stacked.weight.data_ptr(),
                    prepared.stacked.weight_scales.data_ptr(),
                ),
            )
            self.assertTrue(prepared.matches(linears))
            for i, linear in enumerate(linears):
                exact(self, committed[i], linear(x))
                exact(self, proposed[i][..., -512:], linear(x))
                exact(self, proposals[i].weight[:128], query[i])
            if factor == -1.0:
                self.assertFalse(torch.equal(committed[0], initial_result[0]))
        # Tensor replacement without rebinding is rejected before captured storage changes.
        saved = linears[0].weight
        try:
            linears[0].weight = saved.clone()
            with self.assertRaisesRegex(RuntimeError, "rebuild model and graphs"):
                prepared.reload(owners)
        finally:
            linears[0].weight = saved

    def test_model_weight_hook_reuses_owner_without_history_list(self):
        import rtp_llm.models_py.model_desc.deepseek_v41_base_dspark_model as module
        from rtp_llm.utils.model_weight import W

        owners = [
            {
                W.v4_attn_wkv_w: linear.weight,
                W.v4_attn_wkv_s: torch.ones(16, 160, device="cuda").to(
                    torch.float8_e8m0fnu
                ),
            }
            for linear in self.linears
        ]
        weights = SimpleNamespace(
            weights=owners,
            global_weights={
                W.v4_dspark_main_norm: object(),
                W.v4_dspark_main_proj_w: object(),
                W.v4_dspark_main_proj_s: object(),
            },
        )
        model = SimpleNamespace(
            _v4_args=SimpleNamespace(norm_eps=1e-6),
            v4=SimpleNamespace(
                layers=[
                    SimpleNamespace(attn=SimpleNamespace(wkv=w)) for w in self.linears
                ]
            ),
        )
        with patch.object(module, "RMSNorm"), patch.object(
            module, "_v4_fp8_linear"
        ), patch.dict(
            "os.environ", {"DSV4_DSPARK_COMMIT_WKV": "stacked"}
        ), patch.object(
            commit_wkv, "sleep_enabled", return_value=False
        ):
            module.DeepSeekV4DSparkModel._load_extra_weights(model, weights)
            prepared = model._commit_wkv_projection
            for _ in range(2):
                module.DeepSeekV4DSparkModel._load_extra_weights(model, weights)
                self.assertIs(model._commit_wkv_projection, prepared)
        self.assertFalse(hasattr(model, "_commit_wkv_preparations"))
        self.assertTrue(prepared.matches(self.linears))

    def test_fallback_and_capture_preparation(self):
        self.assertIsNone(commit_wkv.prepare_dspark_commit_wkv([]))
        self.assertIsNone(
            commit_wkv.prepare_dspark_commit_wkv([torch.nn.Identity()] * 3)
        )
        prepared = self.prepare("stacked")
        old = self.linears[0].weight_scales
        try:
            self.linears[0].weight_scales = old.contiguous()
            self.assertIsNone(commit_wkv.prepare_dspark_commit_wkv(self.linears))
            self.assertFalse(prepared.matches(self.linears))
        finally:
            self.linears[0].weight_scales = old
        x = torch.randn(6, 5120, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(prepared.project(x.float()))
        self.assertIsNone(prepared.project(x[:0]))
        with patch.object(commit_wkv, "sleep_enabled", return_value=True):
            automatic = commit_wkv.prepare_dspark_commit_wkv(self.linears, "auto")
        self.assertIsNone(automatic.stacked)
        self.assertIsNone(automatic.project(x))
        with self.assertRaises(ValueError):
            commit_wkv.prepare_dspark_commit_wkv(self.linears, "invalid")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            x.add_(1)
            with self.assertRaisesRegex(RuntimeError, "before capture"):
                self.prepare("stacked")
        torch.cuda.current_stream().wait_stream(stream)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DSparkCommitWKVSM90Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (9, 0):
            raise unittest.SkipTest("SM90 shared-quantization contract")
        cls.linears = make_linears()

    def test_auto_reuses_one_native_group32_quantization(self):
        from rtp_llm.models_py.modules.dsv41 import _sm90_quant_scale_fused

        prepared = commit_wkv.prepare_dspark_commit_wkv(self.linears, "auto")
        self.assertIsNotNone(prepared)
        self.assertTrue(prepared.sm90)
        self.assertIsNone(prepared.stacked)
        for rows in (1, 4, 16):
            x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
            expected = [linear(x) for linear in self.linears]
            with patch.object(
                _sm90_quant_scale_fused,
                "quantize_g32",
                wraps=_sm90_quant_scale_fused.quantize_g32,
            ) as quantize, patch.object(
                V41MXFP8Linear,
                "forward_quantized",
                side_effect=AssertionError("SM90 native scales must bypass wrapper"),
            ):
                actual = prepared.project(x)
            quantize.assert_called_once()
            self.assertIsNotNone(actual)
            for got, wanted in zip(actual, expected):
                exact(self, got, wanted)

        x = torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            prepared.project(x)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = prepared.project(x)
        with torch.cuda.stream(stream):
            for _ in range(3):
                x.normal_()
                graph.replay()
                for actual, linear in zip(captured, self.linears):
                    exact(self, actual, linear(x))
        torch.cuda.current_stream().wait_stream(stream)

        with self.assertRaisesRegex(ValueError, "shared, not stacked"):
            commit_wkv.prepare_dspark_commit_wkv(self.linears, "stacked")


def _make_commit_attention(v4_layout=None, device="cpu"):
    from rtp_llm.models_py.modules.dsv41.fp8 import attention as base
    from rtp_llm.models_py.modules.dsv41.fp8 import attention_v41_commit as commit
    from rtp_llm.utils.model_weight import W

    weights = {
        W.v4_attn_wkv_w: torch.empty(1, device=device),
        W.v4_attn_wkv_s: torch.empty(1, device=device),
        W.v4_attn_kv_norm: torch.ones(512, device=device),
    }
    selection = (
        nullcontext()
        if v4_layout is None
        else patch.object(
            commit, "use_v4_kv_cache", return_value=v4_layout, create=True
        )
    )
    # Keep the real minimal constructor and pool helpers; only projection
    # preparation and RoPE allocation are irrelevant to these cache contracts.
    with selection, patch.object(
        base, "_v4_fp8_linear", return_value=torch.nn.Identity()
    ) as linear, patch.object(
        base, "precompute_freqs_cis", return_value=torch.empty(0)
    ):
        attn = commit.CommitOnlyAttentionV41FP8(
            layer_id=0,
            dim=5120,
            n_heads=64,
            q_lora_rank=1024,
            head_dim=512,
            rope_head_dim=64,
            o_lora_rank=1024,
            o_groups=8,
            window_size=128,
            compress_ratio=0,
            compress_rope_theta=10000.0,
            rope_theta=10000.0,
            rope_factor=1.0,
            beta_fast=32,
            beta_slow=1,
            original_seq_len=0,
            max_batch_size=1,
            max_seq_len=1024,
            index_n_heads=1,
            index_head_dim=128,
            index_topk=1,
            layer_weights=weights,
        )
        linear.assert_called_once()
    return attn


def _bind_commit_cache(attn, raw, entries, cp_size, tag="decoder_swa_kv"):
    attn._swa_cache_region = tag
    attn._cp_ctx = SimpleNamespace(
        cp_size=cp_size, cp_rank=0, kv_cache_sharded=cp_size > 1
    )
    attn._kv_cache = SimpleNamespace(
        get_entries_per_block=Mock(
            side_effect=lambda requested: entries if requested == tag else 128
        ),
        get_layer_cache=Mock(return_value=SimpleNamespace(kv_cache_base=raw)),
    )


class DSparkCommitCacheTest(unittest.TestCase):
    def test_pool_abi_matches_allocator_without_query_weights(self):
        from rtp_llm.models.dsv41_kv_cache import build_v41_kv_cache_spec_descs
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV

        for v4_layout, width in ((True, 584), (False, 528)):
            with self.subTest(v4_layout=v4_layout):
                attn = _make_commit_attention(v4_layout)
                desc = build_v41_kv_cache_spec_descs(
                    1,
                    [0],
                    [],
                    512,
                    bounded_replay=True,
                    draft=True,
                    v4_kv_layout=v4_layout,
                )[0][0]
                self.assertEqual(attn._pool_spec[SWA_KV], (torch.uint8, width))
                self.assertEqual(desc.entry_elems, width)
                self.assertEqual(desc.tag, "decoder_swa_kv")
                self.assertIsNone(attn.wq_a)
                self.assertIsNone(attn.wo_b)

    def test_logical_ring_matches_decode_despite_padding_and_cp_slices(self):
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8

        for v4_layout, width, stride in ((True, 584, 79488), (False, 528, 84480)):
            for cp_size in (1, 8):
                for tag in (SWA_KV, "decoder_swa_kv"):
                    with self.subTest(layout=v4_layout, cp_size=cp_size, tag=tag):
                        attn = _make_commit_attention(v4_layout)
                        raw = torch.zeros(4, stride // cp_size, dtype=torch.uint8)
                        _bind_commit_cache(attn, raw, 136, cp_size, tag)
                        self.assertEqual(attn._swa_entries_per_block(), 136)
                        self.assertEqual(
                            attn._swa_entries_per_block(),
                            AttentionV41FP8._swa_entries_per_block(attn),
                        )
                        attn._kv_cache.get_entries_per_block.assert_called_with(tag)
                        view = attn._pool_view_3d_fp8(SWA_KV)
                        self.assertEqual(view.shape[-1], width)
                        self.assertEqual(view.stride(0), stride // cp_size)
                        self.assertEqual(
                            view.shape[1], min(136, stride // cp_size // width)
                        )
                        attn._kv_cache.get_layer_cache.assert_called_with(0, tag)

    def test_cp_missing_logical_metadata_fails_instead_of_guessing(self):
        for v4_layout, stride in ((True, 79488), (False, 84480)):
            for entries in (None, 0, -1):
                with self.subTest(layout=v4_layout, entries=entries):
                    attn = _make_commit_attention(v4_layout)
                    _bind_commit_cache(
                        attn, torch.zeros(4, stride // 8, dtype=torch.uint8), entries, 8
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "logical entries_per_block.*decoder_swa_kv"
                    ):
                        attn._swa_entries_per_block()
                    attn._kv_cache.get_entries_per_block = None
                    with self.assertRaisesRegex(
                        RuntimeError, "logical entries_per_block"
                    ):
                        attn._swa_entries_per_block()

    def test_non_cp_legacy_geometry_and_unbound_cache(self):
        for v4_layout, width in ((True, 584), (False, 528)):
            with self.subTest(layout=v4_layout):
                attn = _make_commit_attention(v4_layout)
                self.assertEqual(attn._swa_entries_per_block(), 0)
                _bind_commit_cache(
                    attn, torch.zeros(4, 132 * width, dtype=torch.uint8), None, 1
                )
                attn._kv_cache.get_entries_per_block = None
                self.assertEqual(attn._swa_entries_per_block(), 132)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DSparkCommitCacheCudaTest(unittest.TestCase):
    def test_planar_cp_partial_updates_preserve_entries_and_padding(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _swa_kv_insert_triton as planar
        from rtp_llm.models_py.modules.dsv41.fp8._swa_cp_byte_sliced import (
            build_cp_byte_sliced_slot_compaction,
        )

        for entries, cp_size in ((136, 8), (128, 3), (130, 5)):
            with self.subTest(entries=entries, cp_size=cp_size):
                alignment = 576 * cp_size
                stride = (entries * 584 + alignment - 1) // alignment * alignment
                initial = torch.full(
                    (5, stride), 173, dtype=torch.uint8, device="cuda"
                )
                expected = initial.clone()
                view = expected.as_strided((5, entries, 584), (stride, 584, 1))
                shards = [
                    part.contiguous().clone() for part in initial.chunk(cp_size, dim=1)
                ]
                for start, end in ((0, 128), (128, 131), (131, 140)):
                    positions = torch.arange(start, end, device="cuda")
                    slots = torch.cat(
                        (
                            3 * entries + positions % entries,
                            entries + positions % entries,
                            positions.new_tensor([-1]),
                        )
                    )
                    indices = torch.arange(slots.numel() * 512, device="cuda")
                    keys = (
                        (((indices * 73 + 19 + start) % 4093 - 2046).float() / 256)
                        .reshape(-1, 512)
                        .bfloat16()
                    )
                    compaction = build_cp_byte_sliced_slot_compaction(
                        slots, entries, 5, "dspark_partial_page_test", "skip_minus_one"
                    )
                    planar.quantize_and_insert_k_cache(keys, view, slots)
                    for rank, raw in enumerate(shards):
                        planar.quantize_and_insert_k_cache_cp_byte_sliced(
                            keys, raw, slots, entries, rank, cp_size, compaction
                        )
                    torch.cuda.synchronize()
                    actual = torch.cat(shards, dim=1)
                    exact(self, actual, expected)
                    exact(self, actual[[0, 2, 4]], initial[[0, 2, 4]])
                    exact(self, actual[:, entries * 584 :], initial[:, entries * 584 :])

                # A skipped-only update must not touch any page or padding.
                slots = torch.tensor([-1], device="cuda", dtype=torch.long)
                compaction = build_cp_byte_sliced_slot_compaction(
                    slots, entries, 5, "dspark_skip_page_test", "skip_minus_one"
                )
                for rank, raw in enumerate(shards):
                    planar.quantize_and_insert_k_cache_cp_byte_sliced(
                        keys[:1], raw, slots, entries, rank, cp_size, compaction
                    )
                exact(self, torch.cat(shards, dim=1), expected)

    def test_cp8_prefill_bytes_match_decode_codec_across_ring_wraps(self):
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
        from rtp_llm.models_py.modules.dsv41.fp8 import _swa_kv_insert_triton as planar
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_swa_triton as compact
        from rtp_llm.models_py.modules.dsv41.fp8._swa_cp_byte_sliced import (
            build_cp_byte_sliced_slot_compaction,
        )
        from rtp_llm.models_py.modules.dsv41.kv_cache_layout import use_v4_kv_cache

        attn = _make_commit_attention(device="cuda")
        planar_decode = use_v4_kv_cache(torch.device("cuda"))
        width, alignment = (584, 576) if planar_decode else (528, 16896)
        entries, cp_size, blocks = 136, 8, 4
        stride = (entries * width + alignment - 1) // alignment * alignment
        expected_codec = planar if planar_decode else compact
        dense_writer = (
            planar.quantize_and_insert_k_cache
            if planar_decode
            else compact.quantize_and_insert_swa_k_cache
        )
        for end in (64, 128, 136, 250, 1024):
            with self.subTest(context_end=end):
                positions = torch.arange(max(0, end - 128), end, device="cuda")
                indices = torch.arange(
                    (positions.numel() + 1) * 512, device="cuda", dtype=torch.int64
                )
                keys = (
                    (((indices * 73 + 19) % 4093 - 2046).float() / 256)
                    .reshape(-1, 512)
                    .bfloat16()
                )
                shards = []
                for rank in range(cp_size):
                    raw = torch.zeros(
                        blocks, stride // cp_size, dtype=torch.uint8, device="cuda"
                    )
                    _bind_commit_cache(attn, raw, entries, cp_size)
                    attn._cp_ctx.cp_rank = rank
                    producer_entries = attn._swa_entries_per_block()
                    pool = attn._pool_view_3d_fp8(SWA_KV)
                    self.assertEqual(pool.shape[-1], width)
                    self.assertEqual(producer_entries, entries)
                    slots = torch.cat(
                        (
                            2 * producer_entries + positions % producer_entries,
                            positions.new_tensor([-1]),
                        )
                    )
                    compaction = build_cp_byte_sliced_slot_compaction(
                        slots,
                        producer_entries,
                        blocks,
                        "dspark_cache_test",
                        "skip_minus_one",
                    )
                    # This is the same pool-view dispatch as forward_commit.
                    producer_codec = compact if pool.shape[-1] == 528 else planar
                    self.assertIs(producer_codec, expected_codec)
                    producer_codec.quantize_and_insert_k_cache_cp_byte_sliced(
                        keys,
                        attn._pool_raw_u8(SWA_KV),
                        slots,
                        producer_entries,
                        rank,
                        cp_size,
                        compaction,
                    )
                    shards.append(raw)
                actual = torch.cat(shards, dim=1)
                expected = torch.zeros_like(actual)
                view = expected.as_strided(
                    (blocks, entries, width), (stride, width, 1)
                )
                expected_slots = torch.cat(
                    (2 * entries + positions % entries, positions.new_tensor([-1]))
                )
                dense_writer(keys, view, expected_slots)
                torch.cuda.synchronize()
                exact(self, actual, expected)
                self.assertEqual(torch.count_nonzero(actual[:2]).item(), 0)
                self.assertEqual(torch.count_nonzero(actual[3:]).item(), 0)


if __name__ == "__main__":
    unittest.main()
