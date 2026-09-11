"""Fixed CMP selection and dependency contracts (no engine or model loading)."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from rtp_llm.models_py.modules.hybrid import hy4_cmp as bridge


def linear(n, k, dtype=torch.float8_e4m3fn):
    weight = torch.empty(n, k, device="cuda", dtype=dtype)
    scale = torch.empty(k // 128, n, device="cuda", dtype=torch.int32).t()
    return NS(
        weight=weight,
        bias=None,
        supports_out=True,
        input_quant_group_size=32,
        input_quant_scale_ue8m0=True,
        _packed_weight_scale=lambda: scale,
        weight_scale=scale,
    )


def native_provider(device):
    return NS(
        output_bmm_gate_quant=Mock(),
        router_proj=Mock(),
        router_topk=Mock(),
        qkv_a_head_gate=Mock(),
        qkv_a_head_gate_partials=Mock(),
        canonicalize_topk=Mock(),
        pack_head_gate_weight=lambda w: torch.empty(96, 64, 32, device=device),
        q_b_proj=Mock(),
        qkv_post=Mock(),
        mtp_norm_quant=Mock(),
        target_norm_quant=Mock(),
        indexer_k_cache=Mock(),
        indexer_q_proj=Mock(),
        indexer_q_post=Mock(),
        indexer_q_post_partials=Mock(),
        get_pdl=Mock(),
    )


def fixture(rows=32, model="hy_v4_mtp"):
    h = torch.empty(rows, 6144, device="cuda", dtype=torch.bfloat16)
    norm = lambda n: NS(
        weight=torch.empty(n, device="cuda", dtype=torch.bfloat16),
        variance_epsilon=1e-6,
    )
    op = NS(
        is_neox_style=False,
        cos_sin_cache=torch.empty(1024, 64, device="cuda"),
        _kv_cache_blocks=lambda c: c.kv_scale_base,
    )
    i = NS(
        use_hadamard=False,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        wk_bf16_input=True,
        bf16_compute=True,
        wq_b=linear(4096, 2048),
        wk=linear(128, 6144, torch.bfloat16),
        k_norm=NS(
            supports_out=True,
            weight=torch.empty(128, device="cuda", dtype=torch.bfloat16),
            beta=torch.empty(128, device="cuda", dtype=torch.bfloat16),
            variance_epsilon=1e-6,
        ),
        indexer_op=op,
        weights_proj=linear(32, 6144, torch.float32),
    )
    # Production raw head weights can be a transpose view.
    i.weights_proj.weight = torch.empty(6144, 32, device="cuda").t()
    a = NS(
        num_heads=64,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        token_per_block=64,
        gating_type="elementwise",
        _fuse_q_a_norm_mode="mxfp8",
        _fuse_kv_a_norm=True,
        _reuse_mxfp8_hidden_quant=True,
        _fuse_gated_mla_quant=True,
        _gated_mla_quant_group_size=32,
        fused_qkv_a_proj=linear(2624, 6144),
        q_b_proj=linear(16384, 2048),
        o_proj=linear(6144, 16384),
        gate_proj=linear(16384, 6144, torch.bfloat16),
        q_a_layernorm=norm(2048),
        kv_a_layernorm=norm(512),
        indexer=i,
        reuse_topk_indices=False,
        layer_idx=0,
        attn_sink=torch.empty(64, device="cuda"),
    )
    cmp = bridge.Hy4Cmp(
        config=NS(model_type=model, gen_num_per_cycle=3),
        parallelism_config=NS(tp_size=1, get_attn_tp_size=lambda: 1),
        self_attn=a,
    )
    params = NS(
        positions_d=torch.empty(rows, device="cuda", dtype=torch.int64),
        slot_mapping=torch.empty(rows, device="cuda", dtype=torch.int64),
        expanded_seq_lens=torch.empty(rows, device="cuda", dtype=torch.int32),
        kvlen_d=torch.empty(rows, device="cuda", dtype=torch.int32),
    )
    inputs = NS(
        is_prefill=False,
        is_target_verify=False,
        is_draft_extend=False,
        kv_cache_kernel_block_id_device_by_group=None,
        kv_cache_layer_to_group=None,
        decode_cu_seqlens_d=torch.empty(rows + 1, device="cuda", dtype=torch.int32),
        kv_cache_kernel_block_id_device=torch.empty(
            rows, 16, device="cuda", dtype=torch.int32
        ),
    )
    cache = NS(
        kv_cache_base=torch.empty(16, 64, 656, device="cuda", dtype=torch.uint8),
        kv_scale_base=torch.empty(16, 64, 132, device="cuda", dtype=torch.uint8),
    )
    fmha = NS(
        hy4_output_weight=lambda _: torch.empty(
            64, 512, 256, device="cuda", dtype=torch.bfloat16
        ),
        is_sparse=lambda: True,
        cp_params=None,
        supports_topk_late_binding=True,
        can_fuse_kv_norm_cache=lambda x, w: True,
        attn_inputs=inputs,
        fmha_params=params,
        _cos_sin_cache=op.cos_sin_cache,
        _is_neox_style=False,
    )
    return cmp, h, fmha, cache


class FixedCmpContractTest(unittest.TestCase):
    def setUp(self):
        capability = patch("torch.cuda.get_device_capability", return_value=(10, 3))
        capability.start()
        self.addCleanup(capability.stop)
        self.mode = FakeTensorMode()
        self.mode.__enter__()
        self.addCleanup(self.mode.__exit__, None, None, None)

    def test_output_epilogue_uses_fused_native_path(self):
        cmp, hidden, fmha, _ = fixture()
        cmp._ops = native_provider(hidden.device)
        cmp._output_bmm_weight = torch.empty(
            64, 256, 512, device="cuda", dtype=torch.bfloat16
        )
        latent = torch.empty(32, 64, 512, device="cuda", dtype=torch.bfloat16)
        gate = torch.empty(32, 16384, device="cuda", dtype=torch.bfloat16)
        ids = torch.empty(32, 2048, device="cuda", dtype=torch.int32)
        fmha.finish_hy4_native_attention = Mock(return_value=latent)
        cmp.self_attn.o_proj = Mock(return_value=hidden)
        output, returned_ids = cmp._finish_attention(None, gate, ids, fmha)
        self.assertIs(output, hidden)
        self.assertIs(returned_ids, ids)
        args = cmp._ops.output_bmm_gate_quant.call_args.args
        self.assertIs(args[0], latent)
        self.assertIs(args[1], cmp._output_bmm_weight)
        self.assertIs(args[2], gate)
        self.assertEqual(args[3].shape, (32, 16384))
        self.assertEqual(args[4].stride(), (1, 32))
        self.assertIs(cmp.self_attn.o_proj.call_args.kwargs["input_scales"], args[4])

    def test_router_uses_fp32_native_weight_without_input_cast(self):
        cmp, hidden, _, _ = fixture()
        cmp._ops = native_provider(hidden.device)
        cmp._router_weight = torch.empty(256, 6144, device="cuda", dtype=torch.float32)
        cmp.mlp = NS(correction_bias=torch.empty(256, device="cuda"))
        ids = torch.empty(32, 8, device="cuda", dtype=torch.int64)
        weights = torch.empty(32, 8, device="cuda")
        cmp._prepare_router(hidden, ids, weights)
        args = cmp._ops.router_proj.call_args.args
        self.assertIs(args[0], hidden)
        self.assertIs(args[1], cmp._router_weight)
        self.assertEqual(args[2].dtype, torch.float32)
        topk_args = cmp._ops.router_topk.call_args.args
        self.assertIs(topk_args[0], args[2])
        self.assertIs(topk_args[2], ids)
        self.assertIs(topk_args[3], weights)

    def test_plan_boundaries(self):
        for rows in (1, 16, 32, 33, 64, 128, 256):
            expected = (
                bridge.Hy4CmpPlan.SMALL_T
                if rows <= 32
                else bridge.Hy4CmpPlan.NATIVE_G32
            )
            self.assertEqual(bridge.select_fixed_plan(rows, False), expected)
            self.assertEqual(
                bridge.select_fixed_plan(rows, True), bridge.Hy4CmpPlan.REUSE_TOPK
            )
        for rows in (0, 257):
            with self.assertRaises(ValueError):
                bridge.select_fixed_plan(rows, False)

    def test_actual_kv_fusion_and_each_layer_cache(self):
        cmp, h, fmha, cache = fixture()
        self.assertIsNone(cmp._dynamic_disabled_reason(h, fmha, cache))
        fmha.can_fuse_kv_norm_cache = lambda x, w: False
        self.assertIn("cannot fuse", cmp._dynamic_disabled_reason(h, fmha, cache))
        fmha.can_fuse_kv_norm_cache = lambda x, w: True
        cache.kv_scale_base = torch.empty(16, 32, 132, device="cuda", dtype=torch.uint8)
        self.assertIn("Indexer cache", cmp._dynamic_disabled_reason(h, fmha, cache))

    def test_native_rope_abi_rejects_bf16_and_neox(self):
        cmp, h, fmha, cache = fixture()
        fmha._cos_sin_cache = fmha._cos_sin_cache.bfloat16()
        self.assertEqual(
            cmp._dynamic_disabled_reason(h, fmha, cache), "invalid MLA RoPE cache"
        )
        fmha._cos_sin_cache = fmha._cos_sin_cache.float()
        fmha._is_neox_style = True
        self.assertEqual(
            cmp._dynamic_disabled_reason(h, fmha, cache), "invalid MLA RoPE cache"
        )

    def test_complete_mtp_preflight_accepts_fixed_weights_and_producers(self):
        cmp, h, fmha, cache = fixture()
        device = h.device
        c = cmp.config
        c.moe_n_group = c.moe_topk_group = 1
        c.has_moe_norm = True
        c.routed_scaling_factor = 2.827
        mega = NS(
            **{
                name: torch.empty(1, device=device)
                for name in ("_mega_l1_w", "_mega_l1_sf", "_mega_l2_w", "_mega_l2_sf")
            }
        )
        cmp.mlp = NS(
            _hy4_mega_moe_prepack=True,
            num_experts=256,
            top_k=8,
            gate_weight=torch.empty(6144, 256, device=device),
            correction_bias=torch.empty(256, device=device),
            shared_expert=NS(
                accepts_mxfp8_input=True,
                up_proj=linear(4096, 6144),
                down_proj=linear(6144, 2048),
            ),
            fused_moe=NS(
                mega_moe=mega,
                prepacked_input_views=lambda rows: (
                    torch.empty(rows, 6144, device=device, dtype=torch.float8_e4m3fn),
                    torch.empty(rows, 48, device=device, dtype=torch.int32),
                    torch.empty(rows, 8, device=device, dtype=torch.int64),
                    torch.empty(rows, 8, device=device, dtype=torch.float32),
                ),
            ),
        )
        norm = NS(weight=torch.empty(6144, device=device, dtype=torch.bfloat16))
        layer = NS(
            hy4_cmp=cmp,
            input_layernorm=norm,
            post_attention_layernorm=norm,
            _fuse_hy4_cmp_input_norm_quant=True,
            _fuse_hy4_cmp_post_norm_quant_moe=True,
        )
        ops = native_provider(device)
        with patch.object(bridge, "_load_hy4_ops", return_value=ops), patch.object(
            bridge, "_is_capturing", return_value=False
        ), patch.object(cmp, "_side_streams"), patch.object(cmp, "_new_events"):
            self.assertTrue(
                bridge.should_enable_hy4_cmp(
                    [layer], 1, h, fmha, NS(get_layer_cache=lambda _: cache), residual=h
                )
            )
            self.assertTrue(cmp._initialized)
            self.assertEqual(cmp._plan, bridge.Hy4CmpPlan.SMALL_T)

    def test_decode_metadata_and_later_group_table(self):
        cmp, h, fmha, cache = fixture()
        fmha.fmha_params.kvlen_d = torch.empty(1, device="cuda", dtype=torch.int32)
        self.assertIn("metadata", cmp._dynamic_disabled_reason(h, fmha, cache))
        fmha.fmha_params.kvlen_d = torch.empty(32, device="cuda", dtype=torch.int32)
        fmha.attn_inputs.kv_cache_kernel_block_id_device_by_group = [
            torch.empty(32, 16, device="cuda", dtype=torch.int64)
        ]
        with patch.object(bridge, "_load_hy4_ops") as load:
            self.assertFalse(
                bridge.should_enable_hy4_cmp(
                    [NS(hy4_cmp=cmp)],
                    1,
                    h,
                    fmha,
                    NS(get_layer_cache=lambda _: cache),
                    residual=h,
                )
            )
            load.assert_not_called()

    def test_invalid_weight_falls_back_before_provider_loading(self):
        cmp, h, fmha, cache = fixture()
        cmp.self_attn.q_b_proj.weight = torch.empty(
            16384, 2048, device="cuda", dtype=torch.bfloat16
        )
        with patch.object(
            bridge, "_producer_supported", return_value=True
        ), patch.object(bridge, "_moe_supported", return_value=True), patch.object(
            bridge, "_load_hy4_ops"
        ) as load:
            self.assertFalse(
                bridge.should_enable_hy4_cmp(
                    [NS(hy4_cmp=cmp)],
                    1,
                    h,
                    fmha,
                    NS(get_layer_cache=lambda _: cache),
                    residual=h,
                )
            )
            load.assert_not_called()

    def test_draft_width_uses_kernel_table_and_clone_flag(self):
        cmp, h, fmha, cache = fixture(rows=16)
        fmha.attn_inputs.is_prefill = True
        fmha.attn_inputs.kv_cache_kernel_block_id_device = torch.empty(
            4, 16, device="cuda", dtype=torch.int32
        )
        fmha.attn_inputs.kv_cache_block_id_device = torch.empty(
            16, 4, device="cuda", dtype=torch.int32
        )
        self.assertIsNotNone(cmp._dynamic_disabled_reason(h, fmha, cache))
        clone = cmp.clone_for_cuda_graph(self_attn=cmp.self_attn, draft_prefill=True)
        self.assertIsNone(clone._dynamic_disabled_reason(h, fmha, cache))
        clone.config.gen_num_per_cycle = 1
        self.assertIsNotNone(clone._dynamic_disabled_reason(h, fmha, cache))

    def test_transposed_head_weight_and_scale_stride(self):
        cmp, _, _, _ = fixture()
        ops = native_provider(cmp.self_attn.fused_qkv_a_proj.weight.device)
        with patch.object(bridge, "_is_capturing", return_value=False), patch(
            "torch.cuda.get_device_capability", return_value=(10, 3)
        ):
            cmp.initialize_for_cmp(ops)
            self.assertTrue(cmp._initialized)
            self.assertTrue(cmp._small_head_weight.is_contiguous())
            clone = cmp.clone_for_cuda_graph(self_attn=cmp.self_attn)
            self.assertIs(clone._packed_head_weight, cmp._packed_head_weight)
            self.assertIsNone(clone._events)
            bad, _, _, _ = fixture()
            bad.self_attn.q_b_proj._packed_weight_scale = lambda: torch.empty(
                16384, 16, device="cuda", dtype=torch.int32
            )
            with self.assertRaisesRegex(ValueError, "column-major"):
                bad.initialize_for_cmp(ops)
            self.assertFalse(bad._initialized)

    def test_all_layers_reject_before_native_initialization(self):
        a, h, fmha, cache = fixture()
        b, _, _, bad_cache = fixture()
        bad_cache.kv_cache_base = torch.empty(
            16, 64, 576, device="cuda", dtype=torch.bfloat16
        )
        cache_owner = NS(get_layer_cache=lambda idx: [cache, bad_cache][idx])
        with patch.object(
            bridge, "_producer_supported", return_value=True
        ), patch.object(bridge, "_moe_supported", return_value=True), patch.object(
            bridge, "_load_hy4_ops"
        ) as load:
            self.assertFalse(
                bridge.should_enable_hy4_cmp(
                    [NS(hy4_cmp=a), NS(hy4_cmp=b)], 2, h, fmha, cache_owner, residual=h
                )
            )
            load.assert_not_called()

    def test_missing_provider_is_error_and_capture_requires_warmup(self):
        cmp, h, fmha, cache = fixture()
        owner = NS(get_layer_cache=lambda _: cache)
        with patch.object(
            bridge, "_producer_supported", return_value=True
        ), patch.object(bridge, "_moe_supported", return_value=True):
            with patch.object(
                bridge, "_is_capturing", return_value=False
            ), patch.object(
                bridge, "_load_hy4_ops", side_effect=ImportError("missing native")
            ):
                with self.assertRaises(ImportError):
                    bridge.should_enable_hy4_cmp(
                        [NS(hy4_cmp=cmp)], 1, h, fmha, owner, residual=h
                    )
            with patch.object(bridge, "_is_capturing", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "warmed"):
                    bridge.should_enable_hy4_cmp(
                        [NS(hy4_cmp=cmp)], 1, h, fmha, owner, residual=h
                    )

    def test_reuse_requires_seed_before_submission(self):
        cmp, h, fmha, cache = fixture()
        owner = NS(get_layer_cache=lambda _: cache)
        with self.assertRaisesRegex(RuntimeError, "seed"):
            bridge.should_enable_hy4_cmp(
                [NS(hy4_cmp=cmp)],
                1,
                h,
                fmha,
                owner,
                force_reuse_topk_indices=True,
                residual=h,
            )
        self.assertIsNone(
            cmp.allocate_raw_head_gate_output(h, force_reuse_topk_indices=True)
        )
        self.assertIsNone(cmp._events)


class FixedCmpDagTest(unittest.TestCase):
    setUp = FixedCmpContractTest.setUp

    def test_three_plans_and_long_kv_dependencies(self):
        for rows, reuse, deferred in [
            (1, False, False),
            (33, False, False),
            (33, False, True),
            (33, True, False),
        ]:
            with self.subTest(rows=rows, reuse=reuse, deferred=deferred):
                cmp, h, fmha, cache = fixture(rows)
                log = []
                current = ["caller"]

                class Stream:
                    def __init__(self, name):
                        self.name = name

                    def wait_event(self, event):
                        log.append((self.name, "wait", event.name))

                class Event:
                    def __init__(self, name):
                        self.name = name
                        self.cuda_event = self

                    def record(self, stream=None):
                        log.append(
                            (stream.name if stream else current[0], "record", self.name)
                        )

                @contextmanager
                def stream_context(stream):
                    previous = current[0]
                    current[0] = stream.name
                    yield
                    current[0] = previous

                streams = tuple(Stream(s) for s in ("main", "index", "index_q"))
                caller = Stream("caller")
                cmp._side_streams = Mock(return_value=streams)
                cmp._events = bridge._Events(
                    *(Event(n) for n in bridge._Events.__annotations__)
                )
                cmp._serialize_score_after_q_path = lambda _: deferred
                make = lambda shape, dtype=torch.bfloat16: torch.empty(
                    shape, device="cuda", dtype=dtype
                )
                raw = make((rows, 32), torch.float32)
                buffers = dict(
                    index_q=make((rows, 32, 128)),
                    index_fp8=make((rows, 32, 128), torch.float8_e4m3fn),
                    index_scale=make((rows, 32), torch.float32),
                    head_weights=make((rows, 32), torch.float32),
                    raw_gate=raw,
                    gate_partials=None,
                )
                cmp._allocate_buffers = lambda *a: buffers
                q_inputs = (
                    make((rows, 2048), torch.float8_e4m3fn),
                    make((16, rows), torch.int32).t(),
                )

                def qkv(*args, native_gate=False, notify_event=0):
                    log.append((current[0], "qkv", native_gate))
                    if notify_event:
                        log.append((current[0], "notify", notify_event.name))
                    return q_inputs

                cmp._qkv_a = qkv

                def main_query(*a):
                    log.append((current[0], "main_query"))
                    return NS(), make((rows, 16384))

                cmp._main_query = main_query

                cmp._prepare_score = lambda *a: None
                cmp._positions = fmha.fmha_params.positions_d.to(torch.int32)
                cmp._index_k_weight = cmp.self_attn.indexer.wk.weight
                cmp._weight_scales["index_q"] = q_inputs[1]
                cmp._small_raw_gate = lambda *a: log.append((current[0], "raw_gate"))

                def post(*args):
                    log.append((current[0], "post", "q"))
                    log.append((current[0], "notify", args[-1].name))

                cmp._ops = NS(
                    indexer_k_cache=lambda *a: log.append((current[0], "post", "k")),
                    indexer_q_proj=lambda *a: log.append((current[0], "Q")),
                    indexer_q_post=post,
                    indexer_q_post_partials=post,
                )

                def score(*a):
                    log.append((current[0], "score"))
                    return make((rows, 2048), torch.int32)

                cmp._score_topk = score
                seed = make((rows, 2048), torch.int32)

                def produce(notify_event):
                    log.append((current[0], "input_producer"))
                    if notify_event:
                        log.append((current[0], "notify", notify_event.name))
                    return h, None, None

                with patch("torch.cuda.current_stream", return_value=caller), patch(
                    "torch.cuda.stream", side_effect=stream_context
                ), patch.object(
                    torch.Tensor,
                    "record_stream",
                    lambda t, s: log.append((s.name, "lifetime")),
                ):
                    _, _, topk = cmp.mla_prologue(
                        h,
                        fmha,
                        cache,
                        seed,
                        bridge.select_fixed_plan(rows, reuse),
                        produce,
                        (h,),
                        raw,
                    )
                if reuse:
                    self.assertIs(topk, seed)
                    cmp._side_streams.assert_not_called()
                    self.assertEqual(
                        log,
                        [
                            ("caller", "input_producer"),
                            ("caller", "qkv", False),
                            ("caller", "main_query"),
                        ],
                    )
                else:
                    self.assertLess(
                        log.index(("main", "input_producer")),
                        log.index(("main", "notify", "frontend_ready")),
                    )
                    self.assertNotIn(("caller", "input_producer"), log)
                    self.assertIn(("index_q", "Q"), log)
                    self.assertLess(
                        log.index(("main", "notify", "q_inputs_ready")),
                        log.index(("index_q", "Q")),
                    )
                    self.assertNotIn(("main", "record", "frontend_ready"), log)
                    self.assertEqual(
                        ("main", "record", "frontend_complete") in log, rows <= 32
                    )
                    self.assertLess(
                        log.index(("index", "post", "k")), log.index(("index", "score"))
                    )
                    self.assertLess(
                        log.index(("index_q", "post", "q")),
                        log.index(("index_q", "notify", "indexer_q_ready")),
                    )
                    self.assertLess(
                        log.index(("index", "wait", "indexer_q_ready")),
                        log.index(("index", "score")),
                    )
                    self.assertIn(("caller", "wait", "side_streams_complete"), log)
                    self.assertEqual(log[-1], ("caller", "lifetime"))
                    self.assertIn(("main", "qkv", rows > 32), log)
                    self.assertEqual(("index_q", "raw_gate") in log, rows <= 32)
                    if deferred:
                        self.assertLess(
                            log.index(("main", "main_query")),
                            log.index(("index", "wait", "q_path_complete")),
                        )
                    else:
                        self.assertLess(
                            log.index(("index", "score")),
                            log.index(("main", "main_query")),
                        )


if __name__ == "__main__":
    unittest.main()
