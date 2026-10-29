import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.config.mega_moe import MegaMoeExecutionPolicy
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    FusedMoe,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeFusedBackend,
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
    Fp8Fp4Router,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_weights import (
    prepare_shared_expert_weights,
)
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W

PACKAGE = "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors"
BACKENDS = (
    ("mega_moe", MegaMoeExecutor, "get_symm_buffer_for_mega_moe"),
    ("mega_moe_fp8", MegaMoeExecutor, "get_symm_buffer_for_mega_moe_fp8"),
    ("mega_moe_se", MegaMoeSEExecutor, "get_symm_buffer_for_mega_moe"),
    ("mega_moe_fp8_se", MegaMoeSEExecutor, "get_symm_buffer_for_mega_moe_fp8"),
    ("mega_moe_fused", MegaMoeFusedBackend, "get_symm_buffer_for_mega_moe_fused"),
)
CHUNK_MODULE = "rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer"


class MegaMoeLifecycleTest(unittest.TestCase):
    def test_routed_arena_is_cached_per_collective_group(self):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe import buffer

        group = SimpleNamespace(size=lambda: 4)
        arena = SimpleNamespace(buffer=torch.empty(1))
        params = dict(
            group=group,
            num_experts=8,
            num_max_tokens_per_rank=16,
            num_topk=2,
            hidden=128,
            intermediate_hidden=256,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        with patch.dict(buffer._MEGA_BUF_CACHE, {}, clear=True):
            with patch(
                "deep_gemm.get_symm_buffer_for_mega_moe", return_value=arena
            ) as allocate:
                generic = buffer._get_or_create_mega_buf(**params)
                cached = buffer._get_or_create_mega_buf(**params)
                self.assertIs(cached, generic)
                allocate.assert_called_once()
                # Another group must retain a separate collective arena.
                buffer._get_or_create_mega_buf(
                    **{**params, "group": SimpleNamespace(size=lambda: 4)}
                )
                self.assertEqual(allocate.call_count, 2)

    def test_warmup_retains_automatic_capacity_buckets(self):
        for backend_type in (MegaMoeExecutor, MegaMoeSEExecutor):
            with self.subTest(backend=backend_type.__name__):
                model = self.backend(backend_type)
                model.cfg = SimpleNamespace(
                    **{
                        **vars(model.cfg),
                        "ep_size": 4,
                        "n_routed_experts": 256,
                        "n_local_experts": 64,
                        "n_activated_experts": 8,
                        "moe_inter_dim": 2048,
                        "max_tokens_per_rank": 64,
                    }
                )
                self.assertEqual(model._resolve_jit_warmup_token_counts(132), [1, 64])

    def backend(self, backend_type, strategy=None):
        if strategy is None:
            strategy = next(name for name, cls, _ in BACKENDS if cls is backend_type)
        cfg = SimpleNamespace(
            layer_id=0,
            moe_strategy=strategy,
            dim=8,
            hidden_size=8,
            moe_inter_dim=4,
            n_routed_experts=4,
            n_activated_experts=2,
            ep_size=1,
            ep_rank=0,
            tp_size=1,
            max_tokens_per_rank=8,
            n_local_experts=4,
            local_expert_start=0,
            local_expert_end=4,
            activation_clamp=5.0,
            fast_math=False,
            swiglu_limit=5.0,
            route_scale=1.0,
            warmup_include_capacity=True,
            max_generate_batch_size=8,
            model_config=SimpleNamespace(max_seq_len=128, gen_num_per_cycle=0),
            parallelism_config=SimpleNamespace(
                role_type=RoleType.PDFUSION,
                tp_size=1,
                prefill_cp_config=SimpleNamespace(is_enabled=lambda: False),
                get_ffn_tp_size=lambda: 1,
            ),
            moe_config=SimpleNamespace(ll_num_max_token=0),
            mega_moe_policy=MegaMoeExecutionPolicy(activation_clamp=5.0),
        )
        model = backend_type.__new__(backend_type)
        nn.Module.__init__(model)
        FusedMoeExpertExecutor.__init__(model, cfg, FusedMoEQuantConfig(), {})
        model.cfg = cfg
        model._is_decode_role = False
        model._num_shared_experts = 0
        model._mega_group = object()
        model._mega_buf = SimpleNamespace(num_max_tokens_per_rank=3)
        model._mega_y = torch.ones(3, 8)
        model._mega_l1_w = torch.ones(2, 2)
        model._mega_l1_sf = torch.ones(2, 2)
        model._mega_l2_w = torch.ones(2, 2)
        model._mega_l2_sf = torch.ones(2, 2)
        model._input_packer = Mock(name="input_packer")
        model._input_packer.name = "fused"
        if backend_type in (
            MegaMoeSEExecutor,
            MegaMoeFusedBackend,
        ):
            model._shared_l1_w = torch.ones(2, 2)
            model._shared_l1_sf = torch.ones(2, 2)
            model._shared_l2_w = torch.ones(2, 2)
            model._shared_l2_sf = torch.ones(2, 2)
            model._num_shared_experts = 2
        if backend_type is MegaMoeFusedBackend:
            model._shared_mid_fp8 = torch.ones(2, 2)
            model._shared_mid_sf = torch.ones(2, 2)
        return model

    def fused_moe(self, model):
        return FusedMoe(
            Fp8Fp4Router(model.config, model.quant_config), model, 4, "mega_moe"
        )

    def test_each_clone_allocates_only_its_recipe_arena(self):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe import buffer

        for strategy, backend_type, allocator_name in BACKENDS:
            with self.subTest(backend=strategy):
                model = self.backend(backend_type, strategy)
                arena = object()
                with ExitStack() as stack:
                    stack.enter_context(
                        patch.dict(buffer._MEGA_CLONE_BUF_CACHE, {}, clear=True)
                    )
                    allocators = {
                        # The legacy fused allocator is optional in newer wheels.
                        # Its recipe still needs the same lifecycle contract.
                        name: stack.enter_context(
                            patch("deep_gemm." + name, return_value=arena, create=True)
                        )
                        for name in {name for _, _, name in BACKENDS}
                    }
                    clone = model.clone_for_cuda_graph()
                for name, allocator in allocators.items():
                    self.assertEqual(allocator.call_count, int(name == allocator_name))
                kwargs = allocators[allocator_name].call_args.kwargs
                self.assertIs(kwargs["group"], model._mega_group)
                self.assertEqual(
                    kwargs["num_max_tokens_per_rank"], model.cfg.max_tokens_per_rank
                )
                if backend_type is MegaMoeSEExecutor:
                    self.assertEqual(kwargs["num_shared_experts"], 2)
                if strategy == "mega_moe_se":
                    self.assertEqual(kwargs["mma_type"], "fp8xfp4")
                self.assertIs(type(clone), backend_type)
                self.assertIs(clone._mega_buf, arena)
                self.assertIs(clone.cfg, model.cfg)
                self.assertIs(clone.config, model.config)
                self.assertIs(clone.quant_config, model.quant_config)
                self.assertIs(clone.weights, model.weights)
                self.assertIs(clone._mega_l1_w, model._mega_l1_w)
                self.assertEqual(
                    clone.includes_shared_expert, model.includes_shared_expert
                )
                self.assertEqual(clone._num_shared_experts, model._num_shared_experts)
                self.assertNotEqual(clone._mega_y.data_ptr(), model._mega_y.data_ptr())
                if model.includes_shared_expert:
                    self.assertIs(clone._shared_l1_w, model._shared_l1_w)
                    self.assertIs(clone._shared_l2_w, model._shared_l2_w)
                if backend_type is MegaMoeFusedBackend:
                    self.assertNotEqual(
                        clone._shared_mid_fp8.data_ptr(),
                        model._shared_mid_fp8.data_ptr(),
                    )
                    self.assertNotEqual(
                        clone._shared_mid_sf.data_ptr(), model._shared_mid_sf.data_ptr()
                    )
                clone._mega_y.fill_(5)
                self.assertEqual(model._mega_y.tolist(), [[1] * 8] * 3)

    def test_ordered_shared_clones_reuse_arena_but_isolate_outputs(self):
        for strategy, backend_type, allocator_name in BACKENDS:
            with self.subTest(backend=strategy):
                model = self.backend(backend_type, strategy)
                with patch(
                    PACKAGE + ".mega_moe.get_or_create_mega_clone_buffer",
                    side_effect=AssertionError("shared clone must not allocate"),
                ):
                    clone = model.clone_for_cuda_graph(share_mega_buf=True)
                self.assertIs(clone._mega_buf, model._mega_buf)
                self.assertNotEqual(clone._mega_y.data_ptr(), model._mega_y.data_ptr())

    def test_uninitialized_shared_arena_is_rejected(self):
        for strategy, backend_type, _ in BACKENDS:
            with self.subTest(backend=strategy):
                model = self.backend(backend_type, strategy)
                model._mega_buf = None
                self.assertRaises(
                    RuntimeError, model.clone_for_cuda_graph, share_mega_buf=True
                )

    def test_prepacked_inputs_reject_a_peer_plan_requiring_multiple_chunks(self):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
            chunked_layer,
        )

        model = self.backend(MegaMoeExecutor)
        buf = model._mega_buf
        buf.x = torch.empty(3, 8)
        buf.x_sf = torch.empty(3, 1, dtype=torch.int32)
        buf.topk_idx = torch.empty(3, 2, dtype=torch.int64)
        buf.topk_weights = torch.empty(3, 2)
        moe = self.fused_moe(model)
        for peer_tokens in (3, 7):
            with self.subTest(peer_tokens=peer_tokens):
                token = chunked_layer._SYNCHRONIZED_CHUNK_TOKENS.set(peer_tokens)
                try:
                    if peer_tokens > buf.num_max_tokens_per_rank:
                        with self.assertRaisesRegex(ValueError, "chunk"):
                            moe.prepacked_input_views(2)
                    else:
                        views = moe.prepacked_input_views(2)
                        for view, source in zip(
                            views, (buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights)
                        ):
                            self.assertEqual(view.shape[0], 2)
                            self.assertEqual(view.data_ptr(), source.data_ptr())
                finally:
                    chunked_layer._SYNCHRONIZED_CHUNK_TOKENS.reset(token)

    def test_shared_recipes_defer_warmup_until_all_weights_are_ready(self):
        for strategy, backend_type, _ in BACKENDS:
            if not backend_type.includes_shared_expert:
                continue
            with self.subTest(backend=strategy):
                model = self.backend(backend_type, strategy)
                model._num_shared_experts = 0
                with patch(
                    PACKAGE + ".mega_moe._get_validated_world_ep_group",
                    return_value=model._mega_group,
                ), patch.object(
                    model, "_allocate_mega_buffer", return_value=model._mega_buf
                ) as allocate, patch.object(
                    model, "_allocate_mega_output", return_value=model._mega_y
                ), patch.object(
                    model, "_maybe_warmup_jit_once"
                ) as warmup:
                    model._setup_buffer_and_warmup()
                    self.assertEqual(
                        allocate.call_count, int(backend_type is MegaMoeFusedBackend)
                    )
                    warmup.assert_not_called()
                    model._num_shared_experts = 2
                    model._setup_buffer_and_warmup()
                    warmup.assert_not_called()
                    model.maybe_warmup_fused_shared_jit_once()
                    warmup.assert_called_once()

    def test_public_moe_rejects_deferred_tp_allreduce_before_execution(self):
        model = self.backend(MegaMoeExecutor)
        with patch.object(model, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "skip_tp_allreduce"):
                self.fused_moe(model)(
                    torch.zeros(2, 8),
                    torch.ones(2, 2),
                    torch.zeros(2, 2, dtype=torch.int64),
                    skip_tp_allreduce=True,
                )
        execute.assert_not_called()

    def test_uneven_ranks_execute_the_same_number_of_chunks(self):
        for strategy, backend_type, _ in BACKENDS:
            with self.subTest(backend=strategy):
                model = self.backend(backend_type, strategy)
                model._mega_buf.num_max_tokens_per_rank = 2
                hidden = torch.zeros(3, 8)
                with patch(
                    CHUNK_MODULE + ".synchronize_moe_chunk_tokens",
                    return_value=7,
                ), patch.object(
                    model, "forward", side_effect=lambda x, *args: x + 1
                ) as forward:
                    result = self.fused_moe(model)(
                        hidden, torch.ones(3, 2), torch.zeros(3, 2, dtype=torch.int64)
                    )
                torch.testing.assert_close(result, hidden + 1)
                self.assertEqual(
                    [call.args[0].size(0) for call in forward.call_args_list],
                    [2, 1, 0, 0],
                )

    def test_decode_overflow_fails_before_any_collective(self):
        model = self.backend(MegaMoeExecutor)
        model._mega_buf.num_max_tokens_per_rank = 2
        model._is_decode_role = True
        with patch(
            CHUNK_MODULE + ".synchronize_moe_chunk_tokens"
        ) as synchronize, patch.object(model, "forward") as forward:
            with self.assertRaisesRegex(ValueError, "max_tokens_per_rank"):
                self.fused_moe(model)(
                    torch.zeros(3, 8),
                    torch.ones(3, 2),
                    torch.zeros(3, 2, dtype=torch.int64),
                )
        synchronize.assert_not_called()
        forward.assert_not_called()

    def test_main_executor_keeps_layer_owned_chunking(self):
        model = self.backend(MegaMoeExecutor)
        model.cfg.mega_moe_policy = None
        hidden = torch.zeros(2, 8)
        with patch(
            CHUNK_MODULE + ".synchronize_moe_chunk_tokens"
        ) as synchronize, patch.object(
            model, "forward", return_value=hidden
        ) as forward:
            result = self.fused_moe(model)(
                hidden, torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.int64)
            )
        synchronize.assert_not_called()
        forward.assert_called_once()
        self.assertIs(result, hidden)

    def test_shared_weight_transform_selects_routed_backend(self):
        cfg = self.backend(MegaMoeExecutor).cfg
        w1 = torch.empty(8, 8, dtype=torch.float8_e4m3fn)
        w2 = torch.empty(8, 4, dtype=torch.float8_e4m3fn)
        sf = torch.ones(2, 2, dtype=torch.int32)
        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                with patch(
                    "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_weights.prepare_fp8_weight_scale_for_deepgemm",
                    return_value=sf,
                ):
                    with patch(
                        "deep_gemm.transform_weights_for_mega_moe",
                        return_value=((w1, sf), (w2, sf)),
                    ) as fp4_transform:
                        with patch(
                            "deep_gemm.transform_weights_for_mega_moe_fp8",
                            return_value=((w1, sf), (w2, sf)),
                            create=True,
                        ) as fp8_transform:
                            result = prepare_shared_expert_weights(
                                cfg, w1, sf, w2, sf, fp8_routed=fp8
                            )
                self.assertEqual(result.num_experts, 1)
                self.assertIs(result.l1[0], w1)
                self.assertEqual(fp4_transform.call_count, int(not fp8))
                self.assertEqual(fp8_transform.call_count, int(fp8))

    def test_shared_weight_mixed_scale_formats_are_rejected(self):
        cfg = self.backend(MegaMoeExecutor).cfg
        self.assertRaises(
            TypeError,
            prepare_shared_expert_weights,
            cfg,
            torch.empty(8, 8, dtype=torch.float8_e4m3fn),
            torch.ones(2, 2),
            torch.empty(8, 4, dtype=torch.float8_e4m3fn),
            torch.ones(2, 2, dtype=torch.int32),
            fp8_routed=False,
        )

    def test_standard_constructor_prepares_routes_then_shared_weights(self):
        for strategy, backend_type, _ in BACKENDS:
            for layout in ("up_gate", "gate_up"):
                with self.subTest(backend=strategy, layout=layout):
                    config = self.backend(backend_type, strategy).cfg
                    config.mega_moe_policy = MegaMoeExecutionPolicy(
                        w1_layout=layout, activation_clamp=5.0, fast_math=False
                    )
                    quant_config = FusedMoEQuantConfig()
                    w13 = torch.arange(256).reshape(4, 8, 8).to(torch.float8_e4m3fn)
                    weights = {
                        W.moe_w1: w13,
                        W.moe_w2: torch.empty(4, 8, 4, dtype=torch.float8_e4m3fn),
                        W.moe_s1: torch.arange(64, dtype=torch.float32).reshape(
                            4, 8, 2
                        ),
                        W.moe_s2: torch.ones(4, 8, 2),
                        W.ffn_w13: torch.empty(8, 8, dtype=torch.float8_e4m3fn),
                        W.ffn_w2: torch.empty(8, 4, dtype=torch.float8_e4m3fn),
                        W.ffn_s13: torch.ones(2, 2),
                        W.ffn_s2: torch.ones(2, 2),
                    }
                    shared_w13 = weights[W.ffn_w13]
                    events = []
                    with ExitStack() as stack:
                        stack.enter_context(patch("torch.cuda.empty_cache"))
                        routed = stack.enter_context(
                            patch.object(
                                backend_type,
                                "setup_weights_from_fp8",
                                side_effect=lambda **kwargs: events.append("routed"),
                            )
                        )
                        if backend_type.includes_shared_expert:
                            shared = stack.enter_context(
                                patch.object(
                                    backend_type,
                                    "setup_shared_expert_from_fp8",
                                    side_effect=lambda **kwargs: events.append(
                                        "shared"
                                    ),
                                )
                            )
                            warmup = stack.enter_context(
                                patch.object(
                                    backend_type,
                                    "maybe_warmup_fused_shared_jit_once",
                                    side_effect=lambda: events.append("warmup"),
                                )
                            )
                        model = backend_type(config, quant_config, weights)
                    self.assertIs(model.config, config)
                    self.assertIs(model.quant_config, quant_config)
                    self.assertIs(model.weights, weights)
                    self.assertIsNot(model.cfg, config)
                    self.assertEqual(config.max_tokens_per_rank, 8)
                    self.assertEqual(model.cfg.max_tokens_per_rank, 128)
                    self.assertEqual(model._activation_clamp, 5.0)
                    self.assertFalse(model._fast_math)
                    gate = w13[:, 4:] if layout == "up_gate" else w13[:, :4]
                    up = w13[:, :4] if layout == "up_gate" else w13[:, 4:]
                    torch.testing.assert_close(
                        routed.call_args.kwargs["w1_fp8"].float(), gate.float()
                    )
                    torch.testing.assert_close(
                        routed.call_args.kwargs["w3_fp8"].float(), up.float()
                    )
                    self.assertNotIn(W.moe_w1, weights)
                    self.assertNotIn(W.moe_s1, weights)
                    self.assertNotIn(W.moe_w2, weights)
                    self.assertNotIn(W.moe_s2, weights)
                    if model.includes_shared_expert:
                        self.assertIs(shared.call_args.kwargs["w1_w"], shared_w13)
                        warmup.assert_called_once()
                        self.assertEqual(events, ["routed", "shared", "warmup"])
                        self.assertNotIn(W.ffn_w13, weights)
                    else:
                        self.assertEqual(events, ["routed"])
                        self.assertIs(weights[W.ffn_w13], shared_w13)

    def test_prepared_capacity_preserves_colocated_decode_and_verify_batches(self):
        for role, gen, expected in (
            (RoleType.PDFUSION, 0, 32768),
            (RoleType.PDFUSION, 5, 196608),
            (RoleType.DECODE, 5, 196608),
            (RoleType.PREFILL, 5, 16384),
        ):
            with self.subTest(role=role, gen=gen):
                config = self.backend(MegaMoeExecutor).cfg
                config.parallelism_config.role_type = role
                config.model_config.max_seq_len = 1048576
                config.model_config.gen_num_per_cycle = gen
                config.max_generate_batch_size = 32768
                config.moe_config.ll_num_max_token = 32768 * (gen + 1)
                weights = {
                    W.moe_w1: torch.empty(4, 8, 4, dtype=torch.int8),
                    W.moe_s1: torch.ones(4, 8, 1),
                    W.moe_w2: torch.empty(4, 8, 2, dtype=torch.int8),
                    W.moe_s2: torch.ones(4, 8, 1),
                }
                with patch.object(MegaMoeExecutor, "setup_weights_from_fp4"), patch(
                    "torch.cuda.empty_cache"
                ):
                    executor = MegaMoeExecutor(config, FusedMoEQuantConfig(), weights)
                self.assertEqual(executor.cfg.max_tokens_per_rank, expected)

    def test_fp4_checkpoint_uses_declared_layout_without_runtime_requantization(self):
        config = self.backend(MegaMoeExecutor).cfg
        config.mega_moe_policy = MegaMoeExecutionPolicy(w1_layout="gate_up")
        weights = {
            W.moe_w1: torch.empty(4, 8, 4, dtype=torch.int8),
            W.moe_s1: torch.ones(4, 8, 1),
            W.moe_w2: torch.empty(4, 8, 2, dtype=torch.int8),
            W.moe_s2: torch.ones(4, 8, 1),
        }
        checkpoint = tuple(weights.values())
        with patch.object(
            MegaMoeExecutor, "setup_weights_from_fp4"
        ) as fp4, patch.object(MegaMoeExecutor, "setup_weights_from_fp8") as fp8, patch(
            "torch.cuda.empty_cache"
        ):
            MegaMoeExecutor(config, FusedMoEQuantConfig(), weights)
        self.assertEqual(fp4.call_args.kwargs, {"w1_layout": "gate_up"})
        for actual, expected in zip(fp4.call_args.args, checkpoint):
            self.assertIs(actual, expected)
        fp8.assert_not_called()
        self.assertEqual(weights, {})


if __name__ == "__main__":
    unittest.main()
