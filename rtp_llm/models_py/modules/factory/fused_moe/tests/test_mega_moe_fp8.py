import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertGatePayload,
    FusedMoe,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
    MegaMoeFp8Executor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se import (
    MegaMoeFp8SEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.mega_moe_fp8 import (
    CudaMegaMoeFp8SEStrategy,
    CudaMegaMoeFp8Strategy,
    MegaMoeFp8Router,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.group import (
    get_validated_world_ep_group,
)


class Checks:
    def __init__(self):
        self.values = []

    def check(self, value):
        self.values.append(bool(value))

    @property
    def passed(self):
        return all(self.values)


class MegaMoeFp8SelectionTest(unittest.TestCase):
    def config(self, **overrides):
        values = dict(
            moe_strategy="mega_moe_fp8",
            moe_quant_method="FP8_PER_BLOCK",
            ep_size=4,
            ep_rank=0,
            tp_size=1,
            world_size=4,
            world_rank=0,
            has_redundant_experts=False,
            enable_cuda_graph=False,
            swiglu_limit=0.0,
            hidden_size=4096,
            moe_inter_dim=1024,
            expert_num=512,
            n_shared_experts=0,
            has_shared_expert_gate=False,
            moe_k=8,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def check_executor(self, executor_cls=MegaMoeFp8Executor, **overrides):
        checker = Checks()
        with patch.object(
            MoeConfigResolver, "get_quant_method", return_value="FP8_PER_BLOCK"
        ), patch.object(MoeConfigResolver, "is_bf16", return_value=True), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8.mega_moe_fp8_available",
            return_value=True,
        ), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se.mega_moe_fp8_se_available",
            return_value=True,
        ):
            executor_cls.check_conditions(checker, self.config(**overrides))
        return checker.passed

    def test_opt_in_only(self):
        cases = [
            (CudaMegaMoeFp8Strategy, "mega_moe_fp8", True, {}),
            (CudaMegaMoeFp8Strategy, "auto", False, {}),
            (CudaMegaMoeFp8Strategy, "mega_moe", False, {}),
            (CudaMegaMoeFp8Strategy, "mega_moe_fp8_se", False, {}),
            (
                CudaMegaMoeFp8SEStrategy,
                "mega_moe_fp8_se",
                True,
                {"n_shared_experts": 1, "has_shared_expert_gate": True},
            ),
            (
                CudaMegaMoeFp8SEStrategy,
                "mega_moe_fp8",
                False,
                {"n_shared_experts": 1, "has_shared_expert_gate": True},
            ),
            (
                CudaMegaMoeFp8SEStrategy,
                "auto",
                False,
                {"n_shared_experts": 1, "has_shared_expert_gate": True},
            ),
            (
                CudaMegaMoeFp8SEStrategy,
                "mega_moe_fp8_se",
                False,
                {"n_shared_experts": 0},
            ),
            (
                CudaMegaMoeFp8SEStrategy,
                "mega_moe_fp8_se",
                False,
                {"n_shared_experts": 1},
            ),
        ]
        for strategy_cls, value, expected, extra in cases:
            with self.subTest(
                strategy=strategy_cls.strategy_name, value=value
            ), patch.object(
                MoeConfigResolver, "get_quant_method", return_value="FP8_PER_BLOCK"
            ):
                checker = Checks()
                strategy_cls.check_conditions(
                    checker, self.config(moe_strategy=value, **extra)
                )
                self.assertEqual(checker.passed, expected)

    def test_supported_parallelism(self):
        self.assertTrue(self.check_executor())
        self.assertTrue(self.check_executor(enable_cuda_graph=True))
        for tp_size in (2, 4):
            self.assertTrue(self.check_executor(tp_size=tp_size))
            self.assertTrue(
                self.check_executor(tp_size=tp_size, enable_cuda_graph=True)
            )
            self.assertFalse(
                self.check_executor(
                    MegaMoeFp8SEExecutor,
                    tp_size=tp_size,
                    n_shared_experts=1,
                    has_shared_expert_gate=True,
                )
            )
        self.assertTrue(
            self.check_executor(
                MegaMoeFp8SEExecutor,
                n_shared_experts=1,
                has_shared_expert_gate=True,
                enable_cuda_graph=True,
            )
        )
        self.assertFalse(self.check_executor(MegaMoeFp8SEExecutor, n_shared_experts=0))
        self.assertFalse(
            self.check_executor(
                MegaMoeFp8SEExecutor,
                n_shared_experts=2,
                has_shared_expert_gate=True,
            )
        )
        for invalid in [
            dict(swiglu_limit=1.0),
            dict(tp_size=0),
            dict(tp_size=3),
            dict(tp_size=8),
            dict(ep_size=1),
            dict(world_size=8),
            dict(world_rank=1),
            dict(has_redundant_experts=True),
            dict(hidden_size=4000),
            dict(moe_inter_dim=1000),
            dict(expert_num=513),
        ]:
            with self.subTest(invalid=invalid):
                self.assertFalse(self.check_executor(**invalid))

    def test_world_group_validation(self):
        world = object()
        dist = SimpleNamespace(
            is_initialized=lambda: True,
            group=SimpleNamespace(WORLD=world),
            get_world_size=lambda group: 4,
            get_rank=lambda group: 0,
        )
        self.assertIs(get_validated_world_ep_group(self.config(), dist), world)
        with self.assertRaises(RuntimeError):
            get_validated_world_ep_group(self.config(ep_rank=1), dist)

    def test_registry_selects_explicit_backend(self):
        registry = StrategyRegistry()
        registry.register(CudaMegaMoeFp8SEStrategy())
        registry.register(CudaMegaMoeFp8Strategy())
        with patch.object(
            MoeConfigResolver, "get_quant_method", return_value="FP8_PER_BLOCK"
        ), patch.object(MoeConfigResolver, "is_bf16", return_value=True), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8.mega_moe_fp8_available",
            return_value=True,
        ), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se.mega_moe_fp8_se_available",
            return_value=True,
        ):
            self.assertEqual(
                registry.get_strategy(self.config()).strategy_name, "mega_moe_fp8"
            )
            self.assertEqual(
                registry.get_strategy(self.config(tp_size=4)).strategy_name,
                "mega_moe_fp8",
            )
            self.assertEqual(
                registry.get_strategy(
                    self.config(
                        moe_strategy="mega_moe_fp8_se",
                        n_shared_experts=1,
                        has_shared_expert_gate=True,
                    )
                ).strategy_name,
                "mega_moe_fp8_se",
            )
            self.assertIs(
                registry.get_strategy(
                    self.config(
                        moe_strategy="mega_moe_fp8_se",
                        n_shared_experts=1,
                        has_shared_expert_gate=True,
                    )
                )
                .get_attributes()
                .executor_class,
                MegaMoeFp8SEExecutor,
            )

    def test_router_and_executor_advertise_gate_pack(self):
        router = MegaMoeFp8Router.__new__(MegaMoeFp8Router)
        self.assertTrue(router.supports_gate_pack)
        self.assertTrue(MegaMoeFp8Executor.gated_shared_expert_requested is False)
        self.assertTrue(MegaMoeFp8SEExecutor.gated_shared_expert_requested)
        executor = MegaMoeFp8Executor.__new__(MegaMoeFp8Executor)
        executor.config = self.config(n_activated_experts=8)
        with patch.dict("os.environ", {"MEGA_MOE_INPUT_PACKER_IMPL": "optimized"}):
            self.assertTrue(executor.supports_gate_pack)
        with patch.dict("os.environ", {"MEGA_MOE_INPUT_PACKER_IMPL": "legacy"}):
            self.assertTrue(executor.supports_gate_pack)
        se = MegaMoeFp8SEExecutor.__new__(MegaMoeFp8SEExecutor)
        self.assertTrue(se.supports_gate_pack)

    def test_se_block_m(self):
        executor = MegaMoeFp8SEExecutor.__new__(MegaMoeFp8SEExecutor)
        get_block_m = Mock(return_value=64)
        executor.config = self.config()
        executor._mega_buf = SimpleNamespace(num_max_tokens_per_rank=32768)
        deep_gemm = SimpleNamespace(
            mega_fp8=SimpleNamespace(
                get_block_m_for_mega_moe_fp8=get_block_m,
            )
        )
        with patch.dict("sys.modules", {"deep_gemm": deep_gemm}):
            self.assertEqual(executor._block_m(123), 64)
        get_block_m.assert_called_once_with(4, 512, 32768, 123, 8)


class _ReferenceMegaMoeFp8Executor(MegaMoeFp8Executor):
    """Exercise the real executor adapter without launching the CUDA kernel."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.input_rows = []

    @staticmethod
    def reference(x, weights, indices):
        return x * (weights * (indices + 1)).sum(dim=-1, keepdim=True)

    def forward(self, x, weights, indices):
        self.input_rows.append(x.size(0))
        return self.reference(x, weights, indices)

    def forward_gate_pack(self, x, gate_payload):
        weights, indices = torch.topk(
            gate_payload.scores.softmax(dim=-1), gate_payload.topk, dim=-1
        )
        return self.forward(x, weights, indices), weights, indices


class MegaMoeFp8TpRouterTest(unittest.TestCase):
    def router(self, tp_size, tp_rank):
        return MegaMoeFp8Router(SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank), None)

    def test_tp_restores_outputs_without_repeating_tokens(self):
        # Each rank starts with the same attention output. Expert dispatch must
        # consume every token exactly once, including short and uneven batches.
        for tp_size in (1, 2, 4):
            for tokens in (0, 1, 2, 3, 4, 5, 7, 8, 9, 129):
                for gate_pack in (False, True):
                    with self.subTest(tp=tp_size, tokens=tokens, gate_pack=gate_pack):
                        x = torch.arange(tokens * 8, dtype=torch.float32).reshape(
                            tokens, 8
                        )
                        scores = (
                            torch.arange(tokens * 6, dtype=torch.float32).reshape(
                                tokens, 6
                            )
                            / 17
                        )
                        weights, indices = torch.topk(scores.softmax(dim=-1), 2, dim=-1)
                        gate = ExpertGatePayload(
                            scores=scores,
                            topk=2,
                            score_func="softmax",
                            route_scale=1.0,
                        )
                        routers = [
                            self.router(tp_size, rank) for rank in range(tp_size)
                        ]
                        payloads = [
                            (
                                router.prepare_gate_pack(x, gate)
                                if gate_pack
                                else router.prepare(x, None, None, weights, indices)
                            )
                            for router in routers
                        ]
                        self.assertEqual(
                            sum(payload.expert_x.size(0) for payload in payloads),
                            tokens,
                        )
                        torch.testing.assert_close(
                            torch.cat([payload.expert_x for payload in payloads]), x
                        )
                        chunk_size = (tokens + tp_size - 1) // tp_size
                        local_outputs = []
                        for payload in payloads:
                            local_weights, local_ids = (
                                torch.topk(
                                    payload.gate_payload.scores.softmax(dim=-1),
                                    2,
                                    dim=-1,
                                )
                                if gate_pack
                                else (
                                    payload.expert_topk_weights,
                                    payload.expert_topk_ids,
                                )
                            )
                            local = _ReferenceMegaMoeFp8Executor.reference(
                                payload.expert_x, local_weights, local_ids
                            )
                            padded = x.new_zeros((chunk_size, x.size(1)))
                            padded[: local.size(0)].copy_(local)
                            local_outputs.append(padded)
                        gathered = torch.cat(local_outputs)
                        expected = _ReferenceMegaMoeFp8Executor.reference(
                            x, weights, indices
                        )
                        for rank, router in enumerate(routers):
                            executor = _ReferenceMegaMoeFp8Executor()
                            moe = FusedMoe(router, executor, expert_num=6)

                            def gather(local, group):
                                self.assertEqual(group, Group.TP)
                                torch.testing.assert_close(local, local_outputs[rank])
                                return gathered

                            with patch(
                                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.mega_moe_fp8.all_gather",
                                side_effect=gather,
                            ) as collective:
                                output = (
                                    moe.forward_gate_pack(x, gate)
                                    if gate_pack
                                    else moe(x, weights, indices)
                                )
                            torch.testing.assert_close(output, expected)
                            # Empty source ranks must still enter MegaMoE's EP
                            # collective, even when they have no local tokens.
                            self.assertEqual(
                                executor.input_rows, [payloads[rank].expert_x.size(0)]
                            )
                            self.assertEqual(
                                collective.call_count, int(tp_size > 1 and tokens > 0)
                            )

    def test_gate_pack_slices_token_metadata_only(self):
        x = torch.arange(40, dtype=torch.float32).reshape(5, 8)
        scores = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        input_ids = torch.arange(5)
        bias = torch.ones(6)
        tid2eid = torch.arange(20).reshape(10, 2)
        gate = ExpertGatePayload(
            scores=scores,
            topk=2,
            score_func="hash",
            route_scale=2.0,
            norm_eps=1e-8,
            bias=bias,
            input_ids=input_ids,
            tid2eid=tid2eid,
        )
        local = self.router(4, 1).prepare_gate_pack(x, gate)
        torch.testing.assert_close(local.expert_x, x[2:4])
        torch.testing.assert_close(local.gate_payload.scores, scores[2:4])
        torch.testing.assert_close(local.gate_payload.input_ids, input_ids[2:4])
        self.assertIs(local.gate_payload.bias, bias)
        self.assertIs(local.gate_payload.tid2eid, tid2eid)
        self.assertEqual(local.gate_payload.route_scale, gate.route_scale)
        self.assertEqual(local.gate_payload.norm_eps, gate.norm_eps)
        self.assertIs(gate.scores, scores)
        self.assertIs(gate.input_ids, input_ids)
        passthrough = self.router(1, 0).prepare_gate_pack(x, gate)
        self.assertIs(passthrough.gate_payload, gate)
        self.assertIs(passthrough.expert_x, x)


class MegaMoeFp8ScaleLifetimeTest(unittest.TestCase):
    def test_checkpoint_scales_released_only_after_success(self):
        import gc
        import os
        import weakref

        import torch

        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
            mega_moe_fp8 as module,
        )
        from rtp_llm.utils.model_weight import W

        config = SimpleNamespace(
            max_tokens_per_rank=256,
            moe_inter_dim=128,
            hidden_size=128,
            n_local_experts=1,
            moe_w1_layout="gate_up",
            expert_num=8,
            moe_k=2,
        )
        for fail in (False, True):
            with self.subTest(conversion_failure=fail):
                weights = {
                    key: torch.ones(4)
                    for key in (W.moe_w1, W.moe_w2, W.moe_s1, W.moe_s2)
                }
                owner_alias = weights
                refs = [weakref.ref(weights[key]) for key in (W.moe_s1, W.moe_s2)]
                packed = (
                    (weights[W.moe_w1], torch.ones(16)),
                    (weights[W.moe_w2], torch.ones(16)),
                )

                def prepare(*args):
                    if fail:
                        raise ValueError("conversion failed")
                    return packed

                # Use a plain function: Mock.call_args would retain old scales.
                with patch.dict(
                    os.environ, {"RTP_QWEN35_FUSED_MEGAMOE_GATED_SE": "0"}
                ), patch.object(
                    module, "mega_moe_fp8_available", return_value=True
                ), patch.object(
                    module, "get_validated_world_ep_group", return_value=None
                ), patch.object(
                    module, "prepare_mega_moe_fp8_weights", new=prepare
                ), patch.object(
                    module,
                    "_get_or_create_mega_fp8_buf",
                    return_value=SimpleNamespace(num_max_tokens_per_rank=256),
                ), patch.object(
                    module,
                    "_get_or_create_mega_output",
                    return_value=torch.empty(256, 128),
                ), patch.object(
                    module, "get_mega_moe_input_packer", return_value=Mock()
                ), patch.object(
                    MegaMoeFp8Executor, "_maybe_warmup_jit_once"
                ), patch.object(
                    torch.cuda, "Event"
                ), patch.object(
                    torch.cuda, "current_stream"
                ):
                    if fail:
                        with self.assertRaisesRegex(ValueError, "conversion failed"):
                            MegaMoeFp8Executor(config, None, weights)
                        self.assertTrue(all(ref() is not None for ref in refs))
                        self.assertIn(W.moe_s1, owner_alias)
                        self.assertIn(W.moe_s2, owner_alias)
                    else:
                        executor = MegaMoeFp8Executor(config, None, weights)
                        gc.collect()
                        self.assertIs(executor.weights, owner_alias)
                        self.assertTrue(all(ref() is None for ref in refs))
                        self.assertNotIn(W.moe_s1, owner_alias)
                        self.assertNotIn(W.moe_s2, owner_alias)
                        self.assertIs(executor.l1, packed[0])
                        self.assertIs(executor.l2, packed[1])


class MegaMoeActivationScaleTest(unittest.TestCase):
    def test_full_buffer_equivalence_and_reuse(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required for packed scale conversion")
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fp8_weights import (
            expand_fp8_scale,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_inputs import (
            expand_packed_activation_scales,
            stage_shared_scales,
        )

        capacity, hidden, shared_rows = 65536, 4096, 537600
        routed = [
            torch.full((32, capacity), 123, dtype=torch.int32, device="cuda").t()
            for _ in range(2)
        ]
        shared = [
            torch.full((32, shared_rows), 456, dtype=torch.int32, device="cuda").t()
            for _ in range(2)
        ]

        def check(source, block_m):
            n = source.shape[0]
            expand_packed_activation_scales(routed[1], source)
            if n:
                expected = expand_fp8_scale(source, n, hidden)
                routed[0][:n].copy_(expected)
                stage_shared_scales(shared[0], expected, block_m)
                stage_shared_scales(shared[1], routed[1][:n], block_m)
            self.assertTrue(torch.equal(routed[0], routed[1]))
            self.assertTrue(torch.equal(shared[0], shared[1]))

        # All exponent byte values, strided input, tails, block padding, and
        # large/small/empty reuse of the same complete destination buffers.
        for n, block_m in (
            (1, 64),
            (127, 128),
            (128, 160),
            (129, 192),
            (257, 240),
            (24601, 240),
            (49202, 240),
            (37, 256),
            (0, 240),
            (49202, 128),
            (1, 240),
            (0, 64),
        ):
            with self.subTest(tokens=n, block_m=block_m):
                source = (
                    torch.arange(8 * max(1, n) * 4, device="cuda")
                    .to(torch.uint8)
                    .reshape(8, max(1, n) * 4)
                    .view(torch.int32)
                    .t()[:n]
                )
                check(source, block_m)

        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterBase,
            is_deep_gemm_e8m0_used,
        )

        # Other GPU architectures can use a floating-point scale producer.
        # The byte-layout checks above apply independently of that producer.
        if not is_deep_gemm_e8m0_used():
            return
        for n in (1, 129, 24601, 49202):
            with self.subTest(real_quantizer_tokens=n):
                x = torch.randn((n, hidden), dtype=torch.bfloat16, device="cuda")
                x[0].zero_()
                _, source = DeepepNormalRouterBase._do_quant_fp8_per_block(None, x)
                check(source, 240)


if __name__ == "__main__":
    unittest.main()
