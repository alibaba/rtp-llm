import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from rtp_llm.config.maxpander_config import MaxPanderConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import ExpertGatePayload
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.maxpander import (
    MaxPanderMegaMoeTilePolicy,
    PackedMoeExecutorMixin,
)


class MaxPanderPackedMoeTest(unittest.TestCase):
    def test_inactive_maxpander_does_not_change_jit_warmup_cache_keys(self):
        from rtp_llm.models_py.modules.dsv41.moe.strategies.mega import MegaMoEStrategy
        from rtp_llm.models_py.modules.dsv41.moe.strategies.mega_se import (
            MegaMoEStrategySE,
        )

        configurations = (
            MaxPanderConfig().resolve("PREFILL"),
            MaxPanderConfig(seed=17, wf_passes=64).resolve("PREFILL"),
            MaxPanderConfig(enable=True, seed=29, wf_passes=8).resolve("DECODE"),
            MaxPanderConfig(enable=True).resolve("PREFILL"),
        )
        for executor_type in (
            MegaMoeExecutor,
            MegaMoeSEExecutor,
            MegaMoEStrategy,
            MegaMoEStrategySE,
        ):
            with self.subTest(executor=executor_type):
                owner = SimpleNamespace(
                    cfg=SimpleNamespace(
                        ep_size=2,
                        physical_expert_num=16,
                        n_local_experts=8,
                        n_activated_experts=4,
                        dim=512,
                        moe_inter_dim=512,
                        n_shared_experts=1,
                        max_tokens_per_rank=128,
                        swiglu_limit=10.0,
                        route_scale=1.0,
                    ),
                    supports_gate_pack=True,
                    _shared_recipe=(1, 128, 128),
                    _resolve_jit_warmup_token_counts=lambda *args: [1, 128],
                )
                warmed_keys = MagicMock()
                warmed_keys.__contains__.return_value = True
                method = executor_type._maybe_warmup_jit_once
                with patch.dict(
                    method.__globals__,
                    {
                        "mega_moe_jit_warmup_enabled": lambda: True,
                        "mega_moe_se_jit_warmup_enabled": lambda: True,
                        "resolve_mega_num_sms": lambda *args: 132,
                        "_MEGA_MOE_JIT_WARMED_KEYS": warmed_keys,
                        "_MEGA_MOE_SE_JIT_WARMED_KEYS": warmed_keys,
                    },
                ), patch("deep_gemm.get_num_sms", return_value=132), patch(
                    "torch.cuda.is_current_stream_capturing", return_value=False
                ):
                    for config in configurations:
                        owner.cfg.maxpander_config = config
                        method(owner)
                keys = [
                    call.args[0] for call in warmed_keys.__contains__.call_args_list
                ]
                self.assertEqual(len(keys), len(configurations))
                self.assertEqual(keys[0], keys[1])
                self.assertEqual(keys[0], keys[2])
                self.assertNotEqual(keys[0], keys[3])

    def test_inactive_roles_do_not_initialize_routing_or_read_placement(self):
        for requested, role in (
            (False, "PREFILL"),
            (False, "PDFUSION"),
            (True, "DECODE"),
            (True, "FRONTEND"),
        ):
            with self.subTest(requested=requested, role=role), patch(
                "rtp_llm.models_py.modules.maxpander.MaxPanderRouter",
                side_effect=AssertionError(
                    "Inactive MaxPander must not create a router"
                ),
            ):
                owner = PackedMoeExecutorMixin()
                owner.cfg = SimpleNamespace(
                    maxpander_config=MaxPanderConfig(enable=requested).resolve(role)
                )
                weights = Mock()
                weights.get.side_effect = AssertionError("Unexpected placement access")
                owner.setup_routing(weights)
                self.assertIsNone(owner.physical_router)
                weights.get.assert_not_called()

    def test_both_pack_paths_remap_once_before_launch(self):
        for executor_type in (MegaMoeExecutor, MegaMoeSEExecutor):
            for enabled in (False, True):
                for tokens in (0, 3):
                    with self.subTest(
                        executor=executor_type, enabled=enabled, tokens=tokens
                    ):
                        self.check_paths(executor_type, enabled, tokens)

    def check_paths(self, executor_type, enabled, tokens):
        executor = executor_type.__new__(executor_type)
        torch.nn.Module.__init__(executor)
        executor.cfg = SimpleNamespace(n_activated_experts=2, max_tokens_per_rank=4)
        executor._mega_y = torch.zeros((4, 128), dtype=torch.bfloat16)
        executor._mega_buf = SimpleNamespace(
            num_max_tokens_per_rank=4,
            x=torch.empty((4, 128)),
            x_sf=torch.empty((4, 1)),
            topk_idx=torch.empty((4, 4), dtype=torch.int64)[:, :2],
            topk_weights=torch.empty((4, 2)),
            shared_l1_acts_sf=torch.empty(1),
        )
        logical = torch.tensor([[0, 3]]).expand(tokens, 2).contiguous()
        weights = torch.full((tokens, 2), 0.25)
        inputs = torch.ones((tokens, 128), dtype=torch.bfloat16)
        events = []

        def pack(*args):
            events.append("pack")
            executor._mega_buf.topk_idx[:tokens].copy_(logical)
            executor._mega_buf.topk_weights[:tokens].copy_(weights)

        def remap(packed):
            events.append("map")
            torch.testing.assert_close(packed, logical)
            packed.add_(10)

        def launch(output, actual_tokens, device):
            events.append("launch")
            self.assertEqual(actual_tokens, tokens)
            torch.testing.assert_close(
                executor._mega_buf.topk_idx[:tokens], logical + (10 if enabled else 0)
            )
            torch.testing.assert_close(
                executor._mega_buf.topk_weights[:tokens], weights
            )

        executor._input_packer = SimpleNamespace(name="fused", pack=pack)
        executor._block_m = lambda token_count: 64
        executor._launch = launch
        executor.physical_router = (
            SimpleNamespace(remap_inplace=remap) if enabled else None
        )
        payload = ExpertGatePayload(
            scores=torch.zeros((tokens, 4)),
            topk=2,
            score_func="sqrtsoftplus",
            route_scale=1.0,
        )
        with patch.dict(os.environ, {"MEGA_MOE_INPUT_PACKER_IMPL": "optimized"}), patch(
            "rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack.fused_pack_mega_moe_gate_inputs",
            side_effect=lambda *args, **kwargs: pack(),
        ), patch(
            "rtp_llm.models_py.triton_kernels.moe.mega_moe_se_input_pack.stage_mega_moe_se_shared_l1_scales"
        ):
            self.assertTrue(executor.supports_gate_pack)
            for _ in range(2):
                events.clear()
                executor.forward(inputs, weights, logical)
                self.assertEqual(
                    events, ["pack", "map", "launch"] if enabled else ["pack", "launch"]
                )
                events.clear()
                executor.forward_gate_pack(inputs, payload)
                self.assertEqual(
                    events, ["pack", "map", "launch"] if enabled else ["pack", "launch"]
                )

    def test_v41_strategies_share_the_same_post_pack_tail(self):
        from rtp_llm.models_py.modules.dsv41.moe.strategies.mega import MegaMoEStrategy
        from rtp_llm.models_py.modules.dsv41.moe.strategies.mega_fused import (
            MegaMoEFusedStrategy,
        )
        from rtp_llm.models_py.modules.dsv41.moe.strategies.mega_se import (
            MegaMoEStrategySE,
        )

        for strategy in (
            MegaMoeExecutor,
            MegaMoeSEExecutor,
            MegaMoEStrategy,
            MegaMoEStrategySE,
            MegaMoEFusedStrategy,
        ):
            self.assertIs(
                strategy._execute_packed_inputs,
                PackedMoeExecutorMixin._execute_packed_inputs,
            )


class MaxPanderTilePolicyTest(unittest.TestCase):
    def make_config(self, enabled=True, role="PREFILL", physical=480):
        return SimpleNamespace(
            maxpander_config=MaxPanderConfig(enable=enabled).resolve(role),
            ep_size=4,
            n_routed_experts=384,
            physical_expert_num=physical,
            max_tokens_per_rank=5760,
            n_activated_experts=6,
        )

    def make_backend(self):
        def kernel(*args, block_m=0, **kwargs):
            return args, kwargs, block_m

        def resolver(ranks, experts, capacity, tokens, topk, mma_type):
            expected = tokens * ranks * topk / experts
            if expected <= 24:
                return 16 if expected <= 10 else 32
            covered = expected + math.sqrt(expected)
            blocks = math.ceil(covered / 240)
            for candidate in (64, 128, 192, 240):
                if blocks * candidate >= covered:
                    return candidate
            raise AssertionError("Invalid test heuristic")

        return SimpleNamespace(
            fp8_fp4_mega_moe=kernel,
            get_block_m_for_mega_moe=Mock(side_effect=resolver),
        )

    def test_all_tiles_match_off_without_changing_kernel_arguments(self):
        backend = self.make_backend()
        policy = MaxPanderMegaMoeTilePolicy(
            self.make_config(),
            backend.fp8_fp4_mega_moe,
            backend.get_block_m_for_mega_moe,
        )
        physical_buffer = object()
        for tokens, selected in (
            (0, 16),
            (32, 16),
            (256, 32),
            (512, 64),
            (1024, 128),
            (2048, 192),
            (3072, 240),
            (4096, 192),
        ):
            with self.subTest(tokens=tokens):
                self.assertEqual(policy.block_m(tokens), selected)
                backend.get_block_m_for_mega_moe.assert_called_with(
                    4, 384, 5760, tokens, 6, "fp8xfp4"
                )
                result = policy.kernel_for(tokens)(
                    "output", physical_buffer, fast_math=True
                )
                self.assertEqual(
                    result,
                    (("output", physical_buffer), {"fast_math": True}, selected),
                )

    def test_selection_and_launch_share_cached_resolver(self):
        backend = self.make_backend()
        policy = MaxPanderMegaMoeTilePolicy(
            self.make_config(),
            backend.fp8_fp4_mega_moe,
            backend.get_block_m_for_mega_moe,
        )
        self.assertEqual(policy.block_m(1024), 128)
        self.assertEqual(backend.get_block_m_for_mega_moe.call_count, 1)
        kernel = policy.kernel_for(1024)
        for _ in range(10):
            policy.kernel_for(1024)("output")
            self.assertEqual(policy.block_m(1024), 128)
            self.assertIs(policy.kernel_for(1024), kernel)
        self.assertEqual(backend.get_block_m_for_mega_moe.call_count, 1)
        self.assertIs(policy.kernel_for(1025), kernel)
        self.assertEqual(backend.get_block_m_for_mega_moe.call_count, 2)

    def test_off_decode_and_zero_redundancy_never_probe_backend(self):
        owner = PackedMoeExecutorMixin()
        for config in (
            self.make_config(enabled=False),
            self.make_config(role="DECODE"),
            self.make_config(role="FRONTEND"),
            self.make_config(physical=384),
        ):
            with self.subTest(config=config), patch(
                "rtp_llm.models_py.modules.maxpander._supports_mega_moe_tile_override",
                side_effect=AssertionError("Backend must not be probed"),
            ):
                owner.cfg = config
                backend = self.make_backend()
                owner.setup_mega_moe_tile_policy(backend)
                self.assertIsNone(owner._maxpander_tile_policy)
                self.assertIs(
                    owner._select_mega_moe_kernel(backend, 2048),
                    backend.fp8_fp4_mega_moe,
                )
                backend.get_block_m_for_mega_moe.assert_not_called()

    def test_old_wheel_keeps_original_callable(self):
        owner = PackedMoeExecutorMixin()
        owner.cfg = self.make_config()

        def original(*args, **kwargs):
            return args, kwargs

        backend = SimpleNamespace(fp8_fp4_mega_moe=original)
        with self.assertLogs(level="WARNING") as captured:
            owner.setup_mega_moe_tile_policy(backend)
        self.assertIn("retaining native tile selection", captured.output[0])
        self.assertIs(owner._select_mega_moe_kernel(backend, 1024), original)

    def test_v4_and_v41_use_shared_policy_for_launch_and_warmup(self):
        from rtp_llm.models_py.modules.dsv41.moe.strategies.mega import MegaMoEStrategy

        for executor_type in (MegaMoeExecutor, MegaMoEStrategy):
            with self.subTest(executor=executor_type):
                owner = executor_type.__new__(executor_type)
                torch.nn.Module.__init__(owner)
                owner.cfg = self.make_config()
                backend = self.make_backend()
                owner.setup_mega_moe_tile_policy(backend)
                for tokens, selected in (
                    (32, 16),
                    (1024, 128),
                    (2048, 192),
                    (4096, 192),
                ):
                    self.assertEqual(
                        owner._maxpander_tile_policy.block_m(tokens), selected
                    )
                    self.assertEqual(
                        owner._select_mega_moe_kernel(backend, tokens)("output")[2],
                        selected,
                    )

    def test_v4_warmup_resolver_covers_all_logical_buckets(self):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
            generate_mega_moe_jit_token_counts,
        )

        backend = self.make_backend()
        policy = MaxPanderMegaMoeTilePolicy(
            self.make_config(),
            backend.fp8_fp4_mega_moe,
            backend.get_block_m_for_mega_moe,
        )
        for include_cap in (False, True):
            tokens = generate_mega_moe_jit_token_counts(
                num_ranks=4,
                num_experts=480,
                num_experts_per_rank=120,
                num_topk=6,
                intermediate_hidden=2048,
                num_sms=132,
                max_tokens_per_rank=5760,
                include_cap=include_cap,
                block_m_resolver=policy.block_m,
            )
            self.assertEqual(
                {policy.block_m(token) for token in tokens},
                {16, 32, 64, 128, 192, 240},
            )
            self.assertEqual(tokens[-1] == 5760, include_cap)


if __name__ == "__main__":
    unittest.main()
