"""V4.1 uses the generic performance runner's route override semantics."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.moe.strategies.base import (
    MoeCfg,
    RoutedExpertsStrategy,
)


def config(**kwargs):
    return MoeCfg(
        layer_id=0,
        dim=128,
        moe_inter_dim=128,
        n_routed_experts=384,
        n_activated_experts=6,
        swiglu_limit=10.0,
        ep_size=4,
        ep_rank=0,
        n_local_experts=96,
        local_expert_start=0,
        local_expert_end=96,
        max_tokens_per_rank=4096,
        **kwargs
    )


class FakeBalanceExpertTest(unittest.TestCase):
    def test_disabled_preserves_ids_and_weights(self):
        strategy = RoutedExpertsStrategy(config())
        ids = torch.tensor([[7, 5, 3, 1, 2, 6]])
        weights = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        before = ids.clone(), weights.clone()
        strategy.apply_fake_balance(ids, weights)
        self.assertTrue(torch.equal(ids, before[0]))
        self.assertTrue(torch.equal(weights, before[1]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_main_and_draft_balanced_counts_and_graph(self):
        for experts, topk, rows in ((384, 6, 384), (128, 3, 320)):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(experts=experts, dtype=dtype):
                    counts = torch.zeros(experts, dtype=torch.int64, device="cuda")
                    for rank in range(4):
                        cfg = replace(
                            config(fake_balance_expert=True, dp_size=4, dp_rank=rank),
                            n_routed_experts=experts,
                            n_activated_experts=topk,
                        )
                        strategy = RoutedExpertsStrategy(cfg)
                        ids = torch.full(
                            (rows + 1, topk), -1, device="cuda", dtype=dtype
                        )
                        weights = torch.zeros_like(ids, dtype=torch.float32)
                        strategy.apply_fake_balance(ids[:rows], weights[:rows])
                        local = experts // 4
                        index = torch.arange(rows * topk, device="cuda").reshape(
                            rows, topk
                        )
                        reference = ((rank + index % 4) % 4) * local + (
                            int(rank * max(local / 4, 1)) + index // 4
                        ) % local
                        self.assertTrue(torch.equal(ids[:rows], reference))
                        expected = ids.clone()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            strategy.apply_fake_balance(ids[:rows], weights[:rows])
                        ids.fill_(-1)
                        weights.zero_()
                        graph.replay()
                        self.assertTrue(torch.equal(ids, expected))
                        self.assertTrue(torch.all(weights[:rows] == 1).item())
                        self.assertTrue(torch.all(ids[rows:] == -1).item())
                        self.assertTrue(torch.all(weights[rows:] == 0).item())
                        counts += torch.bincount(
                            ids[:rows].long().flatten(), minlength=experts
                        )
                    self.assertTrue(
                        torch.all(counts == 4 * rows * topk // experts).item()
                    )
                    self.assertEqual(
                        (counts.reshape(4, -1) > 0).sum(1).tolist(), [experts // 4] * 4
                    )

    def test_config_reaches_target_and_dspark_blocks(self):
        from rtp_llm.models_py.model_desc import deepseek_v41_base_model as model_module
        from rtp_llm.models_py.model_desc.deepseek_v41_base_dspark_model import (
            DeepSeekV4DSparkModel,
        )
        from rtp_llm.models_py.modules.dsv41.transformer import V4Args, _block_kwargs
        from rtp_llm.ops import MoeConfig

        self.assertFalse(MoeConfig().fake_balance_expert)
        model_config = SimpleNamespace(
            gen_num_per_cycle=5,
            inter_size=1536,
            ckpt_path="",
            dspark_noise_token_id=1,
            dspark_target_layer_ids=[0, 10, 20],
            dspark_markov_rank=32,
        )
        parallel = SimpleNamespace(dp_size=8, dp_rank=3, ep_size=4, ep_rank=1)
        for enabled in (False, True):
            for cls, layers, experts, topk in (
                (model_module.DeepSeekV4Model, 40, 384, 6),
                (DeepSeekV4DSparkModel, 3, 128, 3),
            ):
                args = V4Args(
                    n_layers=layers,
                    compress_ratios=[0] * layers,
                    n_routed_experts=experts,
                    n_activated_experts=topk,
                )
                with patch.object(
                    model_module.GptModelBase, "__init__", return_value=None
                ), patch.object(
                    model_module, "_args_from_model_config", return_value=args
                ):
                    model = cls(
                        model_config,
                        parallel,
                        None,
                        SimpleNamespace(
                            fake_balance_expert=enabled, moe_inter_padding_size=0
                        ),
                        64,
                    )
                for layer in range(layers):
                    kwargs = _block_kwargs(layer, model._v4_args, {})
                    self.assertEqual(
                        (
                            kwargs["fake_balance_expert"],
                            kwargs["dp_size"],
                            kwargs["dp_rank"],
                        ),
                        (enabled, 8, 3),
                    )
                    self.assertEqual(
                        (kwargs["n_routed_experts"], kwargs["n_activated_experts"]),
                        (experts, topk),
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_mega_consumers_see_override_after_each_pack_path(self):
        import deep_gemm

        from rtp_llm.models_py.modules.dsv41.moe import mega_gate
        from rtp_llm.models_py.modules.dsv41.moe.strategies import (
            mega,
            mega_fused,
            mega_se,
        )

        for module, cls in (
            (mega, mega.MegaMoEStrategy),
            (mega_fused, mega_fused.MegaMoEFusedStrategy),
            (mega_se, mega_se.MegaMoEStrategySE),
        ):
            for enabled in (False, True):
                strategy = cls.__new__(cls)
                RoutedExpertsStrategy.__init__(
                    strategy, config(fake_balance_expert=enabled)
                )
                rows = 7
                buf = SimpleNamespace(
                    x=torch.zeros((rows, 128), device="cuda", dtype=torch.bfloat16),
                    x_sf=torch.zeros((rows, 4), device="cuda"),
                    shared_l1_acts_sf=torch.zeros((rows, 4), device="cuda"),
                    topk_idx=torch.full(
                        (rows, 6), -1, device="cuda", dtype=torch.int64
                    ),
                    topk_weights=torch.zeros((rows, 6), device="cuda"),
                )
                strategy._mega_buf = buf
                strategy._mega_y = torch.zeros_like(buf.x)
                strategy._block_m = lambda _: 32
                strategy._maybe_pre_kernel_barrier = lambda _: None
                for name in (
                    "_mega_l1_w",
                    "_mega_l1_sf",
                    "_mega_l2_w",
                    "_mega_l2_sf",
                    "_se_l1_w",
                    "_se_l1_sf",
                    "_se_l2_w",
                    "_se_l2_sf",
                    "_se_l1_fp8",
                    "_se_l2_fp8",
                    "_mid_fp8",
                    "_mid_sf",
                ):
                    setattr(strategy, name, None)
                strategy._shared_recipe_kwargs = {}

                def pack(*args, **kwargs):
                    buf.topk_idx.fill_(9)
                    buf.topk_weights.fill_(0.25)

                def consume(y, *args, **kwargs):
                    index = torch.arange(rows * 6, device="cuda").reshape(rows, 6)
                    expected = (
                        (index % 4) * 96 + index // 4
                        if enabled
                        else torch.full_like(index, 9)
                    )
                    self.assertTrue(torch.equal(buf.topk_idx, expected))
                    self.assertTrue(
                        torch.all(buf.topk_weights == (1 if enabled else 0.25)).item()
                    )
                    y.fill_(2)

                strategy._input_packer = SimpleNamespace(pack=pack)
                gate = SimpleNamespace(
                    hash=False,
                    route_scale=2.5,
                    bias=torch.zeros(384, device="cuda"),
                    tid2eid=torch.zeros((1, 6), device="cuda", dtype=torch.int64),
                    _weight_bf16=lambda: torch.zeros(
                        (384, 128), device="cuda", dtype=torch.bfloat16
                    ),
                    _project_scores=lambda x, w: x @ w.T,
                )
                kernel_name = (
                    "fp8_fp4_mega_moe_fused"
                    if module is mega_fused
                    else "fp8_fp4_mega_moe"
                )
                getter = (
                    "_get_mega_se_gate_pack_kernels"
                    if module is mega_se
                    else "_get_gate_pack_kernels"
                )
                with patch.object(
                    module, getter, return_value=(pack, pack, None)
                ), patch.object(module, "sync_cuda_graph_warmup_ranks"), patch.object(
                    deep_gemm, kernel_name, side_effect=consume, create=True
                ) as consumer:
                    strategy(buf.x, buf.topk_weights, buf.topk_idx)
                    for is_hash in (False, True):
                        gate.hash = is_hash
                        strategy.forward_with_gate_pack(
                            buf.x,
                            gate,
                            torch.zeros(rows, device="cuda", dtype=torch.int64),
                        )
                    if module is mega_se:
                        inputs = SimpleNamespace(
                            owner=strategy,
                            x=buf.x,
                            topk_idx=buf.topk_idx,
                            topk_weights=buf.topk_weights,
                        )
                        with patch.object(mega_gate, "mega_gate", side_effect=pack):
                            strategy.forward_from_mhc(buf.x, gate, inputs)
                    self.assertEqual(consumer.call_count, 4 if module is mega_se else 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_nonmega_routes_are_overridden_before_compute_or_dispatch(self):
        from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper
        from rtp_llm.models_py.modules.dsv41.moe.strategies.deepep import DeepEPStrategy
        from rtp_llm.models_py.modules.dsv41.moe.strategies.grouped_fp4 import (
            GroupedFP4Strategy,
        )
        from rtp_llm.models_py.modules.dsv41.moe.strategies.local_loop import (
            LocalLoopStrategy,
        )

        class Dispatched(Exception):
            pass

        for cls in (LocalLoopStrategy, GroupedFP4Strategy, DeepEPStrategy):
            for enabled in (False, True):
                cfg = config(fake_balance_expert=enabled, dp_size=8, dp_rank=3)
                strategy = cls.__new__(cls)
                RoutedExpertsStrategy.__init__(strategy, cfg)
                rows = 65
                ids = torch.full((rows, 6), 9, device="cuda", dtype=torch.int64)
                weights = torch.full((rows, 6), 0.25, device="cuda")
                x = torch.zeros((rows, 128), device="cuda", dtype=torch.bfloat16)
                index = torch.arange(rows * 6, device="cuda").reshape(rows, 6)
                expected = (
                    ((1 + index % 4) % 4) * 96 + (36 + index // 4) % 96
                    if enabled
                    else ids.clone()
                )

                def compute(x, actual_weights, actual_ids, **kwargs):
                    self.assertTrue(torch.equal(actual_ids, expected))
                    self.assertTrue(
                        torch.all(actual_weights == (1 if enabled else 0.25)).item()
                    )
                    return x

                if cls is LocalLoopStrategy:
                    strategy._forward_into_buf = compute
                    self.assertIs(strategy(x, weights, ids), x)
                elif cls is GroupedFP4Strategy:
                    strategy._forward_capture_topk = compute
                    with patch(
                        "torch.cuda.is_current_stream_capturing", return_value=True
                    ):
                        self.assertIs(strategy(x, weights, ids), x)
                else:

                    def dispatch_layout(padded_ids, experts):
                        self.assertTrue(torch.equal(padded_ids[:, :6], expected))
                        self.assertTrue(torch.all(padded_ids[:, 6:] == -1).item())
                        self.assertTrue(
                            torch.all(weights == (1 if enabled else 0.25)).item()
                        )
                        raise Dispatched

                    wrapper = SimpleNamespace(
                        buffer=SimpleNamespace(get_dispatch_layout=dispatch_layout)
                    )
                    with patch.object(
                        DeepEPWrapper, "_instance", wrapper
                    ), self.assertRaises(Dispatched):
                        strategy(x, weights, ids)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_empty_rows_do_not_launch_invalid_grid(self):
        strategy = RoutedExpertsStrategy(config(fake_balance_expert=True))
        ids = torch.empty((0, 6), device="cuda", dtype=torch.int64)
        strategy.apply_fake_balance(ids, ids.float())
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
