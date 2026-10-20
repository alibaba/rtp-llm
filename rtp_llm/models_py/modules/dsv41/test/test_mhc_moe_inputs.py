"""MegaMHC writes the buffers consumed by MegaMoE without another input pack."""

import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.hc import v41_mega_mhc
from rtp_llm.models_py.modules.dsv41.moe import mega_gate
from rtp_llm.models_py.modules.dsv41.moe.moe_layer import MoE
from rtp_llm.models_py.modules.dsv41.moe.strategies.mega_se import MegaMoEStrategySE
from rtp_llm.models_py.modules.dsv41.test.test_v41_mega_mhc import make_case

_DIM = 5120


def make_strategy(capacity=256, device="cpu", block_m=32):
    """Real strategy with small local buffers; no distributed allocation."""
    strategy = MegaMoEStrategySE.__new__(MegaMoEStrategySE)
    torch.nn.Module.__init__(strategy)
    strategy.cfg = SimpleNamespace(
        dim=_DIM,
        shared_fp8_block_size=32,
        is_decode_role=True,
        n_activated_experts=6,
        ep_rank=0,
    )
    sf_rows = ((capacity + 15) // 16) * 128
    strategy._mega_buf = SimpleNamespace(
        num_max_tokens_per_rank=capacity,
        x=torch.empty((capacity, _DIM), device=device, dtype=torch.float8_e4m3fn),
        x_sf=torch.empty((capacity, 40), device=device, dtype=torch.int32),
        shared_l1_acts_sf=torch.empty(
            (40, sf_rows), device=device, dtype=torch.int32
        ).T[:capacity],
        topk_idx=torch.empty((capacity, 6), device=device, dtype=torch.int64),
        topk_weights=torch.empty((capacity, 6), device=device, dtype=torch.float32),
    )
    strategy._mega_y = torch.empty(
        (capacity, _DIM), device=device, dtype=torch.bfloat16
    )
    strategy._block_m = lambda tokens: block_m
    return strategy


def make_moe(strategy):
    moe = MoE.__new__(MoE)
    torch.nn.Module.__init__(moe)
    moe.layer_id, moe.dim, moe.max_tokens_per_rank = 0, _DIM, 256
    moe._is_decode_role = True
    moe._gate_pack_static = True
    moe._has_visual_tokens = False
    moe._routed_includes_shared = True
    moe._strategy = strategy
    moe.gate = SimpleNamespace(
        topk=6,
        route_scale=2.5,
        score_func="sqrtsoftplus",
        bias=torch.zeros(384),
        _weight_bf16=lambda: torch.empty((384, _DIM), dtype=torch.bfloat16),
    )
    return moe


class MHCPreparedMoEHostTest(unittest.TestCase):
    def test_outputs_are_views_of_the_consumer_buffers(self):
        strategy = make_strategy()
        x = torch.zeros((6, _DIM), dtype=torch.bfloat16)
        inputs = strategy.prepare_mhc_inputs(x)
        self.assertIs(inputs.owner, strategy)
        self.assertEqual(inputs.x.shape, x.shape)
        self.assertEqual(inputs.x_sf.shape, (6, 40))
        self.assertEqual(inputs.shared_l1_acts_sf.shape, (6, 40))
        self.assertEqual(inputs.block_m, 32)
        inputs.x.fill_(2)
        inputs.x_sf.fill_(7)
        inputs.topk_idx.fill_(3)
        self.assertTrue(torch.all(strategy._mega_buf.x[:6].float() == 2))
        self.assertTrue(torch.all(strategy._mega_buf.x_sf[:6] == 7))
        self.assertTrue(torch.all(strategy._mega_buf.topk_idx[:6] == 3))
        self.assertEqual(
            inputs.shared_l1_acts_sf.stride(),
            strategy._mega_buf.shared_l1_acts_sf.stride(),
        )

    def test_capacity_and_shared_recipe_keep_the_old_path(self):
        strategy = make_strategy(capacity=8)
        for x in (
            torch.empty((0, _DIM), dtype=torch.bfloat16),
            torch.empty((9, _DIM), dtype=torch.bfloat16),
            torch.empty((6, _DIM), dtype=torch.float32),
            torch.empty((6, _DIM * 2), dtype=torch.bfloat16)[:, ::2],
        ):
            with self.subTest(shape=x.shape, dtype=x.dtype):
                self.assertIsNone(strategy.prepare_mhc_inputs(x))
        strategy.cfg.shared_fp8_block_size = 128
        self.assertIsNone(
            strategy.prepare_mhc_inputs(torch.empty((6, _DIM), dtype=torch.bfloat16))
        )

    def test_moe_only_prepares_for_the_eligible_decode_router(self):
        moe = make_moe(make_strategy())
        x = torch.zeros((1, 6, _DIM), dtype=torch.bfloat16)
        with patch.object(mega_gate, "can_use_mega_gate", return_value=True):
            self.assertIsNotNone(moe.prepare_mhc_inputs(x))
            for attr in ("_is_decode_role", "_gate_pack_static"):
                with patch.object(moe, attr, False):
                    self.assertIsNone(moe.prepare_mhc_inputs(x))
            with patch.object(moe, "_has_visual_tokens", True):
                self.assertIsNone(moe.prepare_mhc_inputs(x))
        with patch.object(mega_gate, "can_use_mega_gate", return_value=False):
            self.assertIsNone(moe.prepare_mhc_inputs(x))

    def test_prepared_forward_consumes_current_values_without_packing(self):
        strategy = make_strategy()
        moe = make_moe(strategy)
        x = torch.ones((1, 6, _DIM), dtype=torch.bfloat16)
        inputs = strategy.prepare_mhc_inputs(x.view(6, _DIM))
        strategy._input_packer = Mock()
        strategy._input_packer.pack.side_effect = AssertionError("repeated quant")

        def gate_kernel(x, *args, out, **kwargs):
            indices, weights = out
            indices.fill_(1)
            weights.copy_(x[:, :6].float())

        def launch(y, tokens, device):
            # Consumer-visible result depends on freshly written activations
            # and routing metadata; stale buffers or reversed out fail.
            buf = strategy._mega_buf
            y.copy_(buf.x[:tokens].float() * buf.topk_weights[:tokens].sum(-1, True))

        strategy._launch = launch
        with patch.object(mega_gate, "_native_gate", return_value=gate_kernel):
            for value in (1, 2):
                x.fill_(value)
                inputs.x.fill_(value + 1)
                actual = moe(x, torch.arange(6), mhc_inputs=inputs)
                torch.testing.assert_close(
                    actual, torch.full_like(x, value * (value + 1) * 6)
                )
                self.assertTrue(torch.all(inputs.topk_idx == 1))

    def test_foreign_or_wrong_sized_inputs_fail_before_routing(self):
        strategy = make_strategy()
        moe = make_moe(strategy)
        x = torch.ones((6, _DIM), dtype=torch.bfloat16)
        inputs = strategy.prepare_mhc_inputs(x)
        with patch.object(mega_gate, "_native_gate") as kernel:
            with self.assertRaises(ValueError):
                moe(x, torch.arange(6), mhc_inputs=replace(inputs, owner=object()))
            with self.assertRaises(ValueError):
                moe(x[:1], torch.arange(1), mhc_inputs=inputs)
        kernel.assert_not_called()

    def test_mhc_writes_external_outputs_and_keeps_bf16_and_state(self):
        case = make_case((1, 6), "cpu")
        inputs = make_strategy().prepare_mhc_inputs(case["attn_out"].view(6, _DIM))

        def operation(**kwargs):
            self.assertNotIn("y_gemm_sf", kwargs)
            self.assertIs(kwargs["y_fp8"], inputs.x)
            self.assertIs(kwargs["y_routed_sf"], inputs.x_sf)
            self.assertIs(kwargs["y_shared_sf"], inputs.shared_l1_acts_sf)
            self.assertEqual(kwargs["shared_sf_block_m"], 32)
            kwargs["y_fp8"].fill_(2)
            kwargs["y_routed_sf"].fill_(7)
            kwargs["y_bf16"].fill_(3)
            kwargs["new_prev_mix"].fill_(0.25)

        with (
            patch.object(v41_mega_mhc, "is_supported", return_value=True),
            patch.object(v41_mega_mhc, "_get_mega_mhc", return_value=operation),
            patch.object(v41_mega_mhc, "_stream_key", return_value=(-1, -1)),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            result = v41_mega_mhc.try_fused_post_pre(**case, moe_inputs=inputs)
        self.assertEqual(len(result), 4)
        self.assertTrue(torch.all(result[1] == 3))
        self.assertTrue(torch.all(inputs.x.float() == 2))
        self.assertTrue(torch.all(case["next_hc"].pre_mix_out == 0.25))

    def test_mhc_rejects_padded_or_conflicting_output_contracts(self):
        case = make_case((1, 6), "cpu")
        inputs = make_strategy().prepare_mhc_inputs(case["attn_out"].view(6, _DIM))
        with patch.object(v41_mega_mhc, "is_supported", return_value=True):
            with self.assertRaises(ValueError):
                v41_mega_mhc.try_fused_post_pre(
                    **case, output_fp8=True, moe_inputs=inputs
                )
            self.assertIsNone(
                v41_mega_mhc.try_fused_post_pre(
                    **case, original_tokens=512, moe_inputs=inputs
                )
            )

    def test_decode_block_consumes_only_successfully_produced_inputs(self):
        from rtp_llm.models_py.modules.dsv41.test.test_v41_cross_layer_mhc import (
            make_blocks,
            reference_transition,
        )

        hidden = torch.randn((1, 6, 4, _DIM), dtype=torch.bfloat16)
        ids = torch.arange(6).view(1, 6)
        results = []
        for fused in (False, True):
            block = make_blocks(dim=_DIM, count=1)[0]
            strategy = make_strategy()
            block.ffn = make_moe(strategy)
            packs = []

            def pack(x, weights, indices, buf, tokens, block_m):
                packs.append(tokens)
                buf.x[:tokens].copy_(x)

            strategy._input_packer = SimpleNamespace(pack=pack)
            strategy._launch = lambda y, tokens, device: y.copy_(
                strategy._mega_buf.x[:tokens].float()
            )

            def gate_kernel(x, *args, out, **kwargs):
                out[0].fill_(0)
                out[1].fill_(1 / 6)

            def mhc(*args, original_tokens=None, moe_inputs=None):
                if not fused:
                    return None
                result = reference_transition(*args)
                if moe_inputs is not None:
                    moe_inputs.x.copy_(result[1].view(6, _DIM))
                return result

            with (
                patch.object(mega_gate, "can_use_mega_gate", return_value=True),
                patch.object(mega_gate, "_native_gate", return_value=gate_kernel),
                patch.object(v41_mega_mhc, "try_fused_post_pre", side_effect=mhc),
            ):
                results.append(block.forward_decode(hidden.clone(), None, ids))
            self.assertEqual(packs, [] if fused else [6])
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MHCPreparedMoECudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("MegaMHC requires Blackwell")

    @torch.inference_mode()
    def test_quant_and_both_scale_layouts_match_existing_packer(self):
        from rtp_llm.models_py.modules.dsv41.moe._mega_se_input_pack_triton import (
            fused_pack_mega_moe_se_inputs,
        )

        for rows, block_m in ((1, 16), (6, 32), (160, 32), (192, 64)):
            with self.subTest(rows=rows, block_m=block_m):
                case = make_case((1, rows))
                strategy = make_strategy(device="cuda", block_m=block_m)
                inputs = strategy.prepare_mhc_inputs(case["attn_out"].view(rows, _DIM))
                reference = make_strategy(device="cuda", block_m=block_m)._mega_buf
                for scale in (1.0, 1e-8, 0.0):
                    case["norm"].weight.fill_(scale)
                    baseline = v41_mega_mhc.try_fused_post_pre(**case)
                    actual = v41_mega_mhc.try_fused_post_pre(**case, moe_inputs=inputs)
                    for got, ref in zip(actual, baseline):
                        torch.testing.assert_close(got, ref, rtol=0, atol=0)
                    fused_pack_mega_moe_se_inputs(
                        baseline[1].view(rows, _DIM),
                        torch.ones((rows, 6), device="cuda"),
                        torch.zeros((rows, 6), device="cuda", dtype=torch.int64),
                        reference.x[:rows],
                        reference.x_sf[:rows],
                        reference.shared_l1_acts_sf,
                        reference.topk_idx[:rows],
                        reference.topk_weights[:rows],
                        block_m,
                    )
                    torch.testing.assert_close(
                        inputs.x.view(torch.uint8),
                        reference.x[:rows].view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        inputs.x_sf, reference.x_sf[:rows], rtol=0, atol=0
                    )
                    # Only logical token scales are consumed. The shared
                    # allocation uses UTCCP rows, not ordinary tensor rows.
                    t = torch.arange(rows, device="cuda")
                    within = t % block_m
                    sf_rows = (
                        t // block_m * ((block_m + 127) // 128 * 128)
                        + within // 128 * 128
                        + within % 32 * 4
                        + within % 128 // 32
                    )
                    capacity = strategy._mega_buf.shared_l1_acts_sf.stride(1)
                    got = torch.as_strided(
                        inputs.shared_l1_acts_sf, (capacity, 40), (1, capacity)
                    )
                    ref = torch.as_strided(
                        reference.shared_l1_acts_sf, (capacity, 40), (1, capacity)
                    )
                    torch.testing.assert_close(
                        got[sf_rows], ref[sf_rows], rtol=0, atol=0
                    )

    @torch.inference_mode()
    def test_real_mega_moe_and_graph_replay_match_packed_path(self):
        """Use native routed/shared GEMMs, including shared-buffer reuse."""
        import deep_gemm
        import torch.distributed as dist
        from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8

        from rtp_llm.models_py.modules.dsv41.moe.mega_se_buf import (
            mega_moe_se_shared_recipe_kwargs,
        )
        from rtp_llm.models_py.modules.dsv41.moe.mega_se_input_packer import (
            FusedMegaMoeSEInputPacker,
        )
        from rtp_llm.models_py.modules.dsv41.moe.strategies.base import MoeCfg

        if dist.is_initialized():
            self.skipTest("This test owns a single-rank NCCL group")
        torch.manual_seed(4141)
        torch.cuda.set_device(torch.cuda.current_device())
        with tempfile.TemporaryDirectory() as tmp:
            dist.init_process_group(
                "nccl",
                init_method=f"file://{tmp}/store",
                rank=0,
                world_size=1,
                device_id=torch.device("cuda", torch.cuda.current_device()),
            )
            try:
                experts, inter, topk, capacity = 32, 1536, 6, 256
                strategy = MegaMoEStrategySE.__new__(MegaMoEStrategySE)
                torch.nn.Module.__init__(strategy)
                strategy.cfg = MoeCfg(
                    layer_id=0,
                    dim=_DIM,
                    moe_inter_dim=inter,
                    n_routed_experts=experts,
                    n_activated_experts=topk,
                    swiglu_limit=10.0,
                    ep_size=1,
                    ep_rank=0,
                    n_local_experts=experts,
                    local_expert_start=0,
                    local_expert_end=experts,
                    max_tokens_per_rank=capacity,
                    shared_fp8_block_size=32,
                    is_decode_role=True,
                )
                strategy._mega_group = dist.group.WORLD
                strategy._mega_buf = deep_gemm.get_symm_buffer_for_mega_moe(
                    group=dist.group.WORLD,
                    num_experts=experts,
                    num_max_tokens_per_rank=capacity,
                    num_topk=topk,
                    hidden=_DIM,
                    intermediate_hidden=inter,
                    num_shared_experts=1,
                    mma_type="fp8xfp4",
                    activation="swiglu",
                )
                strategy._mega_y = torch.empty(
                    (capacity, _DIM), device="cuda", dtype=torch.bfloat16
                )
                strategy._input_packer = FusedMegaMoeSEInputPacker()
                strategy._shared_recipe_kwargs = mega_moe_se_shared_recipe_kwargs(
                    deep_gemm.fp8_fp4_mega_moe, (1, 1, 32)
                )

                def weights(n, k, routed):
                    count = experts if routed else 1
                    quantized, scales = [], []
                    for _ in range(count):
                        value = (torch.randn((n, k), device="cuda") * 0.01).bfloat16()
                        cast = (
                            per_token_cast_to_fp4 if routed else per_token_cast_to_fp8
                        )
                        q, sf = cast(value, use_ue8m0=True, gran_k=32)
                        quantized.append(q)
                        scales.append(sf)
                    q = torch.stack(quantized) if routed else quantized[0]
                    sf = torch.stack(scales) if routed else scales[0]
                    sf = deep_gemm.transform_sf_into_required_layout(
                        sf, n, k, (1, 32), count if routed else None
                    )
                    return q, sf

                (strategy._mega_l1_w, strategy._mega_l1_sf), (
                    strategy._mega_l2_w,
                    strategy._mega_l2_sf,
                ) = deep_gemm.transform_weights_for_mega_moe(
                    weights(inter * 2, _DIM, True), weights(_DIM, inter, True)
                )
                (strategy._se_l1_w, strategy._se_l1_sf), (
                    strategy._se_l2_w,
                    strategy._se_l2_sf,
                ) = deep_gemm.transform_weights_for_mega_moe(
                    weights(inter * 2, _DIM, False), weights(_DIM, inter, False)
                )
                moe = make_moe(strategy)
                gate_weight = (
                    torch.randn((experts, _DIM), device="cuda") * 0.01
                ).bfloat16()
                moe.gate = SimpleNamespace(
                    hash=False,
                    weight=gate_weight,
                    bias=torch.zeros(experts, device="cuda"),
                    score_func="sqrtsoftplus",
                    topk=topk,
                    route_scale=2.5,
                    _weight_bf16=lambda: gate_weight,
                )
                buf = strategy._mega_buf
                # Different token counts reuse one symmetric allocation.
                for rows in (192, 6, 160):
                    case = make_case((1, rows))
                    ids = torch.arange(rows, device="cuda")

                    def packed():
                        out = v41_mega_mhc.try_fused_post_pre(**case)
                        self.assertIsNotNone(out)
                        return moe(out[1], ids)

                    def direct():
                        inputs = moe.prepare_mhc_inputs(case["attn_out"])
                        self.assertIsNotNone(inputs)
                        out = v41_mega_mhc.try_fused_post_pre(**case, moe_inputs=inputs)
                        self.assertIsNotNone(out)
                        return moe(out[1], ids, mhc_inputs=inputs)

                    expected = packed().clone()
                    expected_ids = buf.topk_idx[:rows].clone()
                    expected_weights = buf.topk_weights[:rows].clone()
                    # Simulate arbitrary old scales in padding rows; only
                    # token scales may affect the native shared expert.
                    block_m = strategy._block_m(rows)
                    sf_rows = ((rows + block_m - 1) // block_m) * (
                        (block_m + 127) // 128 * 128
                    )
                    torch.as_strided(
                        buf.shared_l1_acts_sf,
                        (sf_rows, 40),
                        buf.shared_l1_acts_sf.stride(),
                    ).fill_(0x7F7F7F7F)
                    actual = direct().clone()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(buf.topk_idx[:rows], expected_ids)
                    torch.testing.assert_close(
                        buf.topk_weights[:rows], expected_weights
                    )

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        direct()  # Initialize native per-stream barriers before capture.
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = direct()
                    for _ in range(3):
                        case["attn_out"].add_(0.0625)
                        case["previous"].pre_mix_out.mul_(0.97)
                        graph.replay()
                        actual = captured.clone()
                        expected = packed().clone()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            finally:
                try:
                    deep_gemm.destroy_localizer()
                finally:
                    dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
