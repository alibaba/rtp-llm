"""Check shared packing math and the two recipes' non-finite policies."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.mega_moe import MegaMoeExecutionPolicy
from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    prepare_fp4_weight_scale_for_deepgemm,
    prepare_fp8_weight_scale_for_deepgemm,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
    FusedMegaMoEInputPacker,
    TorchMegaMoEInputPacker,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_fp8_scale import (
    shared_fp8_scale_row_indices,
    stage_shared_fp8_input_scales,
)
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W


def executor_packer(*, prepared=False):
    executor = MegaMoeExecutor.__new__(MegaMoeExecutor)
    torch.nn.Module.__init__(executor)
    executor.cfg = SimpleNamespace(
        mega_moe_policy=MegaMoeExecutionPolicy() if prepared else None
    )
    return executor._new_input_packer()


class WeightScaleTest(unittest.TestCase):
    def test_executor_preserves_model_packer_policy(self):
        with patch.dict(
            os.environ, {"MEGA_MOE_INPUT_PACKER": "torch", "MOE_STRICT_FUSED": "0"}
        ):
            ordinary = executor_packer()
            prepared = executor_packer(prepared=True)
        self.assertIsInstance(ordinary, TorchMegaMoEInputPacker)
        self.assertIsInstance(prepared, FusedMegaMoEInputPacker)
        self.assertIsNotNone(prepared.pack_inputs)

    def test_fp4_raw_and_float_scales_use_same_layout(self):
        scales = torch.ones((2, 128, 8), dtype=torch.float32)
        with patch("deep_gemm.transform_sf_into_required_layout") as transform:
            prepare_fp4_weight_scale_for_deepgemm(scales, 128, 256, 2)
            actual, *layout = transform.call_args.args
            torch.testing.assert_close(actual, scales)
            self.assertEqual(layout, [128, 256, (1, 32), 2])
            prepare_fp4_weight_scale_for_deepgemm(
                scales.to(torch.float8_e8m0fnu), 128, 256, 2
            )
            torch.testing.assert_close(transform.call_args.args[0], scales)

    def test_fp8_shape_validation_and_packed_passthrough(self):
        packed = torch.zeros((128, 4), dtype=torch.int32)
        for prepare in (
            prepare_fp4_weight_scale_for_deepgemm,
            prepare_fp8_weight_scale_for_deepgemm,
        ):
            self.assertIs(prepare(packed, 128, 256), packed)
        with self.assertRaises(ValueError):
            prepare_fp8_weight_scale_for_deepgemm(torch.ones(128, 8), 128, 256)
        with patch("deep_gemm.transform_sf_into_required_layout") as transform:
            prepare_fp8_weight_scale_for_deepgemm(torch.ones(1, 2), 128, 256)
            self.assertEqual(transform.call_args.args[1:], (128, 256, (128, 128)))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class InputPackPolicyTest(unittest.TestCase):
    def test_fp8_to_fp4_conversion_preserves_raw_scale_layouts(self):
        from deep_gemm.utils import per_token_cast_to_fp4

        from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
            convert_fp8_moe_to_fp4_ue8m0,
        )

        for rows in (128, 144):
            weight = torch.randn(2, rows, 256, device="cuda").to(torch.float8_e4m3fn)
            for per_block in (False, True):
                scale_rows = (rows + 127) // 128 if per_block else rows
                scale = (
                    torch.randint(-3, 3, (2, scale_rows, 2), device="cuda")
                    .float()
                    .exp2()
                )
                expanded = (
                    scale.repeat_interleave(128, dim=1)[:, :rows]
                    if per_block
                    else scale
                )
                bf16 = (weight.float() * expanded.repeat_interleave(128, dim=2)).to(
                    torch.bfloat16
                )
                reference = [
                    per_token_cast_to_fp4(w, use_ue8m0=True, gran_k=32) for w in bf16
                ]
                expected = tuple(
                    torch.stack([pair[i] for pair in reference]) for i in (0, 1)
                )
                for dtype in (torch.float32, torch.float8_e8m0fnu):
                    for flattened in (False, True):
                        with self.subTest(
                            rows=rows,
                            per_block=per_block,
                            dtype=dtype,
                            flattened=flattened,
                        ):
                            raw_scale = scale.to(dtype)
                            if flattened:
                                raw_scale = raw_scale.flatten()
                            actual = convert_fp8_moe_to_fp4_ue8m0(weight, raw_scale)
                            for value, target in zip(actual, expected):
                                torch.testing.assert_close(
                                    value, target, rtol=0, atol=0
                                )
        with self.assertRaisesRegex(ValueError, "raw floating-point"):
            convert_fp8_moe_to_fp4_ue8m0(weight, scale.to(torch.int32))

    def test_fp8_constructor_preserves_strided_e8m0_checkpoint_scales(self):
        config = SimpleNamespace(
            layer_id=0,
            moe_strategy="mega_moe_fp8",
            dim=256,
            moe_inter_dim=256,
            n_local_experts=2,
            max_generate_batch_size=8,
            model_config=SimpleNamespace(max_seq_len=128, gen_num_per_cycle=0),
            parallelism_config=SimpleNamespace(
                role_type=RoleType.PDFUSION,
                prefill_cp_config=SimpleNamespace(is_enabled=lambda: False),
            ),
            moe_config=SimpleNamespace(ll_num_max_token=0),
        )
        package = "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.weights"
        for layout in ("up_gate", "gate_up"):
            with self.subTest(layout=layout):
                config.mega_moe_policy = MegaMoeExecutionPolicy(w1_layout=layout)
                scales = torch.arange(120, 136, dtype=torch.uint8, device="cuda")
                scales = scales.reshape(2, 4, 2).view(torch.float8_e8m0fnu)
                first, second = scales.chunk(2, dim=1)
                self.assertFalse(first.is_contiguous())
                weights = {
                    W.moe_w1: torch.ones(2, 512, 256, device="cuda").to(
                        torch.float8_e4m3fn
                    ),
                    W.moe_s1: scales,
                    W.moe_w2: torch.ones(2, 256, 256, device="cuda").to(
                        torch.float8_e4m3fn
                    ),
                    W.moe_s2: torch.ones(2, 2, 2, device="cuda").to(
                        torch.float8_e8m0fnu
                    ),
                }
                # Keep CUDA concatenation real: the old contiguous wrapper
                # hid unsupported e8m0 cat dispatch for these checkpoint views.
                with patch(
                    package + ".prepare_fp8_weight_scale_for_deepgemm",
                    side_effect=lambda scale, *args: scale,
                ) as prepare, patch(
                    "deep_gemm.transform_weights_for_mega_moe_fp8",
                    side_effect=lambda l1, l2: (l1, l2),
                ), patch.object(
                    MegaMoeExecutor, "_setup_buffer_and_warmup"
                ):
                    MegaMoeExecutor(config, object(), weights)
                expected = torch.cat(
                    (
                        (second.float(), first.float())
                        if layout == "up_gate"
                        else (first.float(), second.float())
                    ),
                    dim=1,
                )
                actual = prepare.call_args_list[0].args[0]
                self.assertEqual(actual.dtype, torch.float8_e8m0fnu)
                torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)

    def buffer(self, tokens, dim=256, topk=8):
        # Non-contiguous row strides also occur in symmetric-memory views.
        return SimpleNamespace(
            x=torch.empty(tokens, dim * 2, dtype=torch.float8_e4m3fn, device="cuda")[
                :, :dim
            ],
            x_sf=torch.empty(tokens, dim // 64, dtype=torch.int32, device="cuda")[
                :, : dim // 128
            ],
            topk_idx=torch.empty(tokens, topk, dtype=torch.int64, device="cuda"),
            topk_weights=torch.empty(tokens, topk, dtype=torch.float32, device="cuda"),
            shared_l1_acts_sf=torch.full(
                (4096, dim // 128), -1, dtype=torch.int32, device="cuda"
            ),
        )

    def test_finite_inputs_match_for_decode_prefill_and_empty_rank(self):
        for tokens in (0, 1, 17, 2049):
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, 512, dtype=torch.bfloat16, device="cuda")[
                    :, :256
                ]
                weights = torch.rand(tokens, 8, device="cuda")
                ids = torch.randint(0, 256, (tokens, 8), device="cuda")
                ordinary, prepared = self.buffer(tokens), self.buffer(tokens)
                executor_packer().pack(x, weights, ids, ordinary, tokens)
                executor_packer(prepared=True).pack(
                    x, weights, ids.int(), prepared, tokens
                )
                for name in ("x", "x_sf", "topk_idx", "topk_weights"):
                    a, b = getattr(ordinary, name), getattr(prepared, name)
                    if name == "x":
                        a, b = a.view(torch.uint8), b.view(torch.uint8)
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(prepared.topk_idx, ids)
                torch.testing.assert_close(prepared.topk_weights, weights)

    def test_launch_policy_keeps_glm_fixed_and_native_environment_controls(self):
        module = "rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack"
        for tokens in (1, 2049):
            x = torch.ones(tokens, 256, dtype=torch.bfloat16, device="cuda")
            weights = torch.ones(tokens, 8, device="cuda")
            ids = torch.zeros(tokens, 8, dtype=torch.int64, device="cuda")
            for prepared in (False, True):
                with self.subTest(tokens=tokens, prepared=prepared):
                    buf = self.buffer(tokens)
                    indices = ids.int() if prepared else ids
                    env = {
                        "MEGA_MOE_INPUT_PACKER": "fused",
                        "MEGA_MOE_INPUT_PACKER_IMPL": (
                            "legacy" if prepared else "optimized"
                        ),
                        "MEGA_MOE_PACK_BLOCK_M": "invalid" if prepared else "4",
                    }
                    with patch.dict(os.environ, env), patch(
                        module + "._pack_mega_moe_inputs_optimized_kernel"
                    ) as kernel:
                        executor_packer(prepared=prepared).pack(
                            x, weights, indices, buf, tokens
                        )
                    launch = kernel.__getitem__.return_value
                    launch.assert_called_once()
                    self.assertEqual(
                        launch.call_args.kwargs["BLOCK_M"],
                        (8 if tokens >= 2048 else 2) if prepared else 4,
                    )
                    self.assertEqual(
                        launch.call_args.kwargs["SANITIZE_NON_FINITE"], not prepared
                    )
                    self.assertIs(launch.call_args.args[2], indices)
                    self.assertEqual(buf.topk_idx.dtype, torch.int64)
        with patch.dict(
            os.environ,
            {
                "MEGA_MOE_INPUT_PACKER": "fused",
                "MEGA_MOE_INPUT_PACKER_IMPL": "optimized",
            },
        ):
            with self.assertRaisesRegex(ValueError, "indices must be torch.int64"):
                executor_packer().pack(
                    x, weights, ids.int(), self.buffer(tokens), tokens
                )

    def test_non_finite_policy_is_preserved(self):
        x = torch.ones(1, 256, dtype=torch.bfloat16, device="cuda")
        x[0, :3] = torch.tensor(
            [float("nan"), float("inf"), -float("inf")], device="cuda"
        )
        weights = torch.ones(1, 8, device="cuda")
        ids = torch.zeros(1, 8, dtype=torch.int64, device="cuda")
        ordinary, prepared, reference = self.buffer(1), self.buffer(1), self.buffer(1)
        executor_packer().pack(x, weights, ids, ordinary, 1)
        executor_packer().pack(
            torch.nan_to_num(x, nan=0, posinf=0, neginf=0), weights, ids, reference, 1
        )
        executor_packer(prepared=True).pack(x, weights, ids, prepared, 1)
        torch.testing.assert_close(
            ordinary.x.view(torch.uint8), reference.x.view(torch.uint8)
        )
        torch_reference = self.buffer(1)
        with patch.dict(
            os.environ, {"MEGA_MOE_INPUT_PACKER": "torch", "MOE_STRICT_FUSED": "0"}
        ):
            executor_packer().pack(x, weights, ids, torch_reference, 1)
        torch.testing.assert_close(
            torch_reference.x.view(torch.uint8), reference.x.view(torch.uint8)
        )
        torch.testing.assert_close(
            prepared.x.float()[0, :3],
            torch.tensor([448.0, 448.0, -448.0], device="cuda"),
        )

    def test_shared_scale_padding_after_batch_shrink_and_graph_replay(self):
        packer = executor_packer(prepared=True)
        buf = self.buffer(193)
        for tokens, block_m in ((193, 192), (17, 16), (0, 16)):
            x = torch.ones(tokens, 256, dtype=torch.bfloat16, device="cuda")
            weights = torch.ones(tokens, 8, device="cuda")
            ids = torch.zeros(tokens, 8, dtype=torch.int64, device="cuda")
            # Prewarm the specialization before graph capture.
            packer.pack(x, weights, ids, buf, tokens)
            stage_shared_fp8_input_scales(
                buf.x_sf, buf.shared_l1_acts_sf, tokens, block_m
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                packer.pack(x, weights, ids, buf, tokens)
                stage_shared_fp8_input_scales(
                    buf.x_sf, buf.shared_l1_acts_sf, tokens, block_m
                )
            buf.shared_l1_acts_sf.fill_(-1)
            graph.replay()
            active = (
                ((tokens + block_m - 1) // block_m) * ((block_m + 127) // 128) * 128
            )
            if tokens == 0:
                self.assertEqual(torch.count_nonzero(buf.shared_l1_acts_sf).item(), 0)
            else:
                expected = torch.zeros_like(buf.shared_l1_acts_sf[:active])
                rows = shared_fp8_scale_row_indices(tokens, block_m, x.device)
                expected.index_copy_(0, rows, buf.x_sf[:tokens])
                torch.testing.assert_close(buf.shared_l1_acts_sf[:active], expected)


if __name__ == "__main__":
    unittest.main()
