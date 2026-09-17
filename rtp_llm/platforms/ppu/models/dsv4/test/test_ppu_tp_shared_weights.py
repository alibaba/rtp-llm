"""Exercise the real byte-preserving partition helper without GPU imports."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires Torch for tensor partition checks")
class SharedWeightPartitionTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.platforms.ppu.models.dsv4.resources import (
            tp_moe_shared_fp32_preparation,
        )
        from rtp_llm.utils.model_weight import W

        self.names = dict(
            w13_w=W.v4_shared_w13_w,
            w13_s=W.v4_shared_w13_s,
            w2_w=W.v4_shared_w2_w,
            w2_s=W.v4_shared_w2_s,
        )
        self.preparation = tp_moe_shared_fp32_preparation()
        self.shard = lambda weights, dim, inter_dim, tp_size, tp_rank: {
            key: self.preparation.split_function(name, None)(
                weights[key], tp_size, tp_rank
            )
            for key, name in self.names.items()
        }
        self.dim, self.inter = 512, 1024
        torch.manual_seed(17)
        self.weights = {
            key: torch.randint(0, 255, shape, dtype=torch.uint8).view(dtype)
            for key, shape, dtype in (
                ("w13_w", (2048, 512), torch.float8_e4m3fn),
                ("w13_s", (16, 4), torch.float8_e8m0fnu),
                ("w2_w", (512, 1024), torch.float8_e4m3fn),
                ("w2_s", (4, 8), torch.float8_e8m0fnu),
            )
        }

    def test_rank_slices_reconstruct_each_original_byte_once(self):
        pieces = [
            self.shard(
                self.weights, dim=self.dim, inter_dim=self.inter, tp_size=4, tp_rank=r
            )
            for r in range(4)
        ]
        for key in self.weights:
            if key.startswith("w13"):
                halves = [
                    (p[key] if key.endswith("_s") else p[key].view(torch.uint8)).chunk(
                        2, 0
                    )
                    for p in pieces
                ]
                reconstructed = torch.cat(
                    [h[0] for h in halves] + [h[1] for h in halves], 0
                )
            else:
                reconstructed = torch.cat(
                    [
                        p[key] if key.endswith("_s") else p[key].view(torch.uint8)
                        for p in pieces
                    ],
                    1,
                )
            self.assertTrue(
                torch.equal(
                    reconstructed,
                    (
                        self.weights[key].to(torch.float32)
                        if key.endswith("_s")
                        else self.weights[key].view(torch.uint8)
                    ),
                )
            )

    def test_loader_quarters_do_not_retain_or_mutate_full_checkpoint_storage(self):
        before = {k: v.view(torch.uint8).clone() for k, v in self.weights.items()}
        pieces = self.shard(
            self.weights, dim=self.dim, inter_dim=self.inter, tp_size=4, tp_rank=2
        )
        for key, value in pieces.items():
            self.assertEqual(value.numel() * 4, self.weights[key].numel())
            self.assertTrue(value.is_contiguous())
            self.assertNotEqual(
                value.untyped_storage().data_ptr(),
                self.weights[key].untyped_storage().data_ptr(),
            )
            value.view(torch.uint8).fill_(0)
            self.assertTrue(
                torch.equal(before[key], self.weights[key].view(torch.uint8))
            )

    def test_invalid_partition_and_scale_geometry_fail_before_slicing(self):
        for kwargs in ({"tp_rank": 4}, {"tp_size": 2}):
            args = dict(dim=self.dim, inter_dim=self.inter, tp_size=4, tp_rank=0)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                self.shard(self.weights, **args)
        self.weights["w2_s"] = self.weights["w2_s"][:, :-1].contiguous()
        with self.assertRaises(ValueError):
            self.shard(
                self.weights, dim=self.dim, inter_dim=self.inter, tp_size=4, tp_rank=0
            )

    def test_builder_uses_logical_tp_rank_when_local_device_ranks_repeat(self):
        from rtp_llm.platforms.ppu.models.dsv4.pluggable_builders import build_moe_tp

        request = SimpleNamespace(
            module_id="rtp.dsv4.moe", metadata={"layer_id": 0}, path="v4.layers.0.ffn"
        )
        pieces = []
        for tp_rank, local_rank in enumerate((0, 1, 0, 1)):
            context = SimpleNamespace(
                selection=SimpleNamespace(
                    model_metadata={"hidden_size": self.dim, "tp_size": 4},
                    platform=SimpleNamespace(local_rank=local_rank),
                )
            )

            def construct(**kwargs):
                self.assertEqual(kwargs["tp_rank"], tp_rank)
                return self.shard(
                    self.weights,
                    dim=self.dim,
                    inter_dim=self.inter,
                    tp_size=kwargs["tp_size"],
                    tp_rank=kwargs["tp_rank"],
                )

            with patch(
                "rtp_llm.platforms.ppu.models.dsv4.ppu_tp_moe.PpuTPMoE",
                side_effect=construct,
            ):
                pieces.append(
                    build_moe_tp(
                        build_ctx=context,
                        request=request,
                        layer_id=0,
                        dim=self.dim,
                        tp_size=4,
                        tp_rank=tp_rank,
                    )
                )
        reconstructed = torch.cat([p["w2_w"].view(torch.uint8) for p in pieces], 1)
        self.assertTrue(
            torch.equal(reconstructed, self.weights["w2_w"].view(torch.uint8))
        )

    def test_module_uses_owner_buffers_and_loader_updates_reach_gemm(self):
        from rtp_llm.model_loader.model_weight_info import ModelWeights
        from rtp_llm.model_loader.weight_module import AtomicWeight
        from rtp_llm.platforms.ppu.models.dsv4.ppu_tp_shared_expert import (
            PpuTPSharedExpert,
        )
        from rtp_llm.platforms.ppu.modules.linear import fp8_linear
        from rtp_llm.utils.model_weight import identity

        config = SimpleNamespace(
            tp_size=4,
            tp_rank=2,
            ep_size=1,
            ep_rank=0,
            dp_size=1,
            dp_rank=0,
            ffn_tp_rank=2,
            ffn_tp_size=4,
            hidden_size=self.dim,
            head_num=4,
            head_num_kv=4,
            size_per_head=128,
            moe_pure_tp_mode=True,
            bit=8,
            weight_preparation=self.preparation,
            exported_device=SimpleNamespace(maybe_rewrite_weight_by_key=lambda _, t: t),
        )
        owner = ModelWeights(1, "cpu", torch.bfloat16)
        descriptors = {
            key: AtomicWeight(name, [], identity) for key, name in self.names.items()
        }
        for key, desc in descriptors.items():
            for name, tensor in desc.update(self.weights[key], "cpu", config).items():
                owner.set_layer_weight(0, name, tensor)
        expert_weights = {
            key: owner.weights[0][name] for key, name in self.names.items()
        }
        seen = []

        def gemm(lhs, rhs, out):
            seen.append(rhs)
            out.zero_()

        with patch.object(fp8_linear, "_require_m890p"), patch.object(
            fp8_linear, "_resolve_deep_gemm_symbol", return_value=gemm
        ):
            expert = PpuTPSharedExpert(
                self.dim,
                self.inter,
                expert_weights,
                tp_size=4,
                tp_rank=2,
                platform_provider=SimpleNamespace(_bool=lambda _, default: default),
            )
            for key, desc in descriptors.items():
                name = self.names[key]
                linear = getattr(expert, key.split("_")[0])
                buffer = linear.weight_scale if key.endswith("_s") else linear.weight
                self.assertIs(buffer, owner.weights[0][name])
                self.assertEqual(buffer.numel() * 4, self.weights[key].numel())
                new_raw = torch.full_like(
                    self.weights[key].view(torch.uint8),
                    130 if key.endswith("_s") else 32,
                ).view(self.weights[key].dtype)
                updated = desc.update(new_raw, "cpu", config)[name]
                owner.update_layer_weight(0, name, updated)
                torch.testing.assert_close(
                    buffer.view(torch.uint8), updated.view(torch.uint8)
                )
            for name in ("w13", "w2"):
                linear = getattr(expert, name)
                lhs = torch.zeros((2, linear.k), dtype=torch.float8_e4m3fn)
                scales = torch.ones((2, linear.k // 128))
                with patch.object(
                    fp8_linear,
                    "quantize_ppu_fp8_activation",
                    return_value=(lhs, scales),
                ):
                    linear(torch.ones((2, linear.k), dtype=torch.bfloat16))
                self.assertIs(seen[-1][0], owner.weights[0][self.names[name + "_w"]])
                self.assertIs(seen[-1][1], owner.weights[0][self.names[name + "_s"]])
                self.assertTrue(torch.all(seen[-1][1] == 8))

    def test_prepared_scale_rejects_invalid_dtype_shape_and_contiguity(self):
        from rtp_llm.platforms.ppu.modules.linear import fp8_linear

        weight = torch.zeros((256, 256), dtype=torch.float8_e4m3fn)
        for scale, error in (
            (torch.ones(2, 2, dtype=torch.bfloat16), TypeError),
            (torch.ones(2, 1), ValueError),
            (torch.ones(2, 2).t(), ValueError),
        ):

            def check_contiguous(tensor, _name):
                if not tensor.is_contiguous():
                    raise ValueError("must be contiguous")

            with patch.object(
                fp8_linear, "_require_m890p", side_effect=check_contiguous
            ):
                with self.assertRaises(error):
                    fp8_linear.PpuFp8Linear(weight, scale, scale_is_prepared=True)


if __name__ == "__main__":
    unittest.main()
