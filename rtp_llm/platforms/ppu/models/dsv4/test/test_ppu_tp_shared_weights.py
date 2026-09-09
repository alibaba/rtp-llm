"""Exercise the real byte-preserving partition helper without GPU imports."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires Torch for tensor partition checks")
class SharedWeightPartitionTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "ppu_tp_shared_expert.py"
        node = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "shard_shared_weights"
        )
        scope = {"torch": torch}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope
        )
        self.shard = scope["shard_shared_weights"]
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
                halves = [p[key].view(torch.uint8).chunk(2, 0) for p in pieces]
                reconstructed = torch.cat(
                    [h[0] for h in halves] + [h[1] for h in halves], 0
                )
            else:
                reconstructed = torch.cat([p[key].view(torch.uint8) for p in pieces], 1)
            self.assertTrue(
                torch.equal(reconstructed, self.weights[key].view(torch.uint8))
            )

    def test_private_quarter_copies_do_not_mutate_or_alias_loader_weights(self):
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
        for kwargs in ({"tp_rank": 4}, {"tp_size": 2}, {"inter_dim": 640}):
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


if __name__ == "__main__":
    unittest.main()
