"""SM120 loader assembly and numerical bindings, without native scheduling."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_attention_projection_hardware as projections,
)
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_expert_hardware as shared_fixture,
)
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_routed_hardware as routed_fixture,
)
from rtp_llm.models_py.modules.dsv4.v41_engram_stage import V41EngramStage
from rtp_llm.models_py.modules.dsv4.v41_model import build_v41_model_stage
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts
from rtp_llm.utils.model_weight import W


class V41ModelHardwareTest(unittest.TestCase):
    def test_full_width_stage_one_loader_bindings(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        previous = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)
        torch.backends.cuda.matmul.allow_tf32 = False
        reference, _, tensors = projections.fixture(1, full=True)
        path = (
            Path(__file__).resolve().parents[5]
            / "config/test/testdata/deepseek_v41_flash.json"
        )
        metadata = json.loads(path.read_text())
        metadata["text_config"]["n_routed_experts"] = 8
        config = V41Config.from_dict(metadata)
        for name in (
            W.v4_compressor_wkv,
            W.v4_compressor_norm,
            W.v41_indexer_wk,
            W.v41_indexer_k_norm,
        ):
            tensors.pop(name)
        for kind in ("attn", "ffn"):
            for suffix, shape in (
                ("fn", (24, 20480)),
                ("base", (24,)),
                ("scale", (3,)),
            ):
                tensors[getattr(W, "v4_hc_" + kind + "_" + suffix)] = (
                    torch.randn(*shape, device="cuda") * 0.01
                )
            tensors[getattr(W, "v4_" + kind + "_norm")] = torch.ones(
                5120, device="cuda", dtype=torch.bfloat16
            )
        tensors[W.v4_router_w] = (
            torch.randn(8, 5120, device="cuda") / 5120**0.5
        ).bfloat16()
        tensors[W.v4_router_bias] = torch.randn(8, device="cuda")
        for name, value in shared_fixture.V41ExpertHardwareTest.weights().items():
            tensors[getattr(W, "v4_shared_" + name)] = value
        shard = routed_fixture.V41RoutedHardwareTest.weights(2)
        for kind in ("w", "s"):
            gate, up = shard["w13_" + kind].chunk(2, dim=1)
            # Exercise the actual loader's signed packed FP4 and UE8M0 dtypes.
            dtype = torch.int8 if kind == "w" else torch.float8_e8m0fnu
            tensors[getattr(W, "v4_routed_w1_" + kind)] = gate.contiguous().view(dtype)
            tensors[getattr(W, "v4_routed_w3_" + kind)] = up.contiguous().view(dtype)
            tensors[getattr(W, "v4_routed_w2_" + kind)] = shard["w2_" + kind].view(
                dtype
            )
        layers = [{} for _ in range(40)]
        for source in config.layer_sources()[21:]:
            weights = dict(tensors)
            if not source.owns_index:
                for name in (
                    W.v4_indexer_wq_b_w,
                    W.v4_indexer_wq_b_s,
                    W.v4_indexer_weights_proj_w,
                ):
                    weights.pop(name)
            layers[source.layer_id] = weights
        weights = SimpleNamespace(
            weights=layers,
            global_weights={
                W.final_ln_gamma: torch.ones(5120, device="cuda", dtype=torch.bfloat16)
            },
        )
        engram = V41EngramStage(
            {},
            None,
            device=torch.device("cuda", torch.cuda.current_device()),
            max_host_bytes=0,
            max_weight_bytes=0,
        ).initialize()
        self.addCleanup(engram.close)
        loaded = build_v41_model_stage(
            config,
            weights,
            role="prefill",
            pp_rank=1,
            ep_rank=3,
            world_ranks=(4, 5, 6, 7),
            generation="assembly-hardware",
            process_group=object(),
            engram=engram,
            rows_per_rank=4,
            max_positions=256,
        )
        self.assertEqual(len(loaded.stage.blocks), 19)
        self.assertIsNone(loaded.embedding)
        x = torch.randn(3, 5120, device="cuda").bfloat16()
        positions = torch.tensor([0, 99, -999], device="cuda")
        valid = torch.tensor([True, True, False], device="cuda")
        x[-1] = float("nan")
        actual = loaded.stage.blocks[3].attention
        frame, expected = actual.prepare(x, positions, valid), reference.prepare(
            x, positions, valid
        )
        for name in ("query", "index_query", "index_weights"):
            torch.testing.assert_close(
                getattr(frame, name), getattr(expected, name), rtol=0, atol=0
            )
        attended = torch.randn(3, 64, 512, device="cuda").bfloat16()
        torch.testing.assert_close(
            actual.output(attended, frame),
            reference.output(attended, expected),
            rtol=0,
            atol=0,
        )
        local = loaded.stage.blocks[3].feed_forward.routed.local
        expected_local = V41LocalMoE(
            V41RoutedExperts(shard),
            expert_start_id=6,
            max_tokens=16,
            masked_token_limit=16,
        )
        route_ids = torch.tensor(
            [[7, 0, 3, 2, 6, 1], [0, 1, 2, 6, 4, 5], [-1] * 6],
            device="cuda",
            dtype=torch.int32,
        )
        route_weights = torch.tensor([[0.3] * 6, [0.7] * 6, [0.0] * 6], device="cuda")
        torch.testing.assert_close(
            local(x, route_weights, route_ids, valid_rows=valid),
            expected_local(x, route_weights, route_ids, valid_rows=valid),
            rtol=0,
            atol=0,
        )
        for op in loaded.stage.blocks:
            with self.assertRaisesRegex(RuntimeError, "admission"):
                op.feed_forward.routed.require_admitted()
            self.assertIn(W.v4_routed_w1_w, weights.weights[op.layer_id])


if __name__ == "__main__":
    unittest.main()
