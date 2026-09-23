"""Native shared MegaMoE accuracy, graph and reload integration coverage.

Set DSV41_SHARED_TEST_CHECKPOINT to exercise real shared-expert weights.
torchrun with two ranks additionally checks an empty rank's participation.
DSV41_SHARED_TEST_CASE selects full, positive, zero_transition or empty_rank.
DSV41_SHARED_TEST_NODES=2 captures two calls using the same symmetric buffer.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.moe.shared_expert import W13SharedExpert
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.models_py.modules.dsv4.moe.strategies.mega import MegaMoEStrategy
from rtp_llm.models_py.modules.dsv4.moe.strategies.mega_se import MegaMoEStrategySE
from rtp_llm.models_py.modules.dsv4.test import test_mega_v41_compat as mega_compat
from rtp_llm.utils.model_weight import W


@unittest.skipUnless(torch.cuda.is_available(), "requires SM100 CUDA")
class V41MegaSharedCudaTest(unittest.TestCase):
    @staticmethod
    def _shared_weights():
        checkpoint = os.environ.get("DSV41_SHARED_TEST_CHECKPOINT")
        if checkpoint:
            from safetensors import safe_open

            root = Path(checkpoint)
            with (root / "model.safetensors.index.json").open() as stream:
                weight_map = json.load(stream)["weight_map"]
            parts = {}
            layer = int(os.environ.get("DSV41_SHARED_TEST_LAYER", "0"))
            for projection in ("w1", "w3", "w2"):
                for suffix in ("weight", "scale"):
                    name = f"layers.{layer}.ffn.shared_experts.{projection}.{suffix}"
                    with safe_open(root / weight_map[name], framework="pt") as file:
                        value = file.get_tensor(name).to("cuda")
                    parts[(projection, suffix)] = value.view(
                        torch.float8_e4m3fn
                        if suffix == "weight"
                        else torch.float8_e8m0fnu
                    )
            return {
                W.v4_shared_w13_w: torch.cat(
                    [parts[(p, "weight")].view(torch.uint8) for p in ("w1", "w3")]
                ).view(torch.float8_e4m3fn),
                W.v4_shared_w13_s: torch.cat(
                    [parts[(p, "scale")].view(torch.uint8) for p in ("w1", "w3")]
                ).view(torch.float8_e8m0fnu),
                W.v4_shared_w2_w: parts[("w2", "weight")],
                W.v4_shared_w2_s: parts[("w2", "scale")],
            }
        result = {}
        for wk, sk, n, k in (
            (W.v4_shared_w13_w, W.v4_shared_w13_s, 4608, 5120),
            (W.v4_shared_w2_w, W.v4_shared_w2_s, 5120, 2304),
        ):
            result[wk] = (torch.randn(n, k, device="cuda") * 0.025).to(
                torch.float8_e4m3fn
            )
            result[sk] = torch.randint(
                125, 128, ((n + 31) // 32, k // 32), device="cuda", dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
        return result

    @staticmethod
    def _metrics(actual, expected):
        delta = actual.float() - expected.float()
        return {
            "max_abs": delta.abs().max().item(),
            "mean_abs": delta.abs().mean().item(),
            "relative_rms": (
                delta.square().mean()
                / expected.float().square().mean().clamp_min(1e-20)
            )
            .sqrt()
            .item(),
            "cosine": F.cosine_similarity(
                actual.float().flatten(), expected.float().flatten(), dim=0
            ).item(),
            "different_fraction": (actual != expected).float().mean().item(),
        }

    @staticmethod
    def _capture(fn, calls_per_graph=1):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls_per_graph):
                output = fn()
        torch.cuda.synchronize()
        dist.barrier()
        return graph, output

    @staticmethod
    def _time_graph(graph):
        for _ in range(5):
            graph.replay()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        begin.record()
        for _ in range(50):
            graph.replay()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) * 1000 / 50

    @torch.inference_mode()
    def test_accuracy_graph_reload_and_empty_rank(self):
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        rank, world = int(os.environ.get("RANK", "0")), int(
            os.environ.get("WORLD_SIZE", "1")
        )
        experts = int(os.environ.get("DSV41_SHARED_TEST_EXPERTS", "8"))
        calls_per_graph = int(os.environ.get("DSV41_SHARED_TEST_NODES", "1"))
        self.assertIn(calls_per_graph, (1, 2))
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"WARM_UP": "0"}
        ):
            init = "env://" if world > 1 else "file://" + tmpdir + "/init"
            dist.init_process_group(
                "gloo", init_method=init, rank=rank, world_size=world
            )
            try:
                cfg = MoeCfg(
                    layer_id=0,
                    dim=5120,
                    moe_inter_dim=2304,
                    n_routed_experts=experts,
                    n_activated_experts=6,
                    swiglu_limit=10.0,
                    ep_size=world,
                    ep_rank=rank,
                    n_local_experts=experts // world,
                    local_expert_start=rank * (experts // world),
                    local_expert_end=(rank + 1) * (experts // world),
                    max_tokens_per_rank=64,
                    shared_fp8_block_size=32,
                    is_decode_role=True,
                )
                routed_weights = mega_compat.V41MegaCompatibilityTest._weights(
                    experts=experts // world
                )
                shared_weights = self._shared_weights()
                shared = W13SharedExpert(
                    5120,
                    2304,
                    {
                        "w13_w": shared_weights[W.v4_shared_w13_w],
                        "w13_s": shared_weights[W.v4_shared_w13_s],
                        "w2_w": shared_weights[W.v4_shared_w2_w],
                        "w2_s": shared_weights[W.v4_shared_w2_s],
                    },
                    swiglu_limit=10.0,
                )
                routed, fused = MegaMoEStrategy(cfg), MegaMoEStrategySE(cfg)
                routed.setup_weights(dict(routed_weights))
                routed.setup_runtime()
                fused.setup_weights({**routed_weights, **shared_weights})
                fused.setup_runtime()
                self.assertEqual(routed._mega_buf.num_shared_experts, 0)
                self.assertEqual(fused._mega_buf.num_shared_experts, 1)
                self.assertNotEqual(
                    routed._mega_buf.buffer.data_ptr(),
                    fused._mega_buf.buffer.data_ptr(),
                )
                self.assertNotEqual(routed._mega_y.data_ptr(), fused._mega_y.data_ptr())
                cases = [(n, n) for n in (4, 8, 24, 48)]
                if world > 1:
                    cases += [(4, 0)]
                route_scales = (0.0, 1.0, 2.0)
                case = os.environ.get("DSV41_SHARED_TEST_CASE", "full")
                if case == "positive":
                    cases, route_scales = [(4, 4)], (1.0, 2.0, 1.0)
                elif case == "zero_transition":
                    cases, route_scales = [(4, 4)], (0.0, 1.0)
                elif case == "empty_rank":
                    if world != 2:
                        self.skipTest("empty_rank requires exactly two ranks")
                    cases, route_scales = [(4, 0), (0, 4)], (1.0, 2.0, 1.0)
                elif case != "full":
                    raise ValueError(f"Unknown DSV41_SHARED_TEST_CASE={case!r}")
                for counts in cases:
                    tokens = counts[rank] if world > 1 else counts[0]
                    x = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
                    indices = (
                        torch.arange(6, device="cuda").expand(tokens, -1).contiguous()
                    )
                    weights = torch.full((tokens, 6), 1 / 6, device="cuda")

                    def baseline():
                        y = routed(x, weights, indices)
                        if tokens == 0:
                            return y
                        return (y.float() + shared(x).float()).to(torch.bfloat16)

                    def candidate():
                        return fused(x, weights, indices)

                    bg, by = self._capture(baseline, calls_per_graph)
                    fg, fy = self._capture(candidate, calls_per_graph)
                    for route_scale in route_scales:
                        torch.cuda.synchronize()
                        dist.barrier()
                        weights.fill_(route_scale / 6)
                        torch.cuda.synchronize()
                        dist.barrier()
                        bg.replay()
                        torch.cuda.synchronize()
                        dist.barrier()
                        old_snapshot = by.clone()
                        bg.replay()
                        torch.cuda.synchronize()
                        dist.barrier()
                        if not torch.equal(by, old_snapshot):
                            print(
                                "STABILITY_FAILURE "
                                + json.dumps(
                                    {
                                        "stage": "baseline",
                                        "rank": rank,
                                        "tokens": tokens,
                                        "route_scale": route_scale,
                                        "calls_per_graph": calls_per_graph,
                                        "x_first4": x[:, :4].float().cpu().tolist(),
                                        "indices": indices.cpu().tolist(),
                                        "packed_weights": routed._mega_buf.topk_weights[
                                            :tokens
                                        ]
                                        .cpu()
                                        .tolist(),
                                        "old_first2": old_snapshot[:, :2]
                                        .float()
                                        .cpu()
                                        .tolist(),
                                        "new_first2": by[:, :2].float().cpu().tolist(),
                                        "routed_first2": routed._mega_y[:tokens, :2]
                                        .float()
                                        .cpu()
                                        .tolist(),
                                    }
                                ),
                                flush=True,
                            )
                        torch.testing.assert_close(by, old_snapshot, rtol=0, atol=0)
                        fg.replay()
                        torch.cuda.synchronize()
                        dist.barrier()
                        if tokens:
                            metrics = self._metrics(fy, by)
                            record = {
                                "rank": rank,
                                "tokens": tokens,
                                "route_scale": route_scale,
                                "calls_per_graph": calls_per_graph,
                                "checkpoint": bool(
                                    os.environ.get("DSV41_SHARED_TEST_CHECKPOINT")
                                ),
                                **metrics,
                            }
                            print("SHARED_METRICS " + json.dumps(record), flush=True)
                            self.assertTrue(torch.isfinite(fy).all().item())
                            self.assertLess(metrics["relative_rms"], 0.05)
                            self.assertGreater(metrics["cosine"], 0.998)
                        snapshot = fy.clone()
                        fg.replay()
                        torch.cuda.synchronize()
                        dist.barrier()
                        torch.testing.assert_close(fy, snapshot, rtol=0, atol=0)
                    timing = {
                        "rank": rank,
                        "tokens": tokens,
                        "calls_per_graph": calls_per_graph,
                        "baseline_us": self._time_graph(bg),
                        "fused_us": self._time_graph(fg),
                    }
                    print("SHARED_TIMING " + json.dumps(timing), flush=True)
                    if counts == (4, 4):
                        names = (
                            "_mega_l1_w",
                            "_mega_l1_sf",
                            "_mega_l2_w",
                            "_mega_l2_sf",
                            "_se_l1_w",
                            "_se_l1_sf",
                            "_se_l2_w",
                            "_se_l2_sf",
                        )
                        addresses = [getattr(fused, name).data_ptr() for name in names]
                        snapshot = fy.clone()
                        fused.reload_routed_weights(
                            {**routed_weights, **shared_weights}
                        )
                        self.assertEqual(
                            addresses,
                            [getattr(fused, name).data_ptr() for name in names],
                        )
                        fg.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(fy, snapshot, rtol=0, atol=0)
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
