"""Upstream e4a9162f6f hardware checks, scoped to the target model port."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCHead, DelayedHCUnit


class DelayedHCCudaGraphTest(unittest.TestCase):
    def _units(self, weights, device, norm_eps):
        units = []
        for fn, base, scale in weights:
            units.append(
                DelayedHCUnit(
                    fn.to(device),
                    base.to(device),
                    scale.to(device),
                    dim=5120,
                    hc_mult=4,
                    hc_sinkhorn_iters=20,
                    norm_eps=norm_eps,
                    hc_eps=1e-6,
                )
            )
        units[1].set_previous(units[0])
        return units, DelayedHCHead(units[-1])

    @staticmethod
    def _run(x, units, head):
        residual = x.clone()
        for unit in units:
            y, post, comb = unit.pre(residual)
            residual = unit.post(y * 0.25, residual, post, comb)
        return head.head(residual)

    @torch.inference_mode()
    def test_eager_reference_and_graph_replay(self):
        torch.manual_seed(31)
        weights = [
            (
                torch.randn(24, 4 * 5120) * 0.003,
                torch.randn(24) * 0.1,
                torch.tensor([0.2, 0.4, 0.3]),
            )
            for _ in range(2)
        ]
        self._check_weights(weights, ((1, 1), (2, 6), (12,)))

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_CHECKPOINT"),
        "set DSV41_TEST_CHECKPOINT for real weights",
    )
    @torch.inference_mode()
    def test_checkpoint_atomic_loader_eager_and_graph(self):
        from safetensors import safe_open

        from rtp_llm.device.device_impl import CudaImpl
        from rtp_llm.models.deepseek_v41 import DeepSeekV41Weight
        from rtp_llm.utils.model_weight import W

        checkpoint = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        raw_config = json.loads((checkpoint / "config.json").read_text())
        text_config = {**raw_config, **raw_config.get("text_config", {})}
        norm_eps = float(text_config["rms_norm_eps"])
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

        class CheckpointSource:
            def load_tensor(self, name, dtype):
                with safe_open(
                    str(checkpoint / index[name]), framework="pt", device="cpu"
                ) as source:
                    return [source.get_tensor(name).to(dtype)]

        # Exercise the real AtomicWeight raw-load, EP/DP split and CUDA device
        # rewrite hooks. mHC tensors must remain FP32 despite BF16 compute.
        config = SimpleNamespace(
            compute_dtype=torch.bfloat16,
            merge_lora=False,
            tp_size=1,
            tp_rank=0,
            ep_size=4,
            ep_rank=0,
            dp_size=4,
            dp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            hidden_size=5120,
            head_num=64,
            head_num_kv=1,
            size_per_head=512,
            moe_pure_tp_mode=False,
            bit=8,
            exported_device=CudaImpl.__new__(CudaImpl),
        )
        for descriptor_class, layer_id in (
            (DeepSeekV41Weight, 0),
            (DeepSeekV41Weight, 39),
        ):
            with self.subTest(descriptor=descriptor_class.__name__, layer=layer_id):
                descriptor = descriptor_class.__new__(descriptor_class)
                loaded = {}
                for weight in descriptor._build_hc_residual(layer_id):
                    loaded.update(
                        weight.load(CheckpointSource(), layer_id, "cpu", config)
                    )
                weights = []
                for tag in ("attn", "ffn"):
                    fn, base, scale = (
                        loaded[getattr(W, f"v4_hc_{tag}_{part}")]
                        for part in ("fn", "base", "scale")
                    )
                    self.assertEqual(tuple(fn.shape), (24, 20480))
                    self.assertEqual(tuple(base.shape), (24,))
                    self.assertEqual(tuple(scale.shape), (3, 1))
                    for tensor in (fn, base, scale):
                        self.assertEqual(tensor.dtype, torch.float32)
                        self.assertTrue(tensor.is_contiguous())
                    weights.append((fn, base, scale))
                self._check_weights(
                    weights, ((1, 1), (2, 6), (2, 5), (12,)), norm_eps=norm_eps
                )

    @torch.inference_mode()
    def _check_weights(self, weights, layouts, norm_eps=1e-6):
        cpu_units, cpu_head = self._units(weights, "cpu", norm_eps)
        gpu_units, gpu_head = self._units(weights, "cuda", norm_eps)
        for layout in layouts:
            with self.subTest(layout=layout):
                x = torch.randn(*layout, 4, 5120, dtype=torch.bfloat16)
                static_input = x.cuda()
                expected = self._run(x, cpu_units, cpu_head)
                for _ in range(3):
                    eager = self._run(static_input, gpu_units, gpu_head)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    eager.cpu(), expected, rtol=0.02, atol=0.03125
                )

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self._run(static_input, gpu_units, gpu_head)
                for scale in (0.5, -1.25):
                    changed = (x.float() * scale).bfloat16()
                    static_input.copy_(changed)
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = self._run(changed, cpu_units, cpu_head)
                    torch.testing.assert_close(
                        output.cpu(), expected, rtol=0.02, atol=0.03125
                    )


if __name__ == "__main__":
    unittest.main()
