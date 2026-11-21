import os
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.layers.linear import QKVParallelLinear, RowParallelLinear
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


class SmoothOmniTest(unittest.TestCase):
    def test_gqa_tp_checkpoint_layout(self):
        for method in ("smooth_quant", "omni_quant"):
            layer = QKVParallelLinear(
                hidden_size=16,
                num_heads=4,
                num_kv_heads=1,
                head_dim=4,
                tp_size=2,
                tp_rank=1,
                quant_config=QuantizationConfig(method),
            )
            matrices = [
                (torch.arange(n * 16).reshape(n, 16) % 113).to(torch.int8)
                for n in (16, 4, 4)
            ]
            tensors = {}
            for name, w in zip(("q_proj", "k_proj", "v_proj"), matrices):
                tensors[name + ".qweight"] = (
                    w if method == "omni_quant" else w.t()
                ).contiguous()
                tensors[name + ".scales"] = torch.arange(1, w.size(0) + 1).float()
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "weights.safetensors")
                save_file(tensors, path)
                with safe_open(path, framework="pt") as f:
                    for name in f.keys():
                        view = f.get_slice(name)
                        plan = layer.checkpoint_read_spec(name, tuple(view.get_shape()))
                        layer.load_weights(
                            {
                                name: (
                                    plan.materialize(view)
                                    if plan.is_local_shard
                                    else f.get_tensor(name)
                                )
                            }
                        )
            layer.validate_weights_loaded()
            torch.testing.assert_close(
                layer.weight, torch.cat([matrices[0][8:], *matrices[1:]])
            )
            if method == "omni_quant":
                torch.testing.assert_close(
                    layer.bias.float(), layer.weight_scale.flatten()
                )

    def _row(self, device):
        for method in ("smooth_quant", "omni_quant"):
            layer = RowParallelLinear(
                12,
                10,
                quant_config=QuantizationConfig(method),
                prefix="self_attn.o_proj",
                params_dtype=torch.float16,
            )
            w = (torch.arange(120).reshape(12, 10) % 23 - 11).to(torch.int8)
            s = torch.linspace(0.12, 0.28, 10)
            smooth = torch.linspace(0.5, 1.75, 12)
            tensors = {"qweight": w, "scales": s, "smoother": smooth}
            if method == "omni_quant":
                tensors.update(
                    shift=torch.linspace(-0.75, 0.25, 12), bias=torch.arange(10).half()
                )
            layer.load_weights(tensors)
            layer.validate_weights_loaded()
            layer.to(device)
            for m in (1, 17):
                x = torch.arange(m * 12).reshape(m, 12).half() / 13 - 1
                z = x / smooth.half()
                if method == "omni_quant":
                    z += tensors["shift"].half()
                ref = []
                for row in z:
                    peak = row.abs().max().float().clamp_min(1e-6)
                    q = (row.float() * (127 / peak)).round().int()
                    acc = q @ w.int()
                    ref.append((acc * (peak / 127) * s).half())
                result = torch.stack(ref)
                if method == "omni_quant":
                    result += tensors["bias"]
                torch.testing.assert_close(
                    layer(x.to(device)).cpu(), result, rtol=0, atol=0
                )

    def test_online_cpu(self):
        self._row("cpu")

    @unittest.skipUnless(
        os.getenv("TEST_SMOOTH_OMNI_CUDA") == "1", "requires allocated NVIDIA GPU"
    )
    def test_online_cuda(self):
        self._row("cuda")


if __name__ == "__main__":
    unittest.main()
