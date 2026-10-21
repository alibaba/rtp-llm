import importlib
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCHead, DelayedHCUnit


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA graph execution")
class DelayedHCCudaGraphTest(unittest.TestCase):
    def _units(self, weights, device):
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
                    norm_eps=1e-6,
                    hc_eps=1e-6,
                )
            )
        for previous, unit in zip(units, units[1:]):
            unit.set_previous(previous)
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
        self._check_weights(weights, ((1, 1), (2, 6)))

    @torch.inference_mode()
    def test_tilelang_split_prewarm_and_changed_input_graph_replay(self):
        from rtp_llm.models_py.modules.dsv41.hc import v41_jit_warmup as warmup
        from rtp_llm.models_py.modules.dsv41.hc.delayed import _tile_ops

        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("requires Blackwell CUDA")
        ops = _tile_ops()
        norm = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.norm_fn"
        )
        torch.manual_seed(71)
        weights = [(torch.randn(24, 20480) * 0.003, torch.zeros(24), torch.ones(3))]
        units, _ = self._units(weights, "cuda")
        root = torch.nn.ModuleList(units)
        device = units[0].fn.device
        maximum = 4096
        reps = warmup._tilelang_norm_ms(maximum, 20480, device, use_small=True)
        compiled = {"mul": set(), "norm": set()}
        warming = True

        def factory(kind, original):
            def call(*args):
                if warming:
                    compiled[kind].add(args)
                else:
                    self.assertIn(args, compiled[kind], "unwarmed split specialization")
                return original(*args)

            return call

        def legacy_single_split(residual):
            # Reproduce the pre-R1 launch explicitly: the public n_splits hint
            # is ignored by the new planner. S=1 has the original byte layout.
            tokens = residual.numel() // 20480
            mul = torch.empty((1, tokens, 1, 24), dtype=torch.float32, device=device)
            sqr = torch.empty((1, tokens, 1), dtype=torch.float32, device=device)
            total_mul, total_sqr = torch.empty_like(mul[0]), torch.empty_like(sqr[0])
            output = torch.empty((1, tokens, 24), dtype=torch.float32, device=device)
            norm._mhc_pre_norm_fn_fwd_mul(24, 1, 20480)(
                residual.view(tokens, 20480),
                norm.round_to_tf32(units[0].fn),
                mul.view(1, tokens, 24),
                sqr.view(1, tokens),
            )
            norm._mhc_pre_norm_fn_fwd_norm(24, 1, 20480, 1e-6, 1)(
                mul, sqr, total_mul, total_sqr, output.view(tokens, 24)
            )
            return output

        with patch.object(
            norm,
            "_mhc_pre_norm_fn_fwd_mul",
            side_effect=factory("mul", norm._mhc_pre_norm_fn_fwd_mul),
        ), patch.object(
            norm,
            "_mhc_pre_norm_fn_fwd_norm",
            side_effect=factory("norm", norm._mhc_pre_norm_fn_fwd_norm),
        ), patch.object(
            warmup.common, "model_warm_up_enabled", return_value=True
        ):
            warmup._WARMED_KEYS.clear()
            self.addCleanup(warmup._WARMED_KEYS.clear)
            warmup.warmup_v41_hc_jit(root, max_m=maximum, device=device)
            warming = False
            # The old wrapper adds 0x1000; TF32 MMA discards the low 13 bits.
            reference_fn = ((weights[0][0].view(torch.int32) + 0x1000) & -8192).view(
                torch.float32
            )
            for tokens in sorted(set(reps) | {160, 192, maximum}):
                with self.subTest(tokens=tokens):
                    x = torch.randn(1, tokens, 4, 5120, dtype=torch.bfloat16)
                    static = x.to(device)

                    def run():
                        return ops.mhc_pre_norm_fn(
                            static, units[0].fn, None, 1e-6, n_splits=1
                        )

                    for _ in range(3):
                        run()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = run()
                    for scale in (0.5, -1.25):
                        with self.subTest(scale=scale):
                            changed = (x.float() * scale).bfloat16()
                            static.copy_(changed)
                            graph.replay()
                            torch.cuda.synchronize()
                            captured = output.clone()
                            eager = run()
                            torch.testing.assert_close(captured, eager, rtol=0, atol=0)
                            legacy = legacy_single_split(static)
                            splits = norm._tilelang_num_splits(static, 20480)
                            if splits == 1:
                                torch.testing.assert_close(
                                    captured, legacy, rtol=0, atol=0
                                )
                            flat64 = changed.flatten(-2).double()
                            oracle = (flat64 @ reference_fn.double().T) * torch.rsqrt(
                                flat64.square().mean(-1, keepdim=True) + 1e-6
                            )
                            # Long-K S1 itself fails the former scalar CPU check.
                            # Require split-K to preserve or improve BOTH error
                            # measures against the same independent FP64 oracle.
                            error = captured.cpu().double() - oracle
                            legacy_error = legacy.cpu().double() - oracle
                            self.assertTrue(torch.isfinite(error).all().item())
                            self.assertTrue(torch.isfinite(legacy_error).all().item())
                            self.assertLessEqual(
                                error.abs().max().item(),
                                legacy_error.abs().max().item(),
                            )
                            self.assertLessEqual(
                                error.square().mean().sqrt().item(),
                                legacy_error.square().mean().sqrt().item(),
                            )

    @torch.inference_mode()
    def test_split_decode_delayed_chain_graph_replay(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("requires Blackwell CUDA")
        torch.manual_seed(31)
        weights = [
            (
                torch.randn(24, 20480) * 0.003,
                torch.randn(24) * 0.1,
                torch.tensor([0.2, 0.4, 0.3]),
            )
            for _ in range(2)
        ]
        # Frozen decode: standalone split-K entry, then fused S1 predecessor
        # readout, post/residual mixing and final head, including changed inputs.
        with patch.dict(os.environ, {"DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single"}):
            self._check_weights(weights, ((32, 5), (32, 6)))

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_CHECKPOINT"),
        "set DSV41_TEST_CHECKPOINT for real weights",
    )
    @torch.inference_mode()
    def test_checkpoint_atomic_loader_eager_and_graph(self):
        from safetensors import safe_open

        from rtp_llm.device.device_impl import CudaImpl
        from rtp_llm.models.deepseek_v41 import (
            DeepSeekV41DSparkWeight,
            DeepSeekV41Weight,
        )
        from rtp_llm.utils.model_weight import W

        checkpoint = Path(os.environ["DSV41_TEST_CHECKPOINT"])
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
            use_swizzleA=False,
            exported_device=CudaImpl.__new__(CudaImpl),
        )
        for descriptor_class, layer_id in (
            (DeepSeekV41Weight, 0),
            (DeepSeekV41Weight, 39),
            (DeepSeekV41DSparkWeight, 0),
            (DeepSeekV41DSparkWeight, 2),
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
                self._check_weights(weights, ((1, 1), (2, 6), (2, 5)))

    @torch.inference_mode()
    def _check_weights(self, weights, layouts):
        cpu_units, cpu_head = self._units(weights, "cpu")
        gpu_units, gpu_head = self._units(weights, "cuda")
        for batch, q_len in layouts:
            with self.subTest(batch=batch, q_len=q_len):
                x = torch.randn(batch, q_len, 4, 5120, dtype=torch.bfloat16)
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
