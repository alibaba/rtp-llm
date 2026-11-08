"""CPU switch, native TP placement and online-loader delegation contracts."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
from typing import List
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.kimi_k3.nvfp4_config import (
    parse_nvfp4_flag,
    selected_nvfp4_modules,
    reject_legacy_nvfp4,
    validate_nvfp4_backend,
)
from rtp_llm.models.kimi_k3.nvfp4_tensor import (
    OnlineNvfp4Projection,
    nvfp4_global_scale,
    swizzle_nvfp4_scales,
    unswizzle_nvfp4_scales,
    repack_nvfp4_projections,
)


def native_splits():
    path = Path(__file__).resolve().parents[1] / "models/kimi_k3/kimi_k3_weight.py"
    names = {
        "split_kda_input",
        "_merge_kda_qkvg_fa_beta",
        "_merge_mla_input_projections",
    }
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    namespace = dict(torch=torch, List=List)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Nvfp4SwitchTest(unittest.TestCase):
    def test_service_defaults_and_all_four_ablations(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(parse_nvfp4_flag("KDA_NVFP4"))
            self.assertFalse(parse_nvfp4_flag("DSA_NVFP4"))
        for kda in (False, True):
            for dsa in (False, True):
                modules = selected_nvfp4_modules(kda, dsa)
                self.assertEqual(sum(m.startswith("kda.") for m in modules), 2 * kda)
                self.assertEqual(sum(m.startswith("dsa.") for m in modules), 2 * dsa)
                self.assertFalse(any(m.endswith(".output") for m in modules))
                self.assertEqual(selected_nvfp4_modules(kda, dsa, is_draft=True), ())
                self.assertFalse(any(m.startswith("moe.") for m in modules))

    def test_smoke_environment_and_explicit_boolean_precedence(self):
        with patch.dict(os.environ, KDA_NVFP4="0"):
            self.assertFalse(parse_nvfp4_flag("KDA_NVFP4", default=True))
            self.assertTrue(parse_nvfp4_flag("KDA_NVFP4", True, default=True))
        with patch.dict(os.environ, KDA_NVFP4="1"):
            self.assertFalse(parse_nvfp4_flag("KDA_NVFP4", False))
        for invalid in ("true", "", "2"):
            with patch.dict(os.environ, KDA_NVFP4=invalid), self.assertRaises(
                ValueError
            ):
                parse_nvfp4_flag("KDA_NVFP4")
        with self.assertRaises(ValueError):
            parse_nvfp4_flag("KDA_NVFP4", 1)

    def test_legacy_and_backend_errors(self):
        with self.assertRaisesRegex(ValueError, "replaced"):
            reject_legacy_nvfp4({"k31_nvfp4_modules": []})
        with self.assertRaisesRegex(ValueError, "service environment"):
            reject_legacy_nvfp4({"kda_nvfp4": True})
        for env in ({"RTP_LLM_FP4_GEMM_BACKEND": "trtllm"},):
            with patch.dict(os.environ, env), self.assertRaises(ValueError):
                validate_nvfp4_backend(("kda.input",))

    def test_natural_slices_preserve_codes_and_reinterleave_scale_rows(self):
        class Projection:
            def __init__(
                self, weight, weight_scales, weight_scale_2, quant_config, bias=None
            ):
                self.weight, self.weight_scales = weight, weight_scales
                self.weight_scale_2, self.quant_config = weight_scale_2, quant_config
                self.bias, self.dynamic_input_scale = bias, True
                self.output_size, self.hidden_size = (
                    weight.shape[0],
                    weight.shape[1] * 2,
                )

        codes = torch.arange(416 * 32).reshape(416, 32).to(torch.uint8)
        local = torch.arange(416 * 4).reshape(416, 4).to(torch.uint8)
        original = Projection(
            codes, swizzle_nvfp4_scales(local), torch.tensor([0.001]), NS()
        )
        q, kv, gate = [
            repack_nvfp4_projections([(original, a, b)])
            for a, b in ((0, 64), (64, 160), (160, 416))
        ]
        packed = repack_nvfp4_projections([(q, 0, 64), (gate, 0, 256)])
        self.assertTrue(
            torch.equal(packed.weight, torch.cat((codes[:64], codes[160:])))
        )
        self.assertTrue(
            torch.equal(
                unswizzle_nvfp4_scales(packed.weight_scales, 320, 4),
                torch.cat((local[:64], local[160:])),
            )
        )
        self.assertTrue(
            torch.equal(unswizzle_nvfp4_scales(kv.weight_scales, 96, 4), local[64:160])
        )
        self.assertIs(packed.weight_scale_2, original.weight_scale_2)
        gate.weight_scale_2 = original.weight_scale_2.clone()
        with self.assertRaisesRegex(ValueError, "shared global scale"):
            repack_nvfp4_projections([(q, 0, 64), (gate, 0, 256)])

    def test_native_kda_split_is_used_before_packing(self):
        native = native_splits()
        heads, dim, hidden, fa = 32, 32, 64, 64
        parts = [
            torch.arange((heads * dim) * hidden).reshape(heads * dim, hidden)
            + i * 1000000
            for i in range(4)
        ]
        parts += [torch.full((fa, hidden), 7), torch.full((heads, hidden), 9)]
        fused = native["_merge_kda_qkvg_fa_beta"](parts)
        for tp in (1, 2, 4, 8, 16):
            for rank in range(tp):
                source = NS(
                    name="w", get_tensor_names=lambda *args: {"original.weight"}
                )
                source._split = lambda tensor, cfg: {
                    "w": native["split_kda_input"](
                        tensor["w"], heads, dim, cfg.tp_size, cfg.tp_rank
                    )
                }
                projection = OnlineNvfp4Projection(source, module="kda.input")
                result = projection._split({"w": fused}, NS(tp_size=tp, tp_rank=rank))[
                    "w"
                ]
                self.assertEqual(
                    result.shape, (hidden, 4 * heads * dim // tp + fa + heads // tp)
                )
                self.assertTrue(
                    torch.equal(result[:, 4 * heads * dim // tp :][:, :fa], parts[4].T)
                )
                self.assertEqual(
                    projection.get_tensor_names(0, None), {"original.weight"}
                )

    def test_native_dsa_gate_shard_and_query_cp_replication(self):
        merge = native_splits()["_merge_mla_input_projections"]
        q, kv, gate = torch.randn(64, 64), torch.randn(96, 64), torch.randn(512, 64)
        for tp in (1, 2, 4, 8, 16):
            for rank in range(tp):
                result = merge([q, kv, gate], tp_size=tp, tp_rank=rank)
                self.assertTrue(torch.equal(result[:, :160], torch.cat((q, kv)).T))
                self.assertTrue(torch.equal(result[:, 160:], gate.chunk(tp)[rank].T))
        self.assertTrue(
            torch.equal(
                merge([q, kv, gate], tp_size=1, tp_rank=0), torch.cat((q, kv, gate)).T
            )
        )

    def test_global_scale_zero_and_dtype(self):
        for x in (
            torch.zeros(4, 32, dtype=torch.bfloat16),
            torch.ones(4, 32, dtype=torch.bfloat16),
        ):
            scale = nvfp4_global_scale(x)
            self.assertEqual(scale.dtype, torch.float32)
            self.assertEqual(scale.shape, (1,))
            self.assertTrue(torch.isfinite(scale).all() and (scale > 0).all())
        self.assertEqual(
            nvfp4_global_scale(torch.ones(4, 32)).item(), torch.tensor(1 / 2688).item()
        )

    def test_loader_rejects_lora_before_reading(self):
        projection = OnlineNvfp4Projection(NS(name="w"), module="kda.input")
        with self.assertRaisesRegex(ValueError, "LoRA"):
            projection._load_raw_tensor(None, 0, "cpu", NS(merge_lora=True))


if __name__ == "__main__":
    unittest.main()
