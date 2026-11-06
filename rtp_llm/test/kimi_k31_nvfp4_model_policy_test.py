"""CPU execution of production configuration methods without native bindings."""

import ast
from dataclasses import dataclass, replace
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from rtp_llm.models.kimi_k3.nvfp4_config import (
    parse_nvfp4_flag,
    selected_nvfp4_modules,
    reject_legacy_nvfp4,
    validate_nvfp4_backend,
)


@dataclass(frozen=True)
class Dsa:
    indexer_use_fp8: bool


@dataclass(frozen=True)
class Runtime:
    dsa: Dsa


class Base:
    _python_fields = set()

    def __init__(self):
        self.is_mtp = False
        self.model_type = "kimi_k31"
        self.config_dtype, self.quantization = "bf16", "FP8_PER_BLOCK"
        self.attn_config = NS(mla_fp8_compute=False)
        self.k3_runtime_config = Runtime(Dsa(True))
        self.hybrid_attention_config = NS(hybrid_attention_types=["kda", "dsa"])
        self.moe_layer_index, self.kv_cache_spec_descs = [0, 1], []

    def apply_override_args(self, args):
        pass


class KimiBase:
    @classmethod
    def _post_build_model_config(cls, config):
        pass

    @classmethod
    def _parse_kimi_runtime_config(cls, text, config):
        config.k3_runtime_config = NS()


def production_configs():
    bf16 = NS(to_str=lambda: "bf16")
    namespace = dict(
        ModelConfig=Base,
        KimiK3=KimiBase,
        KimiK31Weight=object,
        KimiK3RuntimeConfig=Runtime,
        KimiK31RuntimeConfig=Runtime,
        replace=replace,
        json=json,
        logging=logging,
        os=os,
        WEIGHT_TYPE=NS(BF16=bf16, from_str=lambda value: bf16),
        QuantAlgo=object,
        Fp8BlockWiseQuantConfig=lambda: NS(get_method=lambda: "FP8_PER_BLOCK"),
        KvCacheDataType=NS(BASE="base", FP8="fp8"),
        HybridAttentionType=NS(LINEAR="kda"),
        DsaConfig=NS(from_kimi_config=lambda *args, **kw: Dsa(True)),
        parse_nvfp4_flag=parse_nvfp4_flag,
        selected_nvfp4_modules=selected_nvfp4_modules,
        reject_legacy_nvfp4=reject_legacy_nvfp4,
        validate_nvfp4_backend=validate_nvfp4_backend,
    )
    root = Path(__file__).resolve().parents[1] / "models/kimi_k3"
    for file, names in (
        ("kimi_k3.py", {"KimiK3ModelConfig"}),
        ("kimi_k31.py", {"KimiK31ModelConfig", "KimiK31MtpModelConfig", "KimiK31"}),
    ):
        nodes = [
            n
            for n in ast.parse((root / file).read_text()).body
            if isinstance(n, ast.ClassDef) and n.name in names
        ]
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(root / file), "exec"),
            namespace,
        )
    return namespace


class K31Nvfp4ModelPolicyTest(unittest.TestCase):
    def test_target_flags_keep_indexer_cache_and_mtp_independent(self):
        for kda in ("0", "1"):
            for dsa in ("0", "1"):
                with patch.dict(
                    os.environ,
                    KDA_NVFP4=kda,
                    DSA_NVFP4=dsa,
                    RTP_LLM_FP4_GEMM_BACKEND="cutlass",
                    KIMI_K31_OUTER_NATURAL="0",
                ):
                    classes = production_configs()
                    target = classes["KimiK31ModelConfig"]()
                    classes["KimiK31"]._parse_kimi_runtime_config({}, target)
                    target.init_precision_config(NS(fp8_kv_cache=True), "bf16")
                    classes["KimiK31"]._post_build_model_config(target)
                    self.assertEqual(
                        target.k31_nvfp4_modules,
                        selected_nvfp4_modules(kda == "1", dsa == "1"),
                    )
                    self.assertEqual(
                        target.attention_projection_quant_config.get_method(),
                        "FP8_PER_BLOCK",
                    )
                    self.assertEqual(
                        target.indexer_projection_quant_config.get_method(),
                        "FP8_PER_BLOCK",
                    )
                    self.assertEqual(target.attn_config.kv_cache_dtype, "fp8")
                    self.assertFalse(target.attn_config.mla_fp8_compute)
                    draft = classes["KimiK31MtpModelConfig"]()
                    draft.is_mtp, draft.model_type, draft.quantization = (
                        True,
                        "kimi_k31_mtp",
                        "",
                    )
                    draft.init_precision_config(NS(fp8_kv_cache=True), "bf16")
                    self.assertIsNone(draft.attention_projection_quant_config)
                    self.assertEqual(draft.attn_config.kv_cache_dtype, "base")
                    self.assertFalse(hasattr(draft, "k31_nvfp4_modules"))

    def test_legacy_override_fails_for_target_and_draft(self):
        classes = production_configs()
        for name in ("KimiK31ModelConfig", "KimiK31MtpModelConfig"):
            with self.assertRaisesRegex(ValueError, "replaced"):
                classes[name]().apply_override_args('{"k31_nvfp4_modules":[]}')

    def test_packed_export_rejected_before_runtime_quantization(self):
        classes = production_configs()
        with self.assertRaisesRegex(ValueError, "original native checkpoint"):
            classes["KimiK31"]._parse_kimi_runtime_config(
                {"rtp_llm_nvfp4": {"format": "modelopt_nvfp4"}},
                classes["KimiK31ModelConfig"](),
            )

    def test_missing_selected_layer_fails(self):
        classes = production_configs()
        target = classes["KimiK31ModelConfig"]()
        target.k31_nvfp4_modules = ("kda.input",)
        target.hybrid_attention_config.hybrid_attention_types = ["dsa"]
        with self.assertRaisesRegex(ValueError, "no layers"):
            classes["KimiK31"]._post_build_model_config(target)


if __name__ == "__main__":
    unittest.main()
