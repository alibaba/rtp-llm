"""CPU execution of production configuration methods without native bindings."""

import ast
from contextlib import nullcontext
from dataclasses import dataclass, replace
import json
import logging
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from rtp_llm.models.kimi_k3.nvfp4_config import (
    parse_nvfp4_flag,
    selected_nvfp4_modules,
    reject_legacy_nvfp4,
    uses_fp8_output,
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
    def test_kda_nvfp4_auto_dispatches_at_tp8_boundary(self):
        path = Path(__file__).resolve().parents[1] / "models_py/modules/hybrid/kda_attention.py"
        nodes = ast.parse(path.read_text()).body
        limit = next(
            n for n in nodes if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_AG_LOCAL_NVFP4_OVERLAP_MIN_ROWS_TP8" for t in n.targets)
        )
        cls = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == "KdaAttention")
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        eligible = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "eligible_nvfp4_ag"
        )
        scope = dict(
            torch=NS(bfloat16="bf16", cuda=NS(is_current_stream_capturing=lambda: False)),
            Fp8Activation=type("Fp8Activation", (), {}),
            Group=NS(TP="tp"),
            _get_group=lambda group: group,
            profile_scope=lambda _: nullcontext(),
        )
        exec(compile(ast.Module(body=[limit, eligible, forward], type_ignores=[]), str(path), "exec"), scope)

        class Routed(Exception):
            pass

        module = ModuleType("rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion")

        def gather(hidden, projection, group, *, overlap, direct):
            raise Routed(overlap, direct)

        module.all_gather_nvfp4_gemm = gather
        module.PackedNvfp4Activation = type("PackedNvfp4Activation", (), {})
        direct = object()
        with patch.dict(sys.modules, {module.__name__: module}):
            for mode, tp, rows, expected in (
                ("auto", 8, 8191, (False, None)),
                ("auto", 8, 8192, (True, None)),
                ("auto", 4, 8192, (False, None)),
                ("serial", 8, 8192, (False, None)),
                ("overlap", 8, 128, (True, None)),
                ("direct", 8, 128, (False, direct)),
            ):
                with self.subTest(mode=mode, tp=tp, rows=rows):
                    attention = NS(_fp8_input_collective=None, _ag_local_fp8_overlap=False,
                                   _nvfp4_ag_mode=mode, _nvfp4_direct_ag=direct, tp_size=tp, input=object())
                    attention.eligible_nvfp4_ag = scope["eligible_nvfp4_ag"].__get__(attention)
                    packet = module.PackedNvfp4Activation()
                    packet.shape = (rows, 7168)
                    for hidden in (NS(shape=packet.shape, dtype="bf16", is_cuda=True), packet):
                        with self.assertRaises(Routed) as routed:
                            scope["forward"](attention, hidden, None, None,
                                             NS(is_prefill=True), NS(is_target_verify=False))
                        self.assertEqual(routed.exception.args, expected)

    def test_output_creates_fp8_linear_under_the_nvfp4_switch(self):
        path = Path(__file__).resolve().parents[1] / "models_py/modules/kimi_k3/projection.py"
        function = next(
            node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "create_kimi_projection"
        )
        scope = dict(
            torch=NS(float8_e4m3fn="fp8", uint8="fp4"),
            Fp8BlockWiseQuantConfig=lambda: "grouped_fp8",
            LinearFactory=NS(create_linear=lambda weight, **kwargs: NS(weight=weight, **kwargs)),
            uses_fp8_output=uses_fp8_output,
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
        module = ModuleType("rtp_llm.models.kimi_k3.nvfp4_weight")
        module.ATTENTION_PROJECTION_MODULES = {"kda_o": "kda.output", "dsa_o": "dsa.output"}
        module.nvfp4_scale_keys = None
        config = NS(k31_nvfp4_modules=selected_nvfp4_modules(True, True))
        create = scope["create_kimi_projection"]
        with patch.dict(sys.modules, {module.__name__: module}):
            for name in module.ATTENTION_PROJECTION_MODULES:
                weights = {name: NS(dtype="fp8"), "scale": object()}
                projection = create(config, weights, name, "scale")
                self.assertEqual(projection.quant_config, "grouped_fp8")
                self.assertIs(projection.weight_scales, weights["scale"])
                for dtype in ("bf16", "fp4"):
                    weights[name].dtype = dtype
                    with self.assertRaisesRegex(ValueError, "requires grouped FP8"):
                        create(config, weights, name, "scale")

    def test_weight_loader_keeps_output_fp8_for_each_enabled_group(self):
        class BaseWeight:
            _MLA_SUFFIXES = ()

            def _attention_fp8_quant_config(self, source):
                return self.model_config.attention_projection_quant_config

        roles = NS(PREFILL="PREFILL", DECODE="DECODE", PDFUSION="PDFUSION")
        root = Path(__file__).resolve().parents[1] / "models/kimi_k3"
        path = root / "kimi_k31_weight.py"
        cls = next(
            node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "KimiK31Weight"
        )
        native_fp8 = object()
        scope = dict(
            KimiK3Weight=BaseWeight, RoleType=roles,
            Fp8BlockWiseQuantConfig=lambda: native_fp8,
            uses_fp8_output=uses_fp8_output,
        )
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
        module = ModuleType("rtp_llm.models.kimi_k3.nvfp4_weight")
        names = ("kda.input", "kda.forget", "kda.output", "dsa.input", "dsa.query", "dsa.output")
        module.ATTENTION_PROJECTION_MODULES = {name: name for name in names}
        module.KimiK31Nvfp4Weight = lambda source, module: NS(module=module)

        with patch.dict(sys.modules, {module.__name__: module}):
            for role in vars(roles).values():
                for kda, dsa in ((False, False), (True, False), (False, True), (True, True)):
                    with self.subTest(role=role, kda=kda, dsa=dsa):
                        weight = scope["KimiK31Weight"].__new__(scope["KimiK31Weight"])
                        weight.role_type = role
                        selected = selected_nvfp4_modules(kda, dsa)
                        weight.model_config = NS(
                            k31_nvfp4_modules=selected,
                            attention_projection_quant_config=None,
                        )
                        for name in names:
                            source = NS(name=name)
                            fp8_output = (name == "kda.output" and kda) or (name == "dsa.output" and dsa)
                            fp4 = (name in ("kda.input", "kda.forget") and kda) or (
                                name in ("dsa.input", "dsa.query") and dsa
                            )
                            self.assertEqual(weight._nvfp4_projection(source) is not source, fp4)
                            self.assertIs(
                                weight._attention_fp8_quant_config(source),
                                native_fp8 if fp8_output else None,
                            )

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
