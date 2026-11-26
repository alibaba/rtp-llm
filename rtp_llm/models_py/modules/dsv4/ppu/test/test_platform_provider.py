"""Contracts for the actual instance-bound PPU construction providers."""

import ast
import functools
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).parents[1]
PREFIX = "rtp_llm.models_py.modules.dsv4.ppu"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlatformProviderTest(unittest.TestCase):
    def setUp(self):
        # Keep hardware/math leaves out of the construction-contract test.
        self.stub = patch.dict(sys.modules, {
            "torch": SimpleNamespace(bfloat16=object(), float32=object()),
            "rtp_llm.models_py.kernels.ppu.runtime": SimpleNamespace(PpuStreamPool=Mock()),
            PREFIX + ".attention.attention_contract": SimpleNamespace(
                scheduler_key=Mock(), validate_dual_pool=Mock(), validate_swa_pool=Mock()),
        })
        self.stub.start()
        self.addCleanup(self.stub.stop)
        self.module = load("ppu_contract_provider", ROOT / "ppu_module_provider.py")
        self.manifest = load("ppu_contract_manifest", ROOT / "manifest.py")
        self.mods = patch.dict(sys.modules, {
            PREFIX + ".ppu_module_provider": self.module,
            PREFIX + ".manifest": self.manifest,
        })
        self.mods.start()
        self.addCleanup(self.mods.stop)
        self.decode = load("ppu_contract_decode", ROOT / "ppu_decode_provider.py")

    def test_options_are_owned_and_boolean_parsing_is_strict(self):
        options = {"flag": "yes"}
        provider = self.module.PpuModuleProvider(options)
        options["flag"] = "no"
        self.assertTrue(provider._bool("flag"))
        self.assertFalse(provider._bool("missing"))
        with self.assertRaises(TypeError):
            provider.execution_options["flag"] = "off"
        for value in ("1", "true", "on", "yes", " TRUE "):
            self.assertTrue(self.module.PpuModuleProvider({"flag": value})._bool("flag"))
        for value in ("0", "false", "off", "no"):
            self.assertFalse(self.module.PpuModuleProvider({"flag": value})._bool("flag"))
        with self.assertRaises(ValueError):
            self.module.PpuModuleProvider({"flag": "maybe"})._bool("flag")

    def test_attention_binds_identity_and_rejects_another_instance(self):
        provider = self.module.PpuModuleProvider({})
        ctor = Mock()
        with patch.dict(sys.modules, {PREFIX + ".attention.ppu_fp4_indexer": SimpleNamespace(PpuFP4Attention=ctor)}):
            self.assertIs(provider.build_attention(None, 7), ctor.return_value)
            ctor.assert_called_once_with(7, platform_provider=provider)
            with self.assertRaises(TypeError):
                provider.build_attention(None, platform_provider=object())

    def test_prefill_and_decode_keep_metadata_and_optional_fusion_contracts(self):
        prefill = self.module.PpuModuleProvider({})
        decode = self.decode.PpuDecodeProvider(self.manifest.DECODE_EXECUTION_OPTIONS)
        factory = Mock()
        for provider in (prefill, decode):
            self.assertIs(provider.build_decode_metadata(factory, 3, value=4), factory.return_value)
            factory.assert_called_with(3, value=4)
        self.assertIsNone(prefill.build_mtp_input_fusion(None, None, None, None))
        decode._decode_fusions = False
        self.assertIsNone(decode.build_mtp_input_fusion(None, None, None, None))
        ctor = Mock()
        decode._decode_fusions = True
        with patch.dict(sys.modules, {PREFIX + ".mtp.ppu_mtp_input": SimpleNamespace(PpuMtpInputFusion=ctor)}):
            self.assertIs(decode.build_mtp_input_fusion(1, 2, 3, 4), ctor.return_value)
            ctor.assert_called_once_with(1, 2, 3, 4)

    def test_block_moe_kwargs_match_ppu_constructors_without_kwargs_sink(self):
        def _class_init_params(
            path: Path, cls_name: str
        ) -> tuple[list[str], list[str], bool]:
            tree = ast.parse(path.read_text())
            cls = next(
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == cls_name
            )
            init = next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            )
            pos = init.args.posonlyargs + init.args.args
            n_req = len(pos) - len(init.args.defaults)
            req = [a.arg for a in pos[:n_req] if a.arg != "self"]
            opt = [a.arg for a in pos[n_req:] if a.arg != "self"]
            for a, d in zip(init.args.kwonlyargs, init.args.kw_defaults):
                (req if d is None else opt).append(a.arg)
            return req, opt, init.args.kwarg is not None

        block_tree = ast.parse((ROOT / "block.py").read_text())
        block_cls = next(
            n
            for n in block_tree.body
            if isinstance(n, ast.ClassDef) and n.name == "Block"
        )
        block_init = next(
            n
            for n in block_cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        moe_kwargs_keys = None
        build_moe_call = None
        for node in ast.walk(block_init):
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "moe_kwargs"
                    for t in node.targets
                )
                and isinstance(node.value, ast.Call)
            ):
                moe_kwargs_keys = {kw.arg for kw in node.value.keywords if kw.arg}
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "build_moe"
            ):
                build_moe_call = node
        self.assertIsNotNone(moe_kwargs_keys)
        self.assertNotIn("moe_workspace", moe_kwargs_keys)
        self.assertIsNotNone(build_moe_call)
        factory_arg = build_moe_call.args[0]
        self.assertIsInstance(factory_arg, ast.Call)
        self.assertIsInstance(factory_arg.args[0], ast.Name)
        self.assertEqual(factory_arg.args[0].id, "build_baseline_moe")
        self.assertEqual(
            [kw.arg for kw in factory_arg.keywords], ["moe_workspace"]
        )

        tp_req, tp_opt, tp_has_varkw = _class_init_params(
            ROOT / "moe/ppu_tp_moe.py", "PpuTPMoE"
        )
        ep_req, ep_opt, ep_has_varkw = _class_init_params(
            ROOT / "moe/ppu_ep_moe.py", "PpuEPMoE"
        )
        self.assertFalse(tp_has_varkw)
        self.assertFalse(ep_has_varkw)

        tp_params = set(tp_req) | set(tp_opt)
        ep_params = set(ep_req) | set(ep_opt)
        forwarded = set(moe_kwargs_keys) | {"tp_rank", "platform_provider"}
        self.assertTrue(forwarded <= tp_params, f"extra for PpuTPMoE: {forwarded - tp_params}")
        self.assertTrue(
            (forwarded | {"strategy_type", "strategy_kwargs"}) <= ep_params,
            f"extra for PpuEPMoE: {forwarded - ep_params}",
        )

        # Exercise provider.build_moe against strict-signature stubs (no **kwargs).
        def make_strict_stub(req_names: list[str], opt_names: list[str]):
            recorded = {}
            sig_parts = sorted(req_names) + [f"{n}=None" for n in sorted(opt_names)]
            ns: dict[str, object] = {"recorded": recorded}
            exec(
                f"def ctor({', '.join(sig_parts)}):\n    recorded['kwargs'] = locals().copy()\n    return 'ok'",
                ns,
            )
            return ns["ctor"], recorded

        tp_stub, tp_rec = make_strict_stub(tp_req, tp_opt)
        ep_stub, ep_rec = make_strict_stub(ep_req, ep_opt)
        base_kwargs = {k: f"val_{k}" for k in moe_kwargs_keys}
        base_kwargs.update({"tp_rank": 0, "tp_size": 1, "ep_size": 1, "is_decode_role": False})

        prefill = self.module.PpuModuleProvider({})
        decode = self.decode.PpuDecodeProvider(self.manifest.DECODE_EXECUTION_OPTIONS)
        default_factory = Mock(return_value="baseline")
        with patch.dict(
            sys.modules,
            {
                PREFIX + ".moe.ppu_tp_moe": SimpleNamespace(PpuTPMoE=tp_stub),
                PREFIX + ".moe.ppu_ep_moe": SimpleNamespace(PpuEPMoE=ep_stub),
                PREFIX + ".moe.ppu_legacy_deepep": SimpleNamespace(
                    PpuLegacyDeepEPStrategy=object
                ),
                PREFIX + ".moe.ppu_deepep_fp4": SimpleNamespace(
                    PpuDeepEPFP4Strategy=object
                ),
            },
        ):
            self.assertEqual(prefill.build_moe(default_factory, **base_kwargs), "ok")
            self.assertNotIn("moe_workspace", tp_rec["kwargs"])
            ep_kwargs = dict(base_kwargs, ep_size=8)
            self.assertEqual(prefill.build_moe(default_factory, **ep_kwargs), "ok")
            decode_kwargs = dict(ep_kwargs, is_decode_role=True)
            self.assertEqual(decode.build_moe(default_factory, **decode_kwargs), "ok")
            self.assertNotIn("moe_workspace", ep_rec["kwargs"])
            default_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
