"""Prefill DeepEP registration and topology predicates (source-extracted)."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

_SOURCE_PATH = Path(__file__).resolve().parents[1] / "manifest.py"


def _load_predicates():
    tree = ast.parse(_SOURCE_PATH.read_text())
    wanted = {
        "PREFILL_EXECUTION_OPTIONS",
        "DECODE_EXECUTION_OPTIONS",
        "_check_execution_options",
        "_supports_flash_model",
        "_supports_ppu_prefill",
        "supports_ppu_fp4_prefill",
        "_supports_ppu_tp_moe",
        "supports_ppu_fp4_tp_moe",
        "supports_ppu_fp4_prefill_deepep",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.Assign))
        and (
            getattr(node, "name", None) in wanted
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id in wanted for t in node.targets
                )
            )
        )
    ]
    namespace = {}

    class SupportResult:
        def __init__(self, supported, reason=""):
            self.supported = supported
            self.reason = reason

    namespace["SupportResult"] = SupportResult
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(_SOURCE_PATH), "exec"),
        namespace,
    )
    return namespace, tree


def _prefill_metadata(**changed):
    metadata = dict(
        model_type="deepseek_v4",
        num_layers=43,
        hidden_size=4096,
        tp_size=8,
        ep_size=8,
        dp_size=1,
        pp_size=1,
        world_size=8,
        cp_enabled=False,
        role="PREFILL",
        speculative=False,
        cuda_graph=False,
        reuse_cache=False,
        lora=False,
        eplb=False,
        indexer_cache_mode="fp4",
        fp8_kv_cache=True,
        execution_options=None,
    )
    metadata.update(changed)
    return metadata


def _selection(metadata):
    return SimpleNamespace(model_metadata=metadata)


class PrefillDeepEPManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace, cls.tree = _load_predicates()
        cls.options = dict(cls.namespace["PREFILL_EXECUTION_OPTIONS"])

    def supports_deepep(self, **changed):
        metadata = _prefill_metadata(execution_options=dict(self.options))
        metadata.update(changed)
        return self.namespace["supports_ppu_fp4_prefill_deepep"](
            _selection(metadata), None
        )

    def supports_tp_moe(self, **changed):
        metadata = _prefill_metadata(
            execution_options=dict(self.options), ep_size=1, tp_size=4, world_size=4
        )
        metadata.update(changed)
        return self.namespace["supports_ppu_fp4_tp_moe"](_selection(metadata), None)

    def test_deepep_accepts_ep_world_prefill_with_normal_communication(self):
        for tp, ep, world in ((8, 8, 8), (4, 4, 4), (2, 2, 2)):
            with self.subTest(tp=tp, ep=ep):
                result = self.supports_deepep(
                    tp_size=tp,
                    ep_size=ep,
                    world_size=world,
                    moe_communication={
                        "enabled": True,
                        "low_latency": False,
                        "all_gather": False,
                        "ffn_disaggregate": False,
                        "internode": False,
                    },
                )
                self.assertTrue(result.supported, result.reason)

    def test_deepep_rejects_ep1_missing_or_low_latency_communication(self):
        cases = (
            ("ep1", dict(ep_size=1, moe_communication=self._comm())),
            ("disabled", dict(moe_communication=self._comm(enabled=False))),
            ("low_latency", dict(moe_communication=self._comm(low_latency=True))),
            ("all_gather", dict(moe_communication=self._comm(all_gather=True))),
            (
                "ffn_disaggregate",
                dict(moe_communication=self._comm(ffn_disaggregate=True)),
            ),
            ("internode", dict(moe_communication=self._comm(internode=True))),
            ("missing", dict()),
        )
        for name, changed in cases:
            with self.subTest(case=name):
                result = self.supports_deepep(**changed)
                self.assertFalse(result.supported)

    @staticmethod
    def _comm(**changed):
        values = {
            "enabled": True,
            "low_latency": False,
            "all_gather": False,
            "ffn_disaggregate": False,
            "internode": False,
        }
        values.update(changed)
        return values

    def test_deepep_keeps_the_launch_template_requirement(self):
        result = self.supports_deepep(
            execution_options={k: v for k, v in self.options.items() if k != "DSV4_HC_IMPL"},
            moe_communication=self._comm(),
        )
        self.assertFalse(result.supported)
        self.assertIn("DSV4_HC_IMPL", result.reason)

    def test_tp_moe_still_accepts_ep1_and_now_rejects_shared_experts(self):
        self.assertTrue(self.supports_tp_moe(ep_size=1).supported)
        result = self.supports_tp_moe(ep_size=8, tp_size=8, world_size=8)
        self.assertFalse(result.supported)
        self.assertIn("EP1", result.reason)

    def test_registration_declares_the_new_collectives(self):
        source = _SOURCE_PATH.read_text()
        self.assertIn('impl_id="ppu.dsv4.moe.fp4_prefill_deepep.v1"', source)
        self.assertIn(
            'collective_protocol_id="ppu.dsv4.moe.dpep-normal-mxfp4.v1"', source
        )
        self.assertIn(
            'collective_protocol_id="ppu.dsv4.decode.dpep-mxfp4-ll.v2"', source
        )
        self.assertNotIn("tp1-dpep", source)
        self.assertIn(
            '"rtp_llm.platforms.ppu.models.dsv4.manifest:"\n'
            '            "supports_ppu_fp4_prefill_deepep"',
            source,
        )
        self.assertIn(
            '"build_prefill_deepep_moe"',
            source,
        )

    def test_new_impl_is_explicitly_selected_like_its_siblings(self):
        tree = ast.parse(_SOURCE_PATH.read_text())
        registrations = [
            entry
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register_implementation"
            for entry in node.args[:1]
        ]
        by_impl = {}
        for spec in registrations:
            if not isinstance(spec, ast.Call) or getattr(spec.func, "id", "") != (
                "ModuleImplSpec"
            ):
                continue
            fields = {
                kw.arg: kw.value for kw in spec.keywords if kw.arg is not None
            }
            impl_value = fields.get("impl_id")
            if not isinstance(impl_value, ast.Constant):
                continue
            by_impl[impl_value.value] = fields
        self.assertIn("ppu.dsv4.moe.fp4_prefill_deepep.v1", by_impl)
        fields = by_impl["ppu.dsv4.moe.fp4_prefill_deepep.v1"]
        self.assertEqual(ast.literal_eval(fields["module_id"]), "rtp.dsv4.moe")
        self.assertEqual(ast.literal_eval(fields["auto_selectable"]), False)
        self.assertEqual(ast.unparse(fields["capabilities"]), "{'prefill'}")
        self.assertEqual(
            ast.literal_eval(fields["collective_protocol_id"]),
            "ppu.dsv4.moe.dpep-normal-mxfp4.v1",
        )
        self.assertEqual(ast.literal_eval(fields["priority"]), 110)
        self.assertIn("DeviceType.Ppu", ast.unparse(fields["supported_devices"]))
        self.assertIn("CONTRACTS[", ast.unparse(fields["contract_id"]))


if __name__ == "__main__":
    unittest.main()
