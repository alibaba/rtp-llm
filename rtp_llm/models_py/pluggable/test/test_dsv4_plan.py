"""Framework-level DSV4 plan tests over the public baseline implementations."""

import json
import subprocess
import sys
import unittest
from types import SimpleNamespace

from rtp_llm.config.module_dispatch_config import ModuleDispatchConfig
from rtp_llm.device.device_type import DeviceType
from rtp_llm.device.runtime import DeviceRuntimeContext
from rtp_llm.models.dsv4.adapter import execution_options_snapshot
from rtp_llm.models.dsv4.adapter import get_registry as get_module_registry
from rtp_llm.models.dsv4.specs import CONTRACTS, STATE_FORMAT, request_for
from rtp_llm.models_py.pluggable.factory import (
    ModuleBuildContext,
    ModuleSelectionContext,
)


def selection(rank=0, *, device_name="L20", **changed):
    metadata = dict(
        model_type="deepseek_v4",
        num_layers=43,
        hidden_size=4096,
        tp_size=4,
        ep_size=1,
        dp_size=1,
        pp_size=1,
        world_size=4,
        cp_enabled=False,
        role="PDFUSION",
        speculative=False,
        cuda_graph=False,
        reuse_cache=False,
        lora=False,
        eplb=False,
        indexer_cache_mode="fp8",
        fp8_kv_cache=True,
    )
    metadata.update(changed)
    return ModuleSelectionContext(
        DeviceRuntimeContext(DeviceType.Cuda, device_name, rank), json.dumps(metadata)
    )


def explicit_config():
    return ModuleDispatchConfig.from_dict(
        {
            "mode": "auto",
            "platform": "cuda",
            "impl_overrides": {
                "rtp.dsv4." + kind: f"baseline.dsv4.{kind}.v1"
                for kind in CONTRACTS
            },
        }
    )


class Dsv4PlanTest(unittest.TestCase):
    def context(self, selected=None, config=None):
        return ModuleBuildContext(
            get_module_registry(),
            selected or selection(),
            config or explicit_config(),
            world_size=4,
        )

    def test_real_manifest_plans_all_43_layers_before_build(self):
        ctx = self.context()
        ctx.prepare([request_for("model", ctx.selection)])
        self.assertEqual(len(ctx.bindings), 130)
        paths = {b.request.path for b in ctx.bindings}
        self.assertIn("v4", paths)
        for index in range(43):
            for suffix in ("", ".attn", ".ffn"):
                self.assertIn(f"v4.layers.{index}{suffix}", paths)
        self.assertFalse(
            any(name.endswith("pluggable_builders") for name in sys.modules)
        )
        self.assertEqual(ctx.state, "planned")

    def test_model_rank_digest_is_identical(self):
        digests = set()
        for rank in range(4):
            ctx = self.context(selection(rank))
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
        self.assertEqual(len(digests), 1)

    def test_pd_prefill_plan_keeps_tp_and_forward_contracts(self):
        from rtp_llm.models.dsv4.specs import (
            forward_capabilities,
            validate_forward_phase,
            validate_runtime_role,
        )

        digests = set()
        for rank in range(4):
            ctx = self.context(selection(rank, role="PREFILL"))
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
            self.assertEqual(len(ctx.bindings), 130)
            validate_runtime_role(
                ctx.selection.model_metadata,
                is_decode_role=False,
                is_speculative=False,
            )
            capabilities = forward_capabilities(ctx.bindings)
            validate_forward_phase(
                capabilities,
                is_prefill=True,
                has_decode_fmha=False,
                is_target_verify=False,
            )
            with self.assertRaisesRegex(RuntimeError, "does not support decode"):
                validate_forward_phase(
                    capabilities,
                    is_prefill=False,
                    has_decode_fmha=True,
                    is_target_verify=False,
                )
        self.assertEqual(len(digests), 1)

    def test_parallel_groups_are_not_restricted_to_fixed_sizes(self):
        from rtp_llm.models.dsv4.adapter import validate_parallelism

        for tp, dp, ep in ((2, 1, 1), (8, 1, 1), (4, 1, 4), (8, 1, 8)):
            world = tp * dp
            topology = dict(
                tp_size=tp, dp_size=dp, ep_size=ep, pp_size=1, world_size=world
            )
            with self.subTest(**topology):
                validate_parallelism(SimpleNamespace(**topology))
                digests = set()
                for rank in range(world):
                    ctx = ModuleBuildContext(
                        get_module_registry(),
                        selection(rank, role="PREFILL", **topology),
                        explicit_config(),
                        world_size=world,
                    )
                    digests.add(ctx.prepare([request_for("model", ctx.selection)]))
                    self.assertEqual(len(ctx.bindings), 130)
                self.assertEqual(len(digests), 1)

    def test_parallel_groups_still_require_consistent_engine_geometry(self):
        from rtp_llm.models.dsv4.adapter import validate_parallelism

        for changed in (
            {"tp_size": 0},
            {"dp_size": 0},
            {"ep_size": 0},
            {"pp_size": 2},
            {"world_size": 8},
            {"ep_size": 3},
            {"ep_size": 1},
        ):
            values = dict(tp_size=2, dp_size=2, ep_size=4, pp_size=1, world_size=4)
            values.update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_parallelism(SimpleNamespace(**values))

    def test_declared_protocols_match_the_registered_contracts(self):
        registry = get_module_registry()
        for kind in CONTRACTS:
            impl = registry.implementation(
                "rtp.dsv4." + kind, f"baseline.dsv4.{kind}.v1"
            )
            self.assertEqual(impl.module_id, "rtp.dsv4." + kind)
            self.assertEqual(impl.collective_protocol_id, "rtp.dsv4.tp-reduced.v1")
            self.assertFalse(impl.auto_selectable)

    def test_explicit_state_contracts_are_consistent(self):
        config = json.loads(explicit_config().to_string())
        config["impl_overrides"] = {
            "rtp.dsv4." + kind: f"baseline.dsv4.{kind}.v1" for kind in CONTRACTS
        }
        digests = set()
        for rank in range(4):
            ctx = self.context(selection(rank), ModuleDispatchConfig.from_dict(config))
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
            self.assertEqual(len(ctx.bindings), 130)
            self.assertEqual(
                {b.implementation.state_format_id for b in ctx.bindings},
                {STATE_FORMAT},
            )
        self.assertEqual(len(digests), 1)
        config["impl_overrides"]["rtp.dsv4.attention"] = (
            "baseline.dsv4.attention.retired.v1"
        )
        ctx = self.context(selection(), ModuleDispatchConfig.from_dict(config))
        with self.assertRaisesRegex(ValueError, "Unknown implementation"):
            ctx.prepare([request_for("model", ctx.selection)])

    def test_root_descriptor_declares_allocator_without_device_probe(self):
        from rtp_llm.models.dsv4.specs import declared_indexer_cache_mode

        self.assertEqual(declared_indexer_cache_mode(explicit_config()).value, "fp8")
        config = ModuleDispatchConfig(
            mode="auto", path_overrides=(("v4", "baseline.dsv4.model.v1"),)
        )
        self.assertEqual(declared_indexer_cache_mode(config).value, "fp8")
        with self.assertRaisesRegex(ValueError, "Unknown implementation"):
            declared_indexer_cache_mode(
                ModuleDispatchConfig(
                    mode="auto", impl_overrides=(("rtp.dsv4.model", "missing"),)
                )
            )

    def test_selected_plan_controls_actual_forward_phase(self):
        from rtp_llm.models.dsv4.specs import (
            forward_capabilities,
            validate_forward_phase,
        )

        ctx = self.context()
        ctx.prepare([request_for("model", ctx.selection)])
        capabilities = forward_capabilities(ctx.bindings)
        self.assertEqual(capabilities, frozenset({"prefill"}))
        validate_forward_phase(
            capabilities, is_prefill=True, has_decode_fmha=False, is_target_verify=False
        )
        for graph, verify, phase in (
            (False, False, "decode"),
            (True, False, "decode"),
            (True, True, "target_verify"),
        ):
            with self.assertRaisesRegex(RuntimeError, f"does not support {phase}"):
                validate_forward_phase(
                    capabilities,
                    is_prefill=False,
                    has_decode_fmha=graph,
                    is_target_verify=verify,
                )

    def test_unqualified_modes_rejected_by_real_predicate(self):
        for change in (
            {"model_type": "deepseek_v4_mtp"},
            {"indexer_cache_mode": "fp4"},
            {"role": "DECODE"},
        ):
            with self.subTest(change=change):
                ctx = self.context(selection(**change))
                with self.assertRaisesRegex(ValueError, "No compatible"):
                    ctx.prepare([request_for("model", ctx.selection)])

    def test_engine_feature_flags_no_longer_disqualify(self):
        for flagged in (
            {"cuda_graph": True},
            {"lora": True},
            {"eplb": True},
        ):
            with self.subTest(flagged=flagged):
                ctx = self.context(selection(**flagged))
                ctx.prepare([request_for("model", ctx.selection)])
                self.assertEqual(len(ctx.bindings), 130)

    def test_device_name_no_longer_disqualifies(self):
        for device_name in ("L20", "H20"):
            with self.subTest(device_name=device_name):
                ctx = self.context(selection(device_name=device_name))
                ctx.prepare([request_for("model", ctx.selection)])
                self.assertEqual(len(ctx.bindings), 130)

    def test_execution_digest_ignores_run_paths_but_detects_option_values(self):
        digests = []
        for rank in range(4):
            env = {
                "DSV4_TASK_RUN": f"/tmp/run/rank{rank}",
                "MOEDBG_DIR": f"/tmp/dumps/rank{rank}",
                "WORLD_RANK": str(rank),
                "DSV4_INDEXER_TOPK_BACKEND": "sglang",
                "MOEDBG": "0",
            }
            ctx = self.context(
                selection(rank, execution_options=execution_options_snapshot(env))
            )
            digests.append(ctx.prepare([request_for("model", ctx.selection)]))
        self.assertEqual(len(set(digests)), 1)
        env["DSV4_INDEXER_TOPK_BACKEND"] = "torch"
        ctx = self.context(selection(execution_options=execution_options_snapshot(env)))
        self.assertNotEqual(
            digests[0], ctx.prepare([request_for("model", ctx.selection)])
        )
        env["DSV4_INDEXER_TOPK_BACKEND"] = "sglang"
        env["DSV4_EXTRA_OPTION"] = "1"
        ctx = self.context(selection(execution_options=execution_options_snapshot(env)))
        self.assertNotEqual(
            digests[0], ctx.prepare([request_for("model", ctx.selection)])
        )

    def test_candidate_requires_explicit_selection(self):
        ctx = self.context(config=ModuleDispatchConfig(mode="auto"))
        with self.assertRaisesRegex(ValueError, "explicit selection required"):
            ctx.prepare([request_for("model", ctx.selection)])

    def test_manifest_import_is_light(self):
        code = """
import importlib.abc,sys
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch','triton','deep_gemm','flashinfer') or fullname.startswith('rtp_llm.ops'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectRuntime())
from rtp_llm.models.dsv4.adapter import get_registry as get_module_registry
registry=get_module_registry()
assert registry.frozen
assert get_module_registry() is registry
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
