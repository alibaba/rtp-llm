import dataclasses
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
from rtp_llm.models.dsv4.specs import CONTRACTS, request_for
from rtp_llm.models_py.pluggable.factory import (
    ModuleBuildContext,
    ModuleSelectionContext,
)
from rtp_llm.platforms.ppu.models.dsv4.manifest import (
    DECODE_EXECUTION_OPTIONS,
    PREFILL_EXECUTION_OPTIONS,
)


def selection(rank=0, **changed):
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
        indexer_cache_mode="fp4",
        execution_options=dict(PREFILL_EXECUTION_OPTIONS),
        fp8_kv_cache=True,
    )
    metadata.update(changed)
    return ModuleSelectionContext(
        DeviceRuntimeContext(DeviceType.Ppu, "ZW-M890P", rank), json.dumps(metadata)
    )


def explicit_config():
    return ModuleDispatchConfig.from_dict(
        {
            "mode": "auto",
            "platform": "ppu",
            "impl_overrides": {
                "rtp.dsv4." + kind: f"ppu.dsv4.{kind}.fp4_indexer.v1"
                for kind in CONTRACTS
            },
        }
    )


class Dsv4PlanTest(unittest.TestCase):
    def decode_context(self, rank=0, **changed):
        values = dict(
            tp_size=1,
            dp_size=8,
            ep_size=8,
            world_size=8,
            role="DECODE",
            cache_geometry={"kernel_tokens_per_block": 256},
            cuda_graph=True,
            indexer_cache_mode="fp4",
            execution_options=dict(DECODE_EXECUTION_OPTIONS),
            moe_communication={
                "enabled": True,
                "low_latency": True,
                "all_gather": False,
                "ffn_disaggregate": False,
                "max_generate_batch_size": 128,
            },
        )
        values.update(changed)
        config = ModuleDispatchConfig(
            mode="auto",
            platform="ppu",
            impl_overrides=tuple(
                ("rtp.dsv4." + kind, f"ppu.dsv4.{kind}.fp4_decode.v1")
                for kind in CONTRACTS
            ),
        )
        return ModuleBuildContext(
            get_module_registry(), selection(rank, **values), config,
            world_size=values["world_size"],
        )

    def test_decode_four_and_eight_rank_mtp_plans(self):
        from rtp_llm.models.dsv4.adapter import validate_parallelism

        topology_digests = {}
        for world in (4, 8):
            validate_parallelism(SimpleNamespace(
                tp_size=1, dp_size=world, ep_size=world, pp_size=1,
                world_size=world, role_type=SimpleNamespace(name="DECODE"),
            ))
            for model_type, layers in (("deepseek_v4", 43), ("deepseek_v4_mtp", 1)):
                digests = set()
                for rank in range(world):
                    ctx = self.decode_context(
                        rank, dp_size=world, ep_size=world, world_size=world,
                        model_type=model_type, num_layers=layers,
                        layer_compress_ratios=[0] if layers == 1 else [4] * 43,
                        speculative=True, speculative_type="MTP", gen_num_per_cycle=3,
                    )
                    digests.add(ctx.prepare([request_for("model", ctx.selection)]))
                    self.assertEqual(len(ctx.bindings), 1 + 3 * layers)
                self.assertEqual(len(digests), 1)
                topology_digests[world, model_type] = digests.pop()
        for model_type in ("deepseek_v4", "deepseek_v4_mtp"):
            self.assertNotEqual(
                topology_digests[4, model_type], topology_digests[8, model_type]
            )

    def test_decode_plan_covers_ep_world_before_runtime_imports(self):
        from rtp_llm.models.dsv4.adapter import validate_parallelism

        pc = SimpleNamespace(
            tp_size=1,
            dp_size=8,
            ep_size=8,
            pp_size=1,
            world_size=8,
            role_type=SimpleNamespace(name="DECODE"),
        )
        validate_parallelism(pc)
        digests = set()
        for rank in range(8):
            ctx = self.decode_context(rank)
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
            self.assertEqual(len(ctx.bindings), 130)
            self.assertTrue(
                all(
                    b.request.required_capabilities == frozenset({"decode"})
                    for b in ctx.bindings
                )
            )
            self.assertTrue(
                all(not b.implementation.auto_selectable for b in ctx.bindings)
            )
        self.assertEqual(len(digests), 1)
        pc.ep_size = 4
        with self.assertRaises(ValueError):
            validate_parallelism(pc)

    def test_only_measured_execution_combinations_are_published(self):
        for make_context, expected in (
            (self.decode_context, DECODE_EXECUTION_OPTIONS),
            (lambda **kw: self.context(selection(**kw)), PREFILL_EXECUTION_OPTIONS),
        ):
            for key in expected:
                for value in (None, "unsupported"):
                    options = dict(expected)
                    if value is None:
                        options.pop(key)
                    else:
                        options[key] = value
                    with self.subTest(option=key, value=value):
                        ctx = make_context(execution_options=options)
                        with self.assertRaisesRegex(ValueError, "No compatible"):
                            ctx.prepare([request_for("model", ctx.selection)])

    def test_selected_plan_controls_actual_forward_phase(self):
        from rtp_llm.models.dsv4.specs import (
            forward_capabilities,
            validate_forward_phase,
        )

        for ctx, allowed in (
            (self.context(), {"prefill"}),
            (self.decode_context(), {"decode", "target_verify"}),
        ):
            ctx.prepare([request_for("model", ctx.selection)])
            capabilities = forward_capabilities(ctx.bindings)
            self.assertEqual(capabilities, frozenset(allowed))
            for is_prefill, graph, verify, phase in (
                (True, False, False, "prefill"),
                (False, False, False, "decode"),
                (False, True, False, "decode"),
                (True, True, True, "target_verify"),
            ):
                args = dict(
                    is_prefill=is_prefill,
                    has_decode_fmha=graph,
                    is_target_verify=verify,
                )
                if phase in allowed:
                    validate_forward_phase(capabilities, **args)
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, f"does not support {phase}"
                    ):
                        validate_forward_phase(capabilities, **args)

    def test_decode_rejects_mismatched_runtime_and_communication(self):
        from rtp_llm.models.dsv4.specs import validate_runtime_role

        ctx = self.decode_context()
        validate_runtime_role(
            ctx.selection.model_metadata, is_decode_role=True, is_speculative=False
        )
        with self.assertRaisesRegex(ValueError, "Runtime Decode role"):
            validate_runtime_role(
                ctx.selection.model_metadata, is_decode_role=False, is_speculative=False
            )
        with self.assertRaisesRegex(ValueError, "Runtime speculation"):
            validate_runtime_role(
                ctx.selection.model_metadata, is_decode_role=True, is_speculative=True
            )
        for changed in (
            {"tp_size": 4},
            {"ep_size": 4},
            {"dp_size": 4},
            {"role": "PDFUSION"},
            {"cache_geometry": {}},
            {"cache_geometry": {"kernel_tokens_per_block": 1024}},
            {"speculative": True},
            {"cp_enabled": True},
            {"reuse_cache": True},
            {"moe_communication": {}},
            {"indexer_cache_mode": "fp8"},
            {
                "execution_options": {
                    "DSV4_PPU_SGLANG_MOE": "1",
                    "DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm",
                }
            },
        ):
            with self.subTest(changed=changed):
                ctx = self.decode_context(**changed)
                with self.assertRaisesRegex(ValueError, "No compatible"):
                    ctx.prepare([request_for("model", ctx.selection)])

    def test_mtp3_target_and_draft_use_separate_complete_plans(self):
        self._check_mtp_target_and_draft_plans(3)

    def test_mtp2_target_and_draft_use_separate_complete_plans(self):
        self._check_mtp_target_and_draft_plans(2)

    def test_mtp1_target_and_draft_use_separate_complete_plans(self):
        self._check_mtp_target_and_draft_plans(1)

    def _check_mtp_target_and_draft_plans(self, gamma):
        from rtp_llm.models.dsv4.specs import validate_runtime_role

        for decode in (False, True):
            model_digests = []
            for model_type, layers in (("deepseek_v4", 43), ("deepseek_v4_mtp", 1)):
                digests = set()
                for rank in range(8 if decode else 4):
                    changed = dict(
                        role="DECODE" if decode else "PREFILL",
                        model_type=model_type,
                        num_layers=layers,
                        layer_compress_ratios=[0] if layers == 1 else [4] * 43,
                        speculative=True,
                        speculative_type="MTP",
                        gen_num_per_cycle=gamma,
                    )
                    ctx = (
                        self.decode_context(rank, **changed)
                        if decode
                        else self.context(selection(rank, **changed))
                    )
                    digests.add(ctx.prepare([request_for("model", ctx.selection)]))
                    self.assertEqual(len(ctx.bindings), 1 + 3 * layers)
                    verify = decode and model_type == "deepseek_v4"
                    self.assertEqual(
                        "target_verify"
                        in ctx.bindings[0].request.required_capabilities,
                        verify,
                    )
                    validate_runtime_role(
                        ctx.selection.model_metadata,
                        is_decode_role=decode,
                        is_speculative=True,
                    )
                    with self.assertRaisesRegex(ValueError, "Runtime speculation"):
                        validate_runtime_role(
                            ctx.selection.model_metadata,
                            is_decode_role=decode,
                            is_speculative=False,
                        )
                self.assertEqual(len(digests), 1)
                model_digests.extend(digests)
            self.assertNotEqual(*model_digests)

    def test_mtp_contract_rejects_other_proposals_and_draft_geometry(self):
        valid = dict(
            model_type="deepseek_v4_mtp",
            num_layers=1,
            layer_compress_ratios=[0],
            speculative=True,
            speculative_type="MTP",
            gen_num_per_cycle=3,
        )
        for changed in (
            {"gen_num_per_cycle": 0},
            {"gen_num_per_cycle": 4},
            {"speculative_type": "EAGLE3"},
            {"speculative": False},
            {"model_type": "deepseek_v4_dspark"},
            {"num_layers": 43},
            {"layer_compress_ratios": [4]},
            {"hidden_size": 2048},
        ):
            with self.subTest(changed=changed):
                ctx = self.decode_context(**{**valid, **changed})
                with self.assertRaisesRegex(ValueError, "No compatible"):
                    ctx.prepare([request_for("model", ctx.selection)])

    def test_fp4_requires_consistent_explicit_state_contracts(self):
        from rtp_llm.models.dsv4.specs import STATE_FORMAT_FP4

        config = json.loads(explicit_config().to_string())
        config["impl_overrides"] = {
            "rtp.dsv4." + kind: f"ppu.dsv4.{kind}.fp4_indexer.v1" for kind in CONTRACTS
        }
        digests = set()
        for rank in range(4):
            ctx = self.context(
                selection(
                    rank,
                    indexer_cache_mode="fp4",
                    execution_options=dict(PREFILL_EXECUTION_OPTIONS),
                ),
                ModuleDispatchConfig.from_dict(config),
            )
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
            self.assertEqual(len(ctx.bindings), 130)
            self.assertEqual(
                {b.implementation.state_format_id for b in ctx.bindings},
                {STATE_FORMAT_FP4},
            )
        self.assertEqual(len(digests), 1)
        config["impl_overrides"]["rtp.dsv4.attention"] = "ppu.dsv4.attention.retired.v1"
        ctx = self.context(
            selection(
                indexer_cache_mode="fp4",
                execution_options=dict(PREFILL_EXECUTION_OPTIONS),
            ),
            ModuleDispatchConfig.from_dict(config),
        )
        with self.assertRaisesRegex(ValueError, "Unknown implementation"):
            ctx.prepare([request_for("model", ctx.selection)])

    def test_root_descriptor_declares_allocator_without_device_probe(self):
        from rtp_llm.models.dsv4.specs import declared_indexer_cache_mode

        self.assertEqual(declared_indexer_cache_mode(explicit_config()).value, "fp4")
        config = ModuleDispatchConfig(
            mode="auto", path_overrides=(("v4", "ppu.dsv4.model.fp4_indexer.v1"),)
        )
        self.assertEqual(declared_indexer_cache_mode(config).value, "fp4")
        with self.assertRaisesRegex(ValueError, "Unknown implementation"):
            declared_indexer_cache_mode(
                ModuleDispatchConfig(
                    mode="auto", impl_overrides=(("rtp.dsv4.model", "missing"),)
                )
            )

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

    def test_unqualified_modes_rejected_by_real_predicate(self):
        for change in [
            {"tp_size": 8},
            {"ep_size": 8},
            {"cp_enabled": True},
            {"role": "DECODE"},
            {"speculative": True},
            {"cuda_graph": True},
            {"reuse_cache": True},
            {"lora": True},
            {"eplb": True},
            {"indexer_cache_mode": "fp8"},
            {"fp8_kv_cache": False},
        ]:
            with self.subTest(change=change):
                ctx = self.context(selection(**change))
                with self.assertRaisesRegex(ValueError, "No compatible"):
                    ctx.prepare([request_for("model", ctx.selection)])

    def test_execution_digest_ignores_run_paths_but_detects_numeric_options(self):
        digests = []
        for rank in range(4):
            env = {
                **PREFILL_EXECUTION_OPTIONS,
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
