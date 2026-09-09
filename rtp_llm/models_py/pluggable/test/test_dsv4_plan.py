import dataclasses
import json
import subprocess
import sys
import unittest
from types import SimpleNamespace

from rtp_llm.config.module_dispatch_config import ModuleDispatchConfig
from rtp_llm.device.device_type import DeviceType
from rtp_llm.models.dsv4.adapter import execution_options_snapshot
from rtp_llm.models.dsv4.adapter import get_registry as get_module_registry
from rtp_llm.models.dsv4.specs import CONTRACTS, request_for
from rtp_llm.models_py.pluggable.factory import (
    ModuleBuildContext,
    ModuleSelectionContext,
)
from rtp_llm.models_py.pluggable.platform import PlatformContext


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
        indexer_cache_mode="fp8",
        fp8_kv_cache=True,
    )
    metadata.update(changed)
    return ModuleSelectionContext(
        PlatformContext(DeviceType.Ppu, "ZW-M890P", rank), json.dumps(metadata)
    )


def explicit_config():
    return ModuleDispatchConfig.from_dict(
        {
            "mode": "auto",
            "platform": "ppu",
            "impl_overrides": {
                "rtp.dsv4." + kind: f"ppu.dsv4.{kind}.v1" for kind in CONTRACTS
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
            execution_options={"DSV4_PPU_SGLANG_MOE": "1"},
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
            get_module_registry(), selection(rank, **values), config, world_size=8
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

    def test_decode_shared_overlap_preserves_rank_protocol(self):
        digests = set()
        for rank in range(8):
            ctx = self.decode_context(
                rank,
                execution_options={
                    "DSV4_PPU_SGLANG_MOE": "1",
                    "DSV4_SHARED_EXPERT_MODE": "overlap",
                },
            )
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
            self.assertEqual(len(ctx.bindings), 130)
        self.assertEqual(len(digests), 1)
        sequential = self.decode_context()
        self.assertNotIn(
            sequential.prepare([request_for("model", sequential.selection)]), digests
        )

    def test_decode_attention_overlap_preserves_rank_protocol(self):
        digests = set()
        for rank in range(8):
            ctx = self.decode_context(
                rank,
                execution_options={
                    "DSV4_PPU_SGLANG_MOE": "1",
                    "DSV4_PPU_DECODE_ATTN_MODE": "overlap",
                },
            )
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
            self.assertEqual(len(ctx.bindings), 130)
        self.assertEqual(len(digests), 1)
        sequential = self.decode_context()
        self.assertNotIn(
            sequential.prepare([request_for("model", sequential.selection)]), digests
        )
        unsupported = self.decode_context(
            execution_options={
                "DSV4_PPU_SGLANG_MOE": "1",
                "DSV4_PPU_DECODE_ATTN_MODE": "unknown",
            }
        )
        with self.assertRaisesRegex(ValueError, "No compatible"):
            unsupported.prepare([request_for("model", unsupported.selection)])

    def test_decode_fused_hc_preserves_rank_protocol(self):
        digests = set()
        for rank in range(8):
            ctx = self.decode_context(
                rank,
                execution_options={
                    "DSV4_PPU_SGLANG_MOE": "1",
                    "DSV4_PPU_DECODE_HC_REDUCTION": "fused",
                },
            )
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
            self.assertEqual(len(ctx.bindings), 130)
        self.assertEqual(len(digests), 1)
        sequential = self.decode_context()
        self.assertNotIn(
            sequential.prepare([request_for("model", sequential.selection)]), digests
        )
        unsupported = self.decode_context(
            execution_options={
                "DSV4_PPU_SGLANG_MOE": "1",
                "DSV4_PPU_DECODE_HC_REDUCTION": "unknown",
            }
        )
        with self.assertRaisesRegex(ValueError, "No compatible"):
            unsupported.prepare([request_for("model", unsupported.selection)])

    def test_decode_fused_hc_norm_preserves_rank_protocol(self):
        digests = set()
        for rank in range(8):
            ctx = self.decode_context(
                rank,
                execution_options={
                    "DSV4_PPU_SGLANG_MOE": "1",
                    "DSV4_PPU_DECODE_HC_NORM": "fused",
                },
            )
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
            self.assertEqual(len(ctx.bindings), 130)
        self.assertEqual(len(digests), 1)
        sequential = self.decode_context()
        self.assertNotIn(
            sequential.prepare([request_for("model", sequential.selection)]), digests
        )
        unsupported = self.decode_context(
            execution_options={
                "DSV4_PPU_SGLANG_MOE": "1",
                "DSV4_PPU_DECODE_HC_NORM": "unknown",
            }
        )
        with self.assertRaisesRegex(ValueError, "No compatible"):
            unsupported.prepare([request_for("model", unsupported.selection)])

    def test_decode_shared_schedule_preserves_rank_protocol(self):
        options = {
            "DSV4_PPU_SGLANG_MOE": "1",
            "DSV4_SHARED_EXPERT_MODE": "overlap",
            "DSV4_PPU_DECODE_SHARED_SCHEDULE": "before_route",
        }
        digests = set()
        for rank in range(8):
            ctx = self.decode_context(rank, execution_options=options)
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
        self.assertEqual(len(digests), 1)
        for extra in (
            {"DSV4_PPU_DECODE_SHARED_SCHEDULE": "unknown"},
            {"DSV4_SHARED_EXPERT_MODE": "sequential"},
        ):
            ctx = self.decode_context(execution_options={**options, **extra})
            with self.assertRaisesRegex(ValueError, "No compatible"):
                ctx.prepare([request_for("model", ctx.selection)])
        later = self.decode_context(
            execution_options={
                **options,
                "DSV4_PPU_DECODE_SHARED_SCHEDULE": "after_route",
            }
        )
        self.assertNotIn(
            later.prepare([request_for("model", later.selection)]), digests
        )

    def test_decode_metadata_graph_preserves_rank_protocol(self):
        modes = set()
        for mode in ("eager", "graph", "graph_fused"):
            digests = set()
            for rank in range(8):
                ctx = self.decode_context(
                    rank,
                    execution_options={
                        "DSV4_PPU_SGLANG_MOE": "1",
                        "DSV4_PPU_DECODE_METADATA": mode,
                    },
                )
                digests.add(ctx.prepare([request_for("model", ctx.selection)]))
                self.assertEqual(len(ctx.bindings), 130)
            self.assertEqual(len(digests), 1)
            modes.update(digests)
        self.assertEqual(len(modes), 3)
        unsupported = self.decode_context(
            execution_options={
                "DSV4_PPU_SGLANG_MOE": "1",
                "DSV4_PPU_DECODE_METADATA": "unknown",
            }
        )
        with self.assertRaisesRegex(ValueError, "No compatible"):
            unsupported.prepare([request_for("model", unsupported.selection)])

    def test_decode_shared_rope_preserves_rank_protocol(self):
        options = {
            "DSV4_PPU_SGLANG_MOE": "1",
            "DSV4_PPU_DECODE_METADATA": "graph_fused",
            "DSV4_PPU_DECODE_ATTN_MODE": "overlap",
            "DSV4_PPU_DECODE_ROPE": "shared",
        }
        digests = set()
        for rank in range(8):
            ctx = self.decode_context(rank, execution_options=options)
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
        self.assertEqual(len(digests), 1)
        baseline = self.decode_context(
            execution_options={**options, "DSV4_PPU_DECODE_ROPE": "layer"}
        )
        self.assertNotIn(
            baseline.prepare([request_for("model", baseline.selection)]), digests
        )
        for change in (
            {"DSV4_PPU_DECODE_ROPE": "unknown"},
            {"DSV4_PPU_DECODE_METADATA": "eager"},
            {"DSV4_PPU_DECODE_ATTN_MODE": "sequential"},
        ):
            ctx = self.decode_context(execution_options={**options, **change})
            with self.assertRaisesRegex(ValueError, "No compatible"):
                ctx.prepare([request_for("model", ctx.selection)])

    def test_decode_fp8_v2_preserves_rank_protocol(self):
        digests = set()
        for rank in range(8):
            ctx = self.decode_context(
                rank,
                execution_options={
                    "DSV4_PPU_SGLANG_MOE": "1",
                    "DSV4_PPU_DECODE_FP8_QUANT": "v2",
                },
            )
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
            self.assertEqual(len(ctx.bindings), 130)
        self.assertEqual(len(digests), 1)
        sequential = self.decode_context()
        self.assertNotIn(
            sequential.prepare([request_for("model", sequential.selection)]), digests
        )
        unsupported = self.decode_context(
            execution_options={
                "DSV4_PPU_SGLANG_MOE": "1",
                "DSV4_PPU_DECODE_FP8_QUANT": "unknown",
            }
        )
        with self.assertRaisesRegex(ValueError, "No compatible"):
            unsupported.prepare([request_for("model", unsupported.selection)])

    def test_decode_moe_batch_hint_preserves_rank_protocol(self):
        digests = set()
        for rank in range(8):
            ctx = self.decode_context(
                rank,
                execution_options={
                    "DSV4_PPU_SGLANG_MOE": "1",
                    "DSV4_PPU_DECODE_MOE_HINT": "batch",
                },
            )
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
            self.assertEqual(len(ctx.bindings), 130)
        self.assertEqual(len(digests), 1)
        sequential = self.decode_context()
        self.assertNotIn(
            sequential.prepare([request_for("model", sequential.selection)]), digests
        )
        unsupported = self.decode_context(
            execution_options={
                "DSV4_PPU_SGLANG_MOE": "1",
                "DSV4_PPU_DECODE_MOE_HINT": "unknown",
            }
        )
        with self.assertRaisesRegex(ValueError, "No compatible"):
            unsupported.prepare([request_for("model", unsupported.selection)])

    def test_decode_moe_output_storage_is_part_of_rank_protocol(self):
        digests = set()
        for rank in range(8):
            ctx = self.decode_context(
                rank,
                execution_options={
                    "DSV4_PPU_SGLANG_MOE": "1",
                    "DSV4_PPU_DECODE_MOE_OUTPUT": "bf16",
                },
            )
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
        self.assertEqual(len(digests), 1)
        default = self.decode_context()
        self.assertNotIn(
            default.prepare([request_for("model", default.selection)]), digests
        )
        unsupported = self.decode_context(
            execution_options={
                "DSV4_PPU_SGLANG_MOE": "1",
                "DSV4_PPU_DECODE_MOE_OUTPUT": "fp16",
            }
        )
        with self.assertRaisesRegex(ValueError, "No compatible"):
            unsupported.prepare([request_for("model", unsupported.selection)])

    def test_decode_qkv_mode_is_part_of_rank_protocol(self):
        digests = set()
        for rank in range(8):
            ctx = self.decode_context(
                rank,
                execution_options={
                    "DSV4_PPU_SGLANG_MOE": "1",
                    "DSV4_PPU_DECODE_ATTN_MODE": "overlap",
                    "DSV4_PPU_DECODE_QKV": "merged",
                },
            )
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
        self.assertEqual(len(digests), 1)
        default = self.decode_context(
            execution_options={
                "DSV4_PPU_SGLANG_MOE": "1",
                "DSV4_PPU_DECODE_ATTN_MODE": "overlap",
            }
        )
        self.assertNotIn(
            default.prepare([request_for("model", default.selection)]), digests
        )
        for mode, schedule in (("unknown", "overlap"), ("merged", "sequential")):
            ctx = self.decode_context(
                execution_options={
                    "DSV4_PPU_SGLANG_MOE": "1",
                    "DSV4_PPU_DECODE_QKV": mode,
                    "DSV4_PPU_DECODE_ATTN_MODE": schedule,
                }
            )
            with self.assertRaisesRegex(ValueError, "No compatible"):
                ctx.prepare([request_for("model", ctx.selection)])

    def test_decode_indexer_schedule_is_part_of_rank_protocol(self):
        digests = set()
        for rank in range(8):
            ctx = self.decode_context(
                rank,
                execution_options={
                    "DSV4_PPU_SGLANG_MOE": "1",
                    "DSV4_PPU_DECODE_ATTN_MODE": "overlap",
                    "DSV4_PPU_DECODE_INDEXER": "overlap",
                },
            )
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
        self.assertEqual(len(digests), 1)
        default = self.decode_context(
            execution_options={
                "DSV4_PPU_SGLANG_MOE": "1",
                "DSV4_PPU_DECODE_ATTN_MODE": "overlap",
            }
        )
        self.assertNotIn(
            default.prepare([request_for("model", default.selection)]), digests
        )
        for mode, schedule in (("unknown", "overlap"), ("overlap", "sequential")):
            ctx = self.decode_context(
                execution_options={
                    "DSV4_PPU_SGLANG_MOE": "1",
                    "DSV4_PPU_DECODE_INDEXER": mode,
                    "DSV4_PPU_DECODE_ATTN_MODE": schedule,
                }
            )
            with self.assertRaisesRegex(ValueError, "No compatible"):
                ctx.prepare([request_for("model", ctx.selection)])

    def test_selected_plan_controls_actual_forward_phase(self):
        from rtp_llm.models.dsv4.specs import (
            forward_capabilities,
            validate_forward_phase,
        )

        for ctx, allowed in (
            (self.context(), "prefill"),
            (self.decode_context(), "decode"),
        ):
            ctx.prepare([request_for("model", ctx.selection)])
            capabilities = forward_capabilities(ctx.bindings)
            self.assertEqual(capabilities, frozenset({allowed}))
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
                if phase == allowed:
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
                    execution_options={"DSV4_PPU_SGLANG_MOE": "1"},
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
        config["impl_overrides"][
            "rtp.dsv4.attention"
        ] = "ppu.dsv4.attention.inverse_rope.v2"
        ctx = self.context(
            selection(
                indexer_cache_mode="fp4", execution_options={"DSV4_PPU_SGLANG_MOE": "1"}
            ),
            ModuleDispatchConfig.from_dict(config),
        )
        with self.assertRaisesRegex(ValueError, "state format mismatch"):
            ctx.prepare([request_for("model", ctx.selection)])

    def test_root_descriptor_declares_allocator_without_device_probe(self):
        from rtp_llm.models.dsv4.specs import declared_indexer_cache_mode

        self.assertEqual(declared_indexer_cache_mode(explicit_config()).value, "fp8")
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

    def test_inverse_rope_attention_preserves_weight_and_state_contracts(self):
        config = json.loads(explicit_config().to_string())
        config["impl_overrides"][
            "rtp.dsv4.attention"
        ] = "ppu.dsv4.attention.inverse_rope.v2"
        new = self.context(config=ModuleDispatchConfig.from_dict(config))
        old = self.context()
        new.prepare([request_for("model", new.selection)])
        old.prepare([request_for("model", old.selection)])
        previous = {b.request.path: b.implementation for b in old.bindings}
        changed = [
            b for b in new.bindings if b.request.module_id == "rtp.dsv4.attention"
        ]
        self.assertEqual(len(changed), 43)
        for binding in changed:
            a, b = binding.implementation, previous[binding.request.path]
            self.assertNotEqual(a.impl_id, b.impl_id)
            self.assertEqual(a.contract_id, b.contract_id)
            self.assertEqual(a.weight_format_id, b.weight_format_id)
            self.assertEqual(a.state_format_id, b.state_format_id)
        self.assertFalse(
            any(name.endswith("ppu_rope_attention") for name in sys.modules)
        )

    def test_tp_shared_moe_has_explicit_distinct_rank_invariant_protocol(self):
        selected = selection(execution_options={"DSV4_PPU_SGLANG_MOE": "1"})
        config = json.loads(explicit_config().to_string())
        config["impl_overrides"]["rtp.dsv4.moe"] = "ppu.dsv4.moe.tp_shared_bf16.v2"
        config = ModuleDispatchConfig.from_dict(config)
        digests = set()
        for rank in range(4):
            ctx = self.context(
                selection(rank, execution_options={"DSV4_PPU_SGLANG_MOE": "1"}), config
            )
            digests.add(ctx.prepare([request_for("model", ctx.selection)]))
            moe = [b for b in ctx.bindings if b.request.module_id == "rtp.dsv4.moe"]
            self.assertEqual(len(moe), 43)
            self.assertEqual(
                {b.implementation.collective_protocol_id for b in moe},
                {"ppu.dsv4.moe.tp4-shared-sharded-bf16-reduce.v2"},
            )
        self.assertEqual(len(digests), 1)
        old = self.context(selected)
        self.assertNotIn(old.prepare([request_for("model", old.selection)]), digests)

    def test_tp_shared_moe_rejects_incompatible_execution_options(self):
        config = json.loads(explicit_config().to_string())
        config["impl_overrides"]["rtp.dsv4.moe"] = "ppu.dsv4.moe.tp_shared_bf16.v2"
        config = ModuleDispatchConfig.from_dict(config)
        for options in (
            {},
            {"DSV4_PPU_SGLANG_MOE": "1", "DSV4_MOE_GATHER_FUSED": "0"},
            {"DSV4_PPU_SGLANG_MOE": "1", "DSV4_MOE_SHARED_EXPERT_OVERLAP": "1"},
        ):
            ctx = self.context(selection(execution_options=options), config)
            with self.assertRaises(ValueError):
                ctx.prepare([request_for("model", ctx.selection)])

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

    def test_unqualified_modes_rejected_by_real_predicate(self):
        for change in [
            {"tp_size": 8},
            {"ep_size": 8},
            {"cp_enabled": True},
            {"role": "DECODE"},
            {"role": "PREFILL"},
            {"speculative": True},
            {"cuda_graph": True},
            {"reuse_cache": True},
            {"lora": True},
            {"eplb": True},
            {"indexer_cache_mode": "fp4"},
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
