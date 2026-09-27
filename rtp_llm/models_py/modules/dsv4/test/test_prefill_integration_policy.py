"""CPU source-policy contracts; CUDA kernels and service execution are tested separately."""

from __future__ import annotations

import ast
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "rtp_llm/cpp/models/ModelTypes.h").is_file()
)
FP8 = "rtp_llm/models_py/modules/dsv4/fp8/"


def source_fn(path, name, ns, cls=None):
    """Compile the actual function, injecting only its dependencies."""
    nodes = ast.parse((ROOT / path).read_text()).body
    if cls:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    mod = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(mod), str(ROOT / path), "exec"), ns)
    return ns[name]


class IntegrationPolicyTest(unittest.TestCase):
    def test_woa_and_warmup_choose_same_arm_at_invocation(self):
        def decision(path, method, needle, cls=None):
            ns = {"os": os, "is_sm120": lambda d: True}
            tree = ast.parse((ROOT / path).read_text())
            nodes = (
                tree.body
                if not cls
                else next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == cls
                ).body
            )
            # Locate by method anywhere in the file; class name is not part of the contract.
            f = next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == method
            )
            test = next(
                n.test
                for n in ast.walk(f)
                if isinstance(n, ast.If) and needle in ast.unparse(n.test)
            )
            return eval(
                compile(ast.Expression(test), "<actual-guard>", "eval"),
                dict(ns, device=120, o_fp8=types.SimpleNamespace(device=120)),
            )

        for flag in ("0", "1"):
            with patch.dict(os.environ, {"DSV4_SM120_WOA_EINSUM": flag}, clear=True):
                self.assertEqual(
                    decision(
                        FP8 + "attention.py",
                        "_wo_a_einsum_from_fp8",
                        "DSV4_SM120_WOA_EINSUM",
                    ),
                    flag == "0",
                )
                self.assertEqual(
                    decision(
                        "rtp_llm/models_py/modules/dsv4/dsv4_kernel_jit_warmup.py",
                        "warmup_batched_fp8_einsum_jit",
                        "DSV4_SM120_WOA_EINSUM",
                    ),
                    flag == "0",
                )

    def test_authoritative_rank_layout_profiles(self):
        import importlib.util
        import sys

        p = ROOT / "rtp_llm/models_py/distributed/rank_layout.py"
        spec = importlib.util.spec_from_file_location("_unified_rank_layout", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        for pp, dp, tp in ((4, 1, 2), (2, 1, 4), (1, 4, 1)):
            layout = mod.RankLayout(pp_size=pp, dp_size=dp, tp_size=tp)
            self.assertEqual(
                [layout.world_rank_of(layout.coord_of(r)) for r in range(pp * dp * tp)],
                list(range(pp * dp * tp)),
            )
        layout = mod.RankLayout(pp_size=2, dp_size=1, tp_size=4)
        self.assertEqual(layout.groups(mod.Group.STAGE), [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(
            [layout.ep_rank_of(r, 4) for r in range(8)], [0, 1, 2, 3, 0, 1, 2, 3]
        )

    def test_pp_refactored_helpers_use_resolved_local_cp(self):
        text = (ROOT / "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc").read_text()
        for start, end in (
            (
                "void PPExecutor::prepareDSparkCommitInput",
                "void PPExecutor::sampleTokens",
            ),
            (
                "torch::Tensor PPExecutor::runDraftStep",
                "absl::Status PPExecutor::process(",
            ),
            ("absl::Status PPExecutor::process(", "bool PPExecutor::updateEplbConfig"),
        ):
            self.assertEqual(text.count(start), 1)
            body = text.split(start, 1)[1].split(end, 1)[0]
            self.assertIn("parallelism_config_.local_cp_enabled()", body)
            self.assertNotIn("prefill_cp_config.is_enabled()", body)
        draft = text.split("torch::Tensor PPExecutor::runDraftStep", 1)[1]
        clear = draft.index("draft_input.last_hidden_states = torch::Tensor()")
        sync = draft.index("tpSyncModelInputs", clear)
        bind = draft.index(
            "draft_input.last_hidden_states = target_hidden_states", sync
        )
        forward = draft.index("forwardDraftModel(draft_input)", bind)
        self.assertLess(clear, sync)
        self.assertLess(sync, bind)
        self.assertLess(bind, forward)

    def test_wire_hint_is_not_duplicated_by_automatic_merge(self):
        text = (ROOT / "rtp_llm/cpp/models/ModelTypes.h").read_text()
        enum = text.split("enum GptModelInputIndex")[1].split("};")[0]
        fields = [line.split("//")[0].strip().rstrip(",") for line in enum.splitlines()]
        self.assertEqual(fields.count("pdSeparation"), 1)
        text = (ROOT / "rtp_llm/cpp/models/ModelTypes.cc").read_text()
        self.assertEqual(text.count("shape_hints[GptModelInputIndex::pdSeparation]"), 1)
        self.assertEqual(text.count("inputs.pd_separation                   ="), 1)


class PpEpProfilePolicyTest(unittest.TestCase):
    """Exact-shape PP+EP profile contract: CEP4PP2 target + CEP2PP2 local PD proxy."""

    @staticmethod
    def _load_ep_stage_context():
        import importlib.util
        import sys

        rl_path = ROOT / "rtp_llm/models_py/distributed/rank_layout.py"
        spec = importlib.util.spec_from_file_location(
            "rtp_llm.models_py.distributed.rank_layout", rl_path
        )
        rl = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = rl
        spec.loader.exec_module(rl)
        esc_path = ROOT / "rtp_llm/models_py/distributed/ep_stage_context.py"
        spec = importlib.util.spec_from_file_location(
            "_ep_stage_context_under_test", esc_path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _cfg(pp, dp, tp, ep, world, role=None, cp_size=0, sharded=False):
        cp = types.SimpleNamespace(
            is_prefill_enabled=lambda: True,
            kv_cache_sharded=sharded,
            prefill_cp_size=cp_size,
        )
        return types.SimpleNamespace(
            pp_size=pp,
            dp_size=dp,
            tp_size=tp,
            ep_size=ep,
            world_size=world,
            world_rank=0,
            ep_rank=0,
            role_type=role,
            prefill_cp_config=cp,
        )

    def test_exact_profiles_accepted(self):
        mod = self._load_ep_stage_context()
        for shape in mod.PP_EP_PROFILES:
            cfg = self._cfg(*shape)
            layout = mod.validate_pp_ep_shape(cfg)
            self.assertEqual(layout.world_size(), shape[4])

    def test_near_miss_shapes_rejected(self):
        mod = self._load_ep_stage_context()
        for shape in (
            (2, 1, 4, 4, 4),  # CEP4PP2 with proxy world
            (2, 1, 2, 2, 8),  # CEP2PP2 with target world
            (2, 1, 2, 2, 2),  # truncated world
            (2, 1, 2, 4, 4),  # ep != tp
            (4, 1, 2, 2, 8),  # pp4
            (2, 2, 2, 2, 8),  # dp2
            (2, 1, 3, 3, 6),  # unvalidated width
        ):
            with self.assertRaises(ValueError, msg=f"shape {shape} must be rejected"):
                mod.validate_pp_ep_shape(self._cfg(*shape))

    def test_cp_config_checks_use_configured_tp(self):
        mod = self._load_ep_stage_context()
        # cp_size equal to the profile's own tp is fine; foreign values are not.
        mod.validate_pp_ep_shape(self._cfg(2, 1, 2, 2, 4, cp_size=2))
        mod.validate_pp_ep_shape(self._cfg(2, 1, 4, 4, 8, cp_size=4))
        with self.assertRaises(ValueError):
            mod.validate_pp_ep_shape(self._cfg(2, 1, 2, 2, 4, cp_size=4))
        with self.assertRaises(ValueError):
            mod.validate_pp_ep_shape(self._cfg(2, 1, 4, 4, 8, cp_size=2))
        with self.assertRaises(ValueError):
            mod.validate_pp_ep_shape(self._cfg(2, 1, 2, 2, 4, sharded=True))

    def test_role_gate_admits_prefill_and_pdfusion(self):
        import enum
        import sys

        mod = self._load_ep_stage_context()

        class RoleType(enum.Enum):
            PDFUSION = 0
            PREFILL = 1
            DECODE = 2

        fake_ops = types.ModuleType("rtp_llm.ops")
        fake_ops.RoleType = RoleType
        sys.modules["rtp_llm.ops"] = fake_ops
        try:
            hw = types.SimpleNamespace(
                enable_cuda_graph=False, enable_native_cuda_graph=False
            )
            for role in (RoleType.PDFUSION, RoleType.PREFILL):
                for shape in mod.PP_EP_PROFILES:
                    mod.validate_pp_ep_target(
                        self._cfg(*shape, role=role),
                        hw_kernel_config=hw,
                        is_sm120=True,
                        has_grouped_fp4=True,
                    )
            with self.assertRaises(ValueError, msg="DECODE must stay rejected"):
                mod.validate_pp_ep_target(
                    self._cfg(2, 1, 2, 2, 4, role=RoleType.DECODE),
                    hw_kernel_config=hw,
                    is_sm120=True,
                    has_grouped_fp4=True,
                )
            with self.assertRaises(ValueError, msg="cuda graphs stay rejected"):
                mod.validate_pp_ep_target(
                    self._cfg(2, 1, 2, 2, 4, role=RoleType.PREFILL),
                    hw_kernel_config=types.SimpleNamespace(
                        enable_cuda_graph=True, enable_native_cuda_graph=False
                    ),
                    is_sm120=True,
                    has_grouped_fp4=True,
                )
        finally:
            del sys.modules["rtp_llm.ops"]

    def test_cpp_gate_enumerates_both_profiles(self):
        text = (ROOT / "rtp_llm/cpp/config/ConfigModules.h").read_text()
        self.assertEqual(text.count("pp_ep_shape_valid"), 2)  # declaration + one use
        self.assertIn("tp_size == 4 && world_size == 8", text)
        self.assertIn("tp_size == 2 && world_size == 4", text)
        self.assertIn(
            "role_type == RoleType::PDFUSION || role_type == RoleType::PREFILL", text
        )

    def test_one_token_guard_is_per_request_pd_aware(self):
        # The guard must exempt only streams that took the PD
        # branch (role PREFILL && per-request queryPdSep()), not every request
        # on a PREFILL-role server (a local-bypass multi-token request must
        # still be rejected). This source-policy test only checks that
        # production routes the decision through the shared predicate with the
        # real stream state; the executable truth table lives in the C++ test
        # //rtp_llm/cpp/normal_engine/test:pp_prefill_guard_policy_test, which
        # calls the same predicate.
        text = (ROOT / "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc").read_text()
        start = "void PPExecutor::prepareStreams"
        body = text.split(start, 1)[1].split("void PPExecutor::", 1)[0]
        self.assertIn("dsv4PrefillCpGuardRejects(", body)
        self.assertIn("stream->queryPdSep()", body)
        self.assertIn("parallelism_config_.role_type", body)
        # The predicate itself must require BOTH the PREFILL role and the
        # per-request PD flag for the exemption.
        policy = (
            ROOT / "rtp_llm/cpp/normal_engine/pipeline/PPPrefillGuardPolicy.h"
        ).read_text()
        self.assertIn("role == RoleType::PREFILL) && stream_pd_separation", policy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
