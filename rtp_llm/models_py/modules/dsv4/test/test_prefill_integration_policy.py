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


if __name__ == "__main__":
    unittest.main(verbosity=2)
