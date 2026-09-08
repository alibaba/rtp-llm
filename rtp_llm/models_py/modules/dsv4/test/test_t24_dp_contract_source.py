"""T24 source-shape checks which do not import RTP, torch, or native bindings.

These checks prove only that the reviewed control-flow and call-shape fences
are present in the checked-in sources.  Runtime strategy selection,
allocation ownership, and CUDA-graph behavior remain unverified here.
"""

import ast
from pathlib import Path
import unittest


_DSV4 = Path(__file__).resolve().parents[1]


def _tree(*parts: str) -> ast.Module:
    return ast.parse(_DSV4.joinpath(*parts).read_text(encoding="utf-8"))


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _calls(node: ast.AST, attr: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == attr
    ]


class T24SourceContractTest(unittest.TestCase):
    def test_moe_cfg_has_ep_not_dp_and_ep_gt1_requires_mega(self):
        tree = _tree("moe", "strategies", "base.py")
        cfg = _class(tree, "MoeCfg")
        fields = {
            node.target.id
            for node in cfg.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertIn("ep_size", fields)
        self.assertNotIn("dp_size", fields)

        select = _function(tree, "select_strategy")
        ep_guards = [
            node
            for node in ast.walk(select)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Attribute)
            and node.test.left.attr == "ep_size"
        ]
        self.assertTrue(ep_guards)
        ep_body = ep_guards[-1]
        self.assertTrue(any(_calls(stmt, "can_handle") for stmt in ep_body.body))
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(ep_body)))

    def test_forced_non_mega_branch_raises_before_return(self):
        tree = _tree("moe", "strategies", "base.py")
        select = _function(tree, "select_strategy")
        branches = [
            node
            for node in ast.walk(select)
            if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
        ]
        branch = next(
            node
            for node in branches
            if any(
                isinstance(part, ast.Compare)
                and isinstance(part.left, ast.Attribute)
                and part.left.attr == "ep_size"
                for part in branch_parts(node.test)
            )
            and any(
                isinstance(part, ast.Compare)
                and any(
                    isinstance(name, ast.Name) and name.id == "cls"
                    for name in ast.walk(part)
                )
                for part in branch_parts(node.test)
            )
        )
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(branch)))
        self.assertFalse(any(isinstance(node, ast.Return) for node in branch.body))

    def test_topk_capture_guard_precedes_empty_and_helper_is_called(self):
        tree = _tree("fp8", "indexer.py")
        workspace = _function(tree, "_get_decode_topk_workspace")
        empty = next(
            call
            for call in _calls(workspace, "empty")
            if isinstance(call.func.value, ast.Name) and call.func.value.id == "torch"
        )
        guard = next(
            node
            for node in ast.walk(workspace)
            if isinstance(node, ast.If)
            and any(
                isinstance(name, ast.Name) and name.id == "_decode_topk_capture_active"
                for name in ast.walk(node.test)
            )
        )
        self.assertLess(guard.lineno, empty.lineno)
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(guard)))

        vectorized = _function(tree, "forward_decode_vectorized")
        self.assertTrue(
            any(
                isinstance(call.func, ast.Name) and call.func.id == "_run_decode_topk"
                for call in ast.walk(vectorized)
                if isinstance(call, ast.Call)
            )
        )

    def test_graph_metadata_is_allocated_in_init(self):
        tree = _tree("decode", "decode_fmha_impl.py")
        impl = _class(tree, "DSv4DecodeFmhaImpl")
        init = next(node for node in impl.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        assignment = next(
            node
            for node in ast.walk(init)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Attribute)
            and node.target.attr == "metadata"
        )
        self.assertTrue(any(isinstance(node, ast.Name) and node.id == "allocate_decode_metadata" for node in ast.walk(assignment.value)))

    def test_graph_prepare_forbids_reallocation(self):
        tree = _tree("decode", "decode_fmha_impl.py")
        impl = _class(tree, "DSv4DecodeFmhaImpl")
        prepare = next(node for node in impl.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_cuda_graph")
        calls = _calls(prepare, "prepare")
        self.assertTrue(calls)
        self.assertTrue(
            any(
                keyword.arg == "forbid_realloc"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for call in calls
                for keyword in call.keywords
            )
        )

    def test_decode_abi_declares_numerical_status(self):
        tree = _tree("decode", "forward.py")
        for name in ("forward_layers", "forward_decode"):
            function = _function(tree, name)
            args = function.args.args + function.args.kwonlyargs
            self.assertIn("numerical_status", {arg.arg for arg in args})


def branch_parts(node: ast.AST) -> list[ast.AST]:
    """Flatten boolean terms for a narrow, source-shape-only assertion."""
    if isinstance(node, ast.BoolOp):
        return [part for value in node.values for part in branch_parts(value)]
    return [node]


if __name__ == "__main__":
    unittest.main()
