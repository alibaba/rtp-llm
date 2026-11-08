"""CPU contract test: PPU attention overrides accept shared call-site keywords.

The shared ``AttentionFP8`` base keeps widening the keyword set it passes to
its per-platform hooks (e.g. the SM120 ``swa_slot_mapping`` / ``shared``
prefill-meta threading).  Any base call site the PPU class can reach must
stay bindable against the PPU override, otherwise PPU prefill raises
``TypeError`` at runtime.  Both sides are parsed from source (mirroring
``test_prefill_defaults.py``) so the test needs no native PPU dependency,
and the keyword sets are derived with ``ast`` so call-site drift is caught
automatically.
"""

import ast
import inspect
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
SHARED_ATTENTION = "rtp_llm/models_py/modules/dsv4/fp8/attention.py"
PPU_ATTENTION = "rtp_llm/models_py/modules/dsv4/ppu/attention/fp8/attention.py"

_EMPTY = inspect.Parameter.empty


def _class_methods(path, class_name):
    """``{method_name: FunctionDef}`` for ``class_name`` parsed from source."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(nodes) == 1, f"expected one {class_name} class in {path}"
    return {
        node.name: node
        for node in nodes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _signature(func):
    """``inspect.Signature`` for a parsed method; default values are dummies."""
    args = func.args
    positional = args.posonlyargs + args.args
    assert (
        positional and positional[0].arg == "self"
    ), f"expected 'self' as the first parameter of {func.name}"
    n_posonly = len(args.posonlyargs)
    defaults = [None] * (len(positional) - len(args.defaults))
    defaults += args.defaults
    params = [
        inspect.Parameter(
            node.arg,
            (
                inspect.Parameter.POSITIONAL_ONLY
                if index < n_posonly
                else inspect.Parameter.POSITIONAL_OR_KEYWORD
            ),
            default=_EMPTY if default is None else None,
        )
        for index, (node, default) in enumerate(zip(positional, defaults))
        if index != 0
    ]
    if args.vararg is not None:
        params.append(
            inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL)
        )
    params.extend(
        inspect.Parameter(
            node.arg,
            inspect.Parameter.KEYWORD_ONLY,
            default=_EMPTY if default is None else None,
        )
        for node, default in zip(args.kwonlyargs, args.kw_defaults)
    )
    if args.kwarg is not None:
        params.append(inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(params)


def _self_call_shapes(method, callee_name):
    """``(positional_count, keyword_names)`` for each ``self.<callee>(...)``."""
    shapes = []
    for node in ast.walk(method):
        func = getattr(node, "func", None)
        if (
            isinstance(node, ast.Call)
            and isinstance(func, ast.Attribute)
            and func.attr == callee_name
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
                keyword.arg is None for keyword in node.keywords
            ):
                raise AssertionError(
                    f"shared {method.name} calls self.{callee_name} with "
                    "*args/**kwargs unpacking; that call shape cannot be "
                    "checked statically"
                )
            shapes.append(
                (
                    len(node.args),
                    frozenset(keyword.arg for keyword in node.keywords),
                )
            )
    return shapes


class PpuAttentionOverrideContractTest(unittest.TestCase):
    shared = _class_methods(SHARED_ATTENTION, "AttentionFP8")
    ppu = _class_methods(PPU_ATTENTION, "AttentionFP8")

    def _assert_override_binds(self, callee_name, *caller_names):
        self.assertIn(callee_name, self.ppu, f"PPU override {callee_name} missing")
        signature = _signature(self.ppu[callee_name])
        for caller_name in caller_names:
            shapes = _self_call_shapes(self.shared[caller_name], callee_name)
            self.assertTrue(
                shapes, f"shared {caller_name} never calls self.{callee_name}"
            )
            for positional, keywords in shapes:
                with self.subTest(caller=caller_name, keywords=sorted(keywords)):
                    signature.bind(
                        *[None] * positional, **{name: None for name in keywords}
                    )

    def test_csa_prefill_meta_binds_shared_builder_keywords(self):
        self._assert_override_binds(
            "_build_csa_prefill_meta", "_build_shared_prefill_meta"
        )

    def test_sparse_fwd_override_binds_inherited_call_sites(self):
        # These base methods are inherited (not overridden) by the PPU class,
        # so their call sites run verbatim on PPU hardware; the SM120-only
        # values they pass are always None there.
        self._assert_override_binds(
            "_flash_mla_sparse_fwd_chunked_projected",
            "_attn_fp8_swa_via_concat",
            "_attn_fp8_swa_via_kv_full",
        )


if __name__ == "__main__":
    unittest.main()
