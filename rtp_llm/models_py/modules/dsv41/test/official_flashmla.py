"""Load unmodified quantization and oracle functions from pinned FlashMLA sources."""

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

SOURCE_HASHES = {
    "quant.py": "594959de2ff53bd6f08ff2f43733bf35e36ecc1f2dc48375153a0b06a4a3ac3f",
    "ref.py": "b7abcf463a7c128d0be66193b93e5eb0416050cce086117545aec09de5b98fca",
}


def load_reference(source_root):
    root = Path(source_root) / "tests"
    for name, digest in SOURCE_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"FlashMLA official source {name} differs from the pin")
    spec = importlib.util.spec_from_file_location(
        "dsv41_official_flashmla_quant", root / "quant.py"
    )
    quant = importlib.util.module_from_spec(spec)
    # The immutable upstream archive carries kernelkit beside quant.py.
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(quant)
    finally:
        sys.path.pop(0)
    tree = ast.parse((root / "ref.py").read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "ref_sparse_attn_decode"
    ]
    if len(functions) != 1:
        raise ValueError("pinned FlashMLA reference must define one decode oracle")
    scope = {
        "torch": torch,
        "Optional": Optional,
        "Tuple": Tuple,
        "TestParam": object,
        "TestcaseForDecode": object,
        "KVScope": object,
    }
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]), str(root / "ref.py"), "exec"
        ),
        scope,
    )
    return quant, scope["ref_sparse_attn_decode"]
