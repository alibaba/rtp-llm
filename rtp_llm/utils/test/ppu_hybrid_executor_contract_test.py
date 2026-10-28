import ast
from pathlib import Path

import pytest


pytestmark = [pytest.mark.ppu, pytest.mark.PPU_ZW810E]


def test_bf16_contiguous_activation_call_is_imported():
    source_path = (
        Path(__file__).parents[2]
        / "models_py/modules/factory/fused_moe/ppu_impl/executors/deepgemm_hybrid_executor.py"
    )
    tree = ast.parse(source_path.read_text(), filename=str(source_path))

    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "ppu_silu_and_mul_bf16" not in called_names
    assert "silu_and_mul" in called_names
    assert "silu_and_mul" in imported_names
