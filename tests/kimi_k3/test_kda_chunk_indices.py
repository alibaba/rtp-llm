"""Sequence ordinals must survive requests that contribute no KDA chunks."""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def prepare_indices():
    source = Path(__file__).resolve().parents[2] / (
        "rtp_llm/models_py/modules/kimi_k3/vllm_kda/common/index.py"
    )
    functions = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef)
                 and node.name in ("prepare_lens", "prepare_chunk_indices")]
    for node in functions:
        node.decorator_list = []
    namespace = {
        "torch": torch,
        "triton": SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y),
        "gpu_sync_allowed": contextlib.nullcontext,
        "async_tensor_h2d": lambda data, device, dtype: data.to(device=device, dtype=dtype),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["prepare_chunk_indices"]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("cu,expected", [
    ([0, 0, 3, 3, 132, 132, 193], [[1, 0], [3, 0], [3, 1], [3, 2], [5, 0]]),
    ([0, 64, 129], [[0, 0], [1, 0], [1, 1]]),
    ([0, 0, 0], []),
    ([0], []),
])
def test_chunk_indices_preserve_sequence_identity(prepare_indices, dtype, cu, expected):
    actual = prepare_indices(torch.tensor(cu, dtype=dtype), 64)
    assert actual.dtype == dtype
    assert actual.shape == (len(expected), 2)
    assert actual.tolist() == expected
