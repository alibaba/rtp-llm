"""CPU checks of the RTP/native KDA boundary, independent of CUDA bindings."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def adapter():
    source = (
        Path(__file__).resolve().parents[2]
        / "rtp_llm/models_py/modules/kimi_k3/native_kda_prefill.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    core = Mock(side_effect=lambda q, *args: torch.zeros_like(q))
    namespace = {
        "torch": torch,
        "KimiLinearKDAPrefill": object,
        "flash_kda_paged_prefill": core,
    }
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace
    )
    instance = object.__new__(namespace[cls.name])
    instance.local_num_v_heads = 2
    instance.head_k_dim = instance.head_v_dim = 128
    instance.alog = torch.zeros(2)
    instance.dt_bias = torch.zeros(256)
    instance.gate_lower_bound = -5.0
    return instance, core


def inputs(prefixes=(0, 0)):
    return SimpleNamespace(
        cu_seqlens=torch.tensor([0, 3, 5]),
        prefix_lengths=torch.tensor(prefixes),
        logical_request_count=1,
        kv_cache_kernel_block_id=torch.tensor([[2, 3], [0, 0]]),
    )


def call(instance, meta, cache=None):
    qkv = torch.arange(5 * 768, dtype=torch.float32).reshape(5, 768).bfloat16()
    gate, beta = torch.zeros(5, 256).bfloat16(), torch.zeros(5, 2).bfloat16()
    return instance._fla(qkv, gate, beta, cache, 4096, meta), qkv, beta


def test_native_prefill_preserves_projection_and_cache_views(adapter):
    instance, core = adapter
    states = torch.empty(6, 2, 128, 129)[..., :128]
    cache = object()
    instance._get_ssm_states = Mock(return_value=states)
    output, qkv, beta = call(instance, inputs((4096, 0)), cache)
    args = core.call_args.args
    for index, expected in enumerate(qkv.chunk(3, -1)):
        torch.testing.assert_close(args[index], expected.reshape(5, 2, 128))
    assert args[4] is beta
    assert args[8] is states
    assert args[9:] == ([0, 3, 5], [4096, 0], [[2, 3], [0, 0]], 4096)
    instance._get_ssm_states.assert_called_once_with(cache)
    assert output.shape == (5, 256)


def test_no_cache_allocates_fp32_and_excludes_virtual_request(adapter):
    instance, core = adapter
    output, _, _ = call(instance, inputs())
    args = core.call_args.args
    assert args[8].dtype == torch.float32
    assert args[8].shape == (3, 2, 128, 128)
    assert args[11:] == ([[1], [0]], 3)
    assert output.shape == (5, 256)


def test_cached_history_requires_state_storage(adapter):
    instance, core = adapter
    with pytest.raises(ValueError, match="cached KDA prefix"):
        call(instance, inputs((4096, 0)))
    core.assert_not_called()


def test_inconsistent_host_metadata_rejected_before_kernel(adapter):
    instance, core = adapter
    meta = inputs()
    meta.prefix_lengths = torch.zeros(1, dtype=torch.int64)
    with pytest.raises(ValueError, match="does not match"):
        call(instance, meta)
    core.assert_not_called()
