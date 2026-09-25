"""Planner dispatch checks without loading CUDA bindings."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.mark.parametrize(
    "lengths,verify,valid",
    [
        ([4] * 16, True, True),
        ([4] * 14 + [0] * 2, False, True),
        ([4, 2, 1, 1] + [0] * 12, False, True),
        ([4] * 14 + [0] * 2, True, False),
        ([0] * 16, False, False),
        ([], False, False),
    ],
)
def test_draft_update_preserves_ragged_inputs_for_paged_planner(lengths, verify, valid):
    class Base:
        def __init__(self, *args, **kwargs):
            self.planner_inputs = args[3]

    op = Mock()
    prefix = "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
    modules = {
        prefix + "flashinfer_mla": SimpleNamespace(MlaFlashInferDecodeOp=op),
        prefix + "flashinfer_mla_wrapper": SimpleNamespace(MlaFlashInferImplBase=Base),
        prefix + "mla_kv_cache_write_op": SimpleNamespace(MlaKVCacheWriteOp=Mock()),
    }
    path = Path(__file__).resolve().parents[2] / (
        "rtp_llm/models_py/modules/kimi_k3/mla_verify.py"
    )
    spec = importlib.util.spec_from_file_location("k3_mla_geometry_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    inputs = SimpleNamespace(
        input_lengths=torch.tensor(lengths, dtype=torch.int32),
        physical_token_count=sum(lengths),
        is_target_verify=verify,
        is_mtp_draft_update=not verify,
    )
    config, parallelism = Mock(), Mock()
    if not valid:
        with pytest.raises(ValueError):
            module.KimiK3MlaVerifyImpl(config, parallelism, Mock(), inputs, None, True)
        op.assert_not_called()
        return
    actual = module.KimiK3MlaVerifyImpl(
        config, parallelism, Mock(), inputs, None, True
    )
    assert actual.planner_inputs is inputs
    assert op.call_args.kwargs["max_bs"] == len(lengths)
    assert op.call_args.kwargs["num_tokens"] == sum(lengths)
    assert op.call_args.kwargs["is_cuda_graph"] is True
