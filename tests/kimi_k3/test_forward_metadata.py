"""CPU dispatch regression; actual CUDA capture/replay is a separate GPU gate."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("prefill,verify", [(True, False), (True, True), (False, False)])
def test_conv_metadata_only_for_kda_prefill(linear, prefill, verify):
    # Execute the production forward method without importing CUDA-only bindings.
    source = Path(__file__).resolve().parents[2] / (
        "rtp_llm/models_py/model_desc/kimi_k3.py"
    )
    tree = ast.parse(source.read_text())
    model = next(
        node for node in tree.body if getattr(node, "name", None) == "KimiK3Model"
    )
    method = next(
        node for node in model.body if getattr(node, "name", None) == "_forward_layers"
    )
    prepare = Mock(return_value=object())
    needs_conv = linear and prefill and not verify
    if not needs_conv:
        prepare.side_effect = AssertionError("Unexpected host-side conv metadata access")
    primary = SimpleNamespace(is_prefill=prefill, is_target_verify=verify)
    # MLA-only graph inputs need not expose sequence lengths to the host at all.
    if needs_conv:
        primary.cu_seqlens_device = object()
    received = []

    class Layer:
        layer_type = "linear" if linear else "mla"

        def __call__(self, hidden, anchors, fmha, cache, inputs, metadata, mask):
            received.append(metadata)
            return hidden + 1

    namespace = {
        "HybridAttentionType": SimpleNamespace(LINEAR="linear"),
        "get_primary_attention_inputs": lambda *args: primary,
        "select_attention_inputs_for_layer": lambda *args: primary,
        "select_fmha_impl_for_layer": lambda *args: object(),
        "prepare_causal_conv1d_metadata": prepare,
        "KimiLinearMetadata": lambda conv, verify: (conv, verify),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    instance = SimpleNamespace(
        tp_size=1, tp_rank=0, kv_cache=None, num_blocks=0, layers=[Layer()]
    )
    hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    actual = namespace["_forward_layers"](instance, hidden, object(), object())
    torch.testing.assert_close(actual, hidden + 1)
    assert received == [(prepare.return_value if needs_conv else None, verify)]
    if needs_conv:
        prepare.assert_called_once_with(
            query_start_loc=primary.cu_seqlens_device, device=hidden.device
        )
    else:
        prepare.assert_not_called()
