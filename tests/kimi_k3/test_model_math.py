"""CPU checks; these do not replace CUDA, distributed, or full-model validation."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Residual = load_source(
    "k3_residual", "rtp_llm/models_py/modules/kimi_k3/residual.py"
).KimiK3AttentionResidual
checkpoint = load_source("k3_checkpoint", "rtp_llm/utils/kimi_k3_mtp_checkpoint.py")


@pytest.mark.parametrize("rows", [1, 2, 3, 7, 8, 9])
@pytest.mark.parametrize("blocks", [0, 1, 4])
def test_attention_residual_against_scalar_reference(rows, blocks):
    generator = torch.Generator().manual_seed(731)
    prefix = torch.randn(rows, 16, generator=generator)
    bank = torch.randn(rows, blocks + 2, 16, generator=generator)
    norm = torch.randn(16, generator=generator)
    projection = torch.randn(1, 16, generator=generator)
    module = Residual(norm, projection, 1e-6)
    actual = module(prefix, bank, num_blocks=blocks)
    expected = []
    for row in range(rows):
        candidates = list(bank[row, :blocks].double().unbind()) + [prefix[row].double()]
        scores = torch.stack(
            [
                (
                    (value / (value.square().mean() + 1e-6).sqrt())
                    * norm.double()
                    * projection.double().flatten()
                ).sum()
                for value in candidates
            ]
        )
        probabilities = scores.softmax(0)
        expected.append(
            sum(weight * value for weight, value in zip(probabilities, candidates))
        )
    torch.testing.assert_close(
        actual.double(), torch.stack(expected), atol=3e-6, rtol=3e-6
    )
    # Unused capacity must not influence results when the graph bucket grows.
    bank[:, blocks:] = float("nan")
    torch.testing.assert_close(module(prefix, bank, num_blocks=blocks), actual)


def test_residual_bank_commit_and_output_norm():
    prefix = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    delta = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    bank = torch.zeros(1, 3, 4)
    module = Residual(torch.ones(4), torch.zeros(4), 1e-6)
    result = module(
        prefix,
        bank,
        delta=delta,
        block_write_idx=1,
        num_blocks=2,
        output_norm_weight=torch.ones(4),
        output_norm_eps=1e-6,
    )
    torch.testing.assert_close(bank[:, 1], torch.full((1, 4), 5.0))
    torch.testing.assert_close(result, torch.ones_like(result))


@pytest.mark.parametrize("source", [None, 0, -1, True, "93"])
def test_mtp_rejects_invalid_source_layer(source):
    with pytest.raises(ValueError):
        checkpoint.mtp_source_layer(
            {"num_hidden_layers": source, "num_nextn_predict_layers": 1}
        )


def test_mtp_source_is_nextn_layer_not_last_target_layer():
    assert (
        checkpoint.mtp_source_layer(
            {"num_hidden_layers": 93, "num_nextn_predict_layers": 1}
        )
        == 93
    )
    with pytest.raises(ValueError):
        checkpoint.mtp_source_layer(
            {"num_hidden_layers": 93, "num_nextn_predict_layers": 3}
        )


layout = load_source("k3_layout", "rtp_llm/models/kimi_k3/weight_layout.py")


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_kda_fused_projection_matches_unsharded_components(tp):
    generator = torch.Generator().manual_seed(23)
    inputs = torch.randn(9, 16, generator=generator)
    # Eight heads, four dimensions; F_a rank is deliberately not divisible by TP.
    components = [
        torch.randn(16, width, generator=generator) for width in [32, 32, 32, 32, 7, 8]
    ]
    packed = torch.cat(components, dim=1)
    expected = [inputs @ component for component in components]
    local_outputs = []
    for rank in range(tp):
        local = layout.split_kda_input(packed, 8, 4, tp, rank)
        local_outputs.append(
            (inputs @ local).split([32 // tp] * 4 + [7, 8 // tp], dim=1)
        )
    for index in range(6):
        if index == 4:
            for output in local_outputs:
                torch.testing.assert_close(output[index], expected[index])
        else:
            torch.testing.assert_close(
                torch.cat([output[index] for output in local_outputs], dim=1),
                expected[index],
            )


def test_kda_rejects_split_through_a_head():
    with pytest.raises(ValueError):
        layout.split_kda_input(torch.empty(16, 4 * 6 * 4 + 7 + 6), 6, 4, 4, 0)
