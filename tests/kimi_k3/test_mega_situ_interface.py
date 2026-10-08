import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[2] / (
    "rtp_llm/models_py/modules/factory/fused_moe/utils/mega_moe/activation.py"
)
spec = importlib.util.spec_from_file_location("mega_situ_activation", path)
activation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(activation)


def main_kernel(*, situ_beta, situ_linear_beta):
    return situ_beta, situ_linear_beta


def reference_kernel(*, activation_alpha, activation_beta):
    return activation_alpha, activation_beta


@pytest.mark.parametrize("kernel", [main_kernel, reference_kernel])
def test_gate_and_up_scales_are_not_swapped(kernel):
    assert kernel(**activation.situ_kwargs(kernel, 3.0, 7.0)) == (3.0, 7.0)


def test_unsaturated_up_only_supported_by_reference_interface():
    assert reference_kernel(
        **activation.situ_kwargs(reference_kernel, 3.0, None)
    ) == (3.0, 0.0)
    with pytest.raises(RuntimeError, match="saturated up"):
        activation.situ_kwargs(main_kernel, 3.0, None)


def test_unknown_backend_does_not_silently_drop_activation():
    with pytest.raises(RuntimeError, match="SiTU support"):
        activation.situ_kwargs(lambda **kwargs: None, 3.0, 7.0)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scales_rejected(value):
    with pytest.raises(ValueError):
        activation.situ_kwargs(main_kernel, value, 7.0)
    with pytest.raises(ValueError):
        activation.situ_kwargs(main_kernel, 3.0, value)
