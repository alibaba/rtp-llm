from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.kernels.cuda.sm120.grouped_decode import op as grouped_decode


def test_bound_grouped_adapter_preserves_output_and_rejects_psum():
    device = torch.device("cuda", 1)
    data = SimpleNamespace(device=device)
    packed_scale = torch.empty(1, dtype=torch.int32)
    activations = weights = (data, packed_scale)
    layout = torch.empty(1, dtype=torch.int32)
    output = torch.ones(1, dtype=torch.bfloat16)
    native = Mock(side_effect=lambda _a, _b, out, _layout, **_: out.fill_(2))
    binding = Mock(m_grouped_fp8_fp4_gemm_nt_contiguous=native)

    grouped_decode._grouped_op_for_device.cache_clear()
    try:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=1),
            patch.object(
                torch.cuda,
                "get_device_capability",
                side_effect=lambda index: (12, 0) if index == 1 else (12, 1),
            ),
            patch.object(grouped_decode, "_binding", return_value=binding),
        ):
            assert grouped_decode.is_available(torch.device("cuda"))
            op = grouped_decode.grouped_op(device)
            assert grouped_decode.grouped_op(torch.device("cuda")) is op
            assert not grouped_decode.is_available(torch.device("cuda", 0))
            binding.warmup_device.assert_called_once_with(1)
    finally:
        grouped_decode._grouped_op_for_device.cache_clear()

    assert op(activations, weights, output=output, grouped_layout=layout) is None
    assert native.call_args.args[2] is output
    assert output.item() == 2
    assert native.call_args.kwargs["recipe"] is None
    assert native.call_args.kwargs["recipe_a"] is None
    assert native.call_args.kwargs["recipe_b"] is None
    assert native.call_args.kwargs["compiled_dims"] == "nk"
    assert native.call_args.kwargs["disable_ue8m0_cast"] is False

    recipes = dict(recipe_a=(1, 128), recipe_b=(1, 32), disable_ue8m0_cast=True)
    op(activations, weights, output, layout, **recipes)
    assert all(native.call_args.kwargs[key] == value for key, value in recipes.items())
    native.reset_mock()

    for options in ({"use_psum_layout": True}, {"expected_m_for_psum_layout": 16}):
        with TestCase().assertRaisesRegex(ValueError, "psum"):
            op(activations, weights, output, layout, **options)
    with TestCase().assertRaisesRegex(RuntimeError, "prepacked int32 scales"):
        op((data, torch.ones(1)), weights, output, layout)
    native.assert_not_called()


def test_grouped_decode_accumulates_into_caller_output():
    device = torch.device("cuda")
    op = grouped_decode.grouped_op(device)
    m, n, k = 128, 128, 128
    activation = torch.ones((m, k), device=device).to(torch.float8_e4m3fn)
    # Packed UE8M0 scales of 1.0 in the TMA-aligned MN-major layout.
    activation_scales = torch.empty_strided(
        (m, 1), (1, m), device=device, dtype=torch.int32
    ).fill_(0x7F)
    # Two E2M1 values of +1.0 in each byte.
    weight = torch.full((1, n, k // 2), 0x22, device=device, dtype=torch.int8)
    weight_scales = torch.empty_strided(
        (1, n, 1), (n, 1, n), device=device, dtype=torch.int32
    ).fill_(0x7F7F7F7F)
    output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    groups = torch.zeros((m,), device=device, dtype=torch.int32)

    result = op(
        (activation, activation_scales),
        (weight, weight_scales),
        output,
        groups,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )

    assert result is None
    assert torch.all(output == k)

    # Retain coverage of the native provider's optional FP32 scale conversion.
    output.zero_()
    native = grouped_decode._binding().m_grouped_fp8_fp4_gemm_nt_contiguous
    result = native(
        (activation, torch.ones((m, 1), device=device)),
        (weight, torch.ones((1, n, k // 32), device=device)),
        output,
        groups,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )
    assert result is None
    assert torch.all(output == k)


if __name__ == "__main__":
    test_bound_grouped_adapter_preserves_output_and_rejects_psum()
    test_grouped_decode_accumulates_into_caller_output()
    print("PASS test_grouped_decode_accumulates_into_caller_output")
