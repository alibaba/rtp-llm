"""Independent arithmetic references for the INT8 communication codec tests."""

import torch


def reference_quantize(x, group_size, scale_dtype=torch.bfloat16):
    groups = x.double().reshape(x.shape[0], -1, group_size)
    maximum = groups.abs().amax(-1).float().clamp_min(1e-10)
    raw_scale = maximum / 127
    if scale_dtype == torch.float16:
        raw_scale = raw_scale.clamp(2**-24, 65504)
    scales = raw_scale.to(scale_dtype)
    # Preserve the original BF16 reciprocal and nearest-even integer rounding.
    inverse = (1 / scales.float()).to(torch.bfloat16).double().unsqueeze(-1)
    codes = (groups.abs() * inverse).round().clamp(0, 127) * groups.sign()
    return codes.to(torch.int8).view_as(x), scales


def reference_dequantize(q, scales, group_size):
    groups = q.float().reshape(*scales.shape, group_size)
    return (groups * scales.float().unsqueeze(-1)).to(torch.bfloat16).view_as(q)


def reference_reduce(inputs, scale_dtype):
    """Round each decoded source and each rank-ordered addition to BF16."""
    codes, scales, result = [], [], None
    for x in inputs:
        q, s = reference_quantize(x, 16, scale_dtype)
        decoded = reference_dequantize(q, s, 16)
        result = (
            decoded
            if result is None
            else (result.float() + decoded.float()).to(torch.bfloat16)
        )
        codes.append(q)
        scales.append(s)
    return torch.stack(codes), torch.stack(scales), result


def rank_inputs(m, k, factor, world_size=2):
    """Distinct reproducible rank payloads, independent of GPU RNG and scheduling."""
    inputs = []
    for rank in range(world_size):
        generator = torch.Generator(device="cpu").manual_seed(20261016 + rank)
        x = torch.randn((m, k), generator=generator).to(torch.bfloat16)
        x = (x * factor).to(torch.bfloat16)
        x.view(-1)[:16].zero_()
        x.view(-1)[16:32] = (-1.0) ** rank
        inputs.append(x)
    return inputs


def assert_bits_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(
        actual.cpu().view(torch.int16 if actual.element_size() == 2 else torch.int8),
        expected.cpu().view(
            torch.int16 if expected.element_size() == 2 else torch.int8
        ),
        rtol=0,
        atol=0,
        check_dtype=True,
    )
