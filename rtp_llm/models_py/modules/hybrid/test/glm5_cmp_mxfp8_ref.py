import math

import torch
import torch.nn.functional as F


def _rope_hadamard(value, cos_sin, positions):
    value = value.reshape(value.size(0), -1, 128).float()
    trig = cos_sin[positions.long()]
    cos, sin = trig[:, :32, None], trig[:, 32:, None]
    cos, sin = cos.transpose(1, 2), sin.transpose(1, 2)
    real, imag = value[..., :64:2], value[..., 1:64:2]
    result = value.clone()
    result[..., :64:2] = ((real * cos).double() - imag.double() * sin.double()).float()
    result[..., 1:64:2] = ((imag * cos).double() + real.double() * sin.double()).float()
    result = result.bfloat16().float()
    stride = 1
    while stride < 128:
        paired = result.reshape(*result.shape[:-1], -1, 2, stride)
        left, right = paired[..., 0, :], paired[..., 1, :]
        result = torch.stack((left + right, left - right), dim=-2).reshape_as(result)
        stride *= 2
    return (result * (1 / math.sqrt(128))).bfloat16()


def _mxfp8(value):
    grouped = value.float().reshape(*value.shape[:-1], 4, 32)
    scale = torch.exp2(
        torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    quantized = (
        (grouped / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape_as(value)
    )
    exponent = (torch.log2(scale).to(torch.int32) + 127).to(torch.uint8)
    return quantized, exponent.contiguous().view(torch.int32).squeeze(-1)


def _assert_q(projected, trig, positions, gates, outputs):
    expected_q, expected_sf = _mxfp8(_rope_hadamard(projected, trig, positions))
    torch.testing.assert_close(
        outputs[0].view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(outputs[1], expected_sf, rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], (gates / 64).bfloat16(), rtol=0, atol=0)


def _assert_k(inputs, initial_cache, epsilon):
    (
        activation,
        activation_sf,
        weight,
        weight_sf,
        norm_weight,
        norm_bias,
        trig,
        positions,
        slots,
        cache,
    ) = inputs

    def dequant(value, sf):
        exponents = (
            (sf.unsqueeze(-1).to(torch.int64) >> (8 * torch.arange(4, device="cuda")))
            & 255
        ).reshape(value.size(0), -1)
        return value.float() * torch.exp2(exponents.float() - 127).repeat_interleave(
            128, -1
        )

    projected = (
        dequant(activation, activation_sf) @ dequant(weight, weight_sf).T
    ).bfloat16()
    normalized = F.layer_norm(projected, (128,), norm_weight, norm_bias, epsilon)
    q, sf = _mxfp8(_rope_hadamard(normalized, trig, positions))
    expected = initial_cache.clone().view(2, -1)
    for row, slot in enumerate(slots.cpu().tolist()):
        if 0 <= slot < 128:
            block, offset = divmod(slot, 64)
            expected[block, offset * 128 : (offset + 1) * 128] = (
                q[row].flatten().view(torch.uint8)
            )
            expected[block, 8192 + offset * 4 : 8192 + (offset + 1) * 4] = (
                sf[row].reshape(1).view(torch.uint8)
            )
    torch.testing.assert_close(cache.flatten(), expected.flatten(), rtol=0, atol=0)
