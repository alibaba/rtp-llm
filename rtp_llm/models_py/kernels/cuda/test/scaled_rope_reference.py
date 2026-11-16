"""Independent float64 reference; no production RoPE helpers or native ops."""

import math

import torch


def reference_rope(
    x,
    positions,
    lengths,
    *,
    rotary_dim,
    base,
    scale,
    original_max_position,
    llama3,
    low_freq_factor=1.0,
    high_freq_factor=4.0
):
    rows = []
    for token in range(x.shape[0]):
        length = int(lengths[token])
        theta = base
        if not llama3 and length > original_max_position:
            theta *= (scale * length / original_max_position - (scale - 1)) ** (
                rotary_dim / (rotary_dim - 2)
            )
        frequencies = []
        for index in range(rotary_dim // 2):
            frequency = theta ** (-2 * index / rotary_dim)
            if llama3:
                wave = 2 * math.pi / frequency
                if wave > original_max_position / low_freq_factor:
                    frequency /= scale
                elif wave >= original_max_position / high_freq_factor:
                    smooth = (original_max_position / wave - low_freq_factor) / (
                        high_freq_factor - low_freq_factor
                    )
                    frequency *= smooth + (1 - smooth) / scale
            frequencies.append(frequency)
        angles = float(positions[token]) * torch.tensor(
            frequencies, dtype=torch.float64, device=x.device
        )
        a = x[token, :, : rotary_dim // 2].double()
        b = x[token, :, rotary_dim // 2 : rotary_dim].double()
        rows.append(
            torch.cat(
                (
                    a * angles.cos() - b * angles.sin(),
                    b * angles.cos() + a * angles.sin(),
                    x[token, :, rotary_dim:].double(),
                ),
                dim=-1,
            )
        )
    return torch.stack(rows).to(x.dtype) if rows else x.clone()
