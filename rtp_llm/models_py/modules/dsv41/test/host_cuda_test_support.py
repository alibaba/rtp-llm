"""Real source imports and bounded CPU oracle for shared-host GPU tests."""

from pathlib import Path

import torch
from standalone_load import load_component


def load_host_modules():
    shared = load_component(
        "rtp_llm.model_loader.host_shared_weights",
        "model_loader/host_shared_weights.py",
    )
    load_component(
        "rtp_llm.models_py.modules.dsv41._engram_lookup_triton",
        "models_py/modules/dsv41/_engram_lookup_triton.py",
    )
    load_component(
        "rtp_llm.model_loader.host_shared_metrics",
        "model_loader/host_shared_metrics.py",
    )
    cuda = load_component(
        "rtp_llm.model_loader.host_shared_cuda", "model_loader/host_shared_cuda.py"
    )
    return shared, cuda


def initialized_slices(root, shared_module, rows=1024):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    slices = []
    for layer in (1, 14):
        weight = (
            (torch.arange(rows * 256, dtype=torch.int64, device="cpu") + layer) % 256
        ).to(torch.uint8)
        scales = torch.tensor(
            [0, 1, 64, 126, 127, 128, 192, 254], dtype=torch.uint8, device="cpu"
        ).repeat(rows)
        for kind, values, dim, dtype in (
            ("weight", weight, 256, "F8_E4M3"),
            ("scale", scales, 8, "F8_E8M0"),
        ):
            name = f"layers.{layer}.engram.embed.{kind}"
            path = root / name
            path.write_bytes(bytes(values.tolist()))
            slices.append(
                shared_module.SharedWeightSlice(
                    name, path, 0, values.numel(), (rows, dim), dtype
                )
            )
    return slices


def cpu_lookup_reference(shared, layer, indices, valid_mask=None):
    ids = indices.detach().to("cpu").reshape(-1).tolist()
    valid = (
        [True] * len(ids)
        if valid_mask is None
        else valid_mask.detach().to("cpu").reshape(-1).tolist()
    )
    prefix = f"layers.{layer}.engram.embed."
    weights, scales = shared.view(prefix + "weight"), shared.view(prefix + "scale")
    outputs = []
    try:
        for start in range(0, len(ids), 4096):
            selected = ids[start : start + 4096]
            live = valid[start : start + 4096]
            selected = [row if active else 0 for row, active in zip(selected, live)]
            raw_weight = bytearray(
                b"".join(weights[row * 256 : (row + 1) * 256] for row in selected)
            )
            raw_scale = bytearray(
                b"".join(scales[row * 8 : (row + 1) * 8] for row in selected)
            )
            values = (
                torch.frombuffer(raw_weight, dtype=torch.uint8)
                .view(torch.float8_e4m3fn)
                .float()
                .view(-1, 8, 32)
            )
            scale = (
                torch.frombuffer(raw_scale, dtype=torch.uint8)
                .view(torch.float8_e8m0fnu)
                .float()
                .view(-1, 8, 1)
            )
            output = (values * scale).flatten(1).to(torch.bfloat16)
            output[~torch.tensor(live, dtype=torch.bool)] = 0
            outputs.append(output)
    finally:
        weights.release()
        scales.release()
    flat = (
        torch.cat(outputs) if outputs else torch.empty((0, 256), dtype=torch.bfloat16)
    )
    return flat.reshape(tuple(indices.shape) + (256,))
