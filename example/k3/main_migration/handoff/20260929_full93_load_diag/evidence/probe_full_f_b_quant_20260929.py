"""Isolate the layer-0 K3 FP8 weight conversion without starting a model."""

import argparse
import json
import time

import torch
import deep_gemm.utils.layout
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exact-chunk-loop", action="store_true")
    args = parser.parse_args()
    key = "language_model.model.layers.0.self_attn.f_b_proj.weight"
    with safe_open(args.shard, framework="pt", device="cpu") as archive:
        source = archive.get_tensor(key)
    assert source.shape == (12288, 128) and source.dtype == torch.bfloat16
    x = source.T.contiguous().cuda()
    torch.cuda.synchronize()
    start = time.monotonic()
    rows, columns = x.shape
    if args.exact_chunk_loop:
        weight = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty(
            ((rows + 127) // 128, (columns + 127) // 128),
            dtype=torch.float32,
            device=x.device,
        )
        for begin in range(0, rows, 1024):
            end = min(begin + 1024, rows)
            tile = x[begin:end]
            padded = torch.zeros(
                (((tile.shape[0] + 127) // 128) * 128, ((columns + 127) // 128) * 128),
                dtype=x.dtype,
                device=x.device,
            )
            padded[: tile.shape[0], :columns] = tile
            blocks = padded.view(-1, 128, padded.size(1) // 128, 128)
            amax = blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
            block_scale = torch.pow(2.0, torch.ceil(torch.log2((amax / 448.0).abs())))
            block_weight = (blocks * (1.0 / block_scale)).clamp(-448, 448).to(
                torch.float8_e4m3fn
            ).view_as(padded)[: tile.shape[0], :columns].contiguous()
            weight[begin:end].copy_(block_weight)
            scale[begin // 128 : (end + 127) // 128].copy_(
                block_scale.view(blocks.size(0), blocks.size(2))
            )
    else:
        padded = torch.zeros(
            (((rows + 127) // 128) * 128, ((columns + 127) // 128) * 128),
            dtype=x.dtype,
            device=x.device,
        )
        padded[:rows, :columns] = x
        blocks = padded.view(-1, 128, padded.size(1) // 128, 128)
        amax = blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        scale = torch.pow(2.0, torch.ceil(torch.log2((amax / 448.0).abs())))
        weight = (blocks * (1.0 / scale)).clamp(-448, 448).to(
            torch.float8_e4m3fn
        ).view_as(padded)[:rows, :columns].contiguous()
        scale = scale.view(blocks.size(0), blocks.size(2))
    scale = scale.index_select(-2, torch.arange(rows, device=x.device) // 128)
    packed = deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(
        scale
    )
    torch.cuda.synchronize()
    result = {
        "source": args.shard,
        "source_shape": list(source.shape),
        "quant_shape": list(weight.shape),
        "packed_shape": list(packed.shape),
        "packed_first": int(packed[0, 0].item()),
        "elapsed_s_diagnostic_only": round(time.monotonic() - start, 3),
    }
    with open(args.output, "w") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
