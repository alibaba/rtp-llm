"""Manual TP8 numerical regression for FP8 projection and collective fusion.

Run in the K3 CUDA13/SM10x container with eight free GPUs. The test uses the
actual KDA input/output shapes and compares complete BF16 results, including
the rank order and scale layout of the FP8 AllGather.
"""

from datetime import timedelta
import os

import torch
import torch.distributed as dist

from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

setup_cutlass_import_path()

from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.models_py.distributed.fp8_collective_projection import (
    Fp8CollectiveProjection,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)


def projection(rows, columns, device):
    weight = (torch.arange(rows * columns, device=device).reshape(rows, columns) % 97)
    weight = ((weight.float() - 48) / 256).to(torch.float8_e4m3fn)
    scales = torch.full(
        (columns // 512, rows), 0x7F7F7F7F, dtype=torch.int32, device=device
    ).T
    return CudaFp8DeepGEMMLinear(
        weight=weight,
        weight_scales=scales,
        quant_config=init_quant_config("FP8_PER_BLOCK"),
    )


def main():
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    try:
        input_proj = projection(6288, 7168, device)
        output_proj = projection(7168, 1536, device)
        fused = Fp8CollectiveProjection(
            dist.group.WORLD, device, max_m=65536, hidden_size=7168
        )
        for local_rows in (128, 8192):
            x = torch.arange(local_rows * 7168, device=device).reshape(local_rows, 7168)
            x = (((x % 503).float() - 251) / 32 + rank / 16).to(torch.bfloat16)
            gathered = torch.empty(
                (world * local_rows, 7168), dtype=x.dtype, device=device
            )
            dist.all_gather_into_tensor(gathered, x)
            expected = input_proj(gathered)
            actual = fused.all_gather_gemm(x, input_proj)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            source = torch.arange(
                world * local_rows * 1536, device=device
            ).reshape(world * local_rows, 1536)
            source = (((source % 71).float() - 35) / 16 + rank / 8).to(torch.bfloat16)
            values, scales = output_proj.quantize_input(source)
            partial = output_proj.forward_quantized(values, scales)
            expected_rs = torch.empty(
                (local_rows, 7168), dtype=torch.bfloat16, device=device
            )
            dist.reduce_scatter_tensor(expected_rs, partial)
            actual_rs = fused.gemm_reduce_scatter(
                values, scales, output_proj
            )
            torch.testing.assert_close(actual_rs, expected_rs, rtol=0.02, atol=0.04)
            if rank == 0:
                print(f"PASS TP8 local_rows={local_rows}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    assert os.environ.get("LOCAL_WORLD_SIZE") == "8"
    main()
