"""Distributed staging checks for FP8xFP4 MegaMoE fused shared experts.

Run with the task Python when a four-GPU diagnostic is needed::

    CUDA_VISIBLE_DEVICES=4,5,6,7 /opt/conda310/bin/python -m torch.distributed.run \
      --nproc_per_node=4 \
      rtp_llm/models_py/modules/glm5_mega_moe/test_ut_distributed_se.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.glm5_mega_moe.shared_fp8_scale import (
    shared_fp8_scale_row_indices,
)


def test_shared_scale_staging() -> None:
    from rtp_llm.models_py.modules.glm5_mega_moe.shared_fp8_scale import (
        stage_shared_fp8_input_scales,
    )

    rank = dist.get_rank()
    tokens = 137 + rank
    packed_k = 48
    block_m = 32
    source = torch.arange(tokens * packed_k, dtype=torch.int32, device="cuda").reshape(
        tokens, packed_k
    )
    destination = torch.empty_strided(
        (1024, packed_k),
        (1, 1024),
        dtype=torch.int32,
        device="cuda",
    )
    stage_shared_fp8_input_scales(source, destination, tokens, block_m)
    rows = shared_fp8_scale_row_indices(tokens, block_m, source.device)
    torch.testing.assert_close(destination[rows], source)
    dist.barrier()


def test_se_buffer_contract() -> None:
    import deep_gemm

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group=dist.group.WORLD,
        num_experts=256,
        num_max_tokens_per_rank=192,
        num_topk=8,
        hidden=6144,
        intermediate_hidden=2048,
        num_shared_experts=1,
        mma_type="fp8xfp4",
        activation="swiglu",
    )
    assert buffer.num_shared_experts == 1
    assert buffer.shared_l1_acts_sf is not None
    assert buffer.shared_l2_acts is not None
    assert buffer.shared_l2_acts_sf is not None
    dist.barrier()


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        test_shared_scale_staging()
        test_se_buffer_contract()
        if dist.get_rank() == 0:
            print("MegaMoE SE distributed staging checks passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
