"""Compare TP4 reductions for the same Qwen3.5-397B-A17B hidden row."""

import argparse
import os

import torch

from rtp_llm.config.collective_config import CollectiveConfig
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.ops import NcclCommConfig, ParallelismConfig


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 4
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    parallel = ParallelismConfig()
    parallel.world_size = parallel.local_world_size = parallel.tp_size = 4
    parallel.ffn_tp_size = 4
    parallel.world_rank = parallel.local_rank = parallel.tp_rank = rank
    collective.init_distributed_environment(
        parallel,
        NcclCommConfig(nccl_ip=os.environ["MASTER_ADDR"]),
        int(os.environ["MASTER_PORT"]),
        collective_config=CollectiveConfig(batch_invariant=not args.legacy),
    )
    mismatches = []
    try:
        for seed in (319, 513):
            generator = torch.Generator(device=device).manual_seed(seed + rank)
            x = torch.randn((1024, 4096), device=device, generator=generator).bfloat16()
            expected = collective.all_reduce(x[:1], collective.Group.TP, inplace=False)
            assert expected.data_ptr() != x.data_ptr()
            expected = expected[0].view(torch.int16)
            for rows in (2, 4, 11, 64, 65, 193, 525, 1024):
                batch = x[:rows]
                for inputs, row in ((batch.clone(), 0), (batch.roll(-1, 0), rows - 1)):
                    actual = collective.all_reduce(inputs, collective.Group.TP)
                    assert actual.data_ptr() == inputs.data_ptr()
                    different = (
                        (expected != actual[row].view(torch.int16)).count_nonzero().item()
                    )
                    mismatches.append(different)
                    if rank == 0:
                        print(f"{seed=} {rows=} {row=} {different=}", flush=True)
    finally:
        collective.destroy_distributed_environment()
    if not args.legacy:
        assert not any(mismatches), f"TP rank {rank} changed the target row"


if __name__ == "__main__":
    main()
