"""Compare the bound Qwen4 CUDA MoE to logical-expert math across two EP ranks."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.new_models.qwen4_exp.expert_factory import Qwen4RoutedExperts
from rtp_llm.models_py.new_models.qwen4_exp.runtime_bindings import bind_moe
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig


def _tp_sum(x, group):
    # The test's two-rank NCCL world is exactly the serving TP group.
    dist.all_reduce(x)
    return x


@torch.inference_mode()
def _worker(rank, rendezvous, permutation):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        pc = ParallelismConfig()
        pc.world_size = pc.local_world_size = pc.tp_size = pc.ep_size = 2
        pc.world_rank = pc.local_rank = pc.tp_rank = pc.ep_rank = rank
        config = ModelConfig()
        config.model_type = "qwen4_exp"
        config.data_type = "bf16"
        config.activation_type = "SiGLU"
        config.num_layers = config.moe_k = 1
        config.attn_config.head_num = 1
        config.attn_config.size_per_head = 128
        config.hidden_size = config.moe_inter_size = config.inter_size = 128
        config.max_seq_len = 4
        config.expert_num = 4
        config.has_moe_norm = True
        moe = MoeConfig()
        moe.use_all_gather = True
        moe.moe_strategy = "no_quant_cpp"
        hw = HWKernelConfig()
        hw.enable_cuda_graph = hw.use_swizzleA = False
        device = torch.device("cuda", rank)
        eye = torch.eye(128, dtype=torch.bfloat16, device=device)
        x = eye[:4]
        # Routes 0..3 produce four distinct outputs. The diagonal gate logits
        # make the selected probability round to one in the BF16 result.
        reference = torch.nn.functional.silu(x) * x
        reference *= torch.arange(1, 5, device=device).view(-1, 1)
        for path, redundant in (("", 0), ("", 2), (permutation, 0)):
            config.phy2log_path = path
            config.eplb_config.redundant_expert = redundant
            experts = Qwen4RoutedExperts(
                4,
                1,
                128,
                128,
                1,
                0,
                2,
                rank,
                torch.bfloat16,
                config,
                pc,
                moe,
                None,
                0,
            ).to(device)
            for e in range(4):
                experts.load_weights(
                    {
                        f"{e}.gate_proj.weight": eye,
                        f"{e}.up_proj.weight": eye,
                        f"{e}.down_proj.weight": (1 + e) * eye,
                    }
                )
            experts.process_weights_after_loading()
            experts.runtime.router.tp_all_reduce = _tp_sum
            gate = torch.nn.Linear(
                128, 4, bias=False, dtype=torch.bfloat16, device=device
            )
            gate.weight.copy_(16 * x)
            owner = SimpleNamespace(
                experts=experts, gate=gate, shared_expert=None, shared_expert_gate=None
            )
            runtime = bind_moe(owner, config, pc, moe_config=moe, hw_kernel_config=hw)
            torch.testing.assert_close(runtime(x), reference, atol=0.005, rtol=0.01)
            assert (runtime.fused_moe is experts.runtime) == (
                not path and not redundant
            )
    finally:
        dist.destroy_process_group()


class Qwen4StaticRoutingTest(unittest.TestCase):
    def test_identity_redundancy_and_permutation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "placement.json"
            path.write_text("[[3, 2, 1, 0]]")
            torch.multiprocessing.spawn(
                _worker, args=((Path(folder) / "init").as_uri(), str(path)), nprocs=2
            )


if __name__ == "__main__":
    unittest.main()
