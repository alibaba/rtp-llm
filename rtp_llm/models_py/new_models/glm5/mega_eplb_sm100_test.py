"""No-mock Blackwell launch and CUDA Graph replay across a GLM5 FP4 EPLB swap."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rtp_llm.config.mega_moe import MegaMoeExecutionPolicy
from rtp_llm.models_py.expert_placement import ExpertPlacement
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
    MegaMoeRouter,
)
from rtp_llm.models_py.new_models.glm5.language import (
    _PackedGlm5Experts,
    _PackedGlm5MoE,
)
from rtp_llm.ops import RoleType


class PackedMegaEplbSM100Test(unittest.TestCase):
    @torch.inference_mode()
    def test_graph_replays_live_kernel_weights_and_cmp_routes(self):
        self.assertEqual(torch.cuda.get_device_capability()[0], 10)
        with tempfile.TemporaryDirectory() as folder:
            dist.init_process_group(
                "nccl",
                init_method="file://" + os.path.join(folder, "rendezvous"),
                rank=0,
                world_size=1,
                device_id=torch.device("cuda", 0),
            )
            try:
                self.run_swap()
            finally:
                dist.destroy_process_group()

    def run_swap(self):
        e, k, dim, inter = 16, 4, 4096, 2048
        pc = SimpleNamespace(
            tp_size=1,
            ep_size=1,
            world_size=1,
            local_world_size=1,
            role_type=RoleType.DECODE,
        )
        mc = SimpleNamespace(
            activation_type="SiGLU",
            model_type="glm_5",
            num_layers=1,
            max_seq_len=32,
            gen_num_per_cycle=0,
            eplb_config=SimpleNamespace(enable_eplb=True, redundant_expert=0),
        )
        strategy = SimpleNamespace(moe_strategy="mega_moe")
        leaf = _PackedGlm5Experts(
            num_experts=e,
            top_k=k,
            hidden_size=dim,
            moe_intermediate_size=inter,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            params_dtype=torch.bfloat16,
            model_config=mc,
            parallelism_config=pc,
            moe_config=strategy,
            quant_config=None,
            layer_idx=0,
        )
        swap = tuple(range(e // 2, e)) + tuple(range(e // 2))
        placement = ExpertPlacement(e, 1, 0, swap)

        def prepare(owner):
            # Experts in the two halves have different FP4 nibbles AND scales.
            for expert in range(e):
                hot = int(expert >= e // 2)
                for j, (proj, rows, cols) in enumerate(
                    (
                        ("gate_proj", inter, dim),
                        ("up_proj", inter, dim),
                        ("down_proj", dim, inter),
                    )
                ):
                    key = f"{expert}.{proj}"
                    owner.load_weights(
                        {
                            key
                            + ".weight": torch.full(
                                (rows, cols // 2), 0x11 * (hot + 1), dtype=torch.int8
                            ),
                            key
                            + ".scale": torch.full(
                                (rows, cols // 32), 117 + hot - j, dtype=torch.uint8
                            ).view(torch.float8_e8m0fnu),
                        }
                    )
            owner.to("cuda")
            return owner.prepare_eplb_reload_weights()

        raw = prepare(leaf)
        cfg = SimpleNamespace(
            mega_moe_policy=MegaMoeExecutionPolicy(),
            moe_strategy="mega_moe",
            model_config=mc,
            parallelism_config=pc,
            moe_config=strategy,
            ep_size=1,
            ep_rank=0,
            tp_size=1,
            n_routed_experts=e,
            n_local_experts=e,
            local_expert_start=0,
            n_activated_experts=k,
            max_generate_batch_size=8,
            dim=dim,
            moe_inter_dim=inter,
            layer_id=0,
        )
        kernel = MegaMoeExecutor(cfg, None, dict(raw))
        leaf.fused_moe = FusedMoe(MegaMoeRouter(cfg, None), kernel, e, "mega_moe")
        block = _PackedGlm5MoE.__new__(_PackedGlm5MoE)
        torch.nn.Module.__init__(block)
        block.experts = leaf
        block.fake_balance_expert = block.shared_experts = None
        clone = block.clone_for_cuda_graph(share_mega_buf=True)
        x = torch.ones((2, dim), dtype=torch.bfloat16, device="cuda")
        ids = torch.arange(k, device="cuda").expand(2, -1).contiguous()
        ids[1] += e // 2
        weights = torch.full((2, k), 1 / k, device="cuda")
        _, _, buf_ids, buf_weights = kernel.prepacked_input_views(2)

        def graph_forward():
            kernel._pack_inputs(x, weights, ids, 2)
            return clone.forward_prepacked(x, buf_ids, buf_weights)

        baseline = graph_forward().clone()
        self.assertTrue(torch.isfinite(baseline).all())
        self.assertFalse(torch.allclose(baseline[0], baseline[1]))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = graph_forward()

        def check(swapped):
            # Raw dispatch reads physical slots. The captured dispatch reads
            # logical ids; it must still return the original logical output.
            physical = kernel(x, weights, ids).clone()
            torch.testing.assert_close(
                physical, baseline.flip(0) if swapped else baseline
            )
            torch.testing.assert_close(leaf(x, weights, ids), baseline)
            graph.replay()
            torch.testing.assert_close(y, baseline)

        staged = prepare(leaf.clone_empty_for_placement(placement))
        staged.update(leaf.eplb_kernel_weights(staged, prepare=True))
        phy = torch.tensor(swap, dtype=torch.int32)
        routing = leaf._routing_tensors(placement)
        check(False)
        leaf.apply_eplb_plan(phy, *routing, staged, defer_commit=True)
        check(True)
        leaf.finalize_eplb_install(commit=False)
        check(False)
        leaf.apply_eplb_plan(phy, *routing, staged)
        check(True)


if __name__ == "__main__":
    unittest.main()
