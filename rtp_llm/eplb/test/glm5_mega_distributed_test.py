import json
import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.eplb.runtime import EplbLayer, RuntimeExpertBalancer
from rtp_llm.eplb.test.test_support import make_config, make_layer
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import GLM5MegaMoE, GLM5MegaMoeCfg
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_se import GLM5MegaMoESE


def _run_mega(rank, rendezvous, fused_routing=False, gate_pack=False, shared=False):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("GLM5 FP4 MegaMoE integration requires SM100/SM103 GPUs")
    os.environ["GLM5_MEGA_MOE_JIT_WARMUP"] = "0"
    os.environ["GLM5_MEGA_MOE_SE_JIT_WARMUP"] = "0"
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=300),
    )
    try:
        dist.all_reduce(torch.zeros(1, device="cuda"))
        torch.manual_seed(7401)
        dim, intermediate, tokens = 512, 256, 64
        topk = 2 if gate_pack else 1
        model_class = GLM5MegaMoESE if shared else GLM5MegaMoE
        moe = model_class(
            GLM5MegaMoeCfg(
                layer_id=0,
                dim=dim,
                moe_inter_dim=intermediate,
                n_routed_experts=8,
                n_activated_experts=topk,
                swiglu_limit=0,
                ep_size=2,
                ep_rank=rank,
                n_local_experts=4,
                local_expert_start=rank * 4,
                local_expert_end=(rank + 1) * 4,
                max_tokens_per_rank=tokens,
            )
        )
        w1 = torch.randn(4, intermediate, dim, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(4, dim, intermediate, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn_like(w1)
        initial = make_layer(
            (0, 0, 1, 1, 2, 2, 3, 3), ep_rank=rank, ep_size=2, device="cuda"
        )
        initial_local = initial.physical_to_logical[rank * 4 : (rank + 1) * 4].cuda()
        moe.setup_weights_from_bf16(
            w1[initial_local] * 0.05,
            w2[initial_local] * 0.05,
            w3[initial_local] * 0.05,
        )
        shared_tensors = ()
        shared_bytes = []
        routed_reference = None
        if shared:
            sw1 = torch.randn(2 * intermediate, dim, device="cuda").to(
                torch.float8_e4m3fn
            )
            sw2 = torch.randn(dim, intermediate, device="cuda").to(torch.float8_e4m3fn)
            ss1 = torch.full(
                (2 * intermediate // 128, dim // 128),
                2**-5,
                device="cuda",
                dtype=torch.float8_e8m0fnu,
            )
            ss2 = torch.full(
                (dim // 128, intermediate // 128),
                2**-5,
                device="cuda",
                dtype=torch.float8_e8m0fnu,
            )
            moe.setup_shared_expert_from_fp8(sw1, ss1, sw2, ss2)
            shared_tensors = (
                moe._shared_l1_w,
                moe._shared_l1_sf,
                moe._shared_l2_w,
                moe._shared_l2_sf,
            )
            shared_bytes = [
                t.reshape(-1).contiguous().view(torch.uint8).clone()
                for t in shared_tensors
            ]
            routed_reference = GLM5MegaMoE(moe.cfg)
            routed_reference.setup_weights_from_bf16(
                w1[initial_local] * 0.05,
                w2[initial_local] * 0.05,
                w3[initial_local] * 0.05,
            )
        tensors = (moe._mega_l1_w, moe._mega_l1_sf, moe._mega_l2_w, moe._mega_l2_sf)
        plan = EplbLayer(
            0, initial.logical_to_physical, initial.logical_count, tensors, rank
        )
        reference_weights = []
        for tensor in tensors:
            payload = tensor.contiguous().view(torch.uint8).reshape(4, -1)
            gathered = [torch.empty_like(payload) for _ in range(2)]
            dist.all_gather(gathered, payload)
            reference_weights.append(torch.cat(gathered)[::2].clone())
        stable_tensors = (
            *tensors,
            *shared_tensors,
            plan.logical_to_physical,
            plan.logical_count,
        )
        pointers = [tensor.data_ptr() for tensor in stable_tensors]
        runtime = RuntimeExpertBalancer(
            [plan],
            make_config(eplb_update_time=1, eplb_stats_window_size=1),
            rank,
            2,
            "glm5-mega-test",
        )
        counts = torch.zeros(1, 4, device="cuda", dtype=torch.int32)
        moe.expert_token_counts = counts[0]
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        router_weights = torch.full((tokens, topk), 1.0 / topk, device="cuda")
        mixed_ids = (
            torch.arange(tokens, device="cuda")[:, None]
            + torch.arange(topk, device="cuda")[None, :]
        ) % 4
        ids = mixed_ids.clone()
        dist.barrier()
        torch.cuda.synchronize()
        mixed_reference = moe(x, router_weights, plan.map_ids(ids)).clone()
        ids.copy_(torch.arange(topk, device="cuda").expand(tokens, -1))
        hot_reference = moe(x, router_weights, plan.map_ids(ids)).clone()
        if shared:
            gate, up = (x.float() @ (sw1.float() * 2**-5).T).chunk(2, dim=-1)
            shared_reference = (torch.nn.functional.silu(gate) * up) @ (
                sw2.float() * 2**-5
            ).T
            assert shared_reference.abs().mean().item() > 1e-3
            for logical_ids, expected in (
                (mixed_ids, mixed_reference),
                (ids, hot_reference),
            ):
                routed = routed_reference(
                    x, router_weights, plan.map_ids(logical_ids)
                ).clone()
                reference = routed.float() + shared_reference
                relative_error = (expected.float() - reference).abs().mean() / (
                    reference.abs().mean() + 1e-6
                )
                assert relative_error.item() < 0.05, relative_error.item()
        logits = torch.full((tokens, 4), -5.0, device="cuda", dtype=torch.bfloat16)
        logits.scatter_(1, ids, 5.0)
        bias = torch.zeros(4, device="cuda")
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()

        def execute():
            if gate_pack:
                return moe._forward_impl(
                    x,
                    None,
                    None,
                    inputs_prepacked=False,
                    gate_logits=logits,
                    gate_bias=bias,
                    route_scale=1.0,
                    eplb=plan,
                )
            if fused_routing:
                return moe(x, router_weights, ids, eplb=plan)
            return moe(x, router_weights, plan.map_ids(ids))

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                execute()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = execute()
        counts.zero_()
        assert torch.count_nonzero(counts).item() == 0
        assert runtime.groups[0].updates == 0

        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, hot_reference, rtol=0.02, atol=0.02)
        total = counts.sum().clone()
        dist.all_reduce(total)
        assert total.item() == 2 * tokens * topk
        assert runtime.step(counts)
        counts.zero_()
        assert plan.logical_count[0].item() == (3 if gate_pack else 5)
        assert not torch.equal(
            runtime.groups[0].physical_to_logical[0], initial.physical_to_logical
        )
        assert torch.count_nonzero(counts).item() == 0
        layout = runtime.groups[0].physical_to_logical[0]
        local_logical = layout[rank * 4 : (rank + 1) * 4]
        if rank == 1:
            assert (local_logical == 0).any().item()
        for tensor, reference in zip(tensors, reference_weights):
            expected = reference.index_select(0, local_logical.to("cuda"))
            assert torch.equal(
                tensor.contiguous().view(torch.uint8).reshape(4, -1), expected
            ), "Transformed FP4 weight/scale bytes do not match logical experts"
        assert pointers == [tensor.data_ptr() for tensor in stable_tensors]
        for tensor, before in zip(shared_tensors, shared_bytes):
            assert torch.equal(
                tensor.reshape(-1).contiguous().view(torch.uint8), before
            ), "Shared weights/scales must not be shuffled as routed experts"
        layouts = [torch.empty_like(layout) for _ in range(2)]
        dist.all_gather(layouts, layout)
        assert torch.equal(layouts[0], layouts[1])
        dist.barrier()
        torch.cuda.synchronize()

        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, hot_reference, rtol=0.02, atol=0.02)
        hot_error = (output.float() - hot_reference.float()).abs().max().item()
        total = counts.sum().clone()
        active = torch.count_nonzero(counts)
        dist.all_reduce(total)
        dist.all_reduce(active)
        assert total.item() == 2 * tokens * topk
        assert active.item() == (6 if gate_pack else 5)
        counts.zero_()
        ids.copy_(mixed_ids)
        logits.fill_(-5.0).scatter_(1, ids, 5.0)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, mixed_reference, rtol=0.02, atol=0.02)
        assert torch.isfinite(output).all().item()
        assert torch.count_nonzero(output).item() > 0
        print(
            json.dumps(
                {
                    "rank": rank,
                    "fused_routing": fused_routing,
                    "gate_pack": gate_pack,
                    "shared": shared,
                    "physical_to_logical": layout.tolist(),
                    "weight_scale_bytes_equal": True,
                    "stable_pointers": True,
                    "hot_max_abs_error": hot_error,
                    "mixed_max_abs_error": (output.float() - mixed_reference.float())
                    .abs()
                    .max()
                    .item(),
                    "graph_replay_after_balance": True,
                }
            ),
            flush=True,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_legacy_counts(rank, rendezvous):
    from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fused import GLM5MegaMoEFused

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    graph = None
    try:
        dist.all_reduce(torch.zeros(1, device="cuda"))
        moe = object.__new__(GLM5MegaMoEFused)
        torch.nn.Module.__init__(moe)
        moe.cfg = SimpleNamespace(
            ep_size=2,
            local_expert_start=rank * 4,
            local_expert_end=(rank + 1) * 4,
        )
        moe._mega_group = dist.group.WORLD
        moe._eplb_global_counts = torch.zeros(8, dtype=torch.int32, device="cuda")
        source_ids = (
            [[0, 4], [4, -1], [7, 0]] if rank == 0 else [[4, 1], [7, 4], [-1, 4]]
        )
        moe._mega_buf = SimpleNamespace(
            topk_idx=torch.tensor(
                source_ids + [[2, 2]], device="cuda", dtype=torch.int64
            )
        )
        counts = torch.zeros(4, device="cuda", dtype=torch.int32)
        moe.expert_token_counts = counts
        expected = torch.tensor(
            [2, 1, 0, 0] if rank == 0 else [5, 0, 0, 2],
            device="cuda",
            dtype=torch.int32,
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            moe._record_expert_load(3)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(counts, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            moe._record_expert_load(3)
        counts.zero_()
        graph.replay()
        graph.replay()
        torch.testing.assert_close(counts, 2 * expected, rtol=0, atol=0)
        moe._mega_buf.topk_idx[:3].fill_(6)
        counts.zero_()
        graph.replay()
        expected.zero_()
        if rank == 1:
            expected[2] = 12
        torch.testing.assert_close(counts, expected, rtol=0, atol=0)
        moe._record_expert_load(0)
        torch.testing.assert_close(counts, expected, rtol=0, atol=0)
        moe.expert_token_counts = None
        moe._eplb_global_counts = None
        moe._record_expert_load(3)
        torch.cuda.synchronize()
        print(f"legacy fused counts and graph replay passed on rank {rank}", flush=True)
    finally:
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


class Glm5MegaDistributedTest(unittest.TestCase):
    def test_shared_expert_shuffle_and_graph_replay(self):
        for gate_pack in (False, True):
            with self.subTest(gate_pack=gate_pack):
                with tempfile.TemporaryDirectory(prefix="eplb-glm5-se-") as directory:
                    mp.spawn(
                        _run_mega,
                        args=(
                            os.path.join(directory, "rendezvous"),
                            True,
                            gate_pack,
                            True,
                        ),
                        nprocs=2,
                        join=True,
                    )

    def test_sigmoid_gate_pack_shuffle_and_graph_replay(self):
        with tempfile.TemporaryDirectory(prefix="eplb-glm5-gate-pack-") as directory:
            mp.spawn(
                _run_mega,
                args=(os.path.join(directory, "rendezvous"), False, True),
                nprocs=2,
                join=True,
            )

    def test_legacy_fused_destination_counts_and_graph_replay(self):
        with tempfile.TemporaryDirectory(prefix="eplb-glm5-counts-") as directory:
            mp.spawn(
                _run_legacy_counts,
                args=(os.path.join(directory, "rendezvous"),),
                nprocs=2,
                join=True,
            )

    def test_fused_pack_shuffle_and_graph_replay(self):
        with tempfile.TemporaryDirectory(prefix="eplb-glm5-fused-") as directory:
            mp.spawn(
                _run_mega,
                args=(os.path.join(directory, "rendezvous"), True),
                nprocs=2,
                join=True,
            )

    def test_fp4_kernel_shuffle_and_graph_replay(self):
        if torch.cuda.device_count() < 2:
            raise RuntimeError("GLM5 MegaMoE integration requires two visible GPUs")
        with tempfile.TemporaryDirectory(prefix="eplb-glm5-mega-") as directory:
            mp.spawn(
                _run_mega,
                args=(os.path.join(directory, "rendezvous"),),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
