"""Opt-in two-GPU accuracy/replay regression for the pinned DeepGEMM wheel.

Run under the repository GPU lock with RTP_RUN_MEGA_MOE_FP8_UPGRADE_TEST=1.
The independent reference uses dequantized checkpoint weights and FP32 math.
Graph inputs and gates change on each replay to detect stale readiness/data.
"""

import json
import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _quantize(weight):
    grouped = weight.float().reshape(*weight.shape[:-1], -1, 128)
    scale = torch.pow(
        2.0, torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1e-4) / 448.0))
    )
    quantized = (
        (grouped / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(weight.shape)
    )
    packed = (
        (scale.contiguous().view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)
    )
    reference = (
        quantized.float().reshape(grouped.shape) * scale.unsqueeze(-1)
    ).reshape(weight.shape)
    return quantized, packed, reference


def _worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    from deep_gemm import mega_fp8

    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
        MegaMoeFp8Executor,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se import (
        MegaMoeFp8SEExecutor,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
        _get_or_create_mega_fp8_buf,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fp8_weights import (
        expand_fp8_scale,
        prepare_mega_moe_fp8_weights,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
        FusedMegaMoEInputPacker,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.se_buffer import (
        _get_or_create_mega_fp8_se_buf,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.se_input_packer import (
        FusedMegaMoESEInputPacker,
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        device_id=device,
        timeout=timedelta(seconds=300),
    )
    torch.manual_seed(282)
    hidden, inter, experts, topk, cap = 4096, 1024, 16, 4, 129
    w1, s1, ref1 = _quantize(
        torch.randn(experts, 2 * inter, hidden, device=device, dtype=torch.bfloat16)
        / hidden**0.5
    )
    w2, s2, ref2 = _quantize(
        torch.randn(experts, hidden, inter, device=device, dtype=torch.bfloat16)
        / inter**0.5
    )
    sw1, ss1, sr1 = _quantize(
        torch.randn(2 * inter, hidden, device=device, dtype=torch.bfloat16)
        / hidden**0.5
    )
    sw2, ss2, sr2 = _quantize(
        torch.randn(hidden, inter, device=device, dtype=torch.bfloat16) / inter**0.5
    )
    local = slice(rank * experts // 2, (rank + 1) * experts // 2)
    l1, l2 = prepare_mega_moe_fp8_weights(
        w1[local].clone(), s1[local], w2[local].clone(), s2[local], inter, "gate_up"
    )
    shared = mega_fp8.transform_weights_for_mega_moe_fp8(
        (sw1, expand_fp8_scale(ss1, 2 * inter, hidden)),
        (sw2, expand_fp8_scale(ss2, hidden, inter)),
    )
    shared = tuple(
        (w, s.transpose(-1, -2).contiguous().transpose(-1, -2)) for w, s in shared
    )

    def reference(x, weights, indices, gates, with_shared):
        out = torch.zeros(x.shape, device=device, dtype=torch.float32)
        for expert in range(experts):
            rows, slots = torch.where(indices == expert)
            h = x[rows].float() @ ref1[expert].t()
            gate, up = h.chunk(2, -1)
            y = (torch.nn.functional.silu(gate) * up) @ ref2[expert].t()
            out.index_add_(0, rows, y * weights[rows, slots, None])
        if with_shared:
            h = x.float() @ sr1.t()
            gate, up = h.chunk(2, -1)
            out += ((torch.nn.functional.silu(gate) * up) @ sr2.t()) * gates[:, None]
        return out

    for reserve in ("0", "1"):
        os.environ["MEGA_MOE_FP8_RESERVE_SM"] = reserve
        for with_shared in (False, True):
            cls = MegaMoeFp8SEExecutor if with_shared else MegaMoeFp8Executor
            executor = cls.__new__(cls)
            torch.nn.Module.__init__(executor)
            executor.config = executor.cfg = SimpleNamespace(
                layer_id=0, ep_size=2, expert_num=experts, moe_k=topk
            )
            executor.l1, executor.l2 = l1, l2
            executor.shared_l1, executor.shared_l2 = shared
            executor.uses_shared_expert_gates = with_shared
            kwargs = dict(
                group=dist.group.WORLD,
                num_experts=experts,
                num_max_tokens_per_rank=cap,
                num_topk=topk,
                hidden=hidden,
                intermediate_hidden=inter,
                use_fp8_dispatch=True,
                activation="swiglu",
            )
            if with_shared:
                executor._mega_buf = _get_or_create_mega_fp8_se_buf(
                    **kwargs, num_shared_experts=1
                )
                executor._input_packer = FusedMegaMoESEInputPacker()
            else:
                executor._mega_buf = _get_or_create_mega_fp8_buf(**kwargs)
                executor._input_packer = FusedMegaMoEInputPacker()
            executor._mega_y = torch.empty(
                cap, hidden, device=device, dtype=torch.bfloat16
            )
            for counts in ((1, 1), (32, 32), (129, 65), (0, 32), (32, 0)):
                tokens = counts[rank]
                x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
                scores = torch.randn(tokens, experts, device=device)
                weights, indices = scores.softmax(-1).topk(topk, dim=-1)
                weights /= weights.sum(-1, keepdim=True)
                gates = torch.rand(tokens, device=device)

                def launch():
                    if with_shared:
                        return executor.forward(
                            x, weights, indices, {"shared_expert_gates": gates}
                        )
                    return executor.forward(x, weights, indices)

                for _ in range(2):
                    launch()
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = launch()
                min_cosine, max_relative = 1.0, 0.0
                for replay in range(10):
                    x.copy_(torch.randn_like(x))
                    gates.copy_(torch.rand_like(gates))
                    # Change both routing and weights while keeping graph addresses fixed.
                    new_weights, new_indices = (
                        torch.randn_like(scores).softmax(-1).topk(topk, dim=-1)
                    )
                    weights.copy_(new_weights / new_weights.sum(-1, keepdim=True))
                    indices.copy_(new_indices)
                    graph.replay()
                    actual = captured.clone()
                    eager = launch().clone()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
                    assert torch.isfinite(actual).all()
                    if tokens:
                        expected = reference(x, weights, indices, gates, with_shared)
                        cosine = torch.nn.functional.cosine_similarity(
                            actual.float().flatten(), expected.flatten(), dim=0
                        ).item()
                        relative = (
                            (actual.float() - expected).norm() / expected.norm()
                        ).item()
                        assert cosine > 0.99, (rank, counts, cosine)
                        assert relative < 0.1, (rank, counts, relative)
                        min_cosine, max_relative = min(min_cosine, cosine), max(
                            max_relative, relative
                        )
                print(
                    json.dumps(
                        dict(
                            rank=rank,
                            reserve=reserve,
                            shared=with_shared,
                            tokens=counts,
                            replays=10,
                            min_cosine=min_cosine,
                            max_relative_l2=max_relative,
                        )
                    ),
                    flush=True,
                )
                del graph
            dist.barrier()
    dist.destroy_process_group()


@unittest.skipUnless(
    os.environ.get("RTP_RUN_MEGA_MOE_FP8_UPGRADE_TEST") == "1", "opt-in two-GPU test"
)
class MegaMoeFp8UpgradeTest(unittest.TestCase):
    def test_accuracy_and_changing_graph_inputs(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _worker,
                args=(os.path.join(directory, "rendezvous"),),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
