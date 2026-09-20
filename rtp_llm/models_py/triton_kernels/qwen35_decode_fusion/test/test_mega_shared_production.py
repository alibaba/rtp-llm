"""Four-rank A/B through the production FP8 MegaMoE executor and RTP MLP."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, rendezvous, output_dir):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method="file://" + rendezvous, rank=rank, world_size=4
    )
    import deep_gemm

    from rtp_llm.models_py.modules.base import FusedSiluAndMul
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
        MegaMoeFp8Executor,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se import (
        MegaMoeFp8SEExecutor,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_inputs import (
        shared_expert_sigmoid,
    )
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
        CudaFp8DeepGEMMLinear,
    )
    from rtp_llm.models_py.triton_kernels.common.moe_gating import (
        sigmoid_gate_scale_add_triton,
    )
    from rtp_llm.utils.model_weight import W

    rows = []
    try:
        hidden, inter, experts, topk = 4096, 1024, 512, 10
        cfg = SimpleNamespace(
            ep_size=4,
            ep_rank=rank,
            max_tokens_per_rank=65536,
            model_config=SimpleNamespace(max_seq_len=65536),
            hidden_size=hidden,
            dim=hidden,
            moe_inter_dim=inter,
            expert_num=experts,
            n_local_experts=128,
            n_routed_experts=experts,
            moe_k=topk,
            n_activated_experts=topk,
            local_expert_start=rank * 128,
            route_scale=1.0,
            swiglu_limit=0.0,
            layer_id=0,
            warmup_include_capacity=False,
            moe_w1_layout="gate_up",
        )
        torch.manual_seed(3000 + rank)

        def weights(shape):
            # Exactly representable power-of-two scales, normal-sized projections.
            value = torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            )
            exponent = 122  # scale = 1/32
            scale = torch.full(
                (*shape[:-1], shape[-1] // 512),
                exponent * 0x01010101,
                device="cuda",
                dtype=torch.int32,
            )
            return value, scale.transpose(-1, -2).contiguous().transpose(-1, -2)

        w1, s1 = weights((128, 2 * inter, hidden))
        w2, s2 = weights((128, hidden, inter))
        plain = MegaMoeFp8Executor(
            cfg,
            None,
            {W.moe_w1: w1.clone(), W.moe_s1: s1, W.moe_w2: w2.clone(), W.moe_s2: s2},
        )
        fused = MegaMoeFp8SEExecutor(
            cfg, None, {W.moe_w1: w1, W.moe_s1: s1, W.moe_w2: w2, W.moe_s2: s2}
        )
        torch.manual_seed(3210)
        up = CudaFp8DeepGEMMLinear(*weights((2 * inter, hidden)))
        down = CudaFp8DeepGEMMLinear(*weights((hidden, inter)))
        shared = SimpleNamespace(up_proj=up, down_proj=down)
        fused.configure_gated_shared_expert(shared)
        activation = FusedSiluAndMul()
        assert (
            plain._mega_buf is not fused._mega_buf
        ), "Shared workspace reused for routed-only executor"
        shared_buffer = fused._mega_buf

        def k32_linear(x_fp8, x_sf, linear):
            from deep_gemm.utils.math import unpack_ue8m0_from_int

            from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fp8_weights import (
                expand_fp8_scale,
            )

            weight = linear.weight
            act_sf = unpack_ue8m0_from_int(x_sf.contiguous()).view(
                x_sf.size(0), x_sf.size(1) * 4
            )
            weight_sf = unpack_ue8m0_from_int(
                expand_fp8_scale(linear.weight_scales, weight.size(0), weight.size(1))
            ).view(weight.size(0), weight.size(1) // 32)
            out = torch.empty(
                (x_fp8.size(0), weight.size(0)),
                dtype=torch.bfloat16,
                device=x_fp8.device,
            )
            deep_gemm.fp8_gemm_nt(
                (x_fp8, act_sf.contiguous()),
                (weight, weight_sf.contiguous()),
                out,
                recipe=(1, 1, 32),
            )
            return out

        def run(executor, x, ids, routing, gate=None):
            if gate is None:
                return executor.forward(x, routing, ids)
            return executor.forward(
                x, routing, ids, extra_expert_args={"shared_expert_gates": gate}
            )

        def checkpoint(stage, tokens):
            torch.cuda.synchronize()
            with Path(output_dir, f"stage-{rank}.jsonl").open("a") as f:
                f.write(
                    json.dumps({"rank": rank, "tokens": tokens, "stage": stage}) + "\n"
                )
            dist.barrier()

        # Real B1/B2 sizes followed by decode and another B1 to exercise workspace reuse.
        for tokens in (24601, 49202, 1, 37, 24601):
            torch.manual_seed(9000 + tokens + rank)
            x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
            ids = (
                torch.arange(tokens * topk, device="cuda", dtype=torch.int64).view(
                    tokens, topk
                )
                % experts
            )
            routing = torch.softmax(torch.randn((tokens, topk), device="cuda"), dim=-1)
            checkpoint("inputs_ready", tokens)
            baseline = run(plain, x, ids, routing)
            checkpoint("routed_done", tokens)
            activated = activation(
                k32_linear(plain._mega_buf.x[:tokens], plain._mega_buf.x_sf[:tokens], up)
            )
            checkpoint("shared_activation_done", tokens)
            mlp = down(activated)
            checkpoint("shared_mlp_done", tokens)
            for kind in ("zero", "one", "dynamic"):
                logits = torch.randn((tokens, 1), device="cuda", dtype=torch.bfloat16)
                gate = shared_expert_sigmoid(logits)
                expected = baseline.clone()
                if kind == "zero":
                    gate.zero_()
                elif kind == "one":
                    gate.fill_(1)
                    expected = (baseline.float() + mlp.float()).bfloat16()
                else:
                    sigmoid_gate_scale_add_triton(logits, mlp, expected)
                actual = run(fused, x, ids, routing, gate)
                checkpoint("fused_" + kind, tokens)
                delta = (actual.float() - expected.float()).abs()
                row = {
                    "rank": rank,
                    "tokens": tokens,
                    "gate": kind,
                    "max_abs": delta.max().item(),
                    "mismatches": (delta > (0.01 + 0.01 * expected.float().abs()))
                    .sum()
                    .item(),
                    "deepgemm": deep_gemm.__file__,
                }
                rows.append(row)
                Path(output_dir, f"rank-{rank}.json").write_text(
                    json.dumps(rows, indent=2)
                )
                torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
                assert fused._mega_buf is shared_buffer
            del x, ids, routing, baseline, mlp, expected, actual, delta
    finally:
        dist.destroy_process_group()


class ProductionSharedExpertTest(unittest.TestCase):
    def test_production_executor(self):
        self.assertEqual(torch.cuda.device_count(), 4)
        output = Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp/mega-shared-production")
        )
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                worker,
                args=(str(Path(directory, "rendezvous")), str(output)),
                nprocs=4,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
