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
            moe_inter_dim=inter,
            expert_num=experts,
            n_local_experts=128,
            moe_k=topk,
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
        os.environ["RTP_QWEN35_FUSED_MEGAMOE_GATED_SE"] = "0"
        plain = MegaMoeFp8Executor(
            cfg,
            None,
            {W.moe_w1: w1.clone(), W.moe_s1: s1, W.moe_w2: w2.clone(), W.moe_s2: s2},
        )
        os.environ["RTP_QWEN35_FUSED_MEGAMOE_GATED_SE"] = "1"
        fused = MegaMoeFp8Executor(
            cfg, None, {W.moe_w1: w1, W.moe_s1: s1, W.moe_w2: w2, W.moe_s2: s2}
        )
        torch.manual_seed(3210)
        up = CudaFp8DeepGEMMLinear(*weights((2 * inter, hidden)))
        down = CudaFp8DeepGEMMLinear(*weights((hidden, inter)))
        shared = SimpleNamespace(up_proj=up, down_proj=down)
        fused.configure_gated_shared_expert(shared)
        activation = FusedSiluAndMul()
        assert (
            plain._buffer() is not fused._buffer()
        ), "Shared workspace reused for routed-only executor"
        shared_buffer = fused._buffer()

        def run(executor, x, ids, routing, gate=None):
            payload = SimpleNamespace(
                expert_x=x, expert_topk_ids=ids, expert_topk_weights=routing
            )
            return executor.execute(
                payload,
                "SiGLU",
                None,
                None,
                False,
                None if gate is None else {"shared_expert_gates": gate},
            ).fused_expert_output

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
            activated = activation(up(x))
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
                assert fused._buffer() is shared_buffer
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
