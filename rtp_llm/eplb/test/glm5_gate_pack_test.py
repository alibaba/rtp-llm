import json
import os
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.eplb.test.test_support import make_layer
from rtp_llm.models_py.modules.base.cuda.select_topk import GroupTopK
from rtp_llm.models_py.modules.glm5_mega_moe.input_packer_triton import (
    fused_pack_mega_moe_inputs,
)


class Glm5GatePackTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(735)
        self.plan = make_layer(
            tuple(range(256)) + tuple(range(32)) * 2, device="cuda", replica_offset=7
        )
        self.topk = GroupTopK()

    def buffers(self, rows, dim, k, block_m):
        return SimpleNamespace(
            x=torch.empty((rows, dim), device="cuda", dtype=torch.float8_e4m3fn),
            x_sf=torch.empty((rows, dim // 128), device="cuda", dtype=torch.int32),
            topk_idx=torch.empty((rows, k), device="cuda", dtype=torch.int64),
            topk_weights=torch.empty((rows, k), device="cuda"),
            shared_l1_acts_sf=(
                torch.full(
                    (
                        ((rows + block_m - 1) // block_m)
                        * ((block_m + 127) // 128 * 128),
                        dim // 128,
                    ),
                    -1,
                    device="cuda",
                    dtype=torch.int32,
                )
                if block_m
                else None
            ),
        )

    def launch(
        self, x, logits, bias, buf, *, plan=None, offset=0, block_m=0, scale=2.5
    ):
        fused_pack_mega_moe_inputs(
            x,
            buf.topk_weights,
            buf.topk_idx,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            out_shared_sf=buf.shared_l1_acts_sf,
            shared_block_m=block_m,
            eplb=plan,
            token_offset=offset,
            router_logits=logits,
            correction_bias=bias,
            routed_scaling_factor=scale,
        )

    def reference(
        self, x, logits, bias, buf, *, plan=None, offset=0, block_m=0, scale=2.5
    ):
        rows, k = buf.topk_idx.shape
        ids = torch.empty_like(buf.topk_idx)
        weights = torch.empty_like(buf.topk_weights)
        if rows:
            self.topk(weights, ids, logits.float(), bias, 1, 1, k, True, scale)
        if plan is not None:
            tokens = torch.arange(rows, device="cuda", dtype=torch.int64) + offset
            rotation = (tokens * 23333 + plan.replica_offset) & 0xFFFFFFFF
            replicas = rotation[:, None] % plan.logical_count[ids].long()
            ids = plan.logical_to_physical[ids, replicas].long()
        fused_pack_mega_moe_inputs(
            x,
            weights,
            ids,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            out_shared_sf=buf.shared_l1_acts_sf,
            shared_block_m=block_m,
        )

    def compare(self, a, b):
        for name in ("x", "x_sf", "topk_idx", "shared_l1_acts_sf"):
            actual, expected = getattr(a, name), getattr(b, name)
            if actual is not None:
                if name == "x":
                    actual, expected = actual.view(torch.uint8), expected.view(
                        torch.uint8
                    )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            a.topk_weights, b.topk_weights, rtol=2e-6, atol=2e-7, equal_nan=True
        )

    def test_matches_group_topk_and_quant_pack(self):
        for rows, dim, experts, k in (
            (0, 512, 256, 8),
            (1, 512, 256, 8),
            (17, 512, 256, 8),
            (2053, 512, 256, 8),
            (257, 6144, 256, 8),
            (33, 512, 384, 16),
            (7, 512, 512, 32),
            (5, 512, 256, 2),
        ):
            for dtype, block_m in (
                (torch.bfloat16, 0),
                (torch.float32, 64),
                (torch.bfloat16, 192),
            ):
                with self.subTest(
                    rows=rows,
                    dim=dim,
                    experts=experts,
                    k=k,
                    dtype=dtype,
                    block_m=block_m,
                ):
                    x = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16)
                    logits = torch.randn(rows, experts, device="cuda", dtype=dtype) * 4
                    bias = torch.randn(experts, device="cuda") * 0.2
                    plan = self.plan if experts == 256 else None
                    actual, expected = (
                        self.buffers(rows, dim, k, block_m) for _ in range(2)
                    )
                    self.reference(
                        x,
                        logits,
                        bias,
                        expected,
                        plan=plan,
                        offset=200000,
                        block_m=block_m,
                    )
                    self.launch(
                        x,
                        logits,
                        bias,
                        actual,
                        plan=plan,
                        offset=200000,
                        block_m=block_m,
                    )
                    self.compare(actual, expected)

    def test_ties_nonfinite_strides_and_uint32_wrap(self):
        rows, dim = 17, 512
        x = torch.randn(rows * 2, dim, device="cuda", dtype=torch.bfloat16)[::2]
        logits = torch.zeros(rows * 2, 256, device="cuda")[::2]
        logits[1, :16] = float("nan")
        logits[2, :16] = float("inf")
        logits[3].fill_(-float("inf"))
        logits[4].fill_(float("nan"))
        bias = torch.zeros(256, device="cuda")
        for offset in (0, 200000, 2**32 - 5):
            actual, expected = (self.buffers(rows, dim, 8, 64) for _ in range(2))
            self.reference(
                x, logits, bias, expected, plan=self.plan, offset=offset, block_m=64
            )
            self.launch(
                x, logits, bias, actual, plan=self.plan, offset=offset, block_m=64
            )
            self.compare(actual, expected)

    def test_graph_reads_new_logits_maps_and_replica_counts(self):
        rows = 17
        x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(256, device="cuda") * 0.1
        actual, expected = (self.buffers(rows, 512, 8, 64) for _ in range(2))
        self.launch(x, logits, bias, actual, plan=self.plan, block_m=64)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.launch(x, logits, bias, actual, plan=self.plan, block_m=64)
        self.plan.logical_count.fill_(1)
        self.plan.logical_to_physical.copy_(self.plan.logical_to_physical.roll(7, 0))
        logits.add_(0.3)
        graph.replay()
        self.reference(x, logits, bias, expected, plan=self.plan, block_m=64)
        self.compare(actual, expected)

    def test_one_gpu_launch_and_no_standalone_routing(self):
        from torch.profiler import ProfilerActivity, profile

        x = torch.randn(257, 6144, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(257, 256, device="cuda", dtype=torch.bfloat16)
        bias = torch.zeros(256, device="cuda")
        buf = self.buffers(257, 6144, 8, 64)
        self.launch(x, logits, bias, buf, plan=self.plan, block_m=64)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            self.launch(x, logits, bias, buf, plan=self.plan, block_m=64)
            torch.cuda.synchronize()
        kernels = [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        self.assertEqual(len(kernels), 1, kernels)
        self.assertIn("_pack_mega_moe_inputs_optimized_kernel", kernels[0])
        print(json.dumps({"gate_pack_gpu_kernels": kernels}))

    def test_all_mega_backends_consume_fused_buffers(self):
        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import GLM5MegaMoE
        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8 import GLM5MegaMoEFP8
        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8_se import (
            GLM5MegaMoEFP8SE,
        )
        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fused import (
            GLM5MegaMoEFused,
        )
        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_se import GLM5MegaMoESE

        x = torch.randn(17, 512, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
        bias = torch.zeros(256, device="cuda")
        for cls in (
            GLM5MegaMoE,
            GLM5MegaMoESE,
            GLM5MegaMoEFP8,
            GLM5MegaMoEFP8SE,
            GLM5MegaMoEFused,
        ):
            with self.subTest(backend=cls.__name__):
                shared = cls in (GLM5MegaMoESE, GLM5MegaMoEFP8SE)
                block_m = 64 if shared else 0
                moe = object.__new__(cls)
                torch.nn.Module.__init__(moe)
                moe.cfg = SimpleNamespace(
                    layer_id=0,
                    ep_size=1,
                    n_routed_experts=320,
                    n_activated_experts=8,
                    local_expert_start=0,
                    local_expert_end=320,
                )
                moe._mega_buf = self.buffers(17, 512, 8, block_m)
                moe._mega_buf.num_max_tokens_per_rank = 17
                moe._mega_y = torch.empty_like(x)
                for name in (
                    "_mega_l1_w",
                    "_mega_l1_sf",
                    "_mega_l2_w",
                    "_mega_l2_sf",
                    "_shared_l1_w",
                    "_shared_l1_sf",
                    "_shared_l2_w",
                    "_shared_l2_sf",
                ):
                    setattr(moe, name, None)
                moe._shared_mid_fp8 = torch.empty_like(x)
                moe._shared_mid_sf = None
                moe._mega_group = None
                moe._eplb_global_counts = torch.zeros(
                    320, device="cuda", dtype=torch.int32
                )
                moe.expert_token_counts = torch.zeros_like(moe._eplb_global_counts)
                moe._check_shared_expert_ready = lambda: None
                moe._maybe_pre_kernel_barrier = lambda tokens: None
                calls = []

                def backend(y, *args, **kwargs):
                    calls.append(kwargs.get("cumulative_local_expert_recv_stats"))
                    y.copy_(x)

                module = SimpleNamespace(
                    fp8_fp4_mega_moe=backend,
                    fp8_fp8_mega_moe=backend,
                    fp8_fp4_mega_moe_fused=backend,
                    get_block_m_for_mega_moe=lambda *args: 64,
                    get_block_m_for_mega_moe_fp8=lambda *args: 64,
                )
                with patch.dict(sys.modules, {"deep_gemm": module}), patch(
                    cls.__module__ + "._sync_cuda_graph_warmup_ranks"
                ):
                    output = moe._forward_impl(
                        x,
                        None,
                        None,
                        inputs_prepacked=False,
                        eplb=self.plan,
                        gate_logits=logits,
                        gate_bias=bias,
                        route_scale=2.5,
                    )
                torch.testing.assert_close(output, x)
                expected = self.buffers(17, 512, 8, block_m)
                self.reference(
                    x, logits, bias, expected, plan=self.plan, block_m=block_m
                )
                self.compare(moe._mega_buf, expected)
                self.assertEqual(len(calls), 1)
                if cls is GLM5MegaMoEFused:
                    torch.testing.assert_close(
                        moe.expert_token_counts.long(),
                        torch.bincount(expected.topk_idx.flatten(), minlength=320),
                    )
                else:
                    self.assertIs(calls[0], moe.expert_token_counts)

    def test_switch_and_supported_layout(self):
        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_wrapper import (
            MegaMoeWrapper,
        )

        wrapper = object.__new__(MegaMoeWrapper)
        torch.nn.Module.__init__(wrapper)
        wrapper.mega_moe = SimpleNamespace(_input_packer=SimpleNamespace(name="fused"))
        config = SimpleNamespace(
            model_type="glm_5",
            moe_k=8,
            moe_n_group=1,
            moe_topk_group=1,
            has_moe_norm=True,
        )
        x = torch.empty(17, 512, device="cuda", dtype=torch.bfloat16)
        bias = torch.zeros(256, device="cuda")
        for mode, plan, expected in (
            ("auto", self.plan, True),
            ("auto", None, False),
            ("0", self.plan, False),
            ("1", None, True),
        ):
            with patch.dict("os.environ", {"RTP_LLM_GLM5_GATE_PACK": mode}):
                self.assertEqual(
                    wrapper.supports_gate_pack(x, bias, config, plan), expected
                )
        with patch.dict("os.environ", {"RTP_LLM_GLM5_GATE_PACK": "1"}):
            self.assertFalse(
                wrapper.supports_gate_pack(x.float(), bias, config, self.plan)
            )
            config.moe_n_group = 8
            self.assertFalse(wrapper.supports_gate_pack(x, bias, config, self.plan))
            config.moe_n_group = 1
            wrapper.mega_moe._input_packer.name = "torch"
            self.assertFalse(wrapper.supports_gate_pack(x, bias, config, self.plan))

    def test_profile_group_topk_pack_against_fusion(self):
        from torch.profiler import ProfilerActivity, profile

        from rtp_llm.models_py.modules.glm5_mega_moe import mega_gate

        measurements = []
        for rows in (1, 257, 1024, 2048, 4096, 8192):
            x = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
            gate_weight = (
                torch.randn(256, 6144, device="cuda", dtype=torch.bfloat16) * 0.01
            )
            logits = torch.nn.functional.linear(x, gate_weight)
            bias = torch.randn(256, device="cuda") * 0.1
            config = SimpleNamespace(moe_k=8, routed_scaling_factor=2.5)
            buf = self.buffers(rows, 6144, 8, 0)
            ids, weights = torch.empty_like(buf.topk_idx), torch.empty_like(
                buf.topk_weights
            )

            def baseline():
                self.topk(weights, ids, logits.float(), bias, 1, 1, 8, True, 2.5)
                fused_pack_mega_moe_inputs(
                    x,
                    weights,
                    ids,
                    buf.x,
                    buf.x_sf,
                    buf.topk_idx,
                    buf.topk_weights,
                    eplb=self.plan,
                )

            def fused():
                self.launch(x, logits, bias, buf, plan=self.plan)

            def linear_gate_pack():
                self.launch(
                    x,
                    torch.nn.functional.linear(x, gate_weight),
                    bias,
                    buf,
                    plan=self.plan,
                )

            def native_gate_pack():
                mega_gate.mega_gate(
                    x, gate_weight, bias, config, self.plan, out=(ids, weights)
                )
                fused_pack_mega_moe_inputs(
                    x, weights, ids, buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights
                )

            for name, call in (
                ("group_topk_pack", baseline),
                ("sigmoid_gate_pack", fused),
                ("linear_gate_and_fused_pack", linear_gate_pack),
                ("native_gate_and_pack", native_gate_pack),
            ):
                for _ in range(30):
                    call()
                torch.cuda.synchronize()
                iterations = 100
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                ) as prof:
                    for _ in range(iterations):
                        call()
                    torch.cuda.synchronize()
                kernels = [
                    e
                    for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                ]
                names = Counter(e.name for e in kernels)
                self.assertTrue(
                    all(count % iterations == 0 for count in names.values()), names
                )
                if name == "sigmoid_gate_pack":
                    self.assertEqual(len(kernels), iterations)
                    self.assertTrue(
                        all(
                            "_pack_mega_moe_inputs_optimized_kernel" in n for n in names
                        )
                    )
                elif name == "group_topk_pack":
                    self.assertTrue(
                        any("group_idx_and_topk_idx_kernel" in n for n in names)
                    )
                    self.assertTrue(
                        any(
                            "_pack_mega_moe_inputs_optimized_kernel" in n for n in names
                        )
                    )
                else:
                    self.assertTrue(
                        any(
                            "_pack_mega_moe_inputs_optimized_kernel" in n for n in names
                        )
                    )
                    self.assertGreater(len(kernels), iterations)
                measurements.append(
                    {
                        "path": name,
                        "rows": rows,
                        "hidden": 6144,
                        "experts": 256,
                        "topk": 8,
                        "warmup": 30,
                        "iterations": iterations,
                        "kernel_sum_us_per_iter": sum(
                            e.time_range.elapsed_us() for e in kernels
                        )
                        / iterations,
                        "launches_per_iter": len(kernels) / iterations,
                        "kernels_per_iter": {
                            n: c / iterations for n, c in names.items()
                        },
                        "torch_compile": False,
                    }
                )
        result = {"gpu": torch.cuda.get_device_name(), "measurements": measurements}
        print(json.dumps(result))
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        (output / "glm5_gate_pack_kernel_profile.json").write_text(
            json.dumps(result, indent=2)
        )


if __name__ == "__main__":
    unittest.main()
