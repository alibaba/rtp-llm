"""Scale layout must preserve routed FP8 values and the grouped GEMM result."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
    deepgemm_hybrid_executor as owner,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128
from rtp_llm.utils.model_weight import W


def make_executor(graph=True):
    model = ModelConfig()
    model.quant_config = Fp8BlockWiseQuantConfig()
    model.data_type = "bf16"
    model.expert_num, model.moe_k = 512, 10
    model.hidden_size, model.moe_inter_size = 2560, 640
    parallel = ParallelismConfig()
    parallel.world_size = parallel.local_world_size = 2
    parallel.tp_size = parallel.ep_size = 2
    parallel.tp_rank = parallel.ep_rank = parallel.world_rank = 0
    config = MoEConfigAdapter(
        model_config=model,
        parallelism_config=parallel,
        moe_config=MoeConfig(),
        enable_cuda_graph=graph,
    )
    weights = {}
    for key, scale_key, shape in (
        (W.moe_w1, W.moe_s1, (256, 1280, 2560)),
        (W.moe_w2, W.moe_s2, (256, 2560, 640)),
    ):
        weights[key] = torch.randint(-8, 9, shape, device="cuda", dtype=torch.int8).to(
            torch.float8_e4m3fn
        )
        weights[scale_key] = (
            torch.rand((shape[0], shape[1] // 128, shape[2] // 128), device="cuda")
            * 0.004
            + 0.003
        )
    return owner.DeepGemmHybridExecutor(
        config,
        FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128]),
        weights,
    )


def make_payload(rows, routing="uniform", offset=0):
    hidden = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
    values, scales = trt_fp8_quantize_128(hidden, False)
    tokens = torch.arange(rows, device="cuda")[:, None]
    routes = torch.arange(10, device="cuda")[None, :]
    ids = ((tokens * 10 + routes + offset) % 256).to(torch.int64)
    if routing == "skewed":
        ids = ((tokens % 4 + routes + offset) % 256).to(torch.int64)
    # Model local EP work, with half the routes owned by the other rank.
    ids[:, 5:] = -1
    weights = torch.rand(rows, 10, device="cuda", dtype=torch.float32)
    weights /= weights.sum(dim=1, keepdim=True)
    counts = torch.bincount(ids[ids >= 0], minlength=256).to(torch.int32)
    return ExpertForwardPayload(
        expert_x=values,
        expert_x_scale=scales,
        expert_topk_ids=ids,
        expert_topk_weights=weights,
        expert_ids_are_local=True,
        expert_tokens_meta=ExpertTokensMetadata(
            expert_num_tokens=counts, expert_num_tokens_cpu=counts.cpu().tolist()
        ),
    )


def invoke(executor, payload, masked):
    fn = executor.execute_masked if masked else executor.execute_contiguous
    return fn(payload, "SiGLU", None, None, False, None).fused_expert_output


def clone_payload(payload):
    return ExpertForwardPayload(
        expert_x=payload.expert_x.clone(),
        expert_x_scale=payload.expert_x_scale,
        expert_topk_ids=payload.expert_topk_ids,
        expert_topk_weights=payload.expert_topk_weights,
        expert_tokens_meta=payload.expert_tokens_meta,
        expert_ids_are_local=True,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DeepGemmSM90ScaleLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (9, 0):
            raise unittest.SkipTest("SM90 required")
        torch.manual_seed(20261116)
        cls.executor = make_executor()
        assert cls.executor._sm90_mn_major_scales

    def test_scale_strides_and_values(self):
        for shape in ((128, 5), (256, 20), (256, 128, 5), (256, 256, 20)):
            baseline = owner._empty_fp32_moe_scale(shape, "cuda")
            result = owner._empty_fp32_moe_scale(shape, "cuda", mn_major=True)
            values = torch.arange(
                baseline.numel(), device="cuda", dtype=torch.float32
            ).reshape(shape)
            result.copy_(values)
            torch.testing.assert_close(result, values, rtol=0, atol=0)
            self.assertTrue(baseline.is_contiguous())
            self.assertEqual(result.stride(-2), 1)
            self.assertEqual(result.stride(-1), shape[-2])

    def test_native_column_major_quant_is_exact(self):
        for rows in (128, 256, 8192, 32768):
            for width in (640, 2560):
                x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                x[0].zero_()
                a, sa = trt_fp8_quantize_128(x, False)
                b, sb = trt_fp8_quantize_128(x, True)
                self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
                torch.testing.assert_close(sa, sb.t(), rtol=0, atol=0)

    def test_masked_changed_input_graph_replay(self):
        executor = self.executor
        executor.enable_cuda_graph = True
        for rows in (16, 32, 64):
            with self.subTest(rows=rows):
                payload = make_payload(rows)
                graphs, outputs = [], []
                for enabled in (False, True):
                    executor._sm90_mn_major_scales = enabled
                    for _ in range(2):
                        invoke(executor, payload, True)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        outputs.append(invoke(executor, payload, True))
                    graphs.append(graph)
                for routing, offset in (("uniform", 13), ("skewed", 31)):
                    fresh = make_payload(rows, routing, offset)
                    for attr in (
                        "expert_x",
                        "expert_x_scale",
                        "expert_topk_ids",
                        "expert_topk_weights",
                    ):
                        getattr(payload, attr).copy_(getattr(fresh, attr))
                    payload.expert_tokens_meta.expert_num_tokens.copy_(
                        fresh.expert_tokens_meta.expert_num_tokens
                    )
                    for graph in graphs:
                        graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                for graph in graphs:
                    graph.reset()

    def test_contiguous_executor(self):
        executor = self.executor
        executor.enable_cuda_graph = False
        for rows in (8192, 32768):
            for routing in ("uniform", "skewed"):
                with self.subTest(rows=rows, routing=routing):
                    payload = make_payload(rows, routing)
                    executor._sm90_mn_major_scales = False
                    baseline = invoke(executor, clone_payload(payload), False)
                    executor._sm90_mn_major_scales = True
                    actual = invoke(executor, clone_payload(payload), False)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)

    def test_other_architectures_keep_baseline(self):
        executor = self.executor
        for capability in ((8, 0), (10, 0), (12, 0)):
            with patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ):
                other = owner.DeepGemmHybridExecutor(
                    executor.config, executor.quant_config, executor.weights
                )
            self.assertFalse(other._sm90_mn_major_scales)

    def test_unqualified_rows_keep_contiguous_scales(self):
        executor = self.executor
        executor._sm90_mn_major_scales = True
        for masked, rows_list in (
            (True, (1, 15, 17, 31, 33, 63, 65)),
            (False, (2048, 8193, 32769)),
        ):
            executor.enable_cuda_graph = masked
            for rows in rows_list:
                with self.subTest(masked=masked, rows=rows):
                    payload = make_payload(rows)
                    with patch.object(
                        owner,
                        "_empty_fp32_moe_scale",
                        wraps=owner._empty_fp32_moe_scale,
                    ) as allocation:
                        output = invoke(executor, payload, masked)
                    self.assertTrue(
                        all(not c.kwargs["mn_major"] for c in allocation.call_args_list)
                    )
                    self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
