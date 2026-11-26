"""Compare the effective W8A8 operands and outputs across Qwen batch shapes."""

import argparse
import os
from contextlib import ExitStack
from unittest.mock import patch

import torch
from torch.nn.functional import linear

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import DEFAULT_FP8_BLOCK_WISE_QUANT_CONFIG
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    _transform_scale_ue8m0,
    quant_weight_ue8m0,
    quant_weight_ue8m0_packed,
)
from rtp_llm.models_py.modules import SelectTopk
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
    deepgemm_hybrid_executor as moe_impl,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
    PureTpRouterFp8PerBlock,
)
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

# Separate FP32 accumulation can round to a neighboring BF16 value.
REFERENCE_RTOL = 0.008


def randn(shape, scale=0.125):
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * scale


def batch_cases(width):
    target = randn((1, width))
    yield target, 0
    for size, row, scale in (
        (64, 0, 0.03125),
        (256, 255, 16),
        (257, 0, 16),
        (525, 524, 0.5),
    ):
        batch = randn((size, width), scale)
        batch[row] = target[0]
        yield batch, row


def logical_scale(scale, width):
    if scale.dtype == torch.int32:
        shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=scale.device)
        exponent = ((scale.unsqueeze(-1) >> shifts) & 0xFF).flatten(-2)
        exponent = exponent[..., : width // 128]
        return torch.exp2(exponent.float() - 127)
    assert scale.dtype == torch.float32
    return scale


def dequantize(weight, scale):
    scale = logical_scale(scale, weight.shape[-1])
    return weight.float() * scale.repeat_interleave(128, dim=-1)


def compare(reference, actual, name, *, require_equal):
    differences = {}
    for key, expected in reference.items():
        observed = actual[key]
        assert expected.shape == observed.shape and expected.dtype == observed.dtype
        different = (
            (
                expected.contiguous().view(torch.uint8).flatten()
                != observed.contiguous().view(torch.uint8).flatten()
            )
            .reshape(-1, expected.element_size())
            .any(dim=1)
        )
        indices = different.nonzero().flatten()
        if len(indices):
            differences[key] = (len(indices), indices[0].item())
    print(f"{name}, mismatches (count, first flat index) {differences}", flush=True)
    if require_equal:
        assert not differences, differences


def check_linear(tp4, require_equal):
    dims = (
        ((4096, 5120), (4096, 3072), (2048, 4096), (4096, 256), (256, 4096))
        if tp4
        else ((2048, 12288), (2048, 8192), (4096, 2048), (2048, 512), (512, 2048))
    )
    hw = HWKernelConfig()
    hw.batch_invariant = require_equal
    for k, n in dims:
        weight, scale = quant_weight_ue8m0_packed(randn((n, k)))
        projection = LinearFactory.create_linear(
            weight, None, scale, DEFAULT_FP8_BLOCK_WISE_QUANT_CONFIG, hw
        )
        assert isinstance(projection, CudaFp8GEMMLinear)
        assert projection._flashinfer_linear is None
        for hidden, row in batch_cases(k):
            q, s = projection._deepgemm_linear.quantize_input(hidden)
            actual = {
                "input.fp8": q[row].clone(),
                "input.scale": logical_scale(s[row], k),
                "output": projection(hidden)[row].clone(),
            }
            if len(hidden) == 1:
                expected = actual
                ref = linear(dequantize(q[row], s[row]), dequantize(weight, scale))
                torch.testing.assert_close(
                    actual["output"].float(),
                    ref.to(torch.bfloat16).float(),
                    rtol=REFERENCE_RTOL,
                    atol=1e-4,
                )
            else:
                compare(
                    expected,
                    actual,
                    f"linear {k}x{n}, {len(hidden)} rows, target {row}",
                    require_equal=require_equal,
                )


def trace_experts(executor, payload, row):
    """Read one live route from the existing scatter and GEMM buffers."""
    fields = {}
    dispatch = []

    def scatter(op, index_arg):
        def record(*args, **kwargs):
            op(*args, **kwargs)
            fields["indices"] = args[index_arg][row].long()

        return record

    def gemm(op):
        def record(activation, weight, output, *args, **kwargs):
            x, scale = activation
            op(activation, weight, output, *args, **kwargs)
            indices = fields["indices"]
            name = "down" if "up.output" in fields else "up"
            dispatch.append(op.__name__)
            fields[f"{name}.fp8"] = (
                x.view(torch.uint8).reshape(-1, x.shape[-1])[indices].view(x.dtype)
            )
            fields[f"{name}.scale"] = logical_scale(
                scale.reshape(-1, scale.shape[-1])[indices], x.shape[-1]
            )
            fields[f"{name}.output"] = output.reshape(-1, output.shape[-1])[indices]

        return record

    with ExitStack() as hooks:
        for name, index_arg in (("ep_scatter", 8), ("ep_scatter_v2", 7)):
            hooks.enter_context(
                patch.object(
                    moe_impl, name, scatter(getattr(moe_impl, name), index_arg)
                )
            )
        for name in (
            "m_grouped_fp8_gemm_nt_masked",
            "m_grouped_fp8_gemm_nt_contiguous",
        ):
            hooks.enter_context(
                patch.object(moe_impl, name, gemm(getattr(moe_impl, name)))
            )
        result = executor.execute(payload, "SiGLU", None, None, False, None)
    del fields["indices"]
    fields["merged"] = result.fused_expert_output[row].clone()
    assert len(dispatch) == 2 and dispatch[0] == dispatch[1]
    return dispatch[0], fields


def check_graph_replay(router, executor, width, experts, topk):
    hidden = randn((3, width))
    slots = torch.arange(topk, dtype=torch.int32, device=hidden.device)
    ids = slots.expand(3, -1).clone()
    probabilities = torch.full_like(ids, 1 / topk, dtype=torch.float32)

    def forward():
        payload = router.prepare(hidden, None, None, probabilities, ids)
        result = executor.execute(payload, "SiGLU", None, None, False, None)
        return result.fused_expert_output

    # Capture with repeated experts and replay with disjoint expert ranges.
    # Both layouts use the same bucket and leave different amounts of spare workspace.
    executor.enable_cuda_graph = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with (
        torch.cuda.stream(stream),
        patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}),
    ):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    for offsets in ((0, 0, 0), (0, topk, experts - topk)):
        ids.copy_(slots + torch.tensor(offsets, device=ids.device)[:, None])
        expected = forward()
        graph.replay()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


def check_moe(args):
    k, e, i, topk = (4096, 512, 256, 10) if args.tp4_397b else (2048, 256, 512, 8)
    cfg = ModelConfig()
    cfg.model_type = "qwen35_moe"
    cfg.data_type = "bf16"
    cfg.hidden_size, cfg.expert_num, cfg.moe_k = k, e, topk
    cfg.moe_inter_size = i * (4 if args.tp4_397b else 1)
    cfg.has_moe_norm = True
    parallel = ParallelismConfig()
    parallel.tp_size = parallel.ffn_tp_size = 4 if args.tp4_397b else 1
    moe_config = MoeConfig()
    if args.contiguous:
        moe_config.masked_max_token_num = 0
    adapter = MoEConfigAdapter(
        cfg, parallel, moe_config, batch_invariant=not args.legacy
    )
    quant = FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128])
    weights = {}
    for wkey, skey, shape in (
        (W.moe_w1, W.moe_s1, (e, 2 * i, k)),
        (W.moe_w2, W.moe_s2, (e, k, i)),
    ):
        weights[wkey], scale = quant_weight_ue8m0(randn(shape, 0.02), [128, 128])
        weights[skey] = _transform_scale_ue8m0(scale, mn=shape[-2])
    router = PureTpRouterFp8PerBlock(adapter, quant)
    executor = moe_impl.DeepGemmHybridExecutor(adapter, quant, weights)
    hw = HWKernelConfig()
    hw.batch_invariant = not args.legacy
    gate = LinearFactory.create_linear(
        randn((k, e), 0.02), None, None, DEFAULT_FP8_BLOCK_WISE_QUANT_CONFIG, hw
    )
    select = SelectTopk(cfg)

    for hidden, row in batch_cases(k):
        logits = gate(hidden)
        ids = torch.empty((len(hidden), topk), dtype=torch.int32, device="cuda")
        probs = torch.empty_like(ids, dtype=torch.float32)
        select(logits, ids, probs)
        payload = router.prepare(hidden, None, None, probs, ids)
        actual = dict(
            router_logits=logits[row].clone(),
            ids=ids[row].clone(),
            probabilities=probs[row].clone(),
            input_fp8=payload.expert_x[row].clone(),
            input_scale=logical_scale(payload.expert_x_scale[row], k),
        )
        dispatch, trace = trace_experts(executor, payload, row)
        actual.update(trace)
        if len(hidden) == 1:
            expected = actual
            print(f"MoE solo uses {dispatch}", flush=True)
            route = ids[row].long()
            for name, wkey, skey in (
                ("up", W.moe_w1, W.moe_s1),
                ("down", W.moe_w2, W.moe_s2),
            ):
                w = weights[wkey].view(torch.uint8)[route].view(torch.float8_e4m3fn)
                w = dequantize(w, weights[skey][route])
                x = dequantize(actual[f"{name}.fp8"], actual[f"{name}.scale"])
                ref = torch.bmm(w, x.unsqueeze(-1)).squeeze(-1).to(torch.bfloat16)
                torch.testing.assert_close(
                    actual[f"{name}.output"].float(),
                    ref.float(),
                    rtol=REFERENCE_RTOL,
                    atol=1e-4,
                )
            merged = (
                (actual["down.output"].float() * probs[row, :, None])
                .sum(dim=0)
                .to(torch.bfloat16)
            )
            torch.testing.assert_close(
                actual["merged"].float(), merged.float(), rtol=REFERENCE_RTOL, atol=1e-4
            )
        else:
            compare(
                expected,
                actual,
                f"MoE {len(hidden)} rows, target {row}, {dispatch}",
                require_equal=not args.legacy,
            )
    if not args.legacy:
        check_graph_replay(router, executor, k, e, topk)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--contiguous", action="store_true")
    parser.add_argument("--tp4-397b", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    check_linear(args.tp4_397b, require_equal=not args.legacy)
    check_moe(args)


if __name__ == "__main__":
    main()
