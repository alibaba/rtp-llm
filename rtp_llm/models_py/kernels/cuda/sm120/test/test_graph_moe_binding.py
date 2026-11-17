from unittest.mock import Mock

import torch
from flashinfer import block_scale_interleave, mxfp8_quantize
from flashinfer.tllm_enums import ActivationType

from rtp_llm.models_py.kernels.cuda.sm120.graph_moe.op import _tuning_config
from rtp_llm.models_py.kernels.cuda.sm120.graph_moe.provider import (
    GraphMoeOps,
    bind,
    is_available,
)


def test_hybrid_tuning_buckets_round_up():
    (spec,) = _tuning_config(8192).dynamic_tensor_specs
    assert spec.gen_tuning_buckets == (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        768,
        1024,
        1280,
        1536,
        1792,
        2048,
        2560,
        3072,
        3584,
        4096,
        8192,
    )
    for value, expected in [(0, 1), (3, 4), (257, 512), (2049, 2560), (4097, 8192)]:
        assert spec.map_to_tuning_buckets(value) == expected
    (capped,) = _tuning_config(300).dynamic_tensor_specs
    assert capped.gen_tuning_buckets[-1] == 300
    assert capped.map_to_tuning_buckets(275) == 300
    assert capped.map_to_tuning_buckets(8192) == 300


def test_provider_normalizes_into_caller_output():
    workspace_size = Mock(return_value=4)
    output = torch.empty(1, dtype=torch.bfloat16)
    workspace = torch.empty(4, dtype=torch.uint8)

    def write_output(*, output, **kwargs):
        output.fill_(2)
        return [output]

    execute = Mock(side_effect=write_output)
    provider = GraphMoeOps(workspace_size, execute)
    assert provider.workspace_size() == workspace.numel()
    options = dict(
        use_fused_finalize=False,
        enable_pdl=False,
        workspace_buffer=workspace,
    )
    assert provider.run(output=output, **options) is output
    assert output.item() == 2
    execute.assert_called_once_with(output=output, **options)


def test_separate_finalize_graph_replays_into_caller_buffers():
    device = torch.device("cuda")
    assert is_available(device)
    provider = bind(device)
    assert bind(torch.device("cuda", torch.cuda.current_device())) is provider
    experts, dim, inter, top_k, rows = 16, 512, 512, 6, 8
    mode = dict(
        activation_type=ActivationType.Swiglu,
        use_mxfp8_act_scaling=True,
        use_fused_finalize=False,
    )
    workspace_bytes = provider.workspace_size(
        rows,
        dim,
        inter,
        experts,
        top_k,
        x_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.long,
        output_dtype=torch.bfloat16,
        device=device,
        **mode,
    )
    generator = torch.Generator(device=device).manual_seed(1729)

    def weights(n, k):
        packed = torch.randint(
            0,
            256,
            (experts, n, k // 2),
            generator=generator,
            device=device,
            dtype=torch.uint8,
        ).view(torch.long)
        scales = torch.full(
            (experts, n, k // 32), 120, device=device, dtype=torch.uint8
        )
        return packed, block_scale_interleave(scales).reshape(scales.shape).view(
            torch.int32
        )

    w1, s1 = weights(2 * inter, dim)
    w2, s2 = weights(dim, inter)
    expert_ones = torch.ones(experts, device=device)
    x = torch.randn(
        (rows, dim), generator=generator, device=device, dtype=torch.bfloat16
    )
    q, q_scale = mxfp8_quantize(x, is_sf_swizzled_layout=True)
    router = torch.full((rows, top_k), 1.0 / top_k, device=device)
    selected = torch.arange(top_k, device=device, dtype=torch.int32).repeat(rows, 1)
    workspace = torch.empty(workspace_bytes, device=device, dtype=torch.uint8)
    output = torch.empty_like(x)

    def run():
        return provider.run(
            input=q,
            input_sf=q_scale,
            token_selected_experts=selected,
            token_final_scales=router,
            fc1_expert_weights=w1,
            fc2_expert_weights=w2,
            output_dtype=output.dtype,
            quant_scales=[s1, expert_ones, s2, expert_ones],
            swiglu_limit=torch.full_like(expert_ones, 1.5),
            output=output,
            workspace_buffer=workspace,
            enable_pdl=False,
            tune_max_num_tokens=rows,
            **mode,
        )

    output.fill_(float("nan"))
    assert run() is output
    first = output.clone()
    assert torch.isfinite(first).all()
    assert torch.count_nonzero(first).item() > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    q_other, scale_other = mxfp8_quantize(-x, is_sf_swizzled_layout=True)
    q.copy_(q_other)
    q_scale.copy_(scale_other)
    selected.add_(1)
    run()
    expected = output.clone()
    assert not torch.equal(first, expected)
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


if __name__ == "__main__":
    test_hybrid_tuning_buckets_round_up()
    test_provider_normalizes_into_caller_output()
    test_separate_finalize_graph_replays_into_caller_buffers()
    print("PASS test_separate_finalize_graph_replays_into_caller_buffers")
