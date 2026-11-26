import sys
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
from flashinfer import block_scale_interleave, mxfp8_quantize
from flashinfer.tllm_enums import ActivationType

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import sm120_decode
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_grouped_fp4 import (
    _GraphMoeOps,
    bind_graph_moe,
    sm120_graph_fp4_available,
)


def _fake_deep_gemm(implementation=None, version="2.8.2+a5165ad.cu132"):
    return SimpleNamespace(
        __version__=version,
        m_grouped_fp8_fp4_gemm_nt_contiguous=(
            Mock() if implementation is None else implementation
        ),
    )


def test_provider_normalizes_into_caller_output():
    workspace_size = Mock(return_value=4)
    output = torch.empty(1, dtype=torch.bfloat16)
    workspace = torch.empty(4, dtype=torch.uint8)

    def write_output(*, output, **kwargs):
        output.fill_(2)
        return [output]

    execute = Mock(side_effect=write_output)
    provider = _GraphMoeOps(workspace_size, execute)
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
    assert sm120_graph_fp4_available(device)
    provider = bind_graph_moe(device)
    assert bind_graph_moe(torch.device("cuda", torch.cuda.current_device())) is provider
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
    swiglu_limit = torch.full_like(expert_ones, 1.5)
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
            swiglu_limit=swiglu_limit,
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


def test_bound_grouped_adapter_preserves_output_and_rejects_psum():
    device = torch.device("cuda", 1)
    data = SimpleNamespace(device=device)
    packed_scale = torch.empty(1, dtype=torch.int32)
    activations = weights = (data, packed_scale)
    layout = torch.empty(1, dtype=torch.int32)
    output = torch.ones(1, dtype=torch.bfloat16)
    native = Mock(side_effect=lambda _a, _b, out, _layout, **_: out.fill_(2))

    sm120_decode._grouped_gemm_for_device.cache_clear()
    try:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=1),
            patch.object(
                torch.cuda,
                "get_device_capability",
                side_effect=lambda index: (12, 0) if index == 1 else (12, 1),
            ),
            patch.dict(sys.modules, {"deep_gemm": _fake_deep_gemm(native)}),
        ):
            assert sm120_decode._has_sm120_grouped_fp8_fp4_kernel(torch.device("cuda"))
            op = sm120_decode._sm120_grouped_op(device)
            assert sm120_decode._sm120_grouped_op(torch.device("cuda")) is op
            assert not sm120_decode._has_sm120_grouped_fp8_fp4_kernel(
                torch.device("cuda", 0)
            )
    finally:
        sm120_decode._grouped_gemm_for_device.cache_clear()

    assert op(activations, weights, output=output, grouped_layout=layout) is None
    assert native.call_args.args[2] is output
    assert output.item() == 2
    assert native.call_args.kwargs["recipe"] is None
    assert native.call_args.kwargs["recipe_a"] is None
    assert native.call_args.kwargs["recipe_b"] is None
    assert native.call_args.kwargs["compiled_dims"] == "nk"
    assert native.call_args.kwargs["disable_ue8m0_cast"] is False

    recipes = dict(recipe_a=(1, 128), recipe_b=(1, 32), disable_ue8m0_cast=True)
    op(activations, weights, output, layout, **recipes)
    assert all(native.call_args.kwargs[key] == value for key, value in recipes.items())
    native.reset_mock()

    for options in ({"use_psum_layout": True}, {"expected_m_for_psum_layout": 16}):
        with TestCase().assertRaisesRegex(ValueError, "psum"):
            op(activations, weights, output, layout, **options)
    with TestCase().assertRaisesRegex(RuntimeError, "prepacked int32 scales"):
        op((data, torch.ones(1)), weights, output, layout)
    native.assert_not_called()


def test_grouped_adapter_requires_deep_gemm_sm120_support():
    sm120_decode._grouped_gemm_for_device.cache_clear()
    try:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
        ):
            device = torch.device("cuda")
            with patch.dict(sys.modules, {"deep_gemm": None}):
                assert not sm120_decode._has_sm120_grouped_fp8_fp4_kernel(device)
            with patch.dict(
                sys.modules,
                {"deep_gemm": _fake_deep_gemm(version="2.8.1")},
            ):
                assert not sm120_decode._has_sm120_grouped_fp8_fp4_kernel(device)
            with patch.dict(
                sys.modules,
                {
                    "deep_gemm": SimpleNamespace(
                        __version__="2.8.2",
                        m_grouped_fp8_fp4_gemm_nt_contiguous=None,
                    )
                },
            ):
                assert not sm120_decode._has_sm120_grouped_fp8_fp4_kernel(device)
            with patch.dict(
                sys.modules, {"deep_gemm": _fake_deep_gemm(version="2.8.2")}
            ):
                assert sm120_decode._has_sm120_grouped_fp8_fp4_kernel(device)
    finally:
        sm120_decode._grouped_gemm_for_device.cache_clear()


def test_grouped_decode_accumulates_into_caller_output():
    device = torch.device("cuda")
    op = sm120_decode._sm120_grouped_op(device)
    m, n, k = 128, 128, 128
    activation = torch.ones((m, k), device=device).to(torch.float8_e4m3fn)
    # Packed UE8M0 scales of 1.0 in the TMA-aligned MN-major layout.
    activation_scales = torch.empty_strided(
        (m, 1), (1, m), device=device, dtype=torch.int32
    ).fill_(0x7F)
    # Two E2M1 values of +1.0 in each byte.
    weight = torch.full((1, n, k // 2), 0x22, device=device, dtype=torch.int8)
    weight_scales = torch.empty_strided(
        (1, n, 1), (n, 1, n), device=device, dtype=torch.int32
    ).fill_(0x7F7F7F7F)
    output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    groups = torch.zeros((m,), device=device, dtype=torch.int32)

    result = op(
        (activation, activation_scales),
        (weight, weight_scales),
        output,
        groups,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )

    assert result is None
    assert torch.all(output == k)


if __name__ == "__main__":
    test_provider_normalizes_into_caller_output()
    test_separate_finalize_graph_replays_into_caller_buffers()
    test_bound_grouped_adapter_preserves_output_and_rejects_psum()
    test_grouped_adapter_requires_deep_gemm_sm120_support()
    test_grouped_decode_accumulates_into_caller_output()
    print("PASS test_sm120_moe_bindings")
