"""KDA Decode precision boundaries, replay routing and fused output regression."""

import unittest
from types import SimpleNamespace as NS

import torch
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.attention.kda_impl import KdaDecode, KdaMetadata
from rtp_llm.models_py.modules.hybrid.kda_attention import KdaAttention
from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d_update
from rtp_llm.models_py.triton_kernels.common.activation import (
    rmsnorm_sigmoid_gate_per_token_group_quant_fp8,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import FusedRMSNormGated
from rtp_llm.models_py.triton_kernels.kda import fused_recurrent_kda
from rtp_llm.models_py.triton_kernels.linear_replay import LinearReplay
from rtp_llm.utils.model_weight import W


def _round_bfloat16_rne_cpu(value):
    """Direct binary64 -> BF16 RNE, without an intervening FP32 rounding.

    Test-only oracle. Preserve signed zero/infinity and canonicalize NaNs.
    The tiny underflow range rounds to zero; the overflow range to infinity.
    """
    assert value.device.type == "cpu" and value.dtype == torch.float64
    bits = value.contiguous().view(torch.int64)
    sign = (bits >> 63) & 1
    exponent = (bits >> 52) & 0x7FF
    fraction = bits & ((1 << 52) - 1)
    unbiased = exponent - 1023
    significand = fraction | (1 << 52)
    normal = (significand + (1 << 44) - 1 + ((significand >> 45) & 1)) >> 45
    payload = ((unbiased + 127) << 7) + normal - 128
    shift = (-unbiased - 81).clamp(46, 54)
    half = torch.ones_like(shift) << (shift - 1)
    subnormal = (significand + half - 1 + ((significand >> shift) & 1)) >> shift
    payload = torch.where(unbiased < -126, subnormal, payload)
    payload = torch.where(unbiased > 127, 0x7F80, payload)
    payload = torch.where((exponent == 0x7FF) & (fraction != 0), 0x7FC0, payload)
    return ((sign << 15) | payload).to(torch.int16).view(torch.bfloat16)


def _metric(actual, expected):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    return float(100 * (actual - expected).norm() / expected.norm().clamp_min(1e-30))


def _norm_reference(core, gate, weight):
    x = core.detach().cpu().double()
    result = (
        x
        * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
        * weight.detach().cpu().double()
        * gate.detach().cpu().double().sigmoid()
    )
    return _round_bfloat16_rne_cpu(result)


def _kda_reference(raw, gate, beta, weight, alog, bias, initial, history):
    raw, gate, beta, weight, alog, bias, state, history = [
        x.detach().cpu().double().clone()
        for x in (raw, gate, beta, weight, alog, bias, initial, history)
    ]
    batch, steps, channels = raw.shape
    heads = channels // 384
    output = []
    for token in range(steps):
        window = torch.cat((history, raw[:, token : token + 1]), dim=1)
        convolution = sum(window[:, tap] * weight[:, tap] for tap in range(4))
        convolution = (convolution * convolution.sigmoid()).float().double()
        q, k, v = [part.reshape(batch, heads, 128) for part in convolution.chunk(3, -1)]
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        decay = (
            -5.0
            * torch.sigmoid(
                alog.exp().reshape(1, heads, 1)
                * (gate[:, token].reshape(batch, heads, 128) + bias.reshape(heads, 128))
            )
        ).exp()
        state = state * decay.unsqueeze(-1)
        innovation = v - torch.einsum("bhk,bhkv->bhv", k, state)
        state = state + k.unsqueeze(-1) * (
            innovation * beta[:, token].sigmoid().unsqueeze(-1)
        ).unsqueeze(-2)
        output.append(torch.einsum("bhk,bhkv->bhv", q * (128**-0.5), state).float())
        history = window[:, 1:]
    return torch.stack(output, dim=1)


def _graph_function(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


def _integration_case(verify, graph_mode):
    batch, heads, dim, blocks, slots = 3, 12, 128, 10, 4
    steps = 4 if verify else 1
    width, channels = heads * dim, 3 * heads * dim
    cfg = NS(
        linear_key_head_dim=dim,
        linear_value_head_dim=dim,
        linear_num_key_heads=heads,
        linear_num_value_heads=heads,
        linear_conv_kernel_dim=4,
        ssm_state_dtype=torch.float32,
        conv_state_dtype=torch.bfloat16,
    )
    parallel = NS(tp_size=1)
    weight = torch.randn(channels, 4, device="cuda") * 0.2
    alog = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads * dim, device="cuda") * 0.1
    decoder = KdaDecode(
        cfg,
        parallel,
        {
            W.linear_attn_conv1d_w: weight[:, None],
            W.linear_attn_alog: alog,
            W.linear_attn_dt_b_kda: bias,
        },
    )
    decoder.gate_lower_bound = -5.0
    block_bytes = decoder.linear_cache_converter.block_size_bytes
    assert block_bytes % torch.tensor([], dtype=torch.bfloat16).element_size() == 0
    base = torch.empty((blocks, block_bytes // 2), device="cuda", dtype=torch.bfloat16)
    states, histories = decoder._get_ssm_states(base), decoder._get_conv_states(base)
    states.copy_(torch.randn_like(states) * 0.03)
    histories.copy_(torch.randn_like(histories) * 0.2)

    def zeros(shape, dtype=torch.float32):
        return torch.zeros(shape, device="cuda", dtype=dtype)

    cache = NS(
        k=zeros((slots, steps, heads, dim)),
        u=zeros((slots, steps, heads, dim)),
        g=zeros((slots, steps, heads, dim)),
        conv_inputs=zeros((slots, steps, channels), torch.bfloat16),
        slot_generations=zeros((slots,), torch.int64),
        log_epochs=zeros((slots,), torch.int64),
        valid_counts=zeros((slots,), torch.int32),
        error_flags=zeros((slots,), torch.int32),
    )
    src = torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int32)
    dst = torch.tensor([[4, 5, 6]], device="cuda", dtype=torch.int32)
    replay = NS(
        slot_ids=torch.tensor([0, 1, -1], device="cuda", dtype=torch.int32),
        slot_generations=torch.ones(batch, device="cuda", dtype=torch.int64),
        prev_accept_lengths=zeros((batch,), torch.int32),
        history_valid_lengths=zeros((batch,), torch.int32),
        history_epochs=zeros((batch,), torch.int64),
        verify_epochs=torch.ones(batch, device="cuda", dtype=torch.int64),
        init_kinds=torch.ones(batch, device="cuda", dtype=torch.int32),
        state_read_block_ids=src,
        active_block_ids=dst,
    )
    table = torch.stack((src[0], dst[0]), 1)
    table[-1].zero_()
    valid = torch.tensor([True, True, False], device="cuda").repeat_interleave(steps)
    inputs = NS(
        is_prefill=False,
        is_mtp_draft_update=False,
        linear_replay=replay if verify else None,
        valid_token_mask=valid,
        prefix_lengths=torch.ones(batch, dtype=torch.int32),
        kv_cache_kernel_block_id_device=table,
        sequence_lengths_plus_1_device=torch.tensor(
            [2, 2, 0], device="cuda", dtype=torch.int32
        ),
    )
    layer_cache = NS(
        kv_cache_base=base,
        group_id=0,
        seq_size_per_block=1,
        linear_replay=cache if verify else None,
    )
    initial, history = states[src[0].long()].clone(), histories[src[0].long()].clone()
    raw = torch.randn(batch, steps, channels, device="cuda", dtype=torch.bfloat16) * 0.3
    gate = torch.randn(batch, steps, heads, dim, device="cuda", dtype=torch.bfloat16)
    forget = torch.randn(batch, steps, heads * dim, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(batch, steps, heads, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.2
    fused = torch.cat(
        (
            raw.flatten(0, 1),
            gate.reshape(batch * steps, width),
            forget.flatten(0, 1),
            beta.flatten(0, 1),
        ),
        -1,
    )
    norm = FusedRMSNormGated(gamma, 1e-5)
    observed = []
    norm.register_forward_pre_hook(
        lambda _, args: observed.append((args[0].dtype, args[1].dtype))
    )
    prefill_shape = NS(
        local_num_k_heads=heads,
        local_num_v_heads=heads,
        head_k_dim=dim,
        head_v_dim=dim,
        parallelism_config=parallel,
    )
    attention = KdaAttention(
        input_projection=torch.nn.Identity(),
        forget_projection=torch.nn.Identity(),
        output_projection=torch.nn.Identity(),
        fa_width=width,
        norm=norm,
        prefill=prefill_shape,
        decode=decoder,
    )
    metadata = KdaMetadata(is_target_verify=verify)

    def run():
        return attention(fused, None, layer_cache, inputs, metadata)

    if graph_mode:
        graph, output = _graph_function(run)
        graph.replay()
    else:
        output = run()
    assert observed and all(x == (torch.float32, torch.bfloat16) for x in observed)
    assert output.dtype == torch.bfloat16
    if verify:
        assert decoder._linear_replay is not None
        assert decoder._linear_replay.output_dtype == torch.float32
    else:
        assert decoder._linear_replay is None
    core = _kda_reference(
        raw,
        forget.reshape(batch, steps, heads, dim),
        beta,
        weight,
        alog,
        bias,
        initial,
        history,
    )
    expected = _norm_reference(core.flatten(0, 1), gate.flatten(0, 1), gamma)
    result = _metric(
        output.reshape(batch, steps, heads, dim)[:2],
        expected.reshape(batch, steps, heads, dim)[:2],
    )
    assert result < 0.12
    torch.testing.assert_close(
        output.reshape(batch, steps, heads, dim)[-1],
        torch.zeros_like(core[-1], device="cuda", dtype=torch.bfloat16),
        atol=0,
        rtol=0,
    )
    if verify:
        legacy_replay = LinearReplay.bind(
            decoder._linear_replay,
            decoder,
            layer_cache,
            vector_gate=True,
            state_v_first=False,
            lower_bound=-5.0,
        )
        assert legacy_replay is not decoder._linear_replay
        assert legacy_replay.output_dtype is None
        legacy_core = legacy_replay(
            raw.flatten(0, 1),
            forget.flatten(0, 1),
            beta.flatten(0, 1),
            replay,
        )
        assert legacy_core.dtype == torch.bfloat16
        assert histories.dtype == cache.conv_inputs.dtype == torch.bfloat16
        assert states.dtype == torch.float32
    return result


class BFloat16ReferenceTest(unittest.TestCase):
    def test_direct_rne_fixed_binary64_cases(self):
        cases = [
            (0x0000000000000000, 0x0000),
            (0x8000000000000000, 0x8000),
            (0x3FF0000000000000, 0x3F80),
            (0xBFF0000000000000, 0xBF80),
            (0x3FF0100000000000, 0x3F80),
            (0x3FF0100000000001, 0x3F81),
            (0x3FF0300000000000, 0x3F82),
            (0xBFF0100000000001, 0xBF81),
            (0x3810000000000000, 0x0080),
            (0x380FC00000000000, 0x007F),
            (0x37A0000000000000, 0x0001),
            (0x3790000000000000, 0x0000),
            (0x3790000000000001, 0x0001),
            (0x380FE00000000000, 0x0080),
            (0x47EFE00000000000, 0x7F7F),
            (0x47EFF00000000000, 0x7F80),
            (0x7FF0000000000000, 0x7F80),
            (0xFFF0000000000000, 0xFF80),
            (0x7FF8000000000001, 0x7FC0),
            (0xFFF8000000000001, 0xFFC0),
            (0x0000000000000001, 0x0000),
            (0x8000000000000001, 0x8000),
        ]
        signed = [bits if bits < (1 << 63) else bits - (1 << 64) for bits, _ in cases]
        value = torch.tensor(signed, dtype=torch.int64).view(torch.float64)
        actual = _round_bfloat16_rne_cpu(value).view(torch.int16).int() & 0xFFFF
        self.assertEqual(actual.tolist(), [expected for _, expected in cases])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class KdaDecodePrecisionRegressionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3119)

    def test_decode_and_target_verify_forward_keep_fp32_until_norm(self):
        # Identity projections supply synthetic, already-projected operands.
        # Cache conversion, normal/replay kernels and gated norm are real.
        for verify in (False, True):
            for graph in (False, True):
                with self.subTest(verify=verify, graph=graph):
                    _integration_case(verify, graph)

    def test_default_kernel_dtypes_and_mapped_fp32_norm(self):
        heads, dim = 2, 128
        channels = 3 * heads * dim
        raw = torch.randn(1, channels, 1, device="cuda", dtype=torch.bfloat16)
        histories = torch.randn(3, channels, 3, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(channels, 4, device="cuda")
        table = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([2], device="cuda", dtype=torch.int32)
        legacy_history, fp32_history = histories.clone(), histories.clone()
        kwargs = dict(
            activation="silu",
            block_map=table,
            sequence_lengths=lengths,
            seq_size_per_block=1,
        )
        legacy = causal_conv1d_update(raw, legacy_history, weight, **kwargs)
        full = causal_conv1d_update(
            raw, fp32_history, weight, output_dtype=torch.float32, **kwargs
        )
        self.assertEqual(legacy.dtype, torch.bfloat16)
        self.assertEqual(full.dtype, torch.float32)
        torch.testing.assert_close(legacy_history, fp32_history, atol=0, rtol=0)
        q, k, v = [
            x.reshape(1, 1, heads, dim).contiguous()
            for x in legacy.transpose(1, 2).chunk(3, -1)
        ]
        state = torch.randn(3, heads, dim, dim, device="cuda") * 0.03
        state32 = state.clone()
        options = dict(
            q=q,
            k=k,
            v=v,
            g=torch.zeros_like(q),
            beta=torch.zeros(1, 1, heads, device="cuda", dtype=torch.bfloat16),
            initial_state=state,
            use_qk_l2norm_in_kernel=True,
            block_map=table,
            sequence_lengths=lengths,
            seq_size_per_block=1,
            state_v_first=False,
            transpose_state_layout=False,
        )
        core_legacy, _ = fused_recurrent_kda(**options)
        options["initial_state"] = state32
        core32, _ = fused_recurrent_kda(**options, output_dtype=torch.float32)
        self.assertEqual(core_legacy.dtype, torch.bfloat16)
        self.assertEqual(core32.dtype, torch.float32)
        self.assertEqual(state32.dtype, torch.float32)
        torch.testing.assert_close(state, state32, atol=0, rtol=0)
        # This cast is FP32 -> BF16, so it has no binary64 double-rounding oracle.
        torch.testing.assert_close(core_legacy, core32.bfloat16(), atol=0, rtol=0)
        norm = FusedRMSNormGated(
            torch.ones(dim, device="cuda", dtype=torch.bfloat16), 1e-5
        )
        gate = torch.zeros(1, heads, dim, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(
            norm(core_legacy.reshape(1, heads, dim), gate).dtype, torch.bfloat16
        )
        mapping = torch.tensor([0, -1], device="cuda", dtype=torch.int64)
        mapped = norm(
            core32.reshape(1, heads, dim),
            gate,
            row_indices=mapping,
            output_dtype=torch.bfloat16,
        )
        self.assertEqual(mapped.dtype, torch.bfloat16)
        torch.testing.assert_close(
            mapped[1], torch.zeros_like(mapped[1]), atol=0, rtol=0
        )

    def test_fp32_fused_norm_keeps_fp8_path_with_graph_and_mask(self):
        torch.manual_seed(19291)
        for dtype in (torch.bfloat16, torch.float32):
            for rows in (1, 9):
                for graph_mode in (False, True):
                    with self.subTest(dtype=dtype, rows=rows, graph=graph_mode):
                        heads, dim = 12, 128
                        x = torch.randn(
                            rows, heads, dim + 1, device="cuda", dtype=dtype
                        )[..., :dim]
                        gate = torch.randn(
                            rows, heads, dim + 2, device="cuda", dtype=torch.bfloat16
                        )[..., :dim]
                        weight = torch.linspace(0.5, 1.5, dim, device="cuda").bfloat16()
                        valid = torch.ones(rows, device="cuda", dtype=torch.bool)
                        if rows > 1:
                            valid[-2:] = False
                        norm = FusedRMSNormGated(weight, 1e-5)
                        normalized = norm(x, gate, output_dtype=torch.bfloat16).reshape(
                            rows, -1
                        )
                        normalized = torch.where(
                            valid[:, None], normalized, 0
                        ).contiguous()
                        expected, expected_scales = sgl_per_token_group_quant_fp8(
                            normalized,
                            group_size=128,
                            eps=1e-4,
                            column_major_scales=True,
                            scale_tma_aligned=True,
                            scale_ue8m0=True,
                        )

                        def run():
                            return rmsnorm_sigmoid_gate_per_token_group_quant_fp8(
                                x, gate, weight, 1e-5, valid_rows=valid
                            )

                        if graph_mode:
                            graph, actual = _graph_function(run)
                            graph.replay()
                        else:
                            actual = run()
                        values, scales = actual
                        # Same native synthetic fixture is exact, including scales.
                        # No CPU FP64 -> BF16 cast is used as a bitwise oracle.
                        torch.testing.assert_close(
                            values.view(torch.uint8),
                            expected.view(torch.uint8),
                            atol=0,
                            rtol=0,
                        )
                        torch.testing.assert_close(
                            scales, expected_scales, atol=0, rtol=0
                        )


if __name__ == "__main__":
    unittest.main()
