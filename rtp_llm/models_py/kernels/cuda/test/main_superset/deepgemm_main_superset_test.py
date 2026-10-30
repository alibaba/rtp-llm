"""Exercise the installed DeepGEMM main contracts and the separate K3 MQA path."""

import platform
import sys

import deep_gemm
import pytest
import torch
from deep_gemm.testing import calc_diff
from deep_gemm.utils import (
    cast_back_from_fp4,
    cast_back_from_fp8,
    per_block_cast_to_fp8,
    per_token_cast_to_fp4,
    per_token_cast_to_fp8,
)


@pytest.fixture(scope="module")
def arch():
    assert torch.cuda.is_available(), "This target requires a CUDA GPU"
    major = torch.cuda.get_device_capability()[0]
    assert major in (9, 10, 12), "Route this target to SM90, SM100/103 or SM12x"
    return major


@pytest.fixture(autouse=True)
def exact_fp32_reference(arch):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _is_arm():
    return platform.machine().lower() in ("aarch64", "arm64")


def _values(shape, phase):
    size = 1
    for extent in shape:
        size *= extent
    # Quarter-integers make the dequantized dot-product reference exact in FP32.
    values = torch.arange(size, device="cuda").reshape(shape)
    return ((values * 3 + phase) % 13 - 6).float() / 4


def _mqa_operand(shape, fmt, phase):
    values = _values(shape, phase)
    if fmt == "fp8":
        packed = values.to(torch.float8_e4m3fn)
        return (packed, None), packed.float()
    head_dim = shape[-1]
    rows = values.numel() // head_dim
    quantize = per_token_cast_to_fp4 if fmt == "mxfp4" else per_token_cast_to_fp8
    dequantize = cast_back_from_fp4 if fmt == "mxfp4" else cast_back_from_fp8
    packed, scales = quantize(
        values.reshape(rows, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    decoded = dequantize(packed, scales, gran_k=32, use_packed_ue8m0=True).reshape(
        shape
    )
    packed_dim = head_dim // 2 if fmt == "mxfp4" else head_dim
    operand = (packed.reshape(*shape[:-1], packed_dim), scales.reshape(shape[:-1]))
    return operand, decoded


def _kv_operand(shape, fmt, phase):
    operand, decoded = _mqa_operand(shape, fmt, phase)
    if fmt != "fp8":
        return operand, decoded
    # Legacy E4M3 has a per-token FP32 KV scale and no Q scale.
    scales = torch.pow(2.0, (torch.arange(shape[0], device="cuda") % 3 - 2).float())
    return (operand[0], scales), decoded * scales[:, None]


def _mqa_reference(q, kv, weights):
    scores = torch.einsum("mhd,nd->mhn", q.float(), kv.float()).relu()
    return (scores * weights.float()[:, :, None]).sum(1)


def _assert_logits(actual, reference, dtype):
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    # The SM100 v2 kernel reduces head scores in BF16; legacy FP32 keeps FP32 reduction.
    rtol, atol = (0.008, 0.02) if dtype == torch.bfloat16 else (2e-4, 1e-4)
    torch.testing.assert_close(actual.float(), reference.float(), rtol=rtol, atol=atol)
    if actual.numel():
        assert calc_diff(actual.float(), reference.float()) < (
            3e-5 if dtype == torch.bfloat16 else 5e-6
        )


def _paged_inputs(fmt, heads, next_n):
    batch, page, blocks_per_request, head_dim = 2, 64, 8, 128
    num_blocks = batch * blocks_per_request
    q, q_ref = _mqa_operand((batch, next_n, heads, head_dim), fmt, 1)
    kv, kv_ref = _kv_operand((num_blocks * page, head_dim), fmt, 2)
    head_bytes = kv[0].size(-1)
    payload = kv[0].view(torch.uint8).reshape(num_blocks, page * head_bytes)
    scales = kv[1].contiguous().view(torch.uint8).reshape(num_blocks, page * 4)
    cache = torch.cat((payload, scales), dim=1).reshape(
        num_blocks, page, 1, head_bytes + 4
    )
    table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).reshape(
        batch, blocks_per_request
    )
    return q, q_ref, cache, kv_ref.reshape(num_blocks, page, head_dim), table


def _check_paged(out, q_ref, kv_ref, table, lengths, weights, dtype):
    batch, next_n = lengths.shape
    assert out.shape == (batch * next_n, 512)
    for request in range(batch):
        keys = kv_ref.index_select(0, table[request].long()).reshape(512, -1)
        reference = _mqa_reference(
            q_ref[request], keys, weights[request * next_n : (request + 1) * next_n]
        )
        for token in range(next_n):
            end = int(lengths[request, token])
            _assert_logits(
                out[request * next_n + token, :end], reference[token, :end], dtype
            )


def test_installed_native_exports_and_architecture_contract(arch):
    # Missing wheel capabilities must fail directly.
    required = (
        deep_gemm.fp8_gemm_nt_skip_head_mid,
        deep_gemm.get_bf16_mega_gate_config,
        deep_gemm.bf16_mega_gate,
        deep_gemm.epilogue.FP8Quantization,
        deep_gemm.fp8_fp4_mqa_logits_main_compat,
        deep_gemm.get_mqa_logits_metadata_main_compat,
        deep_gemm.fp8_fp4_paged_mqa_logits_main_compat,
        deep_gemm.get_paged_mqa_logits_metadata_main_compat,
        deep_gemm.get_legacy_paged_mqa_logits_metadata,
        deep_gemm.fp8_fp4_mqa_logits_v2,
        deep_gemm.get_mqa_logits_metadata_v2,
        deep_gemm.fp8_fp4_paged_mqa_logits_v2,
        deep_gemm.get_paged_mqa_logits_metadata_v2,
    )
    assert all(callable(function) for function in required)
    expected = (
        deep_gemm.fp8_fp4_mqa_logits_v2
        if _is_arm()
        else deep_gemm.fp8_fp4_mqa_logits_main_compat
    )
    assert deep_gemm.fp8_fp4_mqa_logits is expected


def test_fp8_skip_head_mid_preserves_gaps_and_padding(arch):
    torch.manual_seed(31)
    m, n, k = 17, 256, 256
    use_ue8m0 = arch in (10, 12)
    a = per_token_cast_to_fp8(
        torch.randn((m, k), device="cuda", dtype=torch.bfloat16), use_ue8m0=use_ue8m0
    )
    b = per_block_cast_to_fp8(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16), use_ue8m0=use_ue8m0
    )
    a_ref = a[0].float() * a[1].repeat_interleave(128, dim=1)
    b_ref = b[0].float() * b[1].repeat_interleave(128, dim=0).repeat_interleave(
        128, dim=1
    )
    reference = a_ref @ b_ref.T
    storage = torch.full((m + 2, 448), 23, device="cuda", dtype=torch.bfloat16)
    out = storage[1 : m + 1, :384]
    deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, out, (64, 64, 64), recipe=(1, 128, 128))
    actual = out.view(m, 2, 192)
    selected = torch.cat((actual[:, :, :64], actual[:, :, 128:]), dim=-1).reshape(m, n)
    assert calc_diff(selected.float(), reference) < 1e-5
    assert torch.all(actual[:, :, 64:128] == 23)
    assert torch.all(storage[[0, -1]] == 23)
    assert torch.all(storage[:, 384:] == 23)


@pytest.mark.parametrize("scoring", ["identity", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_mega_gate_main_scoring_modes_and_default(arch, scoring, with_bias):
    if arch != 10:
        pytest.skip("MegaGate requires SM100/SM103")
    torch.manual_seed(41)
    x = torch.randn((17, 512), device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn((128, 512), device="cuda") * 0.02).bfloat16()
    bias = torch.randn((128,), device="cuda") * 0.01 if with_bias else None
    indices, weights = deep_gemm.bf16_mega_gate(
        x, weight, 4, False, 0, 1.0, 0, scoring_func=scoring, bias=bias
    )
    scores = x.float() @ weight.float().T
    if scoring == "sigmoid":
        scores = scores.sigmoid()
    elif scoring == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(scores).sqrt()
    expected_indices = (
        (scores if bias is None else scores + bias).topk(4, dim=-1).indices
    )
    expected_weights = scores.gather(1, expected_indices)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    assert torch.equal(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights, atol=1e-4, rtol=1e-4)
    if scoring == ("sqrtsoftplus" if _is_arm() else "identity"):
        default = deep_gemm.bf16_mega_gate(x, weight, 4, False, 0, 1.0, 0, bias=bias)
        assert torch.equal(default[0], indices)
        assert torch.equal(default[1], weights)
    config = deep_gemm.get_bf16_mega_gate_config(17, 512, 128, 4)
    assert set(config) == {
        "block_tokens",
        "num_mma_ctas",
        "num_split_k",
        "num_expert_groups",
        "num_gate_warpgroups",
        "num_sms",
    }
    assert all(value > 0 for value in config.values())


@pytest.mark.parametrize("fmt", ["fp8", "mxfp8", "mxfp4"])
def test_main_mqa_full_clean_fp32_and_metadata(arch, fmt):
    if arch == 9 and fmt != "fp8":
        pytest.skip("SM90 has no MX MQA")
    if arch == 12 and fmt == "mxfp8":
        pytest.skip("SM12x dense MQA supports FP8 with FP32 KV scales and MXFP4")
    m, heads, dim, tokens = 8, 32, 128, 512
    q, q_ref = _mqa_operand((m, heads, dim), fmt, 1)
    kv, kv_ref = _kv_operand((tokens, dim), fmt, 2)
    weights = torch.full((m, heads), 0.125, device="cuda")
    starts = torch.arange(m, device="cuda", dtype=torch.int32) * 3
    ends = starts + 385
    # Only SM100 has this optional contiguous scheduler. SM12x explicitly disallows it.
    metadata = (
        deep_gemm.get_mqa_logits_metadata_main_compat(starts, ends, tokens, heads)
        if arch == 10
        else None
    )
    out = deep_gemm.fp8_fp4_mqa_logits_main_compat(
        q, kv, weights, starts, ends, True, 0, torch.float32, metadata
    )
    reference = _mqa_reference(q_ref, kv_ref, weights)
    columns = torch.arange(tokens, device="cuda")[None, :]
    valid = (columns >= starts[:, None]) & (columns < ends[:, None])
    assert out.shape == (m, tokens) and out.dtype == torch.float32
    assert torch.all(torch.isneginf(out[~valid]))
    _assert_logits(out[valid], reference[valid], torch.float32)
    if not _is_arm():
        public = deep_gemm.fp8_fp4_mqa_logits(q, kv, weights, starts, ends)
        assert torch.all(torch.isneginf(public[~valid]))
        _assert_logits(public[valid], reference[valid], torch.float32)
    if arch == 12:
        modern = deep_gemm.fp8_fp4_mqa_logits_v2(
            q,
            kv,
            weights,
            starts,
            ends,
            max_seqlen_k=0,
            clean_logits=True,
            logits_dtype=torch.float32,
        )
        assert torch.all(torch.isneginf(modern[~valid]))
        _assert_logits(modern[valid], reference[valid], torch.float32)


def test_main_paged_mqa_fixed_request_rows_fp32_and_256_metadata(arch):
    heads = 32
    q, q_ref, cache, kv_ref, table = _paged_inputs("fp8", heads, next_n=2)
    weights = torch.full((4, heads), 0.125, device="cuda")
    lengths = torch.tensor([[257, 509], [383, 447]], device="cuda", dtype=torch.int32)
    metadata = deep_gemm.get_paged_mqa_logits_metadata_main_compat(
        lengths, 64, deep_gemm.get_num_sms()
    )
    out = deep_gemm.fp8_fp4_paged_mqa_logits_main_compat(
        q, cache, weights, lengths, table, metadata, 512, False, torch.float32
    )
    _check_paged(out, q_ref, kv_ref, table, lengths, weights, torch.float32)
    if arch in (9, 10):
        legacy = deep_gemm.get_legacy_paged_mqa_logits_metadata(
            lengths, 64, deep_gemm.get_num_sms()
        )
        legacy_out = deep_gemm.fp8_fp4_paged_mqa_logits_main_compat(
            q, cache, weights, lengths, table, legacy, 512, False, torch.float32
        )
        _check_paged(legacy_out, q_ref, kv_ref, table, lengths, weights, torch.float32)
    if arch == 12:
        modern_meta = deep_gemm.get_paged_mqa_logits_metadata_v2(
            lengths, 64, deep_gemm.get_num_sms()
        )
        modern = deep_gemm.fp8_fp4_paged_mqa_logits_v2(
            q,
            cache,
            weights,
            lengths,
            table,
            modern_meta,
            512,
            logits_dtype=torch.float32,
        )
        _check_paged(modern, q_ref, kv_ref, table, lengths, weights, torch.float32)


@pytest.mark.parametrize("fmt", ["fp8", "mxfp8", "mxfp4"])
def test_modern_mqa_v2_compressed_and_paged_pair(arch, fmt):
    if arch == 9:
        pytest.skip("This case checks v2 MQA on SM100/103 and SM12x")
    if arch == 10 and fmt == "fp8":
        pytest.skip("SM100/103 v2 MQA requires MX Q scales")
    if arch == 12 and fmt == "mxfp8":
        pytest.skip("SM12x dense MQA supports FP8 with FP32 KV scales and MXFP4")
    m, heads, dim, tokens = 8, 32, 128, 512
    q, q_ref = _mqa_operand((m, heads, dim), fmt, 1)
    kv, kv_ref = _kv_operand((tokens, dim), fmt, 2)
    dtype = torch.bfloat16 if arch == 10 else torch.float32
    weights = torch.full((m, heads), 0.125, device="cuda", dtype=dtype)
    starts = torch.arange(m, device="cuda", dtype=torch.int32) * 3
    ends = starts + 385
    metadata = (
        deep_gemm.get_mqa_logits_metadata_v2(starts, ends, tokens, heads)
        if arch == 10
        else None
    )
    kwargs = {} if arch == 10 else {"logits_dtype": dtype}
    out = deep_gemm.fp8_fp4_mqa_logits_v2(
        q,
        kv,
        weights,
        starts,
        ends,
        max_seqlen_k=385,
        schedule_meta=metadata,
        **kwargs,
    )
    reference = _mqa_reference(q_ref, kv_ref, weights)
    assert out.shape == (m, 385)
    for row in range(m):
        start = int(starts[row])
        _assert_logits(out[row], reference[row, start : start + 385], dtype)
    q, q_ref, cache, kv_ref, table = _paged_inputs(fmt, heads, next_n=1)
    lengths = torch.tensor([[509], [447]], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    weights = torch.full((2, heads), 0.125, device="cuda", dtype=dtype)
    paged_meta = deep_gemm.get_paged_mqa_logits_metadata_v2(
        lengths, 64, deep_gemm.get_num_sms(), indices=indices
    )
    paged = deep_gemm.fp8_fp4_paged_mqa_logits_v2(
        q, cache, weights, lengths, table, paged_meta, 512, indices=indices, **kwargs
    )
    _check_paged(paged, q_ref, kv_ref, table, lengths, weights, dtype)


@pytest.mark.parametrize("zero_output", [False, True])
def test_fp8_quantization_preserves_unclamped_native_policy(arch, zero_output):
    if arch != 10:
        pytest.skip("The main FP8 output epilogue requires SM100/SM103")
    batch, heads, reduction, dim = 16, 4, 256, 128
    x = _values((batch, heads, reduction), 1)
    if zero_output:
        x.zero_()
    weight = _values((heads, dim, reduction), 2)
    # Unit FP32 scales make the input products independent of the output quantizer.
    x_fp8 = (x.to(torch.float8_e4m3fn), torch.ones((batch, heads, 2), device="cuda"))
    weight_fp8 = (
        weight.to(torch.float8_e4m3fn),
        torch.ones((heads, 1, 2), device="cuda"),
    )
    reference = torch.einsum("bhr,hdr->bhd", x, weight).bfloat16()
    expected, expected_sf = per_token_cast_to_fp8(
        reference.reshape(batch, heads * dim),
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    output = torch.empty_like(reference, dtype=torch.float8_e4m3fn)
    sf = torch.empty_strided(
        (batch, heads * dim // 128), (1, batch), device="cuda", dtype=torch.int32
    )
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd",
        x_fp8,
        weight_fp8,
        output,
        epilogue=deep_gemm.epilogue.FP8Quantization(sf),
    )
    assert torch.equal(
        output.reshape(batch, heads * dim).view(torch.uint8), expected.view(torch.uint8)
    )
    assert torch.equal(sf.contiguous(), expected_sf)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
