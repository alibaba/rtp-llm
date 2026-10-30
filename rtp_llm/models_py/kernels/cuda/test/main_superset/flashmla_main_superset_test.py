"""Real-wheel regression for the explicit FlashMLA main/K3 API contracts.

These tests launch installed native operators; they do not load a source checkout
or replace operators with mocks. V4.1 quantization follows FlashMLA
49e20002f72f50309cf0a984a8dc56e49d453610/tests/quant.py and
bdd41db11f4a7de284608ac3820a92bd23ad4a5e/tests/quant.py. The V3.2 reference
and BF16 dequantization follow bdd41db/tests/test_sparse_decode_fp8_kv.py.
The small, standalone helpers below keep the test independent of those trees.
"""

import hashlib
import importlib
import importlib.metadata
import inspect
import pickle
import platform
import sys
from pathlib import Path

import pytest
import torch

ARM_LEGACY_VERSION = "1.0.0+92fd68b.rtp1"
# Native member of the main ARM wheel whose archive SHA256 is
# 9a0d330ee0875d7bbca8772f7467a561c31627583373c1d5c06c7e183810ec7a.
ARM_LEGACY_NATIVE_SHA256 = (
    "01ff7395898268012f05013be299b863607a38c11d008dd3e58de3abfd84455b"
)


def _require_cuda():
    if torch.version.hip is not None or not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU and the real CUDA FlashMLA wheels")


def _require_sm100():
    _require_cuda()
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        pytest.skip(
            f"V4.1 and metadata-less V3.2 kernels require SM100/SM103; got {capability}"
        )


@pytest.fixture(scope="module")
def apis():
    _require_cuda()
    # Missing packages or symbols on a CUDA test host are failures, not skips.
    root = importlib.import_module("flash_mla")
    current = importlib.import_module("flash_mla.current")
    legacy = importlib.import_module("flash_mla.legacy")
    build = importlib.import_module("flash_mla._build_info")
    assert build.BACKEND == "cuda"
    return root, current, legacy, build


def test_real_wheel_api_and_architecture_default(apis):
    root, current, legacy, build = apis
    machine = platform.machine().lower()
    aliases = {"arm64": "aarch64", "amd64": "x86_64"}
    assert aliases.get(build.CPU_ARCH, build.CPU_ARCH) == aliases.get(machine, machine)
    expected = "legacy" if machine in ("x86_64", "amd64") else "current"
    assert build.DEFAULT_API == expected
    selected = legacy if expected == "legacy" else current
    for name in selected.__all__:
        assert getattr(root, name) is getattr(selected, name), name

    root_interface = importlib.import_module("flash_mla.flash_mla_interface")
    assert root_interface.flash_mla_with_kvcache is selected.flash_mla_with_kvcache
    for api in (current, legacy):
        interface = importlib.import_module(api.__name__ + ".flash_mla_interface")
        metadata, placeholder = api.get_mla_metadata()
        assert placeholder is None
        assert type(metadata) is interface.FlashMLASchedMeta
        assert type(pickle.loads(pickle.dumps(metadata))) is type(metadata)
        for name in api.__all__:
            assert hasattr(api, name), (api.__name__, name)
        fused = api.fused_norm_rope_attn_rope_cast
        for name in ("prefill", "decode", "permute_q_b_proj", "permute_wv_proj"):
            assert callable(getattr(fused, name)), (api.__name__, name)

    modern_signature = inspect.signature(current.flash_mla_with_kvcache)
    old_signature = inspect.signature(legacy.flash_mla_with_kvcache)
    assert modern_signature.parameters["is_fp8_kvcache"].default is True
    assert old_signature.parameters["is_fp8_kvcache"].default is False
    assert "enable_batch_invariant" in modern_signature.parameters
    assert "enable_batch_invariant" not in old_signature.parameters
    assert callable(legacy.flash_mla_sparse_decode_fp8_kv)
    assert callable(legacy.flash_mla_sparse_fp8_fwd)

    modern_native = importlib.import_module("flash_mla.current.cuda")
    old_native = importlib.import_module("flash_mla.legacy.cuda")
    assert importlib.import_module("flash_mla.cuda") is (
        old_native if expected == "legacy" else modern_native
    )
    for native in (modern_native, old_native):
        for name in (
            "sparse_decode_fwd",
            "sparse_prefill_fwd",
            "dense_prefill_fwd",
            "dense_prefill_bwd",
            "fused_norm_rope_attn_rope_cast_fwd",
            "fused_norm_rope_attn_rope_cast_decode",
            "permute_q_b_proj",
            "permute_wv_proj",
        ):
            assert callable(getattr(native, name)), (native.__name__, name)
    assert callable(old_native.sparse_decode_fp8_kv_fwd)
    assert old_native.sparse_decode_fwd is not modern_native.sparse_decode_fwd


def test_arm_keeps_exact_main_92_native_and_coexists(apis):
    _, current, legacy, build = apis
    requirements = importlib.metadata.requires("flash_mla") or []
    standalone_requirements = [
        item.replace(" ", "").replace("_", "-").lower()
        for item in requirements
        if item.replace("_", "-").lower().startswith("flash-mla-legacy")
    ]
    if build.CPU_ARCH not in ("aarch64", "arm64"):
        # An independently installed legacy distribution on x86 is optional.
        assert standalone_requirements == []
        return
    assert standalone_requirements == [f"flash-mla-legacy=={ARM_LEGACY_VERSION}"]
    assert importlib.metadata.version("flash-mla-legacy") == ARM_LEGACY_VERSION
    standalone = importlib.import_module("flash_mla_legacy")
    standalone_native = importlib.import_module("flash_mla_legacy.cuda")
    assert standalone_native.__file__.endswith(".so")
    native_hash = hashlib.sha256(
        Path(standalone_native.__file__).read_bytes()
    ).hexdigest()
    assert native_hash == ARM_LEGACY_NATIVE_SHA256
    assert standalone.get_mla_metadata is not legacy.get_mla_metadata
    assert standalone.get_mla_metadata is not current.get_mla_metadata
    for name in ("get_mla_metadata", "flash_mla_with_kvcache", "flash_mla_sparse_fwd"):
        assert callable(getattr(standalone, name)), name
    for name in ("dense_decode_fwd", "sparse_decode_fwd", "sparse_prefill_fwd"):
        assert callable(getattr(standalone_native, name)), name


def _pack_v41(payload, scale_bytes, layout):
    blocks, block_size, payload_size = payload.shape
    scale_size = scale_bytes.shape[-1]
    token_bytes = payload_size + scale_size
    if layout == "current":
        packed = torch.empty(
            (blocks, block_size, token_bytes), dtype=torch.uint8, device="cuda"
        )
        packed[..., :payload_size] = payload
        packed[..., payload_size:] = scale_bytes
    else:
        # 49e2000 layout: the entire page's data precedes the page's scales.
        page_bytes = block_size * token_bytes
        padded_bytes = (page_bytes + 511) // 512 * 512
        page = torch.empty((blocks, padded_bytes), dtype=torch.uint8, device="cuda")
        used = page[:, :page_bytes]
        used[:, : block_size * payload_size] = payload.reshape(blocks, -1)
        used[:, block_size * payload_size :] = scale_bytes.reshape(blocks, -1)
        packed = used.view(blocks, block_size, token_bytes)
    return packed.unsqueeze(2)


def _v41_cache(x, layout, fp4=False):
    blocks, block_size, dimension = x.shape
    assert dimension == 512
    group_size = 16 if fp4 else 32
    grouped = x.float().view(blocks, block_size, -1, group_size)
    amax = grouped.abs().amax(dim=-1)
    if fp4:
        scales = (amax / 6.0).clamp(2.0**-9, 448.0).to(torch.float8_e4m3fn)
        normalized = grouped / scales.float().unsqueeze(-1)
        magnitudes = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda"
        )
        midpoints = (magnitudes[:-1] + magnitudes[1:]) / 2
        absolute = normalized.abs().clamp_max(6.0).contiguous()
        codes = torch.bucketize(absolute, midpoints, right=True)
        on_tie = (absolute.unsqueeze(-1) == midpoints).any(dim=-1)
        lower = torch.bucketize(absolute, midpoints, right=False)
        codes = torch.where(on_tie, lower + (lower & 1), codes)
        codes = codes.to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) << 3)
        decoded = magnitudes[(codes & 7).long()]
        decoded = torch.where((codes & 8) != 0, -decoded, decoded)
        dequantized = (decoded * scales.float().unsqueeze(-1)).to(torch.bfloat16)
        codes = codes.view(blocks, block_size, dimension)
        payload = codes[..., 0::2] | (codes[..., 1::2] << 4)
        scale_bytes = scales.view(torch.uint8)
    else:
        scales = torch.pow(2.0, (amax / 448.0).clamp_min(1e-4).log2().ceil())
        quantized = (grouped / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
        dequantized = quantized.to(torch.bfloat16) * scales.to(
            torch.bfloat16
        ).unsqueeze(-1)
        payload = quantized.view(blocks, block_size, dimension).view(torch.uint8)
        # All scales are finite powers of two: UE8M0 is exponent + bias 127.
        scale_bytes = (scales.log2().to(torch.int16) + 127).to(torch.uint8)
    return (
        _pack_v41(payload, scale_bytes, layout),
        dequantized.view(blocks * block_size, dimension),
    )


def _v32_cache(x):
    blocks, block_size, dimension = x.shape
    assert dimension == 576
    packed = torch.empty((blocks, block_size, 656), dtype=torch.uint8, device="cuda")
    grouped = x[..., :512].float().view(blocks, block_size, 4, 128)
    scales = grouped.abs().amax(dim=-1).clamp_min(1e-6) / 448.0
    quantized = (grouped / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    packed[..., :512] = quantized.view(blocks, block_size, 512).view(torch.uint8)
    packed[..., 512:528] = scales.contiguous().view(torch.uint8)
    packed[..., 528:] = x[..., 512:].contiguous().view(torch.uint8)
    # Native V3.2 dequantization rounds the FP32 scale to BF16 first.
    nope = quantized.to(torch.bfloat16) * scales.to(torch.bfloat16).unsqueeze(-1)
    decoded = torch.cat((nope.view(blocks, block_size, 512), x[..., 512:]), dim=-1)
    return packed.unsqueeze(2), decoded.view(-1, 576)


def _reference(q, pools, softmax_scale):
    """FP32 attention over exactly the packed cache's BF16 decoded values."""
    batch, query_length, heads, dimension = q.shape
    gathered_parts, valid_parts = [], []
    for values, indices, lengths in pools:
        topk = indices.shape[-1]
        valid = indices >= 0
        if lengths is not None:
            if lengths.numel() == batch:
                lengths = lengths.view(batch, 1).expand(batch, query_length)
            lengths = lengths.reshape(batch, query_length)
            positions = torch.arange(topk, device="cuda").view(1, 1, topk)
            valid = valid & (positions < lengths.unsqueeze(-1))
        gathered = values[indices.clamp_min(0).long()].float()
        gathered_parts.append(gathered)
        valid_parts.append(valid)
    keys = torch.cat(gathered_parts, dim=2)
    valid = torch.cat(valid_parts, dim=2)
    logits = torch.einsum("bshd,bskd->bshk", q.float(), keys) * softmax_scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    lse = logits.logsumexp(dim=-1)
    probabilities = torch.nan_to_num(logits.softmax(dim=-1), nan=0.0)
    out = torch.einsum("bshk,bskd->bshd", probabilities, keys[..., :512])
    assert out.shape == (batch, query_length, heads, 512)
    assert keys.shape[-1] == dimension
    return out.to(torch.bfloat16), lse.transpose(1, 2)


def _check_attention(actual, expected):
    output, lse = actual
    reference_output, reference_lse = expected
    assert output.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    assert output.shape == reference_output.shape
    assert lse.shape == reference_lse.shape
    assert torch.isfinite(output).all()
    assert torch.equal(torch.isneginf(lse), torch.isneginf(reference_lse))
    finite = torch.isfinite(reference_lse)
    torch.testing.assert_close(
        lse[finite], reference_lse[finite], atol=1e-3, rtol=8.01 / 65536
    )
    # Same FP32-reference cosine gate as the upstream V3.2 correctness harness.
    # Elementwise relative error is unstable near cancellation and zero.
    a, r = output.float().flatten(), reference_output.float().flatten()
    denominator = a.norm() * r.norm()
    if denominator.item() == 0:
        assert torch.equal(output, reference_output)
    else:
        cosine = (torch.dot(a, r) / denominator).item()
        assert cosine > 1 - 5e-6, f"attention output cosine={cosine:.9f}"
        # Cosine alone cannot detect a uniformly mis-scaled output.
        relative_l2 = ((a - r).norm() / r.norm()).item()
        assert relative_l2 < 5e-3, f"attention output relative L2={relative_l2:.6f}"
    empty = torch.isneginf(reference_lse).transpose(1, 2)
    assert torch.count_nonzero(output[empty]).item() == 0


def _indices(batch, query_length, topk, slots):
    indices = torch.randint(
        slots, (batch, query_length, topk), dtype=torch.int32, device="cuda"
    )
    # Only -1 is a valid invalid-slot sentinel in the current V4.1 contract.
    indices[..., 11::23] = -1
    return indices


@pytest.mark.parametrize("layout", ["legacy", "current"])
@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("extra_fp4", [False, True], ids=["fp8-extra", "fp4-extra"])
@pytest.mark.parametrize("topk", [128, 512], ids=["no-split", "split-kv"])
def test_v41_explicit_layout_native_decode(apis, layout, heads, extra_fp4, topk):
    _require_sm100()
    root, current, legacy, build = apis
    torch.manual_seed(1729)
    api = legacy if layout == "legacy" else current
    batch, query_length, blocks, block_size = 2, 1, 4, 64
    q = torch.randn(
        batch, query_length, heads, 512, dtype=torch.bfloat16, device="cuda"
    )
    # Token-dependent amplitude makes the two incompatible scale placements
    # observably different, even though both formats have the same tensor shape.
    gain = torch.linspace(0.25, 2.0, block_size, device="cuda").view(1, -1, 1)
    primary = torch.randn(blocks, block_size, 512, device="cuda") * gain
    secondary = torch.randn_like(primary) * gain.flip(1)
    cache, decoded = _v41_cache(primary, layout)
    extra_cache, extra_decoded = _v41_cache(secondary, layout, fp4=extra_fp4)
    indices = _indices(batch, query_length, topk, blocks * block_size)
    extra_topk = 128 if topk == 128 else 256
    extra_indices = _indices(batch, query_length, extra_topk, blocks * block_size)
    lengths = torch.tensor([topk, topk - 33], dtype=torch.int32, device="cuda")
    extra_lengths = torch.tensor(
        [extra_topk - 17, extra_topk], dtype=torch.int32, device="cuda"
    )
    metadata, num_splits = api.get_mla_metadata()
    kwargs = dict(
        num_splits=num_splits,
        softmax_scale=512**-0.5,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=lengths,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_lengths,
    )
    actual = api.flash_mla_with_kvcache(q, cache, None, None, 512, metadata, **kwargs)
    expected = _reference(
        q,
        [(decoded, indices, lengths), (extra_decoded, extra_indices, extra_lengths)],
        512**-0.5,
    )
    _check_attention(actual, expected)
    # Root must execute the old bytes on x86 and the new bytes on ARM. Use
    # fresh metadata; layout selection never depends on shape or request data.
    if build.DEFAULT_API == layout:
        default_metadata, _ = root.get_mla_metadata()
        default_actual = root.flash_mla_with_kvcache(
            q, cache, None, None, 512, default_metadata, **kwargs
        )
        torch.testing.assert_close(default_actual, actual, atol=0, rtol=0)


@pytest.mark.parametrize("heads", [64, 96, 128])
@pytest.mark.parametrize("length_shape", ["batch", "row", "flat"])
def test_k3_metadataless_dynamic_topk_cuda_graph(apis, heads, length_shape):
    _require_sm100()
    _, _, legacy, _ = apis
    torch.manual_seed(2718)
    batch, query_length, topk, blocks, block_size = 2, 2, 256, 4, 64
    q = torch.randn(
        batch, query_length, heads, 576, dtype=torch.bfloat16, device="cuda"
    )
    source = torch.randn(blocks, block_size, 576, dtype=torch.bfloat16, device="cuda")
    cache, decoded = _v32_cache(source)
    indices = _indices(batch, query_length, topk, blocks * block_size)
    shape = (batch,) if length_shape == "batch" else (batch, query_length)
    if length_shape == "flat":
        shape = (batch * query_length,)
    lengths = torch.full(shape, topk, dtype=torch.int32, device="cuda")
    output = torch.empty(
        batch, query_length, heads, 512, dtype=torch.bfloat16, device="cuda"
    )
    lse = torch.empty(
        batch, query_length, heads, dtype=torch.float32, device="cuda"
    ).transpose(1, 2)

    def launch():
        result = legacy.flash_mla_sparse_decode_fp8_kv(
            q, cache, indices, lengths, out=output, softmax_lse=lse
        )
        assert result[0].data_ptr() == output.data_ptr()
        assert result[1].data_ptr() == lse.data_ptr()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    pointer = lengths.data_ptr()
    cases = (
        [[0, 0], [0, 0]],
        [[32, 32], [173, 173]],
        [[256, 256], [256, 256]],
        [[0, 97], [256, 31]],
    )
    for case in cases:
        changed = torch.tensor(case, dtype=torch.int32, device="cuda")
        if length_shape == "batch":
            changed = changed[:, 0]
        lengths.copy_(changed.reshape(shape))
        assert lengths.data_ptr() == pointer
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference(q, [(decoded, indices, lengths)], 576**-0.5)
        _check_attention((output, lse), expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
