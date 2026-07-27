import pytest
import torch

from rtp_llm.models_py.triton_kernels.fla.aiter_flydsl_decode import (
    aiter_flydsl_gdn_decode,
    prepare_aiter_flydsl_gdn_decode_state_indices,
)
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="AITER FlyDSL GDN decode requires ROCm",
)


def test_prepare_decode_indices_honors_noncontiguous_block_map_stride():
    padded_block_map = torch.tensor(
        [[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]],
        device="cuda",
        dtype=torch.int32,
    )
    block_map = padded_block_map[:, :3]
    assert block_map.stride() == (5, 1)
    sequence_lengths_plus_1 = torch.tensor(
        [1002, 1025], device="cuda", dtype=torch.int32
    )

    read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
        block_map, sequence_lengths_plus_1, 1024
    )
    torch.cuda.synchronize()

    assert read_indices.cpu().tolist() == [1, 4]
    assert write_indices.cpu().tolist() == [1, 5]


@pytest.mark.parametrize("sequence_length", [1001, 1024])
@pytest.mark.parametrize(
    ("key_heads", "value_heads"),
    [
        pytest.param(2, 8, id="small-head-config"),
        pytest.param(16, 32, id="qwen35-4b-tp1"),
    ],
)
@pytest.mark.parametrize(
    "state_dtype",
    [
        pytest.param(torch.float32, id="state-fp32"),
        pytest.param(torch.bfloat16, id="state-bf16"),
    ],
)
def test_aiter_flydsl_gdn_decode_matches_triton_reference(
    sequence_length: int,
    key_heads: int,
    value_heads: int,
    state_dtype: torch.dtype,
):
    pytest.importorskip("aiter.ops.flydsl.linear_attention_kernels")

    torch.manual_seed(17)
    device = "cuda"
    # CUDA Graph decode commonly pads several requests into one graph batch.
    # Cover that serving shape rather than testing only B=1.
    batch, dim = 4, 128
    q = torch.randn(batch, 1, key_heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, 1, value_heads, dim, device=device, dtype=torch.bfloat16)
    a = torch.randn(batch, value_heads, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.randn(value_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(value_heads, device=device, dtype=torch.bfloat16)

    block_map = torch.arange(
        1, batch * 3 + 1, device=device, dtype=torch.int32
    ).reshape(batch, 3)
    sequence_lengths_plus_1 = torch.tensor(
        [sequence_length + 1] * batch, device=device, dtype=torch.int32
    )

    # RTP packs convolution state after each SSM state. Preserve that larger
    # pool stride so the test covers the real non-contiguous cache layout.
    state_elements = value_heads * dim * dim
    packed_state = (
        torch.randn(
            batch * 3 + 1,
            state_elements + 12288,
            device=device,
            dtype=state_dtype,
        )
        * 0.01
    )
    packed_reference = packed_state.clone()
    packed_flydsl = packed_state.clone()
    state_reference = packed_reference[:, :state_elements].view(
        batch * 3 + 1, value_heads, dim, dim
    )
    state_flydsl = packed_flydsl[:, :state_elements].view(
        batch * 3 + 1, value_heads, dim, dim
    )
    assert state_flydsl.stride(0) > state_elements

    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    output_reference, _ = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g.view(batch, 1, value_heads),
        beta=beta.view(batch, 1, value_heads),
        initial_state=state_reference,
        block_map=block_map,
        sequence_lengths=sequence_lengths_plus_1,
        seq_size_per_block=1024,
        use_qk_l2norm_in_kernel=True,
    )
    read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
        block_map, sequence_lengths_plus_1, 1024
    )
    output_flydsl = aiter_flydsl_gdn_decode(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        state=state_flydsl,
        read_indices=read_indices,
        write_indices=write_indices,
        copy_state=sequence_length % 1024 == 0,
    )
    torch.cuda.synchronize()

    write_ids = write_indices.to(torch.int64)
    for name, actual, expected in (
        ("output", output_flydsl.float(), output_reference.float()),
        (
            "state",
            state_flydsl[write_ids].float(),
            state_reference[write_ids].float(),
        ),
    ):
        diff = (actual - expected).abs()
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        )
        assert torch.isfinite(actual).all(), name
        assert diff.mean().item() < 2e-4, (name, diff.mean().item())
        assert diff.max().item() < 2e-2, (name, diff.max().item())
        assert cosine.item() > 0.9999, (name, cosine.item())
