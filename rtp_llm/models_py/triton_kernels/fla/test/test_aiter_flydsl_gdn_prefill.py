"""Numerical tests for the AITER FlyDSL GDN prefill adapter."""

import pytest
import torch

from rtp_llm.models_py.triton_kernels.fla.aiter_flydsl_gdn_prefill import (
    _get_aiter_flydsl_gdn_prefill_ops,
    _is_aiter_flydsl_gdn_prefill_disabled,
    build_aiter_flydsl_gdn_prefill_metadata,
    chunk_gated_delta_rule_aiter_flydsl_with_intermediate_states,
    is_aiter_flydsl_gdn_prefill_supported,
)
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="AITER FlyDSL GDN prefill requires ROCm",
)


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_abs_error: float = 5e-2,
    min_cosine: float = 0.998,
) -> None:
    error = (actual.float() - expected.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert torch.isfinite(actual).all(), f"{name} contains non-finite values"
    assert error.mean().item() < 1e-3, (name, "mean", error.mean().item())
    assert error.max().item() < max_abs_error, (name, "max", error.max().item())
    assert cosine.item() > min_cosine, (name, "cosine", cosine.item())


def test_pinned_aiter_api_passes_support_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the complete adapter API and a supported production head shape."""
    tokens, key_heads, value_heads, dim = 64, 16, 32, 128
    q = torch.empty(1, tokens, key_heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty(1, tokens, value_heads, dim, device="cuda", dtype=torch.bfloat16)
    g = torch.empty(1, tokens, value_heads, device="cuda", dtype=torch.float32)
    beta = torch.empty_like(g)

    monkeypatch.delenv("DISABLE_AITER_FLYDSL_GDN_PREFILL", raising=False)
    _is_aiter_flydsl_gdn_prefill_disabled.cache_clear()
    _get_aiter_flydsl_gdn_prefill_ops.cache_clear()
    try:
        assert _get_aiter_flydsl_gdn_prefill_ops() is not None
        assert is_aiter_flydsl_gdn_prefill_supported(q, k, v, g, beta)
    finally:
        _is_aiter_flydsl_gdn_prefill_disabled.cache_clear()
        _get_aiter_flydsl_gdn_prefill_ops.cache_clear()


@pytest.mark.parametrize(
    "cu_values,use_initial_state,snapshot_dtype",
    [
        ([0, 128], False, torch.bfloat16),
        ([0, 256], True, torch.float32),
        ([0, 73, 256], True, torch.bfloat16),
        ([0, 73, 256], True, torch.float32),
    ],
)
def test_flydsl_k1_k5_matches_rtp(
    cu_values: list[int],
    use_initial_state: bool,
    snapshot_dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260824)
    tokens = cu_values[-1]
    key_heads, value_heads, dim = 16, 32, 128
    q = torch.randn(1, tokens, key_heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, value_heads, dim, device="cuda", dtype=torch.bfloat16)
    g = -torch.nn.functional.softplus(
        torch.randn(1, tokens, value_heads, device="cuda", dtype=torch.float32)
    )
    beta = torch.sigmoid(
        torch.randn(1, tokens, value_heads, device="cuda", dtype=torch.float32)
    )
    cu_seqlens = torch.tensor(cu_values, device="cuda", dtype=torch.int32)
    initial_state = None
    if use_initial_state:
        initial_state = (
            torch.randn(
                len(cu_values) - 1,
                value_heads,
                dim,
                dim,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.01
        )

    expected_o, expected_h, expected_final = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    metadata = build_aiter_flydsl_gdn_prefill_metadata(
        tuple(
            cu_values[index + 1] - cu_values[index]
            for index in range(len(cu_values) - 1)
        ),
        cu_seqlens,
    )
    actual_o, actual_h, actual_final = (
        chunk_gated_delta_rule_aiter_flydsl_with_intermediate_states(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            state_dtype=torch.float32,
            snapshot_dtype=snapshot_dtype,
            prefill_metadata=metadata,
            use_qk_l2norm_in_kernel=True,
        )
    )
    torch.cuda.synchronize()

    assert actual_h.dtype == snapshot_dtype
    assert actual_final is not None and expected_final is not None
    assert actual_final.dtype == torch.float32
    _assert_close("output", actual_o, expected_o)
    _assert_close("chunk state", actual_h, expected_h.to(snapshot_dtype))
    _assert_close("final state", actual_final, expected_final)


def test_shape_gate_rejects_non_bf16_inputs() -> None:
    q = torch.empty(1, 64, 2, 128, device="cuda", dtype=torch.float32)
    k = torch.empty_like(q)
    v = torch.empty(1, 64, 8, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.empty(1, 64, 8, device="cuda", dtype=torch.float32)
    beta = torch.empty_like(g)
    assert not is_aiter_flydsl_gdn_prefill_supported(q, k, v, g, beta)


def test_disable_env_forces_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    q = torch.empty(1, 64, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty(1, 64, 8, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.empty(1, 64, 8, device="cuda", dtype=torch.float32)
    beta = torch.empty_like(g)

    monkeypatch.delenv("DISABLE_AITER_FLYDSL_GDN_PREFILL", raising=False)
    _is_aiter_flydsl_gdn_prefill_disabled.cache_clear()
    assert not _is_aiter_flydsl_gdn_prefill_disabled()

    monkeypatch.setenv("DISABLE_AITER_FLYDSL_GDN_PREFILL", "1")
    _is_aiter_flydsl_gdn_prefill_disabled.cache_clear()
    try:
        assert not is_aiter_flydsl_gdn_prefill_supported(q, k, v, g, beta)
    finally:
        _is_aiter_flydsl_gdn_prefill_disabled.cache_clear()


@pytest.mark.parametrize("transposed_input", ["g", "beta"])
def test_shape_gate_rejects_transposed_gate_inputs(transposed_input: str) -> None:
    """Reject same-numel layouts that reshape would silently reinterpret."""
    q = torch.empty(1, 64, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty(1, 64, 8, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.empty(1, 64, 8, device="cuda", dtype=torch.float32)
    beta = torch.empty_like(g)
    if transposed_input == "g":
        g = g.transpose(1, 2)
    else:
        beta = beta.transpose(1, 2)

    assert not is_aiter_flydsl_gdn_prefill_supported(q, k, v, g, beta)


def test_varlen_requires_reusable_metadata() -> None:
    q = torch.empty(1, 64, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty(1, 64, 8, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.empty(1, 64, 8, device="cuda", dtype=torch.float32)
    beta = torch.empty_like(g)
    cu_seqlens = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="requires prefill_metadata"):
        chunk_gated_delta_rule_aiter_flydsl_with_intermediate_states(
            q,
            k,
            v,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            state_dtype=torch.float32,
            snapshot_dtype=torch.float32,
        )


def test_output_final_state_false_returns_no_final_state() -> None:
    tokens, key_heads, value_heads, dim = 64, 2, 8, 128
    q = torch.randn(1, tokens, key_heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, value_heads, dim, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, tokens, value_heads, device="cuda", dtype=torch.float32)
    beta = torch.rand_like(g)

    _, _, final_state = chunk_gated_delta_rule_aiter_flydsl_with_intermediate_states(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        state_dtype=torch.float32,
        snapshot_dtype=torch.float32,
    )
    torch.cuda.synchronize()
    assert final_state is None


def test_repeated_key_slow_decay_matches_rtp() -> None:
    """Cover the padded-prompt pattern that exposed K1-K4 precision loss."""
    tokens, key_heads, value_heads, dim = 128, 16, 32, 128
    q = torch.zeros(1, tokens, key_heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    q[..., 0] = 1
    k[..., 0] = 1
    generator = torch.Generator(device="cuda").manual_seed(20260824)
    v = (
        torch.randn(
            1,
            tokens,
            value_heads,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.2
    )
    g = torch.full((1, tokens, value_heads), -1e-3, device="cuda")
    beta = torch.full((1, tokens, value_heads), 0.835, device="cuda")
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)

    expected = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    metadata = build_aiter_flydsl_gdn_prefill_metadata((tokens,), cu_seqlens)
    actual = chunk_gated_delta_rule_aiter_flydsl_with_intermediate_states(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_dtype=torch.float32,
        snapshot_dtype=torch.float32,
        prefill_metadata=metadata,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    for name, actual_tensor, expected_tensor in zip(
        ("output", "chunk state", "final state"), actual, expected, strict=True
    ):
        assert actual_tensor is not None and expected_tensor is not None
        # The persistent FP32 state accumulates BF16 w/u rounding across two
        # chunks. Keep the model-visible output and snapshots on the default
        # strict bound while allowing that localized state accumulation.
        is_final_state = name == "final state"
        _assert_close(
            name,
            actual_tensor,
            expected_tensor,
            max_abs_error=7.5e-2 if is_final_state else 5e-2,
            min_cosine=0.994 if is_final_state else 0.998,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
