"""Native MLA dependency fallback and real decode/verify query grouping."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    glm53_native_sparse as native,
)


@pytest.mark.parametrize(
    "rows,requests,tokens",
    [(48, 48, 1), (80, 80, 1), (192, 48, 4), (240, 48, 5), (288, 48, 6)],
)
def test_query_length_uses_verify_metadata(rows, requests, tokens):
    assert (
        native.native_query_length(
            SimpleNamespace(is_target_verify=True), rows, requests
        )
        == tokens
    )
    assert (
        native.native_query_length(
            SimpleNamespace(is_target_verify=False), rows, requests
        )
        == 1
    )
    assert native.native_query_length(None, rows, requests) == 1
    assert (
        native.native_query_length(SimpleNamespace(is_target_verify=True), rows, 0) == 1
    )


def test_missing_optional_library_keeps_fallback():
    native._load_native.cache_clear()
    with patch.object(torch.ops, "load_library") as load:
        assert native._load_native(None) is None
        assert native._load_native("/nonexistent/glm53_native_sparse.so") is None
        load.assert_not_called()


def test_disabled_or_unsupported_arch_does_not_load_extension():
    with (
        patch.object(native, "_ENABLED", False),
        patch.object(native, "_load_native") as load,
    ):
        assert native.Glm53NativeSparse().native is None
        load.assert_not_called()
    with (
        patch.object(native, "_ENABLED", True),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(10, 3)),
        patch.object(native, "_load_native") as load,
    ):
        assert native.Glm53NativeSparse().native is None
        load.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows,tokens", [(80, 1), (192, 4), (240, 5), (288, 6)])
def test_native_query_shape_and_scheduler_cache(rows, tokens):
    op = native.Glm53NativeSparse.__new__(native.Glm53NativeSparse)
    op.schedulers = {}
    calls = []
    metadata, splits = object(), object()

    def record(query, cache, indices, *args):
        calls.append((query.shape, indices.shape, args[2], args[3]))
        return query.clone(), None, metadata, splits

    op.native = record
    q = torch.randn(rows, 64, 512, device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(1, 128, 528, device="cuda", dtype=torch.uint8)
    indices = torch.full((rows, 1, 2176), -1, device="cuda", dtype=torch.int32)
    for _ in range(2):
        out = op.forward(q, cache, indices, 0.125, tokens_per_request=tokens)
        torch.testing.assert_close(out, q, rtol=0, atol=0)
    assert calls[0] == (
        (rows // tokens, tokens, 64, 512),
        (rows // tokens, tokens, 2176),
        None,
        None,
    )
    assert calls[1][2:] == (metadata, splits)
    assert op.forward(q, cache, indices, 0.125, tokens_per_request=7) is None
    assert (
        op.forward(q, cache, indices.long(), 0.125, tokens_per_request=tokens) is None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unsupported_geometry_is_rejected_before_index_padding():
    op = native.Glm53NativeSparse.__new__(native.Glm53NativeSparse)
    op.native = object()
    q = torch.empty(48, 64, 512, device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(1, 128, 528, device="cuda", dtype=torch.uint8)
    assert op.supports(q, cache)
    assert not op.supports(q[:, :8], cache)
    assert not op.supports(q[:, :, :256], cache)
    assert not op.supports(
        torch.empty(385, 64, 512, device="cuda", dtype=torch.bfloat16), cache
    )
    assert not op.supports(q, cache[:, :64])
    op.native = None
    assert not op.supports(q, cache)
