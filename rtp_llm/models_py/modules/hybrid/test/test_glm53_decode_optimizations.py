"""Independent precision controls and applicability of the GLM53 decode port."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.glm53_router import Glm53FP32Router
from rtp_llm.models_py.modules import glm53_mega_mhc as mhc


def test_router_default_and_explicit_fp32(monkeypatch):
    monkeypatch.delenv("GLM53_ROUTER_BF16_PARTS", raising=False)
    weight = torch.randn(4096, 288) * 0.01
    assert Glm53FP32Router(weight).split_parts == 2
    monkeypatch.setenv("GLM53_ROUTER_BF16_PARTS", "0")
    router = Glm53FP32Router(weight)
    assert router.split_parts == 0
    assert not hasattr(router, "split_weight_0")
    x = torch.randn(7, 4096).bfloat16()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.testing.assert_close(
            router(x), F.linear(x.float(), weight.T.contiguous()), rtol=0, atol=0
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    # Other model geometry does not allocate BF16 weight components.
    monkeypatch.delenv("GLM53_ROUTER_BF16_PARTS")
    assert not hasattr(Glm53FP32Router(torch.randn(32, 8)), "split_weight_0")


def test_missing_mega_mhc_assets_keep_fallback(monkeypatch):
    monkeypatch.setenv("GLM53_MEGA_MHC_LIBRARY_DIR", "/nonexistent/mega_mhc")
    mhc._get_library.cache_clear()
    try:
        assert mhc._get_library() is None
    finally:
        mhc._get_library.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_router_disabled_and_unsupported_layout_are_fp32(monkeypatch):
    torch.backends.cuda.matmul.allow_tf32 = False
    weight = torch.randn(4096, 288, device="cuda") * 0.01
    x = torch.randn(48, 8192, device="cuda").bfloat16()[:, ::2]
    monkeypatch.delenv("GLM53_ROUTER_BF16_PARTS", raising=False)
    router = Glm53FP32Router(weight)
    torch.testing.assert_close(
        router(x), F.linear(x.float(), router.weight), rtol=0, atol=0
    )
    monkeypatch.setenv("GLM53_ROUTER_BF16_PARTS", "0")
    router = Glm53FP32Router(weight)
    torch.testing.assert_close(
        router(x.contiguous()), F.linear(x.float(), router.weight), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mega_mhc_abi_and_disable_guards():
    x = torch.empty(48, 4096, device="cuda", dtype=torch.bfloat16)
    residual = torch.empty(48, 4, 4096, device="cuda", dtype=torch.bfloat16)
    post = torch.empty(48, 4, 1, device="cuda")
    comb = torch.empty(48, 4, 4, device="cuda")
    unit = SimpleNamespace(
        hc_mult=4,
        dim=4096,
        fn=torch.empty(24, 16384, device="cuda"),
        scale=torch.empty(3, device="cuda"),
        base=torch.empty(24, device="cuda"),
    )
    norm = SimpleNamespace(
        weight=torch.empty(4096, device="cuda", dtype=torch.bfloat16)
    )
    with (
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        patch.object(mhc, "_ENABLED", True),
        patch.object(mhc, "_get_library", return_value=object()) as load,
    ):
        assert mhc.can_use_mega_mhc(x, residual, post, comb, unit, norm)
        with patch.object(mhc, "_ENABLED", False):
            assert not mhc.can_use_mega_mhc(x, residual, post, comb, unit, norm)
        assert not mhc.can_use_mega_mhc(
            x[:47], residual[:47], post[:47], comb[:47], unit, norm
        )
        assert not mhc.can_use_mega_mhc(x, residual, post.bfloat16(), comb, unit, norm)
        with patch.object(unit, "hc_mult", 8):
            assert not mhc.can_use_mega_mhc(x, residual, post, comb, unit, norm)
        assert load.call_count == 1


@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_precision_switches_are_read_at_import(monkeypatch, flag):
    import importlib.util
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
        glm53_native_sparse,
    )

    for module, env in (
        (mhc, "GLM53_MEGA_MHC"),
        (glm53_native_sparse, "GLM53_NATIVE_SPARSE"),
    ):
        monkeypatch.delenv(env, raising=False)
        if flag is not None:
            monkeypatch.setenv(env, flag)
        spec = importlib.util.spec_from_file_location(
            "isolated_precision_control", module.__file__
        )
        isolated = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(isolated)
        assert isolated._ENABLED == (flag != "0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [48, 192, 240, 288])
@pytest.mark.parametrize("ties", [False, True])
def test_splitk_router_epilogue_matches_native_routes(monkeypatch, rows, ties):
    from rtp_llm.models_py.modules import GroupTopK

    monkeypatch.delenv("GLM53_ROUTER_BF16_PARTS", raising=False)
    monkeypatch.delenv("GLM53_ROUTER_SPLIT_K", raising=False)
    monkeypatch.delenv("GLM53_ROUTER_REDUCE_TOPK", raising=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(547809 + rows)
    weight = torch.randn(4096, 288, device="cuda") * 0.01
    if ties:
        weight.zero_()
    router = Glm53FP32Router(weight)
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(288, device="cuda")
    expected_weights = torch.empty(rows, 8, device="cuda")
    expected_ids = torch.empty(rows, 8, device="cuda", dtype=torch.int32)
    GroupTopK()(
        topk_weights=expected_weights,
        topk_ids=expected_ids,
        scores=router(x),
        correction_bias=bias,
        n_group=1,
        topk_group=1,
        topk=8,
        renormalize=True,
        routed_scaling_factor=2.5,
    )
    weights, ids = torch.empty_like(expected_weights), torch.empty_like(expected_ids)
    assert router.forward_routes(x, weights, ids, bias, 1, 1, 8, True, 2.5)
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=1e-7)
    # Unsupported routing groups and output layouts keep GroupTopK fallback.
    assert not router.forward_routes(x, weights, ids, bias, 2, 1, 8, True, 2.5)
    assert not router.forward_routes(x, weights[:, ::2], ids, bias, 1, 1, 8, True, 2.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("width", [0, 13, 1100])
def test_skipping_topk_fill_preserves_short_rows_and_padding(monkeypatch, topk, width):
    from rtp_llm.models_py.modules.dsv4.fp8.indexer import _run_decode_topk

    monkeypatch.setenv("DSV4_INDEXER_TOPK_CANONICALIZE", "1")
    logits = torch.randn(5, width, device="cuda")
    lengths = torch.tensor(
        [0, 1, 7, 100, 1100], device="cuda", dtype=torch.int32
    ).clamp_max(width)
    expected = torch.full((5, topk), 123456, device="cuda", dtype=torch.int32)
    actual = torch.full_like(expected, 234567)
    monkeypatch.setenv("GLM53_INDEXER_SKIP_TOPK_FILL", "0")
    _run_decode_topk(logits, lengths, expected, "topk_v3_tie_break")
    monkeypatch.setenv("GLM53_INDEXER_SKIP_TOPK_FILL", "1")
    _run_decode_topk(logits, lengths, actual, "topk_v3_tie_break")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for row, length in enumerate(lengths.cpu().tolist()):
        assert torch.all(actual[row, min(length, topk) :] == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_incompatible_mega_mhc_binding_keeps_fallback(tmp_path, monkeypatch):
    for relative in (
        "csrc/jit_kernels/impls/sm100_mega_mhc.hpp",
        "include/deep_jit/utils/parser.hpp",
        "mhc_isolated.so",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setenv("GLM53_MEGA_MHC_LIBRARY_DIR", str(tmp_path))
    fake = SimpleNamespace(
        init=lambda root: None
    )  # Old API misses set_num_sms/set_pdl.
    mhc._get_library.cache_clear()
    try:
        with (
            patch.object(mhc.importlib.util, "module_from_spec", return_value=fake),
            patch.object(
                mhc.importlib.util,
                "spec_from_file_location",
                return_value=SimpleNamespace(
                    loader=SimpleNamespace(exec_module=lambda m: None)
                ),
            ),
        ):
            assert mhc._get_library() is None
    finally:
        mhc._get_library.cache_clear()
