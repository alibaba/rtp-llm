"""Numerical and dispatch contracts for the four independent prefill optimizations."""

import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.common.prefill_fusion import (
    GATING,
    METADATA,
    MROPE,
    SIGMOID,
    enabled,
    in_prefill,
    prefill_fusion_scope,
    prefill_quantized_linear,
)
from rtp_llm.models_py.triton_kernels.common.prefill_mrope_cache import (
    maybe_prefill_mrope_cache,
)
from rtp_llm.models_py.triton_kernels.fla.flashinfer_prefill import (
    flashinfer_gdn_prefill,
    prepare_flashinfer_prefill_metadata,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating_prefill import gdn_gating_prefill
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, RopeStyle
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeAttnParams,
    FusedRopeKVCachePrefillOpQKVOut,
    FusedRopeKVCachePrefillOpQOut,
    _get_fused_rope_kvcache,
)


def make_case(lengths, prefixes, fp8=False, explicit=True, qh=4, kh=2):
    cfg = AttentionConfigs()
    cfg.head_num, cfg.kv_head_num, cfg.size_per_head = qh, kh, 256
    cfg.tokens_per_block = cfg.kernel_tokens_per_block = 64
    cfg.max_seq_len = sum(lengths) + max(prefixes) + 64
    cfg.dtype = torch.bfloat16
    cfg.kv_cache_dtype = KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE
    cfg.use_logn_attn = False
    r = cfg.rope_config
    r.style, r.dim, r.base, r.scale = RopeStyle.Mrope, 64, 10000000, 1.0
    r.index_factor = 3
    r.mrope_dim1, r.mrope_dim2, r.mrope_dim3 = 11, 11, 10
    r.mrope_interleaved = True
    cu = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
        device="cuda",
        dtype=torch.int32,
    )
    total = sum(lengths)
    padding = torch.cat(
        [
            torch.full(
                (n,),
                i * max(lengths) - int(sum(lengths[:i])),
                device="cuda",
                dtype=torch.int32,
            )
            for i, n in enumerate(lengths)
        ]
    )
    pos = torch.cat(
        [
            torch.arange(p, p + n, device="cuda", dtype=torch.int32)
            for n, p in zip(lengths, prefixes)
        ]
    )
    pos = torch.stack([pos // 5, pos % 17, pos % 13], -1) if explicit else None
    blocks = [(p + n + 63) // 64 for p, n in zip(prefixes, lengths)]
    pages = sum(blocks) + 3
    table = torch.zeros(len(lengths), max(blocks), dtype=torch.int32)
    cursor = pages - 1
    for i, n in enumerate(blocks):
        table[i, :n] = torch.arange(cursor, cursor - n, -1)
        cursor -= n
    offsets = _get_fused_rope_kvcache().convert_offset_to_block_array(table.cuda())
    params = FusedRopeAttnParams(
        offsets,
        None,
        padding,
        pos,
        cu,
        cu,
        torch.tensor(lengths, dtype=torch.int32).pin_memory(),
        torch.tensor(prefixes, device="cuda", dtype=torch.int32),
        torch.tensor(lengths, device="cuda", dtype=torch.int32),
        max(lengths),
        max(prefixes),
        total + sum(prefixes),
        False,
        torch.bfloat16,
    )
    qkv = (
        torch.randn(total, (qh + 2 * kh) * 256, device="cuda", dtype=torch.bfloat16)
        * 0.25
    )
    cache = torch.full(
        (pages, 2, kh, 64, 256), 0.125, device="cuda", dtype=torch.bfloat16
    )
    if fp8:
        cache = cache.to(torch.float8_e4m3fn)
    scales = torch.full((pages, 2, kh, 64), -7.0, device="cuda") if fp8 else None
    return cfg, params, qkv, SimpleNamespace(kv_cache_base=cache, kv_scale_base=scales)


class PrefillFourTest(unittest.TestCase):
    def test_phase_and_flags(self):
        flags = (SIGMOID, MROPE, GATING, METADATA)
        with patch.dict(os.environ):
            for name in flags:
                os.environ.pop(name, None)
            self.assertTrue(all(enabled(name) for name in flags))
            for name in flags:
                for value in ("0", "false", "off", "no"):
                    with patch.dict(os.environ, {name: value}):
                        self.assertEqual(
                            [enabled(x) for x in flags],
                            [x != name for x in flags],
                        )
        with patch.dict(
            os.environ, {x: "0" for x in (SIGMOID, MROPE, GATING, METADATA)}
        ):
            self.assertFalse(in_prefill())
            with prefill_fusion_scope(True):
                self.assertTrue(in_prefill())
                for name in (SIGMOID, MROPE, GATING, METADATA):
                    with patch.dict(os.environ, {name: "1"}):
                        self.assertEqual(
                            [enabled(x) for x in (SIGMOID, MROPE, GATING, METADATA)],
                            [x == name for x in (SIGMOID, MROPE, GATING, METADATA)],
                        )
                with prefill_fusion_scope(False):
                    self.assertFalse(in_prefill())
                self.assertTrue(in_prefill())
            self.assertFalse(in_prefill())

    def test_automatic_prefill_norm_dispatch(self):
        from rtp_llm.models_py.triton_kernels.common import (
            gated_rmsnorm_prefill as tiled,
        )
        from rtp_llm.models_py.triton_kernels.common import layernorm_gated as norm

        x = torch.ones(2, 128)
        module = norm.RmsNormGated(torch.ones(128))
        for ordinary, supported, expected in (
            (True, True, "tiled"),
            (False, True, "original"),
            (True, False, "original"),
        ):
            with prefill_fusion_scope(ordinary), patch.object(
                tiled, "supports_gated_rmsnorm_prefill", return_value=supported
            ), patch.object(
                tiled, "gated_rmsnorm_prefill", return_value="tiled"
            ) as fused, patch.object(
                norm, "layer_norm_fwd", return_value=("original",)
            ) as original:
                self.assertEqual(module(x, x), expected)
                self.assertEqual(fused.call_count, int(expected == "tiled"))
                self.assertEqual(original.call_count, int(expected == "original"))

    def test_automatic_qk_norm_dispatch(self):
        import importlib
        import inspect

        chunk = importlib.import_module("rtp_llm.models_py.triton_kernels.fla.chunk")
        forward = inspect.unwrap(chunk.ChunkGatedDeltaRuleFunction.forward)
        for ordinary, supported, expected in (
            (True, True, "exact"),
            (False, True, "original"),
            (True, False, "original"),
        ):
            with prefill_fusion_scope(ordinary), patch.object(
                chunk, "is_amd", False
            ), patch.object(
                chunk, "supports_exact_qk_norm", return_value=supported
            ), patch.object(
                chunk, "fused_l2norm_qk_exact", side_effect=RuntimeError("exact")
            ), patch.object(
                chunk, "l2norm_fwd", side_effect=RuntimeError("original")
            ), self.assertRaisesRegex(
                RuntimeError, expected
            ):
                forward(
                    None,
                    object(),
                    object(),
                    None,
                    None,
                    None,
                    1.0,
                    None,
                    False,
                    use_qk_l2norm_in_kernel=True,
                )

    def test_automatic_gating_selects_one_output_mode(self):
        from rtp_llm.models_py.model_desc import qwen3_next as model

        obj = SimpleNamespace(
            alog=object(), dt_bias=object(), head_k_dim=128, head_v_dim=128
        )
        for backend, adapted, supported in (
            ("native", True, True),
            ("flashinfer", True, True),
            ("flashinfer", False, True),
            ("native", True, False),
            ("flashinfer", True, False),
        ):
            with patch.dict(
                os.environ,
                {
                    "RTP_QWEN35_GDN_PREFILL_BACKEND": backend,
                    GATING: "1" if adapted else "0",
                },
            ), prefill_fusion_scope(True), patch.object(
                model, "supports_gdn_gating_prefill", return_value=supported
            ) as supports, patch.object(
                model,
                "gdn_gating_prefill",
                side_effect=RuntimeError("stop after gating"),
            ) as tiled, patch.object(
                model, "fused_gdn_gating", side_effect=RuntimeError("stop after gating")
            ) as original:
                with self.assertRaisesRegex(RuntimeError, "stop after gating"):
                    model.Qwen3NextGatedDeltaNetPrefill._fla(
                        obj, object(), object(), object(), None, 2048, SimpleNamespace()
                    )
                supports.assert_called_once()
                self.assertEqual(tiled.call_count, int(supported))
                self.assertEqual(original.call_count, int(not supported))
                if supported:
                    self.assertEqual(
                        tiled.call_args.kwargs["flashinfer"],
                        backend == "flashinfer" and adapted,
                    )

    def test_default_backend_and_topk(self):
        from rtp_llm.models_py.modules.base.cuda import select_topk
        from rtp_llm.models_py.triton_kernels.common.prefill_fusion import (
            gdn_prefill_backend,
        )

        flag = "RTP_QWEN35_GDN_PREFILL_BACKEND"
        tensor = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, device="cuda")
        with patch.dict(os.environ), patch.object(
            torch.version, "hip", None
        ), patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ) as capability:
            os.environ.pop(flag, None)
            self.assertEqual(gdn_prefill_backend(tensor), "flashinfer")
            self.assertEqual(gdn_prefill_backend(tensor, key_dim=64), "native")
            capability.return_value = (9, 0)
            self.assertEqual(gdn_prefill_backend(tensor), "native")
            capability.return_value = (10, 3)
            tensor.dtype = torch.float16
            self.assertEqual(gdn_prefill_backend(tensor), "native")
            tensor.dtype = torch.bfloat16
            tensor.is_cuda = False
            self.assertEqual(gdn_prefill_backend(tensor), "native")
            for backend in ("native", "flashinfer"):
                os.environ[flag] = backend
                self.assertEqual(gdn_prefill_backend(tensor), backend)

        config = SimpleNamespace(expert_num=512, moe_k=10, has_moe_norm=True)
        with patch.dict(os.environ), patch.object(
            select_topk.compute_ops, "SelectTopkOp"
        ) as op:
            os.environ.pop("RTP_FUSED_TOPK_512", None)
            self.assertTrue(select_topk.SelectTopk(config).fuse_bf16_cast)
            op.assert_called_with(config, use_fused_512=True)
            os.environ["RTP_FUSED_TOPK_512"] = "0"
            self.assertFalse(select_topk.SelectTopk(config).fuse_bf16_cast)
            op.assert_called_with(config, use_fused_512=False)
            os.environ.pop("RTP_FUSED_TOPK_512")
            select_topk.SelectTopk(config, use_fused_512=False)
            op.assert_called_with(config, use_fused_512=False)

    def test_factory_backend_preserved(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            CudaFp8DeepGEMMLinear,
        )
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
            CudaFp8GEMMLinear,
        )

        inner = object.__new__(CudaFp8DeepGEMMLinear)
        torch.nn.Module.__init__(inner)
        inner.scale_ue8m0 = True
        wrapper = object.__new__(CudaFp8GEMMLinear)
        torch.nn.Module.__init__(wrapper)
        wrapper._deepgemm_linear = inner
        for choice in (False, True):
            with patch.object(
                CudaFp8GEMMLinear, "_should_use_flashinfer", return_value=choice
            ):
                self.assertIs(
                    prefill_quantized_linear(wrapper, torch.empty(0)),
                    None if choice else inner,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gating_rounding_and_metadata(self):
        torch.manual_seed(381)
        for lengths in (
            [37],
            [2047],
            [2048],
            [2049],
            [24601],
            [24601, 24601],
            [37, 4096, 20468],
        ):
            n = sum(lengths)
            h = 64
            packed = torch.randn(n, 2 * h, device="cuda", dtype=torch.bfloat16)
            a, b = packed.split(h, -1)
            al = torch.randn(h, device="cuda", dtype=torch.bfloat16)
            dt = torch.randn(h, device="cuda", dtype=torch.bfloat16)
            g, beta = gdn_gating_prefill(al, a, b, dt)
            ge, be = gdn_gating_prefill(al, a, b, dt, flashinfer=True)
            torch.testing.assert_close(ge, g.exp(), atol=0, rtol=0)
            self.assertTrue(torch.equal(be, beta.float()))
            cu = torch.tensor(
                [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                device="cuda",
                dtype=torch.int64,
            )
            for interval in (128, 2048, 4096):
                meta = prepare_flashinfer_prefill_metadata(cu, n, interval)
                ref = torch.cat(
                    [
                        torch.zeros(1, device="cuda", dtype=torch.int64),
                        ((cu[1:] - cu[:-1]) // interval).cumsum(0),
                    ]
                )
                self.assertTrue(torch.equal(meta.checkpoint_starts.long(), ref))
                self.assertTrue(torch.equal(meta.cu_seqlens.long(), cu))
                self.assertEqual(meta.checkpoint_capacity, n // interval)
        # Same shape, changed contents must produce fresh starts on each forward.
        cu = torch.tensor([0, 2047, 4096], device="cuda", dtype=torch.int32)
        before = prepare_flashinfer_prefill_metadata(cu, 4096).checkpoint_starts.clone()
        cu[1] = 2048
        after = prepare_flashinfer_prefill_metadata(cu, 4096).checkpoint_starts
        self.assertFalse(torch.equal(before, after))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_mrope_output_and_entire_cache(self):
        torch.manual_seed(391)
        for lengths, prefix in [
            ([1], [0]),
            ([63, 65], [0, 37]),
            ([2047, 2048, 2049], [65, 0, 128]),
            ([24601], [0]),
            ([24601, 24601], [37, 2048]),
        ]:
            for fp8 in (False, True):
                for qout in (False, True):
                    for explicit in (False, True):
                        with self.subTest(
                            lengths=lengths, fp8=fp8, qout=qout, explicit=explicit
                        ):
                            cfg, params, x, cache = make_case(
                                lengths, prefix, fp8, explicit
                            )
                            y = x.clone()
                            other = SimpleNamespace(
                                kv_cache_base=cache.kv_cache_base.clone(),
                                kv_scale_base=(
                                    cache.kv_scale_base.clone() if fp8 else None
                                ),
                            )
                            op = (
                                FusedRopeKVCachePrefillOpQOut
                                if qout
                                else FusedRopeKVCachePrefillOpQKVOut
                            )(cfg)
                            with patch.dict(
                                os.environ, {MROPE: "0"}
                            ), prefill_fusion_scope(True):
                                ref = op.forward(x, cache, params)
                            with patch.dict(
                                os.environ, {MROPE: "1"}
                            ), prefill_fusion_scope(True):
                                # Production planning lengths live on the host; the kernel
                                # must reuse the already prepared device lengths.
                                fused_params = replace(
                                    params,
                                    prefix_lengths=params.prefix_lengths.cpu().pin_memory(),
                                    prefix_lengths_device=params.prefix_lengths,
                                )
                                actual = op.forward(y, other, fused_params)
                            self.assertIsNotNone(actual)
                            self.assertTrue(
                                torch.equal(actual, ref),
                                f"output mismatch {torch.max((actual-ref).abs()).item()}",
                            )
                            self.assertTrue(torch.equal(x, y), "inplace QKV mismatch")
                            self.assertTrue(
                                torch.equal(
                                    cache.kv_cache_base.view(torch.uint8),
                                    other.kv_cache_base.view(torch.uint8),
                                ),
                                "cache bytes differ",
                            )
                            if fp8:
                                self.assertTrue(
                                    torch.equal(
                                        cache.kv_scale_base, other.kv_scale_base
                                    )
                                )
                            with patch.dict(
                                os.environ, {MROPE: "1"}
                            ), prefill_fusion_scope(False):
                                self.assertIsNone(
                                    maybe_prefill_mrope_cache(
                                        y,
                                        other,
                                        params,
                                        cfg,
                                        qout=qout,
                                        qkvout=not qout,
                                    )
                                )
                            del x, y, ref, actual, cache, other

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_sigmoid_fp8_exact(self):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )
        from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.sigmoid_mul_fp8_quant import (
            _tensors_supported,
            sigmoid_mul_fp8_quant,
        )

        for n in (1, 37, 2047, 2048, 2049, 24601, 49202):
            x = torch.randn(n, 8192, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn_like(x)
            refq, refs = sgl_per_token_group_quant_fp8(
                x * torch.sigmoid(gate),
                128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            q, s = sigmoid_mul_fp8_quant(x, gate)
            self.assertTrue(torch.equal(q.view(torch.uint8), refq.view(torch.uint8)))
            self.assertTrue(torch.equal(s, refs))
            self.assertFalse(
                _tensors_supported(
                    x,
                    torch.empty(n, 16384, device="cuda", dtype=torch.bfloat16)[:, ::2],
                )
            )
            del x, gate, refq, refs, q, s

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_flashinfer_adapter_combinations(self):
        torch.manual_seed(19)
        for lengths in ([37], [2048], [2049], [37, 4096]):
            n = sum(lengths)
            hq, hv = 2, 4
            q = torch.randn(1, n, hq, 128, device="cuda", dtype=torch.bfloat16) * 0.05
            k = torch.randn_like(q)
            v = torch.randn(1, n, hv, 128, device="cuda", dtype=torch.bfloat16)
            a, b = torch.randn(n, 2 * hv, device="cuda", dtype=torch.bfloat16).split(
                hv, -1
            )
            al, dt = torch.zeros(hv, device="cuda", dtype=torch.bfloat16), torch.zeros(
                hv, device="cuda", dtype=torch.bfloat16
            )
            g, beta = gdn_gating_prefill(al, a, b, dt)
            ge, be = gdn_gating_prefill(al, a, b, dt, flashinfer=True)
            cu = torch.tensor(
                [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                device="cuda",
                dtype=torch.int32,
            )
            meta = prepare_flashinfer_prefill_metadata(cu, n)
            ref = flashinfer_gdn_prefill(q, k, v, g, beta, cu)
            for prepared, md in ((True, None), (False, meta), (True, meta)):
                actual = flashinfer_gdn_prefill(
                    q,
                    k,
                    v,
                    ge if prepared else g,
                    be if prepared else beta,
                    cu,
                    gates_prepared=prepared,
                    metadata=md,
                )
                for i in (0, 1):
                    self.assertTrue(torch.equal(actual[i], ref[i]))
                valid = sum(x // 2048 for x in lengths)
                self.assertTrue(torch.equal(actual[2][:valid], ref[2][:valid]))
                self.assertTrue(torch.equal(actual[3].long(), ref[3].long()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_ssm_cache_and_initial_state(self):
        from rtp_llm.models_py.triton_kernels.fla.flashinfer_prefill import (
            store_flashinfer_ssm_state,
        )

        for lengths, interval in (
            ([1, 127, 129], 128),
            ([2047, 2048, 2049], 2048),
            ([24601, 24601], 2048),
            ([37, 4096], 4096),
        ):
            with self.subTest(lengths=lengths, interval=interval):
                n = sum(lengths)
                hq, hv = 2, 4
                q = (
                    torch.randn(1, n, hq, 128, device="cuda", dtype=torch.bfloat16)
                    * 0.05
                )
                k = torch.randn_like(q)
                v = torch.randn(1, n, hv, 128, device="cuda", dtype=torch.bfloat16)
                a, b = torch.randn(
                    n, 2 * hv, device="cuda", dtype=torch.bfloat16
                ).split(hv, -1)
                al, dt = torch.zeros(
                    hv, device="cuda", dtype=torch.bfloat16
                ), torch.zeros(hv, device="cuda", dtype=torch.bfloat16)
                g, beta = gdn_gating_prefill(al, a, b, dt)
                ge, be = gdn_gating_prefill(al, a, b, dt, flashinfer=True)
                cu = torch.tensor(
                    [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                    device="cuda",
                    dtype=torch.int32,
                )
                initial = torch.randn(len(lengths), hv, 128, 128, device="cuda") * 0.01
                meta = prepare_flashinfer_prefill_metadata(cu, n, interval)
                prefix = torch.tensor(
                    [37 + i * interval for i in range(len(lengths))],
                    device="cuda",
                    dtype=torch.int32,
                )
                width = max(
                    (37 + i * interval + x + interval - 1) // interval
                    for i, x in enumerate(lengths)
                )
                pages = len(lengths) * width + 2
                page_map = torch.arange(
                    pages - 1, 1, -1, device="cuda", dtype=torch.int32
                ).reshape(len(lengths), width)
                # Padded cache rows ensure writes respect CACHE_STRIDE and leave all padding untouched.
                storage = torch.full((pages, hv * 128 * 128 + 17), -13.0, device="cuda")
                cache = storage[:, : hv * 128 * 128].view(pages, hv, 128, 128)

                def run(prepared, md, target):
                    result = flashinfer_gdn_prefill(
                        q,
                        k,
                        v,
                        ge if prepared else g,
                        be if prepared else beta,
                        cu,
                        initial_state=initial,
                        checkpoint_interval=interval,
                        gates_prepared=prepared,
                        metadata=md,
                    )
                    store_flashinfer_ssm_state(
                        result[2],
                        result[3],
                        result[1],
                        prefix,
                        cu,
                        page_map,
                        target,
                        interval,
                        n,
                    )
                    return result

                ref = run(False, None, cache)
                for prepared, md in ((True, None), (False, meta), (True, meta)):
                    other = torch.full_like(storage, -13.0)
                    out = run(
                        prepared,
                        md,
                        other[:, : hv * 128 * 128].view(pages, hv, 128, 128),
                    )
                    self.assertTrue(torch.equal(storage, other))
                    for i in (0, 1):
                        self.assertTrue(torch.equal(ref[i], out[i]))
                self.assertTrue(torch.all(storage[:2] == -13.0))
                self.assertTrue(torch.all(storage[:, -17:] == -13.0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_mrope_unsupported_and_no_cache(self):
        from dataclasses import replace

        cfg, params, qkv, cache = make_case([37], [0])
        op = FusedRopeKVCachePrefillOpQKVOut(cfg)
        with patch.dict(os.environ, {MROPE: "0"}), prefill_fusion_scope(True):
            ref = op.forward(qkv.clone(), None, params)
        with patch.dict(os.environ, {MROPE: "1"}), prefill_fusion_scope(True):
            actual = op.forward(qkv.clone(), None, params)
            self.assertTrue(torch.equal(ref, actual))
            for dim in (32, 48, 64, 128, 256):
                cfg.rope_config.dim = dim
                pairs = dim // 2
                cfg.rope_config.mrope_dim2 = (pairs + 1) // 3
                cfg.rope_config.mrope_dim3 = pairs // 3
                cfg.rope_config.mrope_dim1 = (
                    pairs - cfg.rope_config.mrope_dim2 - cfg.rope_config.mrope_dim3
                )
                with patch.dict(os.environ, {MROPE: "0"}):
                    expected = op.forward(qkv.clone(), None, params)
                result = op.forward(qkv.clone(), None, params)
                self.assertTrue(torch.equal(expected, result), dim)
            for data, meta in (
                (qkv.float(), params),
                (qkv, replace(params, decode_plan=True)),
            ):
                self.assertIsNone(
                    maybe_prefill_mrope_cache(
                        data, cache, meta, cfg, qout=False, qkvout=True
                    )
                )
            cfg.rope_config.mrope_interleaved = False
            self.assertIsNone(
                maybe_prefill_mrope_cache(
                    qkv, cache, params, cfg, qout=False, qkvout=True
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_causal_attention_factory_dispatch(self):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0
        from rtp_llm.models_py.modules.factory import LinearFactory
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
            CudaFp8GEMMLinear,
        )
        from rtp_llm.models_py.modules.hybrid.causal_attention import CausalAttention
        from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.env import (
            fusion_phase,
        )
        from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.sigmoid_mul_fp8_quant import (
            sigmoid_mul_fp8_quant,
        )

        # Match the production weight loader's packed, TMA-aligned scale layout.
        weight, scales = requant_weight_ue8m0(
            torch.randn(128, 512, device="cuda").to(torch.float8_e4m3fn),
            torch.ones(1, 4, device="cuda"),
        )
        linear = LinearFactory.create_linear_from_weights(
            {"w": weight, "s": scales},
            "w",
            "s",
            quant_config=SimpleNamespace(get_method=lambda: "FP8_PER_BLOCK"),
        )
        self.assertIsInstance(linear, CudaFp8GEMMLinear)
        x = torch.randn(2049, 512, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(x)
        attn = SimpleNamespace(
            qkv_proj=torch.nn.Identity(),
            qk_fuse_norm=None,
            layer_idx=0,
            o_proj=linear,
            tp_size=1,
        )
        fmha = SimpleNamespace(forward=lambda qkv, cache, layer_idx: qkv)
        with torch.inference_mode():
            reference = linear(x * gate.sigmoid())
            for ordinary, flag, contiguous in (
                (True, "1", True),
                (True, "0", True),
                (False, "1", True),
                (True, "1", False),
            ):
                current_gate = (
                    gate if contiguous else torch.stack((gate, gate), dim=-1)[..., 0]
                )
                with patch.dict(
                    os.environ, {SIGMOID: flag, "RTP_QWEN35_DECODE_FUSION": "0"}
                ), fusion_phase(is_prefill=True), prefill_fusion_scope(ordinary):
                    with patch(
                        "rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.sigmoid_mul_fp8_quant.sigmoid_mul_fp8_quant",
                        wraps=sigmoid_mul_fp8_quant,
                    ) as observed:
                        output = CausalAttention.forward(
                            attn, x, fmha, None, gate=current_gate
                        )
                    self.assertEqual(
                        observed.call_count,
                        int(ordinary and flag == "1" and contiguous),
                    )
                    self.assertTrue(torch.equal(output, reference))

    def test_decoder_scope_excludes_verify_cp_decode(self):
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextDecoderLayer

        obj = SimpleNamespace(_forward_with_phase=lambda *args: in_prefill())
        for prefill, verify, cp in (
            (True, False, False),
            (True, True, False),
            (True, False, True),
            (False, False, False),
        ):
            meta = SimpleNamespace(is_target_verify=verify, is_cp_linear_attn=cp)
            result = Qwen3NextDecoderLayer.forward(
                obj,
                None,
                None,
                None,
                attention_inputs=SimpleNamespace(is_prefill=prefill),
                attn_meta=meta,
            )
            self.assertEqual(result, prefill and not verify and not cp)
        self.assertFalse(in_prefill())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_metadata_groups_rebuilt_each_forward(self):
        import rtp_llm.models_py.model_desc.qwen3_next as model_module
        from rtp_llm.models_py.triton_kernels.fla.flashinfer_prefill import (
            flashinfer_metadata_key,
        )

        cu_a = torch.tensor([0, 2047, 4096], device="cuda", dtype=torch.int32)
        cu_b = cu_a.clone()
        attention = [
            SimpleNamespace(
                cu_seqlens_device=cu, is_prefill=True, is_target_verify=False
            )
            for cu in (cu_a, cu_a, cu_b, cu_b, cu_a, cu_a)
        ]
        intervals = [2048, 2048, 2048, 2048, 128, 128]
        seen = []

        class Layer:
            layer_type = model_module.HybridAttentionType.LINEAR

            def __init__(self, i):
                self.i = i

            def __call__(self, h, r, fmha, **kwargs):
                mapping = kwargs["attn_meta"].flashinfer_prefill_metadata
                if mapping:
                    self.assert_group(mapping)
                    seen.append(
                        mapping[
                            flashinfer_metadata_key(
                                attention[self.i].cu_seqlens_device, intervals[self.i]
                            )
                        ]
                    )
                return h, r

            def assert_group(self, mapping):
                if len(mapping) != 3:
                    raise AssertionError(
                        "Metadata must be prepared for all groups before the first layer"
                    )

        obj = SimpleNamespace(
            word_embedding=lambda inputs: torch.zeros(4096, 8, device="cuda"),
            kv_cache=SimpleNamespace(
                get_layer_cache=lambda i: SimpleNamespace(
                    seq_size_per_block=intervals[i]
                )
            ),
            parallelism_config=SimpleNamespace(
                prefill_cp_config=SimpleNamespace(is_enabled=lambda: False)
            ),
            layers=[Layer(i) for i in range(6)],
            norm=lambda h, r: (h, r),
        )
        with patch.object(
            model_module, "get_primary_attention_inputs", return_value=attention[0]
        ), patch.object(
            model_module,
            "select_attention_inputs_for_layer",
            side_effect=lambda inputs, cache, i: attention[i],
        ), patch.object(
            model_module, "prepare_causal_conv1d_metadata", return_value=None
        ), patch.dict(
            os.environ, {METADATA: "1", "RTP_QWEN35_GDN_PREFILL_BACKEND": "flashinfer"}
        ):
            for changed in (False, True):
                if changed:
                    cu_a[1] = 2048
                with patch(
                    "rtp_llm.models_py.triton_kernels.fla.flashinfer_prefill.prepare_flashinfer_prefill_metadata",
                    wraps=prepare_flashinfer_prefill_metadata,
                ) as build:
                    model_module.Qwen3NextModel.forward(
                        obj, object(), fmha_impl=object()
                    )
                    self.assertEqual(build.call_count, 3)
            self.assertIs(seen[0], seen[1])
            self.assertIs(seen[2], seen[3])
            self.assertIs(seen[4], seen[5])
            self.assertIsNot(seen[0], seen[2])
            self.assertIsNot(seen[0], seen[6])
            self.assertFalse(
                torch.equal(seen[0].checkpoint_starts, seen[6].checkpoint_starts)
            )
            for flag, backend, prefill, verify in (
                ("0", "flashinfer", True, False),
                ("1", "native", True, False),
                ("1", "flashinfer", False, False),
                ("1", "flashinfer", True, True),
            ):
                attention[0].is_prefill = prefill
                attention[0].is_target_verify = verify
                with patch.dict(
                    os.environ,
                    {METADATA: flag, "RTP_QWEN35_GDN_PREFILL_BACKEND": backend},
                ), patch(
                    "rtp_llm.models_py.triton_kernels.fla.flashinfer_prefill.prepare_flashinfer_prefill_metadata"
                ) as build:
                    model_module.Qwen3NextModel.forward(
                        obj, object(), fmha_impl=object()
                    )
                    build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
