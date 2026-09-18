import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch
import torch.distributed._symmetric_memory as symm

import rtp_llm.models_py.modules.kimi_k3.all_gather_gemm as ag
import rtp_llm.models_py.modules.kimi_k3.gemm_reduce_scatter as rs
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)


class ReferenceProjection:
    scale_ue8m0 = True

    def __init__(self, weight):
        self.weight = weight.float()
        self.K, self.N = weight.shape

    def __call__(self, x, out=None):
        result = (x.float() @ self.weight).to(torch.bfloat16)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def forward_quantized(self, values, scales, out=None):
        groups = torch.arange(self.K // 128, device=values.device)
        exponents = (scales[:, groups // 4].long() >> (8 * (groups % 4))) & 255
        factors = torch.exp2(exponents.float() - 127).repeat_interleave(128, dim=1)
        return self(values.float() * factors, out=out)


def activation(rows, k, rank=0, device="cpu"):
    values = ((torch.arange(rows * k, device=device).view(rows, k) + rank) % 7 - 3).to(
        torch.float8_e4m3fn
    )
    groups = (k + 511) // 512
    wire = torch.full(
        (groups, (rows + 3) // 4 * 4), 0x7F7F7F7F, dtype=torch.int32, device=device
    )
    # Different scales across rows, source ranks and K-groups detect mispacking.
    for g in range(groups):
        exponent = 125 + (torch.arange(rows, device=device) + rank + g) % 3
        wire[g, :rows] = exponent * 0x01010101
    return QuantizedActivation(values, wire)


class KimiK3Fp8NcclTest(unittest.TestCase):
    def test_fp8_initialization_keeps_ag_fused_and_selects_rs_backend(self):
        for size in (8, 16):
            with self.subTest(size=size):
                group = Mock(size=Mock(return_value=size))
                device = torch.device("cuda:0")
                workspace = SimpleNamespace(
                    num_bytes=1024,
                )
                backend = SimpleNamespace(
                    GemmRSBuffer=Mock(return_value=workspace),
                    bf16_gemm_rs_nn=Mock(),
                    fp8_gemm_rs_nt=Mock(),
                )
                ready = torch.tensor([1], dtype=torch.int32)
                with patch.dict(ag._STATES, {}, clear=True), patch.dict(
                    rs._STATES, {}, clear=True
                ), patch.dict(
                    sys.modules, {"deep_gemm": backend if size == 8 else None}
                ), patch.object(
                    symm,
                    "_pipelined_multi_all_gather_and_consume",
                    Mock(),
                    create=True,
                ), patch.object(
                    ag, "reserve_fused_all_gather_matmul_workspace"
                ) as reserve, patch.object(
                    torch.cuda, "get_device_capability", return_value=(10, 0)
                ) as capability, patch.object(
                    rs.torch, "tensor", return_value=ready
                ), patch.object(
                    rs.dist, "all_reduce"
                ) as all_reduce:
                    self.assertTrue(
                        ag.configure_all_gather_gemm(
                            group,
                            device,
                            max_m=32,
                            k=512,
                            dtype=torch.bfloat16,
                            fp8=True,
                        )
                    )
                    self.assertTrue(
                        rs.configure_gemm_reduce_scatter(
                            group,
                            device,
                            max_m=32,
                            n=128,
                            fp8=True,
                        )
                    )
                    reserve.assert_called_once()
                    self.assertIs(reserve.call_args.args[0], group)
                    if size == 8:
                        backend.GemmRSBuffer.assert_called_once_with(
                            group,
                            max_m=32,
                            n=128,
                            device=device,
                        )
                    else:
                        backend.GemmRSBuffer.assert_not_called()
                        capability.assert_not_called()
                        all_reduce.assert_not_called()

    def test_tp16_initialization_shares_fp8_and_bf16_state(self):
        group = Mock(size=Mock(return_value=16))
        for order in ((True, False), (False, True)):
            with self.subTest(order=order), patch.dict(
                rs._STATES, {}, clear=True
            ), patch.dict(sys.modules, {"deep_gemm": None}):
                for fp8 in order:
                    self.assertTrue(
                        rs.configure_gemm_reduce_scatter(
                            group, "cuda:0", max_m=32, n=128, fp8=fp8
                        )
                    )
                self.assertEqual(len(rs._STATES), 1)
                state = rs._STATES[(group, 0)]
                self.assertIsNone(state.workspace)
                self.assertTrue(state.fp8)
                with self.assertRaisesRegex(RuntimeError, "different shape"):
                    rs.configure_gemm_reduce_scatter(group, "cuda:0", max_m=64, n=128)

    def test_bf16_decode_gemm_precedes_nccl_with_padding(self):
        for size in (2, 4, 8, 16):
            group, k, n = Mock(size=Mock(return_value=size)), 8, 4
            weight = (torch.arange(k * n).view(k, n) % 3 / 16).to(torch.bfloat16)
            for m, pad_rows in (
                (0, True),
                (1, True),
                (16, False),
                (17, True),
                (177, True),
            ):
                with self.subTest(rows=m, pad_rows=pad_rows):
                    # Real CPU BF16 math; only CUDA eligibility and NCCL are mocked.
                    x = (torch.arange(m * k * 2).view(m, k * 2) % 7 - 3).to(
                        torch.bfloat16
                    )[:, ::2]
                    physical_m = (m + size - 1) // size * size
                    expected = torch.nn.functional.pad(
                        (x.float() @ weight.float()).to(torch.bfloat16),
                        (0, 0, 0, physical_m - m),
                    )
                    state = rs._GemmReduceScatterState(
                        group,
                        x.device,
                        size,
                        max(size, physical_m),
                        n,
                        fp8=True,
                        use_fused=False,
                    )

                    def scatter(output, partial, *, op, group):
                        self.assertIs(group, state.group)
                        self.assertIs(op, rs.dist.ReduceOp.SUM)
                        self.assertTrue(partial.is_contiguous())
                        torch.testing.assert_close(partial, expected, rtol=0, atol=0)
                        output.fill_(7)

                    with patch.object(
                        torch.Tensor,
                        "is_cuda",
                        new_callable=PropertyMock,
                        return_value=True,
                    ), patch.object(
                        rs, "collective_gemm_state_key", return_value=(group, 0)
                    ), patch.dict(
                        rs._STATES, {(group, 0): state}
                    ), patch.object(
                        rs.dist, "reduce_scatter_tensor", side_effect=scatter
                    ) as nccl:
                        result = rs.gemm_reduce_scatter(
                            x, weight, group, pad_rows=pad_rows
                        )
                    self.assertEqual(tuple(result.shape), (physical_m // size, n))
                    self.assertEqual(result.dtype, torch.bfloat16)
                    if m:
                        nccl.assert_called_once()
                        torch.testing.assert_close(result, torch.full_like(result, 7))
                    else:
                        nccl.assert_not_called()

    def test_fp8_dispatch_keeps_ag_fused_and_selects_rs_backend(self):
        device = torch.device("cuda:0")
        projection = Mock(K=512, N=128, scale_ue8m0=True)
        expected = torch.zeros(8, 128)
        for size in (8, 16):
            with self.subTest(ag_size=size):
                group = Mock(size=Mock(return_value=size))
                payload = Mock(shape=(1, 512), device=device)
                state = SimpleNamespace(max_m=32, k=512, device=device, use_fused=True)
                with patch.object(
                    ag, "get_process_group", return_value=group
                ), patch.dict(ag._STATES, {(group, 0, True): state}), patch.object(
                    ag, "fused_all_gather_fp8_linear", return_value=[expected]
                ) as fused, patch.object(
                    ag.dist, "all_gather_into_tensor"
                ) as nccl:
                    ag._all_gather_quantized(
                        payload, [projection], logical_m=size, group=None
                    )
                    fused.assert_called_once_with(payload, [projection], group)
                    nccl.assert_not_called()
        for size in (8, 16):
            with self.subTest(size=size):
                group = Mock(size=Mock(return_value=size))
                source = Mock(
                    is_cuda=True,
                    device=device,
                    ndim=2,
                    dtype=torch.bfloat16,
                    shape=(32, 512),
                )
                state = rs._GemmReduceScatterState(
                    group, device, size, 32, 128, fp8=True
                )
                with patch.dict(rs._STATES, {(group, 0): state}), patch.object(
                    rs, "_fp8_fused_gemm_reduce_scatter", return_value=expected
                ) as fused, patch.object(
                    rs, "_fp8_nccl_gemm_reduce_scatter", return_value=expected
                ) as nccl:
                    self.assertIs(
                        rs.gemm_reduce_scatter(
                            source, projection, group, pad_rows=True
                        ),
                        expected,
                    )
                    nccl.assert_called_once_with(source, projection, state, 32)
                    fused.assert_not_called()

    def test_rs_prefill_threshold_uses_unpadded_input_for_both_precisions(self):
        # Exercise the actual dispatcher and real CPU math, mocking only CUDA
        # eligibility and the two collective boundaries. All ranks have the
        # same partial; check the last output shard, including zero padding.
        k, n = 128, 8
        weight = (torch.arange(k * n).view(k, n) % 3 / 16).to(torch.bfloat16)
        reference = ReferenceProjection(weight)
        projection = Mock(K=k, N=n, bias=None, scale_ue8m0=True)
        projection.weight = weight.T.contiguous().to(torch.float8_e4m3fn)
        projection.weight_scales = torch.ones((1, 1))
        projection.side_effect = reference
        projection.forward_quantized.side_effect = reference.forward_quantized

        def quantize(x):
            if isinstance(x, QuantizedActivation):
                return x.values, x.scales
            return (
                x.to(torch.float8_e4m3fn),
                torch.full((1, x.shape[0]), 0x7F7F7F7F, dtype=torch.int32).T,
            )

        projection.quantize_input.side_effect = quantize
        for size in (2, 4, 8, 16):
            for prefill in (False, True):
                for m in (0, 1, 7, 504, 511, 512, 513, 1024):
                    for precision in ("bf16", "fp8", "prequantized"):
                        with self.subTest(
                            tp=size, prefill=prefill, m=m, precision=precision
                        ):
                            payload = activation(m, k)
                            x = (
                                payload
                                if precision == "prequantized"
                                else payload.values.to(torch.bfloat16)
                            )
                            physical_m = (m + size - 1) // size * size
                            partial = (
                                reference.forward_quantized(x.values, x.scales)
                                if precision == "prequantized"
                                else reference(x)
                            )
                            partial = torch.nn.functional.pad(
                                partial, (0, 0, 0, physical_m - m)
                            )
                            expected = (partial.float() * size).to(torch.bfloat16)[
                                physical_m - physical_m // size :
                            ]
                            group = Mock(size=Mock(return_value=size))
                            workspace = (
                                object()
                            )  # Public API must not need private workspace fields.

                            def scatter(out, actual, *, op, group):
                                torch.testing.assert_close(
                                    actual, partial, rtol=0, atol=0
                                )
                                out.copy_(
                                    (actual.float() * size).to(torch.bfloat16)[
                                        physical_m - physical_m // size :
                                    ]
                                )

                            def bf16(a, b, out, ws, *, compiled_dims):
                                self.assertIs(ws, workspace)
                                self.assertEqual(compiled_dims, "nk")
                                scatter(
                                    out, a @ b, op=rs.dist.ReduceOp.SUM, group=group
                                )

                            def fp8(a, b, out, ws, *, compiled_dims):
                                self.assertIs(ws, workspace)
                                self.assertIs(b[0], projection.weight)
                                self.assertIs(b[1], projection.weight_scales)
                                self.assertEqual(compiled_dims, "nk")
                                self.assertEqual(a[0].shape[0], physical_m)
                                if precision == "prequantized" and physical_m == m:
                                    self.assertIs(a[0], payload.values)
                                    self.assertEqual(
                                        a[1].data_ptr(), payload.scales.data_ptr()
                                    )
                                scatter(
                                    out,
                                    reference.forward_quantized(*a),
                                    op=rs.dist.ReduceOp.SUM,
                                    group=group,
                                )

                            backend = SimpleNamespace(
                                bf16_gemm_rs_nn=Mock(side_effect=bf16),
                                fp8_gemm_rs_nt=Mock(side_effect=fp8),
                            )
                            state = rs._GemmReduceScatterState(
                                group,
                                x.device,
                                size,
                                2048,
                                n,
                                backend,
                                workspace,
                                fp8=True,
                                use_fused=prefill,
                            )
                            with patch.object(
                                torch.Tensor,
                                "is_cuda",
                                new_callable=PropertyMock,
                                return_value=True,
                            ), patch.object(
                                rs, "collective_gemm_state_key", return_value=(group, 0)
                            ), patch.dict(
                                rs._STATES, {(group, 0): state}
                            ), patch.object(
                                rs.dist, "reduce_scatter_tensor", side_effect=scatter
                            ) as nccl:
                                actual = rs.gemm_reduce_scatter(
                                    x,
                                    weight if precision == "bf16" else projection,
                                    group,
                                    pad_rows=True,
                                )
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            fused = prefill and m >= 512 and size != 16
                            self.assertEqual(nccl.call_count, int(m > 0 and not fused))
                            self.assertEqual(
                                backend.bf16_gemm_rs_nn.call_count,
                                int(fused and precision == "bf16"),
                            )
                            self.assertEqual(
                                backend.fp8_gemm_rs_nt.call_count,
                                int(fused and precision != "bf16"),
                            )

    def test_fp8_bias_or_fp32_scales_keep_projection_semantics(self):
        group = Mock(size=Mock(return_value=8))
        x = torch.ones(512, 128, dtype=torch.bfloat16)
        expected = torch.full((64, 8), 7, dtype=torch.bfloat16)
        for scale_ue8m0, bias in ((False, None), (True, torch.ones(8))):
            with self.subTest(scale_ue8m0=scale_ue8m0, bias=bias is not None):
                projection = Mock(K=128, N=8, scale_ue8m0=scale_ue8m0, bias=bias)
                state = rs._GemmReduceScatterState(group, x.device, 8, 512, 8, fp8=True)
                with patch.object(
                    torch.Tensor,
                    "is_cuda",
                    new_callable=PropertyMock,
                    return_value=True,
                ), patch.object(
                    rs, "collective_gemm_state_key", return_value=(group, 0)
                ), patch.dict(
                    rs._STATES, {(group, 0): state}
                ), patch.object(
                    rs, "_fp8_nccl_gemm_reduce_scatter", return_value=expected
                ) as nccl, patch.object(
                    rs, "_fp8_fused_gemm_reduce_scatter"
                ) as fused:
                    self.assertIs(
                        rs.gemm_reduce_scatter(x, projection, group, pad_rows=False),
                        expected,
                    )
                nccl.assert_called_once_with(x, projection, state, 512)
                fused.assert_not_called()

    def test_missing_public_fp8_api_is_rejected_before_allocating_workspace(self):
        group = Mock(size=Mock(return_value=8))
        backend = SimpleNamespace(GemmRSBuffer=Mock(), bf16_gemm_rs_nn=Mock())
        readiness = torch.tensor([0], dtype=torch.int32)
        with patch.dict(rs._STATES, {}, clear=True), patch.dict(
            sys.modules, {"deep_gemm": backend}
        ), patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ), patch.object(
            rs.torch, "tensor", return_value=readiness
        ), patch.object(
            rs.dist, "all_reduce"
        ) as all_reduce:
            with self.assertRaisesRegex(RuntimeError, "fp8_gemm_rs_nt"):
                rs.configure_gemm_reduce_scatter(
                    group, "cuda:0", max_m=512, n=128, fp8=True
                )
            all_reduce.assert_called_once()
            backend.GemmRSBuffer.assert_not_called()
            self.assertFalse(rs._STATES)

    def test_reduce_scatter_receives_one_full_gemm_with_padding(self):
        size, k, n = 16, 640, 8
        for m in (1, 15, 16, 17, 177):
            for quantized in (False, True):
                with self.subTest(rows=m, quantized=quantized):
                    payload = activation(m, k)
                    x = (
                        payload
                        if quantized
                        else payload.values.float().to(torch.bfloat16)
                    )
                    reference = ReferenceProjection(
                        torch.arange(k * n).view(k, n) % 3 / 16
                    )
                    physical_m = (m + size - 1) // size * size
                    valid = (
                        reference.forward_quantized(x.values, x.scales)
                        if quantized
                        else reference(x)
                    )
                    expected_partial = torch.nn.functional.pad(
                        valid, (0, 0, 0, physical_m - m)
                    )
                    projection = Mock(wraps=reference, K=k, N=n)
                    group = object()
                    state = SimpleNamespace(
                        fp8=True, world_size=size, max_m=physical_m, n=n, group=group
                    )

                    def scatter(output, partial, *, op, group):
                        self.assertIs(op, rs.dist.ReduceOp.SUM)
                        self.assertIs(group, state.group)
                        self.assertEqual(output.shape, (physical_m // size, n))
                        self.assertEqual(output.dtype, torch.bfloat16)
                        self.assertTrue(partial.is_contiguous())
                        torch.testing.assert_close(
                            partial, expected_partial, rtol=0, atol=0
                        )
                        # This checks delegation/output propagation, not NCCL arithmetic.
                        output.fill_(7)

                    with patch.object(
                        rs.dist, "reduce_scatter_tensor", side_effect=scatter
                    ) as nccl:
                        result = rs._fp8_nccl_gemm_reduce_scatter(
                            x, projection, state, physical_m
                        )
                    nccl.assert_called_once()
                    if quantized:
                        projection.forward_quantized.assert_called_once()
                        projection.assert_not_called()
                    else:
                        projection.assert_called_once()
                        projection.forward_quantized.assert_not_called()
                    torch.testing.assert_close(result, torch.full_like(result, 7))

    def test_decode_initialization_avoids_all_fused_resources(self):
        for size in (2, 4, 8, 16):
            group = Mock(size=Mock(return_value=size))
            with self.subTest(size=size), patch.dict(
                ag._STATES, {}, clear=True
            ), patch.dict(rs._STATES, {}, clear=True), patch.dict(
                sys.modules, {"deep_gemm": None}
            ), patch.object(
                symm, "_pipelined_multi_all_gather_and_consume", None
            ), patch.object(
                ag, "reserve_fused_all_gather_matmul_workspace"
            ) as reserve:
                for fp8 in (False, True):
                    ag.configure_all_gather_gemm(
                        group,
                        "cuda:0",
                        max_m=32,
                        k=640,
                        dtype=torch.bfloat16,
                        fp8=fp8,
                        use_fused=False,
                    )
                    rs.configure_gemm_reduce_scatter(
                        group,
                        "cuda:0",
                        max_m=32,
                        n=128,
                        fp8=fp8,
                        use_fused=False,
                    )
                reserve.assert_not_called()
                self.assertIsNone(rs._STATES[(group, 0)].workspace)
                self.assertFalse(rs._STATES[(group, 0)].use_fused)
                if size != 16:
                    with self.assertRaisesRegex(RuntimeError, "backend"):
                        rs.configure_gemm_reduce_scatter(
                            group, "cuda:0", max_m=32, n=128
                        )

    def test_decode_fp8_ag_preserves_rank_rows_and_packed_scales(self):
        k = 640
        for size in (2, 4, 8, 16):
            for m in (1, 3, 4, 5):
                with self.subTest(size=size, rows=m):
                    group = Mock(size=Mock(return_value=size))
                    payloads = [activation(m, k, rank) for rank in range(size)]
                    projections = [
                        ReferenceProjection(
                            (torch.arange(k * n).view(k, n) % 3 - 1) / 16
                        )
                        for n in (8, 16)
                    ]
                    logical_m = m * size - 1
                    expected = [
                        torch.cat(
                            [p.forward_quantized(a.values, a.scales) for a in payloads]
                        )[:logical_m]
                        for p in projections
                    ]
                    state = ag._AllGatherGemmState(
                        True,
                        group,
                        torch.device("cpu"),
                        size,
                        m * size,
                        k,
                        torch.bfloat16,
                        0,
                        use_fused=False,
                    )

                    def gather(output, source, *, group):
                        sources = (
                            [a.values.view(torch.uint8) for a in payloads]
                            if source.dtype == torch.uint8
                            else [a.scale_wire for a in payloads]
                        )
                        output.copy_(torch.cat(sources))

                    with patch.object(
                        ag, "get_process_group", return_value=group
                    ), patch.object(
                        ag, "collective_gemm_state_key", return_value=(group, 0)
                    ), patch.dict(
                        ag._STATES, {(group, 0, True): state}
                    ), patch.object(
                        ag.dist, "all_gather_into_tensor", side_effect=gather
                    ) as nccl, patch.object(
                        ag, "fused_all_gather_fp8_linear"
                    ) as fused:
                        actual = ag.all_gather_gemm(
                            payloads[0], projections, logical_m=logical_m
                        )
                    self.assertEqual(nccl.call_count, 2)
                    fused.assert_not_called()
                    for result, reference in zip(actual, expected):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_empty_input_does_not_communicate(self):
        x = activation(0, 128)
        p = ReferenceProjection(torch.ones(128, 8))
        state = SimpleNamespace(fp8=True, world_size=16, max_m=16, n=8, group=object())
        with patch.object(rs.dist, "reduce_scatter_tensor") as scatter:
            result = rs._fp8_nccl_gemm_reduce_scatter(x, p, state, 0)
            self.assertEqual(tuple(result.shape), (0, 8))
            scatter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
