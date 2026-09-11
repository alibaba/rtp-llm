import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
                    _data_offset_bytes=128,
                    _mapping_handle=object(),
                    _launch_lock=object(),
                    _last_stream=None,
                    _barrier=Mock(),
                )
                backend = SimpleNamespace(
                    GemmRSBuffer=Mock(return_value=workspace),
                    bf16_gemm_rs_nn=Mock(),
                    _C=SimpleNamespace(bf16_gemm_rs_reduce=Mock()),
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

    def test_fp8_dispatch_keeps_ag_fused_and_selects_rs_backend(self):
        device = torch.device("cuda:0")
        projection = Mock(K=512, N=128, scale_ue8m0=True)
        expected = torch.zeros(8, 128)
        for size in (8, 16):
            with self.subTest(ag_size=size):
                group = Mock(size=Mock(return_value=size))
                payload = Mock(shape=(1, 512), device=device)
                state = SimpleNamespace(max_m=32, k=512, device=device)
                with patch.object(ag, "get_process_group", return_value=group), patch.dict(
                    ag._STATES, {(group, 0, True): state}
                ), patch.object(
                    ag, "fused_all_gather_fp8_linear", return_value=[expected]
                ) as fused, patch.object(ag.dist, "all_gather_into_tensor") as nccl:
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
                    rs, "_fp8_remote_gemm_reduce_scatter", return_value=expected
                ) as fused, patch.object(
                    rs, "_fp8_nccl_gemm_reduce_scatter", return_value=expected
                ) as nccl:
                    self.assertIs(
                        rs.gemm_reduce_scatter(
                            source, projection, group, pad_rows=True
                        ),
                        expected,
                    )
                    selected, unused = (fused, nccl) if size == 8 else (nccl, fused)
                    selected.assert_called_once_with(source, projection, state, 32)
                    unused.assert_not_called()

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
