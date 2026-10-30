"""CPU contracts for Prefill GEMM/RS dispatch and Decode staging isolation."""

import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.distributed._symmetric_memory as symm


@contextmanager
def projection_context(precision, *, world_size=16, decode=False, peers=True):
    communication = types.ModuleType("rtp_llm.models_py.distributed.collective_torch")
    communication.STAGING_MAX_LOCAL_M = 128
    communication.Fp8Activation = type("Fp8Activation", (), {})
    communication.is_same_host_group = Mock(return_value=False)
    communication.create_custom_all_gather = Mock(
        return_value=object() if peers else None
    )
    push = Mock()
    communication.create_push_reduce_scatter = Mock(return_value=push)

    deep_gemm = types.ModuleType("deep_gemm")
    workspace = object()
    deep_gemm.GemmRSBuffer = Mock(return_value=workspace)

    def publish(a, weight, output, buffer, **kwargs):
        output.fill_(3)

    deep_gemm.fp8_gemm_rs_nt = Mock(side_effect=publish)
    deep_gemm.bf16_gemm_rs_nn = Mock(side_effect=publish)
    group = Mock()
    group.size.return_value = world_size
    group.group_name = "projection_contract"
    projection = Mock()
    projection.K, projection.N = 128, 7168
    projection.bias = None
    projection.scale_ue8m0 = True
    dtype = torch.float8_e4m3fn if precision == "fp8" else torch.bfloat16
    projection.weight = torch.zeros((7168, 128), dtype=dtype)
    projection.weight_scales = torch.full((7168, 1), 0x7F7F7F7F, dtype=torch.int32)

    path = Path(__file__).resolve().parents[1] / f"{precision}_comm_fusion.py"
    spec = importlib.util.spec_from_file_location(f"{precision}_rs_contract", path)
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(
            sys.modules, {communication.__name__: communication, "deep_gemm": deep_gemm}
        ),
        patch.object(torch.distributed, "get_world_size", return_value=world_size),
        patch.object(symm, "get_symm_mem_workspace"),
    ):
        spec.loader.exec_module(module)
        if precision == "fp8":
            collective = module.Fp8CollectiveProjection(
                group,
                torch.device("cpu"),
                max_m=1024,
                hidden_size=7168,
                enable_ag=False,
                decode_staging=decode,
            )
        else:
            collective = module.Bf16CollectiveProjection(
                group,
                torch.device("cpu"),
                max_m=1024,
                hidden_size=7168,
                output_projections=[projection],
            )
        yield collective, projection, deep_gemm, workspace, push, group


class CollectiveProjectionGemmRSTest(unittest.TestCase):
    def test_tp8_and_tp16_prefill_create_fused_workspace(self):
        for precision in ("bf16", "fp8"):
            for world_size in (8, 16):
                with self.subTest(precision=precision, tp=world_size):
                    with projection_context(precision, world_size=world_size) as ctx:
                        collective, _, backend, workspace, _, group = ctx
                        self.assertIs(collective.rs_workspace, workspace)
                        backend.GemmRSBuffer.assert_called_once_with(
                            group, max_m=1024, n=7168, device=torch.device("cpu")
                        )

    def test_tp16_prefill_uses_fusion_at_and_above_boundary(self):
        for precision in ("bf16", "fp8"):
            for rows in (512, 1024):
                with self.subTest(precision=precision, rows=rows):
                    with projection_context(precision) as ctx:
                        collective, projection, backend, workspace, push, _ = ctx
                        values = torch.empty(
                            (rows, 128),
                            dtype=(
                                torch.float8_e4m3fn
                                if precision == "fp8"
                                else torch.bfloat16
                            ),
                        )
                        scales = torch.full((rows, 1), 0x7F7F7F7F, dtype=torch.int32)
                        if precision == "fp8":
                            result = collective.gemm_reduce_scatter(
                                values, scales, projection
                            )
                            call = backend.fp8_gemm_rs_nt.call_args
                            backend.fp8_gemm_rs_nt.assert_called_once()
                            self.assertIs(call.args[0][0], values)
                            self.assertIs(call.args[0][1], scales)
                            self.assertIs(call.args[1][0], projection.weight)
                            self.assertIs(call.args[1][1], projection.weight_scales)
                            projection.forward_quantized.assert_not_called()
                        else:
                            result = collective.gemm_reduce_scatter(values, projection)
                            call = backend.bf16_gemm_rs_nn.call_args
                            backend.bf16_gemm_rs_nn.assert_called_once()
                            self.assertIs(call.args[0], values)
                            self.assertTrue(call.args[1].is_contiguous())
                            torch.testing.assert_close(
                                call.args[1], projection.weight.T
                            )
                            projection.assert_not_called()
                        self.assertIs(call.args[2], result)
                        self.assertIs(call.args[3], workspace)
                        self.assertEqual(call.kwargs, {"compiled_dims": "nk"})
                        self.assertEqual(result.shape, (rows // 16, 7168))
                        self.assertEqual(result.dtype, torch.bfloat16)
                        self.assertTrue(torch.all(result == 3))
                        push.reduce_scatter.assert_not_called()

    def test_tp16_small_prefill_and_decode_keep_separate_gemm_and_push(self):
        for precision in ("bf16", "fp8"):
            cases = (
                ((False, 496), (True, 512)) if precision == "fp8" else ((False, 496),)
            )
            for decode, rows in cases:
                with self.subTest(precision=precision, decode=decode, rows=rows):
                    with projection_context(precision, decode=decode) as ctx:
                        collective, projection, backend, _, push, _ = ctx
                        partial = torch.empty((rows, 7168), dtype=torch.bfloat16)
                        projection.return_value = partial
                        projection.forward_quantized.return_value = partial
                        values = torch.empty(
                            (rows, 128),
                            dtype=(
                                torch.float8_e4m3fn
                                if precision == "fp8"
                                else torch.bfloat16
                            ),
                        )
                        if precision == "fp8":
                            scales = torch.empty((rows, 1), dtype=torch.int32)
                            result = collective.gemm_reduce_scatter(
                                values, scales, projection
                            )
                            projection.forward_quantized.assert_called_once_with(
                                values, scales
                            )
                        else:
                            result = collective.gemm_reduce_scatter(values, projection)
                            projection.assert_called_once_with(values)
                        backend.fp8_gemm_rs_nt.assert_not_called()
                        backend.bf16_gemm_rs_nn.assert_not_called()
                        push.reduce_scatter.assert_called_once()
                        self.assertIs(push.reduce_scatter.call_args.args[0], partial)
                        self.assertIs(push.reduce_scatter.call_args.args[1], result)
                        self.assertEqual(result.shape, (rows // 16, 7168))
                        if decode:
                            backend.GemmRSBuffer.assert_not_called()
                            self.assertIsNone(collective.rs_workspace)

    def test_bf16_without_peer_access_keeps_nccl_fallback(self):
        with projection_context("bf16", peers=False) as ctx:
            collective, projection, backend, _, _, _ = ctx
            self.assertIsNone(collective.rs_workspace)
            backend.GemmRSBuffer.assert_not_called()
            values = torch.empty((512, 128), dtype=torch.bfloat16)
            projection.return_value = torch.empty((512, 7168), dtype=torch.bfloat16)
            with patch.object(
                torch.distributed, "reduce_scatter_tensor"
            ) as reduce_scatter:
                result = collective.gemm_reduce_scatter(values, projection)
            reduce_scatter.assert_called_once()
            self.assertIs(reduce_scatter.call_args.args[0], result)
            self.assertIs(reduce_scatter.call_args.args[1], projection.return_value)


if __name__ == "__main__":
    unittest.main()
