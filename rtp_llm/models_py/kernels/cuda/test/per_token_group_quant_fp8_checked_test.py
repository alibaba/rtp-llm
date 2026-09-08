from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from rtp_llm.models_py.utils.arch import is_hip
from rtp_llm.ops.compute_ops import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_checked,
    per_token_group_quant_fp8_v2,
    per_token_group_quant_fp8_v2_checked,
)


FP8 = torch.float8_e4m3fn
GROUP_SIZE = 128
EPS = 1.0e-10
FP8_MAX = torch.finfo(FP8).max


def _plain_quant_oracle(x):
    groups = x.float().view(-1, GROUP_SIZE)
    scale = groups.abs().amax(dim=-1).clamp_min(EPS) / FP8_MAX
    payload = (groups / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return payload.view_as(x), scale.view(*x.shape[:-1], x.shape[-1] // GROUP_SIZE)


class PerTokenGroupQuantFp8CheckedTest(TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        if is_hip():
            raise SkipTest("checked FP8 quantization is CUDA-only")
        torch.cuda.set_device(0)

    def _outputs(
        self,
        x,
        *,
        fuse=False,
        column_major=False,
        ue8m0=False,
    ):
        shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse else 1))
        payload = torch.empty(shape, device=x.device, dtype=FP8)
        scale = create_per_token_group_quant_fp8_output_scale(
            x_shape=shape,
            device=x.device,
            group_size=GROUP_SIZE,
            column_major_scales=column_major,
            scale_tma_aligned=ue8m0,
            scale_ue8m0=ue8m0,
        )
        return payload, scale

    def _legacy(self, x, payload, scale, status=None):
        args = (x, payload, scale, GROUP_SIZE, EPS, -FP8_MAX, FP8_MAX, False)
        if status is None:
            per_token_group_quant_fp8(*args)
        else:
            per_token_group_quant_fp8_checked(*args, status)

    def _v2(
        self,
        x,
        payload,
        scale,
        *,
        fuse=False,
        masked_m=None,
        ue8m0=False,
        status=None,
    ):
        args = (
            x,
            payload,
            scale,
            GROUP_SIZE,
            EPS,
            -FP8_MAX,
            FP8_MAX,
            ue8m0,
            fuse,
            masked_m,
        )
        if status is None:
            per_token_group_quant_fp8_v2(*args)
        else:
            per_token_group_quant_fp8_v2_checked(*args, status)

    def test_legacy_abi_and_finite_output_are_unchanged(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.randn(5, 3 * GROUP_SIZE, device="cuda", dtype=dtype)
                old_q, old_s = self._outputs(x)
                checked_q, checked_s = self._outputs(x)
                status = torch.zeros(1, device=x.device, dtype=torch.int32)
                self._legacy(x, old_q, old_s)
                self._legacy(x, checked_q, checked_s, status)
                self.assertTrue(torch.equal(old_q.view(torch.uint8), checked_q.view(torch.uint8)))
                self.assertTrue(torch.equal(old_s.view(torch.int32), checked_s.view(torch.int32)))
                self.assertEqual(status.item(), 0)
                with self.assertRaises(TypeError):
                    per_token_group_quant_fp8(
                        x,
                        old_q,
                        old_s,
                        GROUP_SIZE,
                        EPS,
                        -FP8_MAX,
                        FP8_MAX,
                        False,
                        status,
                    )
                with self.assertRaises(TypeError):
                    per_token_group_quant_fp8_v2(
                        x,
                        old_q,
                        old_s,
                        GROUP_SIZE,
                        EPS,
                        -FP8_MAX,
                        FP8_MAX,
                        False,
                        False,
                        None,
                        status,
                    )

    def test_legacy_reports_every_nonfinite_kind_and_position(self):
        positions = (0, GROUP_SIZE + 17, 3 * GROUP_SIZE - 1)
        values = (float("nan"), float("inf"), -float("inf"))
        for dtype in (torch.float16, torch.bfloat16):
            for position in positions:
                for value in values:
                    with self.subTest(dtype=dtype, position=position, value=value):
                        x = torch.ones(2, 3 * GROUP_SIZE, device="cuda", dtype=dtype)
                        x.view(-1)[position] = value
                        payload, scale = self._outputs(x)
                        status = torch.zeros(1, device=x.device, dtype=torch.int32)
                        self._legacy(x, payload, scale, status)
                        self.assertEqual(status.item(), 1)

    def test_legacy_reports_bad_group_across_all_warps_in_sixteen_group_block(self):
        x = torch.ones(1, 16 * GROUP_SIZE, device="cuda", dtype=torch.bfloat16)
        # Group 11 executes in a later warp of the 256-thread block.
        x[0, 11 * GROUP_SIZE + 73] = float("nan")
        payload, scale = self._outputs(x)
        status = torch.zeros(1, device=x.device, dtype=torch.int32)
        self._legacy(x, payload, scale, status)
        self.assertEqual(status.item(), 1)

    def test_legacy_sixteen_group_finite_result_matches_independent_oracle(self):
        x = torch.arange(16 * GROUP_SIZE, device="cuda", dtype=torch.float32)
        x = x.remainder(251).sub_(125).view(1, 16 * GROUP_SIZE).to(torch.bfloat16)
        expected_q, expected_s = _plain_quant_oracle(x)
        checked_q, checked_s = self._outputs(x)
        status = torch.zeros(1, device=x.device, dtype=torch.int32)
        self._legacy(x, checked_q, checked_s, status)
        self.assertTrue(torch.equal(checked_q.view(torch.uint8), expected_q.view(torch.uint8)))
        self.assertTrue(torch.equal(checked_s.view(torch.int32), expected_s.view(torch.int32)))
        self.assertEqual(status.item(), 0)

    def test_status_is_sticky_and_caller_reset_is_observed(self):
        bad = torch.ones(2, GROUP_SIZE, device="cuda", dtype=torch.bfloat16)
        bad[0, 0] = float("nan")
        good = torch.ones_like(bad)
        payload, scale = self._outputs(bad)
        status = torch.zeros(1, device=bad.device, dtype=torch.int32)
        self._legacy(bad, payload, scale, status)
        self.assertEqual(status.item(), 1)
        self._legacy(good, payload, scale, status)
        self.assertEqual(status.item(), 1)
        status.zero_()
        self._legacy(good, payload, scale, status)
        self.assertEqual(status.item(), 0)

    def test_v2_plain_row_and_column_scale_are_bitwise_unchanged(self):
        for dtype in (torch.float16, torch.bfloat16):
            for column_major in (False, True):
                with self.subTest(dtype=dtype, column_major=column_major):
                    x = torch.randn(8, 2 * GROUP_SIZE, device="cuda", dtype=dtype)
                    old_q, old_s = self._outputs(x, column_major=column_major)
                    checked_q, checked_s = self._outputs(x, column_major=column_major)
                    status = torch.zeros(1, device=x.device, dtype=torch.int32)
                    self._v2(x, old_q, old_s)
                    self._v2(x, checked_q, checked_s, status=status)
                    self.assertTrue(torch.equal(old_q.view(torch.uint8), checked_q.view(torch.uint8)))
                    self.assertTrue(torch.equal(old_s.view(torch.int32), checked_s.view(torch.int32)))
                    self.assertEqual(status.item(), 0)

    def test_v2_plain_reports_nonfinite_across_groups(self):
        for dtype in (torch.float16, torch.bfloat16):
            for position in (0, GROUP_SIZE + 23, 3 * GROUP_SIZE - 1):
                for value in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(dtype=dtype, position=position, value=value):
                        x = torch.ones(3, 3 * GROUP_SIZE, device="cuda", dtype=dtype)
                        x.view(-1)[position] = value
                        payload, scale = self._outputs(x)
                        status = torch.zeros(1, device=x.device, dtype=torch.int32)
                        self._v2(x, payload, scale, status=status)
                        self.assertEqual(status.item(), 1)

    def test_v2_partial_and_multiwarp_finite_results_match_independent_oracle(self):
        for num_groups in (14, 16):
            with self.subTest(num_groups=num_groups):
                x = torch.arange(num_groups * GROUP_SIZE, device="cuda", dtype=torch.float32)
                x = x.remainder(193).sub_(96).view(num_groups, GROUP_SIZE).to(torch.bfloat16)
                expected_q, expected_s = _plain_quant_oracle(x)
                checked_q, checked_s = self._outputs(x)
                status = torch.zeros(1, device=x.device, dtype=torch.int32)
                self._v2(x, checked_q, checked_s, status=status)
                self.assertTrue(
                    torch.equal(checked_q.view(torch.uint8), expected_q.view(torch.uint8))
                )
                self.assertTrue(
                    torch.equal(checked_s.view(torch.int32), expected_s.view(torch.int32))
                )
                self.assertEqual(status.item(), 0)

    def test_v2_fused_checks_primary_secondary_and_transformed_values(self):
        for dtype in (torch.float16, torch.bfloat16):
            for source in ("primary", "secondary"):
                with self.subTest(dtype=dtype, source=source):
                    x = torch.ones(4, 2 * GROUP_SIZE, device="cuda", dtype=dtype)
                    x[1, 11 if source == "primary" else GROUP_SIZE + 11] = float("nan")
                    payload, scale = self._outputs(x, fuse=True)
                    status = torch.zeros(1, device=x.device, dtype=torch.int32)
                    self._v2(x, payload, scale, fuse=True, status=status)
                    self.assertEqual(status.item(), 1)

        x = torch.ones(4, 2 * GROUP_SIZE, device="cuda", dtype=torch.float16)
        x[2, 5] = torch.finfo(torch.float16).max
        x[2, GROUP_SIZE + 5] = 2.0
        payload, scale = self._outputs(x, fuse=True)
        status = torch.zeros(1, device=x.device, dtype=torch.int32)
        self._v2(x, payload, scale, fuse=True, status=status)
        self.assertEqual(status.item(), 1)

    def test_v2_fused_finite_output_is_bitwise_unchanged(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.randn(8, 4 * GROUP_SIZE, device="cuda", dtype=dtype)
                old_q, old_s = self._outputs(x, fuse=True)
                checked_q, checked_s = self._outputs(x, fuse=True)
                status = torch.zeros(1, device=x.device, dtype=torch.int32)
                self._v2(x, old_q, old_s, fuse=True)
                self._v2(x, checked_q, checked_s, fuse=True, status=status)
                self.assertTrue(torch.equal(old_q.view(torch.uint8), checked_q.view(torch.uint8)))
                self.assertTrue(torch.equal(old_s.view(torch.int32), checked_s.view(torch.int32)))
                self.assertEqual(status.item(), 0)

    def test_v2_masked_consumes_only_active_rows_with_column_ue8m0_scale(self):
        hidden_size = 16 * GROUP_SIZE
        x = torch.ones(2, 4, 2 * hidden_size, device="cuda", dtype=torch.bfloat16)
        masked_m = torch.tensor([1, 3], device=x.device, dtype=torch.int32)
        x[0, 3, 0] = float("nan")
        old_payload, old_scale = self._outputs(x, fuse=True, column_major=True, ue8m0=True)
        payload, scale = self._outputs(x, fuse=True, column_major=True, ue8m0=True)
        old_payload.zero_()
        payload.zero_()
        old_scale.zero_()
        scale.zero_()
        status = torch.zeros(1, device=x.device, dtype=torch.int32)
        self._v2(x, old_payload, old_scale, fuse=True, masked_m=masked_m, ue8m0=True)
        self._v2(x, payload, scale, fuse=True, masked_m=masked_m, ue8m0=True, status=status)
        self.assertEqual(status.item(), 0)
        self.assertTrue(torch.equal(old_payload.view(torch.uint8), payload.view(torch.uint8)))
        self.assertTrue(torch.equal(old_scale.view(torch.int32), scale.view(torch.int32)))

        x[1, 2, hidden_size - 1] = -float("inf")
        self._v2(x, payload, scale, fuse=True, masked_m=masked_m, ue8m0=True, status=status)
        self.assertEqual(status.item(), 1)

    def test_checked_status_contract_is_strict(self):
        x = torch.ones(2, GROUP_SIZE, device="cuda", dtype=torch.bfloat16)
        payload, scale = self._outputs(x)
        bad_statuses = (
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, device=x.device, dtype=torch.int64),
            torch.zeros(1, 1, device=x.device, dtype=torch.int32),
        )
        for status in bad_statuses:
            with self.subTest(status=status):
                with self.assertRaises(RuntimeError):
                    self._legacy(x, payload, scale, status)
        if torch.cuda.device_count() > 1:
            wrong_device = torch.zeros(1, device="cuda:1", dtype=torch.int32)
            with self.assertRaisesRegex(RuntimeError, "same device"):
                self._legacy(x, payload, scale, wrong_device)

    def test_checked_status_must_not_share_storage_with_any_operand(self):
        x = torch.ones(2, GROUP_SIZE, device="cuda", dtype=torch.bfloat16)
        payload, scale = self._outputs(x)

        input_storage = torch.zeros(x.numel(), device=x.device, dtype=torch.int32)
        input_alias = input_storage.view(torch.bfloat16)[: x.numel()].view_as(x)
        with self.assertRaisesRegex(RuntimeError, "share storage with input"):
            self._legacy(input_alias, payload, scale, input_storage[:1])

        output_storage = torch.zeros(payload.numel(), device=x.device, dtype=torch.int32)
        output_alias = output_storage.view(torch.uint8)[: payload.numel()].view(FP8).view_as(payload)
        with self.assertRaisesRegex(RuntimeError, "share storage with output_q"):
            self._legacy(x, output_alias, scale, output_storage[:1])

        scale_storage = torch.zeros(scale.numel(), device=x.device, dtype=torch.int32)
        scale_alias = scale_storage.view(torch.float32).view_as(scale)
        with self.assertRaisesRegex(RuntimeError, "share storage with output_s"):
            self._legacy(x, payload, scale_alias, scale_storage[:1])

        masked_storage = torch.tensor([1, 1, 0], device=x.device, dtype=torch.int32)
        fused = torch.ones(1, 2 * GROUP_SIZE, device="cuda", dtype=torch.bfloat16)
        fused_q, fused_s = self._outputs(fused, fuse=True)
        with self.assertRaisesRegex(RuntimeError, "share storage with masked_m"):
            self._v2(
                fused,
                fused_q,
                fused_s,
                fuse=True,
                masked_m=masked_storage[1:2],
                status=masked_storage[:1],
            )

    def test_checked_launch_uses_current_stream(self):
        x = torch.ones(2, GROUP_SIZE, device="cuda", dtype=torch.bfloat16)
        x[0, 0] = float("inf")
        payload, scale = self._outputs(x)
        status = torch.zeros(1, device=x.device, dtype=torch.int32)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self._legacy(x, payload, scale, status)
        stream.synchronize()
        self.assertEqual(status.item(), 1)

    def test_checked_launches_can_be_captured_and_replayed_with_sticky_status(self):
        for version in ("legacy", "v2"):
            with self.subTest(version=version):
                x = torch.ones(2, GROUP_SIZE, device="cuda", dtype=torch.bfloat16)
                payload, scale = self._outputs(x)
                status = torch.zeros(1, device=x.device, dtype=torch.int32)

                warm_x = x.clone()
                warm_q, warm_s = self._outputs(warm_x)
                warm_status = torch.zeros_like(status)
                warm_stream = torch.cuda.Stream()
                with torch.cuda.stream(warm_stream):
                    if version == "legacy":
                        self._legacy(warm_x, warm_q, warm_s, warm_status)
                    else:
                        self._v2(warm_x, warm_q, warm_s, status=warm_status)
                warm_stream.synchronize()

                graph = torch.cuda.CUDAGraph()
                capture_stream = torch.cuda.Stream()
                captured_ptrs = (
                    x.data_ptr(),
                    payload.data_ptr(),
                    scale.data_ptr(),
                    status.data_ptr(),
                )
                with torch.cuda.graph(graph, stream=capture_stream):
                    if version == "legacy":
                        self._legacy(x, payload, scale, status)
                    else:
                        self._v2(x, payload, scale, status=status)

                torch.cuda.synchronize()
                self.assertEqual(status.item(), 0)

                good = torch.ones_like(x)
                bad = good.clone()
                bad[-1, -1] = float("nan")
                for _ in range(100):
                    status.zero_()
                    x.copy_(bad)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertEqual(status.item(), 1)

                    x.copy_(good)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertEqual(status.item(), 1)

                    status.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertEqual(status.item(), 0)
                    self.assertEqual(
                        (
                            x.data_ptr(),
                            payload.data_ptr(),
                            scale.data_ptr(),
                            status.data_ptr(),
                        ),
                        captured_ptrs,
                    )


if __name__ == "__main__":
    main()
