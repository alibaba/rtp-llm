"""Address-only full-frequency-table lookup, against each unchanged old helper."""

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant, fused_ops, graph_swa_write


class _Launch:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            if self.failure is not None:
                raise self.failure
        return launch


@contextmanager
def _cpu_metadata_device():
    """Exercise real shape/stride/raw/storage checks without a GPU allocation."""
    stream = object()
    with ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        stack.enter_context(patch.object(torch.Tensor, "is_cuda", property(lambda self: True)))
        stack.enter_context(patch.object(torch.cuda, "current_device", return_value=None))
        stack.enter_context(patch.object(torch.cuda, "current_stream", return_value=stream))
        stack.enter_context(patch.object(torch.cuda, "default_stream", return_value=stream))
        stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
        yield


def _bytes(test, left, right):
    test.assertEqual(left.shape, right.shape)
    test.assertTrue(torch.equal(left.contiguous().view(torch.uint8),
                                right.contiguous().view(torch.uint8)))


class LocalPrefillSGRopeLookupCPU(unittest.TestCase):
    def test_full_table_launch_owners_and_old_helper_abi(self):
        value = torch.ones((3, 64, 512), dtype=torch.bfloat16)
        table = torch.ones((17, 32), dtype=torch.complex64)
        for dtype in (torch.int32, torch.int64):
            positions = torch.tensor([16, 0, 7], dtype=dtype)
            selected = table[positions.long()]
            inplace, fresh = _Launch(), _Launch()
            with _cpu_metadata_device(), \
                    patch.object(graph_swa_write, "_kernels", return_value=(None, inplace, None, None, None)), \
                    patch.object(fused_ops, "_kernel", return_value=(None, fresh)):
                self.assertIs(graph_swa_write.try_rope_(value, table, positions=positions,
                                                       inverse=True, preserve_signed_zero=True), value)
                result = fused_ops.try_rope_tail(value, table, True, positions=positions)
                self.assertIsNotNone(result)
                self.assertNotEqual(result.untyped_storage().data_ptr(), value.untyped_storage().data_ptr())
                self.assertIs(graph_swa_write.try_rope_(value, selected, preserve_signed_zero=True), value)
                self.assertIsNotNone(fused_ops.try_rope_tail(value, selected, False))
            for launch in (inplace, fresh):
                self.assertIs(launch.calls[0][2]["positions"], positions)
                self.assertNotIn("positions", launch.calls[1][2])
                frequencies = launch.calls[0][1][1]
                self.assertEqual(frequencies.shape, (17, 32, 2))
                self.assertEqual(frequencies.untyped_storage().data_ptr(), table.untyped_storage().data_ptr())
                self.assertFalse(launch.calls[0][2]["enable_fp_fusion"])
            self.assertEqual(inplace.calls[0][0], (3, 8))
            self.assertEqual(fresh.calls[0][0], (192,))

    def test_real_metadata_rejections_and_no_position_value_certification(self):
        value = torch.ones((3, 64, 512), dtype=torch.bfloat16)
        table = torch.ones((17, 32), dtype=torch.complex64)
        positions = torch.tensor([16, 0, 7], dtype=torch.int64)
        before = value.clone()
        with _cpu_metadata_device():
            self.assertTrue(fused_ops._rope_positions_supported(value, table, positions))
            # Range/birth proof belongs to the caller. Admission must not read
            # or silently clamp these deliberately invalid fixture values.
            self.assertTrue(fused_ops._rope_positions_supported(
                value, table, torch.tensor([-1, 17, 7], dtype=torch.int64)))
            cases = (
                (value, table[:, ::2], positions),
                (value, torch.ones((34, 32), dtype=torch.complex64)[::2], positions),
                (value, table.conj(), positions),
                (value, table._neg_view(), positions),
                (value, table, torch.arange(6, dtype=torch.int64)[::2]),
                (value, table, positions.int().short()),
                (value, table, positions[:2]),
                (value._neg_view(), table, positions),
                (value[:, :, ::2], table, positions),
                (value, table.requires_grad_(), positions),
            )
            for x, frequencies, pos in cases:
                self.assertFalse(fused_ops._rope_positions_supported(x, frequencies, pos))
            table.requires_grad_(False)
            with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
                self.assertFalse(fused_ops._rope_positions_supported(value, table, positions))
            with patch.object(torch.cuda, "current_stream", return_value=object()):
                self.assertFalse(fused_ops._rope_positions_supported(value, table, positions))
            with torch.enable_grad():
                self.assertFalse(fused_ops._rope_positions_supported(value, table, positions))
        _bytes(self, value, before)

    def test_alias_unsupported_never_reaches_kernel_or_mutates(self):
        value = torch.ones((3, 64, 512), dtype=torch.bfloat16)
        table = torch.ones((17, 32), dtype=torch.complex64)
        positions = torch.tensor([16, 0, 7], dtype=torch.int64)
        before = value.clone()
        # Table and output are different dtype views of the same backing bank.
        alias_table = value.view(torch.complex64).reshape(-1, 32)
        with _cpu_metadata_device(), \
                patch.object(graph_swa_write, "_kernels", side_effect=AssertionError("unsupported launch")), \
                patch.object(fused_ops, "_kernel", side_effect=AssertionError("unsupported launch")):
            for frequencies, pos in ((alias_table, positions), (table[:, ::2], positions),
                                     (table, positions[:1]), (table, positions._neg_view())):
                self.assertIsNone(graph_swa_write.try_rope_(value, frequencies, positions=pos))
                self.assertIsNone(fused_ops.try_rope_tail(value, frequencies, False, positions=pos))
        _bytes(self, value, before)

    def test_admitted_native_errors_propagate_without_retry(self):
        value = torch.ones((3, 64, 512), dtype=torch.bfloat16)
        table = torch.ones((17, 32), dtype=torch.complex64)
        positions = torch.tensor([16, 0, 7], dtype=torch.int64)
        before = value.clone()
        inplace, fresh = _Launch(RuntimeError("inplace launch")), _Launch(RuntimeError("fresh launch"))
        with _cpu_metadata_device(), \
                patch.object(graph_swa_write, "_kernels", return_value=(None, inplace, None, None, None)), \
                patch.object(fused_ops, "_kernel", return_value=(None, fresh)):
            with self.assertRaisesRegex(RuntimeError, "inplace launch"):
                graph_swa_write.try_rope_(value, table, positions=positions)
            with self.assertRaisesRegex(RuntimeError, "fresh launch"):
                fused_ops.try_rope_tail(value, table, False, positions=positions)
        self.assertEqual(len(inplace.calls), 1)
        self.assertEqual(len(fresh.calls), 1)
        _bytes(self, value, before)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU test")
class LocalPrefillSGRopeLookupDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(_quant.is_ppu())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        self.assertEqual(torch.cuda.current_stream(), torch.cuda.default_stream())
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)
        torch.manual_seed(1212)

    def _inputs(self, rows, *, special=False):
        table = torch.complex(torch.randn((131200, 32), device="cuda"),
                              torch.randn((131200, 32), device="cuda"))
        positions = torch.arange(126976, 126976 + rows, device="cuda", dtype=torch.int64)
        positions[0], positions[1], positions[-1] = 0, table.shape[0] - 1, 37
        if special:
            bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to("cuda")
            count = rows * 64 * 512
            value = bits.repeat((count + 65535) // 65536)[:count].reshape(rows, 64, 512)
            patterns = torch.tensor([0., -0., torch.inf, -torch.inf, torch.nan, 1., -1.], device="cuda")
            table.view(torch.float32).reshape(-1)[:patterns.numel()].copy_(patterns)
        else:
            value = torch.randn((rows, 64, 512), device="cuda", dtype=torch.bfloat16)
        return value, table, positions

    def test_fresh4096_4097_compact140_qo_and_kv_match_old_helpers_exactly(self):
        for rows in (4096, 4097, 140):
            value, table, base_positions = self._inputs(rows)
            kv = value[:, 0].contiguous()
            kv_before = kv.clone()
            for dtype in (torch.int32, torch.int64):
                positions = base_positions.to(dtype)
                selected = table[positions.long()]
                for inverse in (False, True):
                    expected, actual = value.clone(), value.clone()
                    self.assertIs(graph_swa_write.try_rope_(expected, selected, inverse=inverse,
                                                           preserve_signed_zero=True), expected)
                    self.assertIs(graph_swa_write.try_rope_(actual, table, inverse=inverse,
                                                           preserve_signed_zero=True, positions=positions), actual)
                    _bytes(self, actual, expected)
                    _bytes(self, actual[..., :448], value[..., :448])
                    fresh_expected = fused_ops.try_rope_tail(kv, selected, inverse)
                    fresh_actual = fused_ops.try_rope_tail(kv, table, inverse, positions=positions)
                    self.assertIsNotNone(fresh_actual)
                    _bytes(self, fresh_actual, fresh_expected)
                    self.assertNotEqual(fresh_actual.untyped_storage().data_ptr(), kv.untyped_storage().data_ptr())
                    self.assertNotEqual(fresh_actual.untyped_storage().data_ptr(), table.untyped_storage().data_ptr())
            _bytes(self, kv, kv_before)

    def test_all_bf16_bits_signedzero_nonfinite_table_and_valid_mask(self):
        value, table, positions = self._inputs(140, special=True)
        positions[2:6] = 0
        selected = table[positions]
        table_before = table.clone()
        valid = torch.arange(140, device="cuda") % 3 != 0
        kv = value.reshape(-1)[:140 * 512].reshape(140, 512).clone()
        original = kv.clone()
        for inverse in (False, True):
            for signed_zero in (False, True):
                expected, actual = value.clone(), value.clone()
                graph_swa_write.try_rope_(expected, selected, inverse=inverse, valid_mask=valid,
                                         preserve_signed_zero=signed_zero)
                graph_swa_write.try_rope_(actual, table, inverse=inverse, valid_mask=valid,
                                         preserve_signed_zero=signed_zero, positions=positions)
                _bytes(self, actual, expected)
            _bytes(self, fused_ops.try_rope_tail(kv, table, inverse, positions=positions),
                   fused_ops.try_rope_tail(kv, selected, inverse))
        _bytes(self, kv, original)
        _bytes(self, table, table_before)

    def test_strided_raw_views_nondefault_stream_reject_without_mutation(self):
        value, table, positions = self._inputs(140)
        before = value.clone()
        bad_table = torch.empty((table.shape[0], 64), device="cuda", dtype=torch.complex64)[:, ::2]
        bad_positions = torch.empty((280,), dtype=torch.int64, device="cuda")[::2]
        cases = ((bad_table, positions), (table, bad_positions), (table.conj(), positions),
                 (table._neg_view(), positions), (table, positions._neg_view()))
        with patch.object(graph_swa_write, "_kernels", side_effect=AssertionError("unsupported launch")), \
                patch.object(fused_ops, "_kernel", side_effect=AssertionError("unsupported launch")):
            for frequencies, pos in cases:
                self.assertIsNone(graph_swa_write.try_rope_(value, frequencies, positions=pos))
                self.assertIsNone(fused_ops.try_rope_tail(value, frequencies, False, positions=pos))
            with torch.cuda.stream(torch.cuda.Stream()):
                self.assertIsNone(graph_swa_write.try_rope_(value, table, positions=positions))
                self.assertIsNone(fused_ops.try_rope_tail(value, table, False, positions=positions))
        _bytes(self, value, before)

    def test_old_direct_kernel_abi_and_fulltable_profile_no_frequency_index(self):
        directory = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or tempfile.gettempdir())
        directory.mkdir(parents=True, exist_ok=True)
        for rows, fresh, inverse in ((4096, False, False), (140, False, True), (4097, True, False)):
            value, table, positions = self._inputs(rows)
            if fresh:
                value = value[:, 0].contiguous()
            selected = table[positions]
            expected = (fused_ops.try_rope_tail(value, selected, inverse) if fresh else value.clone())
            if not fresh:
                # Direct old private ABI omits the new trailing optional arg.
                old_kernel = graph_swa_write._kernels()[1]
                old_kernel[(rows, 8)](expected, torch.view_as_real(selected), None,
                                      64, 512, inverse, True,
                                      num_warps=4, enable_fp_fusion=False)
            if fresh:
                operation = lambda: fused_ops.try_rope_tail(value, table, inverse, positions=positions)
            else:
                operation = lambda: graph_swa_write.try_rope_(value, table, inverse=inverse,
                                                             preserve_signed_zero=True, positions=positions)
            # Warm the exact dtype/mode before profiling; no clone, gather,
            # value observation or JIT belongs to the measured native chain.
            warmed = operation()
            _bytes(self, warmed, expected)
            if not fresh:
                # Both helpers now rotate the already warmed BF16 input once
                # more; the first comparison above checked the direct old ABI.
                expected = value.clone()
                graph_swa_write.try_rope_(expected, selected, inverse=inverse, preserve_signed_zero=True)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                actual = operation()
                torch.cuda.synchronize()
            _bytes(self, actual, expected)
            path = directory / f"sg-rope-lookup-{rows}-{'fresh' if fresh else 'inplace'}.json"
            profile.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
            kernels = [event.get("name", "") for event in events if event.get("cat") == "kernel"]
            self.assertEqual(len(kernels), 1, kernels)
            self.assertIn("fused_rope_tail" if fresh else "rope_inplace", kernels[0])
            self.assertFalse(any(event.get("name") in ("aten::index", "aten::index_select", "aten::gather")
                                 for event in events))
            self.assertFalse(any("DtoH" in event.get("name", "") for event in events))


if __name__ == "__main__":
    unittest.main()
