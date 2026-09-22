"""Independent stable-sort oracle and opt-in CUDA coverage for index plans."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

# This leaf module can be tested without initializing the model/native engine.
_spec = importlib.util.spec_from_file_location(
    "v41_index_plan_under_test",
    Path(__file__).resolve().parents[1] / "_v41_prefill_index_plan.py",
)
fused = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fused
_spec.loader.exec_module(fused)


def reference_plan(
    selected, positions, offset, global_count, swa_start, window_size=128
):
    """Frozen eager attention path, deliberately using stable argsort."""
    positions = positions.long()
    swpos = (
        positions[:, None]
        - window_size
        + 1
        + torch.arange(window_size, device=selected.device)[None]
    )
    swidx = torch.where(
        swpos >= swa_start, offset + global_count + swpos - swa_start, -1
    )
    global_idx = torch.where(selected >= 0, offset + selected, -1)
    indices = torch.cat((global_idx, swidx), -1).int()
    indices = indices.gather(1, torch.argsort(indices < 0, dim=-1, stable=True))
    lengths = (indices >= 0).sum(-1).int()
    if indices.shape[1] % 64:
        indices = F.pad(indices, (0, 64 - indices.shape[1] % 64), value=-1)
    return indices, lengths


def make_inputs(rows, k=512, *, device="cpu", strided=False, dtype=torch.int32):
    generator = torch.Generator().manual_seed(41)
    selected = torch.randint(-7, 4096, (rows, k), generator=generator, dtype=dtype)
    selected[:, ::5] = -1
    selected[:, 1::7] = 17  # duplicates and unsorted selected IDs are intentional
    positions = torch.arange(rows, dtype=torch.int64) * 3 + 31
    if rows:
        selected[0].fill_(-1)
        positions[0] = -1
    selected, positions = selected.to(device), positions.to(device)
    if strided:
        backing = torch.empty((rows, k + 19), dtype=dtype, device=device)
        backing[:, 3 : k + 3].copy_(selected)
        selected = backing[:, 3 : k + 3]
        backing_pos = torch.empty(rows * 2 + 1, dtype=torch.int64, device=device)
        backing_pos[1::2].copy_(positions)
        positions = backing_pos[1::2]
    return selected, positions


class IndexPlanCPU(unittest.TestCase):
    def test_oracle_preserves_valid_order_duplicates_and_padding(self):
        selected = torch.tensor([[7, -3, 2, 7, -1], [-1, -1, -1, -1, -1]])
        actual, lengths = reference_plan(selected, torch.tensor([4, -1]), 10, 20, 3, 4)
        self.assertEqual(actual.shape, (2, 64))
        self.assertEqual(actual[0, :5].tolist(), [17, 12, 17, 30, 31])
        self.assertEqual(lengths.tolist(), [5, 0])
        self.assertTrue((actual[0, 5:] == -1).all())
        self.assertTrue((actual[1] == -1).all())

    def test_oracle_matches_scalar_filter_with_strides_and_empty_rows(self):
        for rows in (0, 1, 32):
            selected, positions = make_inputs(rows, 17, strided=True)
            got, lengths = reference_plan(selected, positions, 7, 4096, 29, 7)
            for row, values in enumerate(selected.tolist()):
                expected = [7 + x for x in values if x >= 0]
                expected += [
                    7 + 4096 + p - 29
                    for p in range(int(positions[row]) - 6, int(positions[row]) + 1)
                    if p >= 29
                ]
                self.assertEqual(lengths[row], len(expected))
                self.assertEqual(got[row, : len(expected)].tolist(), expected)
                self.assertTrue((got[row, len(expected) :] == -1).all())

    def test_oracle_cast_precedes_partition_even_at_integer_wrap(self):
        selected = torch.tensor([[2**31 - 1, -9, 3]], dtype=torch.int64)
        got, lengths = reference_plan(selected, torch.tensor([2**31]), 1, 7, 0, 1)
        self.assertEqual(lengths.tolist(), [1])
        self.assertEqual(got[0, :4].tolist(), [4, -(2**31), -1, -(2**31) + 8])

    def test_cpu_and_tensor_metadata_fall_back_without_cuda_queries(self):
        selected, positions = make_inputs(32)
        with patch.object(
            fused, "_device_supported", side_effect=AssertionError("CUDA query")
        ):
            self.assertIsNone(
                fused.try_build_index_plan(selected, positions, 0, 4096, 0)
            )
            self.assertFalse(
                fused.is_supported(selected, positions, torch.tensor(0), 1, 0, 128)
            )

    def test_metadata_shape_dtype_platform_capacity_guards(self):
        def inputs():
            device = torch.device("cuda:0")
            selected = SimpleNamespace(
                is_cuda=True,
                device=device,
                dtype=torch.int32,
                ndim=2,
                shape=(32, 512),
                stride=lambda axis: (531, 1)[axis],
            )
            positions = SimpleNamespace(
                device=device,
                dtype=torch.int64,
                ndim=1,
                shape=(32,),
                stride=lambda axis: 2,
            )
            return selected, positions

        with patch.object(fused, "_prefill_index_plan_kernel", object()), patch.object(
            fused, "_device_supported", return_value=True
        ):
            self.assertTrue(fused.is_supported(*inputs(), 0, 32768, 0, 128))
            for change in (
                {"dtype": torch.float32},
                {"shape": (131073, 512)},
                {"shape": (32, 2048)},
                {"stride": lambda axis: 2},
            ):
                x, p = inputs()
                vars(x).update(change)
                self.assertFalse(fused.is_supported(x, p, 0, 32768, 0, 128))
            for scalars in (
                (True, 1, 0, 128),
                (0, 1, 0, 0),
                (-1, 1, 0, 128),
                (2**31 - 1, 1, 0, 128),
                (0, 1, 2**31, 128),
            ):
                self.assertFalse(fused.is_supported(*inputs(), *scalars))
            x, p = inputs()
            p.shape = (31,)
            self.assertFalse(fused.is_supported(x, p, 0, 32768, 0, 128))
            with patch.object(torch.version, "hip", "test"):
                self.assertFalse(fused.is_supported(*inputs(), 0, 32768, 0, 128))
            with patch.object(fused, "_device_supported", return_value=False):
                self.assertFalse(fused.is_supported(*inputs(), 0, 32768, 0, 128))


@unittest.skipUnless(
    torch.cuda.is_available(), "CUDA required; reserve a GPU explicitly"
)
class IndexPlanCUDA(unittest.TestCase):
    def setUp(self):
        if fused.triton is None or torch.cuda.get_device_capability() not in (
            (10, 0),
            (10, 3),
        ):
            self.skipTest("SM100/SM103 and Triton required")

    def check(self, selected, positions, offset=0, size=4096, start=0, window=128):
        before = selected.clone(), positions.clone()
        expected = reference_plan(selected, positions, offset, size, start, window)
        actual = fused.try_build_index_plan(
            selected, positions, offset, size, start, window
        )
        self.assertIsNotNone(actual)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
        torch.testing.assert_close(selected, before[0], rtol=0, atol=0)
        torch.testing.assert_close(positions, before[1], rtol=0, atol=0)

    def test_small_and_captured_rows_bitexact(self):
        for rows in (0, 1, 17, 32, 1664, 8192, 32768):
            with self.subTest(rows=rows):
                self.check(*make_inputs(rows, device="cuda", strided=bool(rows % 2)))

    def test_shapes_dtypes_padding_offsets_and_invalid_holes(self):
        for k, window in ((1, 1), (17, 7), (511, 127), (512, 128), (1024, 128)):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(k=k, window=window, dtype=dtype):
                    x, p = make_inputs(32, k, device="cuda", strided=True, dtype=dtype)
                    self.check(x, p, 13, 4096, 47, window)
                    x.fill_(-1)
                    self.check(x, p, 0, 0, 10**6, window)

    def test_int64_positions_and_int32_wrap_match_eager(self):
        x = torch.tensor([[2**31 - 1, -9, 3], [2**40, 2**31, -1]], device="cuda")
        p = torch.tensor([2**31, -(2**40)], device="cuda")
        self.check(x, p, 1, 7, 0, 7)

    def test_graph_replay_outputs_alias_checks_and_changed_inputs(self):
        x, p = make_inputs(32, device="cuda", strided=True)
        out = torch.empty((32, 640), device="cuda", dtype=torch.int32)
        lengths = torch.empty(32, device="cuda", dtype=torch.int32)
        call = lambda: fused.try_build_index_plan(
            x, p, 0, 4096, 9, out=out, lengths_out=lengths
        )
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call()
        self.assertIs(captured[0], out)
        self.assertIs(captured[1], lengths)
        for shift in (100, 700):
            p.add_(shift)
            x[:, ::3] = -1
            out.fill_(123456)
            lengths.fill_(-99)
            graph.replay()
            for got, ref in zip(captured, reference_plan(x, p, 0, 4096, 9)):
                torch.testing.assert_close(got, ref, rtol=0, atol=0)
        self.assertIsNone(fused.try_build_index_plan(x, p, 0, 4096, 0, out=out.t()))
        self.assertIsNone(
            fused.try_build_index_plan(
                x, p, 0, 4096, 0, out=out, lengths_out=out.flatten()[:32]
            )
        )
        with patch.object(fused, "_prefill_index_plan_kernel") as kernel:
            kernel.__getitem__.return_value.side_effect = RuntimeError("launch failure")
            with self.assertRaisesRegex(RuntimeError, "launch failure"):
                call()


if __name__ == "__main__":
    unittest.main()
