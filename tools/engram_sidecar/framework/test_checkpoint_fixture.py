"""CPU-only coverage tests for the deterministic v2 Engram fixture."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "checkpoint_fixture_under_test", Path(__file__).with_name("checkpoint_fixture.py")
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU Torch is required; no GPU/native RTP is used")
class DecodeWindowTest(unittest.TestCase):
    def test_saved_decode_cases_have_zero_one_three_history_tokens(self):
        expected = {
            0: [5, -1, -1, -1],
            1: [10, 5, -1, -1],
            7: [13, 8, 30, 25],
        }
        for step, window in expected.items():
            with self.subTest(step=step):
                case = fixture.make_inputs(1, step)
                self.assertEqual(case["engram_token_windows"].tolist(), [window])
                self.assertEqual(case["token_mask"].tolist(), [True])

    def test_advancing_decode_shifts_the_actual_previous_window(self):
        previous = fixture.make_inputs(1, 0)["engram_token_windows"]
        for step in range(1, 16):
            current = fixture.make_inputs(1, step)["engram_token_windows"]
            torch.testing.assert_close(current[:, 1:], previous[:, :3], rtol=0, atol=0)
            previous = current

    def test_multirow_cases_keep_history_boundary_image_and_padding(self):
        case = fixture.make_inputs(33, 3)
        windows, mask = case["engram_token_windows"], case["token_mask"]
        self.assertTrue(bool((windows[1] >= 5).all()))
        self.assertEqual(windows[0, 1:].tolist(), [-1, -1, -1])
        self.assertEqual(int(windows[4, 0]), fixture.IMAGE_TOKEN_ID)
        self.assertFalse(bool(mask[4]))
        self.assertEqual(windows[6].tolist(), [-1, -1, -1, -1])
        self.assertFalse(bool(mask[6]))
        self.assertTrue(bool(mask[1]))

    def test_empty_decode_inputs_preserve_shapes_and_dtypes(self):
        case = fixture.make_inputs(0)
        self.assertEqual(tuple(case["engram_token_windows"].shape), (0, 4))
        self.assertEqual(tuple(case["hidden"].shape), (0, 2, 128))
        self.assertEqual(tuple(case["token_mask"].shape), (0,))
        self.assertEqual(case["engram_token_windows"].dtype, torch.int32)
        self.assertEqual(case["hidden"].dtype, torch.bfloat16)


@unittest.skipIf(torch is None, "CPU Torch is required; no GPU/native RTP is used")
class TableIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = {
            layer: fixture.table_rows(layer, torch.arange(rows))
            for layer, rows in fixture.LAYERS
        }

    @staticmethod
    def dequantize(weights, scales):
        return weights.float() * scales.float().repeat_interleave(32, dim=-1)

    def test_all_actual_rows_are_unique_in_bytes_and_dequantized_values(self):
        raw, dequantized = [], []
        for weights, scales in self.tables.values():
            raw.append(torch.cat((weights.view(torch.uint8), scales.view(torch.uint8)), dim=-1))
            dequantized.append(self.dequantize(weights, scales))
        expected_rows = sum(rows for _, rows in fixture.LAYERS)
        self.assertEqual(torch.unique(torch.cat(raw), dim=0).shape[0], expected_rows)
        self.assertEqual(torch.unique(torch.cat(dequantized), dim=0).shape[0], expected_rows)

    def test_adjacent_and_34_row_offsets_always_change_decoded_values(self):
        for layer, (weights, scales) in self.tables.items():
            values = self.dequantize(weights, scales)
            for offset in (1, 34):
                with self.subTest(layer=layer, offset=offset):
                    self.assertTrue(bool((values[:-offset] != values[offset:]).any(dim=-1).all()))

    def test_rows_remain_finite_mild_and_native_quantized_format(self):
        for weights, scales in self.tables.values():
            self.assertEqual(weights.dtype, torch.float8_e4m3fn)
            self.assertEqual(scales.dtype, torch.float8_e8m0fnu)
            self.assertTrue(weights.is_contiguous() and scales.is_contiguous())
            values = self.dequantize(weights, scales)
            self.assertTrue(bool(torch.isfinite(values).all()))
            self.assertLessEqual(float(values.abs().max()), 4.0)
            self.assertTrue(bool((scales.float() > 0).all()))

    def test_selected_multidimensional_rows_match_full_table_bytes(self):
        for layer, rows in fixture.LAYERS:
            ids = torch.tensor([[0, 34, rows - 1], [33, 1, 34]])
            selected = fixture.table_rows(layer, ids)
            for got, whole in zip(selected, self.tables[layer]):
                expected = whole.view(torch.uint8)[ids]
                torch.testing.assert_close(got.view(torch.uint8), expected, rtol=0, atol=0)

    def test_empty_selection_keeps_leading_dimensions(self):
        weights, scales = fixture.table_rows(0, torch.empty((0, 6), dtype=torch.int64))
        self.assertEqual(tuple(weights.shape), (0, 6, 64))
        self.assertEqual(tuple(scales.shape), (0, 6, 2))

    def test_invalid_rows_and_layers_are_rejected(self):
        for layer, ids in ((99, [0]), (0, [-1]), (0, [4097]), (1, [514])):
            with self.subTest(layer=layer, ids=ids), self.assertRaises(ValueError):
                fixture.table_rows(layer, ids)


if __name__ == "__main__":
    unittest.main()
