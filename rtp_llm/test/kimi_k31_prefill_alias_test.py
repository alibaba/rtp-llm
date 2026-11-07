"""CPU regression for sparse Prefill output ownership and alias detection."""
import importlib.util
from pathlib import Path
import unittest


_path = Path(__file__).resolve().parents[1] / "models_py/triton_kernels/causal_conv1d/prefill_state_alias.py"
_spec = importlib.util.spec_from_file_location("prefill_state_alias", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
plan = _module.prefill_conv_alias_sources


class PrefillConvAliasTest(unittest.TestCase):
    def test_last_one_through_five_alias_only_the_running_destination(self):
        for n in range(1, 6):
            row = [-1] * 24
            row[7] = 9
            for point in range(n):
                row[24 - n + point] = point + 1
            row[-1] = 9
            sources, aliases = plan([1024], [0, 2048], [row], 128)
            self.assertEqual(sources, [9])
            self.assertEqual(aliases, [True])

    def test_fixed_checkpoint_input_does_not_copy_unnecessarily(self):
        row = [-1] * 18
        row[15], row[17] = 4, 5
        self.assertEqual(plan([2048], [0, 256], [row], 128), ([0], [False]))

    def test_one_token_and_history_short_suffixes(self):
        for length in (1, 2, 3, 7):
            row = [-1] * 9
            row[7] = row[8] = 1
            self.assertEqual(plan([1024], [0, length], [row], 128), ([1], [True]))

    def test_cold_and_mixed_batch(self):
        self.assertEqual(plan([0], [0, 128], [[1]], 128), ([0], [False]))
        a, b = [-1] * 9, [-1] * 9
        a[7] = a[8] = 1
        b[7], b[8] = 2, 3
        self.assertEqual(plan([1024, 1024], [0, 1, 2], [a, b], 128), ([1, 0], [True, False]))

    def test_multiple_outputs_cannot_share_one_destination(self):
        row = [-1] * 10
        row[7] = row[8] = row[9] = 1
        with self.assertRaisesRegex(ValueError, "multiple Prefill conv outputs"):
            plan([1024], [0, 256], [row], 128)

    def test_missing_source_and_truncated_rows_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "missing its initial state"):
            plan([1024], [0, 128], [[-1] * 9], 128)
        with self.assertRaisesRegex(ValueError, "outside the execution row"):
            plan([1024], [0, 128], [[1] * 8], 128)


if __name__ == "__main__":
    unittest.main()
