import random
import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.fmha_validation import main_geometry_checks
from rtp_llm.models_py.modules.qwen4_exp.qsa.validation import DeviceChecks


class EagerValidationTest(unittest.TestCase):
    def test_geometry_matches_independent_wide_integer_host_reference(self):
        rng = random.Random(19)
        values = [-(2**63), -1, 0, 1, 3, 7, 8, 2**31, 2**32 + 1, 2**63 - 1]
        for padded in (False, True):
            for _ in range(300):
                bases = [rng.choice(values) for _ in range(2)]
                lengths = [rng.choice(values) for _ in range(2)]
                table = [
                    [rng.choice([-1, 0, 1, 2, 3, 4]) for _ in range(2)]
                    for _ in range(2)
                ]
                required = [(p + n + 3) // 4 for p, n in zip(bases, lengths)]
                active = [not padded or any(row) for row in table]
                expected = None
                if any(p < 0 or n <= 0 for p, n in zip(bases, lengths)):
                    expected = "sequence bases"
                elif any(page >= 4 for row in table for page in row):
                    expected = "physical block"
                elif any(a and r > 2 for a, r in zip(active, required)):
                    expected = "cover the visible KV"
                elif any(
                    a and col < r and (page <= 0 or page >= 4)
                    for a, r, row in zip(active, required, table)
                    for col, page in enumerate(row)
                ):
                    expected = "unallocated"
                checks = DeviceChecks()
                main_geometry_checks(
                    checks,
                    torch.tensor(bases),
                    torch.tensor(lengths),
                    torch.tensor(table, dtype=torch.int32),
                    4,
                    4,
                    padded,
                )
                with self.subTest(
                    bases=bases, lengths=lengths, table=table, padded=padded
                ):
                    if expected is None:
                        checks.verify()
                    else:
                        with self.assertRaisesRegex(RuntimeError, expected):
                            checks.verify()

    def test_grouped_flags_keep_bit_offsets_and_error_precedence(self):
        for mask, message in (
            (1, "first"),
            (8, "pool"),
            (16, "capacity"),
            (32, "page"),
        ):
            checks = DeviceChecks()
            checks.add(torch.tensor(False), "unrelated")
            checks.add_flags(
                torch.tensor(mask, dtype=torch.int32),
                ["first", None, None, "pool", "capacity", "page"],
            )
            checks.add(torch.tensor(True), "later")
            with self.assertRaisesRegex(RuntimeError, message):
                checks.verify()

    def test_partition_payload_and_verdict_keep_actual_input_check(self):
        expected = torch.tensor([0, 3, 5], dtype=torch.int64)
        actual = torch.tensor([0, 3, 5], dtype=torch.int32)
        checks = DeviceChecks()
        checks.add((actual != expected).any(), "partition", ValueError)
        self.assertEqual(checks.verify(expected), [0, 3, 5])
        actual[1] = 2
        next_step = DeviceChecks()
        next_step.add((actual != expected).any(), "partition", ValueError)
        with self.assertRaisesRegex(ValueError, "partition"):
            next_step.verify(expected)


if __name__ == "__main__":
    unittest.main()
