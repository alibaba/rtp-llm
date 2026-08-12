import argparse
import unittest

from rtp_llm.server.server_args.util import (
    MAX_RUNTIME_MEMORY_MIB,
    bounded_float,
    bounded_int,
    non_negative_mib_int,
)


class BoundedConverterTest(unittest.TestCase):
    def assertArgumentError(self, converter, value, *fragments):
        with self.assertRaises(argparse.ArgumentTypeError) as caught:
            converter(value)
        message = str(caught.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def test_bounded_int_boundaries_and_invalid_inputs(self):
        self.assertEqual(bounded_int("0", max_value=3), 0)
        self.assertEqual(bounded_int("3", max_value=3), 3)

        for value in ("-1", "4"):
            with self.subTest(value=value):
                self.assertArgumentError(
                    lambda raw: bounded_int(raw, max_value=3), value, "[0, 3]", repr(value)
                )
        for value in ("", "x"):
            with self.subTest(value=value):
                self.assertArgumentError(bounded_int, value, "must be an integer", repr(value))

    def test_bounded_float_boundaries_and_invalid_inputs(self):
        self.assertEqual(bounded_float("0", max_value=1.0), 0.0)
        self.assertEqual(bounded_float("1", max_value=1.0), 1.0)
        exclusive = lambda raw: bounded_float(raw, max_value=1.0, max_value_exclusive=True)

        for value in ("1", "nan", "inf", "-0.1"):
            with self.subTest(value=value):
                self.assertArgumentError(exclusive, value, "finite", repr(value))
        self.assertArgumentError(bounded_float, "", "must be a number", "''")

    def test_runtime_memory_mib_limit_matches_size_t_conversion(self):
        self.assertEqual(MAX_RUNTIME_MEMORY_MIB, (1 << 64) // (1024 * 1024) - 1)
        self.assertEqual(
            bounded_int(str(MAX_RUNTIME_MEMORY_MIB), max_value=MAX_RUNTIME_MEMORY_MIB),
            MAX_RUNTIME_MEMORY_MIB,
        )
        self.assertArgumentError(
            lambda raw: bounded_int(raw, max_value=MAX_RUNTIME_MEMORY_MIB),
            str(MAX_RUNTIME_MEMORY_MIB + 1),
            str(MAX_RUNTIME_MEMORY_MIB),
        )

    def test_lenient_reserve_converter_keeps_value_error_contract(self):
        self.assertEqual(non_negative_mib_int("0"), 0)
        for value in ("-1", str(MAX_RUNTIME_MEMORY_MIB + 1), "x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                non_negative_mib_int(value)


if __name__ == "__main__":
    unittest.main()
