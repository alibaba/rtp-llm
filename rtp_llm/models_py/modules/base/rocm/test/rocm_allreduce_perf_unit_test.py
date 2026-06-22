import unittest

from rocm_allreduce_perf_lib import (
    BenchShape,
    generate_shapes,
    parse_byte_target,
    parse_shapes,
)


class RocmAllReducePerfShapeTest(unittest.TestCase):
    def test_parse_byte_target_supports_tokens_kb_and_mb(self):
        self.assertEqual(parse_byte_target("1_token"), ("token", 1))
        self.assertEqual(parse_byte_target("16_token"), ("token", 16))
        self.assertEqual(parse_byte_target("32KB"), ("bytes", 32 * 1024))
        self.assertEqual(parse_byte_target("64MB"), ("bytes", 64 * 1024 * 1024))

    def test_parse_byte_target_rejects_bad_values(self):
        for value in ("token", "0_token", "3GB", "abc"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_byte_target(value)

    def test_generate_shapes_from_token_and_byte_targets(self):
        shapes = generate_shapes(
            byte_targets=["1_token", "32KB"],
            hidden_sizes=[4096, 5120],
            dtype_size=2,
            one_token_hidden_size=5120,
            max_bytes=64 * 1024 * 1024,
            explicit_shapes=None,
        )

        self.assertIn(
            BenchShape(target="1_token", rows=1, hidden_size=5120, dtype_size=2),
            shapes,
        )
        self.assertIn(
            BenchShape(target="32KB", rows=4, hidden_size=4096, dtype_size=2),
            shapes,
        )
        self.assertIn(
            BenchShape(target="32KB", rows=4, hidden_size=5120, dtype_size=2),
            shapes,
        )

    def test_generate_shapes_filters_above_max_bytes(self):
        shapes = generate_shapes(
            byte_targets=["64MB"],
            hidden_sizes=[4096],
            dtype_size=2,
            one_token_hidden_size=5120,
            max_bytes=32 * 1024 * 1024,
            explicit_shapes=None,
        )

        self.assertEqual(shapes, [])

    def test_explicit_shapes_override_byte_generation(self):
        shapes = generate_shapes(
            byte_targets=["64MB"],
            hidden_sizes=[4096],
            dtype_size=2,
            one_token_hidden_size=5120,
            max_bytes=64 * 1024 * 1024,
            explicit_shapes=parse_shapes("1x5120,2x4096"),
        )

        self.assertEqual(
            shapes,
            [
                BenchShape(target="1x5120", rows=1, hidden_size=5120, dtype_size=2),
                BenchShape(target="2x4096", rows=2, hidden_size=4096, dtype_size=2),
            ],
        )


if __name__ == "__main__":
    unittest.main()
