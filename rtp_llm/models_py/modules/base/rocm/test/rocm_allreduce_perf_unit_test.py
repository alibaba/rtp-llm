import unittest

from rocm_allreduce_perf_lib import (
    BackendSpec,
    BenchShape,
    ResultRow,
    format_result_row,
    generate_shapes,
    parse_backends,
    parse_byte_target,
    parse_shapes,
    summarize_us,
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


class RocmAllReducePerfBackendAndRowTest(unittest.TestCase):
    def test_parse_backends_maps_quick_reduce_quantization(self):
        specs = parse_backends("rccl,trt,vllm_custom,quick_reduce_int6")
        self.assertEqual(
            specs,
            [
                BackendSpec(name="rccl", quantization=None),
                BackendSpec(name="trt", quantization=None),
                BackendSpec(name="vllm_custom", quantization=None),
                BackendSpec(name="quick_reduce_int6", quantization="INT6"),
            ],
        )

    def test_parse_backends_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            parse_backends("rccl,bad_backend")

    def test_summarize_us_returns_avg_percentiles_and_extremes(self):
        summary = summarize_us([4.0, 1.0, 3.0, 2.0])
        self.assertEqual(summary["avg_us"], 2.5)
        self.assertEqual(summary["p50_us"], 2.5)
        self.assertEqual(summary["p90_us"], 3.7)
        self.assertEqual(summary["min_us"], 1.0)
        self.assertEqual(summary["max_us"], 4.0)

    def test_format_result_row_includes_ok_and_skip_fields(self):
        ok_row = ResultRow(
            backend="rccl",
            dtype="fp16",
            target="1_token",
            shape="1x5120",
            bytes=10240,
            status="OK",
            avg_us=12.3,
            p50_us=12.0,
            p90_us=13.0,
            min_us=11.0,
            max_us=14.0,
            algbw_GBps=0.83,
            note="",
        )
        skip_row = ResultRow(
            backend="quick_reduce_int8",
            dtype="fp16",
            target="1_token",
            shape="1x5120",
            bytes=10240,
            status="SKIP",
            avg_us=None,
            p50_us=None,
            p90_us=None,
            min_us=None,
            max_us=None,
            algbw_GBps=None,
            note="below min size",
        )

        self.assertIn("rccl", format_result_row(ok_row))
        self.assertIn("12.300", format_result_row(ok_row))
        self.assertIn("quick_reduce_int8", format_result_row(skip_row))
        self.assertIn("below min size", format_result_row(skip_row))


if __name__ == "__main__":
    unittest.main()
