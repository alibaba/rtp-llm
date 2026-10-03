import unittest
from unittest import mock

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl


class BatchPerfImplTest(unittest.TestCase):
    def test_real_output_is_explicit_and_legacy_payload_unchanged(self):
        impl = BatchPerfImpl.__new__(BatchPerfImpl)
        impl.batch_size, impl.dp_size, impl.base_port, impl.is_decode = (
            128,
            8,
            12345,
            False,
        )
        for enabled in ("0", "1"):
            with self.subTest(enabled=enabled), mock.patch.dict(
                "os.environ",
                {"PERF_REAL_OUTPUT": enabled},
            ), mock.patch(
                "rtp_llm.test.perf_test.batch_perf_impl.requests.post"
            ) as post:
                post.return_value.status_code = 200
                post.return_value.json.return_value = {"status": "ok"}
                impl._set_concurrency()
                expected = {"batch_size": 16, "mode": "prefill"}
                if enabled == "1":
                    expected["real_output"] = True
                self.assertEqual(post.call_args.kwargs["json"], expected)

    def test_string_seed_fills_fixed_global_batch(self):
        impl = BatchPerfImpl.__new__(BatchPerfImpl)
        impl.batch_size = 16
        self.assertEqual(impl._normalize_seed_queries("prefix"), ["prefix"] * 16)

    def test_explicit_seed_list_is_preserved(self):
        impl = BatchPerfImpl.__new__(BatchPerfImpl)
        impl.batch_size = 16
        self.assertEqual(impl._normalize_seed_queries(["a", "b"]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
