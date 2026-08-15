import unittest

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl


class BatchPerfImplTest(unittest.TestCase):
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
