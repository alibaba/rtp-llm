import math
import unittest

from rtp_llm.test.perf_test.dataclass import ResponseInfo, counted_throughput


def response(**updates):
    aux = dict(
        input_len=100,
        reuse_len=90,
        output_len=16,
        cost_time=1000,
        wait_time=0,
        first_token_cost_time=100,
    )
    aux.update(updates)
    return ResponseInfo({"finished": True, "aux_info": aux}, expected_output_len=16)


class CountedThroughputTest(unittest.TestCase):
    def test_real_counts_and_gpu_normalization(self):
        result = counted_throughput([response(), response()], 2, 2)
        self.assertTrue(result["valid"])
        self.assertEqual(result["generated_tokens"], 32)
        self.assertEqual(result["generated_tps"], 16)
        self.assertEqual(result["generated_tps_per_gpu"], 8)
        self.assertEqual(result["logical_input_tpm"], 6000)
        self.assertEqual(result["uncached_input_tpm"], 600)

    def test_empty_or_failed_mix_is_invalid(self):
        self.assertFalse(counted_throughput([], 1, 1)["valid"])
        result = counted_throughput([response(), ResponseInfo({}, False)], 1, 1)
        self.assertFalse(result["valid"])
        self.assertEqual(result["generated_tokens"], 16)
        self.assertEqual(result["completed_requests"], 1)

    def test_missing_terminal_early_stop_and_invalid_cache_count(self):
        incomplete = ResponseInfo(
            {"aux_info": {"input_len": 100, "output_len": 16, "cost_time": 1000}}
        )
        for sample in (
            incomplete,
            response(output_len=2),
            response(reuse_len=101),
            response(cost_time=math.nan),
            response(cost_time=0),
        ):
            self.assertFalse(counted_throughput([sample], 1, 1)["valid"])

    def test_bad_measurement_window_and_gpu_count(self):
        for elapsed in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                counted_throughput([response()], elapsed, 1)
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                counted_throughput([response()], 1, count)


if __name__ == "__main__":
    unittest.main()
