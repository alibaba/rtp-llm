import time
import unittest
import logging
from maga_transformer.utils.util import has_overlap_kmp

class TestUtil(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basic_overlap(self):
        self.assertTrue(has_overlap_kmp("abcde", "de"))
        self.assertTrue(has_overlap_kmp("hello", "lo"))
        self.assertFalse(has_overlap_kmp("test", "xyz"))

    def test_no_overlap(self):
        self.assertFalse(has_overlap_kmp("abc", "def"))
        self.assertFalse(has_overlap_kmp("aaa", "bbb"))

    def test_edge_cases(self):
        self.assertFalse(has_overlap_kmp("", "a"))  # one is empty
        self.assertFalse(has_overlap_kmp("abc", "a"))  # short b, no overlap
        self.assertTrue(has_overlap_kmp("a", "a"))  # one character, equal

    def test_performance(self):
        test_turn = 100
        start_time = time.perf_counter()
        result = ""
        for _ in range(test_turn):
            # 大量的输入数据，b的长度小于10
            a = "a" * 100000
            b = "aaaaaa"  # b 的长度小于 10
            result = has_overlap_kmp(a, b)
        end_time = time.perf_counter()
        avg_cost_time = (end_time - start_time) / test_turn # unit: second
        self.assertTrue(avg_cost_time < 0.0001)
        logging.info(f"Performance test for large input: Result = {result}, Time taken = {avg_cost_time:.6f} seconds")
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
