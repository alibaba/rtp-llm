import unittest

from rtp_llm.utils.awi_weights import _balanced_owner_map


class BalancedOwnerMapTest(unittest.TestCase):
    def test_balances_large_tensors_and_keeps_excluded_ranks_empty(self):
        sizes = {"large": 700, "medium": 300, "small": 20, "tiny": 5}
        owners = _balanced_owner_map(sizes, [1, 2, 3])
        self.assertEqual(set(owners), set(sizes))
        self.assertTrue(set(owners.values()) <= {1, 2, 3})
        loads = {rank: 0 for rank in (1, 2, 3)}
        for name, owner in owners.items():
            loads[owner] += sizes[name]
        self.assertLessEqual(max(loads.values()) - min(loads.values()), 300)

    def test_each_name_has_exactly_one_owner(self):
        sizes = {f"k{i}": i + 1 for i in range(20)}
        owners = _balanced_owner_map(sizes, [0, 1, 2, 3])
        self.assertEqual(len(owners), len(sizes))
        self.assertEqual(sorted(owners), sorted(sizes))


if __name__ == "__main__":
    unittest.main()
