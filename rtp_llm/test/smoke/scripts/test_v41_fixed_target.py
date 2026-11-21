import unittest

from compare_v41_fixed_target import HISTORY, PROMPT, compare


class FixedFirstVerify(unittest.TestCase):
    def setUp(self):
        self.old = dict(
            role="oldfst",
            nonce="a" * 32,
            history=HISTORY,
            prompt=PROMPT,
            prefix=512,
            anchor=1606,
            candidates=[11556, 2521, 304],
            scores={"305": 28.5, "16": 18.0},
            target=305,
            first_committed=305,
        )
        self.new = dict(
            self.old,
            role="newfst",
            scores={"305": 21.0, "16": 28.0},
            target=16,
            first_committed=16,
        )

    def test_both_columns_same_history(self):
        d = compare(self.old, self.new)
        self.assertEqual(d["target_ids"], [305, 16])
        self.assertEqual(d["new_minus_old"], {"305": -7.5, "16": 10.0})

    def test_fail_closed(self):
        for change in (
            dict(candidates=[6271, 28, 270]),
            dict(nonce="f" * 32),
            dict(anchor=28),
            dict(prefix=0),
            dict(history="0" * 64),
            dict(scores={"305": float("nan"), "16": 1}),
        ):
            with self.assertRaises(ValueError):
                compare(self.old, dict(self.new, **change))
        with self.assertRaises(ValueError):
            compare(dict(self.old, target=16, first_committed=16), self.new)
