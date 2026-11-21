import unittest

from compare_v41_layer_trace import FIXTURE, compare_traces


def trace(role):
    groups = {}
    for k in range(2):
        stages = {}
        for s in ("embedding", "layer_0", "layer_1", "hc_reduce", "norm"):
            stages[s] = dict(
                stage=s,
                rank=k,
                layers=2,
                dtype="torch.bfloat16",
                valid_shape=[391 + k, 3],
                positions_sha=str(k) * 64,
                valid_sha="b" * 64,
                last_sha="c" * 64 if k else None,
            )
        stages["lm_head"] = dict(
            stage="lm_head",
            rank=k,
            world=2,
            hidden_dtype="BF16",
            head_dtype="BF16",
            head_shape=[64640, 5120],
            head_sha="a" * 64,
            last_sha="d" * 64,
            candidate_sha={} if k else {"28": "6" * 64, "1606": "4" * 64},
        )
        groups[str(k)] = stages
    return dict(schema=2, loader=role, input_ids_sha256=FIXTURE, ranks=groups)


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.old = trace("oldfst")
        self.new = trace("newfst")

    def test_identical(self):
        result = compare_traces(self.old, self.new)
        self.assertIsNone(result["first_valid_rows_difference"])
        self.assertIsNone(result["first_last_token_difference"])
        self.assertTrue(result["lm_head_by_rank"]["0"]["weight_equal"])

    def test_other_row_can_diverge_before_last(self):
        self.new["ranks"]["0"]["layer_0"]["valid_sha"] = "d" * 64
        self.new["ranks"]["1"]["layer_1"].update(last_sha="e" * 64, valid_sha="f" * 64)
        self.new["ranks"]["1"]["lm_head"]["last_sha"] = "d" * 63 + "a"
        result = compare_traces(self.old, self.new)
        self.assertEqual(
            result["first_valid_rows_difference"], {"stage": "layer_0", "ranks": [0]}
        )
        self.assertEqual(
            result["first_last_token_difference"], {"stage": "layer_1", "ranks": [1]}
        )
        self.assertFalse(result["lm_head_by_rank"]["1"]["last_hidden_equal"])

    def test_head_weight_only(self):
        self.new["ranks"]["0"]["lm_head"].update(head_sha="1" * 64)
        self.new["ranks"]["0"]["lm_head"]["candidate_sha"]["1606"] = "2" * 64
        result = compare_traces(self.old, self.new)
        self.assertIsNone(result["first_last_token_difference"])
        self.assertFalse(result["lm_head_by_rank"]["0"]["weight_equal"])
        self.assertEqual(
            result["lm_head_by_rank"]["0"]["candidate_rows_equal"],
            {"28": True, "1606": False},
        )

    def test_invalid_traces_do_not_look_equal(self):
        for corrupt in (
            lambda d: d["ranks"].pop("0"),
            lambda d: d["ranks"]["1"].pop("layer_1"),
            lambda d: d.update(input_ids_sha256="0" * 64),
            lambda d: d["ranks"]["1"]["layer_1"].update(last_sha=None),
            lambda d: d["ranks"]["0"]["norm"].update(positions_sha="2" * 64),
            lambda d: d["ranks"]["0"]["lm_head"].update(candidate_sha={}),
            lambda d: d["ranks"]["0"]["lm_head"].update(head_sha="bogus"),
        ):
            new = trace("newfst")
            corrupt(new)
            with self.assertRaises(ValueError):
                compare_traces(self.old, new)


if __name__ == "__main__":
    unittest.main()
