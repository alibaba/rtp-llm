import unittest

from rtp_llm.models_py.new_models.deepseek_v3.score_checkpoint import score_name_filter


class ScoreOwnershipTest(unittest.TestCase):
    def test_only_declared_inactive_owners_are_excluded(self):
        keep = score_name_filter(2, 4, 1)
        self.assertEqual(
            [keep(f"model.layers.{i}.self_attn.q_b_proj.weight") for i in range(6)],
            [True, True, False, False, False, True],
        )
        for unexpected in ("alien.weight", "model.layers.invalid.gate.weight", "model.layers.99.mlp.weight"):
            self.assertTrue(keep(unexpected))  # allow strict dispatch to report the error

    def test_no_appended_mtp_and_invalid_depth(self):
        self.assertTrue(score_name_filter(4, 4, 0)("model.layers.4.x"))
        for shape in ((5, 4, 0), (4, 4, -1), (0, 4, 0), (4, 4, True)):
            with self.assertRaises(ValueError):
                score_name_filter(*shape)


if __name__ == "__main__":
    unittest.main()
