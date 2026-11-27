import unittest
from unittest.mock import Mock

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

    def test_undeclared_draft_needs_physical_marker(self):
        for marker in (None, 2, 78):
            with self.subTest(marker=marker):
                resolve = Mock(return_value=marker)
                keep = score_name_filter(1, 2, 0, resolve_appended=resolve)
                self.assertTrue(keep("model.layers.0.x"))
                self.assertFalse(keep("model.layers.1.x"))
                resolve.assert_not_called()
                for idx in (2, 78, 79):
                    self.assertEqual(keep(f"model.layers.{idx}.x"), idx != marker)
                resolve.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
