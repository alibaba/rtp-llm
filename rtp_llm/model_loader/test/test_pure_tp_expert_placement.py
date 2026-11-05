"""CPU checks of the production multi-node EP=1 placement implementation."""

import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.utils.expert_placement import pure_tp_identity_placement


class PureTpExpertPlacementTest(unittest.TestCase):
    def test_all_layers_retain_all_512_experts_for_each_physical_topology(self):
        for nodes in (2, 4, 8):
            with self.subTest(nodes=nodes):
                placement = pure_tp_identity_placement(60, 512, 512, nodes)
                self.assertEqual(placement, [list(range(512)) for _ in range(60)])
                placement[0][0] = 511
                self.assertEqual(placement[1][0], 0)

    def test_custom_identity_mapping_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "placement.json"
            expected = [list(range(4)), list(range(4))]
            path.write_text(json.dumps(expected))
            self.assertEqual(pure_tp_identity_placement(2, 4, 4, 4, str(path)), expected)

    def test_custom_missing_permuted_or_nonidentity_layers_are_rejected(self):
        invalid = (
            [],
            [[0, 1, 2, 3]],
            [[0, 1, 2, 3], [0, 1, 2]],
            [[0, 1, 2, 3], [1, 0, 2, 3]],
            [[0, 1, 2, 3], [0, 1, 2, 2]],
            [[0, 1, 2, 3], [False, 1, 2, 3]],
            [[0, 1, 2, 3], [0.0, 1, 2, 3]],
            {"layers": [[0, 1, 2, 3], [0, 1, 2, 3]]},
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "placement.json"
            for placement in invalid:
                with self.subTest(placement=placement):
                    path.write_text(json.dumps(placement))
                    with self.assertRaisesRegex(ValueError, "complete identity"):
                        pure_tp_identity_placement(2, 4, 4, 4, str(path))

    def test_redundant_experts_and_invalid_dimensions_are_rejected(self):
        for physical in (3, 5, 8):
            with self.subTest(physical=physical):
                with self.assertRaisesRegex(ValueError, "nonredundant"):
                    pure_tp_identity_placement(2, 4, physical, 4)
        for position in range(4):
            for invalid in (0, -1, True, 1.5, "4"):
                values = [2, 4, 4, 4]
                values[position] = invalid
                with self.subTest(position=position, value=invalid):
                    with self.assertRaises(ValueError):
                        pure_tp_identity_placement(*values)
        with self.assertRaisesRegex(ValueError, "num_nodes > 1"):
            pure_tp_identity_placement(2, 4, 4, 1)


if __name__ == "__main__":
    unittest.main()
