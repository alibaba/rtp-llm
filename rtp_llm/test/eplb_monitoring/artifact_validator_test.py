"""CPU-only unit tests for artifact_validator (frozen window schema v1).

All expected ratios below are hand-computed literals, not oracle outputs, so
the tests pin the validator's recomputation independently.

Run directly:
    /opt/conda310/bin/python -I \\
        rtp_llm/test/eplb_monitoring/artifact_validator_test.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifact_validator import main as validator_main  # noqa: E402
from artifact_validator import validate_window_artifact, validate_window_sequence


def _artifact(rank_history, hist, ratios, **meta):
    """Build a schema-v1 artifact.  ratios: list per layer of
    (valid, total, rank_activations, moe_lb, moe_lb_window)."""
    layer_count = len(rank_history[0])
    ep_size = len(rank_history[0][0])
    artifact = {
        "schema_version": 1,
        "model_id": 0,
        "world_rank": 0,
        "world_size": 2,
        "ep_size": ep_size,
        "layer_count": layer_count,
        "physical_expert_count": len(hist[0]),
        "mapping_epoch": 0,
        "step_begin": 1,
        "step_end": len(rank_history),
        "sample_count": len(rank_history),
        "window_capacity": max(len(rank_history), 2),
        "physical_expert_activations": hist,
        "rank_history": rank_history,
        "layers": [
            {
                "layer": l,
                "valid": ratios[l][0],
                "total_activations": ratios[l][1],
                "rank_activations": ratios[l][2],
                "moe_lb": ratios[l][3],
                "moe_lb_window": ratios[l][4],
            }
            for l in range(layer_count)
        ],
    }
    artifact.update(meta)
    return artifact


class ValidateWindowArtifactTest(unittest.TestCase):
    def test_frozen_schema_example_accepted(self):
        """The coordinator's frozen example must validate as-is."""
        artifact = {
            "schema_version": 1,
            "model_id": 0,
            "world_rank": 0,
            "world_size": 2,
            "ep_size": 2,
            "layer_count": 1,
            "physical_expert_count": 4,
            "mapping_epoch": 0,
            "step_begin": 1,
            "step_end": 2,
            "sample_count": 2,
            "window_capacity": 2,
            "physical_expert_activations": [[1, 1, 2, 2]],
            "rank_history": [[[2, 4]], [[0, 0]]],
            "layers": [
                {
                    "layer": 0,
                    "valid": True,
                    "total_activations": 6,
                    "rank_activations": [2, 4],
                    "moe_lb": 4.0 / 3.0,
                    "moe_lb_window": 4.0 / 3.0,
                }
            ],
        }
        res = validate_window_artifact(artifact)
        self.assertEqual(res["errors"], [])
        self.assertTrue(res["ok"])

    def test_balanced_window_ratios_2_and_1(self):
        """Steps [4,0],[0,4] -> moe_lb 2, moe_lb_window 1 (literal)."""
        artifact = _artifact(
            rank_history=[[[4, 0]], [[0, 4]]],
            hist=[[2, 2, 2, 2]],
            ratios=[(True, 8, [4, 4], 2.0, 1.0)],
        )
        res = validate_window_artifact(artifact)
        self.assertEqual(res["errors"], [])
        self.assertTrue(res["ok"])

    def test_sample_weighted_ratio_not_unweighted(self):
        """Steps [8,2],[1,1] -> moe_lb 1.5; unweighted 1.3 must be rejected."""
        artifact = _artifact(
            rank_history=[[[8, 2]], [[1, 1]]],
            hist=[[5, 4, 2, 1]],
            ratios=[(True, 12, [9, 3], 1.5, 1.5)],
        )
        res = validate_window_artifact(artifact)
        self.assertEqual(res["errors"], [])
        self.assertTrue(res["ok"])

        bad = _artifact(
            rank_history=[[[8, 2]], [[1, 1]]],
            hist=[[5, 4, 2, 1]],
            ratios=[(True, 12, [9, 3], 1.3, 1.5)],  # unweighted trap
        )
        res = validate_window_artifact(bad)
        self.assertFalse(res["ok"])
        self.assertTrue(any("moe_lb" in e for e in res["errors"]))

    def test_zero_layer_requires_invalid_and_null(self):
        artifact = _artifact(
            rank_history=[[[0, 0]], [[0, 0]]],
            hist=[[0, 0, 0, 0]],
            ratios=[(False, 0, [0, 0], None, None)],
        )
        res = validate_window_artifact(artifact)
        self.assertEqual(res["errors"], [])
        self.assertTrue(res["ok"])

    def test_zero_layer_fabricated_ratio_rejected(self):
        artifact = _artifact(
            rank_history=[[[0, 0]], [[0, 0]]],
            hist=[[0, 0, 0, 0]],
            ratios=[(True, 0, [0, 0], 1.0, 1.0)],
        )
        res = validate_window_artifact(artifact)
        self.assertFalse(res["ok"])
        self.assertTrue(any("zero activations" in e for e in res["errors"]))

    def test_rank_activations_must_match_history(self):
        artifact = _artifact(
            rank_history=[[[2, 4]], [[0, 0]]],
            hist=[[1, 1, 2, 2]],
            ratios=[(True, 6, [4, 2], 4.0 / 3.0, 4.0 / 3.0)],  # swapped ranks
        )
        res = validate_window_artifact(artifact)
        self.assertFalse(res["ok"])
        self.assertTrue(any("rank_activations" in e for e in res["errors"]))

    def test_expert_rank_conservation_enforced(self):
        """hist rank segments must equal rank_activations (contiguous slots)."""
        artifact = _artifact(
            rank_history=[[[2, 4]], [[0, 0]]],
            hist=[[1, 1, 1, 3]],  # rank0 experts sum 2 ok, rank1 sums 4 ok...
            ratios=[(True, 6, [2, 4], 4.0 / 3.0, 4.0 / 3.0)],
        )
        self.assertTrue(validate_window_artifact(artifact)["ok"])
        broken = _artifact(
            rank_history=[[[2, 4]], [[0, 0]]],
            hist=[[3, 1, 2, 2]],  # rank0 segment now 4 != rank_activations 2
            ratios=[(True, 8, [2, 4], 4.0 / 3.0, 4.0 / 3.0)],
        )
        res = validate_window_artifact(broken)
        self.assertFalse(res["ok"])
        self.assertTrue(any("rank 0" in e for e in res["errors"]))

    def test_metadata_and_shape_errors(self):
        good = _artifact(
            rank_history=[[[2, 4]], [[0, 0]]],
            hist=[[1, 1, 2, 2]],
            ratios=[(True, 6, [2, 4], 4.0 / 3.0, 4.0 / 3.0)],
        )
        # missing metadata
        bad = dict(good)
        del bad["ep_size"]
        self.assertFalse(validate_window_artifact(bad)["ok"])
        # wrong schema version
        bad = dict(good, schema_version=2)
        self.assertFalse(validate_window_artifact(bad)["ok"])
        # negative count
        bad = dict(good, rank_history=[[[2, -4]], [[0, 0]]])
        self.assertFalse(validate_window_artifact(bad)["ok"])
        # sample_count != step span
        bad = dict(good, sample_count=1)
        self.assertFalse(validate_window_artifact(bad)["ok"])
        # sample_count > capacity
        bad = dict(good, window_capacity=1)
        self.assertFalse(validate_window_artifact(bad)["ok"])
        # non-divisible experts
        bad = dict(
            good, physical_expert_count=3, physical_expert_activations=[[1, 1, 4]]
        )
        self.assertFalse(validate_window_artifact(bad)["ok"])
        # wrong history shape
        bad = dict(good, rank_history=[[[2, 4]]])
        self.assertFalse(validate_window_artifact(bad)["ok"])
        # bool where int expected
        bad = dict(good, sample_count=True)
        self.assertFalse(validate_window_artifact(bad)["ok"])

    def test_multi_layer_independence(self):
        """Layer 0 zero/invalid, layer 1 with traffic: mixed validity ok."""
        artifact = _artifact(
            rank_history=[[[0, 0], [2, 4]]],
            hist=[[0, 0, 0, 0], [1, 1, 2, 2]],
            ratios=[
                (False, 0, [0, 0], None, None),
                (True, 6, [2, 4], 4.0 / 3.0, 4.0 / 3.0),
            ],
            sample_count=1,
            step_begin=1,
            step_end=1,
            window_capacity=2,
            layer_count=2,
        )
        res = validate_window_artifact(artifact)
        self.assertEqual(res["errors"], [])
        self.assertTrue(res["ok"])

    def test_cli_exit_codes(self):
        artifact = _artifact(
            rank_history=[[[2, 4]], [[0, 0]]],
            hist=[[1, 1, 2, 2]],
            ratios=[(True, 6, [2, 4], 4.0 / 3.0, 4.0 / 3.0)],
        )
        with tempfile.TemporaryDirectory() as d:
            good_path = Path(d) / "good.json"
            good_path.write_text(json.dumps(artifact))
            bad_path = Path(d) / "bad.json"
            bad_artifact = dict(artifact, sample_count=5)
            bad_path.write_text(json.dumps(bad_artifact))
            self.assertEqual(validator_main([str(good_path)]), 0)
            self.assertEqual(validator_main([str(bad_path)]), 1)
            self.assertEqual(validator_main([str(Path(d) / "missing.json")]), 2)


class ValidateWindowSequenceTest(unittest.TestCase):
    def _window(self, step_begin, step_end, epoch=0, model_id=0):
        n = step_end - step_begin + 1
        return _artifact(
            rank_history=[[[2, 4]]] * n,
            hist=[[n, n, 2 * n, 2 * n]],
            ratios=[(True, 6 * n, [2 * n, 4 * n], 4.0 / 3.0, 4.0 / 3.0)],
            step_begin=step_begin,
            step_end=step_end,
            sample_count=n,
            window_capacity=4,
            mapping_epoch=epoch,
            model_id=model_id,
        )

    def test_continuous_sequence_ok(self):
        res = validate_window_sequence([self._window(1, 2), self._window(3, 4)])
        self.assertEqual(res["errors"], [])
        self.assertTrue(res["ok"])
        self.assertTrue(res["chains"][0]["continuous"])

    def test_single_window_sequence_ok(self):
        """No minimum window count at this layer (smoke enforces that)."""
        res = validate_window_sequence([self._window(1, 2)])
        self.assertTrue(res["ok"])

    def test_step_gap_detected(self):
        res = validate_window_sequence([self._window(1, 2), self._window(4, 5)])
        self.assertFalse(res["ok"])
        self.assertTrue(any("gap" in e for e in res["errors"]))

    def test_step_overlap_detected(self):
        res = validate_window_sequence([self._window(1, 3), self._window(3, 4)])
        self.assertFalse(res["ok"])
        self.assertTrue(any("overlap" in e for e in res["errors"]))

    def test_mapping_epoch_change_starts_new_chain(self):
        """Epoch change flushes a partial window, but production step
        numbering does NOT restart: continuity holds across epochs too."""
        res = validate_window_sequence(
            [self._window(1, 2, epoch=0), self._window(3, 3, epoch=1)]
        )
        self.assertEqual(res["errors"], [])
        self.assertTrue(res["ok"])
        self.assertEqual(sorted(res["chains"].keys()), [0, 1])

    def test_cross_epoch_step_gap_detected(self):
        """A remap must not silently drop production steps between windows."""
        res = validate_window_sequence(
            [self._window(1, 2, epoch=0), self._window(4, 4, epoch=1)]
        )
        self.assertFalse(res["ok"])
        self.assertTrue(any("not continuous" in e for e in res["errors"]))

    def test_mapping_epoch_decrease_detected(self):
        res = validate_window_sequence(
            [self._window(1, 2, epoch=1), self._window(3, 4, epoch=0)]
        )
        self.assertFalse(res["ok"])
        self.assertTrue(any("mapping_epoch decreases" in e for e in res["errors"]))

    def test_step_begin_must_be_at_least_one(self):
        """step numbers are production forwards (warmup excluded), 1-based."""
        artifact = self._window(0, 1)
        res = validate_window_artifact(artifact)
        self.assertFalse(res["ok"])
        self.assertTrue(any("step_begin" in e for e in res["errors"]))

    def test_schema_version_must_be_true_int(self):
        artifact = self._window(1, 2)
        artifact["schema_version"] = True
        res = validate_window_artifact(artifact)
        self.assertFalse(res["ok"])
        self.assertTrue(any("schema_version" in e for e in res["errors"]))

    def test_mixed_model_or_rank_rejected(self):
        res = validate_window_sequence(
            [self._window(1, 2), self._window(3, 4, model_id=1)]
        )
        self.assertFalse(res["ok"])
        self.assertTrue(any("mixed model_id" in e for e in res["errors"]))

    def test_empty_sequence_rejected(self):
        self.assertFalse(validate_window_sequence([])["ok"])


if __name__ == "__main__":
    unittest.main()
