import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios


class RequestShapeTests(unittest.TestCase):
    def test_grade_is_validated_and_preserved_as_instance_metadata(self):
        docs = [("shape.yaml", self.source())]
        self.assertEqual(compile_scenarios(docs)[0]["grade"], "normal")
        self.assertEqual(compile_scenarios(docs, grade="strict")[0]["grade"], "strict")
        with self.assertRaises(ValueError):
            compile_scenarios(docs, grade="invented")

    def source(self):
        source = copy.deepcopy(load_scenarios(ROOT / "scenarios/core")[0][1])
        source["stages"] = source.pop("variants")[0]["stages"]
        return source

    def test_unset_is_not_materialized_and_protocol_values_preserved(self):
        source = self.source()
        params = source["stages"][1]["params"]
        for extra in (
            {},
            {"priority": 0},
            {"priority": 70, "qos_level": 30},
            {"priority": -1},
            {"block_keys": [-(2**63), 0, 2**63 - 1]},
        ):
            source["stages"][1]["params"] = dict(params, **extra)
            compiled = compile_scenarios([("shape.yaml", source)])[0]["stages"][1][
                "params"
            ]
            if "priority" not in extra:
                self.assertNotIn("priority", compiled)
            for key, value in extra.items():
                self.assertEqual(compiled[key], value)

    def test_invalid_shapes_are_rejected_before_backend_setup(self):
        for extra in (
            {"block_keys": []},
            {"block_keys": [True]},
            {"block_keys": [-(2**63) - 1]},
            {"block_keys": [2**63]},
            {"block_keys": [0] * 4097},
            {"priority": True},
            {"priority": 2**31},
            {"qos_level": "70"},
            {"schedule_timeout_s": 0},
            {"stream_timeout_s": 61},
            {"stream_timeout_s": float("inf")},
            {"post_issue_delay_s": -1},
            {"post_issue_delay_s": 3},
        ):
            source = self.source()
            source["stages"][1]["params"].update(extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                compile_scenarios([("shape.yaml", source)])


if __name__ == "__main__":
    unittest.main()
