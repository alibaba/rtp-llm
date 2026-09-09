import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from online_eval.rt_model import (  # noqa: E402
    PerformanceModel,
    RequestShape,
    extract_prefill_expression_from_master_config,
)


EXPRESSION = (
    "max(10, 1 + 2*log(batchSize + 1) "
    "+ 0.5*sum(hitCacheTokens) "
    "+ 0.25*(sum(hasHitCache)/batchSize) "
    "+ 0.75*(sum(hitCacheTokens/(inputTokens + 1))/batchSize))"
)


class RtModelConfigTest(unittest.TestCase):
    def test_extracts_formula_estimator_expression_from_flexlb_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "master.json"
            path.write_text(
                json.dumps(
                    {
                        "zone_process_setting": {
                            "process_info": {
                                "envs": [
                                    [
                                        "FLEXLB_CONFIG",
                                        json.dumps(
                                            {
                                                "router": {
                                                    "roles": {
                                                        "prefill": {
                                                            "executionTimeEstimator": {
                                                                "type": "FORMULA",
                                                                "expression": EXPRESSION,
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        ),
                                    ]
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            expression = extract_prefill_expression_from_master_config(str(path))

        self.assertEqual(EXPRESSION, expression)
        model = PerformanceModel({"prefill": {"expression": expression}})
        request = RequestShape(
            1, input_len=100, output_len=1, block_keys=[], hit_tokens=50
        )
        self.assertGreater(model.prefill_ms([request]), 10)

    def test_non_formula_estimator_has_no_mock_expression(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "master.json"
            path.write_text(
                json.dumps(
                    {
                        "zone_process_setting": {
                            "process_info": {
                                "envs": [
                                    [
                                        "FLEXLB_CONFIG",
                                        json.dumps(
                                            {
                                                "router": {
                                                    "roles": {
                                                        "prefill": {
                                                            "executionTimeEstimator": {
                                                                "type": "LEARNING"
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        ),
                                    ]
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            expression = extract_prefill_expression_from_master_config(str(path))

        self.assertIsNone(expression)


if __name__ == "__main__":
    unittest.main()
