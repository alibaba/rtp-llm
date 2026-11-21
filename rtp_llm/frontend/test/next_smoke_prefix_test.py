import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.frontend import next_smoke_prefix as audit


class NextPrefixTest(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[2] / "test/smoke/data/model/qwen35"
        queries = json.loads((base / "q_r_35b_nvfp4_py_tp2_cg_sm100_arm.json").read_text())["query_result"]
        self.single, self.batch = [q["query"] for q in queries]
        # Smoke GenerateConfig renders the JSON integer top_p as a float on the wire.
        self.single["generate_config"]["top_p"] = float(self.single["generate_config"]["top_p"])
        out = tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR") or None)
        self.addCleanup(out.cleanup)
        self.path = Path(out.name)
        env = patch.dict(os.environ, {
            "RTP_NEXT_SMOKE_PREFIX": "1", "TEST_TARGET": sorted(audit._TARGETS)[0],
            "TEST_UNDECLARED_OUTPUTS_DIR": out.name,
        })
        env.start()
        self.addCleanup(env.stop)

    @staticmethod
    def response(ids, finished=True):
        out = NS(output_ids=torch.tensor([ids], dtype=torch.int32), finished=finished)
        return NS(generate_outputs=NS(generate_outputs=[out]))

    def test_fixture_admission_and_unrelated_rejection(self):
        for target in audit._TARGETS:
            with patch.dict(os.environ, TEST_TARGET=target):
                self.assertTrue(audit.approve("single", {**self.single, "__request_id__": 123, "headers": {}}))
                self.assertTrue(audit.approve("batch", self.batch))
                self.assertIsNone(audit.approve("single", {**self.single, "unknown": 1}))
                self.assertIsNone(audit.approve("batch", {**self.batch, "prompt_batch": ["other"] * 3}))
                self.assertIsNone(audit.approve("single", {**self.single, "generate_config": {}}))
        for env in ({"TEST_TARGET": "//rtp_llm/test/smoke:another"}, {"RTP_NEXT_SMOKE_PREFIX": "0"},
                    {"TEST_UNDECLARED_OUTPUTS_DIR": ""}):
            with patch.dict(os.environ, **env):
                self.assertIsNone(audit.approve("single", self.single))

    def test_terminal_cpu_only_and_no_raw_ids(self):
        path = audit.approve("single", self.single)
        res = self.response([1234567, 7654321])
        original = res.generate_outputs.generate_outputs[0].output_ids.clone()
        audit.capture(path, [self.response([1234567], False)])
        self.assertFalse(path.exists())
        audit.capture(path, [res])
        self.assertTrue(torch.equal(original, res.generate_outputs.generate_outputs[0].output_ids))
        data = json.loads(path.read_text())
        self.assertEqual(set(data), {"version", "rows"})
        self.assertEqual([len(row) for row in data["rows"]], [2])
        self.assertNotIn("1234567", path.read_text())
        self.assertNotIn("7654321", path.read_text())
        audit.capture(path, [res])
        self.assertTrue(path.with_suffix(".duplicate").exists())

    def test_batch_order_and_first_different_token(self):
        path = audit.approve("batch", self.batch)
        audit.capture(path, [self.response([13, 17]), self.response([13, 18]), self.response([13, 17])])
        a, b, c = json.loads(path.read_text())["rows"]
        self.assertEqual(a, c)
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])

    def test_invalid_shapes_are_not_diagnostics(self):
        for t in (None, torch.tensor([[1.0]]), torch.zeros(2, 2, dtype=torch.int32),
                  torch.arange(21).reshape(1, 21), torch.tensor([[-1]])):
            self.assertIsNone(audit._row(t))

    def test_request_cannot_provide_internal_capture_file(self):
        injected = str(self.path / "attacker-chosen.json")
        bodies = ({**self.single, "_next_fixture_prefix_file": injected},
                  {**self.single, "generate_config": {**self.single["generate_config"],
                                                      "_next_fixture_prefix_file": injected}})
        for body in bodies:
            self.assertIsNone(audit.approve("single", body))
        with patch.object(audit, "_row", side_effect=AssertionError("must not inspect tokens")):
            audit.capture(injected, [self.response([1])])  # JSON cannot transmit a Path.
            with patch.dict(os.environ, TEST_TARGET="//an/unrelated:target"):
                audit.capture(Path(injected), [self.response([1])])
            self.assertFalse(Path(injected).exists())


if __name__ == "__main__":
    unittest.main()
