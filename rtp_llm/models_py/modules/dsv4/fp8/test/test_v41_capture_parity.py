"""Compare independently captured same-mode MoE boundaries byte-for-byte."""

import hashlib
import json
import os
import unittest
from pathlib import Path

import torch


class V41CaptureParity(unittest.TestCase):
    def test_same_mode_captures(self):
        baseline = Path(os.environ["DSV41_BASELINE_FIXTURES"])
        candidate = Path(os.environ["DSV41_CANDIDATE_FIXTURES"])
        files = sorted(path.name for path in baseline.glob("rank*-layer*-chunk*.pt"))
        self.assertEqual(len(files), 8)
        self.assertEqual(
            files,
            sorted(path.name for path in candidate.glob("rank*-layer*-chunk*.pt")),
        )
        report = {
            "baseline": str(baseline),
            "candidate": str(candidate),
            "fixtures": [],
            "passed": True,
        }
        for name in files:
            before = torch.load(baseline / name, map_location="cpu", weights_only=True)
            after = torch.load(candidate / name, map_location="cpu", weights_only=True)
            entry = {"file": name, "tensors": {}}
            for field in ("rank", "layer", "call", "segments", "collectives"):
                self.assertEqual(before[field], after[field], (name, field))
            removed = after.get("removed_intermediates", [])
            self.assertIn(removed, ([], ["shared_hidden"]))
            self.assertEqual(
                set(before["tensors"]) - set(removed), set(after["tensors"]), name
            )
            for boundary, expected in before["tensors"].items():
                if boundary in removed:
                    continue
                actual = after["tensors"][boundary]
                self.assertEqual(actual.shape, expected.shape, (name, boundary))
                self.assertEqual(actual.dtype, expected.dtype, (name, boundary))
                a = actual.contiguous().view(torch.uint8).reshape(-1)
                b = expected.contiguous().view(torch.uint8).reshape(-1)
                same = torch.equal(a, b)
                result = {
                    "exact": same,
                    "bytes": a.numel(),
                    "shape": list(actual.shape),
                    "baseline_sha256": hashlib.sha256(
                        memoryview(b.numpy())
                    ).hexdigest(),
                    "candidate_sha256": hashlib.sha256(
                        memoryview(a.numpy())
                    ).hexdigest(),
                }
                if not same:
                    mismatch = torch.nonzero(a != b).flatten()
                    result.update(
                        mismatched_bytes=mismatch.numel(), first_byte=int(mismatch[0])
                    )
                    report["passed"] = False
                entry["tensors"][boundary] = result
            report["fixtures"].append(entry)
            del before, after
        before_output = baseline.parent / "warmup-output.pt"
        after_output = candidate.parent / "warmup-output.pt"
        report["final_output"] = {
            "available": before_output.exists() and after_output.exists()
        }
        if report["final_output"]["available"]:
            before = torch.load(before_output, map_location="cpu", weights_only=True)
            after = torch.load(after_output, map_location="cpu", weights_only=True)
            self.assertEqual(before["token"], after["token"])
            for key in ("hidden", "logits"):
                self.assertEqual(before[key].shape, after[key].shape)
                self.assertEqual(before[key].dtype, after[key].dtype)
                exact = torch.equal(
                    before[key].contiguous().view(torch.uint8),
                    after[key].contiguous().view(torch.uint8),
                )
                report["final_output"][key] = {
                    "exact": exact,
                    "max_abs_difference": float(
                        (before[key].float() - after[key].float()).abs().max()
                    ),
                }
                report["passed"] &= exact
        Path(os.environ["DSV41_PARITY_OUTPUT"]).write_text(
            json.dumps(report, indent=2) + "\n"
        )
        self.assertTrue(
            report["passed"], "Same-mode boundary mismatch; inspect parity output"
        )


if __name__ == "__main__":
    unittest.main()
