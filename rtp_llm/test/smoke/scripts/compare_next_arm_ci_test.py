import io
import json
import unittest
from zipfile import ZipFile

from compare_next_arm_ci import SUITES, first_difference, read_case


def fixture(diag=False, invalid="", old=False):
    suite = SUITES[1]
    lab = suite + ("_sample_diag" if diag else "")
    mem, inner = io.BytesIO(), io.BytesIO()
    with ZipFile(inner, "w") as w:
        events = []
        for rank in range(2):
            pid = str(100 + rank)
            w.writestr("main_logs/batch_record/owner-" + pid + "-1/manifest.json", json.dumps({"world_rank": rank}))
            method = "scratch" if invalid == "reader" and rank else "fastsafetensors"
            events.append("[process-" + pid + "] " + ("finally choose load method: " if old else
                          "model loaded successfully: model_type=qwen35_moe selected=") + method)
            events.append("[process-" + pid + "] Streamed checkpoint tensors with fastsafetensors")
        for group, n in (("single", 1), ("batch", 3)):
            w.writestr("next-prefix-v1-" + group + ".json", json.dumps({"version": 1, "rows": [["a" * 64]] * n}))
        if invalid == "duplicate":
            w.writestr("next-prefix-v1-batch.duplicate", "")
        if diag or invalid == "contaminated":
            for slot in range(4):
                if invalid == "incomplete" and slot == 2:
                    continue
                rank = int(slot == 0)
                if invalid == "rank":
                    rank = 1 - rank
                events.append(f"[{100+rank}:1] [RANK {rank}] NEXT_SMOKE_AUDIT_V1 slot={slot} step=0 pick=1 margin=0.125")
        w.writestr("main_logs/process.log", "\n".join(events))
    with ZipFile(mem, "w") as z:
        z.writestr("fake/" + lab + "/test.outputs/outputs.zip", inner.getvalue())
        z.writestr("fake/" + lab + "/test.log", "total count:[2], suc count:[2], compare diff count:[0], visit_failed_count:[0], other_count: [0]")
    return ZipFile(mem)


class CompareTest(unittest.TestCase):
    def test_routes_and_native_rank_evidence(self):
        for old in (True, False):
            for diag in (True, False):
                with fixture(diag, old=old) as z:
                    good, rows, evidence = read_case(z, SUITES[1], "old" if old else "new", diag)
                self.assertEqual((good, len(rows), len(evidence)), (2, 4, 4 if diag else 0))

    def test_missing_changed_or_misbound_data_fail_closed(self):
        for diag, reason in ((True,"rank"), (True,"incomplete"), (False,"duplicate"),
                             (False,"contaminated"), (False,"reader")):
            with fixture(diag, reason) as z, self.assertRaises(ValueError):
                read_case(z, SUITES[1], "new", diag)

    def test_prefix_length_and_order(self):
        self.assertEqual(first_difference(["a", "b"], ["a", "c"]), 1)
        self.assertEqual(first_difference(["a"], ["a", "b"]), 1)
        self.assertIsNone(first_difference(["a"], ["a"]))


if __name__ == "__main__":
    unittest.main()
