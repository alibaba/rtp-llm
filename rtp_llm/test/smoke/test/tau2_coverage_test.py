import json
import tempfile
import unittest
from pathlib import Path

from smoke.tau2_coverage import require_all_reviews


class Tau2CoverageTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        self.manifest = self.work / "tasks.json"
        self.manifest.write_text(json.dumps({"retail": ["17", "75"], "airline": ["4"]}))
        self.report = self.work / "current/reports/model/score.json"
        self.report.parent.mkdir(parents=True)
        self.report.write_text('{"score": 1}')
        self.reviews = self.work / "current/reviews/model"
        self.reviews.mkdir(parents=True)
        self.write_reviews("airline", ["4"])

    def write_reviews(self, domain, ids):
        rows = [json.dumps({"sample_score": {"sample_metadata": {"id": k}}}) for k in ids]
        (self.reviews / f"tau2_bench_{domain}.jsonl").write_text("\n".join(rows))

    def check(self):
        require_all_reviews(self.work, self.report, self.manifest, "model")

    def test_complete_out_of_order(self):
        self.write_reviews("retail", ["75", "17"])
        self.check()

    def test_missing_repeated_or_substituted_task_fails_even_with_full_score(self):
        for ids in (["17"], ["17", "17"], ["17", "99"]):
            with self.subTest(ids=ids):
                self.write_reviews("retail", ids)
                with self.assertRaisesRegex(ValueError, "retail"):
                    self.check()

    def test_missing_domain_or_score_from_another_run_cannot_pass(self):
        with self.assertRaises(FileNotFoundError):
            self.check()
        self.write_reviews("retail", ["17", "75"])
        self.report = self.work / "previous/reports/model/score.json"
        with self.assertRaisesRegex(ValueError, "same tau2 run"):
            self.check()


if __name__ == "__main__":
    unittest.main()
