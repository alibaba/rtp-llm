import tempfile
import unittest
from pathlib import Path

from eplb_smoke_checks import check_eplb_applied


def applied(model=0, layer=3, update=1, changed=8, rank=0):
    return (
        f"[EPLB] applied model_id={model} layer_id={layer} update={update} "
        f"changed_slots={changed} ep_rank={rank}\n"
    )


class EplbSmokeChecksTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.logs = Path(folder.name)

    def write(self, name, text):
        path = self.logs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_target_completes_across_worker_and_process_logs(self):
        self.write("process.log", applied())
        self.write("rank1/worker.log.1", applied(rank=1))
        # A repeated stdout record cannot stand in for the other rank.
        self.write("duplicate.log", applied())
        records = check_eplb_applied(self.logs, 2)
        self.assertEqual([item["model_id"] for item in records], [0])
        for record in records:
            self.assertEqual(record["changed_slots"], 8)
            self.assertEqual(record["ep_ranks"], [0, 1])

    def test_target_migration_is_sufficient(self):
        self.write("process.log", applied() + applied(rank=1))
        self.assertEqual(len(check_eplb_applied(self.logs, 2)), 1)

    def test_statistics_or_noop_are_not_migration(self):
        for text in (
            "EPLB monitor sample_count=4\n",
            applied(changed=0) + applied(changed=0, rank=1),
            "",
        ):
            with self.subTest(text=text):
                self.write("process.log", text)
                with self.assertRaisesRegex(AssertionError, "model_id=0"):
                    check_eplb_applied(self.logs, 2)

    def test_draft_migration_is_rejected_even_with_zero_changed_slots(self):
        for changed in (0, 8):
            with self.subTest(changed=changed):
                self.write(
                    "process.log",
                    applied() + applied(rank=1) + applied(model=1, changed=changed),
                )
                with self.assertRaisesRegex(AssertionError, "draft.*model_id=1"):
                    check_eplb_applied(self.logs, 2)

    def test_duplicate_rank_is_not_a_collective_update(self):
        self.write("process.log", applied() * 2)
        with self.assertRaisesRegex(AssertionError, "all 2 ranks"):
            check_eplb_applied(self.logs, 2)

    def test_rank_must_agree_on_layer_update_and_changed_slots(self):
        for peer in (
            applied(layer=4, rank=1),
            applied(update=2, rank=1),
            applied(changed=9, rank=1),
            applied(rank=2),
        ):
            with self.subTest(peer=peer):
                self.write("process.log", applied() + peer)
                with self.assertRaises(AssertionError):
                    check_eplb_applied(self.logs, 2)


if __name__ == "__main__":
    unittest.main()
