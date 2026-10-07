import unittest

from pdfusion_schedule_trace import align, summarize


def fixture(steps=2):
    lines = []
    for step in range(1, steps + 1):
        for rank in range(4):
            common = (
                f"v=1 run=test rank={rank} control_epoch={step} model_step_id={step}"
            )
            lines.append(
                f"SCHEDULE_TRACE event=begin {common} epoch_scope=local_model_order global_plan=local "
                f"start_us={step*100} execute_start_us={step*100+2} snapshot_valid=1 intent=prefill "
                "committed_prefill=1 committed_decode=0 prefill=1 decode=0 fake=0 "
                "reject_batch=0 reject_tokens=0 reject_kv=0"
            )
            lines.append(
                f"SCHEDULE_TRACE event=end {common} end_us={step*100+90} output_tokens=1 ok=1"
            )
    return lines


class TraceTest(unittest.TestCase):
    def test_valid_explicit_ids(self):
        rows, metadata = align(fixture(), "test")
        self.assertEqual(metadata["rank_begin_counts"], [2] * 4)
        self.assertEqual(
            summarize(rows)["prefill_present"]["prefill_rank_hist"], {4: 2}
        )

    def test_graph_observation_uses_explicit_brackets(self):
        lines = fixture()
        lines.insert(
            1,
            "MODEL_EXECUTION rank=0 mode=graph prefill=0 batch=1 graph_bs=4 count=1 real=1",
        )
        rows, _ = align(lines, "test")
        summary = summarize(rows)["prefill_present"]
        self.assertEqual(summary["observed_graph_hist"], {4: 1})
        self.assertEqual(summary["model_calls_unobserved"], 7)

    def test_missing_middle_step_not_realigned(self):
        lines = [
            x for x in fixture(3) if not ("rank=2 " in x and "model_step_id=2 " in x)
        ]
        with self.assertRaises(ValueError):
            align(lines, "test", allow_drain_tail=True)

    def test_duplicate_rejected(self):
        lines = fixture()
        with self.assertRaises(ValueError):
            align(lines + [lines[0]], "test")

    def test_other_run_rejected(self):
        with self.assertRaises(ValueError):
            align(fixture(), "different")

    def test_failed_execution_rejected(self):
        lines = fixture()
        lines[-1] = lines[-1].replace("ok=1", "ok=0")
        with self.assertRaises(ValueError):
            align(lines, "test")

    def test_explicit_drain_tail_only(self):
        lines = fixture()[:-1]
        with self.assertRaises(ValueError):
            align(lines, "test")
        rows, metadata = align(lines, "test", allow_drain_tail=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(metadata["excluded_drain_tail"], [2])

    def test_failed_partial_tail_is_not_discarded(self):
        lines = fixture()[:-1]
        lines[-2] = lines[-2].replace("ok=1", "ok=0")
        with self.assertRaises(ValueError):
            align(lines, "test", allow_drain_tail=True)

    def test_epoch_divergence_rejected(self):
        lines = [
            x.replace("control_epoch=2", "control_epoch=3") if "rank=2 " in x else x
            for x in fixture()
        ]
        with self.assertRaises(ValueError):
            align(lines, "test")

    def test_nonoverlap_rejected(self):
        lines = [
            (
                x.replace("start_us=200", "start_us=400")
                .replace("execute_start_us=202", "execute_start_us=402")
                .replace("end_us=290", "end_us=490")
                if "rank=2 " in x
                else x
            )
            for x in fixture()
        ]
        with self.assertRaises(ValueError):
            align(lines, "test")


if __name__ == "__main__":
    unittest.main()
