import unittest

from pdfusion_schedule_trace import align, summarize, validate_cadence


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


def global_fixture():
    lines = [
        line.replace(
            "epoch_scope=local_model_order global_plan=local",
            "epoch_scope=global_control global_plan=prefill",
        )
        .replace("control_epoch=1", "control_epoch=10")
        .replace("control_epoch=2", "control_epoch=30")
        for line in fixture()
    ]
    for i, line in enumerate(lines):
        if "event=begin" in line:
            line += " ready_prefill=1 ready_decode=7 oldest_ready_us=100 decode_unserved_us=20 ready_mask=15 commit_mask=7 prepare_us=10 control_us=20 commit_us=5"
            if "rank=3 " in line:
                line = (
                    line.replace("committed_prefill=1", "committed_prefill=0")
                    .replace(" prefill=1", " prefill=0")
                    .replace("fake=0", "fake=1")
                )
        elif "rank=3 " in line:
            line = line.replace("output_tokens=1", "output_tokens=0")
        lines[i] = line
    return lines


class TraceTest(unittest.TestCase):
    def test_adaptive_replay_has_no_legacy_ratio_dependency(self):
        for ratio in (None, "", "0", "7", "1/7", "invalid"):
            self.assertEqual(
                validate_cadence([self.untimed_step()], ratio)["reasons"],
                {"accumulate": 1},
            )

    def untimed_step(self, draining=False):
        return [
            dict(
                rank=i,
                control_epoch=1,
                epoch_scope="global_control",
                global_plan="prefill" if draining else "decode",
                ready_prefill=int(i == 0),
                ready_decode=128,
                oldest_ready_us=600000000,
                decode=0 if draining else 128,
                prefill=int(draining and i == 0),
                cadence_adaptive=1,
                decode_window=64,
                low_water_pct=50,
                prefill_max_wait_ms=0,
                min_decode_share_pct=0,
                decode_peak=128,
                low_decode_mask=0,
                cadence_reason="drain" if draining else "accumulate",
                local_draining=int(draining and i == 3),
                drain_mask=8 if draining else 0,
            )
            for i in range(4)
        ]

    def test_untimed_replay_keeps_partial_prefill_despite_long_wait(self):
        rows = self.untimed_step()
        self.assertEqual(validate_cadence([rows], 7)["reasons"], {"accumulate": 1})
        for row in rows:
            row.update(global_plan="prefill", cadence_reason="prefill_timeout")
        with self.assertRaisesRegex(ValueError, "replay"):
            validate_cadence([rows], 7)

    def test_explicit_drain_replay_requires_actual_signal_and_rank_consensus(self):
        rows = self.untimed_step(True)
        self.assertEqual(validate_cadence([rows], 7)["reasons"], {"drain": 1})
        rows[0]["drain_mask"] = 0
        with self.assertRaisesRegex(ValueError, "drain mask"):
            validate_cadence([rows], 7)
        rows = self.untimed_step(True)
        rows[0].pop("local_draining")
        with self.assertRaisesRegex(ValueError, "drain evidence"):
            validate_cadence([rows], 7)

    def test_explicit_drain_signal_is_monotonic(self):
        first, second = self.untimed_step(True), self.untimed_step()
        for row in second:
            row["control_epoch"] = 2
        with self.assertRaisesRegex(ValueError, "drain signal regressed"):
            validate_cadence([first, second], 7)

    def paced_steps(self):
        transitions = [
            ("prefill", 15, "all_ready", 0, 0, 0, 0, 1),
            ("decode", 0, "decode_only", 1, 0, 17500000, 700000, 0),
            # A reservation loaded before the gate closed remains executable.
            ("decode", 1, "accumulate", 2, 17500000, 14500000, 40000, 0),
        ]
        return [
            [
                dict(
                    rank=i,
                    control_epoch=epoch,
                    epoch_scope="global_control",
                    global_plan=plan,
                    ready_prefill=int(bool(mask & (1 << i))),
                    ready_decode=128,
                    oldest_ready_us=0,
                    decode=128 if plan == "decode" else 0,
                    prefill=1 if plan == "prefill" else 0,
                    cadence_adaptive=1,
                    decode_window=64,
                    low_water_pct=50,
                    prefill_max_wait_ms=1000,
                    min_decode_share_pct=25,
                    decode_peak=128,
                    low_decode_mask=0,
                    cadence_reason=reason,
                    debt_before_scaled=before,
                    debt_after_scaled=after,
                    charged_plan=charged,
                    last_model_us=cost - i * 100 if cost else 0,
                    admit_before=admit,
                    prefill_admit_allowed=admit,
                    decode_debt_us=max(0, after) // 75,
                )
                for i in range(4)
            ]
            for epoch, (
                plan,
                mask,
                reason,
                charged,
                before,
                after,
                cost,
                admit,
            ) in enumerate(transitions, 1)
        ]

    def test_pacing_replay_checks_cost_consensus_admission_and_reservation(self):
        steps = self.paced_steps()
        result = validate_cadence(steps, 7)
        self.assertEqual(result["min_decode_share_pct"], 25)
        self.assertEqual(result["budget_continuity_gaps"], 0)
        steps[2][0]["prefill_admit_allowed"] = 1
        with self.assertRaisesRegex(ValueError, "admission gate"):
            validate_cadence(steps, 7)
        steps = self.paced_steps()
        steps[1][3]["debt_after_scaled"] += 1
        with self.assertRaisesRegex(ValueError, "budget mismatch"):
            validate_cadence(steps, 7)

    def test_pacing_replay_checks_continuity_and_marks_zero_work_epoch_gaps(self):
        steps = self.paced_steps()
        for row in steps[2]:
            row["debt_before_scaled"] += 1
            row["debt_after_scaled"] += 1
        with self.assertRaisesRegex(ValueError, "consecutive epochs"):
            validate_cadence(steps, 7)
        steps = self.paced_steps()
        for row in steps[2]:
            row["control_epoch"] += 2
        self.assertEqual(validate_cadence(steps, 7)["budget_continuity_gaps"], 1)

    def lookahead_steps(self):
        steps = self.paced_steps()[:2]
        before = 17500000
        for n in range(6):
            rows = [dict(r) for r in self.paced_steps()[2]]
            for r in rows:
                r.update(
                    control_epoch=n + 3,
                    ready_prefill=0,
                    cadence_reason="decode_only",
                    debt_before_scaled=before,
                    debt_after_scaled=before - 3000000,
                    decode_debt_us=max(0, before - 3000000) // 75,
                    admit_before=int(before <= (3000000 if n else 0)),
                    prefill_admit_allowed=int(before <= (3000000 if n else 0)),
                )
            steps.append(rows)
            before -= 3000000
        previous_credit = 0
        for rows in steps:
            after_credit = 3000000 if rows[0]["charged_plan"] == 2 else 0
            for r in rows:
                r["admit_credit_before_scaled"] = previous_credit
                r["admit_credit_after_scaled"] = after_credit
            previous_credit = after_credit
        return steps

    def test_lookahead_replay_allows_one_quantum_without_erasing_debt(self):
        steps = self.lookahead_steps()
        self.assertEqual(steps[-1][0]["debt_before_scaled"], 2500000)
        self.assertEqual(steps[-1][0]["admit_before"], 1)
        self.assertEqual(validate_cadence(steps, 7)["budget_continuity_gaps"], 0)
        for r in steps[3]:
            r["admit_credit_after_scaled"] += 1
        with self.assertRaisesRegex(ValueError, "exceeds completed Decode"):
            validate_cadence(steps, 7)

    def test_lookahead_replay_rejects_invented_or_cross_rank_credit(self):
        steps = self.lookahead_steps()
        steps[3][1]["admit_credit_before_scaled"] += 1
        with self.assertRaisesRegex(ValueError, "cross-rank admission lookahead"):
            validate_cadence(steps, 7)
        steps = self.lookahead_steps()
        for r in steps[3]:
            r["admit_credit_before_scaled"] += 1
        with self.assertRaisesRegex(
            ValueError, "lookahead changed between consecutive"
        ):
            validate_cadence(steps, 7)

    def test_adaptive_policy_replays_all_triggers_and_ignores_ratio(self):
        def rows(plan, ready, mask, reason, low=0, age=0, peaks=None):
            return [
                dict(
                    rank=i,
                    epoch_scope="global_control",
                    global_plan=plan,
                    ready_prefill=int(bool(mask & (1 << i))),
                    ready_decode=n,
                    oldest_ready_us=age if mask & (1 << i) else 0,
                    decode=n if plan == "decode" else 0,
                    cadence_adaptive=1,
                    decode_window=64,
                    low_water_pct=50,
                    prefill_max_wait_ms=1000,
                    decode_peak=(peaks or ready)[i],
                    low_decode_mask=low,
                    cadence_reason=reason,
                )
                for i, n in enumerate(ready)
            ]

        steps = [
            rows("decode", [128] * 4, 0, "decode_only"),
            rows("decode", [128] * 4, 1, "accumulate"),
            rows(
                "prefill", [128, 128, 64, 128], 1, "decode_low", low=4, peaks=[128] * 4
            ),
            rows("prefill", [128] * 4, 1, "prefill_timeout", age=1000000),
            rows("prefill", [128] * 4, 15, "all_ready"),
        ]
        evidence = validate_cadence(steps, 1)
        self.assertEqual(evidence["phases"], dict(decode=2, prefill=3))
        self.assertEqual(
            set(evidence["reasons"]),
            {"decode_only", "accumulate", "decode_low", "prefill_timeout", "all_ready"},
        )
        steps[2][0]["decode_peak"] = 127
        with self.assertRaisesRegex(ValueError, "replay"):
            validate_cadence(steps, 1)

    def test_adaptive_replay_counts_commits_and_expires_old_peak(self):
        def rows(count, peak, committed, mask=0, reason="decode_only"):
            return [
                dict(
                    rank=i,
                    epoch_scope="global_control",
                    global_plan="decode",
                    ready_prefill=int(bool(mask & (1 << i))),
                    ready_decode=count,
                    oldest_ready_us=0,
                    decode=committed,
                    cadence_adaptive=1,
                    decode_window=2,
                    low_water_pct=50,
                    prefill_max_wait_ms=1000,
                    decode_peak=peak,
                    low_decode_mask=0,
                    cadence_reason=reason,
                )
                for i in range(4)
            ]

        steps = [
            rows(128, 128, 8),
            rows(8, 8, 4),
            rows(7, 8, 4),
            rows(3, 4, 3, mask=1, reason="accumulate"),
        ]
        self.assertEqual(validate_cadence(steps, 1)["phases"]["decode"], 4)
        steps[3][2]["decode_window"] = 3
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            validate_cadence(steps, 1)

    def test_two_rank_global_trace_rejects_inactive_mask_bits(self):
        lines = [
            line.replace("ready_mask=15", "ready_mask=3").replace(
                "commit_mask=7", "commit_mask=3"
            )
            for line in global_fixture()
            if "rank=2 " not in line and "rank=3 " not in line
        ]
        rows, metadata = align(lines, "test", ranks=2)
        self.assertEqual(metadata["rank_begin_counts"], [2, 2])
        self.assertEqual(len(rows), 2)
        with self.assertRaisesRegex(ValueError, "global masks"):
            align(
                [line.replace("ready_mask=3", "ready_mask=7") for line in lines],
                "test",
                ranks=2,
            )

    def test_global_control_with_idle_epoch_gaps_and_cancelled_rank(self):
        rows, _ = align(global_fixture(), "test")
        self.assertEqual(len(rows), 2)
        self.assertEqual(summarize(rows)["prefill_present"]["global_steps"], 2)

    def test_global_plan_masks_reservations_and_fake_are_checked(self):
        for old, new in (
            ("global_plan=prefill", "global_plan=decode"),
            ("commit_mask=7", "commit_mask=15"),
            ("ready_prefill=1", "ready_prefill=0"),
            ("fake=1", "fake=0"),
            ("control_us=20", "control_us=-1"),
            ("control_epoch=30", "control_epoch=10"),
        ):
            lines = global_fixture()
            lines = [line.replace(old, new) for line in lines]
            with self.subTest(old=old), self.assertRaises(ValueError):
                align(lines, "test")

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
