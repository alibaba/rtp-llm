"""Offline regression checks for functional K3 smoke result validation."""

import copy
import json
import pathlib
import tempfile
import unittest

from rtp_llm.test.smoke.kimi_k3.result_check import (
    _check_protocol,
    _group_artifact_errors,
    _raw_consistency,
    answer_from_question,
    audit_raw_answers,
    check_result,
    independent_answer_check,
)


class ResultCheckTest(unittest.TestCase):
    def test_answer_audit_rejects_duplicate_json_and_truncation(self):
        row = {
            "name": "x",
            "phase": "formal",
            "pd_sep": True,
            "finish_reason": "length",
            "content": '{"value":"A","value":"A"}',
            "expected_json": {"value": "A"},
            "output_len": 8,
        }
        errors = independent_answer_check(row)
        self.assertTrue(any("duplicate" in error for error in errors))
        self.assertTrue(any("incomplete" in error for error in errors))

    def test_four_layer_diagnostic_garble_does_not_weaken_formal_audit(self):
        row = {
            "name": "filler",
            "phase": "formal",
            "pd_sep": True,
            "finish_reason": "length",
            "content": "x\ufffd",
            "expected_regex": ".",
            "output_len": 2,
        }
        self.assertEqual(independent_answer_check(row, diagnostic=True), [])
        self.assertTrue(independent_answer_check(row, diagnostic=False))

    def test_raw_answer_is_derived_from_question(self):
        self.assertEqual(answer_from_question("只回答数字：81 的平方是多少？"), "6561")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "0001.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "case",
                        "phase": "formal",
                        "status": 200,
                        "request": {
                            "messages": [
                                {"content": '只输出 JSON {"value":"expected"}'}
                            ]
                        },
                        "response_body": json.dumps(
                            {
                                "choices": [
                                    {
                                        "finish_reason": "stop",
                                        "message": {"content": '{"value":"wrong"}'},
                                    }
                                ],
                                "aux_info": {"pd_sep": True},
                            }
                        ),
                    }
                )
            )
            self.assertTrue(
                any(
                    "prompt-derived answer mismatch" in error
                    for error in audit_raw_answers(
                        path.parent, {"case": {"phase": "formal"}}
                    )
                )
            )

    def test_group_rpc_raw_answer_requires_rpc_ack_and_exact_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "0001.json"
            artifact = {
                "name": "case",
                "phase": "formal",
                "transport": "pd_group_rpc",
                "request_id": 17,
                "rpc_status": "OK",
                "request": {
                    "messages": [{"content": '只输出 JSON {"value":"expected"}'}]
                },
                "response_body": json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": '{"value":"expected"}'},
                            }
                        ],
                        "aux_info": {"pd_sep": True},
                    }
                ),
            }
            path.write_text(json.dumps(artifact))
            rows = {"case": {"phase": "formal"}}
            self.assertEqual(audit_raw_answers(path.parent, rows), [])
            artifact["rpc_status"] = "UNKNOWN"
            path.write_text(json.dumps(artifact))
            self.assertTrue(audit_raw_answers(path.parent, rows))

    def fixture(self):
        role = {"role": "DECODE", "ip": "127.0.0.1", "http_port": 1, "grpc_port": 2}
        cases, stages = [], []
        for boundary in (4096, 32768):
            for delta in (-1, 0, 1):
                name = f"orthogonal_page_{boundary}_{delta:+d}"
                cases.append(
                    {
                        "name": name,
                        "phase": "formal",
                        "pd_sep": True,
                        "content": "5",
                        "finish_reason": "stop",
                        "expected_regex": ".",
                        "input_len": boundary + delta,
                        "output_len": 8,
                        "effective_reuse_len": 0,
                        "expected_reuse": "miss",
                        "decode_owner_rank": 0,
                        "selected_decode_role_addr": role,
                        "observed_decode_role_addrs": [role],
                    }
                )
                stages.append({"name": name, "case_names": [name], "passed": True})
        return {
            "passed": True,
            "suite": "orthogonal-flow",
            "case_profile": "compact-64k-v1",
            "orthogonal_phases": ["page"],
            "cases": cases,
            "stages": stages,
            "summary": {"case_count": 6, "skipped_case_count": 0},
            "block_size": 4096,
            "decode_role_addrs": [role],
            "skipped_cases": [],
        }

    def checked(self, result):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "accuracy.json"
            path.write_text(json.dumps(result))
            return check_result(path)

    def test_complete_diagnostic_matrix_has_only_functional_scope(self):
        verdict = self.checked(self.fixture())
        self.assertTrue(verdict["passed"], verdict["errors"])
        self.assertEqual(verdict["formal_case_count"], 6)
        self.assertFalse(verdict["kernel_paths_verified"])
        self.assertFalse(verdict["host_load_timing_verified"])
        self.assertFalse(verdict["full_original_suite_passed"])

    def test_missing_page_stage_cannot_pass_by_updating_counts(self):
        result = self.fixture()
        result["cases"].pop()
        result["stages"].pop()
        result["summary"]["case_count"] -= 1
        verdict = self.checked(result)
        self.assertFalse(verdict["passed"])
        self.assertTrue(
            any("missing functional stage" in error for error in verdict["errors"])
        )

    def test_duplicate_cases_and_stage_membership_fail(self):
        for mode in ("duplicate_case", "duplicate_assignment", "orphan_case"):
            result = self.fixture()
            if mode == "duplicate_case":
                result["cases"].append(copy.deepcopy(result["cases"][0]))
                result["summary"]["case_count"] += 1
            elif mode == "duplicate_assignment":
                result["stages"][1]["case_names"] = result["stages"][0]["case_names"]
            else:
                result["stages"][0]["case_names"] = ["unknown"]
            with self.subTest(mode=mode):
                self.assertFalse(self.checked(result)["passed"])

    def test_cache_and_owner_checks_do_not_trust_runner_pass(self):
        for changes in (
            {"expected_reuse": "hit"},
            {"effective_reuse_len": 1},
            {"expected_input_len": 123},
            {"expected_cache_tier": "memory"},
            {"observed_decode_role_addrs": []},
            {"decode_owner_rank": 1},
            {"decode_crossings": [99999]},
        ):
            result = self.fixture()
            result["cases"][0].update(changes)
            with self.subTest(changes=changes):
                self.assertFalse(self.checked(result)["passed"])

    def add_mixed_probe(self, result, attempt, *, host_demoted):
        kinds = ("device", "memory", "partial", "miss")
        names = []
        for kind in kinds:
            row = copy.deepcopy(result["cases"][0])
            name = f"tier-mixed-{attempt}-{kind}"
            names.append(name)
            reuse = 0 if kind == "miss" else 4096
            memory = 4096 if kind == "memory" and host_demoted else 0
            row.update(
                name=name,
                input_len=8193,
                expected_reuse="any",
                effective_reuse_len=reuse,
                prefill_cache_tiers={
                    "total": reuse,
                    "local": reuse,
                    "device": reuse - memory,
                    "memory": memory,
                    "disk": 0,
                },
            )
            result["cases"].append(row)
        result["summary"]["case_count"] += 4
        result["stages"].append(
            {
                "name": f"orthogonal_cache_mixed_{attempt}",
                "case_names": names,
                "passed": True,
                "cache_tier_checks": dict.fromkeys(kinds, True)
                | {"memory": host_demoted},
                "current_q_total": 4 * 8193 - 3 * 4096,
                "functional_passed": host_demoted,
            }
        )

    def test_mixed_cache_retry_accepts_a_later_complete_tier_combination(self):
        result = self.fixture()
        result["orthogonal_phases"].append("cache")
        self.add_mixed_probe(result, 0, host_demoted=False)
        self.add_mixed_probe(result, 1, host_demoted=True)
        verdict = self.checked(result)
        self.assertTrue(verdict["passed"], verdict["errors"])
        self.assertEqual(verdict["formal_case_count"], 14)

    def test_mixed_cache_retries_without_complete_tiers_fail(self):
        result = self.fixture()
        result["orthogonal_phases"].append("cache")
        for attempt in range(3):
            self.add_mixed_probe(result, attempt, host_demoted=False)
        verdict = self.checked(result)
        self.assertFalse(verdict["passed"])
        self.assertIn("missing successful mixed-cache stage", verdict["errors"])

    def test_retry_success_cannot_hide_bad_answers_or_forged_cache_summary(self):
        for mode in ("bad_answer", "forged_cache_summary"):
            result = self.fixture()
            result["orthogonal_phases"].append("cache")
            self.add_mixed_probe(result, 0, host_demoted=False)
            self.add_mixed_probe(result, 1, host_demoted=True)
            if mode == "bad_answer":
                result["cases"][6]["content"] = ""
            else:
                result["stages"][-2]["cache_tier_checks"]["memory"] = True
            with self.subTest(mode=mode):
                self.assertFalse(self.checked(result)["passed"])

    def test_skips_are_reported_and_not_counted_as_executed(self):
        result = self.fixture()
        skipped = {
            "name": "deferred",
            "reason": "outside selected scope",
            "sent": False,
        }
        result["skipped_cases"] = [skipped]
        result["summary"]["skipped_case_count"] = 1
        verdict = self.checked(result)
        self.assertTrue(verdict["passed"], verdict["errors"])
        self.assertEqual(verdict["case_count"], 6)
        self.assertEqual(verdict["skipped_cases"], [skipped])
        result["stages"][0]["skipped"] = True
        self.assertFalse(self.checked(result)["passed"])

    def test_full_suite_without_protocol_matrix_fails(self):
        result = self.fixture()
        result["suite"] = "main-text-64k-capped"
        verdict = self.checked(result)
        self.assertFalse(verdict["passed"])
        self.assertTrue(
            any("protocol matrix mismatch" in error for error in verdict["errors"])
        )

    def test_cancel_requires_ack_and_recovery_without_claiming_host_timing(self):
        result = self.fixture()
        result["orthogonal_phases"].append("cancel")
        names = ["cancel-same-prefix-recovery", "cancel-independent-recovery"]
        for name in names:
            row = copy.deepcopy(result["cases"][0])
            row["name"] = name
            result["cases"].append(row)
        result["summary"]["case_count"] += 2
        attempt = {
            "cancel_status": 1,
            "fetch": {
                "code": "StatusCode.RESOURCE_EXHAUSTED",
                "details": "preempted by a higher-priority request",
            },
        }
        result["stages"].append(
            {
                "name": "orthogonal_cancel_recovery",
                "case_names": names,
                "passed": True,
                "cancel_attempts": [attempt],
            }
        )
        verdict = self.checked(result)
        self.assertTrue(verdict["passed"], verdict["errors"])
        self.assertFalse(verdict["host_load_timing_verified"])
        attempt["cancel_status"] = 2
        self.assertFalse(self.checked(result)["passed"])

    def test_protocol_presence_requires_all_names_and_saved_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "summary.json").write_text("[]")
            errors = _check_protocol([], root)
            self.assertTrue(
                any("protocol matrix mismatch" in error for error in errors)
            )

    def test_raw_cache_values_and_rpc_ids_are_independently_compared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            row = {
                "name": "x",
                "content": "5",
                "reasoning_content": "",
                "finish_reason": "stop",
                "input_len": 12,
                "output_len": 8,
                "iter_count": 3,
                "effective_reuse_len": 0,
                "mtp_draft_rounds": 2,
                "mtp_accepted_tokens": 5,
                "observed_decode_role_addrs": [],
            }
            artifact = {
                "name": "x",
                "phase": "formal",
                "transport": "pd_group_rpc",
                "request_id": 7,
                "rpc_status": "OK",
                "response_body": json.dumps(
                    {
                        "choices": [
                            {"message": {"content": "5"}, "finish_reason": "stop"}
                        ],
                        "aux_info": {
                            "input_len": 12,
                            "output_len": 8,
                            "iter_count": 3,
                            "prefill_total_reuse_len": 0,
                            "speculative_draft_rounds": 2,
                        },
                    }
                ),
            }
            (root / "x.json").write_text(json.dumps(artifact))
            self.assertEqual(_raw_consistency(root, {"x": row}), [])
            changed = dict(row, effective_reuse_len=4096)
            self.assertTrue(_raw_consistency(root, {"x": changed}))
            stage = {"transport": "pd_group_rpc", "request_ids": {"x": 7}}
            self.assertEqual(_group_artifact_errors([stage], root), [])
            stage["request_ids"]["x"] = 8
            self.assertTrue(_group_artifact_errors([stage], root))

    def test_malformed_result_is_a_failed_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "accuracy.json"
            path.write_text('{"passed": true, "passed": true}')
            verdict = check_result(path)
            self.assertFalse(verdict["passed"])
            self.assertTrue(
                any("duplicate JSON key" in error for error in verdict["errors"])
            )


if __name__ == "__main__":
    unittest.main()
