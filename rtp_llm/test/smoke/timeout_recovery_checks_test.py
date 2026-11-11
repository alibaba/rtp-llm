import json
import unittest

from timeout_recovery_checks import (
    TimeoutRecoveryCheckError,
    analyze_timeout_recovery,
    check_timeout_recovery,
    d2h_timeout_records,
    frozen_host_metric_samples,
)


def record(trace=42, execution="transfer", released=False, **changes):
    fields = {
        "trace": json.dumps(trace),
        "execution_type": execution,
        "operation": '"load"',
        "state": "RELEASED" if released else "FROZEN",
        "frozen_at_us": "12345",
        "released_at_us": "23456" if released else "0",
        "target_block_count": "2",
        "expected_rank_count": "2",
        "failure_reason": '"rpc_timeout"',
        "error": '"error with commas, and escaped \\"quotes\\""',
        "descriptors": "[{index=0,targets=[{pool=abc,block=7}]}]",
        "recovery_query_count": "2" if released else "0",
        "last_recovery_query": (
            "{trace=" + json.dumps(trace) + ',all_safe=true,error="",ranks=[]}'
            if released
            else "unknown"
        ),
    }
    fields.update(changes)
    prefix = (
        "FrozenTask recovery released frozen_duration_ms=11 "
        if released
        else "transfer frozen: " if execution == "transfer" else "KVCM READ frozen: "
    )
    return (
        "[INFO] "
        + prefix
        + "FrozenTask{"
        + ",".join(f"{key}={value}" for key, value in fields.items())
        + "}\n"
    )


class TimeoutRecoveryChecksTest(unittest.TestCase):
    def test_real_trace_pair_with_nested_fields_and_duplicate_logs(self):
        text = record() * 2 + record(released=True) * 2
        report = check_timeout_recovery(text, backend="local", require_frozen=True)
        self.assertEqual(len(report["frozen"]), 1)
        self.assertEqual(len(report["released"]), 1)
        self.assertEqual(report["pending"], [])
        json.dumps(report)

    def test_d2h_requires_eviction_direction_rank_deadline_and_target_pool(self):
        fields = {
            "operation": '"evict"',
            "descriptors": (
                '[{source_tier=DEVICE,target_tier=HOST,targets=[{pool="host0",block=7}]}]'
            ),
            "rank_snapshots": "[{rank=0,grpc_code=0},{rank=1,grpc_code=4}]",
        }
        report = check_timeout_recovery(
            record(**fields) + record(released=True, **fields), backend="local"
        )
        self.assertEqual(d2h_timeout_records(report)[0]["target_pools"], ["host0"])
        for changes in (
            {"operation": '"load"'},
            {"expected_rank_count": "1"},
            {"rank_snapshots": "[{rank=0,grpc_code=0}]"},
            {"descriptors": "[]"},
            {"descriptors": fields["descriptors"].replace("DEVICE", "DISK")},
            {
                "descriptors": (
                    '[{source_tier=HOST,target_tier=DEVICE,targets=[{pool="dev0"}]}]'
                )
            },
        ):
            with self.subTest(changes=changes):
                altered = {**fields, **changes}
                report = analyze_timeout_recovery(record(**altered), backend="local")
                self.assertEqual(d2h_timeout_records(report), [])
        mismatched = {
            **fields,
            "descriptors": fields["descriptors"].replace("host0", "host1"),
        }
        self.assertFalse(
            analyze_timeout_recovery(
                record(**fields) + record(released=True, **mismatched), backend="local"
            )["passed"]
        )

    def test_host_metric_requires_fresh_sample_and_exact_pool(self):
        def sample(timestamp, value, pool="host0", tier="HOST"):
            return (
                f"rtp_llm_kv_cache_frozen_ref_blocks {timestamp} {value} "
                f"pool_name={pool} tier={tier}\n"
            )

        self.assertEqual(frozen_host_metric_samples("", after_ms=100), {})
        text = sample(99, "0_0_0_0_1") + sample(101, "0_0_0_0_1", tier="DEVICE")
        self.assertEqual(frozen_host_metric_samples(text, after_ms=100), {})
        text += sample(101, "1_1_1_1_1") + sample(101, "0_0_0_0_1")
        text += sample(102, "0_0_0_0_1", pool="other_host")
        samples = frozen_host_metric_samples(text, after_ms=100)
        self.assertEqual(samples["host0"]["maximum"], 1)
        self.assertEqual(samples["other_host"]["maximum"], 0)
        text += sample(103, "0_0_0_0_2")
        self.assertEqual(
            frozen_host_metric_samples(text, after_ms=100)["host0"]["maximum"], 0
        )
        for invalid in ("0", "0_0_nan_0_1", "0_0_0_0_0"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                frozen_host_metric_samples(sample(101, invalid), after_ms=100)

    def test_namespaces_are_distinct(self):
        text = (
            record()
            + record("42", "kvcm_read")
            + record(released=True)
            + record("42", "kvcm_read", released=True)
        )
        report = check_timeout_recovery(text, backend="kvcm", require_frozen=True)
        self.assertEqual(len(report["frozen"]), 2)

    def test_local_rpc_deadline_uses_original_rank_status_with_empty_reason(self):
        fields = {
            "failure_reason": '""',
            "rank_snapshots": "[{rank=0,grpc_code=0},{rank=1,grpc_code=4}]",
        }
        report = check_timeout_recovery(
            record(**fields) + record(released=True, **fields),
            backend="local",
            require_frozen=True,
            required_evidence=["local_rpc_timeout"],
        )
        self.assertEqual(report["frozen"][0]["rpc_deadline_ranks"], [1])

    def test_quoted_grpc_timeout_is_not_rank_status(self):
        fields = {
            "failure_reason": '""',
            "rank_snapshots": '[{rank=0,grpc_code=0,error="grpc_code=4"}]',
        }
        report = analyze_timeout_recovery(
            record(**fields) + record(released=True, **fields), backend="local"
        )
        self.assertFalse(report["passed"])

    def test_wrong_trace_cannot_release_frozen_execution(self):
        with self.assertRaises(TimeoutRecoveryCheckError) as caught:
            check_timeout_recovery(
                record() + record(43, released=True), backend="local"
            )
        self.assertEqual(len(caught.exception.report["pending"]), 1)

    def test_every_frozen_execution_requires_release(self):
        report = analyze_timeout_recovery(
            record() + record(43) + record(released=True), backend="local"
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["pending"][0]["trace"], 43)

    def test_invalid_release_proof_is_rejected(self):
        for changes in (
            {"recovery_query_count": "0"},
            {"last_recovery_query": "unknown"},
            {"last_recovery_query": "{trace=42,all_safe=false}"},
            {"last_recovery_query": "{trace=43,all_safe=true}"},
            {"last_recovery_query": '{trace=42,error="all_safe=true"}'},
            {"last_recovery_query": '{trace=42,all_safe="true"}'},
            {"target_block_count": "0"},
            {"target_block_count": "3"},
            {"frozen_at_us": "98765"},
            {"state": "FROZEN"},
        ):
            with self.subTest(changes=changes):
                report = analyze_timeout_recovery(
                    record() + record(released=True, **changes), backend="local"
                )
                self.assertFalse(report["passed"])

    def test_missing_or_malformed_freeze_is_rejected(self):
        for text in (
            record(released=True),
            record(target_block_count="0") + record(released=True),
            record()[:-2] + "\n" + record(released=True),
            record() + "FrozenTask recovery released (diagnostics unavailable)\n",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    analyze_timeout_recovery(text, backend="local")["passed"]
                )

    def test_untouched_queue_timeout_does_not_require_freeze(self):
        cases = {
            "local_business_queue": "load expired in business queue, descriptor_count=4",
            "local_workloop_queue": "transfer expired in WorkLoop queue",
            "local_worker_queue": "transfer expired in TE worker queue",
            "local_resource_wait": "staging admission timed out",
            "local_pre_submit": "transfer deadline exceeded before dispatch",
        }
        for category, text in cases.items():
            with self.subTest(category=category):
                check_timeout_recovery(
                    text, backend="local", required_evidence=[category]
                )
                self.assertFalse(
                    analyze_timeout_recovery(
                        text, backend="local", require_frozen=True
                    )["passed"]
                )

    def test_kvcm_timeout_evidence(self):
        cases = {
            "kvcm_pre_io": "storage read deadline exceeded before I/O",
            "kvcm_payload_pre_io": "KVCM READ payload expired before I/O: trace=read_1 timeout_ms=2",
            "kvcm_task_deadline": "KVCM task deadline exceeded: before payload dispatch",
            "kvcm_sdk_timeout": "kvcm client loadKvCaches fail, ec [119]",
            "kvcm_rpc_timeout": "KVCM broadcast timed out: rank=1",
        }
        for category, text in cases.items():
            with self.subTest(category=category):
                check_timeout_recovery(
                    text, backend="kvcm", required_evidence=[category]
                )

    def test_match_to_ready_is_not_backend_queue_or_rpc_evidence(self):
        text = (
            "block tree load failed, stream=42 error=match-to-ready deadline exceeded"
        )
        report = check_timeout_recovery(text, backend="kvcm")
        self.assertEqual(set(report["evidence"]), {"kvcm_match_to_ready"})
        for category in ("kvcm_pre_io", "kvcm_payload_pre_io", "kvcm_rpc_timeout"):
            self.assertFalse(
                analyze_timeout_recovery(
                    text, backend="kvcm", required_evidence=[category]
                )["passed"]
            )
        self.assertFalse(
            analyze_timeout_recovery(
                "configured error=match-to-ready deadline exceeded", backend="kvcm"
            )["passed"]
        )

    def test_sdk_read_timeout_codes_and_actual_read_abort(self):
        for text in (
            "kvcm client loadKvCaches fail, ec [100]",
            "kvcm client loadKvCaches fail, ec [119]",
            "local file sdk timeout abort: op=get done_blocks=3/5",
        ):
            with self.subTest(text=text):
                check_timeout_recovery(
                    text, backend="kvcm", required_evidence=["kvcm_sdk_timeout"]
                )

    def test_write_timeout_or_generic_sdk_error_does_not_prove_read_timeout(self):
        for text in (
            "kvcm client loadKvCaches fail, ec [120]",
            "kvcm client saveKvCaches fail, ec [100]",
            "kvcm client saveKvCaches fail, ec [119]",
            "local file sdk timeout abort: op=put done_blocks=3/5",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    analyze_timeout_recovery(
                        text, backend="kvcm", required_evidence=["kvcm_sdk_timeout"]
                    )["passed"]
                )

    def test_config_or_other_backend_is_not_a_timeout(self):
        for text in (
            "kvcm_get_timeout_ms: 1\nrequest finished",
            "sdk_timeout=1 rpc_timeout=1",
            "kvcm client loadKvCaches fail, ec [104]",
            "transfer expired in WorkLoop queue",
            "",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    analyze_timeout_recovery(text, backend="kvcm")["passed"]
                )

    def test_required_timeout_kind_is_enforced(self):
        report = analyze_timeout_recovery(
            "KVCM task deadline exceeded: before read",
            backend="kvcm",
            required_evidence=["kvcm_sdk_timeout"],
        )
        self.assertFalse(report["passed"])

    def test_frozen_non_timeout_failure_does_not_count_as_timeout(self):
        text = record(failure_reason='"rpc_error"') + record(
            released=True, failure_reason='"rpc_error"'
        )
        self.assertFalse(analyze_timeout_recovery(text, backend="local")["passed"])

    def test_unknown_category_and_backend_are_errors(self):
        with self.assertRaises(ValueError):
            analyze_timeout_recovery("", backend="invalid")
        with self.assertRaises(ValueError):
            analyze_timeout_recovery("", backend="local", required_evidence=["io"])
        with self.assertRaises(ValueError):
            analyze_timeout_recovery(
                "", backend="local", required_evidence=["kvcm_pre_io"]
            )


if __name__ == "__main__":
    unittest.main()
