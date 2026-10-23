"""Cache lifecycle evidence must not be mistaken for request completions."""

import copy
import unittest

from prefill_traffic_audit import audit_run
from test_prefill_traffic_audit import evidence


def cache_evidence(dispatch="NON_BATCH"):
    requests, stages, events = evidence(dispatch)
    for event in events:
        if event["event"] == "decode_done":
            event.update(
                engine_name="decode-0",
                engine_incarnation="decode-incarnation-1",
                engine_address="127.0.0.1:12346",
            )
    events.append(
        {
            "event": "evict_chain",
            "engine_name": "decode-0",
            "engine_incarnation": "decode-incarnation-1",
            "engine_address": "127.0.0.1:12346",
            "timestamp_ms": 100300,
            "leaf_key": 3000100000,
            "chain_keys": [3000100000, 1062, 1061],
            "blocks_freed": 3,
            "reason": "admission",
        }
    )
    return requests, stages, events


class CacheEventAuditTest(unittest.TestCase):
    def test_legitimate_non_request_eviction_is_reported_separately(self):
        for dispatch in ("NON_BATCH", "BATCH"):
            with self.subTest(dispatch=dispatch):
                rows, stages, events = cache_evidence(dispatch)
                without = audit_run(rows, stages, events[:-1], dispatch=dispatch)
                result = audit_run(rows, stages, events, dispatch=dispatch)
                self.assertEqual(
                    result.pop("cache_lifecycle"),
                    {
                        "evict_chain_events": 1,
                        "blocks_freed": 3,
                        "by_reason": {"admission": 1},
                        "by_worker": {"decode-0": 1},
                    },
                )
                without.pop("cache_lifecycle")
                self.assertEqual(result, without)

    def test_pinned_ancestor_is_inspected_but_not_freed(self):
        rows, stages, events = cache_evidence()
        events[-1]["blocks_freed"] = len(events[-1]["chain_keys"]) - 1
        result = audit_run(rows, stages, events, dispatch="NON_BATCH")
        self.assertEqual(result["cache_lifecycle"]["blocks_freed"], 2)

    def test_all_pinned_mock_reasons_and_signed_long_keys_are_accepted(self):
        for reason in ("admission", "growth", "completion", "insertion", "retention"):
            with self.subTest(reason=reason):
                rows, stages, events = cache_evidence()
                events[-1].update(
                    reason=reason,
                    leaf_key=-(1 << 63),
                    chain_keys=[-(1 << 63), 0, (1 << 63) - 1],
                )
                audit_run(rows, stages, events, dispatch="NON_BATCH")

    def test_missing_or_extra_fields_fail_including_rid_none(self):
        base = cache_evidence()
        for key in base[2][-1]:
            with self.subTest(missing=key):
                rows, stages, events = copy.deepcopy(base)
                del events[-1][key]
                with self.assertRaises(ValueError):
                    audit_run(rows, stages, events, dispatch="NON_BATCH")
        for key, value in (("rid", None), ("rid", 1), ("cancelled", False)):
            with self.subTest(extra=(key, value)):
                rows, stages, events = copy.deepcopy(base)
                events[-1][key] = value
                with self.assertRaisesRegex(ValueError, "evict_chain"):
                    audit_run(rows, stages, events, dispatch="NON_BATCH")

    def test_invalid_identity_or_reason_cannot_be_skipped(self):
        for field in ("engine_name", "engine_incarnation", "engine_address", "reason"):
            for value in (None, "", " ", 42, [], "unknown"):
                with self.subTest(field=field, value=value):
                    rows, stages, events = cache_evidence()
                    events[-1][field] = value
                    with self.assertRaisesRegex(ValueError, "evict_chain"):
                        audit_run(rows, stages, events, dispatch="NON_BATCH")

    def test_invalid_keys_chain_and_release_count_fail(self):
        bad_values = {
            "leaf_key": [True, None, "1062", 1.5, -(1 << 63) - 1, 1 << 63, 999],
            "chain_keys": [
                None,
                "1062",
                [],
                [True],
                [3000100000, 1062, 1062],
                [3000100000, []],
                [3000100000, float("nan")],
                [3000100000, 1 << 63],
                [3000100000, -(1 << 63) - 1],
            ],
            "blocks_freed": [None, True, 0, -1, 1, 4, 2.5, "3"],
        }
        for field, values in bad_values.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    rows, stages, events = cache_evidence()
                    events[-1][field] = value
                    with self.assertRaisesRegex(ValueError, "evict_chain"):
                        audit_run(rows, stages, events, dispatch="NON_BATCH")

    def test_timestamps_must_be_integral_and_inside_run_interval(self):
        for stamp in (None, True, "100300", 100300.0, float("nan"), 99997, 101003):
            with self.subTest(stamp=stamp):
                rows, stages, events = cache_evidence()
                events[-1]["timestamp_ms"] = stamp
                with self.assertRaisesRegex(ValueError, "evict_chain"):
                    audit_run(rows, stages, events, dispatch="NON_BATCH")
        for stamp in (99998, 101002):
            rows, stages, events = cache_evidence()
            events[-1]["timestamp_ms"] = stamp
            audit_run(rows, stages, events, dispatch="NON_BATCH")

    def test_unknown_events_fail_even_if_their_request_id_is_known(self):
        for kind in ("unknown", None, [], 123):
            for rid in (None, 1, 9999):
                with self.subTest(kind=kind, rid=rid):
                    rows, stages, events = evidence()
                    events.append({"event": kind, "rid": rid})
                    expected = "unknown Engine event" if rid == 1 else "unknown request"
                    with self.assertRaisesRegex(ValueError, expected):
                        audit_run(rows, stages, events, dispatch="NON_BATCH")

    def test_request_events_still_require_known_request_id(self):
        for kind in ("prefill_done", "decode_done"):
            for rid in (None, 9999):
                with self.subTest(kind=kind, rid=rid):
                    rows, stages, events = evidence()
                    events.append({"event": kind, "rid": rid})
                    with self.assertRaisesRegex(ValueError, "unknown request"):
                        audit_run(rows, stages, events, dispatch="NON_BATCH")

    def test_multiple_cache_events_preserve_independent_accounting(self):
        rows, stages, events = cache_evidence()
        other = dict(events[-1], reason="growth", timestamp_ms=100301, blocks_freed=2)
        events.append(other)
        result = audit_run(rows, stages, events, dispatch="NON_BATCH")
        self.assertEqual(
            result["cache_lifecycle"],
            {
                "evict_chain_events": 2,
                "blocks_freed": 5,
                "by_reason": {"admission": 1, "growth": 1},
                "by_worker": {"decode-0": 2},
            },
        )


if __name__ == "__main__":
    unittest.main()
