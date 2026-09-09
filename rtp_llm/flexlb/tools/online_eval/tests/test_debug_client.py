import copy
import io
import json
import unittest
from unittest.mock import patch

from flexlb_test_framework.debug_client import (
    Capture,
    DebugClient,
    DebugUnavailable,
    check_scheduler_tombstone,
    validate_snapshot,
)


def snapshot():
    return dict(
        schemaVersion=1,
        instanceId="master-1",
        snapshotId="sample-1",
        captureStartedAtMs=1,
        captureFinishedAtMs=2,
        status="ok",
        endpointDirectoryTruncated=False,
        endpointScope="registered",
        components={
            "scheduler": dict(
                status="ok",
                truncated=False,
                scannedCount=0,
                consistency="per_entry",
                metadata={},
                rows=[],
                captureStartedAtMs=1,
                captureFinishedAtMs=2,
            )
        },
    )


class DebugClientTest(unittest.TestCase):
    def test_missing_component_is_not_empty_success(self):
        with self.assertRaises(DebugUnavailable):
            Capture(snapshot(), 1, 2).component("decode")

    def test_every_incomplete_status_is_rejected(self):
        for status in (
            "busy",
            "unavailable",
            "partial",
            "budget_exhausted",
            "not_applicable",
        ):
            data = snapshot()
            data["components"]["scheduler"]["status"] = status
            with self.subTest(status=status), self.assertRaises(DebugUnavailable):
                Capture(validate_snapshot(data), 1, 2).component("scheduler")

    def test_truncation_is_rejected_even_with_ok_status(self):
        data = snapshot()
        data["components"]["scheduler"]["truncated"] = True
        with self.assertRaises(DebugUnavailable):
            Capture(validate_snapshot(data), 1, 2).component("scheduler")

    def test_missing_fields_and_unknown_schema_fail_closed(self):
        for key in (
            "instanceId",
            "components",
            "endpointDirectoryTruncated",
            "captureStartedAtMs",
        ):
            data = snapshot()
            del data[key]
            with self.subTest(key=key), self.assertRaises(DebugUnavailable):
                validate_snapshot(data)

    def test_http_failure_does_not_create_empty_snapshot(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(DebugUnavailable):
                DebugClient("http://localhost:7001").snapshot()

    def test_long_id_is_preserved_in_url_and_json(self):
        data = snapshot()
        data["components"]["scheduler"]["rows"] = [{"request_id": "9007199254740993"}]
        with patch(
            "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(data).encode())
        ) as request:
            capture = DebugClient("http://localhost:7001").snapshot(
                request_id=9007199254740993
            )
        self.assertIn("/requests/9007199254740993?", request.call_args.args[0])
        self.assertEqual(
            "9007199254740993", capture.component("scheduler")["rows"][0]["request_id"]
        )

    def test_numeric_json_identity_is_rejected(self):
        data = snapshot()
        data["components"]["scheduler"]["rows"] = [{"request_id": 9007199254740993}]
        with self.assertRaises(DebugUnavailable):
            validate_snapshot(data)

    def test_tombstone_is_not_equivalent_to_no_resource_ownership(self):
        row = dict(
            storage_phase="TOMBSTONE",
            lifecycle_phase="COMPLETED",
            admission_open=False,
            has_item=False,
            has_engine_fence=False,
            has_preemption=False,
            has_admission_mutation=False,
            has_request_deadline=False,
            has_decision_deadline=False,
            has_inactivity_deadline=False,
            has_cancel_reason=False,
            has_pending_admission_cancel=False,
        )
        self.assertTrue(check_scheduler_tombstone(row)[0])
        retained = copy.deepcopy(row)
        retained["has_engine_fence"] = True
        self.assertFalse(check_scheduler_tombstone(retained)[0])
        del row["has_item"]
        with self.assertRaises(DebugUnavailable):
            check_scheduler_tombstone(row)


if __name__ == "__main__":
    unittest.main()
