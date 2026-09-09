"""Full programs over actual consumer workers and protobuf trailer bytes.

Requires the reviewed e8e536 shared backend and d955 cleanup dependencies.
"""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_scenario_priority_preemption as programs
from flexlb_test_framework.scenario.actions import priority_preemption as preempt
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def details_type():
    file = descriptor_pb2.FileDescriptorProto(
        name="preemption-terminal.proto", syntax="proto3"
    )
    message = file.message_type.add(name="ErrorDetailsPB")
    message.field.add(name="error_code", number=1, type=3, label=1)
    message.field.add(name="error_message", number=2, type=9, label=1)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("ErrorDetailsPB"))


DETAILS = details_type()


class Failure(Exception):
    def __init__(self, status, code):
        self.status, self.typed_code = status, code

    def code(self):
        return NS(name=self.status)

    def trailing_metadata(self):
        if self.typed_code is None:
            return []
        return [
            (
                "grpc-status-details-bin",
                DETAILS(error_code=self.typed_code).SerializeToString(),
            )
        ]


class TerminalPrograms(unittest.TestCase):
    def run_program(self, status="INTERNAL", code=8430):
        backend = programs.Backend()
        backend.ops.pb2.ErrorDetailsPB = DETAILS
        generate = backend.ops.generate

        def call(request, *args, **kwargs):
            stream = generate(request, *args, **kwargs)
            if request[0] == 2:
                stream.error = Failure(status, code)
            return stream

        backend.ops.generate = call
        with patch.object(programs, "Backend", return_value=backend):
            return programs.PreemptionPrograms().run_program(
                variant="same_priority_zero_eviction"
            )

    def test_typed_forbidden_error_reaches_verdict_after_real_consumer_exit(self):
        result, cohorts, backend = self.run_program()
        self.assertEqual("FAIL", result["status"], result)
        stages = {s["id"]: s for s in result["stages"]}
        self.assertEqual("PASS", stages["wave_drain"]["status"])
        self.assertEqual(
            {"PR4": "FAIL", "AT3": "PASS", "P6_terminal": "FAIL"},
            {c["id"]: c["status"] for c in stages["same_priority"]["checks"]},
        )
        row = next(r for wave in cohorts for r in wave if r["wire_request_id"] == 2)
        self.assertTrue(row["consumer_completion_verified"])
        self.assertEqual("INTERNAL", row["stream"]["status"])
        self.assertEqual(8430, row["stream"]["trailer_error_code"])
        self.assertEqual("parsed", row["stream"]["error_trailer"]["status"])

    def test_literal_trailer_two_is_not_remapped_to_engine_cancelled(self):
        result, cohorts, backend = self.run_program(code=2)
        row = next(r for wave in cohorts for r in wave if r["wire_request_id"] == 2)
        self.assertEqual(2, row["stream"]["trailer_error_code"])
        self.assertEqual("parsed", row["stream"]["error_trailer"]["status"])

    def test_transport_cancelled_does_not_become_typed_engine_victim(self):
        result, _, _ = self.run_program(status="CANCELLED", code=8429)
        self.assertEqual("ERROR", result["status"])
        self.assertTrue(all(c["status"] == "PASS" for c in result["cleanup"]))

    def test_missing_typed_evidence_is_error(self):
        result, _, _ = self.run_program(code=None)
        self.assertEqual("ERROR", result["status"])

    def test_rpc_deadline_is_not_swallowed_as_expected_business_failure(self):
        result, _, _ = self.run_program(status="DEADLINE_EXCEEDED")
        self.assertEqual("TIMEOUT", result["status"])

    def test_inband_cancelled_retains_expected_mapping(self):
        entry = dict(batch=NS(entries=[dict(response=NS(code=200, success=True))]))
        with patch.object(preempt, "request_success", return_value=False):
            self.assertEqual(
                (False, 8429),
                preempt._outcome(
                    entry, dict(stream=dict(status="OK"), business_error_code=2)
                ),
            )
