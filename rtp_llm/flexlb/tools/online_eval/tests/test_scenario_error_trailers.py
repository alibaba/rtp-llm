"""Real protobuf bytes and consumer completion, with no Java/network fixture."""

import base64
import tempfile
import time
import unittest
from types import SimpleNamespace as NS

from flexlb_test_framework.scenario.backend import RequestBatch, error_trailer_evidence
from flexlb_test_framework.scenario.runtime import Deadline, RuntimeContext
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from test_scenario_backend import Ops, Stream


def details_type():
    # Exact ErrorDetailsPB wire fields from model_rpc_service.proto; dynamic
    # protobuf decoding exercises unknown fields and malformed wire payloads.
    file = descriptor_pb2.FileDescriptorProto(
        name="trailer-test.proto", syntax="proto3"
    )
    message = file.message_type.add(name="ErrorDetailsPB")
    message.field.add(name="error_code", number=1, type=3, label=1)
    message.field.add(name="error_message", number=2, type=9, label=1)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("ErrorDetailsPB"))


DETAILS = details_type()
PB2 = NS(ErrorDetailsPB=DETAILS)


class RpcFailure(Exception):
    def __init__(self, metadata, status="INTERNAL"):
        self.metadata, self.status = metadata, status

    def code(self):
        return NS(name=self.status)

    def trailing_metadata(self):
        return self.metadata


def trailer(raw):
    return [("grpc-status-details-bin", raw)]


class ErrorTrailerTest(unittest.TestCase):
    def test_real_consumer_retains_transport_typed_code_and_bytes(self):
        for transport in ("INTERNAL", "CANCELLED"):
            with self.subTest(
                transport=transport
            ), tempfile.TemporaryDirectory() as tmp:
                raw = DETAILS(
                    error_code=8430, error_message="victim"
                ).SerializeToString()
                ctx = RuntimeContext({}, None, tmp, time.monotonic, time.sleep)
                ctx.ops = Ops(batch=False)
                ctx.ops.pb2.ErrorDetailsPB = DETAILS
                ctx.ops.pb2_grpc.RpcServiceStub = lambda channel: NS(
                    GenerateStreamCall=lambda *a, **kw: Stream(
                        error=RpcFailure(trailer(raw), transport)
                    )
                )
                ctx.instance_deadline_s = time.monotonic() + 2
                batch = RequestBatch(ctx, dict(count=1, consume="immediate"))
                batch.submit(Deadline(time.monotonic() + 1))
                with self.assertRaisesRegex(RuntimeError, "stream RPC failed"):
                    batch.wait(Deadline(time.monotonic() + 1))
                record = batch.snapshot_records()[0]
                self.assertTrue(record["consumer_completion_verified"])
                self.assertEqual(record["stream"]["status"], transport)
                self.assertEqual(record["stream"]["trailer_error_code"], 8430)
                self.assertIsNone(record["business_error_code"])
                self.assertEqual(
                    base64.b64decode(
                        record["stream"]["error_trailer"]["values"][0]["base64"]
                    ),
                    raw,
                )

    def test_bad_or_missing_evidence_never_becomes_zero(self):
        samples = [
            (None, "absent"),
            (trailer(b""), "invalid_value"),
            (trailer("not bytes"), "invalid_value"),
            (trailer(b"\x08"), "parse_error"),
            (trailer(b"\x18\x01"), "missing_error_code"),
            (
                trailer(DETAILS(error_message="only text").SerializeToString()),
                "missing_error_code",
            ),
            (trailer(b"a" * 65537), "invalid_value"),
            (trailer(b"\x08\x02") * 2, "ambiguous"),
        ]
        for metadata, status in samples:
            with self.subTest(status=status):
                evidence = error_trailer_evidence(RpcFailure(metadata), PB2)
                self.assertEqual(evidence["error_trailer"]["status"], status)
                self.assertIsNone(evidence["trailer_error_code"])
                for value in evidence["error_trailer"]["values"]:
                    if "base64" in value:
                        self.assertLessEqual(
                            len(base64.b64decode(value["base64"])), 65536
                        )

    def test_cancelled_without_typed_trailer_and_inband_enum_are_not_normalized(self):
        absent = error_trailer_evidence(RpcFailure(None, "CANCELLED"), PB2)
        self.assertIsNone(absent["trailer_error_code"])
        raw = DETAILS(error_code=2).SerializeToString()
        typed = error_trailer_evidence(RpcFailure(trailer(raw)), PB2)
        self.assertEqual(typed["trailer_error_code"], 2)

    def test_metadata_reader_failure_is_observable(self):
        class Broken:
            def trailing_metadata(self):
                raise ValueError("unavailable")

        evidence = error_trailer_evidence(Broken(), PB2)
        self.assertEqual(evidence["error_trailer"]["status"], "parse_error")
        self.assertIn("unavailable", evidence["error_trailer"]["error"])
        self.assertIsNone(evidence["trailer_error_code"])


if __name__ == "__main__":
    unittest.main()
