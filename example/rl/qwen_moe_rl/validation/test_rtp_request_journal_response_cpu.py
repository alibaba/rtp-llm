"""CPU-only response retention checks using the real generated RPC protobuf.

The transport is a fixture; this does not issue an RPC or evaluate a model.
"""
import argparse
import asyncio
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

parser = argparse.ArgumentParser()
HERE = Path(__file__).resolve().parent
default_candidate = HERE / "rtp_request_journal.py"
if not default_candidate.is_file():
    default_candidate = HERE.parent / "adapter/rtp_request_journal.py"
default_source = next((p for p in HERE.parents if (p / "rtp_llm").is_dir()), Path.cwd())
parser.add_argument("--candidate", type=Path, default=default_candidate)
parser.add_argument("--rtp-source", type=Path, default=default_source)
parser.add_argument("--output", type=Path)
args, _remaining = parser.parse_known_args()
TORCH_WAS_IMPORTED = "torch" in sys.modules
sys.path.insert(0, str(args.rtp_source))
spec = importlib.util.spec_from_file_location("capture_candidate", args.candidate)
journal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(journal)
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.RL.token_rpc_client import TokenRpcClient, TokenProtocolError

PROMPT = [760, 869, 314, 220, 18, 22, 348, 220, 19, 18, 369]
TARGET = [220, 16, 20, 24, 16, 13]
FINITE_FIXTURE_BITS = [0xBE800000, 0xBF800000, 0xC1200000,
                       0xBE000000, 0x80000000, 0xC2B10000]


def response(bits=FINITE_FIXTURE_BITS, version="fixture:1"):
    result = pb.GenerateOutputsPB(request_id=101, model_version=version,
                                  global_step=0)
    result.flatten_output.finished.append(True)
    result.flatten_output.output_ids.CopyFrom(pb.TensorPB(
        data_type=pb.TensorPB.INT32, shape=[1], int32_data=struct.pack("<i", 7)))
    output = result.flatten_output.prompt_logits
    output.start_pos, output.end_pos = 10, 16
    output.target_logprobs.CopyFrom(pb.TensorPB(data_type=pb.TensorPB.FP32,
        shape=[6], fp32_data=struct.pack("<6I", *bits)))
    output.topk_logprobs.CopyFrom(pb.TensorPB(data_type=pb.TensorPB.FP32,
        shape=[6, 1], fp32_data=struct.pack("<6I", *bits)))
    return result


class ControlledTransport(TokenRpcClient):
    def __init__(self, supplied_response, **kwargs):
        self.supplied_response = supplied_response
        super().__init__("no-rpc.fixture.invalid:1", vocab_size=248320,
                         eos_token_ids=[248044], timeout=600, **kwargs)

    async def _request(self, request_id, ids, config):
        self.transport_inputs = (request_id, list(ids), config)
        return self.supplied_response


class ObservedClient(journal.JournaledTokenRpcMixin, ControlledTransport):
    pass


class ResponseTests(unittest.TestCase):
    def test_real_protobuf_round_trip_preserves_all_fields_and_unknown_fields(self):
        message = response()
        # Unknown field 1000=7 must survive complete-message retention too.
        payload = message.SerializeToString(deterministic=True) + b"\xc0\x3e\x07"
        message.ParseFromString(payload)
        before = message.SerializeToString(deterministic=True)
        record = journal._response_snapshot(message)
        captured = base64.b64decode(record["response_pb_b64"])
        recovered = pb.GenerateOutputsPB.FromString(captured)
        self.assertEqual(captured, before)
        self.assertEqual(recovered.SerializeToString(deterministic=True), before)
        self.assertEqual(message.SerializeToString(deterministic=True), before)
        self.assertEqual(record["response_pb_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(record["response_pb_bytes"], len(before))
        self.assertEqual(record["response_pb_type"], message.DESCRIPTOR.full_name)
        self.assertEqual(record["response_pb_encoding"],
                         "protobuf_deterministic_reserialization")
        scores = record["prompt_logits"]["target_logprobs"]
        self.assertEqual((record["prompt_logits"]["start_pos"],
                          record["prompt_logits"]["end_pos"]), (10, 16))
        self.assertEqual((scores["data_type"], scores["shape"], scores["fp32_bytes"]),
                         (pb.TensorPB.FP32, [6], 24))
        raw = message.flatten_output.prompt_logits.target_logprobs.fp32_data
        self.assertEqual(bytes.fromhex(scores["fp32_hex"]), raw)
        self.assertEqual(scores["fp32_sha256"], hashlib.sha256(raw).hexdigest())

    def test_nonfinite_and_signed_zero_bytes_remain_json_safe_and_exact(self):
        bits = [0x80000000, 0x00000000, 0x7FC12345,
                0x7F800000, 0xFF800000, 0xBF800001]
        record = journal._response_snapshot(response(bits))
        persisted = json.loads(json.dumps(record, allow_nan=False))
        raw = bytes.fromhex(persisted["prompt_logits"]["target_logprobs"]["fp32_hex"])
        self.assertEqual(list(struct.unpack("<6I", raw)), bits)

    def test_version_only_fixture_does_not_invent_payload(self):
        record = journal._response_snapshot(SimpleNamespace(model_version="fixture", global_step=0))
        self.assertIs(record["response_payload_available"], False)
        self.assertNotIn("response_pb_b64", record)
        self.assertNotIn("prompt_logits", record)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def check_call(self, message, *, expected_version="fixture:1", error=None):
        with tempfile.TemporaryDirectory() as directory:
            recording = journal.NativeRpcJournal(directory, 3,
                lambda: dict(native_boot=0, native_incarnation="fixture"))
            client = ObservedClient(message, request_journal=recording,
                                    max_active_requests=1)
            if error is None:
                actual = await client.score_tokens(101, PROMPT, TARGET,
                                                    expected_version=expected_version)
                expected = struct.unpack("<6f", message.flatten_output.prompt_logits.target_logprobs.fp32_data)
                self.assertEqual(actual, list(expected))
            else:
                with self.assertRaises(error):
                    await client.score_tokens(101, PROMPT, TARGET,
                                              expected_version=expected_version)
            records = [json.loads(line) for line in recording.path.read_text().splitlines()]
            self.assertEqual([row["event"] for row in records], ["queued", "start", "finish"])
            self.assertEqual(records[0]["prompt_token_ids"], PROMPT + TARGET)
            self.assertEqual(records[-1]["transport_outcome"], "response_received")
            self.assertEqual(records[-1]["outcome"], "success" if error is None else "error")
            snapshot = records[-1]["response"]
            expected = message.SerializeToString(deterministic=True)
            self.assertEqual(base64.b64decode(snapshot["response_pb_b64"]), expected)
            self.assertEqual(snapshot["response_pb_sha256"], hashlib.sha256(expected).hexdigest())
            self.assertEqual(bytes.fromhex(snapshot["prompt_logits"]["target_logprobs"]["fp32_hex"]),
                             message.flatten_output.prompt_logits.target_logprobs.fp32_data)
            self.assertEqual(client._request_gate._value, 1)
            self.assertEqual(client.transport_inputs[:2], (101, PROMPT + TARGET))
            await client.close()
            self.assertEqual("torch" in sys.modules, TORCH_WAS_IMPORTED)

    async def test_actual_score_client_success_is_durable(self):
        await self.check_call(response())

    async def test_actual_score_client_version_error_keeps_response(self):
        await self.check_call(response(), expected_version="different", error=TokenProtocolError)

    async def test_actual_score_client_invalid_scores_keep_original_nan_bits(self):
        bits = [0x7FC12345, *FINITE_FIXTURE_BITS[1:]]
        await self.check_call(response(bits), error=TokenProtocolError)


if __name__ == "__main__":
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(ResponseTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleTests),
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    no_torch = "torch" not in sys.modules
    record = dict(schema="rtp-response-capture-cpu-v1", tests=result.testsRun,
                  success=result.wasSuccessful(), no_torch_imported=no_torch,
                  GPU_or_RPC_or_model_evaluation_performed=False,
                  real_generated_protobuf_used=True,
                  actual_token_score_validation_used=True,
                  transport_is_explicit_fixture=True,
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  proto_path=pb.__file__,
                  proto_sha256=hashlib.sha256(Path(pb.__file__).read_bytes()).hexdigest())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() and no_torch else 1)
