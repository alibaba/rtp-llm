import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import msgspec

_path = Path(__file__).resolve().parents[2] / "adapters/vllm/VLLMForwardProtocol.py"
_spec = importlib.util.spec_from_file_location("forward_protocol", _path)
protocol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(protocol)


class OldRequest(msgspec.Struct, array_like=True):
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: dict[str, int]
    navi_sched_local_fallback: bool = False
    navi_sched_render_call: bytes | None = None


class NewRequest(msgspec.Struct, array_like=True):
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: dict[str, int]
    reasoning_parser_kwargs: dict[str, str] | None = None
    navi_sched_local_fallback: bool = False
    navi_sched_render_call: bytes | None = None


@dataclass
class PrefillStats:
    num_prompt_tokens: int = 0
    num_computed_tokens: int = 0
    num_cached_tokens: int = 0
    num_local_cached_tokens: int = 0
    num_external_cached_tokens: int = 0


class OldOutput(msgspec.Struct, array_like=True):
    request_id: str
    new_token_ids: list[int]
    finish_reason: int | None = None
    num_cached_tokens: int = 0
    num_external_computed_tokens: int = 0
    num_nans_in_logits: int = 0


class NewOutput(msgspec.Struct, array_like=True):
    request_id: str
    new_token_ids: list[int]
    finish_reason: int | None = None
    prefill_stats: PrefillStats | None = None
    num_nans_in_logits: int = 0


class OldOutputs(msgspec.Struct, array_like=True):
    outputs: list[OldOutput] = []
    finished_requests: set[str] | None = None


class NewOutputs(msgspec.Struct, array_like=True):
    outputs: list[NewOutput] = []
    finished_requests: set[str] | None = None


class VLLMForwardProtocolTest(unittest.TestCase):
    def test_native_request_arrays_reject_both_version_directions(self):
        for source, target in ((OldRequest, NewRequest), (NewRequest, OldRequest)):
            with self.subTest(source=source.__name__):
                request = source(
                    "r", [1, 2], {"max_tokens": 3}, navi_sched_render_call=b"image-url"
                )
                with self.assertRaises(msgspec.ValidationError):
                    msgspec.msgpack.decode(msgspec.msgpack.encode(request), type=target)

    def test_requests_preserve_fields_in_both_version_directions(self):
        for source, target in (
            (OldRequest, NewRequest),
            (NewRequest, OldRequest),
            (OldRequest, OldRequest),
            (NewRequest, NewRequest),
        ):
            with self.subTest(source=source.__name__, target=target.__name__):
                request = source(
                    "r", [7, 8], {"max_tokens": 9}, navi_sched_render_call=b"image-url"
                )
                encoded = msgspec.msgpack.encode(
                    protocol.forward_request_fields(request)
                )
                decoded = msgspec.msgpack.decode(
                    encoded, type=protocol.forward_request_type(target)
                )
                self.assertIsInstance(decoded, target)
                self.assertEqual("r", decoded.request_id)
                self.assertEqual([7, 8], decoded.prompt_token_ids)
                self.assertEqual({"max_tokens": 9}, decoded.sampling_params)
                self.assertFalse(decoded.navi_sched_local_fallback)
                self.assertEqual(b"image-url", decoded.navi_sched_render_call)

    def test_request_projection_reuses_token_and_media_buffers(self):
        request = NewRequest("r", [7, 8], {}, navi_sched_render_call=b"media")
        fields = protocol.forward_request_fields(request)
        self.assertIs(request.prompt_token_ids, fields["prompt_token_ids"])
        self.assertIs(request.navi_sched_render_call, fields["navi_sched_render_call"])

    def test_new_request_preserves_reasoning_configuration(self):
        request = NewRequest("r", [], {}, reasoning_parser_kwargs={"mode": "strict"})
        encoded = msgspec.msgpack.encode(protocol.forward_request_fields(request))
        decoded = msgspec.msgpack.decode(
            encoded, type=protocol.forward_request_type(NewRequest)
        )
        self.assertEqual({"mode": "strict"}, decoded.reasoning_parser_kwargs)

    def decode_output(self, source, target, target_pack):
        encoded = msgspec.msgpack.encode(protocol.forward_outputs_fields(source))
        return msgspec.msgpack.decode(
            encoded,
            type=protocol.forward_outputs_type(target, target_pack),
        )

    def test_native_output_arrays_reject_different_cache_statistics(self):
        old = OldOutputs([OldOutput("r", [3], num_cached_tokens=8)])
        with self.assertRaises(msgspec.ValidationError):
            msgspec.msgpack.decode(msgspec.msgpack.encode(old), type=NewOutputs)

    def test_new_output_to_old_preserves_cache_counts_and_terminal_state(self):
        source = NewOutputs(
            [
                NewOutput(
                    "r",
                    [9],
                    finish_reason=1,
                    prefill_stats=PrefillStats(20, 12, 8, 5, 3),
                    num_nans_in_logits=2,
                )
            ],
            {"r"},
        )
        decoded = self.decode_output(source, OldOutput, OldOutputs)
        output = decoded.outputs[0]
        counts = protocol.restore_cache_statistics(output, OldOutput, 20, 0, 0)
        self.assertEqual((8, 3), counts)
        self.assertEqual(8, output.num_cached_tokens)
        self.assertEqual(3, output.num_external_computed_tokens)
        self.assertEqual([9], output.new_token_ids)
        self.assertEqual(1, output.finish_reason)
        self.assertEqual(2, output.num_nans_in_logits)
        self.assertEqual({"r"}, decoded.finished_requests)

    def test_old_output_to_new_restores_prefill_statistics(self):
        source = OldOutputs(
            [OldOutput("r", [9], num_cached_tokens=8, num_external_computed_tokens=3)]
        )
        decoded = self.decode_output(source, NewOutput, NewOutputs)
        stats_module = types.ModuleType("vllm.v1.metrics.stats")
        stats_module.PrefillStats = PrefillStats
        with mock.patch.dict(sys.modules, {"vllm.v1.metrics.stats": stats_module}):
            counts = protocol.restore_cache_statistics(
                decoded.outputs[0], NewOutput, 20, 0, 0
            )
        self.assertEqual((8, 3), counts)
        self.assertEqual(
            PrefillStats(20, 12, 8, 5, 3), decoded.outputs[0].prefill_stats
        )

    def test_streaming_new_output_keeps_old_cache_counts_after_prefill(self):
        source = NewOutputs([NewOutput("r", [10])])
        output = self.decode_output(source, OldOutput, OldOutputs).outputs[0]
        self.assertEqual(
            (8, 3), protocol.restore_cache_statistics(output, OldOutput, None, 8, 3)
        )
        self.assertEqual(8, output.num_cached_tokens)
        self.assertEqual(3, output.num_external_computed_tokens)

    def test_streaming_old_output_does_not_repeat_new_prefill_statistics(self):
        source = OldOutputs(
            [OldOutput("r", [10], num_cached_tokens=8, num_external_computed_tokens=3)]
        )
        output = self.decode_output(source, NewOutput, NewOutputs).outputs[0]
        self.assertEqual(
            (8, 3), protocol.restore_cache_statistics(output, NewOutput, None, 8, 3)
        )
        self.assertIsNone(output.prefill_stats)

    def test_wire_type_definitions_are_reused(self):
        self.assertIs(
            protocol.forward_request_type(OldRequest),
            protocol.forward_request_type(OldRequest),
        )
        self.assertIs(
            protocol.forward_outputs_type(OldOutput, OldOutputs),
            protocol.forward_outputs_type(OldOutput, OldOutputs),
        )


if __name__ == "__main__":
    unittest.main()
