"""Fixture-driven smoke for ``DashScInferenceServicer.ModelStreamInfer``.

Granular request/response slices live in ``servicer_test.py`` / ``codec_test.py``.
This smoke walks a small set of *realistic dashscope-shaped requests* end to
end (parser -> ``GenerateConfig`` -> fake backend -> response builder) and
golden-compares against the expected values declared in
``test/data/dash_sc_request_smoke_cases.json`` — same fixture-driven shape as
``mrcr_smoke_test.py``.

Adding a new case = appending a JSON entry, no Python change required.
"""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch

from rtp_llm.dash_sc.inference.servicer import (
    DashScInferenceServicer,
    build_think_runtime,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "dash_sc_request_smoke_cases.json"
)


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def _load_cases() -> list[dict]:
    with _FIXTURE.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fakes (kept tiny — same shape as servicer_test._FakeAsyncStream / _FakeVisitor)
# ---------------------------------------------------------------------------


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._emitted = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._emitted >= len(self._chunks):
            raise StopAsyncIteration
        item = self._chunks[self._emitted]
        self._emitted += 1
        return item

    async def aclose(self):
        pass


class _FakeVisitor:
    def __init__(self, stream: _FakeAsyncStream):
        self._stream = stream
        self.enqueue_called = 0
        self.last_generate_input = None
        self.generate_inputs: list = []

    async def enqueue(self, generate_input):
        self.enqueue_called += 1
        self.last_generate_input = generate_input
        self.generate_inputs.append(generate_input)
        return self._stream


class _DSV4Tokenizer:
    """Same dsv4-style mapping as servicer_test._dsv4_tokenizer."""

    eos_token_id = 2
    vocab_size = 200000

    _MAPPING: dict[str, list[int]] = {
        "<think>\n": [128821, 198],
        "</think>\n\n": [128822, 271],
        "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
        "</think>": [128822],
    }

    def encode(self, text, add_special_tokens=True):
        if text in self._MAPPING:
            return list(self._MAPPING[text])
        return [ord(c) % 65535 for c in text]


class _GenerateEnvCfg:
    think_mode = 1
    think_end_token_id = -1
    think_start_tag = "<think>\n"
    think_end_tag = "</think>\n\n"


# ---------------------------------------------------------------------------
# Request building (fixture -> ModelInferRequest)
# ---------------------------------------------------------------------------


def _set_parameter(param, value) -> None:
    """Set a Triton-style ``InferParameter`` from a JSON-decoded value.

    Mapping:
      * ``bool`` -> ``bool_param``
      * ``int`` (non-bool) -> ``int64_param``
      * ``str`` -> ``string_param``
      * ``dict`` / ``list`` -> ``json.dumps`` then ``string_param`` (matches
        how dashscope-serving forwards nested controls like
        ``ds_header_attributes``).
    """
    if isinstance(value, bool):
        param.bool_param = value
    elif isinstance(value, int):
        param.int64_param = value
    elif isinstance(value, str):
        param.string_param = value
    elif isinstance(value, (dict, list)):
        param.string_param = json.dumps(value, ensure_ascii=False)
    else:
        raise TypeError(
            f"unsupported parameter value type {type(value).__name__}: {value!r}"
        )


def _build_request(case: dict) -> predict_v2_pb2.ModelInferRequest:
    spec = case["request"]
    req = predict_v2_pb2.ModelInferRequest()
    req.id = spec.get("id", case["name"])
    req.model_name = spec.get("model_name", "default")

    input_ids = spec.get("input_ids")
    if input_ids is not None:
        inp = req.inputs.add()
        inp.name = "input_ids"
        inp.datatype = "INT32"
        inp.shape[:] = [len(input_ids)]
        req.raw_input_contents.append(
            struct.pack("<%di" % len(input_ids), *input_ids)
        )

    for name, value in spec.get("parameters", {}).items():
        _set_parameter(req.parameters[name], value)
    return req


def _build_invocation_context(case: dict):
    spec = case["request"]
    metadata = spec.get("invocation_metadata") or ()
    if not metadata:
        return MagicMock()
    context = MagicMock()
    context.invocation_metadata.return_value = tuple(
        (k, v) for k, v in metadata
    )
    return context


def _build_fake_stream(case: dict) -> _FakeAsyncStream:
    chunks = []
    for entry in case.get("fake_stream", []):
        chunks.append(
            GenerateOutputs(
                generate_outputs=[
                    GenerateOutput(
                        output_ids=torch.tensor(
                            entry["output_ids"], dtype=torch.int32
                        ),
                        finished=bool(entry["finished"]),
                        aux_info=AuxInfo(
                            input_len=int(entry.get("input_len", 0)),
                            reuse_len=int(entry.get("reuse_len", 0)),
                        ),
                    )
                ]
            )
        )
    return _FakeAsyncStream(chunks)


def _build_servicer(case: dict, visitor: _FakeVisitor) -> DashScInferenceServicer:
    """Wire a tokenizer + think_runtime only when the case explicitly opts in.

    Cases that don't touch the thinking pipeline (e.g. ``max_completion_tokens``
    validation) leave these unset so the smoke also covers the bare-defaults
    construction path the production code falls back to.
    """
    tokenizer_kind = case.get("tokenizer")
    model_type = case.get("model_type")
    if tokenizer_kind == "dsv4" and model_type:
        tok = _DSV4Tokenizer()
        env_cfg = _GenerateEnvCfg()
        return DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, model_type),
        )
    return DashScInferenceServicer(backend_visitor=visitor)


# ---------------------------------------------------------------------------
# Response decoding
# ---------------------------------------------------------------------------


def _outputs_by_name(infer):
    by_name: dict[str, tuple] = {}
    for i, out in enumerate(infer.outputs):
        raw = infer.raw_output_contents[i] if i < len(infer.raw_output_contents) else b""
        by_name[out.name] = (out, raw)
    return by_name


def _unpack_int32_le(raw: bytes) -> list[int]:
    if not raw:
        return []
    return list(struct.unpack("<%di" % (len(raw) // 4), raw))


def _decode_finish_reason(out, raw: bytes) -> int | None:
    if out.datatype == "INT64" and len(raw) >= 8:
        return struct.unpack("<q", raw[:8])[0]
    if out.datatype == "INT32" and len(raw) >= 4:
        return struct.unpack("<i", raw[:4])[0]
    return None


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


async def _drain(aiter):
    return [x async for x in aiter]


async def _areq_iter(requests):
    for r in requests:
        yield r


# ---------------------------------------------------------------------------
# Golden assertions
# ---------------------------------------------------------------------------


def _assert_generate_config(test, case_name: str, golden: dict, cfg) -> None:
    """Compare a small allow-list of ``GenerateConfig`` fields against goldens.

    Allow-list (vs. ``getattr`` over arbitrary names) keeps fixture authors
    from accidentally relying on internal field names that get refactored;
    new fields land here explicitly.
    """
    for field in (
        "max_new_tokens",
        "timeout_ms",
        "ttft_timeout_ms",
        "traffic_reject_priority",
        "in_think_mode",
        "max_thinking_tokens",
        "end_think_token_ids",
    ):
        if field not in golden:
            continue
        actual = getattr(cfg, field)
        if isinstance(actual, list):
            actual = list(actual)
        test.assertEqual(
            actual,
            golden[field],
            msg=f"[{case_name}] generate_config.{field}: {actual!r} != {golden[field]!r}",
        )

    if "response_format_json" in golden:
        test.assertTrue(
            cfg.response_format,
            msg=f"[{case_name}] response_format unexpectedly empty",
        )
        parsed = json.loads(cfg.response_format)
        test.assertEqual(
            parsed,
            golden["response_format_json"],
            msg=f"[{case_name}] response_format mismatch",
        )

    if golden.get("json_schema_is_none"):
        test.assertIsNone(
            cfg.json_schema,
            msg=f"[{case_name}] json_schema should be None, got {cfg.json_schema!r}",
        )


def _assert_response(
    test, case_name: str, idx: int, golden: dict, resp
) -> None:
    test.assertFalse(
        resp.error_message,
        msg=f"[{case_name}][resp {idx}] unexpected error_message {resp.error_message!r}",
    )
    outputs = _outputs_by_name(resp.infer_response)

    if "generated_ids" in golden:
        test.assertIn(
            "generated_ids", outputs, msg=f"[{case_name}][resp {idx}] missing generated_ids"
        )
        gen_out, gen_raw = outputs["generated_ids"]
        actual_ids = _unpack_int32_le(gen_raw)
        # ``generated_ids`` is shape ``[1, len]`` so an empty tensor still has
        # an outer dimension; guard against the encoder advertising 0-len.
        shape = list(gen_out.shape)
        if shape and shape[-1] <= 0:
            actual_ids = []
        test.assertEqual(
            actual_ids,
            list(golden["generated_ids"]),
            msg=f"[{case_name}][resp {idx}] generated_ids mismatch",
        )

    if "finished" in golden:
        test.assertIn("finished", outputs)
        _, finished_raw = outputs["finished"]
        test.assertEqual(
            finished_raw,
            b"\x01" if golden["finished"] else b"\x00",
            msg=f"[{case_name}][resp {idx}] finished mismatch",
        )

    if "finish_reason_in" in golden:
        test.assertIn("finish_reason", outputs)
        fr_out, fr_raw = outputs["finish_reason"]
        actual_fr = _decode_finish_reason(fr_out, fr_raw)
        test.assertIn(
            actual_fr,
            golden["finish_reason_in"],
            msg=(
                f"[{case_name}][resp {idx}] finish_reason={actual_fr!r} "
                f"not in {golden['finish_reason_in']}"
            ),
        )

    if "status_code" in golden:
        test.assertEqual(
            resp.infer_response.parameters["status_code"].int64_param,
            golden["status_code"],
            msg=f"[{case_name}][resp {idx}] status_code mismatch",
        )
    if "status_name" in golden:
        test.assertEqual(
            resp.infer_response.parameters["status_name"].string_param,
            golden["status_name"],
            msg=f"[{case_name}][resp {idx}] status_name mismatch",
        )
    if "status_message_contains" in golden:
        test.assertIn(
            golden["status_message_contains"],
            resp.infer_response.parameters["status_message"].string_param,
            msg=f"[{case_name}][resp {idx}] status_message missing expected fragment",
        )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class DashScRequestSmokeTest(unittest.IsolatedAsyncioTestCase):
    """One ``test_<case_name>`` per fixture entry, generated below by
    ``_install_case_methods`` so each case shows up individually in the bazel
    test output (mirrors ``mrcr_smoke_test.py``'s ``test_…`` granularity)."""

    def test_fixture_well_formed(self) -> None:
        cases = _load_cases()
        self.assertGreater(len(cases), 0, "fixture must contain at least one case")
        seen_names: set[str] = set()
        for case in cases:
            name = case["name"]
            self.assertNotIn(name, seen_names, f"duplicate case name {name!r}")
            seen_names.add(name)
            self.assertIn("request", case)
            self.assertIn("expected_responses", case)


async def _run_case(test: DashScRequestSmokeTest, case: dict) -> None:
    case_name = case["name"]
    visitor = _FakeVisitor(_build_fake_stream(case))
    servicer = _build_servicer(case, visitor)
    request = _build_request(case)
    context = _build_invocation_context(case)

    responses = await _drain(
        servicer.ModelStreamInfer(_areq_iter([request]), context)
    )

    expected_enqueue = case.get("expected_enqueue_called")
    if expected_enqueue is not None:
        test.assertEqual(
            visitor.enqueue_called,
            expected_enqueue,
            msg=f"[{case_name}] enqueue_called mismatch",
        )

    if "expected_generate_config" in case:
        test.assertIsNotNone(
            visitor.last_generate_input,
            msg=f"[{case_name}] expected an enqueue but visitor saw none",
        )
        _assert_generate_config(
            test,
            case_name,
            case["expected_generate_config"],
            visitor.last_generate_input.generate_config,
        )

    if "expected_headers" in case:
        test.assertIsNotNone(visitor.last_generate_input)
        test.assertEqual(
            visitor.last_generate_input.headers,
            case["expected_headers"],
            msg=f"[{case_name}] headers mismatch",
        )

    expected_responses = case.get("expected_responses", [])
    test.assertEqual(
        len(responses),
        len(expected_responses),
        msg=(
            f"[{case_name}] response count {len(responses)} "
            f"!= expected {len(expected_responses)}"
        ),
    )
    for idx, golden in enumerate(expected_responses):
        _assert_response(test, case_name, idx, golden, responses[idx])


def _install_case_methods() -> None:
    """Attach one ``test_<case_name>`` per fixture entry so each case is a
    distinct unittest method; bazel reports them individually."""
    for case in _load_cases():

        async def _method(self, _case=case) -> None:
            await _run_case(self, _case)

        method_name = f"test_{case['name']}"
        _method.__name__ = method_name
        setattr(DashScRequestSmokeTest, method_name, _method)


_install_case_methods()


if __name__ == "__main__":
    unittest.main()
