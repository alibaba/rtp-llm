import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.test.smoke import deepseek_v41_sm90_semantic_checks as checks


def response(path):
    aux = dict(pd_sep=True, output_len=2)
    if path == "/":
        return dict(finished=True, response="Paris.", output_ids=[[12, 1]], aux_info=aux)
    return dict(
        choices=[dict(message=dict(content="42", reasoning_content=""), finish_reason="stop")],
        extra_outputs=dict(output_ids=[[42, 1]]),
        aux_info=aux,
        usage=dict(completion_tokens=2),
        stream_audit=dict(complete=True, eos_token_id=1),
    )


class Sm90SemanticChecksTest(unittest.TestCase):
    def test_known_answers_repeats_and_concurrent_requests(self):
        calls = []
        def post(path, payload):
            calls.append((path, payload))
            return response(path)
        with tempfile.TemporaryDirectory() as directory:
            report = checks.run_semantic_checks(post, directory, 100)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(len(report["requests"]), 10)
            self.assertEqual(len(list(Path(directory).glob("*serial*.json"))), 6)
        self.assertEqual(len(calls), 10)
        for path, payload in calls:
            config = payload["generate_config" if path == "/" else "extra_configs"]
            self.assertEqual(config["thinking_mode"], 1)
            self.assertFalse(config["ignore_eos"])
            self.assertFalse(config["do_sample"])
            self.assertEqual(config["min_new_tokens"], 0)
            self.assertFalse(config["reuse_cache"])
            if path != "/":
                self.assertFalse(payload["enable_thinking"])
                self.assertTrue(payload["stream"])
                self.assertTrue(payload["stream_options"]["include_usage"])
                self.assertTrue(payload["debug_info"])

    def test_wrong_facts_arithmetic_and_repetition_fail(self):
        for kind, path, text in (
            ("native_france", "/", "Berlin."),
            ("native_france", "/", "Not Paris."),
            ("native_france", "/", "Paris Paris Paris"),
            ("chat_arithmetic", "/v1/chat/completions", "43"),
            ("chat_arithmetic", "/v1/chat/completions", "42 42 42"),
        ):
            raw = response(path)
            if path == "/":
                raw["response"] = text
            else:
                raw["choices"][0]["message"]["content"] = text
            with self.subTest(text=text), self.assertRaises(AssertionError):
                checks._check(kind, raw, 100, True)

    def test_natural_stop_pd_and_ids_are_required(self):
        base = response("/v1/chat/completions")
        for field in ("length", "thinking", "pd", "ids", "budget"):
            raw = copy.deepcopy(base)
            if field == "length":
                raw["choices"][0]["finish_reason"] = "length"
            elif field == "thinking":
                raw["choices"][0]["message"]["reasoning_content"] = "Let me think"
            elif field == "pd":
                raw["aux_info"]["pd_sep"] = False
            elif field == "ids":
                raw["extra_outputs"]["output_ids"] = [[9999]]
            else:
                raw["extra_outputs"]["output_ids"] = [[42] * 64]
                raw["aux_info"]["output_len"] = 64
            with self.subTest(field=field), self.assertRaises(AssertionError):
                checks._check("chat_arithmetic", raw, 100, True)

    def test_native_continuation_may_use_its_full24_token_budget(self):
        raw = response("/")
        raw.update(response="Paris is a city in France.", output_ids=[list(range(24))])
        raw["aux_info"]["output_len"] = 24
        self.assertEqual(
            checks._check("native_france", raw, 100, True)["finish"],
            "finished_within_budget",
        )
        self.assertEqual(checks._payload("native_france")[1]["generate_config"]["max_new_tokens"], 24)

    def test_bad_answer_is_saved_and_never_promoted_to_reference(self):
        def post(path, payload):
            raw = response(path)
            if path == "/":
                raw["response"] = "garbage"
            return raw
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):
                checks.run_semantic_checks(post, directory, 100)
            report = json.loads((Path(directory) / "semantic-summary.json").read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["requests"], [])
            self.assertTrue((Path(directory) / "native_france_serial_0.json").is_file())

    def test_correct_but_changed_greedy_ids_fail(self):
        count = 0
        def post(path, payload):
            nonlocal count
            raw = response(path)
            count += 1
            if count == 2:
                raw["output_ids"] = [[13, 1]]
            return raw
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(AssertionError, "serial output changed"):
            checks.run_semantic_checks(post, directory, 100)


def stream_chunks():
    # The observed nonstream answer is42, cumulative length2 and tail EOS1.
    # ID42 here is a synthetic first-token fixture, not a model golden.
    def chunk(delta, ids=None, count=None, finish=None, **other):
        result = dict(choices=[dict(index=0, delta=delta, finish_reason=finish)], **other)
        if ids is not None:
            result["extra_outputs"] = dict(output_ids=[ids])
        if count is not None:
            result["usage"] = dict(completion_tokens=count)
        return result
    return [
        chunk(dict(role="assistant", content=""), debug_info=dict(eos_token_id=1)),
        chunk(dict(content="42"), [42], 1),
        chunk(dict(content=""), [1], 2),
        chunk(dict(content=""), count=2, finish="stop", aux_info=dict(pd_sep=True, output_len=2)),
    ]


def encode_sse(chunks):
    return "".join("data: " + json.dumps(chunk) + "\r\n\r\n" for chunk in chunks)


class Sm90SemanticStreamTest(unittest.TestCase):
    def test_42_then_eos_collects_complete_ids_and_raw_chunks(self):
        chunks = stream_chunks()
        body = encode_sse(chunks)
        raw = checks._collect_chat_sse(body)
        result = checks._check("chat_arithmetic", raw, 100, True)
        self.assertEqual(result["content"], "42")
        self.assertEqual(result["output_ids"], [42, 1])
        self.assertEqual(raw["stream_audit"]["raw_chunks"], chunks)
        self.assertEqual(raw["stream_audit"]["raw_sse"], body)

    def test_actual_openai_eof_and_optional_done(self):
        body = encode_sse(stream_chunks())
        self.assertEqual(checks._collect_chat_sse(body)["extra_outputs"]["output_ids"], [[42, 1]])
        self.assertEqual(checks._collect_chat_sse(body + "data: [DONE]\n\n")["usage"]["completion_tokens"], 2)

    def test_missing_duplicate_or_reordered_token_events_fail(self):
        chunks = stream_chunks()
        for bad in (chunks[:1] + chunks[2:], chunks[:2] + chunks[3:],
                    chunks[:2] + [chunks[1]] + chunks[2:],
                    [chunks[0], chunks[2], chunks[1], chunks[3]], chunks[:-1]):
            with self.subTest(chunks=bad), self.assertRaises(AssertionError):
                checks._collect_chat_sse(encode_sse(bad))

    def test_invalid_ids_eos_finish_and_final_counts_fail(self):
        for field in ("bool", "negative", "float", "empty", "eos", "no_eos_config", "length", "aux", "usage", "post_finish"):
            chunks = stream_chunks()
            if field in ("bool", "negative", "float", "empty"):
                chunks[1]["extra_outputs"]["output_ids"] = [{"bool": [True], "negative": [-1], "float": [42.0], "empty": []}[field]]
            elif field == "eos":
                chunks[2]["extra_outputs"]["output_ids"] = [[2]]
            elif field == "no_eos_config":
                chunks[0].pop("debug_info")
            elif field == "length":
                chunks[-1]["choices"][0]["finish_reason"] = "length"
            elif field == "aux":
                chunks[-1]["aux_info"]["output_len"] = 1
            elif field == "usage":
                chunks[-1]["usage"]["completion_tokens"] = 1
            else:
                chunks.append(copy.deepcopy(chunks[1]))
            with self.subTest(field=field), self.assertRaises(AssertionError):
                checks._collect_chat_sse(encode_sse(chunks))

    def test_out_of_vocab_and_thinking_are_not_hidden_by_aggregation(self):
        for field in ("vocab", "thinking"):
            chunks = stream_chunks()
            if field == "vocab":
                chunks[1]["extra_outputs"]["output_ids"] = [[100]]
            else:
                chunks[1]["choices"][0]["delta"]["reasoning_content"] = "Let me think"
            raw = checks._collect_chat_sse(encode_sse(chunks))
            with self.subTest(field=field), self.assertRaises(AssertionError):
                checks._check("chat_arithmetic", raw, 100, True)

    def test_stream_error_or_invalid_framing_is_rejected(self):
        for body in ("", "not SSE", "data: {bad}\n\n", "data: {\"error_code\": 1}\n\n",
                     encode_sse(stream_chunks()) + "data: [DONE]\n\ndata: {}\n\n"):
            with self.subTest(body=body), self.assertRaises((AssertionError, ValueError)):
                checks._collect_chat_sse(body)

    def test_http_adapter_retains_bad_stream_for_request_artifact(self):
        result = mock.MagicMock()
        result.__enter__.return_value.read.return_value = b"data: {\"error_code\": 1}\r\n\r\n"
        opener = mock.Mock()
        opener.open.return_value = result
        with mock.patch.object(checks.urllib.request, "build_opener", return_value=opener):
            raw = checks.http_post("http://fixture")("/v1/chat/completions", dict(stream=True))
        self.assertEqual(raw["error_code"], "INVALID_CHAT_STREAM")
        self.assertIn('"error_code": 1', raw["raw_sse"])


if __name__ == "__main__":
    unittest.main()
