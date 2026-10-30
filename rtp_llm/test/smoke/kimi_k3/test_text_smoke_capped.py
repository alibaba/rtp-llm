"""Short Decode page coverage in the time-capped full-model smoke."""

import gzip
import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.smoke.kimi_k3.text_smoke import (
    REQUEST_PROGRESS_PREFIX,
    Case,
    Runner,
    SmokeDeadline,
    SmokeFailure,
    TransportFailure,
    prefill_cache_tiers,
)


def formal_response_fixture(model_type="kimi_k3"):
    args = types.SimpleNamespace(
        base_url="http://127.0.0.1:1",
        decode_health_url="http://127.0.0.1:2/health",
        decode_role_addrs=[],
        namespace="response-integrity-test",
        block_size=4096,
        chunk_tokens=65536,
        max_tokens=32,
        suite="main-text-64k-capped",
        model_type=model_type,
        output=None,
        timeout=900,
        case_deadline_s=300,
    )
    case = Case(
        "short-exact-answer",
        "只输出 JSON",
        "",
        "miss",
        expected_json={"value": "BND-4096"},
    )
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": '{"value":"BND-4096"}',
                    "reasoning_content": "",
                },
            }
        ],
        "aux_info": {
            "pd_sep": True,
            "output_len": 16,
            "iter_count": 16,
            "input_len": 1,
            "reuse_len": 0,
        },
        "debug_info": {"output_ids": [[1] * 16]},
    }
    return Runner(args), case, response


class CappedDecodePageTest(unittest.TestCase):
    def test_k31_prompt_correction_is_verified_by_the_serving_tokenizer(self):
        smoke, _, _ = formal_response_fixture("kimi_k31")
        with patch.object(
            smoke,
            "tokenize",
            side_effect=lambda prompt, **_: [1] * (prompt.count(" x") + 5),
        ) as tokenize:
            prompt, ids = smoke.fit_prompt("head", "tail", 40)
            self.assertEqual(prompt, "head" + " x" * 35 + "tail")
            self.assertEqual(len(ids), 40)
            self.assertEqual(tokenize.call_count, 2)
        self.assertEqual(smoke.prompt_timing()["fit_calls"], 1)

    def test_k31_failed_correction_falls_back_to_the_original_search(self):
        def fit(model_type):
            smoke, _, _ = formal_response_fixture(model_type)
            counts = []

            def tokenize(prompt, **_):
                count = prompt.count(" x")
                counts.append(count)
                return [1] * (count + count // 2 + 2)

            with patch.object(smoke, "tokenize", side_effect=tokenize):
                return smoke.fit_prompt("head", "tail", 41), counts

        baseline, old_probes = fit("kimi_k3")
        optimized, new_probes = fit("kimi_k31")
        self.assertEqual(optimized, baseline)
        self.assertEqual(new_probes, [old_probes[0], 19, *old_probes[1:]])

    def test_k31_reuses_a_saved_exact_prompt_with_the_same_thinking_mode(self):
        smoke, _, _ = formal_response_fixture("kimi_k31")
        with tempfile.TemporaryDirectory() as tmp:
            smoke.args.output = Path(tmp) / "accuracy.json"
            smoke.save_token_fixture("same prompt", [9, 8, 7], enable_thinking=False)
            path = next((Path(tmp) / "token-fixtures").glob("*.json.gz"))
            before = path.read_bytes()

            with patch.object(
                smoke.opener,
                "open",
                side_effect=lambda *_a, **_k: io.BytesIO(b'{"token_ids":[4,5]}'),
            ) as fetch:
                self.assertEqual(
                    smoke.verified_input_length("same prompt", enable_thinking=False), 3
                )
                fetch.assert_not_called()
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(
                    smoke.verified_input_length("same prompt", enable_thinking=None), 2
                )
                fetch.assert_called_once()
                path.unlink()
                self.assertEqual(
                    smoke.verified_input_length("same prompt", enable_thinking=False), 2
                )
                self.assertEqual(fetch.call_count, 2)
            with gzip.open(path, "rt", encoding="utf-8") as artifact:
                self.assertEqual(json.load(artifact)["input_ids"], [4, 5])
        self.assertEqual(smoke.prompt_timing()["tokenize_calls"], 2)
        self.assertEqual(smoke.prompt_timing()["token_cache_hits"], 1)

    def test_k3_rechecks_a_saved_prompt_with_the_server(self):
        smoke, _, _ = formal_response_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            smoke.args.output = Path(tmp) / "accuracy.json"
            smoke.save_token_fixture("prompt", [1])
            with patch.object(smoke, "tokenize", return_value=[2, 3]) as tokenize:
                self.assertEqual(
                    smoke.verified_input_length("prompt", enable_thinking=None), 2
                )
            tokenize.assert_called_once_with("prompt", enable_thinking=None)
        self.assertEqual(smoke.prompt_timing()["token_cache_hits"], 0)

    def test_request_stage_reports_start_and_failure_without_request_data(self):
        smoke, case, _ = formal_response_fixture()
        stdout = io.StringIO()
        with redirect_stdout(stdout), patch.object(
            smoke, "health", side_effect=SmokeFailure("secret prompt")
        ), self.assertRaises(SmokeFailure):
            smoke.run_stage("batch_all_miss", [case])
        events = [
            json.loads(line[len(REQUEST_PROGRESS_PREFIX) :])
            for line in stdout.getvalue().splitlines()
            if line.startswith(REQUEST_PROGRESS_PREFIX)
        ]
        self.assertEqual(
            [event["event"] for event in events],
            ["request_stage_start", "request_stage"],
        )
        for event in events:
            self.assertEqual(event["stage"], "batch_all_miss")
            self.assertEqual(event["case_count"], 1)
            self.assertFalse(event["passed"])
            self.assertNotIn("secret prompt", json.dumps(event))

    def test_capped_request_respects_shorter_request_and_case_timeouts(self):
        for request_timeout, case_timeout, expected_deadline in (
            (17, None, 17),
            (900, 11, 11),
            (900, None, 300),
        ):
            with self.subTest(
                request_timeout=request_timeout, case_timeout=case_timeout
            ):
                smoke, case, response = formal_response_fixture()
                smoke.args.timeout = request_timeout
                case = replace(case, timeout_s=case_timeout)
                completed = types.SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(response).encode() + b"\n__K3_HTTP_STATUS__200",
                    stderr=b"",
                )
                with patch.object(smoke, "tokenize", return_value=[1]), patch(
                    "rtp_llm.test.smoke.kimi_k3.text_smoke.subprocess.run",
                    return_value=completed,
                ) as run:
                    result = smoke.request(case)
                command = run.call_args.args[0]
                self.assertEqual(
                    command[command.index("--max-time") + 1], str(expected_deadline)
                )
                self.assertEqual(run.call_args.kwargs["timeout"], expected_deadline + 1)
                self.assertEqual(
                    result["content"], response["choices"][0]["message"]["content"]
                )
                self.assertEqual(len(smoke.records), 1)

    def test_rdma_prewarm_applies_its_shorter_transport_timeout(self):
        smoke, _, response = formal_response_fixture()
        smoke.args.rdma_prewarm_attempts = 1
        smoke.args.rdma_prewarm_timeout = 7
        smoke.args.rdma_prewarm_settle_s = 0
        smoke.args.batch_size = 1
        response["choices"][0]["message"]["content"] = "6400"
        completed = types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(response).encode() + b"\n__K3_HTTP_STATUS__200",
            stderr=b"",
        )
        with patch.object(smoke, "health"), patch.object(
            smoke, "tokenize", return_value=[1]
        ), patch(
            "rtp_llm.test.smoke.kimi_k3.text_smoke.subprocess.run",
            return_value=completed,
        ) as run:
            smoke.prewarm_rdma_pool()
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--max-time") + 1], "7")
        self.assertEqual(run.call_args.kwargs["timeout"], 8)
        self.assertTrue(smoke.rdma_prewarm_attempts[0]["passed"])
        self.assertEqual(smoke.records[0]["phase"], "prewarm")

    def test_rdma_prewarm_retries_without_counting_partial_attempt(self):
        smoke, _, _ = formal_response_fixture()
        smoke.args.rdma_prewarm_attempts = 2
        smoke.args.rdma_prewarm_timeout = 7
        smoke.args.rdma_prewarm_backoff_s = 0
        smoke.args.rdma_prewarm_settle_s = 0
        smoke.args.batch_size = 2

        def request(case, barrier, **kwargs):
            if case.name == "rdma_prewarm_1_0":
                raise TransportFailure("HTTP 503")
            return {"name": case.name, "input_len": 10}

        with patch.object(smoke, "health"), patch.object(
            smoke, "_request", side_effect=request
        ):
            smoke.prewarm_rdma_pool()

        failed, succeeded = smoke.rdma_prewarm_attempts
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["completed_cases"], ["rdma_prewarm_1_1"])
        self.assertEqual(failed["failures"][0]["name"], "rdma_prewarm_1_0")
        self.assertTrue(succeeded["passed"])
        self.assertEqual(
            {row["name"] for row in smoke.records}, set(succeeded["case_names"])
        )
        self.assertEqual(smoke.failures, [])

    def test_failed_request_keeps_case_artifact_and_transport_classification(self):
        smoke, case, _ = formal_response_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            smoke.args.output = Path(tmp) / "accuracy.json"
            with patch.object(
                smoke, "_request", side_effect=TransportFailure("HTTP 503")
            ), self.assertRaisesRegex(
                TransportFailure, "case=short-exact-answer owner=0.*HTTP 503"
            ):
                smoke.request(case)
            failure = smoke.failures[0]
            self.assertTrue(failure["retryable"])
            artifact = json.loads(Path(failure["artifact"]).read_text())
            self.assertEqual(artifact["name"], case.name)
            self.assertIn("HTTP 503", artifact["error"])

    def test_prefill_cache_tier_does_not_misclassify_memory_as_device(self):
        counters = {
            "prefill_total_reuse_len": 8192,
            "prefill_local_reuse_len": 8192,
            "prefill_memory_reuse_len": 4096,
            "prefill_disk_reuse_len": 0,
        }
        self.assertEqual(prefill_cache_tiers(counters)["device"], 4096)
        smoke, case, response = formal_response_fixture()
        response["aux_info"].update(counters, input_len=12288, reuse_len=8192)
        case = Case(
            case.name,
            case.prompt,
            case.expected_regex,
            "partial",
            expected_json=case.expected_json,
            expected_cache_tier="memory",
        )
        with self.assertRaisesRegex(SmokeFailure, "expected memory"):
            smoke.validate(case, response, 0.1, 32)
        response["aux_info"].update(
            prefill_memory_reuse_len=8192, prefill_local_reuse_len=8192
        )
        result = smoke.validate(case, response, 0.1, 32)
        self.assertEqual(result["prefill_cache_tiers"]["device"], 0)
        self.assertEqual(result["prefill_cache_tiers"]["memory"], 8192)

    def test_long_history_requires_actual_uncached_q_below_64k(self):
        smoke, case, response = formal_response_fixture()
        case = Case(
            case.name,
            case.prompt,
            case.expected_regex,
            "hit",
            expected_json=case.expected_json,
            allow_long_history=True,
        )
        response["aux_info"].update(input_len=2_065_000, reuse_len=1_998_848)
        with self.assertRaisesRegex(SmokeFailure, "uncached Q tokens"):
            smoke.validate(case, response, 0.1, 32)
        response["aux_info"].update(input_len=2_063_000)
        self.assertEqual(
            smoke.validate(case, response, 0.1, 32)["input_len"], 2_063_000
        )

    def test_formal_response_rejects_length_finish_reason(self):
        smoke, case, response = formal_response_fixture()
        response["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(SmokeFailure, "finish_reason=length"):
            smoke.validate(case, response, 0.1, 32)

    def test_response_rejects_unicode_replacement_in_reasoning(self):
        smoke, case, response = formal_response_fixture()
        response["choices"][0]["message"]["reasoning_content"] = "\ufffd"
        with self.assertRaisesRegex(SmokeFailure, "Unicode replacement"):
            smoke.validate(case, response, 0.1, 32)

    def test_malformed_choice_still_raises_smoke_failure(self):
        smoke, case, response = formal_response_fixture()
        response["choices"] = [[]]
        with self.assertRaisesRegex(SmokeFailure, "malformed response"):
            smoke.validate(case, response, 0.1, 32)

    def test_cache_preparation_accepts_bounded_reasoning_output(self):
        smoke, case, response = formal_response_fixture()
        case = replace(
            case, preparation_only=True, expected_json=None, expected_regex=r"."
        )
        response["choices"][0]["finish_reason"] = "length"
        response["choices"][0]["message"] = {
            "content": "",
            "reasoning_content": "preparing cache",
        }
        self.assertEqual(
            smoke.validate(case, response, 0.1, 32)["finish_reason"], "length"
        )


if __name__ == "__main__":
    unittest.main()
