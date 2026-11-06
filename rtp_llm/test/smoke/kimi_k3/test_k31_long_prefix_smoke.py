"""CPU tests for the two-request K3.1 cached QA evidence contract."""

import contextlib
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import k31_long_prefix_smoke as smoke

EXPECTED = {
    "early": "CEDAR-1827",
    "middle": "MAPLE-4639",
    "late": "BIRCH-7251",
    "square": 1369,
}


@lru_cache(maxsize=1024)
def token_for(word):
    return int(hashlib.sha256(word.encode()).hexdigest()[:6], 16)


def tokens_for(messages):
    roles = {"user": 1, "assistant": 2}
    tokens = [0]
    for message in messages:
        tokens.append(roles[message["role"]])
        words = re.findall(r"[A-Za-z0-9]+|[^\w\s]", message["content"])
        tokens.extend(map(token_for, words))
    return tokens + [2]


def load_raw(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def save_raw(path, value):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream)


class LongPrefixSmokeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "evidence"
        self.geometry = self.root / "geometry.json"
        self.geometry.write_text(
            json.dumps(
                dict(
                    runtime_verified=True,
                    runtime_layers=93,
                    source_shards=8,
                    page_size=32,
                    kernel_page_size=16,
                    checkpoint_span=128,
                    cache_key_stride=64,
                )
            )
        )
        self.generations = []
        self.seed_reuse = 0
        self.answer = json.dumps(EXPECTED)
        test = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                tokens = tokens_for(payload["messages"])
                if self.path == "/tokenize":
                    body = {"token_ids": tokens}
                elif self.path == "/v1/chat/completions":
                    test.generations.append(payload)
                    seed = len(test.generations) == 1
                    seed_tokens = tokens_for(test.generations[0]["messages"])
                    common = next(
                        (
                            i
                            for i, pair in enumerate(zip(seed_tokens, tokens))
                            if pair[0] != pair[1]
                        ),
                        len(seed_tokens),
                    )
                    body = {
                        "id": f"chatcmpl-{len(test.generations)}",
                        "choices": [
                            dict(
                                finish_reason="stop",
                                message=dict(
                                    content="RECEIVED" if seed else test.answer
                                ),
                            )
                        ],
                        "debug_info": dict(input_ids=tokens, output_ids=[[37]]),
                        "aux_info": dict(
                            pd_sep=True,
                            input_len=len(tokens),
                            output_len=1,
                            prefill_total_reuse_len=(
                                test.seed_reuse if seed else common // 128 * 128
                            ),
                            role_addrs=payload["extra_configs"]["role_addrs"],
                        ),
                    }
                else:
                    self.send_error(404)
                    return
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def test_direct_cli_works_without_pythonpath(self):
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        script = Path(smoke.__file__).resolve()
        result = subprocess.run(
            [sys.executable, str(script), "verify", "--help"],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)

    def run_case(self, target=4096):
        with patch.object(smoke, "TARGET_TOKENS", target), contextlib.redirect_stdout(
            io.StringIO()
        ):
            return smoke.main(
                [
                    "run",
                    "--base-url",
                    f"http://127.0.0.1:{self.server.server_port}",
                    "--namespace",
                    "isolated-test-17",
                    "--output",
                    str(self.output),
                    "--geometry",
                    str(self.geometry),
                    "--decode-role-addr",
                    "127.0.0.2:1234:4321",
                    "--decode-role-addr",
                    "127.0.0.3:1234:4321",
                ]
            )

    def test_exact_continuation_preserves_archive_and_records_serving_evidence(self):
        self.assertEqual(smoke.TARGET_TOKENS, 1_048_576)
        self.assertEqual(self.run_case(1_048_576), 0)
        self.assertEqual(len(self.generations), 2)
        seed, continuation = (payload["messages"] for payload in self.generations)
        self.assertEqual(continuation[:1], seed)
        self.assertEqual(continuation[1], {"role": "assistant", "content": "RECEIVED"})
        self.assertEqual(len(tokens_for(continuation)), 1_048_576)
        archive = seed[0]["content"]
        for value in ("CEDAR-1827", "MAPLE-4639", "BIRCH-7251"):
            self.assertEqual(archive.count(value), 1)
            self.assertNotIn(value, continuation[-1]["content"])
        self.assertNotIn("1369", continuation[-1]["content"])
        self.assertIn("isolated-test-17", archive)
        verified = smoke.verify(self.output, self.geometry)
        self.assertEqual(verified["continuation_tokens"], 1_048_576)
        self.assertEqual(verified["reuse_unit_tokens"], 128)
        self.assertEqual(verified["seed_reuse_tokens"], 0)
        self.assertEqual(verified["reuse_tokens"] % 128, 0)
        self.assertGreater(verified["reuse_tokens"], 1_046_000)
        self.assertEqual(verified["new_tokens"], 1_048_576 - verified["reuse_tokens"])
        self.assertEqual(verified["answer"], self.answer)
        self.assertEqual(
            verified["continuation_token_sha256"],
            hashlib.sha256(json.dumps(tokens_for(continuation)).encode()).hexdigest(),
        )
        self.assertEqual(
            load_raw(self.output / "continuation-response.json.gz")["debug_info"][
                "input_ids"
            ],
            tokens_for(continuation),
        )
        self.assertEqual(
            self.generations[0]["extra_configs"]["role_addrs"][0]["ip"], "127.0.0.2"
        )
        self.assertEqual(
            self.generations[1]["extra_configs"]["role_addrs"][0]["ip"], "127.0.0.3"
        )

    def test_seed_cache_hit_blocks_formal_request_and_keeps_response(self):
        self.seed_reuse = 128
        with self.assertRaisesRegex(ValueError, "seed.*cache"):
            self.run_case()
        self.assertEqual(len(self.generations), 1)
        self.assertTrue((self.output / "seed-response.json.gz").exists())
        failure = json.loads((self.output / "RESULT.json").read_text())
        self.assertIs(failure["passed"], False)
        self.assertIn("cache", failure["error"])

    def test_verifier_rejects_invalid_answers_from_raw_model_output(self):
        self.assertEqual(self.run_case(), 0)
        path = self.output / "continuation-response.json.gz"
        response = load_raw(path)
        for answer in (
            "```json\n" + self.answer + "\n```",
            self.answer.replace('"early":', '"early": "CEDAR-1827", "early":'),
            self.answer.replace("1369", "1369.0"),
            self.answer.replace("MAPLE-4639", "WRONG"),
        ):
            with self.subTest(answer=answer), patch.object(
                smoke, "TARGET_TOKENS", 4096
            ):
                response["choices"][0]["message"]["content"] = answer
                save_raw(path, response)
                with self.assertRaisesRegex(ValueError, "answer"):
                    smoke.verify(self.output, self.geometry)

    def test_verifier_rejects_a_nonstandard_generation_override(self):
        self.assertEqual(self.run_case(), 0)
        path = self.output / "continuation-request.json.gz"
        payload = load_raw(path)
        payload["extra_configs"]["force_sp_accept"] = True
        save_raw(path, payload)
        with patch.object(smoke, "TARGET_TOKENS", 4096):
            with self.assertRaisesRegex(ValueError, "ordinary generation request"):
                smoke.verify(self.output, self.geometry)

    def test_verifier_checks_actual_pd_exact_ids_and_aligned_prefix_reuse(self):
        self.assertEqual(self.run_case(), 0)
        path = self.output / "continuation-response.json.gz"
        original = load_raw(path)
        actual_reuse = original["aux_info"]["prefill_total_reuse_len"]
        for key, value, error in (
            ("pd_sep", 1, "PD"),
            ("input_len", 4095, "input length"),
            ("prefill_total_reuse_len", actual_reuse - 1, "aligned"),
            ("prefill_total_reuse_len", 128, "too little"),
            ("prefill_total_reuse_len", 4096, "common token prefix"),
        ):
            with self.subTest(key=key, value=value), patch.object(
                smoke, "TARGET_TOKENS", 4096
            ):
                response = {
                    **original,
                    "aux_info": {**original["aux_info"], key: value},
                }
                save_raw(path, response)
                with self.assertRaisesRegex(ValueError, error):
                    smoke.verify(self.output, self.geometry)
        original["debug_info"]["input_ids"][-1] += 1
        save_raw(path, original)
        with patch.object(smoke, "TARGET_TOKENS", 4096):
            with self.assertRaisesRegex(ValueError, "token IDs"):
                smoke.verify(self.output, self.geometry)


if __name__ == "__main__":
    unittest.main()
