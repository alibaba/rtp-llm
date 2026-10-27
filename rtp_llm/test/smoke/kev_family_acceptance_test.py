"""One-start HTTP acceptance for an exported Kev family checkpoint.

Required environment: KEV_ACCEPTANCE_MODEL_PATH, KEV_ACCEPTANCE_REFERENCE.
Reference JSON (computed independently; this test never writes goldens)::

    {"source": {"reference_revision": "...",
                "checkpoint_manifest_sha256": "sha256 of kev_config.json"},
     "tolerances": {"max_abs_probability_error": 0.01,
                    "require_same_choice": true},
     "cases": [{"name": "short", "request": {"state": "...",
                "questions": [{"instr": "...", "options": ["a", "b"]}]},
                "probabilities": [[0.2, 0.8]]}, ...]}

Cases named short, long and changed_state are required. Long must span at least
two physical cache blocks. Each case can contain multiple questions. Tolerances
must be selected before running, for the checkpoint, precision and reference.

KEV_ACCEPTANCE_QUANTIZATION is BF16 (default) or FP8_PER_CHANNEL_COMPRESSED.
KEV_ACCEPTANCE_REUSE_CACHE and KEV_ACCEPTANCE_SHARED_PREFIX are 0/1 (default 1/0).
KEV_ACCEPTANCE_SMOKE_ARGS supplies additional server arguments. A single
MagaServerManager owns startup and teardown; GPU ownership is the Bazel
test-execution wrapper's responsibility.

With KEV_ACCEPTANCE_ENDPOINT, only HTTP assertions run against that endpoint.
The caller owns that server's lifecycle (including the final image smoke);
the test neither starts nor stops it. Its actual launch configuration/image
must be recorded separately. Requested env settings do not configure it.

Raw HTTP records and a summary go to KEV_ACCEPTANCE_OUTPUT_DIR, otherwise
TEST_UNDECLARED_OUTPUTS_DIR. Repeated requests exercise cache-eligible behavior;
the current Kev HTTP response has no reuse_len, so this is not cache-hit proof.
"""

import copy
import hashlib
import json
import math
import os
import shlex
import tempfile
import time
import unittest
from pathlib import Path

import requests
from tokenizers import Tokenizer


def _flag(name, default):
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value


class KevFamilyAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.failures = []
        self.records = []
        self.report = {"status": "NOT_RUN", "cache_hit_observation": "NOT_EXPOSED"}
        self.output = Path(
            os.environ.get("KEV_ACCEPTANCE_OUTPUT_DIR")
            or os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
            or tempfile.mkdtemp(prefix="kev-acceptance-")
        )
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ("requests.jsonl", "summary.json"):
            self.assertFalse(
                (self.output / name).exists(),
                f"use a fresh output directory: {self.output}",
            )
        self.addCleanup(self._write_report)
        self.model = Path(os.environ["KEV_ACCEPTANCE_MODEL_PATH"])
        reference_path = Path(os.environ["KEV_ACCEPTANCE_REFERENCE"])
        reference_bytes = reference_path.read_bytes()
        reference = json.loads(reference_bytes)
        manifest_bytes = (self.model / "kev_config.json").read_bytes()
        self.manifest = json.loads(manifest_bytes)
        source = reference["source"]
        self.assertTrue(source["reference_revision"], "independent reference required")
        self.assertEqual(
            source["checkpoint_manifest_sha256"],
            hashlib.sha256(manifest_bytes).hexdigest(),
            "reference belongs to a different checkpoint manifest",
        )
        self.tolerance = reference["tolerances"]["max_abs_probability_error"]
        self.assertIs(type(self.tolerance) in (float, int), True)
        self.assertTrue(math.isfinite(self.tolerance) and 0 <= self.tolerance < 1)
        self.same_choice = reference["tolerances"]["require_same_choice"]
        self.assertIs(type(self.same_choice), bool)
        self.cases = {item["name"]: item for item in reference["cases"]}
        self.assertEqual(len(self.cases), len(reference["cases"]), "duplicate cases")
        self.assertTrue({"short", "long", "changed_state"} <= self.cases.keys())
        tokenizer = Tokenizer.from_file(str(self.model / "tokenizer.json"))
        for item in self.cases.values():
            request = item["request"]
            self.assertEqual(set(request), {"state", "questions"})
            self.assertTrue(1 <= len(request["questions"]) <= 64)
            self._validate_rows(item["probabilities"], request)
        long_tokens = len(
            tokenizer.encode(
                self.cases["long"]["request"]["state"], add_special_tokens=False
            ).ids
        )
        self.assertGreaterEqual(long_tokens, 128, "long state must span two blocks")
        self.assertNotEqual(
            self.cases["short"]["request"]["state"],
            self.cases["changed_state"]["request"]["state"],
        )
        self.quantization = os.environ.get(
            "KEV_ACCEPTANCE_QUANTIZATION", "BF16"
        ).upper()
        if self.quantization == "PTPC":
            self.quantization = "FP8_PER_CHANNEL_COMPRESSED"
        self.assertIn(self.quantization, ("BF16", "FP8_PER_CHANNEL_COMPRESSED"))
        reuse = _flag("KEV_ACCEPTANCE_REUSE_CACHE", "1")
        shared = _flag("KEV_ACCEPTANCE_SHARED_PREFIX", "0")
        self.assertFalse(shared == "1" and reuse == "0")
        self.endpoint = os.environ.get("KEV_ACCEPTANCE_ENDPOINT", "").rstrip("/")
        self.report.update(
            source=source,
            reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
            requested_quantization=self.quantization,
            requested_reuse_cache=reuse,
            requested_shared_prefix=shared,
            lifecycle="external" if self.endpoint else "MagaServerManager",
            tolerances=reference["tolerances"],
            long_state_tokens=long_tokens,
            model_path=str(self.model),
        )
        if not self.endpoint:
            from rtp_llm.test.utils.maga_server_manager import MagaServerManager

            args = [
                "--act_type",
                "bf16",
                "--tp_size",
                "1",
                "--world_size",
                "1",
                "--enable_cuda_graph",
                "false",
                "--frontend_server_count",
                "1",
                "--warm_up",
                "0",
                "--reuse_cache",
                reuse,
                "--seq_size_per_block",
                "64",
                "--kernel_seq_size_per_block",
                "16",
                "--max_seq_len",
                "8192",
                "--max_context_batch_size",
                "16",
                "--max_batch_tokens_size",
                "16384",
                "--kv_cache_mem_mb",
                "4096",
            ]
            args += shlex.split(os.environ.get("KEV_ACCEPTANCE_SMOKE_ARGS", ""))
            # Keep the requested precision explicit even if the shell has QUANTIZATION.
            args += [
                "--quantization",
                "" if self.quantization == "BF16" else self.quantization,
            ]
            self.server = MagaServerManager(
                env_args={
                    "KEV_SHARED_PREFIX": shared,
                    "USE_FLYDSL": "1",
                    "LOAD_PYTHON_MODEL": "1",
                },
                smoke_args_str=shlex.join(args),
                role_name="kev_acceptance",
            )
            self.addCleanup(self.server.stop_server)
            self.assertTrue(
                self.server.start_server(
                    model_path=str(self.model),
                    model_type="",
                    timeout=int(os.environ.get("KEV_ACCEPTANCE_START_TIMEOUT", "1600")),
                ),
                "Kev server failed to become healthy",
            )
            self.endpoint = f"http://127.0.0.1:{self.server.port}"
            self.report["server_log"] = self.server.log_file_path
            self.report["server_args"] = args
        self.assertTrue(self.endpoint.startswith(("http://", "https://")))
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        health = self.session.get(f"{self.endpoint}/health", timeout=30)
        self.assertEqual(health.status_code, 200, health.text)

    def _write_report(self):
        self.report["failures"] = self.failures
        self.report["http_records"] = len(self.records)
        (self.output / "requests.jsonl").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n" for record in self.records
            ),
            encoding="utf-8",
        )
        (self.output / "summary.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _post(self, label, path, request, status=200):
        started = time.perf_counter()
        record = {"case": label, "endpoint": path, "request": request}
        try:
            response = self.session.post(
                self.endpoint + path,
                json=request,
                headers={"x-typesafe-request-id": "kev-acceptance-" + label},
                timeout=float(os.environ.get("KEV_ACCEPTANCE_HTTP_TIMEOUT", "300")),
            )
            record.update(
                status_code=response.status_code,
                response_text=response.text,
                headers={
                    key: response.headers.get(key)
                    for key in ("x-typesafe-request-id", "server-timing")
                },
            )
            self.assertEqual(response.status_code, status, response.text)
            if status != 200:
                return None
            actual = response.json()
            record["response"] = actual
            return actual
        except Exception as error:
            record["error"] = str(error)
            raise
        finally:
            record["http_elapsed_ms"] = (time.perf_counter() - started) * 1000
            self.records.append(record)
            with (self.output / "requests.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _validate_rows(self, rows, request):
        self.assertEqual(len(rows), len(request["questions"]))
        for values, question in zip(rows, request["questions"]):
            self.assertEqual(len(values), len(question["options"]))
            self.assertTrue(
                all(
                    type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1
                    for p in values
                )
            )
            self.assertAlmostEqual(sum(values), 1.0, delta=1e-5)

    def _decide(self, label, request, expected=None):
        actual = self._post(label, "/v1/decide", request)
        rows = [result["probabilities"] for result in actual["results"]]
        self._validate_rows(rows, request)
        for result, question in zip(actual["results"], request["questions"]):
            choice = max(
                range(len(result["probabilities"])),
                key=result["probabilities"].__getitem__,
            )
            self.assertEqual(result["choice"], choice)
            self.assertEqual(result["option"], question["options"][choice])
        if expected is not None:
            self._validate_rows(expected, request)
            error = max(
                abs(p - q) for row, ref in zip(rows, expected) for p, q in zip(row, ref)
            )
            self.records[-1]["max_abs_probability_error"] = error
            self.assertLessEqual(error, self.tolerance, label)
            if self.same_choice:
                self.assertEqual(
                    [max(range(len(row)), key=row.__getitem__) for row in rows],
                    [max(range(len(row)), key=row.__getitem__) for row in expected],
                    label,
                )
        return rows

    def _case(self, name, function):
        try:
            function()
        except Exception as error:
            self.failures.append({"case": name, "error": str(error)})

    def test_family_acceptance(self):
        # Each fixture's first and repeated requests share exact reference input.
        # External endpoints might already be warm; no cold/hit claim is inferred.
        for name, item in self.cases.items():
            for phase in ("first", "repeat"):
                label = f"{name}_{phase}"
                self._case(
                    label,
                    lambda item=item, label=label: self._decide(
                        label, item["request"], item["probabilities"]
                    ),
                )

        short = self.cases["short"]
        for count in (1, 64):
            request = copy.deepcopy(short["request"])
            indexes = [i % len(request["questions"]) for i in range(count)]
            request["questions"] = [request["questions"][i] for i in indexes]
            expected = [short["probabilities"][i] for i in indexes]
            self._case(
                f"questions_{count}",
                lambda request=request, expected=expected, count=count: self._decide(
                    f"questions_{count}", request, expected
                ),
            )

        invalid = [
            {"state": "x", "questions": []},
            {"state": "x", "questions": [{"instr": "x", "options": []}]},
            {"state": "x", "questions": [short["request"]["questions"][0]] * 65},
            {"state": "x", "questions": short["request"]["questions"], "extra": True},
        ]
        for i, request in enumerate(invalid):
            self._case(
                f"invalid_{i}",
                lambda request=request, i=i: self._post(
                    f"invalid_{i}", "/v1/decide", request, 400
                ),
            )
        self._case("systemone_mapping", self._systemone)
        self._case(
            "systemone_invalid",
            lambda: self._post(
                "systemone_invalid",
                "/v1/systemone",
                {"state": "x", "questions": {}},
                422,
            ),
        )
        # Recovery after rejected requests is part of the same server lifecycle.
        self._case(
            "after_invalid",
            lambda: self._decide(
                "after_invalid", short["request"], short["probabilities"]
            ),
        )
        self.report["status"] = "FAIL" if self.failures else "PASS"
        self.assertFalse(self.failures, json.dumps(self.failures, ensure_ascii=False))

    def _systemone(self):
        state = self.cases["short"]["request"]["state"]
        request = {
            "model": "kev-acceptance",
            "state": state,
            "questions": {
                "choice": {
                    "type": "choice",
                    "instructions": "Select a color",
                    "criteria": {"red": "warm", "blue": "cool"},
                },
                "score": {
                    "type": "score",
                    "instructions": "Rate urgency",
                    "criteria": ["low", "high"],
                },
                "noul": {
                    "type": "noul",
                    "instructions": "Does the statement match?",
                    "criteria": {"false": "misses", "true": "matches"},
                },
            },
        }
        raw = {
            "state": state,
            "questions": [
                {"instr": "Select a color", "options": ["red: warm", "blue: cool"]},
                {"instr": "Rate urgency", "options": ["low", "high"]},
                {
                    "instr": "Does the statement match?",
                    "options": ["no: misses", "yes: matches"],
                },
            ],
        }
        self._decide("systemone_warm", raw)
        rows = self._decide("systemone_raw", raw)
        actual = self._post("systemone", "/v1/systemone", request)
        self.assertEqual(actual["model"], "kev-acceptance")
        self.assertEqual(set(actual["answers"]), {"choice", "score", "noul"})
        answers = actual["answers"]
        for index, (name, keys) in enumerate(
            (("choice", ("red", "blue")), ("score", ("0", "1")))
        ):
            self.assertEqual(answers[name]["type"], name)
            for key, probability in zip(keys, rows[index]):
                self.assertAlmostEqual(
                    answers[name]["probabilities"][key],
                    probability,
                    delta=self.tolerance + 0.00005,
                )
            self.assertAlmostEqual(
                answers[name]["confidence"],
                2 * max(rows[index]) - 1,
                delta=2 * self.tolerance + 0.00005,
            )
        if self.same_choice:
            self.assertEqual(
                answers["choice"]["choice"],
                ("red", "blue")[max(range(2), key=rows[0].__getitem__)],
            )
        self.assertEqual(answers["score"]["legend"], {"0": "low", "1": "high"})
        self.assertAlmostEqual(
            answers["score"]["score"], rows[1][1], delta=self.tolerance + 0.00005
        )
        self.assertEqual(answers["noul"]["type"], "noul")
        self.assertAlmostEqual(
            answers["noul"]["noul"], rows[2][1], delta=self.tolerance + 0.00005
        )
        for key in ("input_tokens", "output_tokens"):
            self.assertIs(type(actual["usage"][key]), int)
            self.assertGreater(actual["usage"][key], 0)
        self.assertGreaterEqual(actual["latency_ms"], 0)
        self.assertEqual(
            self.records[-1]["headers"]["x-typesafe-request-id"],
            "kev-acceptance-systemone",
        )
        self.assertRegex(
            self.records[-1]["headers"]["server-timing"], r"^app;dur=\d+\.\d$"
        )


if __name__ == "__main__":
    unittest.main()
