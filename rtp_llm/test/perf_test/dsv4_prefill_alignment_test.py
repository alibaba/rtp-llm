"""CPU tests for workload accounting and the real HTTP collection path."""

import argparse
import copy
import importlib.util
import json
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "bench", Path(__file__).with_name("dsv4_prefill_alignment.py")
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

CASE = {"id": "case", "text": "example", "input_ids": [1, 2, 3]}
RTP = {"finished": True, "aux_info": {"input_len": 3, "output_len": 1,
        "reuse_len": 0, "pd_sep": False}, "input_ids": [[1, 2, 3]],
       "output_ids": [[2]], "logits": [[0.0, 0.5, 1.5, -1.0]]}
SG = {"output_ids": [2], "meta_info": {"prompt_tokens": 3, "completion_tokens": 1,
      "cached_tokens": 0, "finish_reason": {"type": "length"},
      "output_top_logprobs": [[[-0.5, 2, None], [-1.5, 1, None]]]}}


class ContractTests(unittest.TestCase):
    def test_exact_one_token_greedy_requests(self):
        rtp = bench.request_body("rtp", CASE, True)["generate_config"]
        sg = bench.request_body("sglang", CASE, True)
        self.assertEqual((rtp["min_new_tokens"], rtp["max_new_tokens"]), (1, 1))
        self.assertEqual(sg["input_ids"], CASE["input_ids"])
        self.assertTrue(rtp["return_input_ids"])
        self.assertFalse(bench.request_body("rtp", CASE, False)["generate_config"]["return_logits"])

    def test_rtp_and_sg_normalize(self):
        self.assertEqual(bench.normalize("rtp", RTP, CASE, True)["output_id"], 2)
        self.assertEqual(bench.normalize("sglang", SG, CASE, True)["output_id"], 2)

    def test_same_length_different_token_ids_fail(self):
        raw = copy.deepcopy(RTP)
        raw["input_ids"] = [[1, 3, 2]]
        with self.assertRaisesRegex(ValueError, "input IDs"):
            bench.normalize("rtp", raw, CASE, True)

    def test_input_truncation_fails(self):
        raw = copy.deepcopy(SG)
        raw["meta_info"]["prompt_tokens"] = 2
        with self.assertRaisesRegex(ValueError, "input length"):
            bench.normalize("sglang", raw, CASE, False)

    def test_zero_or_extra_outputs_fail(self):
        for length in [0, 2]:
            raw = copy.deepcopy(RTP)
            raw["aux_info"]["output_len"] = length
            with self.assertRaisesRegex(ValueError, "exactly one"):
                bench.normalize("rtp", raw, CASE, False)

    def test_cached_requests_fail(self):
        for backend, original, key, field in [
            ("rtp", RTP, "aux_info", "reuse_len"),
            ("sglang", SG, "meta_info", "cached_tokens"),
        ]:
            for value in [None, 3]:
                raw = copy.deepcopy(original)
                raw[key][field] = value
                with self.assertRaises(ValueError):
                    bench.normalize(backend, raw, CASE, False)

    def test_pd_response_fails_stage_one(self):
        raw = copy.deepcopy(RTP)
        raw["aux_info"]["pd_sep"] = True
        with self.assertRaisesRegex(ValueError, "pd_sep"):
            bench.normalize("rtp", raw, CASE, False)

    def test_nonfinite_logits_fail(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            raw = copy.deepcopy(RTP)
            raw["logits"][0][0] = value
            with self.assertRaisesRegex(ValueError, "non-finite"):
                bench.normalize("rtp", raw, CASE, True)

    def test_stable_logsumexp(self):
        raw = copy.deepcopy(RTP)
        raw["logits"] = [[10000.0, 10001.0, 10002.0, 9999.0]]
        result = bench.normalize("rtp", raw, CASE, True)
        self.assertAlmostEqual(sum(__import__("math").exp(p) for p in result["top_logprobs"].values()), 1.0)

    def test_unfinished_response_fails(self):
        raw = copy.deepcopy(RTP)
        raw["finished"] = False
        with self.assertRaisesRegex(ValueError, "finish"):
            bench.normalize("rtp", raw, CASE, False)

    def test_missing_sg_logprobs_fail(self):
        raw = copy.deepcopy(SG)
        del raw["meta_info"]["output_top_logprobs"]
        with self.assertRaisesRegex(ValueError, "probabilities"):
            bench.normalize("sglang", raw, CASE, True)

    def test_corpus_rejects_empty_duplicate_and_boolean_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            for rows in [[], [CASE, CASE], [{**CASE, "input_ids": [True]}]]:
                path.write_text("\n".join(json.dumps(r) for r in rows))
                with self.assertRaises(ValueError):
                    bench.load_corpus(path)

    def test_unicode_line_separator_inside_prompt_is_not_a_jsonl_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            case = {**CASE, "text": "first\u2028second\u0085third"}
            path.write_text(json.dumps(case, ensure_ascii=False) + "\n")
            self.assertEqual(bench.load_corpus(path), [case])


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = copy.deepcopy(RTP)
        self.http_status = 200
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(owner.http_status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(owner.payload).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.corpus = self.root / "corpus.jsonl"
        self.corpus.write_text(json.dumps(CASE) + "\n")
        self.service = self.root / "service.json"
        self.service.write_text(json.dumps({"tp": 4, "cp": 1, "ep": 1, "dp": 1,
            "mtp": False, "prefix_cache": False, "pd": False,
            "checkpoint_manifest_sha256": "test-fixture", "runtime_verified": True,
            "runtime_evidence": "mock-server-test-only"}))

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.temp.cleanup()

    def args(self, name, mode="accuracy", **kwargs):
        result = dict(backend="rtp", mode=mode, corpus=self.corpus,
            service_manifest=self.service, output=self.root / name,
            accuracy_run=None, concurrency=1, rounds=3, warmup_rounds=1,
            timeout=5, url=self.url)
        result.update(kwargs)
        return argparse.Namespace(**result)

    def test_collection_then_performance_requires_matching_contract(self):
        accuracy = self.args("accuracy")
        bench.collect(accuracy)
        summary = json.loads((accuracy.output / "summary.json").read_text())
        self.assertTrue(summary["valid"])
        self.assertFalse(summary["scored_performance"])
        self.assertFalse(summary["whole_model_accuracy_pass"])
        perf = self.args("perf", "performance", accuracy_run=accuracy.output)
        bench.collect(perf)
        result = json.loads((perf.output / "summary.json").read_text())
        self.assertEqual(len(result["round_results"]), 4)
        self.assertTrue(result["round_results"][0]["warmup"])
        self.assertGreater(result["median_input_tokens_per_request_wall_second"], 0)
        self.corpus.write_text(json.dumps({**CASE, "id": "changed"}) + "\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            bench.collect(self.args("wrong", "performance", accuracy_run=accuracy.output))

    def test_explicit_performance_assumption_remains_unqualified(self):
        args = self.args("explore", "performance", assume_accuracy_for_performance=True)
        bench.collect(args)
        result = json.loads((args.output / "summary.json").read_text())
        self.assertTrue(result["valid"])
        self.assertTrue(result["accuracy_assumed"])
        self.assertFalse(result["stage1_qualified"])
        self.assertFalse(result["whole_model_accuracy_pass"])
        self.assertGreater(result["median_input_tokens_per_request_wall_second"], 0)

    def test_measured_compilation_invalidates_performance_round(self):
        server_log = self.root / "server.log"
        server_log.write_text("")
        original = bench.collect_one
        def request(*args):
            result = original(*args)
            with server_log.open("a") as log:
                log.write("1 warning generated when compiling for ppu0015.\n")
            return result
        args = self.args("jit", "performance", assume_accuracy_for_performance=True,
                         server_log=server_log)
        with mock.patch.object(bench, "collect_one", side_effect=request):
            with self.assertRaisesRegex(RuntimeError, "invalid collection"):
                bench.collect(args)
        summary = json.loads((args.output / "summary.json").read_text())
        self.assertTrue(summary["round_results"][0]["valid"])
        self.assertFalse(summary["round_results"][1]["valid"])
        self.assertIsNone(summary["median_input_tokens_per_request_wall_second"])

    def test_accuracy_collection_rejects_assumption(self):
        with self.assertRaisesRegex(ValueError, "only valid for performance"):
            bench.collect(self.args("bad", assume_accuracy_for_performance=True))

    def test_invalid_warmup_retains_evidence_and_stops(self):
        self.payload["aux_info"]["output_len"] = 0
        args = self.args("invalid")
        with self.assertRaises(RuntimeError):
            bench.collect(args)
        result = json.loads((args.output / "summary.json").read_text())
        self.assertFalse(result["valid"])
        self.assertIsNone(result["median_input_tokens_per_request_wall_second"])
        self.assertEqual(len(result["round_results"]), 1)
        self.assertIn("exactly one", (args.output / "responses.jsonl").read_text())

    def test_nonfinite_raw_response_is_preserved(self):
        self.payload["logits"][0][0] = float("nan")
        args = self.args("nan")
        with self.assertRaises(RuntimeError):
            bench.collect(args)
        record = json.loads((args.output / "responses.jsonl").read_text())
        self.assertIn("NaN", record["raw_response"])
        self.assertFalse(record["ok"])

    def test_http_failure_cannot_pass(self):
        self.http_status = 500
        result = bench.collect_one(self.url, "rtp", CASE, False, 5)
        self.assertFalse(result["ok"])
        self.assertIn("500", result["error"])

    def test_existing_output_is_never_overwritten(self):
        args = self.args("existing")
        args.output.mkdir()
        with self.assertRaises(FileExistsError):
            bench.collect(args)

    def test_compare_rejects_different_checkpoint(self):
        rtp = self.args("rtp")
        bench.collect(rtp)
        self.payload = copy.deepcopy(SG)
        sg = self.args("sg", backend="sglang")
        bench.collect(sg)
        compare = argparse.Namespace(rtp=rtp.output, sglang=sg.output, output=self.root / "compare.json")
        bench.compare(compare)
        result = json.loads(compare.output.read_text())
        self.assertEqual(result["top1_matches"], 3)
        self.assertFalse(result["whole_model_accuracy_pass"])
        path = sg.output / "summary.json"
        changed = json.loads(path.read_text())
        changed["service"]["checkpoint_manifest_sha256"] = "different"
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            bench.compare(compare)


if __name__ == "__main__":
    unittest.main()
