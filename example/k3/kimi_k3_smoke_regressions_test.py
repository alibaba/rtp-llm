from __future__ import annotations

import ast
import copy
from collections import Counter
from contextlib import ExitStack
import gzip
import io
import json
import os
import pathlib
import re
import subprocess
import tempfile
import threading
import unittest
import urllib.error
from types import SimpleNamespace
from unittest import mock

from example.k3.kimi_k3_full_model_pd_cases import (
    Case,
    Runner,
    SmokeFailure,
    TransportFailure,
    numbered_answer_pattern,
)
from example.k3.kimi_k3_full_model_pd_cases_test import make_args
from example.k3.kimi_k3_smoke_runtime_evidence import collect, verify


def response_for(runner, content, *, owner=0, reuse=0, input_len=8193):
    return {
        "choices": [{"message": {"content": content}}],
        "aux_info": {
            "pd_sep": True,
            "input_len": input_len,
            "output_len": 12,
            "iter_count": 9,
            "prefill_total_reuse_len": reuse,
            "role_addrs": [runner.decode_role_addrs[owner]],
        },
        "debug_info": {"output_ids": [[1, 2, 3]]},
    }


class RegressionTest(unittest.TestCase):
    def test_keep_services_reaches_both_roles(self):
        from example.k3.kimi_k3_full_model_two_host_pd_smoke_driver import forwarded_optional_environment
        with mock.patch.dict(os.environ, {"SMOKE_KEEP_SERVICES": "1"}, clear=True):
            for role in ("prefill", "decode"):
                self.assertEqual(forwarded_optional_environment(role)["SMOKE_KEEP_SERVICES"], "1")

    def test_cleanup_retains_service_on_success_and_failure(self):
        script = pathlib.Path(__file__).with_name("kimi_k3_full_model_two_host_pd_smoke.sh").read_text()
        cleanup = re.search(r"(?ms)^cleanup\(\) \{\n.*?^\}", script).group(0)
        for keep in (0, 1):
            for status in (0, 1):
                with self.subTest(keep=keep, status=status), tempfile.TemporaryDirectory() as tmp:
                    command = (
                        'role=prefill; notified=0; service_pid=$$; listener_pid=listener; '
                        'checkpoint_real=checkpoint; role_dir=artifacts; summary_file="$1"; '
                        'smoke_keep_services="$2"; '
                        'notify_decode() { echo notified; }; '
                        'stop_owned_process() { echo stopped:"$1"; }; '
                        + cleanup + '\n(exit "$3"); cleanup'
                    )
                    result = subprocess.run(
                        ["bash", "-c", command, "test", tmp + "/summary", str(keep), str(status)],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, status, result.stderr)
                    self.assertIn("notified", result.stdout)
                    self.assertIn("stopped:listener", result.stdout)
                    self.assertEqual("RETAINED:" in result.stdout, bool(keep))
                    self.assertEqual(len(re.findall(r"stopped:[0-9]+", result.stdout)), 0 if keep else 1)
                    self.assertIn(f"status={status}", pathlib.Path(tmp + "/summary").read_text())

    def test_profile_admits_actual_concurrent_smoke_batches(self):
        from rtp_llm.utils.concurrency_controller import ConcurrencyController

        runner = Runner(make_args())
        with ExitStack() as patches:
            for method in (
                "prewarm_rdma_pool", "run_owner_regressions", "run_prefix_branches",
                "run_padding_boundaries", "run_long_prefix_case",
            ):
                patches.enter_context(mock.patch.object(runner, method))
            stages = patches.enter_context(mock.patch.object(runner, "run_stage"))
            runner.run_all()
        batches = [
            (call.args[0], call.args[1]) for call in stages.call_args_list
            if call.kwargs.get("concurrent", False)
        ]
        self.assertTrue(any(name == "dp_uneven_local_batch" for name, _ in batches))
        profile = pathlib.Path(__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        ).read_text()
        limits = {}
        for role in ("prefill", "decode"):
            exports = []
            for section in ("common", role):
                body = re.search(
                    rf"(?ms)^apply_validated_{section}_profile\(\) \{{\n(.*?)^\}}",
                    profile,
                ).group(1)
                exports.extend(re.findall(r"export CONCURRENCY_LIMIT=(\d+)", body))
            self.assertTrue(exports, role)
            limits[role] = int(exports[-1])
        for name, cases in batches:
            with self.subTest(stage=name):
                # Hold every HTTP slot until the whole submitted batch is admitted.
                controller = ConcurrencyController(limits["prefill"])
                with ExitStack() as active:
                    for _ in cases:
                        active.enter_context(controller)
                    self.assertEqual(controller.get_request_counter(), len(cases))
                self.assertEqual(controller.get_available_concurrency(), limits["prefill"])
                owners = Counter(case.decode_owner_rank for case in cases)
                self.assertLessEqual(max(owners.values()), limits["decode"])

    def test_long_record_budget_includes_reasoning_and_final_json(self):
        runner = Runner(make_args())
        short = runner.record_case("short", 0)
        long = runner.record_case("rolling-6", 6, words=16)
        self.assertEqual(len(long.expected_json["value"].split()), 16)
        self.assertGreaterEqual(long.max_tokens, 1024)
        self.assertGreater(long.max_tokens, short.max_tokens)

    def test_math_answer_cannot_contain_sibling_answer(self):
        pattern = numbered_answer_pattern(6561)
        self.assertIsNotNone(re.search(pattern, " \n6561\n"))
        for text in ("6400 6561", "6561 6724", "答案6561", "16561", "6561.0"):
            self.assertIsNone(re.search(pattern, text), text)

    def test_json_answer_rejects_extra_or_duplicate_fields(self):
        runner = Runner(make_args())
        case = Case("record", "prompt", "", "miss", expected_json={"value": "CEDAR"})
        runner.validate(case, response_for(runner, '{"value":"CEDAR"}'), 0, 128)
        for text in (
            '{"value":"MAPLE"}',
            '{"value":"CEDAR","other":"MAPLE"}',
            '{"value":"MAPLE","value":"CEDAR"}',
            '```json\n{"value":"CEDAR"}\n```',
            "null",
            "[]",
        ):
            with self.subTest(text=text), self.assertRaises(SmokeFailure):
                runner.validate(case, response_for(runner, text), 0, 128)

    def test_reuse_and_token_length_are_exact(self):
        runner = Runner(make_args())
        case = Case(
            "frontier",
            "prompt",
            numbered_answer_pattern(1),
            "hit",
            expected_input_len=8193,
            expected_reuse_len=8192,
        )
        runner.validate(case, response_for(runner, "1", reuse=8192), 0, 128)
        for reuse, length in ((4096, 8193), (8192, 8194)):
            with self.assertRaises(SmokeFailure):
                runner.validate(
                    case,
                    response_for(runner, "1", reuse=reuse, input_len=length),
                    0,
                    128,
                )

    def test_semantic_prewarm_failure_is_never_retried(self):
        args = make_args()
        args.rdma_prewarm_attempts = 3
        runner = Runner(args)
        with mock.patch.object(runner, "health"), mock.patch.object(
            runner, "request_cases", side_effect=SmokeFailure("wrong owner answer")
        ) as request:
            with self.assertRaisesRegex(SmokeFailure, "wrong owner"):
                runner.prewarm_rdma_pool()
        self.assertEqual(request.call_count, 1)
        self.assertFalse(runner.rdma_prewarm_attempts[0]["passed"])

    def test_batch_transport_error_cannot_hide_semantic_failure(self):
        runner = Runner(make_args())
        cases = [runner.record_case("connection", 0), runner.record_case("answer", 1)]

        def request(case, barrier):
            barrier.wait(timeout=2)
            if case.name == "connection":
                raise TransportFailure("connect")
            raise SmokeFailure("wrong answer")

        with mock.patch.object(runner, "request", side_effect=request):
            with self.assertRaisesRegex(SmokeFailure, "wrong answer"):
                runner.request_cases(cases, True)

    def test_failed_response_and_successful_sibling_are_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args()
            args.output = pathlib.Path(tmp) / "accuracy.json"
            runner = Runner(args)
            cases = [runner.record_case("good", 0), runner.record_case("bad", 0)]

            def open_request(request, **kwargs):
                payload = json.loads(request.data)
                good = "/good。" in payload["messages"][0]["content"]
                result = mock.MagicMock()
                result.__enter__.return_value = result
                result.status = 200
                result.read.return_value = json.dumps(
                    response_for(
                        runner,
                        (
                            json.dumps(cases[0].expected_json)
                            if good
                            else '{"value":"WRONG"}'
                        ),
                    )
                ).encode()
                return result

            runner.opener.open = mock.Mock(side_effect=open_request)
            with mock.patch.object(runner, "health"):
                with self.assertRaises(SmokeFailure):
                    runner.run_stage("siblings", cases, concurrent=True)
            runner.save(False)
            self.assertEqual([r["name"] for r in runner.records], ["good"])
            saved = [
                json.loads(p.read_text())
                for p in (pathlib.Path(tmp) / "requests").glob("*.json")
            ]
            self.assertEqual(len(saved), 2)
            self.assertTrue(
                all("response_body" in item and "request" in item for item in saved)
            )
            self.assertEqual(sum(item["passed"] for item in saved), 1)
            self.assertFalse(runner.stages[-1]["passed"])
            self.assertEqual(runner.failures[0]["name"], "bad")

    def test_http_client_error_is_not_retryable(self):
        runner = Runner(make_args())
        runner.opener.open = mock.Mock(
            side_effect=urllib.error.HTTPError(
                "http://test", 400, "bad request", {}, io.BytesIO(b"bad request")
            )
        )
        with self.assertRaises(SmokeFailure) as error:
            runner.request(runner.record_case("bad", 0))
        self.assertNotIsInstance(error.exception, TransportFailure)

    def test_owner_cases_cover_all_ranks_rotation_and_slot_waves(self):
        runner = Runner(make_args())
        stages = {}
        with mock.patch.object(
            runner, "run_stage", side_effect=lambda n, c, **kw: stages.update({n: c})
        ):
            runner.run_owner_regressions()
        for rotation in (0, 3):
            cases = stages[f"owner_records_rotate_{rotation}"]
            self.assertEqual(
                [c.decode_owner_rank for c in cases],
                [(i + rotation) % 8 for i in range(8)],
            )
            self.assertEqual(len({json.dumps(c.expected_json) for c in cases}), 8)
        self.assertEqual(stages["owner_last_only"][0].decode_owner_rank, 7)
        self.assertEqual(
            [len(stages[f"graph_slot_wave_{i}"]) for i in range(3)], [7, 5, 6]
        )
        self.assertEqual(
            [c.decode_owner_rank for c in stages["historical_four_squares"]],
            [0, 0, 1, 2],
        )

    def test_refill_submits_new_request_before_slow_sibling_finishes(self):
        runner = Runner(make_args())
        cases = [runner.record_case(str(i), 0) for i in range(4)]
        refill_arrived = threading.Event()
        observed = []

        def request(case):
            observed.append(case.name)
            if case.name == "0":
                if not refill_arrived.wait(3):
                    raise AssertionError("refill waited for drained batch")
            if case.name == "2":
                refill_arrived.set()
            return {}

        with mock.patch.object(runner, "request", side_effect=request):
            runner.request_refill(cases, window=2)
        self.assertEqual(set(observed), {"0", "1", "2", "3"})

    def test_exact_padding_inputs_use_serving_tokenizer(self):
        runner = Runner(make_args())
        stages = {}
        with mock.patch.object(
            runner, "fit_prompt", side_effect=lambda h, t, n: (h + t, list(range(n)))
        ) as fit, mock.patch.object(
            runner, "run_stage", side_effect=lambda n, c, **kw: stages.update({n: c})
        ):
            runner.run_padding_boundaries()
        self.assertEqual([call.args[2] for call in fit.call_args_list], [65537, 65543])
        for tail in (1, 7):
            cold = stages[f"padding_tail_{tail}_cold"][0]
            hit = stages[f"padding_tail_{tail}_hit"][0]
            self.assertEqual(cold.expected_input_len, 65536 + tail)
            self.assertEqual(hit.expected_reuse_len, 65536)
            self.assertNotEqual(cold.decode_owner_rank, hit.decode_owner_rank)

    def test_token_fixture_saves_exact_chat_input_and_token_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args()
            args.output = pathlib.Path(tmp) / "accuracy.json"
            runner = Runner(args)
            with mock.patch.object(
                runner,
                "tokenize",
                side_effect=lambda p: [13] * (11 + p.count(" x") * 2),
            ):
                prompt, tokens = runner.fit_prompt("head", "tail", 31)
            files = list((pathlib.Path(tmp) / "token-fixtures").glob("*.json.gz"))
            self.assertEqual(len(files), 1)
            with gzip.open(files[0], "rt") as source:
                fixture = json.load(source)
            self.assertEqual(fixture["input_ids"], tokens)
            self.assertEqual(fixture["messages"], [{"role": "user", "content": prompt}])

    def test_prompt_fitting_does_not_assume_character_count(self):
        runner = Runner(make_args())
        with mock.patch.object(
            runner, "tokenize", side_effect=lambda p: [1] * (11 + p.count(" x") * 2)
        ):
            prompt, tokens = runner.fit_prompt("head", "tail", 31)
        self.assertEqual(len(tokens), 31)
        self.assertEqual(prompt.count(" x"), 10)

    def test_prefix_A_B_A_and_parallel_mix_have_different_expected_answers(self):
        runner = Runner(make_args())
        stages = {}
        with mock.patch.object(
            runner, "fit_prompt", side_effect=lambda h, t, n: (h + t, [1] * n)
        ), mock.patch.object(
            runner, "tokenize", return_value=[1] * 8192 + [2] * 128
        ), mock.patch.object(
            runner, "run_stage", side_effect=lambda n, c, **kw: stages.update({n: c})
        ):
            runner.run_prefix_branches()
        self.assertEqual(
            list(stages),
            ["prefix_A_seed", "prefix_B_partial", "prefix_A_return", "prefix_AB_mixed"],
        )
        self.assertNotEqual(
            stages["prefix_A_seed"][0].expected_json,
            stages["prefix_B_partial"][0].expected_json,
        )
        self.assertEqual(stages["prefix_B_partial"][0].expected_reuse_len, 8192)
        self.assertEqual(
            [c.decode_owner_rank for c in stages["prefix_AB_mixed"]], [4, 5, 6, 7]
        )


class RuntimeEvidenceTest(unittest.TestCase):
    def events(self):
        events = []
        for step, (valid, bucket) in enumerate(
            [
                ([1] * 8, 1),
                ([0] * 7 + [2], 2),
                ([0] * 7 + [3], 4),
                ([0] * 7 + [7], 8),
                ([0] * 7 + [5], 8),
                ([0] * 7 + [6], 8),
            ]
        ):
            for rank in range(8):
                events.append(
                    dict(
                        kind="ktp",
                        rank=rank,
                        step=step,
                        valid=valid,
                        bucket=bucket,
                        physical=bucket,
                        graph=True,
                        mode="TARGET_VERIFY",
                        tokens=4,
                    )
                )
        return events

    def test_real_runtime_logger_is_opt_in_and_numbers_steps(self):
        source = (
            pathlib.Path(__file__).resolve().parents[2]
            / "rtp_llm/models_py/modules/kimi_k3/ktp_step.py"
        )
        tree = ast.parse(source.read_text())
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_log_step_plan_once"
        )
        logger = mock.Mock()
        rank = mock.Mock(return_value=7)
        namespace = {
            "os": os,
            "json": json,
            "logger": logger,
            "_SMOKE_STEP": 0,
            "_LOGGED_STEP_PLANS": set(),
            "KtpStepPlan": object,
            "torch": SimpleNamespace(distributed=SimpleNamespace(get_rank=rank)),
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        plan = SimpleNamespace(
            valid_batch_sizes=(0, 0, 0, 0, 0, 0, 0, 7),
            common_physical_batch=8,
            common_graph_bucket=8,
            use_cuda_graph=True,
            forward_mode=SimpleNamespace(name="TARGET_VERIFY"),
            tokens_per_batch=4,
        )
        # A real IntEnum is hashable, as required by the existing dedup cache.
        from enum import IntEnum

        class Mode(IntEnum):
            TARGET_VERIFY = 2

        plan.forward_mode = Mode.TARGET_VERIFY
        with mock.patch.dict(os.environ, {"KIMI_K3_SMOKE_EVIDENCE": "0"}):
            namespace["_log_step_plan_once"](plan)
        rank.assert_not_called()
        with mock.patch.dict(os.environ, {"KIMI_K3_SMOKE_EVIDENCE": "1"}):
            namespace["_log_step_plan_once"](plan)
            idle = copy.copy(plan)
            idle.valid_batch_sizes = (0,) * 8
            namespace["_log_step_plan_once"](idle)
            namespace["_log_step_plan_once"](plan)
        events = [
            json.loads(c.args[1])
            for c in logger.info.call_args_list
            if c.args[0] == "[K3_SMOKE_EVENT] %s"
        ]
        self.assertEqual([e["step"] for e in events], [0, 1])
        self.assertEqual([e["rank"] for e in events], [7, 7])

    def test_complete_decode_evidence_passes_and_mirrors_are_deduplicated(self):
        events = self.events()
        self.assertTrue(verify(events + events, "decode", True)["passed"])

    def test_owner7_bucket8_does_not_require_instantaneous_wave_sizes(self):
        events = copy.deepcopy(self.events())
        for event in events:
            if event["step"] in (3, 4, 5):
                event["valid"][-1] = {3: 6, 4: 5, 5: 6}[event["step"]]
        report = verify(events, "decode", True)
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["observations"]["bucket8_owner7_valid_transitions"],
            [6, 5, 6],
        )
        self.assertNotIn("bucket8_slot_reuse_7_5_6", report["checks"])

    def test_missing_rank_or_disagreed_plan_or_fake_graph_fails(self):
        events = self.events()
        variants = [events[:-1], [dict(e, graph=False) for e in events]]
        bad = copy.deepcopy(events)
        bad[1]["bucket"] = 8
        variants.append(bad)
        for data in variants:
            self.assertFalse(verify(data, "decode", True)["passed"])
        self.assertFalse(verify(events, "decode", False)["passed"])

    def test_bucket8_on_another_owner_does_not_cover_owner7(self):
        events = copy.deepcopy(self.events())
        for event in events:
            if event["step"] in (3, 4, 5):
                event["valid"] = [event["valid"][-1]] + [0] * 7
        report = verify(events, "decode", True)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["bucket8_owner7_target_verify"])

    def test_prefill_requires_actual_padding_both_boundaries(self):
        events = [
            dict(
                kind="chunk",
                tp=8,
                logical_tokens=n,
                physical_tokens=8,
                logical_requests=1,
                physical_requests=2,
            )
            for n in (1, 7)
        ]
        self.assertTrue(verify(events, "prefill")["passed"])
        self.assertFalse(verify(events[:1], "prefill")["passed"])
        events[0]["physical_tokens"] = 9
        self.assertFalse(verify(events, "prefill")["passed"])

    def test_collect_reads_only_current_role_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            event = self.events()[0]
            (root / "service.log").write_text(
                "prefix [K3_SMOKE_EVENT] "
                + json.dumps(event)
                + "\n[K3_PROJECTION_KTP_GRAPH_REPLAY] test\n"
            )
            (root / "old-evidence.json").write_text("[K3_SMOKE_EVENT] invalid")
            events, replay = collect(root)
            self.assertEqual(events, [event])
            self.assertTrue(replay)


if __name__ == "__main__":
    unittest.main()
