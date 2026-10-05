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
from example.k3.kimi_k3_smoke_runtime_evidence import (
    GRAPH_BUCKETS,
    physical_graph_buckets,
    collect,
    verify,
)
from example.k3.mla_request_trace import verify_mla_traces


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
    def test_profile_readiness_consumes_real_k3_stream_frames(self):
        import asyncio

        import torch

        from rtp_llm.config.generate_config import GenerateConfig
        from rtp_llm.openai.api_datatype import (
            ChatCompletionResponseStreamChoice,
            ChatCompletionStreamResponse,
            DeltaMessage,
        )
        from rtp_llm.openai.renderers.custom_renderer import StreamResponseObject
        from rtp_llm.openai.renderers.kimi_k3_renderer import KimiK3Renderer
        from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput

        renderer = object.__new__(KimiK3Renderer)
        config = GenerateConfig(return_output_ids=True)
        frames = []
        for token, text in ((1, "P"), (2, "D")):
            extra = asyncio.run(renderer._generate_extra_outputs(
                GenerateOutput(output_ids=torch.tensor([[token]])), config,
            ))
            response = StreamResponseObject(
                choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(content=text))],
                extra_outputs=extra,
            )
            frames.extend(renderer._format_stream_frames(response, False))
        self.assertTrue(all(frame.aux_info is None for frame in frames))
        frames.extend(renderer._format_stream_frames(StreamResponseObject(
            choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(), finish_reason="stop")],
            aux_info=AuxInfo(output_len=2, iter_count=2, pd_sep=True),
        ), False))
        wire = b"".join(
            b"data: " + ChatCompletionStreamResponse(
                choices=frame.choices, aux_info=frame.aux_info, extra_outputs=frame.extra_outputs,
            ).model_dump_json(exclude_none=True).encode() + b"\n\n"
            for frame in frames
        ) + b"data: [DONE]\n"
        audit = {}
        ready = mock.Mock()
        result = Runner.read_profile_stream(io.BytesIO(wire), ready, audit)
        ready.assert_called_once_with()
        self.assertEqual(audit["decode_ready_output_ids"], [1, 2])
        self.assertEqual(result["choices"][0]["message"]["content"], "PD")
        self.assertEqual(result["debug_info"]["output_ids"], [[1, 2]])

    def test_checkpoint_storage_accepts_3fs_and_local_disks(self):
        script = pathlib.Path(__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        ).read_text()
        for prefix in ("checkpoint", "sp_checkpoint"):
            start = script.index(f'case "${{{prefix}_real}}" in')
            end = script.index(f'[[ -f "${{{prefix}_real}}/config.json" ]]', start)
            guard = script[start:end]
            for path, filesystem, source, accepted in (
                ("/data0/model", "ext4", "/dev/nvme0n1", True),
                ("/mnt/hf3fs/3fs/models/kimi", "fuse.hf3fs", "hf3fs.cluster", True),
                ("/data0/model", "nfs4", "server:/models", False),
                ("/mnt/hf3fs/3fs/models/kimi", "fuse.sshfs", "server:/models", False),
            ):
                with self.subTest(prefix=prefix, path=path, filesystem=filesystem):
                    result = subprocess.run(
                        ["bash", "-c",
                         'set -e; die() { echo "$*" >&2; exit 2; }; '
                         f'{prefix}_real="$1"; '
                         'storage_fs="$2"; storage_source="$3"; '
                         'findmnt() { if [[ "$5" == FSTYPE ]]; then echo "$storage_fs"; '
                         'else echo "$storage_source"; fi; }; '
                         + guard,
                         "storage-test", path, filesystem, source],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode == 0, accepted, result.stderr)

    def test_page_rr_multi_launch_profile_requires_2k_physical_pages(self):
        script = (
            pathlib.Path(__file__)
            .with_name("kimi_k3_full_model_two_host_pd_smoke.sh")
            .read_text()
        )
        profile = re.search(
            r'if \[\[ "\$\{smoke_prefill_page_rr_multi_launch\}" == "1" \]\]; then(.*?)fi',
            script,
            re.S,
        ).group(1)
        self.assertIn('"2048:128"', profile)

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

    def test_flow_covers_chunk_and_both_mixed_dp_owners(self):
        args = make_args()
        args.suite = "flow"
        args.decode_role_addrs = args.decode_role_addrs[:2]
        runner = Runner(args)
        tokens = list(range(args.chunk_tokens + 1))
        with mock.patch.object(runner, "fit_prompt", return_value=("chunk", tokens)) as fit, \
             mock.patch.object(runner, "run_stage") as stage:
            runner.run_flow()
        self.assertEqual(fit.call_args_list[0].args[2], args.chunk_tokens + 1)
        self.assertEqual(
            [call.args[2] for call in fit.call_args_list[1:]],
            [runner.reuse_unit_tokens + args.block_size * (idx + 1) for idx in range(5)],
        )
        calls = stage.call_args_list
        self.assertEqual(calls[0].args[1][0].expected_input_len, len(tokens))
        self.assertEqual([c.args[1][0].decode_owner_rank for c in calls[1:3]], [0, 1])
        self.assertEqual(Counter(c.decode_owner_rank for c in calls[3].args[1]), {0: 4, 1: 1})
        self.assertEqual(Counter(c.decode_owner_rank for c in calls[4].args[1]), {0: 1, 1: 4})
        self.assertTrue(calls[3].kwargs["concurrent"])
        self.assertTrue(all(c.reuse == "hit" for c in calls[4].args[1]))

    def test_profile_admits_actual_concurrent_smoke_batches(self):
        from rtp_llm.utils.concurrency_controller import ConcurrencyController

        runner = Runner(make_args())
        with ExitStack() as patches:
            patches.enter_context(mock.patch.object(
                runner, "fit_prompt",
                side_effect=lambda head, tail, target: (head + tail, [0] * target),
            ))
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
                exports.extend(re.findall(r"export CONCURRENCY_LIMIT=(.+)", body))
            self.assertTrue(exports, role)
            value = exports[-1].strip('"')
            variable = re.fullmatch(r"\$\{(\w+)\}", value)
            if variable:
                value = re.search(
                    rf'(?m)^{variable[1]}="\$\{{[^:]+:-(\d+)\}}"$', profile,
                )[1]
            limits[role] = int(value)
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

    def test_owner_cases_preserve_dp_rotation_without_ktp_slot_waves(self):
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
        self.assertFalse(any(name.startswith("graph_slot_wave_") for name in stages))
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

    def test_page_rr_batch_hit_prompts_cross_the_full_shard_span(self):
        args = make_args()
        args.block_size = 2048
        args.reuse_unit_tokens = 2048 * 8
        runner = Runner(args)
        stages = {}

        def fit_prompt(head, tail, target):
            return f"{head}<target={target}>{tail}", [1] * target

        with ExitStack() as patches:
            for method in (
                "prewarm_rdma_pool",
                "run_owner_regressions",
                "run_prefix_branches",
                "run_padding_boundaries",
                "run_page_rr_boundaries",
                "run_long_prefix_case",
            ):
                patches.enter_context(mock.patch.object(runner, method))
            patches.enter_context(
                mock.patch.object(runner, "fit_prompt", side_effect=fit_prompt)
            )
            patches.enter_context(
                mock.patch.object(runner, "tokenize", return_value=[1] * 22528)
            )
            patches.enter_context(
                mock.patch.object(
                    runner,
                    "run_stage",
                    side_effect=lambda name, cases, **kw: stages.update({name: cases}),
                )
            )
            runner.run_all()

        expected_targets = [18432, 20480, 22528, 24576]
        for stage_name in ("batch_all_miss", "batch_all_hit"):
            self.assertEqual(
                [
                    int(re.search(r"<target=(\d+)>", case.prompt).group(1))
                    for case in stages[stage_name]
                ],
                expected_targets,
            )
        self.assertEqual(
            [case.prompt for case in stages["batch_all_miss"]],
            [case.prompt for case in stages["batch_all_hit"]],
        )
        self.assertTrue(
            all(
                f"<target={expected_targets[index]}>" in case.prompt
                for index, case in enumerate(stages["batch_mixed_then_all_hit"])
            )
        )

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

    def test_page_rr_prefill_requires_a_real_tp8_multi_launch_plan(self):
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
        self.assertFalse(
            verify(events, "prefill", prefill_page_rr=True)["passed"]
        )
        events.append(
            dict(
                kind="mla_prefix",
                backend="page_rr",
                route="hybrid",
                tp=8,
                prefix_tokens=950_272,
                query_tokens=1_587,
                capacity_tokens=559_104,
                alignment_tokens=2_048,
                launch_tokens=[559_104, 391_168],
            )
        )
        report = verify(events, "prefill", prefill_page_rr=True)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["observations"]["page_rr_max_launch_count"], 2)

    def test_page_rr_prefill_rejects_invalid_multi_launch_evidence(self):
        base = [
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
        plan = dict(
            kind="mla_prefix",
            backend="page_rr",
            route="hybrid",
            tp=8,
            prefix_tokens=4096,
            query_tokens=1,
            capacity_tokens=2048,
            alignment_tokens=2048,
            launch_tokens=[2048, 2048],
        )
        for broken in (
            dict(plan, backend="replicated"),
            dict(plan, tp=4),
            dict(plan, route="full"),
            dict(plan, launch_tokens=[2048]),
            dict(plan, launch_tokens=[1152, 896]),
        ):
            with self.subTest(broken=broken):
                self.assertFalse(
                    verify(base + [broken], "prefill", prefill_page_rr=True)[
                        "passed"
                    ]
                )

    def test_replicated_prefill_does_not_require_page_rr_plan_evidence(self):
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
        self.assertTrue(verify(events, "prefill", prefill_page_rr=False)["passed"])

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
            events, replay, _ = collect(root)
            self.assertEqual(events, [event])
            self.assertTrue(replay)

    def dcp_log(self, ranks=range(8), buckets=(2, 4, 8), tp=8):
        lines = [f"[MLA_DCP] backend=a2a tp={tp} rank={rank}" for rank in ranks]
        lines.append(f"[K3_PAGE_RR_TARGET] role=Decode TP={tp} B=128 V=64")
        for bucket in buckets:
            for rank in range(tp):
                lines.append(
                    f"[INFO] [RANK {rank}][10.0.0.1][cuda_graph_runner.cc:1095] "
                    f"[CudaGraph Memory] captured batch size {bucket}: pool_delta=46 MiB"
                )
        return "\n".join(lines) + "\n"

    def test_physical_graph_buckets_match_tp_token_alignment(self):
        self.assertEqual(physical_graph_buckets(8, 3), (2, 4, 8))
        self.assertEqual(physical_graph_buckets(8, 1), (4, 8))
        self.assertEqual(physical_graph_buckets(8, 7), (1, 2, 4, 8))
        self.assertEqual(physical_graph_buckets(1, 3), (1, 2, 4, 8))
        self.assertEqual(physical_graph_buckets(4, 3, (1, 2, 4, 8, 16)), (1, 2, 4, 8, 16))

    def test_mla_request_evidence_requires_correlated_work_in_both_groups(self):
        for selector, qrep in ((s, q) for s in ("AUTO", "NCCL", "CUSTOM") for q in (False, True)):
            with self.subTest(a2a_backend=selector, qrep=qrep), tempfile.TemporaryDirectory() as tmp:
                directory = pathlib.Path(tmp)
                plans = {}
                def branch_names(batch, queries, splits):
                    if splits == 1:
                        return ["_merge_splits_serial"]
                    # Fixture H96/TP4: small target/draft T4 retains NCCL; draft T16
                    # reaches AUTO's custom range. Explicit modes override it.
                    custom = selector == "CUSTOM" or (selector == "AUTO" and batch * queries >= 8)
                    exchange = (["kernel_PeerPullMerge"] if custom
                                else ["_pack_a2a", "ncclDevKernel_SendRecv", "_combine_a2a"])
                    return ["merge_local_splits", *exchange]
                for stage, batch, target_splits in (("small", 1, 24), ("mid", 4, 6), ("large", 16, 1)):
                    for rank in range(8):
                        plans[rank, batch, 4, "torch.float8_e4m3fn"] = dict(TP=4, H=96, S=target_splits, mode="fused" if target_splits == 1 else "unfused",
                                                   q_layout="token_major" if qrep else "head_major")
                        prefix = [] if qrep else ["ncclDevKernel_AllGather", "_pack"]
                        names = prefix + ["kernel_cutlass_split_kv_kernel_PageRRFusedMLAFP8"]
                        names += branch_names(batch, 4, target_splits)
                        # Two attention layers, plus unrelated TP AllGather.
                        names = ["ncclDevKernel_AllGather"] + names * 2
                        events = [dict(ph="X", cat="cpu_op", pid=1, tid=1, ts=10, dur=10,
                                       name=f"cuda_graph.forward(replayDecode,B={batch},capture={batch},Q=4,T={batch*4},fake=0)"),
                                  dict(ph="X", cat="cuda_runtime", pid=1, tid=1, ts=12, dur=1,
                                       name="cudaGraphLaunch", args={"correlation": 7})]
                        # GPU activity is deliberately later than the CPU scope.
                        events += [dict(ph="X", cat="kernel", pid=0, tid=8, ts=100+i, dur=1,
                                        name=name, args={"correlation": 7}) for i, name in enumerate(names)]
                        for kind, queries, capture, correlation, splits in (() if stage == "mid" else (
                            ("replayDecode", 1, max(4, batch), 17, 18 if batch == 1 else 4),
                            ("replayPrefill", 4, 16, 27, 1),
                        )):
                            plans[rank, capture, queries, "torch.bfloat16"] = dict(
                                TP=4, H=96, S=splits, mode="fused" if splits == 1 else "unfused",
                                q_layout="token_major" if qrep else "head_major",
                            )
                            draft_names = prefix + ["kernel_cutlass_split_kv_kernel_PageRRFusedMLABF16"]
                            draft_names += branch_names(capture, queries, splits)
                            events += [dict(ph="X", cat="cpu_op", pid=1, tid=1, ts=correlation*10, dur=10,
                                            name=f"cuda_graph.forward({kind},B={batch},capture={capture},Q={queries},T={batch*queries},fake=0)"),
                                       dict(ph="X", cat="cuda_runtime", pid=1, tid=1, ts=correlation*10+2, dur=1,
                                            name="cudaGraphLaunch", args={"correlation": correlation})]
                            events += [dict(ph="X", cat="kernel", pid=0, tid=8, ts=1000+correlation*10+i, dur=1,
                                            name=name, args={"correlation": correlation}) for i, name in enumerate(draft_names)]
                        (directory / f"mla_{stage}_owner{rank//4}_wr{rank}_0.json").write_text(
                            json.dumps({"traceEvents": events}))
                self.assertTrue(verify_mla_traces(directory, 4, 2, "FIA2A", plans, a2a_backend=selector, q_replicated=qrep)["passed"])
                missing = directory / "mla_large_owner1_wr7_0.json"
                original = missing.read_text()
                full = verify_mla_traces(directory, 4, 2, "FIA2A", plans, require_draft=True, a2a_backend=selector, q_replicated=qrep)
                self.assertEqual(len(full["observations"]), 56)
                wrong_selector = "NCCL" if selector == "CUSTOM" else "CUSTOM"
                with self.assertRaisesRegex(ValueError, "GPU A2A transport"):
                    verify_mla_traces(directory, 4, 2, "FIA2A", plans,
                                      require_draft=True, a2a_backend=wrong_selector, q_replicated=qrep)
                small = directory / "mla_small_owner1_wr7_0.json"
                small_original = small.read_text()
                # A valid small proposal in the large window cannot substitute
                # for the native B16 proposal branch required by that stage.
                def proposal_event(event):
                    return event.get("args", {}).get("correlation") == 17 or (
                        event.get("name", "").startswith("cuda_graph.forward(replayDecode,")
                        and ",Q=1," in event["name"]
                    )
                wrong_bucket = json.loads(original)
                wrong_bucket["traceEvents"] = [e for e in wrong_bucket["traceEvents"] if not proposal_event(e)]
                wrong_bucket["traceEvents"] += [e for e in json.loads(small_original)["traceEvents"] if proposal_event(e)]
                missing.write_text(json.dumps(wrong_bucket))
                with self.assertRaisesRegex(ValueError, "native MTP proposal"):
                    verify_mla_traces(directory, 4, 2, "FIA2A", plans,
                                      require_draft=True, a2a_backend=selector, q_replicated=qrep)
                missing.write_text(original)
                # A complete unfused Q4 replay is still insufficient to prove
                # the MTP update's required fused path.
                wrong_update = json.loads(original)
                wrong_update["traceEvents"] = [e for e in wrong_update["traceEvents"]
                    if e.get("args", {}).get("correlation") != 27
                    and not e.get("name", "").startswith("cuda_graph.forward(replayPrefill,")]
                for event in json.loads(small_original)["traceEvents"]:
                    if event.get("args", {}).get("correlation") == 7 or (
                        event.get("name", "").startswith("cuda_graph.forward(replayDecode,")
                        and ",Q=4," in event["name"]
                    ):
                        event["ts"] += 4000
                        event["name"] = event["name"].replace("replayDecode", "replayPrefill").replace(
                            "PageRRFusedMLAFP8", "PageRRFusedMLABF16")
                        if "args" in event:
                            event["args"]["correlation"] = 27
                        wrong_update["traceEvents"].append(event)
                plans[7, 1, 4, "torch.bfloat16"] = plans[7, 1, 4, "torch.float8_e4m3fn"]
                missing.write_text(json.dumps(wrong_update))
                with self.assertRaisesRegex(ValueError, "native MTP update"):
                    verify_mla_traces(directory, 4, 2, "FIA2A", plans,
                                      require_draft=True, a2a_backend=selector, q_replicated=qrep)
                del plans[7, 1, 4, "torch.bfloat16"]
                missing.write_text(original)
                # A missing PullMerge or unrelated AllGather cannot witness A2A.
                bad = small_original.replace("PeerPullMerge", "missing_pull_merge") if selector == "CUSTOM" else (
                    small_original.replace("ncclDevKernel_SendRecv", "ncclDevKernel_AllGather")
                )
                small.write_text(bad)
                with self.assertRaisesRegex(ValueError, "complete GPU A2A transport"):
                    verify_mla_traces(directory, 4, 2, "FIA2A", plans, a2a_backend=selector, q_replicated=qrep)
                small.write_text(small_original)
                small.write_text(small_original.replace("merge_local_splits", "missing_merge", 1))
                with self.assertRaisesRegex(ValueError, "no local merge"):
                    verify_mla_traces(directory, 4, 2, "FIA2A", plans,
                                      a2a_backend=selector, q_replicated=qrep)
                small.write_text(small_original)
                with self.assertRaisesRegex(ValueError, "Q layout"):
                    verify_mla_traces(directory, 4, 2, "FIA2A", plans,
                                      a2a_backend=selector, q_replicated=not qrep)
                if not qrep:
                    small.write_text(small_original.replace('"_pack"', '"missing_q_pack"', 1))
                    with self.assertRaisesRegex(ValueError, "Q reorder"):
                        verify_mla_traces(directory, 4, 2, "FIA2A", plans,
                                          a2a_backend=selector, q_replicated=qrep)
                    small.write_text(small_original)
                # A CPU launch without recorded GPU work cannot be a witness, but
                # must not discard the complete target/proposal/update witnesses.
                partial = json.loads(original)
                partial["traceEvents"] += [
                    dict(ph="X", cat="cpu_op", pid=1, tid=1, ts=5000, dur=10,
                         name="cuda_graph.forward(replayPrefill,B=16,capture=16,Q=4,T=64,fake=0)"),
                    dict(ph="X", cat="cuda_runtime", pid=1, tid=1, ts=5002, dur=1,
                         name="cudaGraphLaunch", args={"correlation": 37}),
                ]
                missing.write_text(json.dumps(partial))
                retained = verify_mla_traces(directory, 4, 2, "FIA2A", plans, require_draft=True, a2a_backend=selector, q_replicated=qrep)
                self.assertEqual(len(retained["observations"]), 56)
                self.assertEqual([row["correlation"] for row in retained["incomplete_replays"]], [37])
                missing.write_text(original.replace("PageRRFusedMLABF16", "missing_draft_kernel"))
                with self.assertRaisesRegex(ValueError, "native MTP proposal"):
                    verify_mla_traces(directory, 4, 2, "FIA2A", plans, require_draft=True, a2a_backend=selector, q_replicated=qrep)
                missing.write_text(original.replace("fake=0", "fake=1"))
                with self.assertRaisesRegex(ValueError, "no Q4 target"):
                    verify_mla_traces(directory, 4, 2, "FIA2A", plans, a2a_backend=selector, q_replicated=qrep)
                data = json.loads(original)
                data["traceEvents"] = [event for event in data["traceEvents"] if event["cat"] != "kernel"]
                missing.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "no GPU kernels"):
                    verify_mla_traces(directory, 4, 2, "FIA2A", plans, a2a_backend=selector, q_replicated=qrep)
                missing.unlink()
                with self.assertRaisesRegex(ValueError, "missing="):
                    verify_mla_traces(directory, 4, 2, "FIA2A", plans, a2a_backend=selector, q_replicated=qrep)

    def test_dcp_decode_evidence_needs_every_rank_and_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "engine.log").write_text(self.dcp_log())
            events, replay, markers = collect(root)
            report = verify(events, "decode", replay, markers, True)
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["observations"]["dcp_ranks"], list(range(8)))
            # The DCP round must not be reportable through the KTP plan branch.
            self.assertFalse(verify(events, "decode", replay, markers, False)["passed"])
            for bad in (
                self.dcp_log(ranks=range(7)),
                self.dcp_log(buckets=(1, 2, 4)),
                self.dcp_log(tp=4),
            ):
                (root / "engine.log").write_text(bad)
                events, replay, markers = collect(root)
                self.assertFalse(
                    verify(events, "decode", replay, markers, True)["passed"]
                )

    def test_mixed_dcp_requires_both_dp_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            def logs(workers):
                lines = ["[K3_PAGE_RR_TARGET] role=Decode TP=4 B=128 V=64"]
                for world in workers:
                    lines.append(f"[MLA_DCP] backend=a2a tp=4 rank={world % 4} world_rank={world} dp_rank={world // 4}")
                    for bucket in physical_graph_buckets(4):
                        lines.append(f"[RANK {world}] captured batch size {bucket}:")
                return "\n".join(lines)
            for workers, passed in ((range(8), True), (range(4), False), (range(1, 8), False)):
                (root / "engine.log").write_text(logs(workers))
                events, replay, markers = collect(root)
                report = verify(events, "decode", replay, markers, True, 3, 4, 2)
                self.assertEqual(report["passed"], passed, report)
            bad = logs(range(8)).replace("world_rank=7 dp_rank=1", "world_rank=7 dp_rank=0")
            (root / "engine.log").write_text(bad)
            events, replay, markers = collect(root)
            self.assertFalse(verify(events, "decode", replay, markers, True, 3, 4, 2)["passed"])

    def test_mixed_page_rr_rejects_wrong_block_or_checkpoint_span(self):
        markers = {
            "dcp_backends": {(4, rank) for rank in range(4)},
            "dcp_workers": {(rank, rank // 4, rank % 4) for rank in range(8)},
            "graph_captures": {(rank, bucket) for rank in range(8) for bucket in physical_graph_buckets(4)},
        }
        for block, span, passed in ((1024, 8192, True), (128, 1024, False), (1024, 4096, False)):
            markers["page_rr_targets"] = {("Decode", 4, block, span)}
            report = verify([], "decode", False, markers, True, 3, 4, 2, 1024, 8)
            self.assertEqual(report["passed"], passed, report)

    def test_dcp_round_rejects_projection_ktp_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "engine.log").write_text(
                self.dcp_log() + "[K3_PROJECTION_KTP_GRAPH_REPLAY] graph_key=0\n"
            )
            events, replay, markers = collect(root)
            report = verify(events, "decode", replay, markers, True)
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["projection_ktp_inactive"])

    def test_collect_reads_rank_from_main_log_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            log_dir = root / "runtime/logs/decode"
            log_dir.mkdir(parents=True)
            for rank in range(8):
                (log_dir / f"main_{rank}.log").write_text(
                    f"[MLA_DCP] backend=a2a tp=8 rank={rank}\n"
                    f"[K3_PAGE_RR_TARGET] role=Decode TP=8 B=128 V=64\n"
                    + "".join(
                        f"captured batch size {bucket}: pool_delta=46 MiB\n"
                        for bucket in (2, 4, 8)
                    )
                )
            events, replay, markers = collect(root)
            self.assertTrue(
                verify(events, "decode", replay, markers, True)["passed"]
            )


if __name__ == "__main__":
    unittest.main()
