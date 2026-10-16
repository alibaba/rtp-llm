"""CPU-only harness tests; no model imports or GPUs."""

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PATH = Path(__file__).with_name("qwen35_epd_fusion_4gpu_smoke.py")
import sys

sys.path.insert(0, str(PATH.parent))
import http.server
import json
import threading
import time
from types import SimpleNamespace

import qwen35_epd_fusion_4gpu_smoke as smoke


class FusionSmokeTest(unittest.TestCase):
    def test_output_failure_continuation_preserves_strict_validation_and_protocol_failures(
        self,
    ):
        row = dict(
            ok=False,
            http_status=200,
            finish_reason="length",
            reasoning_content="still thinking",
            content="",
            usage=dict(completion_tokens=32000),
            validation_errors=[
                "incomplete, empty, or truncated generation",
                "incomplete, empty, or truncated choice",
                "invalid requested visual-event JSON",
            ],
        )
        with mock.patch.dict(smoke.CLIENT_OPTIONS, continue_output_errors=False):
            self.assertTrue(smoke.steady_response_requires_abort(row))
        with mock.patch.dict(smoke.CLIENT_OPTIONS, continue_output_errors=True):
            self.assertFalse(smoke.steady_response_requires_abort(row))
            self.assertFalse(row["ok"])
            self.assertEqual(len(row["validation_errors"]), 3)
            for extra in [
                "prompt token mismatch",
                "HTTP status",
                "missing/invalid PDFUSION evidence",
                "ReadTimeout()",
            ]:
                failed = dict(row, validation_errors=row["validation_errors"] + [extra])
                self.assertTrue(smoke.steady_response_requires_abort(failed))
            self.assertTrue(
                smoke.steady_response_requires_abort(
                    dict(row, error="connection reset")
                )
            )

    def test_quality_failures_are_not_counted_as_valid_answer_throughput(self):
        good = dict(
            ok=True,
            started_s=0,
            completed_s=0.5,
            client_rank=0,
            usage=dict(completion_tokens=10),
        )
        output = dict(
            good,
            ok=False,
            completed_s=0.6,
            http_status=200,
            finish_reason="length",
            content="",
            reasoning_content="thinking",
            validation_errors=["incomplete, empty, or truncated generation"],
        )
        transport = dict(output, completed_s=0.7, error="connection reset")
        summary = smoke.steady_summary([good, output, transport], 0, 1, 1)
        self.assertFalse(summary["all_valid"])
        window = summary["measurement"]
        self.assertEqual(window["qps"], 1)
        self.assertEqual(window["model_completed_qps"], 2)
        self.assertEqual(window["errors"], 2)
        self.assertEqual(window["output_quality_errors"], 1)

    def test_output_continuation_cli_requires_measured_steady_load(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            smoke.main(
                [
                    "--output",
                    "/quality-accounting",
                    "--steady-continue-output-errors",
                    "1",
                ]
            )
        with mock.patch.object(smoke, "run_launcher", side_effect=smoke.bazel_command):
            command = smoke.main(
                [
                    "--output",
                    "/quality-accounting",
                    "--workload",
                    "text",
                    "--client-mode",
                    "measured",
                    "--steady-concurrency",
                    "4",
                    "--steady-continue-output-errors",
                    "1",
                ]
            )
        self.assertIn("--test_arg=--steady-continue-output-errors=1", command)

    def test_untimed_default_and_explicit_legacy_timeout(self):
        self.assertEqual(smoke.CADENCE_ENV["PDFUSION_CADENCE_PREFILL_MAX_WAIT_MS"], 0)
        for value in ("0", "1000"):
            result = smoke.cadence_env(
                dict(
                    PDFUSION_CADENCE_ADAPTIVE="1",
                    PDFUSION_CADENCE_PREFILL_MAX_WAIT_MS=value,
                ),
                "cadence",
            )
            self.assertEqual(result["PDFUSION_CADENCE_PREFILL_MAX_WAIT_MS"], value)
        with self.assertRaises(ValueError):
            smoke.cadence_env(
                dict(
                    PDFUSION_CADENCE_ADAPTIVE="1",
                    PDFUSION_CADENCE_PREFILL_MAX_WAIT_MS="-1",
                ),
                "cadence",
            )

    def test_drain_marker_never_precedes_submission_end_and_is_run_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            marker = out / "cadence-drain"
            with mock.patch.dict(
                smoke.CLIENT_OPTIONS,
                dict(cadence_drain_file=str(marker), cadence_run_id="this-run"),
            ):
                self.assertFalse(
                    smoke.signal_cadence_drain(out, "steady", 1499.999, 1500)
                )
                self.assertFalse(marker.exists())
                self.assertTrue(smoke.signal_cadence_drain(out, "steady", 1500, 1500))
                self.assertEqual(marker.read_text(), "this-run\n")
                self.assertFalse(smoke.signal_cadence_drain(out, "steady", 1501, 1500))
                self.assertEqual(
                    json.loads((out / "steady-drain.json").read_text())["elapsed_s"],
                    1500,
                )
                marker.write_text("foreign-run\n")
                with self.assertRaisesRegex(RuntimeError, "foreign"):
                    smoke.signal_cadence_drain(out, "steady", 1502, 1500)

    def test_encoder_does_not_inherit_cadence_drain_control(self):
        env = smoke.encoder_cadence_env(
            dict(
                PDFUSION_CADENCE_ADAPTIVE="1",
                PDFUSION_CADENCE_DRAIN_FILE="/tmp/test-drain",
            )
        )
        self.assertNotIn("PDFUSION_CADENCE_DRAIN_FILE", env)
        self.assertEqual(env["PDFUSION_CADENCE_ADAPTIVE"], "0")

    def four_video_fixture(self, directory):
        import hashlib

        directory = Path(directory)
        entries = []
        for i in range(1, 5):
            video = directory / f"video{i}.mp4"
            video.write_bytes(f"video {i}".encode())
            body = dict(
                model="qwen35_moe",
                stream=False,
                max_tokens=8192,
                temperature=0,
                messages=[
                    dict(
                        role="user",
                        content=[
                            dict(
                                type="video_url",
                                video_url=dict(url=f"https://fixture/{i}.mp4"),
                            )
                        ],
                    )
                ],
            )
            template = directory / f"request{i}.json"
            template.write_text(json.dumps(body))
            entries.append(
                dict(
                    id=i,
                    video=video.name,
                    template=template.name,
                    video_url=f"https://fixture/{i}.mp4",
                    video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                    request_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
                )
            )
        (directory / "manifest.json").write_text(json.dumps(dict(requests=entries)))

    def test_four_video_omits_request_cap_and_explicitly_enables_thinking(self):
        from qwen35_mixed_video import load_workload, output_token_budget

        with tempfile.TemporaryDirectory() as directory:
            self.four_video_fixture(directory)
            rows, _ = load_workload(
                directory, omit_max_tokens=True, enable_thinking=True
            )
            for row in rows:
                body = row["payload"]
                self.assertNotIn("max_tokens", body)
                self.assertNotIn("extra_configs", body)
                self.assertFalse(body["stream"])
                self.assertTrue(body["enable_thinking"])
                self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
                self.assertTrue(
                    Path(
                        body["messages"][0]["content"][0]["video_url"]["url"]
                    ).is_file()
                )
            original, _ = load_workload(directory)
            self.assertTrue(all(r["payload"]["max_tokens"] == 8192 for r in original))
        self.assertEqual(output_token_budget({}, 24599, 65536, 32000), 32000)
        self.assertEqual(output_token_budget({}, 40642, 65536, 32000), 24894)

    def test_four_video_checksum_and_path_escape_are_rejected(self):
        from qwen35_mixed_video import load_workload

        with tempfile.TemporaryDirectory() as directory:
            self.four_video_fixture(directory)
            p = Path(directory)
            (p / "video1.mp4").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_workload(directory)
            self.four_video_fixture(directory)
            m = json.loads((p / "manifest.json").read_text())
            m["requests"][0]["video"] = "../outside.mp4"
            (p / "manifest.json").write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_workload(directory)

    def test_seeded_four_video_selection_replays_across_process_strides(self):
        from collections import Counter

        from qwen35_mixed_video import select_request

        workload = [dict(id=i) for i in range(1, 5)]
        serial = [select_request(workload, i, 20261016)["id"] for i in range(1000)]
        replay = {}
        for worker in range(4):
            for i in reversed(range(worker, 1000, 4)):
                replay[i] = select_request(workload, i, 20261016)["id"]
        self.assertEqual(serial, [replay[i] for i in range(1000)])
        self.assertTrue(all(200 < n < 300 for n in Counter(serial).values()))
        self.assertNotEqual(
            serial, [select_request(workload, i, 1)["id"] for i in range(1000)]
        )

    def test_nonstream_thinking_uses_server_ttft_and_keeps_reasoning(self):
        body = dict(
            choices=[
                dict(
                    index=0,
                    message=dict(content="answer", reasoning_content="consider video"),
                    finish_reason="stop",
                )
            ],
            usage=dict(
                prompt_tokens=100,
                completion_tokens=11,
                prompt_tokens_details=dict(video_tokens=80),
            ),
            aux_info=dict(
                pd_sep=False, reuse_len=0, first_token_cost_time=250, cost_time=1250
            ),
        )
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.text = json.dumps(body)
        payload = dict(
            stream=False,
            enable_thinking=True,
            chat_template_kwargs=dict(enable_thinking=True),
        )
        with mock.patch("requests.post", return_value=response) as post:
            row = smoke.request(
                "http://test", payload, 0, dict(prompt_tokens=100, video_tokens=80)
            )
        self.assertTrue(row["ok"], row)
        self.assertTrue(row["thinking_observed"])
        self.assertEqual(row["reasoning_content"], "consider video")
        self.assertEqual(row["ttft_s"], 0.25)
        self.assertEqual(row["tpot_ms"], 100)
        self.assertEqual(row["ttft_source"], "server_aux_first_token_cost_time")
        self.assertNotIn("first_content_s", row)
        self.assertNotIn("max_tokens", post.call_args.kwargs["json"])

    def test_nonstream_missing_first_token_timing_is_not_faked_from_completion(self):
        row = dict(
            http_status=200,
            content="answer",
            finish_reason="stop",
            response_stream=False,
            usage=dict(prompt_tokens=100),
            aux_info=dict(pd_sep=False, reuse_len=0),
            e2e_s=9,
        )
        smoke.record_output_timing(row)
        self.assertNotIn("ttft_s", row)
        self.assertIn(
            "missing nonstream server first-token timing",
            smoke.validate_response(row, dict(prompt_tokens=100, video_tokens=0)),
        )

    def test_long_thinking_read_timeout_is_forwarded_without_an_output_cap(self):
        response = mock.MagicMock(status_code=200)
        response.__enter__.return_value = response
        response.text = json.dumps(
            dict(
                choices=[
                    dict(index=0, message=dict(content="answer"), finish_reason="stop")
                ],
                usage=dict(prompt_tokens=100, completion_tokens=2),
                aux_info=dict(
                    pd_sep=False, reuse_len=0, first_token_cost_time=1, cost_time=2
                ),
            )
        )
        with mock.patch.dict(
            smoke.CLIENT_OPTIONS, read_timeout_seconds=3600
        ), mock.patch("requests.post", return_value=response) as post:
            row = smoke.request(
                "http://test",
                dict(stream=False, enable_thinking=True),
                0,
                dict(prompt_tokens=100, video_tokens=0),
            )
        self.assertTrue(row["ok"], row)
        self.assertEqual(post.call_args.kwargs["timeout"], (10, 3600))
        self.assertNotIn("max_tokens", post.call_args.kwargs["json"])
        with mock.patch.object(smoke, "run_launcher", side_effect=smoke.bazel_command):
            command = smoke.main(
                ["--output", "/long-thinking", "--client-read-timeout-seconds", "3600"]
            )
        self.assertIn("--test_arg=--client-read-timeout-seconds=3600", command)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            smoke.main(
                ["--output", "/long-thinking", "--client-read-timeout-seconds", "0"]
            )

    def test_thinking_stream_ttft_precedes_first_visible_content(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.iter_lines.return_value = [
            b"data: " + json.dumps(x).encode()
            for x in [
                dict(choices=[dict(index=0, delta=dict(reasoning_content="consider"))]),
                dict(
                    choices=[
                        dict(
                            index=0, delta=dict(content="answer"), finish_reason="stop"
                        )
                    ]
                ),
                dict(
                    usage=dict(prompt_tokens=100, completion_tokens=3),
                    aux_info=dict(pd_sep=False, reuse_len=0),
                ),
            ]
        ]
        with mock.patch("requests.post", return_value=response):
            row = smoke.request(
                "http://test",
                dict(stream=True),
                0,
                dict(prompt_tokens=100, video_tokens=0),
            )
        self.assertTrue(row["ok"], row)
        self.assertEqual(row["ttft_s"], row["first_reasoning_s"])
        self.assertLessEqual(row["first_reasoning_s"], row["first_content_s"])
        self.assertEqual(row["reasoning_content"], "consider")

    def test_mixed_cli_preserves_uncapped_thinking_and_capacity_in_worker_arguments(
        self,
    ):
        args = [
            "--output",
            "/mixed-test",
            "--encoder-gpus",
            "0",
            "--gpus",
            "4,5,6,7",
            "--encoder-proxy",
            "1",
            "--mixed-video",
            "--omit-max-tokens",
            "--mixed-thinking",
            "1",
            "--max-input-tokens",
            "65536",
            "--client-mode",
            "measured",
            "--steady-concurrency",
            "512",
            "--rank-concurrency",
            "256",
            "--mixed-seed",
            "20261016",
        ]
        with mock.patch.object(smoke, "run_launcher", side_effect=smoke.bazel_command):
            command = smoke.main(args)
        for field in [
            "--mixed-video",
            "--omit-max-tokens",
            "--mixed-thinking=1",
            "--max-input-tokens=65536",
            "--mixed-seed=20261016",
        ]:
            self.assertIn("--test_arg=" + field, command)
        _, original = smoke.server_config("4,5,6,7")
        larger = smoke.input_capacity_config(original, 65536)
        for name in [
            "max_seq_len",
            "max_batch_tokens_size",
            "max_batch_tokens_without_cache",
        ]:
            self.assertIn("--" + name + " 65536", larger)
        self.assertEqual(smoke.input_capacity_config(original, 32768), original)

    def test_adaptive_environment_validation_and_pd_forwarding(self):
        settings = dict(
            PDFUSION_CADENCE_ADAPTIVE="1",
            PDFUSION_CADENCE_DECODE_WINDOW="32",
            PDFUSION_CADENCE_LOW_WATER_PCT="60",
            PDFUSION_CADENCE_PREFILL_MAX_WAIT_MS="500",
            PDFUSION_CADENCE_MIN_DECODE_SHARE_PCT="30",
        )
        with mock.patch.dict(os.environ, settings, clear=True):
            env, args = smoke.server_config("2,3", "baseline", "mega_moe_fp8", 0)
            env, args = smoke.throughput_config(
                env, args, "prefill-first", "7", 1, "adaptive", "cadence"
            )
            self.assertEqual({k: env[k] for k in settings}, settings)
            configs = smoke.dp2_service_configs(
                "2,3",
                "1",
                env,
                args,
                dict(encoder_0=20000, encoder_1=21000, fusion=22000),
                9,
                "candidate-a",
            )
            for config in configs:
                if config["name"] == "fusion":
                    self.assertEqual(config["env"]["PDFUSION_CADENCE_ADAPTIVE"], "1")
                else:
                    self.assertEqual(config["env"]["PDFUSION_CADENCE_ADAPTIVE"], "0")
                    self.assertFalse(
                        (set(settings) - {"PDFUSION_CADENCE_ADAPTIVE"})
                        & config["env"].keys()
                    )
            with mock.patch.object(
                smoke, "run_launcher", side_effect=smoke.bazel_command
            ):
                command = smoke.main(
                    [
                        "--output",
                        "/new-run",
                        "--coord-mode",
                        "cadence",
                        "--scheduler-policy",
                        "prefill-first",
                        "--decode-prefill-ratio",
                        "7",
                        "--schedule-trace",
                        "1",
                        "--trace-run-id",
                        "adaptive",
                    ]
                )
            for name, value in settings.items():
                self.assertIn(f"--test_env={name}={value}", command)
        self.assertEqual(smoke.cadence_env({}, "off"), {})
        for settings in (
            dict(PDFUSION_CADENCE_ADAPTIVE="2"),
            dict(PDFUSION_CADENCE_ADAPTIVE="1", PDFUSION_CADENCE_LOW_WATER_PCT="0"),
            dict(PDFUSION_CADENCE_ADAPTIVE="1", PDFUSION_CADENCE_DECODE_WINDOW="-1"),
            dict(
                PDFUSION_CADENCE_ADAPTIVE="1",
                PDFUSION_CADENCE_MIN_DECODE_SHARE_PCT="100",
            ),
            dict(
                PDFUSION_CADENCE_ADAPTIVE="1",
                PDFUSION_CADENCE_MIN_DECODE_SHARE_PCT="-1",
            ),
        ):
            with self.assertRaises(ValueError):
                smoke.cadence_env(settings, "cadence")
        self.assertEqual(
            smoke.effective_coord_mode(dict(PDFUSION_CADENCE_ADAPTIVE="1"), "off"),
            "cadence",
        )

    def test_adaptive_auto_selection_omits_legacy_scheduler_and_ratio_arguments(self):
        with mock.patch.dict(os.environ, PDFUSION_CADENCE_ADAPTIVE="1", clear=True):
            for ratio in ("0", "7", "1/7", "invalid"):
                env, args = smoke.throughput_config(
                    {}, "", "fifo", ratio, 1, "automatic-adaptive", "off"
                )
                self.assertEqual(env["PDFUSION_CADENCE_ADAPTIVE"], "1")
                self.assertIn("--pdfusion_coord_mode cadence", args)
                self.assertNotIn("--pdfusion_scheduler_mode", args)
                self.assertNotIn("--decode_prefill_ratio", args)
            with mock.patch.object(
                smoke, "run_launcher", side_effect=smoke.bazel_command
            ):
                command = smoke.main(
                    [
                        "--output",
                        "/automatic-adaptive",
                        "--workload",
                        "text",
                        "--schedule-trace",
                        "1",
                        "--trace-run-id",
                        "automatic-adaptive",
                    ]
                )
            self.assertIn("--test_env=PDFUSION_CADENCE_ADAPTIVE=1", command)
            self.assertIn("--test_arg=--coord-mode=cadence", command)
            self.assertFalse(
                any(
                    arg.startswith(
                        (
                            "--test_arg=--scheduler-policy=",
                            "--test_arg=--decode-prefill-ratio=",
                        )
                    )
                    for arg in command
                )
            )

    def test_adaptive_auto_selection_retains_trace_and_mtp_safety_checks(self):
        with mock.patch.dict(os.environ, PDFUSION_CADENCE_ADAPTIVE="1", clear=True):
            for trace, run_id in ((0, "automatic"), (1, "unset"), (1, "x" * 64)):
                with self.assertRaises(ValueError):
                    smoke.throughput_config({}, "", "fifo", "0", trace, run_id)
            with mock.patch.object(
                smoke, "run_launcher", side_effect=smoke.bazel_command
            ):
                command = smoke.main(
                    [
                        "--output",
                        "/automatic-mtp",
                        "--gpus",
                        "1,2",
                        "--encoder-gpus",
                        "0",
                        "--encoder-proxy",
                        "1",
                        "--mtp-tokens",
                        "1",
                        "--schedule-trace",
                        "1",
                        "--trace-run-id",
                        "automatic-mtp",
                    ]
                )
            self.assertIn("--test_arg=--coord-mode=cadence", command)
            with mock.patch.dict(os.environ, RTP_LLM_STREAM_ASYNC="1"):
                with self.assertRaisesRegex(ValueError, "synchronous host dispatch"):
                    smoke.main(
                        [
                            "--output",
                            "/automatic-mtp",
                            "--workload",
                            "text",
                            "--mtp-tokens",
                            "1",
                            "--schedule-trace",
                            "1",
                            "--trace-run-id",
                            "automatic-mtp",
                        ]
                    )

    def test_disabled_adaptive_retains_legacy_arguments_and_ratio_validation(self):
        with mock.patch.dict(os.environ, PDFUSION_CADENCE_ADAPTIVE="0", clear=True):
            _, fifo = smoke.throughput_config({}, "", "fifo")
            self.assertEqual(fifo, " --pdfusion_coord_mode off")
            _, args = smoke.throughput_config(
                {}, "", "prefill-first", "7", 1, "legacy", "cadence"
            )
            self.assertIn(
                "--pdfusion_scheduler_mode ratio --decode_prefill_ratio 7", args
            )
            with self.assertRaises(ValueError):
                smoke.throughput_config({}, "", "fifo", "0", 1, "legacy", "cadence")
            with self.assertRaises(ValueError):
                smoke.throughput_config(
                    {}, "", "prefill-first", "1/7", 1, "legacy", "cadence"
                )

    def test_cadence_mtp1_forwards_only_pd_with_synchronous_dispatch(self):
        env, args = smoke.server_config("2,3", "baseline", "mega_moe_fp8", 0)
        env, args = smoke.throughput_config(
            env, args, "prefill-first", "7", 1, "mtp-cadence", "cadence"
        )
        configs = smoke.dp2_service_configs(
            "2,3",
            "1",
            env,
            args,
            dict(encoder_0=20000, encoder_1=21000, fusion=22000),
            9,
            "candidate-a",
        )
        encoder, pd = smoke.configure_mtp_services(configs, 1, "/model", "cadence")
        self.assertEqual(encoder["env"]["SP_TYPE"], "")
        self.assertNotIn("--sp_type", encoder["args"])
        self.assertIn("--sp_type eagle", pd["args"])
        self.assertIn("--pdfusion_coord_mode cadence", pd["args"])
        self.assertEqual(pd["env"]["RTP_MEGAMOE_CHUNK_TOKENS"], "0")

    def test_cadence_mtp_rejects_async_dispatch_and_width_above_four(self):
        for name in (
            "RTP_LLM_STREAM_ASYNC",
            "RTP_LLM_DROP_BROAD_SYNC",
            "RTP_LLM_MTP_ASYNC_DEVICE_STATE",
            "RTP_LLM_DEVICE_INPUT",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "synchronous host"
            ):
                smoke.mtp_config({name: "1"}, "", 1, "/model", "cadence")
        with self.assertRaisesRegex(ValueError, "1 through 4"):
            smoke.mtp_config({}, "", 5, "/model", "cadence")
        for width in (1, 2, 3, 4):
            env, args = smoke.mtp_config({}, "", width, "/model", "cadence")
            self.assertIn(f"--gen_num_per_cycle {width}", args)

    def test_two_pd_cadence_mtp1_cli_is_allowed_with_explicit_trace(self):
        with mock.patch.object(smoke, "run_launcher", side_effect=smoke.bazel_command):
            command = smoke.main(
                [
                    "--output",
                    "/new-run",
                    "--gpus",
                    "2,3",
                    "--encoder-gpus",
                    "1",
                    "--encoder-proxy",
                    "1",
                    "--mtp-tokens",
                    "1",
                    "--coord-mode",
                    "cadence",
                    "--scheduler-policy",
                    "prefill-first",
                    "--decode-prefill-ratio",
                    "7",
                    "--schedule-trace",
                    "1",
                    "--trace-run-id",
                    "cadence-mtp1-test",
                ]
            )
        self.assertIn("--test_arg=--coord-mode=cadence", command)
        self.assertIn("--test_arg=--mtp-tokens=1", command)
        self.assertIn("--test_env=GPU_COUNT=3", command)

    def test_image_probe_checks_answers_embeddings_and_uneven_rank_load(self):
        from collections import Counter

        cases = [
            dict(name=color, payload={}, expected={}, visual_tokens=256)
            for color in ("red", "green", "blue")
        ]

        def batch(manager, payload, expected, concurrency, count, rank_count):
            self.assertEqual((concurrency, count, rank_count), (3, 3, 2))
            color = cases[batch.calls]["name"]
            batch.calls += 1
            return [
                dict(
                    ok=True,
                    content=color + ".",
                    client_rank=i % rank_count,
                    usage=dict(completion_tokens=2),
                    aux_info=dict(
                        multimodal_lengths={"1": 256},
                        output_len=2,
                        speculative_draft_rounds=1,
                        speculative_accepted_tokens_per_pos=[1],
                    ),
                )
                for i in range(count)
            ]

        batch.calls = 0
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            smoke, "image_probe_cases", return_value=cases
        ), mock.patch.object(smoke, "run_throughput_batch", side_effect=batch):
            result = smoke.run_image_probe(
                None, "/model", Path(directory), "before", 2, 1
            )
            saved = json.loads((Path(directory) / "image-before.json").read_text())
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["requests"], 9)
        self.assertEqual(result["speculative"]["draft_rounds"], 9)
        self.assertEqual(
            Counter(r["client_rank"] for r in saved["requests"]), {0: 6, 1: 3}
        )

    def test_image_probe_rejects_wrong_answers_missing_embeddings_and_bad_http(self):
        case = dict(name="red", payload={}, expected={}, visual_tokens=256)
        valid = dict(
            ok=True,
            content="red",
            usage=dict(completion_tokens=2),
            aux_info=dict(multimodal_lengths={"1": 256}, output_len=2),
        )
        for patch in [
            dict(content="blue"),
            dict(aux_info={}),
            dict(ok=False),
            dict(usage=dict(completion_tokens=33)),
        ]:
            with self.subTest(
                patch=patch
            ), tempfile.TemporaryDirectory() as directory, mock.patch.object(
                smoke, "image_probe_cases", return_value=[case]
            ), mock.patch.object(
                smoke, "run_throughput_batch", return_value=[dict(valid, **patch)]
            ):
                with self.assertRaisesRegex(RuntimeError, "image sanity check failed"):
                    smoke.run_image_probe(None, "/model", Path(directory), "before", 2)
                saved = json.loads((Path(directory) / "image-before.json").read_text())
                self.assertFalse(saved["all_passed"])

    def test_image_probe_cli_reaches_worker_and_defaults_off(self):
        with mock.patch.object(smoke, "run_launcher", side_effect=smoke.bazel_command):
            command = smoke.main(
                [
                    "--image-probe",
                    "1",
                    "--output",
                    "/new-run",
                    "--gpus",
                    "2,3",
                    "--encoder-gpus",
                    "1",
                ]
            )
            default = smoke.main(["--output", "/new-run", "--gpus", "4,5,6,7"])
        self.assertIn("--test_arg=--image-probe=1", command)
        self.assertIn("--test_arg=--image-probe=0", default)

    def test_image_probe_requires_encoder(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            smoke.main(
                ["--image-probe", "1", "--output", "/new-run", "--gpus", "4,5,6,7"]
            )

    def test_mtp_disabled_preserves_existing_configuration(self):
        env, args = smoke.server_config("4,5,6,7", "baseline", "mega_moe_fp8", 1)
        actual_env, actual_args = smoke.mtp_config(env, args, 0, "/model", "cadence")
        self.assertIs(actual_env, env)
        self.assertEqual(actual_args, args)
        configs = smoke.separated_service_configs(
            "4,5,6,7",
            "0",
            "baseline",
            dict(encoder_0=20000, encoder_1=21000, fusion=22000),
            9,
        )
        self.assertIs(smoke.configure_mtp_services(configs, 0, "/model"), configs)

    def test_mtp_services_enable_only_pd_and_keep_checkpoint_path_intact(self):
        import shlex

        for proxy in (0, 1):
            for tokens in (1, 4):
                with self.subTest(proxy=proxy, tokens=tokens):
                    env, args = smoke.server_config(
                        "4,5,6,7", "baseline", "mega_moe_fp8", 0, 1, 1, 4096
                    )
                    env, args = smoke.throughput_config(
                        env, args, "prefill-first", "7", coord_mode="off"
                    )
                    ports = dict(encoder_0=20000, encoder_1=21000, fusion=22000)
                    if proxy:
                        configs = smoke.dp2_service_configs(
                            "4,5,6,7", "0", env, args, ports, 9, "candidate-a"
                        )
                    else:
                        configs = smoke.separated_service_configs(
                            "4,5,6,7",
                            "0",
                            "baseline",
                            ports,
                            9,
                            "mega_moe_fp8",
                            0,
                            1,
                            1,
                            4096,
                        )
                    original_encoder_args = configs[0]["args"]
                    actual = smoke.configure_mtp_services(
                        configs, tokens, "/model with spaces", "off"
                    )
                    self.assertEqual(actual[0]["args"], original_encoder_args)
                    self.assertEqual(actual[0]["env"]["SP_TYPE"], "")
                    self.assertNotIn("qwen35_moe_mtp", actual[0]["args"])
                    pd = actual[-1]
                    words = shlex.split(pd["args"])
                    for name, value in (
                        ("sp_model_type", "qwen35_moe_mtp"),
                        ("sp_type", "eagle"),
                        ("sp_checkpoint_path", "/model with spaces"),
                        ("sp_act_type", "bf16"),
                        ("gen_num_per_cycle", str(tokens)),
                        ("enable_mixed_continuous_batching", "0"),
                        ("use_batch_decode_scheduler", "0"),
                    ):
                        self.assertEqual(words[words.index("--" + name) + 1], value)
                    self.assertEqual(pd["env"]["RTP_MEGAMOE_CHUNK_TOKENS"], "0")
                    self.assertIn("--fp8_kv_cache 1", pd["args"])
                    self.assertEqual(pd["env"]["RTP_QWEN35_NATIVE_FP8_ATTN"], "1")
                    self.assertNotIn("SP_TYPE", configs[0]["env"])

    def test_mtp_rejects_negative_tokens_cadence_and_mixed_batching(self):
        for tokens, coord, env, args in (
            (-1, "off", {}, ""),
            (5, "cadence", {}, ""),
            (4, "off", {"ENABLE_MIXED_CONTINUOUS_BATCHING": "1"}, ""),
            (1, "off", {}, "--enable_mixed_continuous_batching 1"),
            (1, "off", {}, "--enable_mixed_continuous_batching=true"),
        ):
            with self.subTest(tokens=tokens, coord=coord, env=env, args=args):
                with self.assertRaises(ValueError):
                    smoke.mtp_config(env, args, tokens, "/model", coord)
        with mock.patch.dict(os.environ, ENABLE_MIXED_CONTINUOUS_BATCHING="1"):
            with self.assertRaisesRegex(ValueError, "mixed continuous"):
                smoke.mtp_config({}, "", 1, "/model")
            # Explicit server arguments have the same precedence as server startup.
            smoke.mtp_config({}, "--enable_mixed_continuous_batching 0", 1, "/model")

    def test_mtp_cli_reaches_worker_and_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "new-run")
            argv = [
                "--output",
                output,
                "--gpus",
                "4,5,6,7",
                "--encoder-gpus",
                "0",
                "--encoder-proxy",
                "1",
                "--mtp-tokens",
                "4",
                "--coord-mode",
                "off",
            ]
            with mock.patch.object(
                smoke, "run_launcher", side_effect=smoke.bazel_command
            ):
                command = smoke.main(argv)
            self.assertIn("--test_arg=--mtp-tokens=4", command)
            self.assertIn("--test_env=GPU_COUNT=5", command)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(smoke.main(argv), 0)
            self.assertIn("--test_arg=--mtp-tokens=4", stdout.getvalue())
            self.assertFalse(Path(output).exists())
            for extra in (
                ["--mtp-tokens=-1"],
                ["--mtp-tokens=5", "--coord-mode=cadence"],
            ):
                with self.assertRaises(ValueError):
                    smoke.main(argv + extra)

    def test_mtp_worker_passes_pd_settings_to_real_manager_boundary(self):
        from types import ModuleType

        module = ModuleType("rtp_llm.test.utils.maga_server_manager")
        config_module = ModuleType("rtp_llm.config.py_config_modules")
        config_module.MIN_WORKER_INFO_PORT_NUM = 9
        managers = [mock.Mock(port=20000 + i * 1000, exit_code=None) for i in range(2)]
        for manager in managers:
            manager.start_server.return_value = False
        module.MagaServerManager = mock.Mock(side_effect=managers)
        module.MagaServerManager.get_free_port.side_effect = [20000, 21000, 22000]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=directory,
                gpus="4,5,6,7",
                encoder_gpus="0",
                data_dir=str(PATH.parent / "qwen35_e2p4d2_data"),
                model_dir=directory,
                mtp_tokens=4,
                coord_mode="off",
                encoder_dp2=1,
            )
            with mock.patch.dict(
                os.environ, CUDA_VISIBLE_DEVICES="0,4,5,6,7"
            ), mock.patch.dict(
                sys.modules,
                {module.__name__: module, config_module.__name__: config_module},
            ), mock.patch.object(
                smoke, "check_gpu_idle", return_value={}
            ), mock.patch.object(
                smoke, "expected_tokens", return_value={}
            ), mock.patch.object(
                smoke.threading.Thread, "start"
            ), mock.patch.object(
                smoke.threading.Thread, "join"
            ), mock.patch.object(
                smoke.concurrent.futures, "ThreadPoolExecutor"
            ) as pool:
                pool.return_value.__enter__.return_value.map.side_effect = (
                    lambda fn, items: list(map(fn, items))
                )
                self.assertEqual(smoke.run_worker(args), 1)
            calls = module.MagaServerManager.call_args_list
            self.assertEqual(calls[0].kwargs["env_args"]["SP_TYPE"], "")
            self.assertIn("--gen_num_per_cycle 4", calls[1].kwargs["smoke_args_str"])
            self.assertEqual(
                calls[1].kwargs["env_args"]["RTP_MEGAMOE_CHUNK_TOKENS"], "0"
            )
            report = json.loads(Path(directory, "result.json").read_text())
            self.assertEqual(report["mtp_tokens"], 4)
            self.assertIn("--sp_model_type qwen35_moe_mtp", report["args"])
            for manager in managers:
                manager.stop_server.assert_called_once()

    def test_coordinator_graph_oracle_uses_observed_execution_rank(self):
        row = dict(rank=1, prefill=0, batch=63, mode="graph", graph_bs=64)
        self.assertEqual(smoke.validate_coordinator_boundary([row], 63, 1), [row])
        for records in (
            [],
            [dict(row, batch=62)],
            [dict(row, graph_bs=96)],
            [dict(row, mode="eager")],
        ):
            with self.assertRaises(RuntimeError):
                smoke.validate_coordinator_boundary(records, 63, 1)
        eager = dict(row, mode="eager", graph_bs=0)
        self.assertEqual(smoke.validate_coordinator_boundary([eager], 63, 0), [eager])

    def test_steady_completion_windows_exclude_warmup_and_drain(self):
        rows = [
            dict(
                started_s=0,
                completed_s=end,
                ok=ok,
                client_rank=rank,
                usage=dict(completion_tokens=100),
            )
            for end, ok, rank in (
                (9, True, 0),
                (10, True, 0),
                (15, False, 1),
                (19, True, 2),
                (20, True, 3),
                (30, True, 0),
            )
        ]
        result = smoke.steady_summary(rows, 10, 10, 2)
        self.assertEqual(result["measurement"]["successful"], 3)
        self.assertEqual(result["measurement"]["errors"], 1)
        self.assertEqual(result["measurement"]["qps"], 3 / 20)
        self.assertEqual([r["successful"] for r in result["rounds"]], [2, 1])
        self.assertEqual(result["warmup_completions"], 1)
        self.assertEqual(result["drain_completions"], 1)
        self.assertFalse(result["all_valid"])
        self.assertFalse(result["rounds_within_10_percent"])

    def test_steady_windows_count_carry_in_and_track_inflight(self):
        rows = [
            dict(
                started_s=start,
                completed_s=end,
                ok=True,
                client_rank=0,
                usage=dict(completion_tokens=200),
            )
            for start, end in ((0, 11), (11, 21), (21, 31))
        ]
        result = smoke.steady_summary(rows, 10, 10, 2)
        self.assertEqual(result["measurement"]["qps"], 0.1)
        self.assertEqual(result["measurement"]["completed_output_tokens_per_s"], 20)
        self.assertEqual(result["measurement"]["in_flight_at_start"], 1)
        self.assertEqual(result["measurement"]["in_flight_at_end"], 1)
        self.assertTrue(result["rounds_within_10_percent"])

    def test_steady_lanes_refill_until_deadline_and_drain(self):
        import sys
        import time
        from types import SimpleNamespace

        sleep = time.sleep

        def response(url, payload, index, expected):
            sleep(0.015)
            return dict(
                index=index,
                url=url,
                ok=True,
                usage=dict(completion_tokens=2),
                e2e_s=0.015,
                ttft_s=0.01,
            )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.config.py_config_modules": SimpleNamespace(
                        MIN_WORKER_INFO_PORT_NUM=9
                    )
                },
            ):
                with mock.patch.object(
                    smoke, "request", side_effect=response
                ), mock.patch.object(
                    smoke.time, "sleep", side_effect=lambda x: sleep(min(x, 0.02))
                ):
                    rows, result = smoke.run_steady_load(
                        SimpleNamespace(port=20000),
                        {},
                        {},
                        4,
                        0,
                        1,
                        2,
                        Path(directory),
                        "steady_test",
                    )
            self.assertGreater(len(rows), 100)
            self.assertTrue(all(r["started_s"] < 2 for r in rows))
            self.assertGreater(result["drain_completions"], 0)
            self.assertEqual(result["peak_in_flight"], 4)
            self.assertFalse(result["aborted"])
            for lane in range(4):
                local = [r for r in rows if r["client_lane"] == lane]
                self.assertGreater(len(local), 10)
                self.assertEqual(
                    {r["url"] for r in local},
                    {f"http://127.0.0.1:{20000+lane*9}/v1/chat/completions"},
                )

    def test_global_cadence_is_explicit_and_rejects_invalid_arguments(self):
        env, original = smoke.server_config("0,1,2,3", "baseline", "mega_moe_fp8", 1)
        args = smoke.throughput_config(
            env, original, "prefill-first", "16", 1, "g1-run", "cadence"
        )[1]
        self.assertIn("--pdfusion_coord_mode cadence", args)
        self.assertIn("--decode_prefill_ratio 16", args)
        self.assertIn(
            "--pdfusion_coord_mode off",
            smoke.throughput_config(env, original, "prefill-first")[1],
        )
        for ratio, trace, run in (
            ("0", 1, "x"),
            ("1/2", 1, "x"),
            ("16", 0, "x"),
            ("16", 1, "x" * 64),
        ):
            with self.assertRaises(ValueError):
                smoke.throughput_config(
                    env, original, "prefill-first", ratio, trace, run, "cadence"
                )

    def test_schedule_trace_and_ratio_reach_server_argv(self):
        env, original = smoke.server_config("0,1,2,3", "baseline", "mega_moe_fp8", 1)
        _, args = smoke.throughput_config(
            env, original, "prefill-first", "2", 1, "o0-b1"
        )
        self.assertIn("--decode_prefill_ratio 2", args)
        self.assertNotIn("--decode_prefill_ratio 0", args)
        self.assertIn("--pdfusion_schedule_trace 1 --pdfusion_trace_run_id o0-b1", args)
        self.assertNotIn(
            "--pdfusion_schedule_trace",
            smoke.throughput_config(env, original, "prefill-first")[1],
        )
        for policy, ratio, run in (
            ("fifo", "2", "x"),
            ("prefill-first", "-1", "x"),
            ("prefill-first", "2", "unset"),
            ("prefill-first", "2", "x y"),
        ):
            with self.assertRaises(ValueError):
                smoke.throughput_config(env, original, policy, ratio, 1, run)

    def test_throughput_policy_and_balanced_sweep(self):
        env, original = smoke.server_config("0,1,2,3", "baseline", "mega_moe_fp8", 1)
        _, fifo = smoke.throughput_config(env, original, "fifo")
        _, packed = smoke.throughput_config(env, original, "prefill-first")
        self.assertNotIn("--pdfusion_scheduler_mode", fifo)
        self.assertIn(
            "--pdfusion_scheduler_mode ratio --decode_prefill_ratio 0", packed
        )
        for args in (fifo, packed):
            self.assertIn("--concurrency_limit 4", args)
            self.assertIn("--max_context_batch_size 1", args)
            self.assertIn("--decode_capture_config 1,2,4", args)
        plan = smoke.throughput_plan(2)
        self.assertEqual([c for _, c, _ in plan[1:]], [4, 8, 16, 16, 8, 4])
        for _, concurrency, count in plan:
            assignments = [
                [i % 4 for i in range(w, count, concurrency)]
                for w in range(concurrency)
            ]
            self.assertTrue(all(len(set(a)) == 1 for a in assignments))
            self.assertEqual(
                [sum(a.count(r) for a in assignments) for r in range(4)],
                [count // 4] * 4,
            )

    def test_throughput_requests_use_all_four_frontends_and_refill_lanes(self):
        import sys
        from types import SimpleNamespace

        fake_config = SimpleNamespace(MIN_WORKER_INFO_PORT_NUM=9)

        def response(url, payload, index, expected):
            return dict(index=index, url=url, ok=True)

        with mock.patch.dict(
            sys.modules, {"rtp_llm.config.py_config_modules": fake_config}
        ):
            with mock.patch.object(smoke, "request", side_effect=response):
                for concurrency in (4, 8, 16):
                    rows = smoke.run_throughput_batch(
                        SimpleNamespace(port=20000), {}, {}, concurrency, 32
                    )
                    self.assertEqual(sorted(r["index"] for r in rows), list(range(32)))
                    for rank in range(4):
                        local = [r for r in rows if r["client_rank"] == rank]
                        self.assertEqual(len(local), 8)
                        self.assertEqual(
                            {r["url"] for r in local},
                            {
                                f"http://127.0.0.1:{20000 + rank * 9}/v1/chat/completions"
                            },
                        )
                    for lane in range(concurrency):
                        local = [r for r in rows if r["client_lane"] == lane]
                        self.assertEqual(
                            [r["index"] for r in local],
                            list(range(lane, 32, concurrency)),
                        )

    def test_capacity_configuration_keeps_graph_coverage_and_kv_budget(self):
        env, args = smoke.server_config("0,1,2,3", "baseline", "mega_moe_fp8", 1)
        _, configured = smoke.capacity_config(env, args, 24, 16384, "1,2,4,8,16,24", 1)
        self.assertIn("--concurrency_limit 24", configured)
        self.assertIn("--kv_cache_mem_mb 16384", configured)
        self.assertIn("--decode_capture_config 1,2,4,8,16,24", configured)
        self.assertIn("--max_batch_tokens_size 32768", configured)
        for captures in ("", "1,2,4,8,16", "1,2,2,24"):
            with self.assertRaises(ValueError):
                smoke.capacity_config(env, args, 24, 16384, captures, 1)
        plan = smoke.capacity_plan("8,16,24", 2)
        self.assertEqual([c for _, c, _ in plan], [16, 32, 32, 64, 64, 96, 96])

    def test_capacity_auto_kv_preserves_runtime_reserve(self):
        env, args = smoke.server_config("0,1,2,3", "baseline", "mega_moe_fp8", 1)
        _, configured = smoke.capacity_config(
            env, args, 160, 0, "1,2,4,8,16,32,64,96,128,160", 1
        )
        self.assertIn("--kv_cache_mem_mb 0", configured)
        self.assertIn("--reserver_runtime_mem_mb 24576", configured)
        self.assertIn("--concurrency_limit 160", configured)
        with self.assertRaises(ValueError):
            smoke.capacity_config(env, args, 160, -1, "1,160", 1)

    def test_capacity_refinement_uses_weakest_rank_and_tests_next_batch(self):
        summary = dict(
            max_observed_decode_batch_by_rank={"0": 119, "1": 118, "2": 120, "3": 119}
        )
        plan = smoke.capacity_refinement(summary, [32, 64, 96, 128, 160], 2, 160)
        self.assertEqual([n for _, n, _ in plan], [472, 472, 476, 476])
        self.assertEqual(smoke.capacity_refinement(summary, [118, 119], 2, 160), [])
        self.assertEqual(len(smoke.capacity_refinement(summary, [], 2, 118)), 2)

    def test_capacity_evidence_does_not_count_queued_requests_or_graph_padding(self):
        phase = dict(
            phase="capacity_b24_r1",
            concurrency=96,
            count=96,
            wall_s=100,
            output_tokens_per_s=1000,
        )
        report = dict(
            batches=[phase], requests=[dict(phase=phase["phase"], ok=True)] * 96
        )
        records = [dict(rank=r, prefill=0, batch=18, graph_bs=24) for r in range(4)]
        evidence = smoke.capacity_summary(report, dict(records=records))
        self.assertFalse(evidence["levels"]["24"]["all_ranks_reached_requested_batch"])
        self.assertEqual(
            evidence["max_observed_decode_batch_by_rank"],
            {str(r): 18 for r in range(4)},
        )
        records += [dict(rank=r, prefill=0, batch=24, graph_bs=24) for r in range(4)]
        self.assertTrue(
            smoke.capacity_summary(report, dict(records=records))["levels"]["24"][
                "all_ranks_reached_requested_batch"
            ]
        )

    def test_gpu_mapping_and_config(self):
        for value in ("0,1,2", "4,4,6,7", "-1,2,3,4"):
            with self.assertRaises(ValueError):
                smoke.gpu_ids(value)
        env, args = smoke.server_config("4,5,6,7")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
        self.assertEqual(env["ROLE_TYPE"], "PDFUSION")
        self.assertIn("--vit_separation 0", args)
        self.assertIn("--enable_cuda_graph 0", args)

    def test_mega_graph_configuration(self):
        for graph in (0, 1):
            env, args = smoke.server_config(
                "0,1,2,3", "baseline", "mega_moe_fp8", graph
            )
            self.assertEqual(env["MOE_STRATEGY"], "mega_moe_fp8")
            self.assertIn("--use_deepep_moe 0", args)
            self.assertIn(f"--enable_cuda_graph {graph}", args)
            self.assertEqual("--decode_capture_config 1,2,4" in args, bool(graph))
            self.assertEqual(env["RTP_QWEN35_DECODE_FUSION"], "0")
        with self.assertRaises(ValueError):
            smoke.server_config("0,1,2,3", decode_graph=1)

    def test_fp8_switches_control_both_environment_and_server_argument(self):
        for cache, native in ((0, 0), (1, 0), (1, 1)):
            env, args = smoke.server_config(
                "0,1,2,3", "baseline", "mega_moe_fp8", 1, cache, native
            )
            self.assertEqual(env["FP8_KV_CACHE"], str(cache))
            self.assertEqual(env["RTP_QWEN35_NATIVE_FP8_ATTN"], str(native))
            self.assertIn(f"--fp8_kv_cache {cache}", args)
            self.assertIn(f"--seq_size_per_block {4096 if cache else 2048}", args)
            self.assertIn("--kernel_seq_size_per_block 64", args)
            self.assertIn("--act_type BF16", args)
            self.assertIn("--enable_cuda_graph 1", args)
        with self.assertRaises(ValueError):
            smoke.server_config("0,1,2,3", fp8_kv_cache=0, native_fp8_attn=1)

    def test_fp8_shared_pool_rejects_undersized_allocation_block(self):
        with self.assertRaisesRegex(ValueError, "requires seq_size_per_block=4096"):
            smoke.server_config(
                "0,1,2,3", fp8_kv_cache=1, native_fp8_attn=1, seq_size_per_block=2048
            )
        _, args = smoke.server_config("0,1,2,3", seq_size_per_block=4096)
        self.assertIn("--seq_size_per_block 4096", args)
        self.assertIn("--fp8_kv_cache 0", args)

    def test_native_fp8_evidence_requires_dispatched_fp8_tensors_per_phase(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            logs = root / "fusion_logs"
            logs.mkdir()
            lines = []
            for rank in range(4):
                pid = 100 + rank
                lines.append(f"[process-{pid}] [rank: {rank}] initialize process_group")
                lines.append(
                    f"[process-{pid}] FP8_ATTN_EXECUTION phase=prefill native=1 q=torch.float8_e4m3fn kv=torch.float8_e4m3fn out=torch.bfloat16"
                )
                dtype = "bfloat16" if rank == 3 else "float8_e4m3fn"
                lines.append(
                    f"[process-{pid}] FP8_ATTN_EXECUTION phase=decode native=1 q=torch.{dtype} kv=torch.float8_e4m3fn out=torch.bfloat16"
                )
            (logs / "process.log").write_text("\n".join(lines))
            evidence = smoke.execution_evidence(root)["native_fp8"]
            self.assertTrue(all(evidence[str(r)]["prefill"] for r in range(4)))
            self.assertTrue(all(evidence[str(r)]["decode"] for r in range(3)))
            self.assertFalse(evidence["3"]["decode"])

    def test_mega_options_reach_worker_and_separated_pd(self):
        from types import SimpleNamespace

        a = SimpleNamespace(
            gpus="0,1,2,3",
            encoder_gpus="",
            cache_root="/tmp/cache",
            model_dir="/tmp/model",
            data_dir="/tmp/data",
            output="/tmp/out",
            profile="baseline",
            perf_repeats=2,
            bazel_option=[],
            moe_strategy="mega_moe_fp8",
            decode_graph=1,
            graph_edge_cases=1,
            scheduler_policy="prefill-first",
            throughput_sweep=1,
            steady_concurrency="348,512,640",
            steady_warmup_seconds=240,
            steady_window_seconds=300,
            steady_windows=2,
            fp8_kv_cache=1,
            native_fp8_attn=1,
            seq_size_per_block=4096,
        )
        command = smoke.bazel_command(a)
        for option in (
            "moe-strategy=mega_moe_fp8",
            "decode-graph=1",
            "graph-edge-cases=1",
            "scheduler-policy=prefill-first",
            "throughput-sweep=1",
            "steady-concurrency=348,512,640",
            "steady-warmup-seconds=240",
            "steady-window-seconds=300",
            "steady-windows=2",
            "fp8-kv-cache=1",
            "native-fp8-attn=1",
            "seq-size-per-block=4096",
        ):
            self.assertIn("--test_arg=--" + option, command)
        configs = smoke.separated_service_configs(
            "4,5,6,7",
            "0,1",
            "baseline",
            dict(encoder_0=20000, encoder_1=21000, fusion=22000),
            9,
            "mega_moe_fp8",
            1,
            1,
            1,
        )
        self.assertIn("--enable_cuda_graph 1", configs[-1]["args"])
        self.assertIn("--moe_strategy mega_moe_fp8", configs[-1]["args"])
        self.assertIn("--fp8_kv_cache 1", configs[-1]["args"])
        self.assertEqual(configs[-1]["env"]["RTP_QWEN35_NATIVE_FP8_ATTN"], "1")
        for encoder in configs[:2]:
            self.assertIn("--enable_cuda_graph 0", encoder["args"])

    def test_execution_evidence_requires_real_replay_on_every_rank(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            logs = root / "fusion_logs"
            logs.mkdir()
            lines = ["MegaMoE FP8 weights prepared during model construction"]
            for rank in range(4):
                lines += [
                    f"GRAPH_CAPTURE_READY rank={rank}",
                    f"MODEL_EXECUTION rank={rank} mode=eager prefill=1 batch=1 graph_bs=0 count=1",
                    f"MODEL_EXECUTION rank={rank} mode=graph prefill=0 batch=3 graph_bs=4 count=1",
                ]
            (root / "logs").mkdir()
            (logs / "process.log").write_text(
                "\n".join(
                    f"[process-{100 + rank}][rank: {rank}] initialize process_group\n"
                    f"[process-{100 + rank}] MegaMoE FP8 weights prepared during model construction"
                    for rank in range(4)
                )
            )
            path = root / "logs/engine.log"
            path.write_text("\n".join(lines[1:]))
            evidence = smoke.execution_evidence(root)
            self.assertEqual(smoke.validate_execution_evidence(evidence, 1, 1), [])
            prefill_graph = dict(
                evidence, records=evidence["records"] + [dict(mode="graph", prefill=1)]
            )
            self.assertIn(
                "Prefill unexpectedly used graph execution",
                smoke.validate_execution_evidence(prefill_graph, 1, 1),
            )
            with path.open("a") as f:
                f.write(
                    "\nMODEL_EXECUTION rank=0 mode=eager prefill=0 batch=1 graph_bs=0 count=1"
                )
            self.assertIn(
                "Decode silently used eager execution in graph validation",
                smoke.validate_execution_evidence(smoke.execution_evidence(root), 1, 1),
            )
            path.write_text("\n".join(lines[:-1]))
            errors = smoke.validate_execution_evidence(
                smoke.execution_evidence(root), 1, 1
            )
            self.assertIn("rank 3: missing replay evidence", errors)
            # Capture alone never establishes actual request-time graph use.
            path.write_text(
                "\n".join(line for line in lines if "mode=graph" not in line)
            )
            self.assertTrue(
                smoke.validate_execution_evidence(smoke.execution_evidence(root), 1, 1)
            )

    def test_fallback_evidence_excludes_metric_registration_warnings(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            logs = root / "fusion_logs"
            logs.mkdir()
            actual = [
                "decode cuda graph: batch size 5 exceeds max captured 4, fallback to normal run",
                "decode cuda graph: capture_range_ is empty, cannot run",
            ]
            (logs / "process.log").write_text(
                "\n".join(
                    ["no metric named VIT_CUDA_GRAPH_FALLBACK_QPS_METRIC", *actual]
                )
            )
            self.assertEqual(smoke.execution_evidence(root)["fallbacks"], actual)

    def test_response_requires_fusion_and_complete_output(self):
        expected = {"prompt_tokens": 123, "video_tokens": 20240}
        result = dict(
            http_status=200,
            content="video description",
            done=True,
            finish_reason="stop",
            usage=dict(
                prompt_tokens=123, prompt_tokens_details=dict(video_tokens=20240)
            ),
            aux_info=dict(pd_sep=False, reuse_len=0),
        )
        self.assertEqual(smoke.validate_response(result, expected), [])
        result.pop("done")
        self.assertEqual(smoke.validate_response(result, expected), [])
        for patch in (
            dict(finish_reason=None),
            dict(finish_reason="length"),
            dict(content=""),
            dict(aux_info={}),
            dict(aux_info=dict(pd_sep=True, reuse_len=0)),
            dict(usage={}),
        ):
            self.assertTrue(smoke.validate_response(dict(result, **patch), expected))

    def test_multi_choice_stream_requires_each_completion(self):
        import sys
        from types import SimpleNamespace

        expected = dict(prompt_tokens=123, video_tokens=0)

        def run(missing=False, truncated=False):
            rows = []
            for i in range(3 if not missing else 2):
                rows.append(
                    dict(choices=[dict(index=i, delta=dict(content=f"answer-{i}"))])
                )
            for i in range(3 if not missing else 2):
                rows.append(
                    dict(
                        choices=[
                            dict(
                                index=i,
                                delta={},
                                finish_reason=(
                                    "length" if truncated and i == 1 else "stop"
                                ),
                            )
                        ]
                    )
                )
            rows.append(
                dict(
                    choices=[],
                    usage=dict(prompt_tokens=123, completion_tokens=9),
                    aux_info=dict(pd_sep=False, reuse_len=0),
                )
            )
            response = mock.MagicMock(status_code=200)
            response.__enter__.return_value = response
            response.iter_lines.return_value = [
                b"data: " + smoke.json.dumps(row).encode() for row in rows
            ]
            fake = SimpleNamespace(post=mock.Mock(return_value=response))
            with mock.patch.dict(sys.modules, requests=fake):
                return smoke.request("http://test", dict(n=3), 0, expected)

        result = run()
        self.assertTrue(result["ok"], result)
        self.assertEqual(
            [c["content"] for c in result["choices"].values()],
            ["answer-0", "answer-1", "answer-2"],
        )
        self.assertFalse(run(missing=True)["ok"])
        self.assertFalse(run(truncated=True)["ok"])

    def test_selected_gpu_check_ignores_other_cards(self):
        with mock.patch.object(
            smoke.subprocess,
            "check_output",
            side_effect=["4, 1, 0\n5, 1, 0\n6, 1, 0\n7, 1, 0\n", ""],
        ) as call:
            smoke.check_gpu_idle("4,5,6,7")
            for args in call.call_args_list:
                self.assertIn("--id=4,5,6,7", args.args[0])
        with mock.patch.object(
            smoke.subprocess,
            "check_output",
            return_value="4, 2000, 0\n5, 1, 0\n6, 1, 0\n7, 1, 0\n",
        ):
            with self.assertRaises(RuntimeError):
                smoke.check_gpu_idle("4,5,6,7")

    def test_profiles_preserve_fusion_topology(self):
        for profile in ("baseline", "fused", "flashinfer"):
            env, args = smoke.server_config("0,1,2,3", profile)
            self.assertEqual(
                env["RTP_QWEN35_DECODE_FUSION"], "0" if profile == "baseline" else "1"
            )
            self.assertEqual(
                env["RTP_QWEN35_FUSED_CONV_QKV_NORM"],
                "0" if profile == "baseline" else "1",
            )
            self.assertEqual(
                env["RTP_QWEN35_GDN_DECODE_BACKEND"],
                "flashinfer" if profile == "flashinfer" else "native",
            )
            self.assertEqual(env["ROLE_TYPE"], "PDFUSION")
            self.assertIn("--enable_cuda_graph 0", args)
            self.assertIn("--moe_strategy fp8_per_block_ep_normal", args)

    def test_performance_excludes_warmup(self):
        measured = dict(
            phase="single_video_1",
            ok=True,
            ttft_s=2,
            e2e_s=6,
            tpot_ms=1000,
            usage=dict(completion_tokens=5),
        )
        warm = dict(measured, phase="warmup", ttft_s=999)
        report = dict(
            requests=[warm, measured],
            batches=[
                dict(phase="warmup", wall_s=999),
                dict(phase="single_video_1", wall_s=6),
            ],
        )
        summary = smoke.performance_summary(report)["single_video"]
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["ttft_s"]["median"], 2)
        self.assertAlmostEqual(summary["output_tokens_per_s"], 5 / 6)

    def test_e2pd4_gpu_assignment_and_route(self):
        self.assertEqual(smoke.selected_gpus("4,5,6,7", "0,1"), "0,1,4,5,6,7")
        self.assertEqual(smoke.selected_gpus("4,5,6,7", "0"), "0,4,5,6,7")
        for encoder in ("0,1,2", "0,0", "0,4", "-1", "0,"):
            with self.assertRaises(ValueError):
                smoke.selected_gpus("4,5,6,7", encoder)
        ports = dict(encoder_0=20000, encoder_1=21000, fusion=22000)
        configs = smoke.separated_service_configs(
            "4,5,6,7", "0,1", "baseline", ports, 9
        )
        self.assertEqual([c["gpus"] for c in configs], [[0], [1], [4, 5, 6, 7]])
        for encoder in configs[:2]:
            self.assertEqual(encoder["env"]["ROLE_TYPE"], "VIT")
            self.assertEqual(encoder["env"]["WORLD_SIZE"], "1")
            self.assertIn("--vit_separation 1", encoder["args"])
        pd = configs[2]
        self.assertEqual(pd["env"]["ROLE_TYPE"], "PDFUSION")
        self.assertEqual(pd["env"]["VIT_SEPARATION"], "2")
        self.assertEqual(pd["env"]["WORLD_SIZE"], "4")
        self.assertNotIn("REMOTE_SERVER_PORT", pd["env"])
        group = smoke.json.loads(pd["env"]["MODEL_SERVICE_CONFIG"])["role_endpoints"][0]
        self.assertEqual(
            group["vit_endpoint"]["address"], "127.0.0.1:20000,127.0.0.1:21000"
        )
        self.assertEqual(
            group["pd_fusion_endpoint"]["address"],
            "127.0.0.1:22000,127.0.0.1:22009,127.0.0.1:22018,127.0.0.1:22027",
        )
        self.assertNotIn("prefill_endpoint", group)
        self.assertNotIn("decode_endpoint", group)

    def test_e2pd4_bazel_selection(self):
        from types import SimpleNamespace

        a = SimpleNamespace(
            gpus="4,5,6,7",
            encoder_gpus="0,1",
            cache_root="/tmp/bazel",
            model_dir="/tmp/model",
            data_dir="/tmp/data",
            output="/tmp/out",
            profile="baseline",
            perf_repeats=2,
            bazel_option=[],
        )
        command = smoke.bazel_command(a)
        self.assertIn(smoke.ENCODER_TARGET, command)
        self.assertIn("--test_env=GPU_COUNT=6", command)
        self.assertIn("--test_env=CUDA_VISIBLE_DEVICES=0,1,4,5,6,7", command)
        self.assertIn("--test_arg=--gpus=4,5,6,7", command)

    def test_encoder_evidence_requires_success_on_both_services(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(2):
                logs = root / f"encoder_{i}_logs"
                logs.mkdir()
                row = dict(
                    request_id=i,
                    exception=None,
                    response="MMEmbeddingRes(embeddings_shape=[20424,4096])",
                )
                (logs / "mm_access_r0_s0.log").write_text(smoke.json.dumps(row) + "\n")
            self.assertTrue(all(smoke.encoder_log_evidence(root).values()))
            (root / "encoder_1_logs/mm_access_r0_s0.log").write_text(
                '{"exception": "failure", "response": null}\n'
            )
            self.assertFalse(all(smoke.encoder_log_evidence(root).values()))

    def test_encoder_start_failure_cleans_all_owned_services(self):
        import sys
        from types import ModuleType, SimpleNamespace

        module = ModuleType("rtp_llm.test.utils.maga_server_manager")
        config_module = ModuleType("rtp_llm.config.py_config_modules")
        config_module.MIN_WORKER_INFO_PORT_NUM = 9
        managers = [mock.Mock(port=20000 + i * 1000, exit_code=None) for i in range(3)]
        for i, m in enumerate(managers):
            m.start_server.return_value = i != 0
        module.MagaServerManager = mock.Mock(side_effect=managers)
        module.MagaServerManager.get_free_port.side_effect = [20000, 21000, 22000]
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(
                output=d,
                gpus="4,5,6,7",
                encoder_gpus="0,1",
                data_dir=str(PATH.parent / "qwen35_e2p4d2_data"),
                model_dir=d,
            )
            with mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,4,5,6,7"}
            ), mock.patch.dict(
                sys.modules,
                {module.__name__: module, config_module.__name__: config_module},
            ), mock.patch.object(
                smoke, "check_gpu_idle", return_value={}
            ), mock.patch.object(
                smoke, "expected_tokens", return_value={}
            ), mock.patch.object(
                smoke.threading.Thread, "start"
            ), mock.patch.object(
                smoke.threading.Thread, "join"
            ), mock.patch.object(
                smoke.concurrent.futures, "ThreadPoolExecutor"
            ) as pool:
                pool.return_value.__enter__.return_value.map.side_effect = (
                    lambda fn, items: list(map(fn, items))
                )
                self.assertEqual(smoke.run_worker(args), 1)
            for m in managers:
                m.stop_server.assert_called_once()
            result = smoke.json.loads(Path(d, "result.json").read_text())
            self.assertEqual(
                result["readiness"], dict(encoder_0=False, encoder_1=True, fusion=True)
            )

    def test_text_response_has_no_video_usage(self):
        expected = dict(prompt_tokens=24601, video_tokens=0)
        row = dict(
            http_status=200,
            content="report",
            finish_reason="stop",
            usage=dict(prompt_tokens=24601, prompt_tokens_details=None),
            aux_info=dict(pd_sep=False, reuse_len=0),
        )
        self.assertEqual(smoke.validate_response(row, expected), [])
        row["usage"]["prompt_tokens_details"] = dict(video_tokens=1)
        self.assertIn("video token mismatch", smoke.validate_response(row, expected))

    def test_text_bazel_target_preserves_four_gpu_selection(self):
        from types import SimpleNamespace

        a = SimpleNamespace(
            gpus="4,5,6,7",
            encoder_gpus="",
            workload="text",
            cache_root="/tmp/bazel",
            model_dir="/tmp/model",
            data_dir="/tmp/data",
            output="/tmp/out",
            profile="baseline",
            perf_repeats=2,
            bazel_option=[],
        )
        command = smoke.bazel_command(a)
        self.assertIn(smoke.TEXT_TARGET, command)
        self.assertIn("--test_env=CUDA_VISIBLE_DEVICES=4,5,6,7", command)
        self.assertIn("--test_env=GPU_COUNT=4", command)
        self.assertIn("--test_arg=--workload=text", command)

    def test_passive_gpu_lock_never_signals_compute_processes(self):
        spec = importlib.util.spec_from_file_location(
            "device_resource", PATH.parent.parent / "utils/device_resource.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        resource = object.__new__(mod.DeviceResource)
        resource.passive_lock = True
        resource.gpu_ids = ["4", "5", "6", "7"]
        with mock.patch.object(
            mod, "_nvidia_smi", return_value="nvidia-smi"
        ), mock.patch.object(mod.os, "kill") as kill, mock.patch.object(
            resource, "_get_gpu_pids"
        ) as query:
            for pids in ([123], None):
                query.return_value = pids
                self.assertFalse(resource._ensure_gpus_released())
                self.assertTrue(resource._has_zombie_gpu_contexts("4"))
            query.return_value = []
            self.assertTrue(resource._ensure_gpus_released())
            self.assertFalse(resource._has_zombie_gpu_contexts("4"))
            kill.assert_not_called()
        resource.query_timeout = 60
        with mock.patch.object(
            mod.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="")
        ) as run:
            self.assertEqual(resource._get_gpu_pids("4"), [])
            self.assertEqual(run.call_args.kwargs["timeout"], 60)

    def test_fixture_integrity(self):
        data = PATH.parent / "qwen35_e2p4d2_data"
        smoke.validate_fixture(data)
        request = smoke.payload_for(data)
        self.assertEqual(request["max_tokens"], 4096)
        self.assertEqual(
            request["messages"][0]["content"][0]["preprocess_config"]["fps"], 6
        )
        with tempfile.TemporaryDirectory() as d:
            from shutil import copy

            copy(data / "manifest.json", d)
            for name in ("video.mp4", "messages.json"):
                Path(d, name).write_text("bad")
            with self.assertRaises(ValueError):
                smoke.validate_fixture(Path(d))

    def test_foreign_gpu_use_after_lock_stops_before_model(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(output=d, gpus="4,5,6,7", data_dir=d, model_dir=d)
            with mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": args.gpus}
            ), mock.patch.object(
                smoke, "check_gpu_idle", side_effect=RuntimeError("foreign GPU use")
            ), mock.patch.object(
                smoke, "expected_tokens"
            ) as tokens, mock.patch.object(
                smoke.threading.Thread, "start"
            ), mock.patch.object(
                smoke.threading.Thread, "join"
            ):
                self.assertEqual(smoke.run_worker(args), 1)
                tokens.assert_not_called()
            result = smoke.json.loads(Path(d, "result.json").read_text())
            self.assertEqual(result["phase"], "gpu_availability_after_lock")
            self.assertEqual(result["requests"], [])

    def test_worker_failure_is_persisted(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(output=d, gpus="4,5,6,7", data_dir=d, model_dir=d)
            with mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": args.gpus}
            ), mock.patch.object(
                smoke, "check_gpu_idle", return_value={}
            ), mock.patch.object(
                smoke.threading.Thread, "start"
            ), mock.patch.object(
                smoke.threading.Thread, "join"
            ):
                self.assertEqual(smoke.run_worker(args), 1)
            self.assertEqual(
                smoke.json.loads(Path(d, "result.json").read_text())["status"], "FAILED"
            )

    def test_startup_failure_stops_only_owned_manager(self):
        import sys
        from types import ModuleType, SimpleNamespace

        module = ModuleType("rtp_llm.test.utils.maga_server_manager")
        manager = mock.Mock()
        manager.port = 12345
        manager.exit_code = 1
        manager.start_server.return_value = False
        module.MagaServerManager = mock.Mock(return_value=manager)
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(
                output=d,
                gpus="4,5,6,7",
                data_dir=str(PATH.parent / "qwen35_e2p4d2_data"),
                model_dir=d,
            )
            with mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": args.gpus}
            ), mock.patch.object(
                smoke, "check_gpu_idle", return_value={}
            ), mock.patch.dict(
                sys.modules, {module.__name__: module}
            ), mock.patch.object(
                smoke, "expected_tokens", return_value={}
            ), mock.patch.object(
                smoke.threading.Thread, "start"
            ), mock.patch.object(
                smoke.threading.Thread, "join"
            ):
                self.assertEqual(smoke.run_worker(args), 1)
            manager.stop_server.assert_called_once()
            result = smoke.json.loads(Path(d, "result.json").read_text())
            self.assertEqual(result["phase"], "model_initialization")


class MeasurementTest(unittest.TestCase):
    def setUp(self):
        self.options = dict(smoke.CLIENT_OPTIONS)

    def tearDown(self):
        smoke.CLIENT_OPTIONS.clear()
        smoke.CLIENT_OPTIONS.update(self.options)

    def test_fixed_output_rejects_early_stop_and_checks_backend_count(self):
        payload, expected = smoke.fixed_output_payload(
            {}, dict(prompt_tokens=24601, video_tokens=0), 1024
        )
        self.assertEqual(payload["extra_configs"]["min_new_tokens"], 1024)
        response = dict(
            http_status=200,
            content="text",
            finish_reason="length",
            usage=dict(prompt_tokens=24601, completion_tokens=1024),
            aux_info=dict(pd_sep=False, reuse_len=0, output_len=1024),
        )
        self.assertEqual(smoke.validate_response(response, expected), [])
        response["usage"]["completion_tokens"] = 1000
        self.assertIn(
            "fixed output token mismatch", smoke.validate_response(response, expected)
        )
        response["usage"]["completion_tokens"] = 1024
        response["aux_info"]["output_len"] = 1000
        self.assertIn(
            "fixed backend output token mismatch",
            smoke.validate_response(response, expected),
        )
        with self.assertRaises(ValueError):
            smoke.fixed_output_payload({}, expected, 9000)

    def test_measured_two_rank_lanes_drain_journal_without_losing_rows(self):
        smoke.CLIENT_OPTIONS.update(mode="measured", rank_count=2)

        def response(url, payload, index, expected):
            time.sleep(0.01)
            return dict(
                index=index,
                ok=True,
                e2e_s=0.01,
                ttft_s=0.001,
                usage=dict(completion_tokens=10),
            )

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.config.py_config_modules": SimpleNamespace(
                        MIN_WORKER_INFO_PORT_NUM=9
                    )
                },
            ), mock.patch.object(smoke, "request", side_effect=response):
                rows, summary = smoke.run_steady_load(
                    SimpleNamespace(port=20000), {}, {}, 4, 0, 1, 1, out, "test"
                )
            saved = [
                json.loads(line)
                for line in (out / "test-requests.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), len(saved))
            self.assertEqual({r["index"] for r in rows}, {r["index"] for r in saved})
            self.assertEqual(summary["peak_in_flight"], 4)
            self.assertEqual(
                set(summary["measurement"]["completed_by_rank"]), {"0", "1"}
            )
            self.assertEqual({r["client_rank"] for r in rows}, {0, 1})
            self.assertTrue(all(r["started_s"] < 1 for r in saved))
            self.assertGreater(summary["drain_completions"], 0)
            self.assertTrue(all("journal_queue_s" in r for r in saved))

    def test_measured_output_failure_keeps_refilling_without_changing_failed_row(self):
        smoke.CLIENT_OPTIONS.update(
            mode="measured", rank_count=2, continue_output_errors=True
        )

        def response(url, payload, index, expected):
            time.sleep(0.01)
            row = dict(
                index=index,
                ok=True,
                http_status=200,
                finish_reason="stop",
                content="answer",
                usage=dict(completion_tokens=10),
                e2e_s=0.01,
            )
            if index == 0:
                row.update(
                    ok=False,
                    content="",
                    reasoning_content="thinking",
                    finish_reason="length",
                    validation_errors=["incomplete, empty, or truncated generation"],
                )
            return row

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules,
            {
                "rtp_llm.config.py_config_modules": SimpleNamespace(
                    MIN_WORKER_INFO_PORT_NUM=9
                )
            },
        ), mock.patch.object(smoke, "request", side_effect=response):
            rows, summary = smoke.run_steady_load(
                SimpleNamespace(port=20000), {}, {}, 4, 0, 1, 1, Path(directory), "test"
            )
        self.assertGreater(len(rows), 4)
        self.assertFalse(summary["aborted"])
        self.assertFalse(summary["all_valid"])
        self.assertEqual(summary["measurement"]["output_quality_errors"], 1)
        self.assertFalse(next(r for r in rows if r["index"] == 0)["ok"])

    def test_journal_failure_is_fatal(self):
        smoke.CLIENT_OPTIONS["mode"] = "measured"
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "test-requests.jsonl").write_text("preserve")
            with mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.config.py_config_modules": SimpleNamespace(
                        MIN_WORKER_INFO_PORT_NUM=9
                    )
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "journal failed"):
                    smoke.run_steady_load(
                        SimpleNamespace(port=1), {}, {}, 4, 0, 1, 1, out, "test"
                    )
            self.assertEqual((out / "test-requests.jsonl").read_text(), "preserve")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fail = False

    def log_message(self, *args):
        pass

    def do_POST(self):
        json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(0.03)
        event = dict(
            choices=[
                dict(index=0, delta=dict(content="test output"), finish_reason="length")
            ],
            usage=dict(prompt_tokens=24601, completion_tokens=4, total_tokens=24605),
            aux_info=dict(pd_sep=False, reuse_len=0, output_len=4),
        )
        self.send_response(500 if self.fail else 200)
        self.send_header("Connection", "close")
        self.close_connection = True
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        body = ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
        self.wfile.write(f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n")
        self.wfile.flush()


def fail_one_client(port, options, *args):
    if options["index_offset"] == 0:
        raise RuntimeError("injected client startup failure")
    return smoke._client_process_entry(port, options, *args)


class ProcessLoadTest(unittest.TestCase):
    def setUp(self):
        self.options = dict(smoke.CLIENT_OPTIONS)

    def tearDown(self):
        smoke.CLIENT_OPTIONS.clear()
        smoke.CLIENT_OPTIONS.update(self.options)

    @classmethod
    def setUpClass(cls):
        for base in range(18800, 19000, 4):
            servers = []
            try:
                for rank in range(4):
                    servers.append(
                        http.server.ThreadingHTTPServer(
                            ("127.0.0.1", base + rank), Handler
                        )
                    )
            except OSError:
                for server in servers:
                    server.server_close()
                continue
            cls.servers = servers
            cls.port = base
            break
        else:
            raise RuntimeError("no free four-port test range")
        cls.threads = [
            threading.Thread(target=s.serve_forever, daemon=True) for s in cls.servers
        ]
        for thread in cls.threads:
            thread.start()

    @classmethod
    def tearDownClass(cls):
        for server in cls.servers:
            server.shutdown()
            server.server_close()
        for thread in cls.threads:
            thread.join()

    def run_load(self, out, rank_count=4):
        smoke.CLIENT_OPTIONS.update(
            mode="measured",
            rank_count=rank_count,
            processes=rank_count,
            worker_stride=1,
            chunk_size=4096,
            profile_requests=0,
            profile_dir=str(out),
        )
        return smoke.run_steady_load(
            SimpleNamespace(port=self.port),
            {},
            dict(prompt_tokens=24601, video_tokens=0, fixed_output_tokens=4),
            8,
            0,
            1,
            1,
            out,
            "steady_c8",
        )

    def test_shared_clock_unique_indices_rank_mapping_and_drain(self):
        Handler.fail = False
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            rows, summary = self.run_load(out)
            self.assertTrue(summary["all_valid"])
            self.assertFalse(summary["aborted"])
            self.assertEqual(summary["client_processes"], 4)
            self.assertGreater(summary["drain_completions"], 0)
            self.assertEqual(len({r["index"] for r in rows}), len(rows))
            self.assertEqual({r["client_lane"] for r in rows}, set(range(8)))
            self.assertTrue(all(r["client_rank"] == r["client_lane"] % 4 for r in rows))
            summaries = [
                json.loads(p.read_text())
                for p in out.glob("steady_c8-processes/*/steady_c8-summary.json")
            ]
            self.assertEqual(
                {s["started_at"] for s in summaries}, {summary["started_at"]}
            )
            self.assertEqual(len(rows), sum(s["total_requests"] for s in summaries))
            saved = [
                json.loads(line)
                for line in (out / "steady_c8-requests.jsonl").read_text().splitlines()
            ]
            self.assertEqual(saved, rows)
            self.assertTrue(all(r["started_s"] < 1 for r in rows))
            for pid in summary["client_pids"]:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_two_rank_processes_use_only_two_ports_and_preserve_concurrency(self):
        Handler.fail = False
        with tempfile.TemporaryDirectory() as directory:
            rows, summary = self.run_load(Path(directory), rank_count=2)
            self.assertTrue(summary["all_valid"])
            self.assertEqual(summary["client_processes"], 2)
            self.assertEqual(summary["peak_in_flight"], 8)
            self.assertEqual(
                set(summary["measurement"]["completed_by_rank"]), {"0", "1"}
            )
            self.assertEqual({r["client_lane"] for r in rows}, set(range(8)))
            self.assertTrue(all(r["client_rank"] == r["client_lane"] % 2 for r in rows))

    def test_response_error_aborts_all_processes_and_preserves_rows(self):
        Handler.fail = True
        try:
            with tempfile.TemporaryDirectory() as directory:
                rows, summary = self.run_load(Path(directory))
                self.assertTrue(summary["aborted"])
                self.assertFalse(summary["all_valid"])
                self.assertTrue(rows)
                self.assertTrue(all(not r["ok"] for r in rows))
                self.assertLessEqual(len(rows), 8)
        finally:
            Handler.fail = False

    def test_startup_crash_releases_other_processes_from_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            start = time.monotonic()
            with mock.patch.object(smoke, "_client_process_entry", fail_one_client):
                with self.assertRaisesRegex(RuntimeError, "load client process failed"):
                    self.run_load(out)
            self.assertLess(time.monotonic() - start, 10)
            pids = json.loads((out / "steady_c8-processes/processes.json").read_text())[
                "pids"
            ]
            for pid in pids:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)


class StreamingBufferTest(unittest.TestCase):
    def test_actual_parser_preserves_split_utf8_and_parses_before_done(self):
        import requests

        text = "中🧪文" * 3000
        parsed = threading.Event()
        observations = []
        first = (
            "data: "
            + json.dumps(
                dict(
                    choices=[
                        dict(index=0, delta=dict(content=text), finish_reason=None)
                    ]
                ),
                ensure_ascii=False,
            )
            + "\n\n"
        ).encode()
        last = (
            "data: "
            + json.dumps(
                dict(
                    choices=[dict(index=0, delta={}, finish_reason="length")],
                    usage=dict(
                        prompt_tokens=24601, completion_tokens=4, total_tokens=24605
                    ),
                    aux_info=dict(pd_sep=False, reuse_len=0, output_len=4),
                )
            )
            + "\n\ndata: [DONE]\n\n"
        ).encode()
        split_utf8 = first.index("🧪".encode()) + 1

        class StreamingHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
                try:
                    for begin, end in (
                        (0, split_utf8),
                        (split_utf8, 4094),
                        (4094, len(first) - 1),
                        (len(first) - 1, len(first)),
                    ):
                        piece = first[begin:end]
                        self.wfile.write(
                            f"{len(piece):x}\r\n".encode() + piece + b"\r\n"
                        )
                        self.wfile.flush()
                    # No [DONE] until the actual request parser decoded the
                    # first JSON event. A buffering regression times out.
                    observations.append(parsed.wait(timeout=3))
                    self.wfile.write(
                        f"{len(last):x}\r\n".encode() + last + b"\r\n0\r\n\r\n"
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        original_loads, original_post = json.loads, requests.post

        def observe(value, *args, **kwargs):
            decoded = original_loads(value, *args, **kwargs)
            if decoded.get("choices", [{}])[0].get("delta", {}).get("content") == text:
                parsed.set()
            return decoded

        def bounded_post(*args, **kwargs):
            kwargs["timeout"] = 2
            return original_post(*args, **kwargs)

        options = dict(smoke.CLIENT_OPTIONS)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StreamingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for chunk in (1, 4096):
                parsed.clear()
                smoke.CLIENT_OPTIONS.update(
                    mode="measured", chunk_size=chunk, profile_requests=0
                )
                with mock.patch.object(
                    json, "loads", side_effect=observe
                ), mock.patch.object(requests, "post", side_effect=bounded_post):
                    result = smoke.request(
                        f"http://127.0.0.1:{server.server_port}/",
                        {},
                        0,
                        dict(
                            prompt_tokens=24601, video_tokens=0, fixed_output_tokens=4
                        ),
                    )
                self.assertTrue(result["ok"], result.get("validation_errors"))
                self.assertEqual(result["content"], text)
                self.assertTrue(parsed.is_set())
            self.assertEqual(observations, [True, True])
        finally:
            smoke.CLIENT_OPTIONS.clear()
            smoke.CLIENT_OPTIONS.update(options)
            server.shutdown()
            server.server_close()
            thread.join()


class LongBudgetTest(unittest.TestCase):
    def args(self):
        return [
            "smoke",
            "--output",
            "/tmp/not-launched-budget-test",
            "--workload",
            "text",
            "--gpus",
            "4,5,6,7",
            "--moe-strategy",
            "mega_moe_fp8",
            "--decode-graph",
            "1",
            "--fp8-kv-cache",
            "1",
            "--native-fp8-attn",
            "1",
            "--seq-size-per-block",
            "4096",
            "--rank-concurrency",
            "256",
            "--graph-batches",
            "1,2,4,8,16,32,64,96,99,128,160,192,196,224,256",
            "--steady-concurrency",
            "768,784",
            "--steady-warmup-seconds",
            "240",
            "--steady-window-seconds",
            "900",
            "--steady-windows",
            "4",
            "--client-mode",
            "measured",
            "--client-processes",
            "4",
        ]

    def test_old_budget_rejects_long_plan_before_launch(self):
        with mock.patch.object(sys, "argv", self.args()), mock.patch.object(
            smoke, "run_launcher"
        ) as launch:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(
                SystemExit
            ) as error:
                smoke.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("exceeds the configured test budget", stderr.getvalue())
            launch.assert_not_called()

    def test_explicit_budget_reaches_bazel_and_worker_with_graph_fp8_preserved(self):
        with mock.patch.object(
            sys, "argv", self.args() + ["--test-timeout-seconds", "14400"]
        ), mock.patch.object(smoke, "run_launcher", side_effect=smoke.bazel_command):
            command = smoke.main()
        self.assertIn("--test_timeout=14400", command)
        self.assertNotIn("--test_timeout=7200", command)
        for name, value in [
            ("test-timeout-seconds", "14400"),
            ("decode-graph", "1"),
            ("fp8-kv-cache", "1"),
            ("native-fp8-attn", "1"),
            ("seq-size-per-block", "4096"),
            ("steady-window-seconds", "900"),
        ]:
            self.assertIn("--test_arg=--" + name + "=" + value, command)


class EncoderTopologyTest(unittest.TestCase):
    def test_bazel_target_resource_declarations_match_worker_args(self):
        targets = []
        scope = dict(
            load=lambda *a: None,
            SMOKE_FRAMEWORK_DEPS=[],
            native=SimpleNamespace(py_test=lambda **kw: targets.append(kw)),
        )
        exec(PATH.with_suffix(".bzl").read_text(), scope)
        scope["qwen35_epd_fusion_4gpu_suite"]()
        for count in (1, 2):
            target = next(
                t for t in targets if t["name"] == f"qwen35_e{count}pd4_smoke"
            )
            self.assertEqual(target["exec_properties"]["gpu_count"], str(count + 4))
            self.assertEqual(target["env"]["GPU_COUNT"], str(count + 4))
            self.assertEqual(target["env"]["WORLD_SIZE"], str(count + 4))
            self.assertIn(
                "--encoder-gpus=" + ("0" if count == 1 else "0,1"), target["args"]
            )

    def test_proxy_configs_and_five_six_gpu_resources(self):
        ports = dict(encoder_0=20000, encoder_1=21000, fusion=22000)
        env, args = smoke.server_config("4,5,6,7", "baseline", "mega_moe_fp8", 1)
        for ids, count in [("0", 1), ("0,1", 2)]:
            configs = smoke.dp2_service_configs(
                "4,5,6,7", ids, env, args, ports, 9, "candidate-a", 137438953472
            )
            enc, pd = configs
            self.assertEqual(enc["name"], f"encoder_dp{count}")
            self.assertEqual(enc["gpus"], list(range(count)))
            for field in ("WORLD_SIZE", "TP_SIZE", "DP_SIZE", "EP_SIZE"):
                self.assertEqual(enc["env"][field], "1")
            for field in ("GPU_COUNT", "VIT_SERVER_COUNT"):
                self.assertEqual(enc["env"][field], str(count))
            self.assertIn(f"--vit_server_count {count}", enc["args"])
            self.assertEqual(
                enc["env"].get("RTP_VIT_SINGLE_WORKER_PROXY"),
                "1" if count == 1 else None,
            )
            self.assertEqual(enc["env"]["CUDA_VISIBLE_DEVICES"], ids)
            self.assertEqual(pd["env"]["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
            route = json.loads(pd["env"]["MODEL_SERVICE_CONFIG"])["role_endpoints"][0]
            self.assertEqual(route["vit_endpoint"]["address"], "127.0.0.1:20000")
            for field in ("MM_CACHE_CPU_MAX_BYTES", "MM_CACHE_GPU_MAX_BYTES"):
                self.assertEqual(enc["env"][field], "0")
            a = SimpleNamespace(
                gpus="4,5,6,7",
                encoder_gpus=ids,
                encoder_dp2=1,
                cache_root="/tmp/cache",
                model_dir="/tmp/model",
                data_dir="/tmp/data",
                output="/tmp/out",
                profile="baseline",
                perf_repeats=0,
                bazel_option=[],
            )
            command = smoke.bazel_command(a)
            self.assertIn(
                smoke.SINGLE_ENCODER_TARGET if count == 1 else smoke.ENCODER_TARGET,
                command,
            )
            self.assertIn(f"--test_env=GPU_COUNT={count+4}", command)
            self.assertIn(f"--test_env=WORLD_SIZE={count+4}", command)
            self.assertIn(f"--test_arg=--encoder-gpus={ids}", command)
            self.assertIn("--run_under=//rtp_llm/test/utils:gpu_lock", command)
        with self.assertRaises(ValueError):
            smoke.encoder_gpu_ids("")

    def test_worker_evidence_does_not_require_absent_worker_or_accept_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            logs = root / "encoder_dp1_logs"
            logs.mkdir()
            access = logs / "mm_access_r0_s0.log"
            access.write_text(
                json.dumps(
                    dict(exception=None, response="MMEmbeddingRes(shape=[20424,4096])")
                )
                + "\n"
            )
            evidence = smoke.encoder_log_evidence(root, 1)
            self.assertEqual(set(evidence), {"encoder_0"})
            self.assertTrue(all(evidence.values()))
            access.write_text(
                json.dumps(dict(exception="failed", response="MMEmbeddingRes")) + "\n"
            )
            self.assertFalse(all(smoke.encoder_log_evidence(root, 1).values()))

    def test_single_proxy_steady_cli_alias(self):
        argv = [
            "smoke",
            "--output",
            "/tmp/not-launched-encoder-test",
            "--gpus",
            "4,5,6,7",
            "--encoder-gpus",
            "0",
            "--encoder-proxy",
            "1",
            "--steady-concurrency",
            "512",
            "--rank-concurrency",
            "256",
            "--client-mode",
            "measured",
            "--client-processes",
            "4",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            smoke, "run_launcher", side_effect=smoke.bazel_command
        ):
            command = smoke.main()
        self.assertIn(smoke.SINGLE_ENCODER_TARGET, command)
        self.assertIn("--test_arg=--encoder-dp2=1", command)

    def test_real_spawn_function_keeps_single_proxy_and_worker_route(self):
        # Execute the actual launcher function with process creation mocked, avoiding
        # model/CUDA imports while checking worker arguments and proxy endpoints.
        import ast
        import logging

        start_path = PATH.parents[2] / "start_server.py"
        tree = ast.parse(start_path.read_text())
        func = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "start_vit_server_impl"
        )
        func.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
        process = mock.Mock()
        scope = dict(
            PyEnvConfigs=object,
            ProcessManager=object,
            logging=logging,
            os=os,
            load_gpu_nic_affinity=mock.Mock(),
            torch=SimpleNamespace(multiprocessing=SimpleNamespace(Process=process)),
        )
        exec(compile(module, str(start_path), "exec"), scope)
        vit = mock.Mock()
        proxy = mock.Mock()
        modules = {
            "rtp_llm.multimodal.vit_start_server": SimpleNamespace(
                vit_start_server=vit
            ),
            "rtp_llm.multimodal.vit_proxy_start_server": SimpleNamespace(
                vit_proxy_start_server=proxy
            ),
        }
        for count, enabled, expected in [(1, "0", 1), (1, "1", 2), (2, "0", 3)]:
            process.reset_mock()
            config = SimpleNamespace(
                server_config=SimpleNamespace(
                    start_port=20000,
                    vit_server_count=count,
                    rpc_server_port=20001,
                    server_port=20000,
                ),
                vit_config=SimpleNamespace(
                    output_transport=SimpleNamespace(rdma=SimpleNamespace(port=22000))
                ),
            )
            with mock.patch.dict(sys.modules, modules), mock.patch.dict(
                os.environ, {"RTP_VIT_SINGLE_WORKER_PROXY": enabled}
            ):
                result = scope["start_vit_server_impl"](config)
            self.assertEqual(len(result), expected)
            self.assertEqual(process.call_count, expected)
            if expected > 1:
                calls = process.call_args_list
                self.assertEqual(
                    [c.kwargs["args"][0] for c in calls[:-1]], list(range(count))
                )
                self.assertTrue(all(c.kwargs["args"][4] is True for c in calls[:-1]))
                self.assertEqual(
                    calls[-1].kwargs["args"][1],
                    [f"127.0.0.1:{20002+2*i}" for i in range(count)],
                )
                self.assertEqual(
                    [c.kwargs["args"][5] for c in calls[:-1]],
                    [22000 + i for i in range(count)],
                )


class TwoPdTopologyTest(unittest.TestCase):
    def cli(self):
        return [
            "--output",
            "/tmp/not-launched-e1pd2",
            "--gpus",
            "2,3",
            "--encoder-gpus",
            "1",
            "--encoder-proxy",
            "1",
            "--mtp-tokens",
            "1",
            "--coord-mode",
            "off",
        ]

    def test_three_gpu_target_lock_and_worker_arguments(self):
        with mock.patch.object(smoke, "run_launcher", side_effect=smoke.bazel_command):
            command = smoke.main(self.cli())
        for expected in (
            smoke.TWO_PD_TARGET,
            "--test_env=GPU_COUNT=3",
            "--test_env=WORLD_SIZE=3",
            "--test_env=CUDA_VISIBLE_DEVICES=1,2,3",
            "--test_arg=--gpus=2,3",
            "--test_arg=--encoder-gpus=1",
            "--test_arg=--mtp-tokens=1",
            "--run_under=//rtp_llm/test/utils:gpu_lock",
        ):
            self.assertIn(expected, command)
        targets = []
        scope = dict(
            load=lambda *a: None,
            SMOKE_FRAMEWORK_DEPS=[],
            native=SimpleNamespace(py_test=lambda **kw: targets.append(kw)),
        )
        exec(PATH.with_suffix(".bzl").read_text(), scope)
        scope["qwen35_epd_fusion_4gpu_suite"]()
        target = next(t for t in targets if t["name"] == "qwen35_e1pd2_smoke")
        self.assertEqual(target["exec_properties"]["gpu_count"], "3")
        self.assertEqual(target["env"]["GPU_COUNT"], "3")
        self.assertEqual(target["env"]["WORLD_SIZE"], "3")

    def test_two_pd_routes_and_encoder_mtp_isolation(self):
        import shlex

        env, args = smoke.server_config(
            "2,3", "baseline", "mega_moe_fp8", 0, 1, 1, 4096
        )
        ports = dict(encoder_0=20000, encoder_1=21000, fusion=22000)
        for proxy in (False, True):
            with self.subTest(proxy=proxy):
                configs = (
                    smoke.dp2_service_configs(
                        "2,3", "1", env, args, ports, 9, "candidate-a"
                    )
                    if proxy
                    else smoke.separated_service_configs(
                        "2,3", "1", "baseline", ports, 9
                    )
                )
                encoder, pd = smoke.configure_mtp_services(configs, 1, "/model")
                self.assertEqual(encoder["gpus"], [1])
                self.assertEqual(encoder["env"]["SP_TYPE"], "")
                self.assertNotIn("--sp_type", encoder["args"])
                self.assertEqual(pd["gpus"], [2, 3])
                argv = shlex.split(pd["args"])
                for flag, value in (
                    ("--world_size", "2"),
                    ("--tp_size", "1"),
                    ("--dp_size", "2"),
                    ("--ep_size", "2"),
                ):
                    self.assertEqual(argv[argv.index(flag) + 1], value)
                self.assertEqual(pd["env"]["RTP_MEGAMOE_CHUNK_TOKENS"], "0")
                route = json.loads(pd["env"]["MODEL_SERVICE_CONFIG"])["role_endpoints"][
                    0
                ]
                self.assertEqual(
                    route["pd_fusion_endpoint"]["address"],
                    "127.0.0.1:22000,127.0.0.1:22009",
                )

    def test_request_batch_only_contacts_existing_two_rank_ports(self):
        urls = []
        modules = {
            "rtp_llm.config.py_config_modules": SimpleNamespace(
                MIN_WORKER_INFO_PORT_NUM=9
            )
        }

        def response(url, payload, index, expected):
            urls.append(url)
            return dict(ok=True, index=index)

        with mock.patch.dict(sys.modules, modules), mock.patch.object(
            smoke, "request", side_effect=response
        ):
            rows = smoke.run_throughput_batch(
                SimpleNamespace(port=22000), {}, {}, 4, 4, 2
            )
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["client_rank"] for r in rows}, {0, 1})
        self.assertEqual(
            set(urls),
            {
                "http://127.0.0.1:22000/v1/chat/completions",
                "http://127.0.0.1:22009/v1/chat/completions",
            },
        )

    def test_execution_evidence_only_requires_configured_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = smoke.execution_evidence(Path(directory), 2)
        self.assertEqual(set(evidence["ranks"]), {"0", "1"})
        self.assertEqual(set(evidence["native_fp8"]), {"0", "1"})

    def test_two_rank_cli_rejects_four_rank_suites_before_launch(self):
        for extra in (
            ["--coord-checks", "1"],
            ["--graph-edge-cases", "1"],
            ["--throughput-sweep", "2"],
            ["--capacity-batches", "1,2"],
        ):
            with self.subTest(extra=extra), mock.patch.object(
                smoke, "run_launcher"
            ) as launch:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                    SystemExit
                ):
                    smoke.main(self.cli() + extra)
                launch.assert_not_called()

    def test_rejects_invalid_pd_count_and_encoder_allocations(self):
        for ids in ("2", "1,2,3", "2,2", "-1,2"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                smoke.pd_gpu_ids(ids)
        for enc in ("", "0,1", "2"):
            with self.subTest(enc=enc), self.assertRaises(ValueError):
                smoke.selected_gpus("2,3", enc)

    def test_two_rank_cadence_graph_steady_cli_all_mtp_widths(self):
        for width in range(5):
            argv = self.cli() + [
                "--mtp-tokens",
                str(width),
                "--coord-mode",
                "cadence",
                "--scheduler-policy",
                "prefill-first",
                "--decode-prefill-ratio",
                "7",
                "--schedule-trace",
                "1",
                "--trace-run-id",
                "two-rank-steady",
                "--steady-concurrency",
                "8",
                "--rank-concurrency",
                "4",
                "--decode-graph",
                "1",
                "--graph-batches",
                "1,2,4",
                "--moe-strategy",
                "mega_moe_fp8",
                "--client-mode",
                "measured",
                "--client-processes",
                "2",
            ]
            with self.subTest(width=width), mock.patch.object(
                smoke, "run_launcher", side_effect=smoke.bazel_command
            ):
                command = smoke.main(argv)
                self.assertIn("--test_arg=--steady-concurrency=8", command)
                self.assertIn(f"--test_arg=--mtp-tokens={width}", command)
            for bad in ("3", "10"):
                with mock.patch.object(smoke, "run_launcher") as launch:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                        SystemExit
                    ):
                        smoke.main(argv + ["--steady-concurrency", bad])
                    launch.assert_not_called()


class ServicePythonPathTest(unittest.TestCase):
    def test_prefix_wins_real_child_import_without_replacing_other_environment(self):
        import subprocess

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix, inherited = root / "prefix", root / "inherited"
            prefix.mkdir()
            inherited.mkdir()
            (prefix / "service_import_fixture.py").write_text("version = 'fixed'\n")
            (inherited / "service_import_fixture.py").write_text("version = 'shadow'\n")
            env = {"CUDA_VISIBLE_DEVICES": "2,3", "DP_SIZE": "2"}
            with mock.patch.dict(os.environ, {"PYTHONPATH": str(inherited)}):
                smoke.service_pythonpath_env(env, str(prefix))
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import service_import_fixture as m; print(m.version)",
                    ],
                    env=dict(os.environ, **env),
                    capture_output=True,
                    text=True,
                    check=True,
                )
            self.assertEqual(result.stdout.strip(), "fixed")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,3")
            self.assertEqual(env["DP_SIZE"], "2")

    def test_disabled_preserves_environment_and_rejects_invalid_directories(self):
        env = {"PYTHONPATH": "/original"}
        self.assertIs(smoke.service_pythonpath_env(env, ""), env)
        self.assertEqual(env, {"PYTHONPATH": "/original"})
        for path in ("relative/path", str(PATH)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                smoke.service_pythonpath_env({}, path)

    def test_prefix_reaches_worker_arguments(self):
        with tempfile.TemporaryDirectory() as prefix, mock.patch.object(
            smoke, "run_launcher", side_effect=smoke.bazel_command
        ):
            command = smoke.main(
                [
                    "--output",
                    "/tmp/service-prefix-not-launched",
                    "--service-pythonpath-prefix",
                    prefix,
                ]
            )
        self.assertIn("--test_arg=--service-pythonpath-prefix=" + prefix, command)


class MtpEvidenceTest(unittest.TestCase):
    def graph_evidence(self, directory, real=1):
        self.evidence(directory, {0: [(1, 2, 1)], 1: [(1, 2, 1)]})
        path = Path(directory) / "fusion_logs/process.log"
        lines = [path.read_text()]
        for rank in range(2):
            lines += [
                f"GRAPH_CAPTURE_READY rank={rank}",
                f"MODEL_EXECUTION rank={rank} mode=graph prefill=1 batch=2 graph_bs=2 count=1 real={real} model_id=0 target_verify=1 draft_commit=0",
                f"MODEL_EXECUTION rank={rank} mode=graph prefill=1 batch=2 graph_bs=1 count=1 real={real} model_id=1 target_verify=0 draft_commit=1",
            ]
        path.write_text("\n".join(lines))
        return smoke.execution_evidence(Path(directory), 2)

    def test_mtp_graph_requires_real_verify_and_commit_on_every_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.graph_evidence(directory)
        self.assertEqual(smoke.validate_execution_evidence(evidence, 1, 0, 1), [])
        evidence["ranks"]["1"]["draft_commit_replay"] = False
        self.assertEqual(
            smoke.validate_execution_evidence(evidence, 1, 0, 1),
            ["rank 1: missing draft_commit_replay evidence"],
        )

    def test_mtp_graph_does_not_accept_fake_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.graph_evidence(directory, real=0)
        errors = smoke.validate_execution_evidence(evidence, 1, 0, 1)
        self.assertTrue(any("missing verify_replay" in e for e in errors))
        self.assertTrue(any("missing draft_commit_replay" in e for e in errors))

    def test_mtp_graph_still_rejects_ordinary_prompt_prefill_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.graph_evidence(directory)
        evidence["records"].append(
            dict(
                rank=0,
                mode="graph",
                prefill=1,
                batch=1,
                graph_bs=1,
                real=1,
                model_id=0,
                target_verify=0,
                draft_commit=0,
            )
        )
        self.assertIn(
            "Prefill unexpectedly used graph execution",
            smoke.validate_execution_evidence(evidence, 1, 0, 1),
        )

    def test_mtp_graph_rejects_silent_eager_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.graph_evidence(directory)
        evidence["records"].append(
            dict(
                rank=0,
                mode="eager",
                prefill=1,
                batch=2,
                graph_bs=0,
                real=1,
                model_id=0,
                target_verify=1,
                draft_commit=0,
            )
        )
        self.assertIn(
            "Decode silently used eager execution in graph validation",
            smoke.validate_execution_evidence(evidence, 1, 0, 1),
        )

    def evidence(self, directory, witnesses):
        root = Path(directory)
        logs = root / "fusion_logs"
        logs.mkdir(exist_ok=True)
        lines = []
        for rank in range(2):
            pid = 200 + rank
            lines += [
                f"[process-{pid}] [rank: {rank}] initialize process_group",
                f"[process-{pid}] MegaMoE FP8 weights prepared during model construction",
                f"MODEL_EXECUTION rank={rank} mode=eager prefill=1 batch=1 graph_bs=0 count=1",
            ]
            lines += [
                f"MEASURE_STEP rank={rank} start_us=1 execute_start_us=2 end_us=3 prefill=0 decode={decode} fake=1 input_tokens=0 output_tokens={output} ok={ok}"
                for decode, output, ok in witnesses.get(rank, [])
            ]
        (logs / "process.log").write_text("\n".join(lines))
        return smoke.execution_evidence(root, 2)

    def test_mtp_verify_uses_logical_decode_and_preserves_non_mtp_check(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.evidence(directory, {0: [(1, 2, 1)], 1: [(1, 1, 1)]})
        self.assertFalse(any(r["decode"] for r in evidence["ranks"].values()))
        self.assertEqual(
            smoke.validate_execution_evidence(evidence, 0, 0, mtp_tokens=1), []
        )
        self.assertEqual(len(smoke.validate_execution_evidence(evidence, 0, 0)), 2)

    def test_fake_zero_output_or_failed_steps_are_not_real_decode_evidence(self):
        for witness in ((0, 2, 1), (1, 0, 1), (1, 2, 0)):
            with self.subTest(
                witness=witness
            ), tempfile.TemporaryDirectory() as directory:
                evidence = self.evidence(directory, {0: [witness], 1: [witness]})
                errors = smoke.validate_execution_evidence(evidence, 0, 0, mtp_tokens=1)
                self.assertEqual(len(errors), 2)
                self.assertTrue(all("logical_decode" in e for e in errors))

    def test_every_rank_needs_its_own_real_decode_witness(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self.evidence(directory, {0: [(1, 2, 1)]})
        self.assertEqual(
            smoke.validate_execution_evidence(evidence, 0, 0, mtp_tokens=1),
            ["rank 1: missing logical_decode evidence"],
        )

    def test_acceptance_summary_uses_real_request_counters(self):
        rows = [
            {
                "aux_info": {
                    "speculative_draft_rounds": 69,
                    "speculative_accepted_tokens_per_pos": [58],
                }
            }
            for _ in range(3)
        ]
        result = smoke.speculative_summary(rows, 1)
        self.assertEqual(result["draft_rounds"], 207)
        self.assertEqual(result["accepted_draft_tokens"], 174)
        self.assertAlmostEqual(result["accepted_draft_per_round"], 58 / 69)

    def test_missing_or_impossible_acceptance_is_rejected(self):
        for aux in (
            {},
            {"speculative_draft_rounds": 0, "speculative_accepted_tokens_per_pos": []},
            {"speculative_draft_rounds": 2, "speculative_accepted_tokens_per_pos": [3]},
            {
                "speculative_draft_rounds": 3,
                "speculative_accepted_tokens_per_pos": [1, 2],
            },
        ):
            with self.subTest(aux=aux), self.assertRaises(RuntimeError):
                smoke.speculative_summary([{"aux_info": aux}], 4)

    def test_all_rejected_or_short_proposals_are_valid_if_draft_rounds_exist(self):
        result = smoke.speculative_summary(
            [
                {
                    "aux_info": {
                        "speculative_draft_rounds": 3,
                        "speculative_accepted_tokens_per_pos": [],
                    }
                }
            ],
            4,
        )
        self.assertEqual(result["accepted_tokens_per_pos"], [0, 0, 0, 0])
        self.assertEqual(result["accepted_draft_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
