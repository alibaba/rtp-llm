"""Performance framework behavior and failure regressions."""

import array
import copy
import hashlib
import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.perf_test.modes import ModeExecutionContext, PrefillAdapter
from rtp_llm.test.perf_test.prefill import runner as prefill_runner_module
from rtp_llm.test.perf_test.prefill.prompts import (
    PrefixPromptFactory,
    build_grouped_prompts,
)
from rtp_llm.test.perf_test.prefill.runner import PrefillRunner
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore
from rtp_llm.test.perf_test.tests.support.prefill import (
    SeedBatchClient,
    prefill_resume_config,
    prefill_resume_plan,
    prefill_runner_case,
    prefill_runner_Client,
    prefill_runner_grouped_case,
    prefill_runner_response,
    prefill_runner_run,
    prefill_runner_Tokenizer,
)


class ShortColdPromptTest(unittest.TestCase):

    def test_short_cold_runs_have_distinct_exact_length_inputs(self):
        tokenizer = SimpleNamespace(encode=list)
        prompts = PrefixPromptFactory(tokenizer).build_case_prompts(
            "prefill-58bcef9044dc7047168737e3_g0_r0", 7, 0, 3
        )
        self.assertEqual([7, 7, 7], [len(ids) for ids in prompts.run_ids])
        self.assertEqual(3, len({tuple(ids) for ids in prompts.run_ids}))


class PreflightReuseTest(unittest.TestCase):

    class Tokenizer(prefill_runner_Tokenizer):

        def __init__(self, *, collide=False):
            self.long_encodes = 0
            self.reject_long = False
            self.collide = collide
            self.vocab = {}

        def encode(self, text):
            ids = super().encode(text)
            if len(ids) >= 64:
                self.long_encodes += 1
                if self.reject_long:
                    raise ValueError("unexpected repeated full encoding")
            if self.collide and ids and ids[0].startswith("cache_grid_shared_prefix_"):
                ids[0] = "shared"
            return [self.vocab.setdefault(word, len(self.vocab)) for word in ids]

    def case(self, *, cache_len=64, case_id="sampled", batch_size=2):
        return dict(
            prefill_runner_case(cache_len=cache_len, batch_size=batch_size),
            case_id=case_id,
            sample_index=1,
            input_len=128,
            compute_len=128 - cache_len,
            prefix_policy="independent",
            request_groups=[
                {"count": batch_size, "input_len": 128, "cache_len": cache_len}
            ],
        )

    def runner(self, tokenizer, case, **kwargs):
        client = SeedBatchClient([case["cache_len"]] * case["batch_size"])
        return (
            PrefillRunner(
                client,
                tokenizer,
                commit_tail_tokens=4,
                max_batch_tokens=4096,
                cache_block_size=4,
                **kwargs,
            ),
            client,
        )

    def test_successful_global_preflight_is_reused_without_long_reencoding(self):
        case = self.case()
        tokenizer = self.Tokenizer()
        (runner, client) = self.runner(tokenizer, case)
        runner.preflight([case])
        tokenizer.reject_long = True
        result = runner.run_case(copy.deepcopy(case))
        self.assertEqual("ok", result["status"])
        self.assertEqual(2, len(client.batches))
        self.assertEqual(4, len(result["metrics"]["measurements"]))

    def test_changed_case_with_same_id_is_validated_again(self):
        case = self.case()
        tokenizer = self.Tokenizer()
        (runner, client) = self.runner(tokenizer, case)
        runner.preflight([case])
        tokenizer.reject_long = True
        result = runner.run_case(dict(case, measure_runs=3))
        self.assertEqual("request_failed", result["status"])
        self.assertEqual([], client.batches)

    def test_standalone_run_validates_before_dispatch(self):
        case = self.case(cache_len=0)
        tokenizer = self.Tokenizer()
        (runner, client) = self.runner(tokenizer, case)
        tokenizer.reject_long = True
        result = runner.run_case(case)
        self.assertEqual("request_failed", result["status"])
        self.assertEqual([], client.batches)

    def test_failed_global_preflight_does_not_mark_partial_cases_valid(self):
        case = self.case(batch_size=1)
        other = self.case(case_id="other", batch_size=1)
        tokenizer = self.Tokenizer(collide=True)
        (runner, client) = self.runner(tokenizer, case)
        with self.assertRaisesRegex(ValueError, "not independent"):
            runner.preflight([case, other])
        tokenizer.reject_long = True
        result = runner.run_case(case)
        self.assertEqual("request_failed", result["status"])
        self.assertEqual([], client.batches)

    def test_id_transport_only_full_encodes_initial_verified_shapes(self):
        case = self.case(batch_size=16)
        tokenizer = self.Tokenizer()
        (runner, _) = self.runner(tokenizer, case, input_ids_only=True)
        runner.preflight([case])
        self.assertEqual(5, tokenizer.long_encodes)

    def test_text_transport_retains_full_text_id_consistency_check(self):
        for input_ids_only in (False, True):
            with self.subTest(input_ids_only=input_ids_only):
                case = self.case()
                (runner, _) = self.runner(
                    self.Tokenizer(), case, input_ids_only=input_ids_only
                )

                def corrupt_text(*args):
                    members = build_grouped_prompts(*args)
                    members[0]["prompts"].run_texts[0] += " extra"
                    return members

                with patch(
                    "rtp_llm.test.perf_test.prefill.runner.build_grouped_prompts",
                    side_effect=corrupt_text,
                ):
                    if input_ids_only:
                        runner.preflight([case])
                    else:
                        with self.assertRaisesRegex(
                            ValueError, "text and input IDs differ"
                        ):
                            runner.preflight([case])

    def test_id_transport_rejects_wrong_prefix_and_duplicate_round_ids(self):
        for corruption in ("prefix", "duplicate", "short_texts"):
            with self.subTest(corruption=corruption):
                case = self.case()
                (runner, client) = self.runner(
                    self.Tokenizer(), case, input_ids_only=True
                )

                def corrupt_ids(*args):
                    members = build_grouped_prompts(*args)
                    prompts = members[0]["prompts"]
                    if corruption == "prefix":
                        prompts.run_ids[0][0] = "wrong-prefix"
                    elif corruption == "duplicate":
                        prompts.run_ids[1] = prompts.run_ids[0]
                    else:
                        prompts.run_texts.pop()
                    return members

                with patch(
                    "rtp_llm.test.perf_test.prefill.runner.build_grouped_prompts",
                    side_effect=corrupt_ids,
                ):
                    result = runner.run_case(case)
                self.assertEqual("request_failed", result["status"])
                self.assertEqual([], client.batches)


class TokenDigestTest(unittest.TestCase):

    def test_digest_matches_little_endian_int32_boundary_bytes(self):
        ids = [0, 1, -1, -(2**31), 2**31 - 1]
        payload = bytes.fromhex("00000000 01000000 ffffffff 00000080 ffffff7f")
        self.assertEqual(
            hashlib.sha256(payload).digest(), prefill_runner_module._token_digest(ids)
        )

    def test_digest_preserves_equality_order_and_length(self):
        digest = prefill_runner_module._token_digest
        self.assertEqual(digest([1, 2]), digest([1, 2]))
        for left, right in (
            ([], [0]),
            ([1, 2], [2, 1]),
            ([1, 2], [1, 2, 0]),
            ([1, 23], [12, 3]),
        ):
            with self.subTest(left=left, right=right):
                self.assertNotEqual(digest(left), digest(right))

    def test_digest_rejects_out_of_range_and_non_integer_tokens(self):
        for token in (-(2**31) - 1, 2**31, 1.5, "1"):
            with self.subTest(token=token):
                with self.assertRaises(ValueError):
                    prefill_runner_module._token_digest([token])

    def test_big_endian_native_array_is_normalized(self):

        def big_endian_array(typecode, values):
            packed = array.array(typecode, values)
            if sys.byteorder == "little":
                packed.byteswap()
            return packed

        with patch.object(
            prefill_runner_module, "array", SimpleNamespace(array=big_endian_array)
        ), patch.object(prefill_runner_module, "sys", SimpleNamespace(byteorder="big")):
            actual = prefill_runner_module._token_digest([1, -1])
        expected = hashlib.sha256(bytes.fromhex("01000000 ffffffff")).digest()
        self.assertEqual(expected, actual)

    def test_preflight_rejects_shared_block_with_distinct_complete_sequences(self):
        setup = PreflightReuseTest()
        first = setup.case(batch_size=1)
        second = dict(
            setup.case(case_id="other", batch_size=1),
            input_len=132,
            compute_len=68,
            request_groups=[{"count": 1, "input_len": 132, "cache_len": 64}],
        )
        (runner, client) = setup.runner(
            setup.Tokenizer(collide=True), first, input_ids_only=True
        )
        with self.assertRaisesRegex(ValueError, "cache blocks are not independent"):
            runner.preflight([first, second])
        self.assertEqual([], client.batches)

    def test_preflight_rejects_extra_reuse_with_distinct_complete_sequences(self):
        setup = PreflightReuseTest()
        case = setup.case()
        (runner, client) = setup.runner(setup.Tokenizer(), case, input_ids_only=True)

        def corrupt_round(*args):
            members = build_grouped_prompts(*args)
            ids = members[0]["prompts"].run_ids
            ids[1][:68] = ids[0][:68]
            ids[1][-1] = -1
            return members

        with patch.object(
            prefill_runner_module, "build_grouped_prompts", side_effect=corrupt_round
        ):
            with self.assertRaisesRegex(ValueError, "rounds are not independent"):
                runner.preflight([case])
        self.assertEqual([], client.batches)


class SeedBatchTest(unittest.TestCase):

    def run_seeds(
        self,
        cache_lengths,
        *,
        max_batch_tokens=4096,
        workspace_tokens=None,
        cache_block_size=4,
        client=None,
    ):
        client = client or SeedBatchClient(cache_lengths)
        case = dict(
            prefill_runner_case(batch_size=len(cache_lengths), measure_runs=1),
            input_len=32,
            cache_len=max(cache_lengths),
            prefix_policy="independent",
            request_groups=[
                {"count": 1, "input_len": 32, "cache_len": length}
                for length in cache_lengths
            ],
        )
        result = PrefillRunner(
            client,
            prefill_runner_Tokenizer(),
            commit_tail_tokens=4,
            cache_block_size=cache_block_size,
            max_batch_tokens=max_batch_tokens,
            workspace_tokens=workspace_tokens,
        ).run_case(case)
        return (client, result)

    def test_seeds_overlap_in_batches_of_eight_with_short_tail(self):
        (client, result) = self.run_seeds([8] * 10)
        self.assertEqual("ok", result["status"])
        self.assertEqual([8, 2, 20], [len(b["requests"]) for b in client.batches])
        self.assertEqual(10, len(result["metrics"]["measurements"]))

    def test_token_budget_includes_output_and_skips_cold_members(self):
        for budget, expected in ((30, [[0, 2], [3, 4]]), (29, [[0], [2, 3], [4]])):
            with self.subTest(budget=budget):
                (client, result) = self.run_seeds(
                    [8, 0, 12, 4, 8], max_batch_tokens=budget
                )
                self.assertEqual("ok", result["status"])
                self.assertEqual(
                    expected,
                    [
                        sorted((r["request_index"] for r in b["requests"]))
                        for b in client.batches[:-1]
                    ],
                )

    def test_workspace_budget_counts_block_padding(self):
        (client, result) = self.run_seeds(
            [8, 0, 12, 4, 8], workspace_tokens=32, cache_block_size=8
        )
        self.assertEqual("ok", result["status"])
        self.assertEqual(
            [[0], [2], [3, 4]],
            [
                sorted((r["request_index"] for r in b["requests"]))
                for b in client.batches[:-1]
            ],
        )

    def test_oversized_seed_fails_before_any_dispatch(self):
        for limits in ({"max_batch_tokens": 24}, {"workspace_tokens": 27}):
            with self.subTest(limits=limits):
                (client, result) = self.run_seeds([8, 20], **limits)
                self.assertEqual("request_failed", result["status"])
                self.assertIsNone(result["metrics"])
                self.assertEqual([], client.batches)

    def test_failed_seed_batch_drains_peers_and_stops(self):
        for failure in ("failed_member", "raised_member"):
            with self.subTest(failure=failure):
                client = SeedBatchClient([8] * 10, **{failure: 0})
                (_, result) = self.run_seeds([8] * 10, client=client)
                self.assertEqual("request_failed", result["status"])
                self.assertIsNone(result["metrics"])
                self.assertEqual(1, len(client.batches))
                self.assertEqual(set(range(8)), client.completed_seeds)

    def test_submission_failure_aborts_before_dispatch(self):
        submitted = []

        class Executor(ThreadPoolExecutor):

            def submit(self, fn, *args, **kwargs):
                if submitted:
                    raise RuntimeError("submission failed")
                future = super().submit(fn, *args, **kwargs)
                submitted.append(future)
                return future

        with patch(
            "rtp_llm.test.perf_test.prefill.runner.ThreadPoolExecutor", Executor
        ):
            (client, result) = self.run_seeds([8] * 4)
        self.assertEqual("request_failed", result["status"])
        self.assertEqual([], client.batches[0]["requests"])
        self.assertEqual(1, len(submitted))
        self.assertTrue(submitted[0].done())

    def test_scheduler_failure_stops_before_seed_dispatch(self):

        class Client(SeedBatchClient):

            def configure_scheduler(self, batch_size, mode="prefill"):
                raise RuntimeError("scheduler unavailable")

        client = Client([8] * 4)
        (_, result) = self.run_seeds([8] * 4, client=client)
        self.assertEqual("request_failed", result["status"])
        self.assertEqual([], client.batches)


class PrefillWarmupTest(unittest.TestCase):

    def test_warmup_is_concurrent_independent_and_excluded_from_metrics(self):
        for cache_len in (0, 8):
            with self.subTest(cache_len=cache_len):
                barrier = threading.Barrier(2, timeout=2)
                requests = []
                warmed = set()
                lock = threading.Lock()

                class Client:

                    def configure_scheduler(self, batch_size, mode="prefill"):
                        assert batch_size == 2

                    def request(self, **request):
                        with lock:
                            requests.append(request)
                        barrier.wait()
                        phase = request["phase"]
                        member = request["request_index"]
                        if phase == "warmup":
                            with lock:
                                warmed.add(member)
                        elif phase == "measure":
                            assert warmed == {0, 1}, "measurement started before warmup"
                        return dict(
                            prefill_runner_response(
                                len(request["input_ids"]),
                                0 if phase == "seed" else cache_len,
                                latency=10000 if phase == "warmup" else 10,
                            ),
                            wait_time_ms=9999 if phase == "warmup" else 2,
                        )

                result = prefill_runner_run(
                    Client(),
                    prefill_runner_case(cache_len=cache_len, batch_size=2),
                    monotonic=iter([0.0, 0.2, 1.0, 1.5]).__next__,
                )
                self.assertEqual("ok", result["status"], result)
                warmup = [r for r in requests if r["phase"] == "warmup"]
                formal = [r for r in requests if r["phase"] == "measure"]
                self.assertEqual(2, len(warmup))
                self.assertEqual(4, len(formal))
                self.assertEqual({0, 1}, {r["run_index"] for r in formal})
                for member in (0, 1):
                    ids = [
                        r["input_ids"]
                        for r in warmup + formal
                        if r["request_index"] == member
                    ]
                    self.assertEqual(3, len({tuple(tokens) for tokens in ids}))
                    self.assertEqual(
                        3, len({tuple(tokens[: cache_len + 4]) for tokens in ids})
                    )
                    self.assertTrue(
                        all(
                            (tokens[:cache_len] == ids[0][:cache_len] for tokens in ids)
                        )
                    )
                metrics = result["metrics"]
                self.assertEqual(2, result["runs"])
                self.assertEqual(4, len(metrics["measurements"]))
                self.assertEqual(10, metrics["latency_ms"])
                self.assertEqual(2, metrics["wait_time_ms"])
                self.assertEqual(
                    56 if cache_len else 112, metrics["throughput_tokens_per_second"]
                )

    def test_failed_warmup_drains_batch_and_prevents_formal_measurement(self):
        for failure in ("request", "shape", "reuse", "timing", "exception"):
            with self.subTest(failure=failure):
                completed = set()
                requests = []
                barrier = threading.Barrier(2, timeout=2)

                class Client:

                    def configure_scheduler(self, batch_size, mode="prefill"):
                        pass

                    def request(self, **request):
                        requests.append(request)
                        barrier.wait()
                        try:
                            response = prefill_runner_response(16, 0)
                            if (
                                request["phase"] == "warmup"
                                and request["request_index"] == 0
                            ):
                                if failure == "exception":
                                    raise RuntimeError("warmup transport failed")
                                (field, value) = {
                                    "request": ("success", False),
                                    "shape": ("input_len", 15),
                                    "reuse": ("reuse_len", 4),
                                    "timing": ("latency_ms", float("nan")),
                                }[failure]
                                response[field] = value
                            return response
                        finally:
                            completed.add(request["request_index"])

                result = prefill_runner_run(
                    Client(), prefill_runner_case(cache_len=0, batch_size=2)
                )
                self.assertNotEqual("ok", result["status"])
                self.assertIsNone(result["metrics"])
                self.assertEqual({0, 1}, completed)
                self.assertEqual(["warmup", "warmup"], [r["phase"] for r in requests])


class PrefillRunnerTest(unittest.TestCase):

    def test_batch_four_measure_requests_overlap(self):
        barrier = threading.Barrier(4, timeout=2)

        class Client(prefill_runner_Client):

            def request(self, **request):
                if request["phase"] == "measure":
                    barrier.wait()
                return super().request(**request)

        client = Client(
            [prefill_runner_response(12, 0)] * 4 + [prefill_runner_response(16, 8)] * 4,
            4,
        )
        result = prefill_runner_run(
            client, prefill_runner_case(batch_size=4, measure_runs=1)
        )
        self.assertEqual("ok", result["status"])
        self.assertEqual([4, 4], client.scheduler_calls)
        self.assertEqual(["seed"] * 4, [r["phase"] for r in client.requests[:4]])

    def test_seed_failure_stops_before_measurement(self):
        client = prefill_runner_Client(
            [prefill_runner_response(12, 0, success=False)] * 4, 4
        )
        result = prefill_runner_run(
            client, prefill_runner_case(batch_size=4, measure_runs=1)
        )
        self.assertEqual("request_failed", result["status"])
        self.assertEqual([4], client.scheduler_calls)
        self.assertEqual(["seed"] * 4, [r["phase"] for r in client.requests])

    def test_round_setup_failure_aborts_and_drains_submitted_workers(self):
        for failure in ("submit", "clock"):
            with self.subTest(failure=failure):
                release = threading.Event()
                submitted = []
                results = []
                errors = []

                class Executor(ThreadPoolExecutor):

                    def submit(self, fn, *args, **kwargs):
                        if failure == "submit" and submitted:
                            raise RuntimeError("submission failed")
                        future = super().submit(fn, *args, **kwargs)
                        submitted.append(future)
                        return future

                def monotonic():
                    raise RuntimeError("clock failed")

                client = prefill_runner_Client([prefill_runner_response(16, 0)] * 4, 4)

                def run():
                    try:
                        results.append(
                            prefill_runner_run(
                                client,
                                prefill_runner_case(
                                    cache_len=0, batch_size=4, measure_runs=1
                                ),
                                monotonic=monotonic,
                            )
                        )
                    except Exception as error:
                        errors.append(error)

                with patch(
                    "rtp_llm.test.perf_test.prefill.runner.ThreadPoolExecutor", Executor
                ), patch(
                    "rtp_llm.test.perf_test.prefill.runner.threading",
                    SimpleNamespace(Event=lambda: release),
                ):
                    worker = threading.Thread(target=run)
                    worker.start()
                    try:
                        worker.join(timeout=2)
                        self.assertFalse(worker.is_alive(), "round setup deadlocked")
                    finally:
                        release.set()
                        worker.join(timeout=5)
                self.assertEqual([], errors)
                self.assertEqual(1 if failure == "submit" else 8, len(submitted))
                self.assertTrue(all((future.done() for future in submitted)))
                self.assertEqual(
                    [] if failure == "submit" else ["warmup"] * 4,
                    [request["phase"] for request in client.requests],
                )
                self.assertEqual(
                    "request_failed" if failure == "submit" else "invalid_timing",
                    results[0]["status"],
                )
                self.assertIsNone(results[0]["metrics"])

    def test_throughput_is_mean_of_round_rates(self):
        times = iter([0.0, 0.2, 1.0, 1.5])
        client = prefill_runner_Client(
            [prefill_runner_response(12, 0)] + [prefill_runner_response(16, 8)] * 2
        )
        result = prefill_runner_run(
            client, prefill_runner_case(), monotonic=lambda: next(times)
        )
        self.assertEqual(28.0, result["metrics"]["throughput_tokens_per_second"])

    def test_valid_cached_case_seeds_then_measures_exact_reuse(self):
        client = prefill_runner_Client(
            [
                prefill_runner_response(12, 0),
                dict(
                    prefill_runner_response(16, 8, latency=10.0),
                    ttft_ms=10.0,
                    wait_time_ms=1.0,
                ),
                dict(
                    prefill_runner_response(16, 8, latency=14.0),
                    ttft_ms=14.0,
                    wait_time_ms=3.0,
                ),
            ]
        )
        result = prefill_runner_run(client)
        self.assertEqual("ok", result["status"])
        self.assertEqual(4, len(client.requests))
        self.assertEqual("seed", client.requests[0]["phase"])
        self.assertEqual(
            ["warmup", "measure", "measure"], [r["phase"] for r in client.requests[1:]]
        )
        self.assertEqual(12.0, result["metrics"]["latency_ms"])
        self.assertEqual(12.0, result["metrics"]["ttft_ms"])
        self.assertEqual(2.0, result["metrics"]["wait_time_ms"])
        self.assertEqual(
            [1.0, 3.0], [m["wait_time_ms"] for m in result["metrics"]["measurements"]]
        )
        self.assertEqual(8, result["metrics"]["observed_reuse_length"])
        self.assertEqual(2, len(result["metrics"]["measurements"]))

    def test_cold_case_skips_seed_and_requires_zero_reuse(self):
        client = prefill_runner_Client(
            [prefill_runner_response(16, 0), prefill_runner_response(16, 0)]
        )
        result = prefill_runner_run(client, prefill_runner_case(cache_len=0))
        self.assertEqual("ok", result["status"])
        self.assertEqual(
            ["warmup", "measure", "measure"], [r["phase"] for r in client.requests]
        )


class GroupedRunnerTest(unittest.TestCase):

    def run_group(self, bad_reuse=False):
        barrier = threading.Barrier(3, timeout=2)

        class Client(prefill_runner_Client):

            def request(self, **request):
                self.requests.append(request)
                if request["phase"] in ("warmup", "measure"):
                    barrier.wait()
                    return prefill_runner_response(
                        len(request["input_ids"]),
                        (
                            8
                            if request["request_index"] == 0
                            or (bad_reuse and request["phase"] == "measure")
                            else 0
                        ),
                        latency=10 * (request["request_index"] + 1),
                    )
                return prefill_runner_response(len(request["input_ids"]), 0)

        client = Client([])
        times = iter([0, 0.2])
        result = prefill_runner_run(
            client, prefill_runner_grouped_case(), monotonic=lambda: next(times)
        )
        return (client, result)

    def test_grouped_members_use_own_geometry_and_round_wall_time(self):
        (client, result) = self.run_group()
        self.assertEqual("ok", result["status"])
        self.assertEqual([1, 3], client.scheduler_calls)
        self.assertEqual(
            [12], [len(r["input_ids"]) for r in client.requests if r["phase"] == "seed"]
        )
        metrics = result["metrics"]
        self.assertEqual(
            (64, 8, 56),
            (
                metrics["input_length"],
                metrics["expected_reuse_length"],
                metrics["compute_length"],
            ),
        )
        self.assertEqual(280, metrics["throughput_tokens_per_second"])
        self.assertEqual(30, metrics["latency_ms"])
        self.assertEqual(
            [16, 24, 24], [m["input_length"] for m in metrics["measurements"]]
        )
        self.assertEqual(
            [8, 0, 0], [m["expected_reuse_length"] for m in metrics["measurements"]]
        )
        self.assertTrue(
            all((m["round_elapsed_ms"] == 200 for m in metrics["measurements"]))
        )

    def test_grouped_reuse_is_validated_per_member(self):
        (_, result) = self.run_group(bad_reuse=True)
        self.assertEqual("invalid_reuse", result["status"])


class ProbeTest(unittest.TestCase):

    def test_probe_checks_seed_and_hit(self):
        from rtp_llm.test.perf_test.prefill import probe

        for broken_phase in (None, "seed", "measure"):
            requests = []

            class Client:

                def configure_scheduler(self, batch_size, mode):
                    self.batch = batch_size

                def request(self, **request):
                    requests.append(request)
                    reuse = 0 if request["phase"] == "seed" else 8
                    if request["phase"] == broken_phase:
                        reuse += 1
                    return prefill_runner_response(len(request["input_ids"]), reuse)

            client = Client()
            if broken_phase:
                with self.assertRaisesRegex(ValueError, "reuse"):
                    probe.probe_cache_block(client, prefill_runner_Tokenizer(), 8, 4)
            else:
                result = probe.probe_cache_block(
                    client, prefill_runner_Tokenizer(), 8, 4
                )
                self.assertEqual("ok", result["status"])
                self.assertEqual([12, 16], [len(r["input_ids"]) for r in requests])
                self.assertEqual(
                    requests[0]["input_ids"][:8], requests[1]["input_ids"][:8]
                )
                self.assertNotEqual(
                    requests[0]["input_ids"][8], requests[1]["input_ids"][8]
                )
            self.assertEqual(1 if broken_phase == "seed" else 2, len(requests))

    def test_probe_failure_persisted_before_any_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            plan = prefill_resume_plan(root)
            initialize_run_directory(
                root,
                source_config=prefill_resume_config(),
                plan=plan,
                run_id="probe",
                code_version="test",
            )
            calls = []

            class Runner:

                def run_case(self, case):
                    calls.append(case)

            def fail(_context):
                raise ValueError("wrong block reuse")

            adapter = PrefillAdapter(lambda context: Runner(), probe=fail)
            with self.assertRaisesRegex(ValueError, "wrong block reuse"):
                adapter(
                    ModeExecutionContext(
                        plan=plan, session=object(), store=RunStore(root)
                    )
                )
            self.assertEqual([], calls)
            self.assertEqual({}, RunStore(root).read_committed_cases())
            import json

            artifact = json.loads(
                (root / "checkpoints" / "prefill_cache_probe.json").read_text()
            )
            self.assertEqual("failed", artifact["status"])


class WarmupPhaseClient:
    """Scripted client recording phases with per-round latency control."""

    def __init__(
        self,
        cache_lengths,
        *,
        default_latency_ms=10.0,
        round_latency_ms=None,
        raise_on=(),
    ):
        self.cache_lengths = list(cache_lengths)
        self.default_latency_ms = default_latency_ms
        self.round_latency_ms = round_latency_ms or {}
        self.raise_on = set(raise_on)
        self.requests = []
        self.scheduler_calls = []

    def configure_scheduler(self, batch_size, mode="prefill"):
        assert mode == "prefill"
        self.scheduler_calls.append(batch_size)

    def request(self, **request):
        self.requests.append(request)
        if request["phase"] in self.raise_on:
            raise RuntimeError(f"{request['phase']} transport failed")
        seed = request["phase"] == "seed"
        member = request["request_index"]
        if seed:
            latency = self.round_latency_ms.get("seed", self.default_latency_ms)
            reuse = 0
        else:
            round_key = (
                "warmup" if request["phase"] == "warmup" else request["run_index"]
            )
            latency = self.round_latency_ms.get(round_key, self.default_latency_ms)
            reuse = self.cache_lengths[member]
        return prefill_runner_response(
            len(request["input_ids"]), reuse, latency=latency
        )


def _warmup_runner(client, *, monotonic=None, **kwargs):
    kwargs.setdefault("commit_tail_tokens", 4)
    kwargs.setdefault("max_batch_tokens", 1024)
    if monotonic is not None:
        kwargs["monotonic"] = iter(monotonic).__next__
    return PrefillRunner(client, prefill_runner_Tokenizer(), **kwargs)


class WarmupSweepModeTest(unittest.TestCase):
    """Conditional warmup rounds with a deviation sentinel in sweep mode."""

    def test_first_case_of_shape_warms_and_second_skips(self):
        client = WarmupPhaseClient([8])
        # Keep round durations stable so thread scheduling cannot trip the sentinel.
        runner = _warmup_runner(client, monotonic=[0.0, 1.0] * 5)
        first = runner.run_case(
            prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2)
        )
        self.assertEqual("ok", first["status"])
        self.assertTrue(first["metrics"]["warmup_used"])
        self.assertEqual(
            1, sum(1 for request in client.requests if request["phase"] == "warmup")
        )
        second = runner.run_case(
            prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2)
        )
        self.assertEqual("ok", second["status"])
        self.assertFalse(second["metrics"]["warmup_used"])
        self.assertIsNone(second["metrics"].get("warmup_anomaly"))
        self.assertEqual(
            1, sum(1 for request in client.requests if request["phase"] == "warmup")
        )

    def test_single_measure_run_always_keeps_warmup(self):
        client = WarmupPhaseClient([8])
        runner = _warmup_runner(client)
        result = runner.run_case(
            prefill_runner_case(cache_len=8, batch_size=1, measure_runs=1)
        )
        self.assertEqual("ok", result["status"])
        self.assertTrue(result["metrics"]["warmup_used"])
        self.assertEqual(
            1, sum(1 for request in client.requests if request["phase"] == "warmup")
        )

    def test_sentinel_drops_polluted_first_round(self):
        client = WarmupPhaseClient([8])
        runner = _warmup_runner(client, monotonic=[0.0, 100.0, 0.0, 10.0])
        runner._warmed_shapes.add(1)  # simulate an already-swept shape
        client.requests.clear()
        result = runner.run_case(
            prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2)
        )
        self.assertEqual("ok", result["status"])
        self.assertFalse(result["metrics"]["warmup_used"])
        anomaly = result["metrics"]["warmup_anomaly"]
        self.assertAlmostEqual(0.9, anomaly["deviation_ratio"], places=3)
        self.assertEqual([0], anomaly["dropped_runs"])
        runs = {measurement["run"] for measurement in result["metrics"]["measurements"]}
        self.assertEqual({1}, runs)

    def test_sentinel_tolerates_healthy_rounds(self):
        client = WarmupPhaseClient([8])
        runner = _warmup_runner(client, monotonic=[0.0, 10.0, 0.0, 10.0])
        runner._warmed_shapes.add(1)  # simulate an already-swept shape
        client.requests.clear()
        result = runner.run_case(
            prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2)
        )
        self.assertEqual("ok", result["status"])
        self.assertIsNone(result["metrics"].get("warmup_anomaly"))
        runs = {measurement["run"] for measurement in result["metrics"]["measurements"]}
        self.assertEqual({0, 1}, runs)


class WarmupSweepCombinationsTest(unittest.TestCase):
    def test_profiles_span_the_length_axis(self):
        cases = [
            dict(
                prefill_runner_case(cache_len=8, batch_size=2, measure_runs=2),
                case_id=f"prefill-case-{index}",
                input_len=16 + index * 8,
                compute_len=8 + index * 8,
            )
            for index in range(5)
        ]
        combos = PrefillRunner._sweep_combinations(cases, profiles_per_shape=3)
        self.assertEqual([0, 1, 2], [combo["profile"] for combo in combos])
        self.assertEqual([16, 32, 48], [combo["case"]["input_len"] for combo in combos])
        self.assertTrue(
            all(
                combo["case"]["case_id"].startswith("warmup-sweep-") for combo in combos
            )
        )
        self.assertTrue(all(combo["case"]["measure_runs"] == 1 for combo in combos))

    def test_make_sweep_case_clones_grouped_shape_with_fresh_identity(self):
        grouped = prefill_runner_grouped_case()
        sweep = PrefillRunner._make_sweep_case(grouped, 1)
        self.assertTrue(sweep["case_id"].startswith("warmup-sweep-"))
        self.assertEqual(1, sweep["measure_runs"])
        self.assertEqual(grouped["batch_size"], sweep["batch_size"])
        self.assertEqual(grouped["request_groups"], sweep["request_groups"])
        self.assertIsNot(grouped["request_groups"], sweep["request_groups"])
        self.assertNotIn("sample_index", sweep)


class WarmupSweepExecutionTest(unittest.TestCase):
    def test_sweep_warms_shapes_and_real_cases_skip_warmup(self):
        client = WarmupPhaseClient([8, 0, 0])
        runner = _warmup_runner(client)
        report = runner.warmup_sweep(
            [
                prefill_runner_grouped_case(),
                prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2),
            ]
        )
        self.assertEqual(2, report["warmed_shapes"])
        self.assertEqual([], report["failures"])
        self.assertFalse(report["budget_exhausted"])
        self.assertEqual({1, 3}, set(runner._warmed_shapes))
        result = runner.run_case(
            prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2)
        )
        self.assertEqual("ok", result["status"])
        self.assertFalse(result["metrics"]["warmup_used"])
        # grouped warmup round issues one request per member (3) plus one for
        # the ungrouped combo; the real case afterwards skips its warmup round.
        self.assertEqual(
            4,
            sum(1 for request in client.requests if request["phase"] == "warmup"),
        )

    def test_sweep_uses_fresh_prompt_material(self):
        client = WarmupPhaseClient([8])
        runner = _warmup_runner(client)
        runner.warmup_sweep(
            [
                prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2),
                dict(
                    prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2),
                    case_id="prefill-case-b",
                ),
            ]
        )
        sweep_seed_ids = [
            request["input_ids"]
            for request in client.requests
            if request["phase"] == "seed"
        ]
        runner.run_case(prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2))
        real_seed_ids = [
            request["input_ids"]
            for request in client.requests
            if request["phase"] == "seed"
        ][len(sweep_seed_ids) :]
        self.assertGreaterEqual(len(sweep_seed_ids), 1)
        self.assertEqual(1, len(real_seed_ids))
        for sweep_ids in sweep_seed_ids:
            self.assertNotEqual(sweep_ids, real_seed_ids[0])

    def test_sweep_respects_wall_clock_budget(self):
        client = WarmupPhaseClient([8])
        runner = _warmup_runner(client, monotonic=[0.0, 10_000.0])
        report = runner.warmup_sweep(
            [
                prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2),
                dict(
                    prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2),
                    case_id="prefill-case-b",
                ),
            ]
        )
        self.assertTrue(report["budget_exhausted"])
        self.assertEqual(0, report["warmed_shapes"])
        self.assertEqual([], client.requests)

    def test_sweep_tolerates_failing_combos(self):
        client = WarmupPhaseClient([8], raise_on=("seed",))
        runner = _warmup_runner(client)
        report = runner.warmup_sweep(
            [
                prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2),
                dict(
                    prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2),
                    case_id="prefill-case-b",
                ),
            ]
        )
        self.assertEqual(2, len(report["failures"]))
        self.assertEqual(0, report["warmed_shapes"])
        self.assertEqual("request_failed", report["combos"][0]["status"])


class WarmupSweepAdapterTest(unittest.TestCase):
    def test_adapter_records_sweep_checkpoint_and_skips_case_warmup(self):
        client = WarmupPhaseClient([8])
        case = prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2)
        second = dict(case, case_id="prefill-case-b")
        planned = [
            SimpleNamespace(case_id=case["case_id"], identity=case),
            SimpleNamespace(case_id=second["case_id"], identity=second),
        ]
        checkpoints = []

        class Store:
            @staticmethod
            def write_checkpoint(name, payload):
                checkpoints.append((name, payload))

            @staticmethod
            def write_prefill_case(case_id, result):
                pass

            @staticmethod
            def read_prefill_cases(plan):
                return {}

        adapter = PrefillAdapter(
            lambda context: PrefillRunner(
                context.session,
                prefill_runner_Tokenizer(),
                commit_tail_tokens=4,
                max_batch_tokens=1024,
            )
        )
        result = adapter(
            ModeExecutionContext(
                plan=SimpleNamespace(mode="prefill", cases=planned),
                session=client,
                store=Store(),
            )
        )
        self.assertEqual(RunState.COMPLETED, result.state)
        self.assertEqual(["prefill_warmup_sweep"], [name for name, _ in checkpoints])
        self.assertEqual(1, checkpoints[0][1]["warmed_shapes"])
        # Only the sweep's own combos issue warmup rounds; both real cases
        # skip theirs because the shape was already warmed.
        warmup_rounds = sum(
            1 for request in client.requests if request["phase"] == "warmup"
        )
        sweep_combos = len(checkpoints[0][1]["combos"])
        self.assertGreaterEqual(sweep_combos, 1)
        self.assertEqual(sweep_combos, warmup_rounds)
        for case_result in result.case_results:
            self.assertFalse(case_result["metrics"]["warmup_used"])


if __name__ == "__main__":
    unittest.main()
