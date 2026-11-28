import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

from writeback_acceptance import (
    AcceptanceRun,
    MultiturnAcceptance,
    build_source,
    build_turn,
    cached_keys,
    cached_usage,
    common_prefix_length,
    compare_handoff,
    compare_outputs,
    complete_prefix,
    expected_reuse,
    generation_query,
    max_overlap,
    validate_billing,
    validate_cold_control,
    validate_generation,
    validate_metrics,
    validate_reuse,
    validate_stream_usage,
    validate_topology,
    wait_for_publication,
)


class CharacterTokenizer:
    def encode(self, text):
        return [ord(character) for character in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


def service_info():
    common = (
        " --seq_size_per_block 1024 --prefill_cp_kv_cache_sharded 1"
        " --p2p_writeback_enable 1"
        " --load_cache_timeout_ms 120000 --reuse_cache 1 --enable_device_cache 1"
        " --enable_memory_cache 0 --enable_disk_cache 0 --enable_remote_cache 0"
    )
    return {
        "servers": {"prefill": {"port": 8000}, "decode": {"port": 8001}},
        "smoke_args": {
            "prefill": "--tp_size 2 --cp_rotate_method ALL_GATHER" + common,
            "decode": "--tp_size 1 --cp_rotate_method PREFILL_CP --prefill_cp_size 2" + common,
        },
    }


def handoff_response(logits):
    winner = max(range(len(logits)), key=logits.__getitem__)
    return {"input_ids": [[10, 11]], "output_ids": [[12, winner]], "logits": [logits]}


class V4CpWritebackAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.response = {
            "finished": True,
            "response": "result",
            "input_ids": [[1, 2, 3]],
            "output_ids": [[4, 5]],
            "aux_info": {
                "input_len": 3,
                "output_len": 2,
                "pd_sep": True,
                "reuse_len": 2048,
                "actual_reuse_len": 2048,
                "writeback_first_use_len": 0,
                "prefill_total_reuse_len": 2048,
                "prefill_local_reuse_len": 2048,
                "decode_total_reuse_len": 2048,
                "decode_local_reuse_len": 2048,
            },
        }

    def test_source_and_query_construction(self):
        tokenizer = CharacterTokenizer()
        prompt, input_ids, output_count = build_source(tokenizer, 2048)
        self.assertEqual(input_ids, tokenizer.encode(prompt))
        self.assertLess(len(input_ids), 2048)
        self.assertEqual(len(input_ids) + output_count - 1, 2048)
        self.assertTrue(1 < output_count <= 128)

        warm = generation_query("prompt", 8, True)["generate_config"]
        cold = generation_query("prompt", 8, False)["generate_config"]
        for key in warm:
            if key not in ("reuse_cache", "enable_device_cache"):
                self.assertEqual(warm[key], cold[key])
        self.assertFalse(cold["reuse_cache"] or cold["enable_device_cache"])

    def test_generation_and_output_validation(self):
        validate_generation(self.response, [1, 2, 3], 2, True)
        compare_outputs(self.response, self.response)
        for key, value in (("finished", False), ("input_ids", [[1, 2]]), ("output_ids", [[4]])):
            with self.subTest(gen_key=key), self.assertRaises(AssertionError):
                validate_generation(dict(self.response, **{key: value}), [1, 2, 3], 2, True)
        for key, value in (("output_ids", [[9, 5]]), ("response", "different")):
            with self.subTest(cmp_key=key), self.assertRaises(AssertionError):
                compare_outputs(self.response, dict(self.response, **{key: value}))
        self.response["aux_info"]["pd_sep"] = False
        with self.assertRaisesRegex(AssertionError, "PD separation"):
            validate_generation(self.response, [1, 2, 3], 2, True)

    def test_reuse_and_first_use_accounting(self):
        validate_reuse(self.response, 2048, True)
        with self.assertRaises(AssertionError):
            validate_reuse({}, 2048, True)
        for field in ("actual_reuse_len", "writeback_first_use_len"):
            broken = copy.deepcopy(self.response)
            del broken["aux_info"][field]
            with self.subTest(missing=field), self.assertRaisesRegex(AssertionError, field):
                validate_reuse(broken, 2048, True)

        # First-use credit is deducted from billed reuse_len and cannot repeat on subsequent hits.
        credit = copy.deepcopy(self.response)
        credit["aux_info"].update(local_reuse_len=2048, reuse_len=0, writeback_first_use_len=2048)
        for pd_sep in (False, True):
            validate_reuse(credit, 2048, pd_sep, expected_first_use=2048)
        with self.assertRaisesRegex(AssertionError, "writeback_first_use_len"):
            validate_reuse(credit, 2048, True, expected_first_use=0)
        credit["aux_info"]["reuse_len"] = 2048
        with self.assertRaisesRegex(AssertionError, "billed reuse"):
            validate_reuse(credit, 2048, False)

        # Decode cache cannot mask missing Prefill writeback or leak into cold control.
        masked = copy.deepcopy(self.response)
        masked["aux_info"]["prefill_local_reuse_len"] = 0
        with self.assertRaisesRegex(AssertionError, "prefill_local_reuse_len"):
            validate_reuse(masked, 2048, True)
        cold = copy.deepcopy(self.response)
        cold["aux_info"].update(prefill_total_reuse_len=0, prefill_local_reuse_len=0)
        with self.assertRaisesRegex(AssertionError, "Cold control reused D cache"):
            validate_reuse(cold, 0, True)

    def test_cache_keys_and_publication_wait(self):
        self.assertEqual(cached_keys({"cache_key_list": ["123", "-456"], "cache_keys": {"9": True}}), {123, -456})
        self.assertEqual(cached_keys({"cache_key_list": [], "cache_keys": {"123": True, "456": False}}), {123})
        with self.assertRaisesRegex(AssertionError, "omitted keys"):
            cached_keys({"block_size": 2048})

        now, calls = [0.0], []
        advance = lambda d: now.__setitem__(0, now[0] + d)
        status = wait_for_publication(
            lambda t: calls.append(t) or {"block_size": "2048", "cache_keys": {} if len(calls) == 1 else {"123": True}},
            2048, 0.3, lambda: now[0], advance,
        )
        self.assertEqual(cached_keys(status), {123})
        self.assertEqual(len(calls), 2)

        now[0] = 0.0
        with self.assertRaisesRegex(AssertionError, "did not publish"):
            wait_for_publication(lambda _: {"block_size": 2048, "cache_keys": {}}, 2048, 0.3, lambda: now[0], advance)
        self.assertAlmostEqual(now[0], 0.3)

        for bad_block_size in (1024, "1024", None, True, 2048.5, "invalid"):
            with self.subTest(block_size=bad_block_size), self.assertRaises(AssertionError):
                wait_for_publication(lambda _: {"block_size": bad_block_size, "cache_keys": {"1": True}}, 2048, 1)
        for bad_timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=bad_timeout), self.assertRaises(AssertionError):
                wait_for_publication(lambda _: None, 2048, bad_timeout)

    def test_live_topology_validation(self):
        live_info = service_info()
        validate_topology(live_info, 1024, 2)
        with self.assertRaisesRegex(AssertionError, "launch arguments"):
            validate_topology({"servers": {}}, 1024, 2)
        for role, old, new in (
            ("prefill", "--tp_size 2", "--tp_size 1"),
            ("decode", "--tp_size 1", "--tp_size 2"),
            ("prefill", "--cp_rotate_method ALL_GATHER", "--cp_rotate_method DISABLED --prefill_cp_size 1"),
            ("prefill", "--cp_rotate_method ALL_GATHER", ""),
            ("decode", "--cp_rotate_method PREFILL_CP", "--cp_rotate_method ALL_GATHER"),
            ("decode", "--prefill_cp_size 2", "--prefill_cp_size 1"),
            ("decode", "--prefill_cp_size 2", ""),
            ("prefill", "--p2p_writeback_enable 1", "--p2p_writeback_enable 0"),
            ("decode", "--p2p_writeback_enable 1", "--p2p_writeback_enable 0"),
            ("prefill", "--load_cache_timeout_ms 120000", "--load_cache_timeout_ms 0"),
            ("decode", "--enable_memory_cache 0", "--enable_memory_cache 1"),
            ("prefill", "--seq_size_per_block 1024", "--seq_size_per_block 512"),
        ):
            with self.subTest(role=role, option=old):
                broken = copy.deepcopy(live_info)
                broken["smoke_args"][role] = broken["smoke_args"][role].replace(old, new)
                with self.assertRaises(AssertionError):
                    validate_topology(broken, 1024, 2)

    def _mock_service(self, calls, corrupt_control=False):
        generation_count = status_count = 0

        def post(role, endpoint, query, timeout=None):
            nonlocal generation_count, status_count
            calls.append((role, endpoint, query))
            if endpoint == "/rtp_llm/cache_status":
                status_count += 1
                return {"block_size": "2048", "cache_keys": {} if status_count == 1 else {"123": True}}
            generation_count += 1
            count = query["generate_config"]["max_new_tokens"]
            reuse_len = 2048 if generation_count in (2, 3) else 0
            first_use = 2048 if generation_count == 2 else 0
            output_token = 121 if corrupt_control and generation_count == 4 else 120
            input_ids = CharacterTokenizer().encode(query["prompt"])
            return {
                "finished": True,
                "response": chr(output_token) * count,
                "input_ids": [input_ids],
                "output_ids": [[output_token] * count],
                "aux_info": {
                    "input_len": len(input_ids),
                    "output_len": count,
                    "pd_sep": role == "decode",
                    "reuse_len": reuse_len - first_use,
                    "actual_reuse_len": reuse_len,
                    "writeback_first_use_len": first_use,
                    "local_reuse_len": reuse_len,
                    "prefill_total_reuse_len": reuse_len,
                    "prefill_local_reuse_len": reuse_len,
                    "decode_total_reuse_len": 0,
                    "decode_local_reuse_len": 0,
                },
            }

        return post

    def test_full_flow_and_guards(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            run = AcceptanceRun(service_info(), directory)
            with patch.object(run, "post", side_effect=self._mock_service(calls)):
                result = run.run(CharacterTokenizer(), 2048)
            self.assertTrue(result["passed"])
            self.assertEqual(
                [(role, endpoint) for role, endpoint, _ in calls],
                [
                    ("prefill", "/rtp_llm/cache_status"),
                    ("decode", "/"),
                    ("prefill", "/rtp_llm/cache_status"),
                    ("prefill", "/"),
                    ("decode", "/"),
                    ("prefill", "/"),
                    ("decode", "/"),
                ],
            )
            self.assertEqual(
                [q["generate_config"]["reuse_cache"] for _, ep, q in calls if ep == "/"],
                [True, True, True, False, False],
            )
            self.assertEqual(
                json.loads((Path(directory) / "warm-prefill.json").read_text())["aux_info"]["local_reuse_len"], 2048
            )

        with tempfile.TemporaryDirectory() as directory:
            run = AcceptanceRun(service_info(), directory)
            with patch.object(run, "post", side_effect=self._mock_service([], True)):
                with self.assertRaisesRegex(AssertionError, "output token IDs differ"):
                    run.run(CharacterTokenizer(), 2048)
            self.assertTrue((Path(directory) / "cold-prefill.json").is_file())
            self.assertFalse((Path(directory) / "cold-pd-request.json").exists())

        with tempfile.TemporaryDirectory() as directory:
            run = AcceptanceRun(service_info(), directory)
            with patch.object(run, "post", return_value={"block_size": 2048, "cache_keys": {"123": True}}) as post:
                with self.assertRaisesRegex(AssertionError, "empty P cache"):
                    run.run(CharacterTokenizer(), 2048)
            post.assert_called_once()


class HandoffComparisonTest(unittest.TestCase):
    def test_handoff_tolerances_and_shift_invariance(self):
        near_tie = compare_handoff(handoff_response([2.0, 2.01, 0.0]), handoff_response([2.01, 2.0, 0.0]))
        self.assertFalse(near_tie["same_greedy_output"])
        self.assertLess(near_tie["total_variation"], 0.02)

        rel = compare_handoff(
            handoff_response([math.log(0.139), math.log(0.861)]),
            handoff_response([math.log(0.15), math.log(0.85)]),
        )
        self.assertGreater(rel["max_probability_error"], 0.01)
        self.assertLess(rel["max_tolerance_ratio"], 1)
        self.assertEqual(
            compare_handoff(handoff_response([1002.0, 1001.0]), handoff_response([2.0, 1.0]))["total_variation"], 0
        )

    def test_handoff_rejects_divergence_and_invalid_inputs(self):
        for warm_l, cold_l, pattern in (
            ([5.0, 0.0, -1.0], [0.5, 0.0, -1.0], "probability error"),
            ([3.0, 0.0], [0.0, 3.0], "probability error"),
            ([0.1] * 50 + [-0.1] * 50, [0.0] * 100, "Total variation"),
            ([2.0, 1.0], [2.0, 1.0, 0.0], "vocabulary sizes"),
        ):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(AssertionError, pattern):
                compare_handoff(handoff_response(warm_l), handoff_response(cold_l))

        cold = handoff_response([2.0, 1.0])
        for key, replacement in (("input_ids", [[10, 13]]), ("output_ids", [[13, 0]]), ("output_ids", [[12, 1]])):
            with self.subTest(key=key, replacement=replacement), self.assertRaises(AssertionError):
                compare_handoff(dict(cold, **{key: replacement}), cold)
        for bad_logits in (None, [], [[]], [[1.0, float("nan")]], [[1.0, float("inf")]], [[True, 0]]):
            with self.subTest(logits=bad_logits), self.assertRaises(AssertionError):
                compare_handoff(dict(cold, logits=bad_logits), cold)


class MultiturnAndBillingAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.aux = {
            "reuse_len": 1024,
            "actual_reuse_len": 1536,
            "writeback_first_use_len": 512,
            "prefill_total_reuse_len": 1536,
            "prefill_local_reuse_len": 1536,
            "decode_total_reuse_len": 1792,
        }

    def test_prefix_replay_and_turn_building(self):
        self.assertEqual((complete_prefix(480, 32, 512), complete_prefix(480, 33, 512)), (0, 512))
        self.assertEqual((expected_reuse(10252, 10240, 512, 128), expected_reuse(10368, 10240, 512, 128)), (9728, 10240))
        self.assertEqual((common_prefix_length([1, 2, 3], [1, 2, 4]), common_prefix_length([], [1])), (2, 0))

        tokenizer, history = CharacterTokenizer(), []
        for turn in range(1, 6):
            prompt, input_ids, count = build_turn(tokenizer, history, "aligned", turn, 1024, True)
            self.assertEqual(input_ids[: len(history)], history)
            self.assertEqual(complete_prefix(len(input_ids), count, 1024), turn * 1024)
            history = input_ids + tokenizer.encode("x" * count)

        _, nat_hist, _ = build_turn(tokenizer, [], "natural", 1, 1024, False)
        prompt, input_ids, count = build_turn(tokenizer, nat_hist, "natural", 2, 1024, False)
        self.assertEqual((input_ids[: len(nat_hist)], count), (nat_hist, 128))
        self.assertNotIn("repeated words", prompt[len(nat_hist) :])

    def test_multiturn_metrics_and_run_case_guards(self):
        validate_metrics(self.aux, 1536, 512)
        for mutation, expected_first, pattern in (
            ({"prefill_total_reuse_len": 0}, 512, "prefill_total_reuse_len"),
            ({"actual_reuse_len": 1024}, 512, "does not equal"),
            ({}, 0, "writeback_first_use_len"),
            ({"actual_reuse_len": None}, None, "Missing integer metric"),
        ):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(AssertionError, pattern):
                validate_metrics({**self.aux, **mutation}, 1536, expected_first)

        cold = {**dict.fromkeys(self.aux, 0), "decode_total_reuse_len": 2048, "decode_local_reuse_len": 0,
                "decode_memory_reuse_len": 0, "decode_disk_reuse_len": 0}
        validate_cold_control(cold)
        with self.assertRaisesRegex(AssertionError, "decode_local_reuse_len"):
            validate_cold_control({**cold, "decode_local_reuse_len": 512})

        with tempfile.TemporaryDirectory() as directory:
            runner = MultiturnAcceptance(CharacterTokenizer(), 1, 2, directory, 512)
            with self.assertRaisesRegex(AssertionError, "at least two"):
                runner.generate(Path(directory), "probe", "a", [97], 1, True)

        zero_aux = {**dict.fromkeys(self.aux, 0), "decode_local_reuse_len": 0,
                    "decode_memory_reuse_len": 0, "decode_disk_reuse_len": 0}
        warm = {"output_ids": [[120, 120]], "response": ["xx"], "aux_info": zero_aux}
        for output_ids, response, error in (
            ([120, 120], "xx", None),
            ([120, 121], "xx", "output token IDs differ"),
            ([120, 120], "xy", "response text differs"),
        ):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                runner = MultiturnAcceptance(CharacterTokenizer(), 1, 2, directory, 512)
                runner.status = mock.Mock(return_value={"cached_keys": []})
                runner.wait_for_prefix = mock.Mock(return_value={"cached_keys": []})
                cold_resp = {"output_ids": [output_ids], "response": [response], "aux_info": zero_aux}
                runner.generate = mock.Mock(side_effect=[warm, warm, warm, warm, cold_resp])
                with mock.patch("writeback_acceptance.build_turn", return_value=("a", [97], 2)):
                    if error:
                        with self.assertRaisesRegex(AssertionError, error):
                            runner.run_case("comparison", 2, False)
                        self.assertFalse((Path(directory) / "comparison/result.json").exists())
                    else:
                        result = runner.run_case("comparison", 2, False)
                        self.assertTrue(result["passed"] and result["warm_cold_next_token_equal"])

    def test_billing_overlap_and_stream_usage(self):
        def make_chat(first_use=512):
            return {
                "aux_info": {
                    "pd_sep": True, "input_len": 672, "output_len": 4,
                    "reuse_len": 512 - first_use, "actual_reuse_len": 512,
                    "writeback_first_use_len": first_use,
                    "prefill_total_reuse_len": 512, "prefill_local_reuse_len": 512,
                    "decode_total_reuse_len": 512,
                },
                "usage": {
                    "prompt_tokens": 672, "completion_tokens": 4, "total_tokens": 676,
                    "prompt_tokens_details": {"cached_tokens": 512 - first_use},
                },
            }

        for first_use in (0, 512):
            validate_billing(make_chat(first_use), list(range(672)), 4, 512, {0, 512})
        bad_cached = make_chat()
        bad_cached["usage"]["prompt_tokens_details"]["cached_tokens"] = 512
        with self.assertRaisesRegex(AssertionError, "Usage differs"):
            validate_billing(bad_cached, list(range(672)), 4, 512, {512})
        with self.assertRaisesRegex(AssertionError, "first-use snapshot"):
            validate_billing(make_chat(0), list(range(672)), 4, 512, {512})
        bad_total = make_chat()
        bad_total["usage"]["total_tokens"] += 1
        with self.assertRaisesRegex(AssertionError, "total tokens"):
            validate_billing(bad_total, list(range(672)), 4, 512, {512})

        self.assertEqual(cached_usage({"usage": {}}), 0)
        self.assertEqual(max_overlap([{"start": 0, "end": 2}, {"start": 1, "end": 3}]), 2)
        self.assertEqual(max_overlap([{"start": 0, "end": 1}, {"start": 1, "end": 2}]), 1)

        frames = [
            {"usage": None},
            *[
                {"usage": {
                    "prompt_tokens": 672, "completion_tokens": c, "total_tokens": 672 + c,
                    "prompt_tokens_details": {"cached_tokens": 512},
                }}
                for c in (1, 2, 3, 4, 4)
            ],
        ]
        validate_stream_usage(frames, 672, 4, 512)
        with self.assertRaisesRegex(AssertionError, "final completion"):
            validate_stream_usage(frames[:3], 672, 4, 512)
        frames[-1]["usage"]["prompt_tokens"] *= 2
        with self.assertRaisesRegex(AssertionError, "accumulated prompt"):
            validate_stream_usage(frames, 672, 4, 512)


if __name__ == "__main__":
    unittest.main()
