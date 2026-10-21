import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v4_cp_writeback_acceptance import (
    AcceptanceRun,
    build_source,
    cached_keys,
    compare_outputs,
    generation_query,
    validate_generation,
    validate_reuse,
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
        " --p2p_writeback_enable 1 --p2p_writeback_timeout_ms 30000"
        " --load_cache_timeout_ms 120000 --reuse_cache 1 --enable_device_cache 1"
        " --enable_memory_cache 0 --enable_disk_cache 0 --enable_remote_cache 0"
    )
    return {
        "servers": {"prefill": {"port": 8000}, "decode": {"port": 8001}},
        "smoke_args": {
            "prefill": "--tp_size 2" + common,
            "decode": "--tp_size 1" + common,
        },
    }


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
                "prefill_total_reuse_len": 2048,
                "prefill_local_reuse_len": 2048,
                "decode_total_reuse_len": 2048,
                "decode_local_reuse_len": 2048,
            },
        }

    def test_source_crosses_complete_cp_unit_excluding_last_sample(self):
        tokenizer = CharacterTokenizer()
        prompt, input_ids, output_count = build_source(tokenizer, 2048)
        self.assertEqual(input_ids, tokenizer.encode(prompt))
        self.assertLess(len(input_ids), 2048)
        self.assertEqual(len(input_ids) + output_count - 1, 2048)
        self.assertGreater(output_count, 1)
        self.assertLessEqual(output_count, 128)

    def test_control_disables_cache_but_keeps_sampling_identical(self):
        warm = generation_query("prompt", 8, True)["generate_config"]
        cold = generation_query("prompt", 8, False)["generate_config"]
        for key in warm:
            if key not in ("reuse_cache", "enable_device_cache"):
                self.assertEqual(warm[key], cold[key])
        self.assertFalse(cold["reuse_cache"])
        self.assertFalse(cold["enable_device_cache"])

    def test_generation_checks_pd_and_token_identity(self):
        validate_generation(self.response, [1, 2, 3], 2, True)
        for key, value in (
            ("finished", False),
            ("input_ids", [[1, 2]]),
            ("output_ids", [[4]]),
        ):
            with self.subTest(key=key):
                broken = dict(self.response, **{key: value})
                with self.assertRaises(AssertionError):
                    validate_generation(broken, [1, 2, 3], 2, True)
        self.response["aux_info"]["pd_sep"] = False
        with self.assertRaisesRegex(AssertionError, "PD separation"):
            validate_generation(self.response, [1, 2, 3], 2, True)

    def test_decode_cache_cannot_mask_missing_prefill_writeback(self):
        validate_reuse(self.response, 2048, True)
        self.response["aux_info"]["prefill_local_reuse_len"] = 0
        with self.assertRaisesRegex(AssertionError, "prefill_local_reuse_len"):
            validate_reuse(self.response, 2048, True)

    def test_missing_reuse_fields_are_not_silently_ignored(self):
        with self.assertRaises(AssertionError):
            validate_reuse({}, 2048, True)

    def test_cold_control_must_not_reuse_decode_cache(self):
        self.response["aux_info"].update(
            prefill_total_reuse_len=0, prefill_local_reuse_len=0
        )
        with self.assertRaisesRegex(AssertionError, "Cold control reused D cache"):
            validate_reuse(self.response, 0, True)

    def test_prefill_only_uses_local_reuse_counters(self):
        self.response["aux_info"].update(reuse_len=2048, local_reuse_len=2048)
        validate_reuse(self.response, 2048, False)
        self.response["aux_info"]["local_reuse_len"] = 1024
        with self.assertRaises(AssertionError):
            validate_reuse(self.response, 2048, False)

    def test_text_and_tokens_must_both_match_control(self):
        compare_outputs(self.response, self.response)
        for key, value in (("output_ids", [[9, 5]]), ("response", "different")):
            with self.subTest(key=key):
                control = copy.deepcopy(self.response)
                control[key] = value
                with self.assertRaises(AssertionError):
                    compare_outputs(self.response, control)

    def test_publication_wait_never_generates_requests(self):
        now = [0.0]
        calls = []

        def post(timeout):
            calls.append(timeout)
            return {
                "block_size": 2048,
                "cache_keys": {} if len(calls) == 1 else {"123": True},
            }

        status = wait_for_publication(
            post,
            2048,
            0.3,
            lambda: now[0],
            lambda duration: now.__setitem__(0, now[0] + duration),
        )
        self.assertEqual(cached_keys(status), {123})
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(0 < timeout <= 0.3 for timeout in calls))

    def test_absent_writeback_times_out(self):
        now = [0.0]
        with self.assertRaisesRegex(AssertionError, "did not publish"):
            wait_for_publication(
                lambda timeout: {"block_size": 2048, "cache_keys": {"123": False}},
                2048,
                0.3,
                lambda: now[0],
                lambda duration: now.__setitem__(0, now[0] + duration),
            )
        self.assertAlmostEqual(now[0], 0.3)

    def test_wrong_cp_geometry_is_rejected(self):
        for block_size in (1024, "1024"):
            with self.subTest(block_size=block_size):
                with self.assertRaisesRegex(AssertionError, "canonical CP unit"):
                    wait_for_publication(
                        lambda timeout: {
                            "block_size": block_size,
                            "cached_keys": [123],
                        },
                        2048,
                        1,
                    )

    def test_publication_accepts_integer_and_proto_json_block_sizes(self):
        for block_size in (2048, "2048"):
            with self.subTest(block_size=block_size):
                status = {"block_size": block_size, "cache_keys": {"123": True}}
                self.assertEqual(
                    wait_for_publication(lambda timeout: status, 2048, 1), status
                )

    def test_publication_rejects_malformed_block_sizes(self):
        for block_size in (None, True, 2048.0, 2048.5, "", "2048.0", "invalid"):
            with self.subTest(block_size=block_size), self.assertRaises(AssertionError):
                wait_for_publication(
                    lambda timeout: {
                        "block_size": block_size,
                        "cache_keys": {"123": True},
                    },
                    2048,
                    1,
                )

    def test_publication_requires_explicit_keys_and_valid_timeout(self):
        with self.assertRaisesRegex(AssertionError, "omitted keys"):
            cached_keys({"block_size": 2048})
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(AssertionError):
                wait_for_publication(lambda unused: None, 2048, timeout)

    def test_live_topology_requires_asymmetric_cp_and_both_writeback_flags(self):
        live_info = service_info()
        validate_topology(live_info, 1024, 2)
        for role, old, new in (
            ("prefill", "--tp_size 2", "--tp_size 1"),
            ("decode", "--tp_size 1", "--tp_size 2"),
            ("prefill", "--p2p_writeback_enable 1", "--p2p_writeback_enable 0"),
            ("decode", "--p2p_writeback_enable 1", "--p2p_writeback_enable 0"),
            (
                "prefill",
                "--load_cache_timeout_ms 120000",
                "--load_cache_timeout_ms 5000",
            ),
            ("decode", "--enable_memory_cache 0", "--enable_memory_cache 1"),
            ("prefill", "--seq_size_per_block 1024", "--seq_size_per_block 512"),
        ):
            with self.subTest(role=role, option=old):
                broken = copy.deepcopy(live_info)
                broken["smoke_args"][role] = broken["smoke_args"][role].replace(
                    old, new
                )
                with self.assertRaises(AssertionError):
                    validate_topology(broken, 1024, 2)

    def test_missing_launch_metadata_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "launch arguments"):
            validate_topology({"servers": {}}, 1024, 2)

    def mock_service(self, calls, corrupt_control=False):
        generation_count = 0
        status_count = 0

        def post(role, endpoint, query, timeout=None):
            nonlocal generation_count, status_count
            calls.append((role, endpoint, query))
            if endpoint == "/rtp_llm/cache_status":
                status_count += 1
                return {
                    "block_size": "2048",
                    "cache_keys": {} if status_count == 1 else {"123": True},
                }
            generation_count += 1
            config = query["generate_config"]
            count = config["max_new_tokens"]
            reuse_len = 2048 if generation_count in (2, 3) else 0
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
                    "reuse_len": reuse_len,
                    "local_reuse_len": reuse_len,
                    "prefill_total_reuse_len": reuse_len,
                    "prefill_local_reuse_len": reuse_len,
                    "decode_total_reuse_len": 0,
                    "decode_local_reuse_len": 0,
                },
            }

        return post

    def test_full_flow_probes_publication_and_warm_p_before_cold_controls(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            run = AcceptanceRun(service_info(), directory)
            with patch.object(run, "post", side_effect=self.mock_service(calls)):
                result = run.run(CharacterTokenizer(), 2048)
            self.assertTrue(result["passed"])
            self.assertGreater(result["generated_tokens_reused"], 0)
            self.assertEqual(
                [(role, endpoint) for role, endpoint, query in calls],
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
                [
                    query["generate_config"]["reuse_cache"]
                    for _, endpoint, query in calls
                    if endpoint == "/"
                ],
                [True, True, True, False, False],
            )
            self.assertEqual(
                json.loads((Path(directory) / "warm-prefill.json").read_text())[
                    "aux_info"
                ]["local_reuse_len"],
                2048,
            )

    def test_full_flow_rejects_bad_outputs_and_keeps_failure_response(self):
        with tempfile.TemporaryDirectory() as directory:
            run = AcceptanceRun(service_info(), directory)
            with patch.object(run, "post", side_effect=self.mock_service([], True)):
                with self.assertRaisesRegex(AssertionError, "output token IDs differ"):
                    run.run(CharacterTokenizer(), 2048)
            self.assertTrue((Path(directory) / "cold-prefill.json").is_file())
            self.assertFalse((Path(directory) / "cold-pd-request.json").exists())

    def test_nonempty_prefill_cache_fails_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = AcceptanceRun(service_info(), directory)
            with patch.object(
                run,
                "post",
                return_value={"block_size": 2048, "cache_keys": {"123": True}},
            ) as post:
                with self.assertRaisesRegex(AssertionError, "empty P cache"):
                    run.run(CharacterTokenizer(), 2048)
            post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
