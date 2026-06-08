import os
import unittest
from enum import Enum

from smoke.flexlb_smoke_checks import (
    FLEXLB_SMOKE_DEFAULT_TIMEOUT_MS,
    apply_flexlb_frontend_timeouts,
    decode_frontend_json_response,
    find_role_addr,
    flexlb_check_enabled,
    flexlb_smoke_positive_int_list,
    load_flexlb_config,
    make_fast_probe_qr,
    make_worker_dp_leader_addrs,
    max_reuse_len,
    queue_snapshot_count,
    role_hit_cache_len,
)


class Role(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class FlexLbSmokeChecksTest(unittest.TestCase):
    def tearDown(self):
        for name in (
            "FLEXLB_CONFIG",
            "FLEXLB_SMOKE_GENERATE_TIMEOUT_MS",
            "FLEXLB_SMOKE_MASTER_TIMEOUT_MS",
            "FLEXLB_SMOKE_MASTER_SESSION_TIMEOUT_S",
            "FLEXLB_SMOKE_CACHE_PROMPT_REPEAT_CANDIDATES",
            "FLEXLB_SMOKE_MAX_TOKENS",
        ):
            os.environ.pop(name, None)

    def test_apply_frontend_timeouts_defaults_and_preserves_existing(self):
        envs = {}
        apply_flexlb_frontend_timeouts(envs)
        self.assertEqual(
            FLEXLB_SMOKE_DEFAULT_TIMEOUT_MS, envs["MASTER_DEFAULT_TIMEOUT_MS"]
        )
        self.assertEqual("10", envs["MASTER_SESSION_TIMEOUT_S"])

        envs = {
            "MASTER_DEFAULT_TIMEOUT_MS": "77",
            "MASTER_SESSION_TIMEOUT_S": "88",
        }
        apply_flexlb_frontend_timeouts(envs)
        self.assertEqual("77", envs["MASTER_DEFAULT_TIMEOUT_MS"])
        self.assertEqual("88", envs["MASTER_SESSION_TIMEOUT_S"])

    def test_make_fast_probe_qr_chat_sets_timeout_and_keeps_source_unchanged(self):
        source_qr = {
            "query": {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }
        }
        probe = make_fast_probe_qr(
            source_qr,
            suffix=" suffix",
            prompt_repeat=2,
            max_tokens=3,
            timeout_ms=1234,
        )

        self.assertTrue(source_qr["query"]["stream"])
        query = probe["query"]
        self.assertFalse(query["stream"])
        self.assertEqual(3, query["max_tokens"])
        self.assertEqual("hello\nhello suffix", query["messages"][0]["content"])
        self.assertEqual(1234, query["extra_configs"]["timeout_ms"])
        self.assertEqual(1234, query["extra_configs"]["ttft_timeout_ms"])
        self.assertTrue(query["extra_configs"]["aux_info"])

    def test_make_worker_dp_leader_addrs_skips_tp_non_leaders(self):
        self.assertEqual(
            ["10.0.0.1:10100", "10.0.0.1:10116"],
            make_worker_dp_leader_addrs(
                base_port=10100,
                dp_size=2,
                tp_size=2,
                world_size=4,
                host="10.0.0.1",
            ),
        )
        self.assertEqual(
            ["127.0.0.1:10300"],
            make_worker_dp_leader_addrs(
                base_port=10300,
                dp_size=1,
                tp_size=2,
                world_size=2,
            ),
        )

    def test_make_fast_probe_qr_raw_prompt_sets_generate_config(self):
        source_qr = {"query": {"prompt": "cap", "yield_generator": True}}
        probe = make_fast_probe_qr(source_qr, prompt_repeat=2, timeout_ms=4321)

        query = probe["query"]
        self.assertFalse(query["yield_generator"])
        self.assertEqual("cap\ncap", query["prompt"])
        self.assertEqual(1, query["generate_config"]["max_new_tokens"])
        self.assertEqual(4321, query["generate_config"]["timeout_ms"])
        self.assertEqual(4321, query["generate_config"]["ttft_timeout_ms"])

    def test_find_role_addr_from_aux_info_and_schedule_status(self):
        response = {
            "aux_info": {
                "role_addrs": [
                    {"role": Role.PREFILL, "http_port": 10100},
                    {"role": "DECODE", "http_port": 10300},
                ]
            }
        }
        self.assertEqual(10100, find_role_addr(response, "prefill")["http_port"])
        self.assertEqual(10300, find_role_addr(response, "decode")["http_port"])

        schedule = {
            "server_status": [
                {"role": "RoleType.PREFILL", "http_port": 10100},
            ]
        }
        self.assertEqual(10100, find_role_addr(schedule, "PREFILL")["http_port"])

    def test_max_reuse_len_reads_prefill_reuse_fields(self):
        response = {
            "aux_info": {
                "role_addrs": [],
                "reuse_len": 1,
                "prefill_total_reuse_len": "7",
            }
        }
        self.assertEqual(7, max_reuse_len(response))

    def test_decode_frontend_json_response_supports_streaming_last_chunk(self):
        ok, body = decode_frontend_json_response(
            [b"", b'data: {"ok": false}', b'data: {"ok": true}']
        )
        self.assertTrue(ok)
        self.assertEqual({"ok": True}, body)

    def test_flexlb_config_and_flags(self):
        self.assertEqual(
            {"maxQueueSize": 20},
            load_flexlb_config({"FLEXLB_CONFIG": '{"maxQueueSize": 20}'}),
        )
        self.assertEqual({}, load_flexlb_config({"FLEXLB_CONFIG": "not-json"}))
        self.assertTrue(
            flexlb_check_enabled(
                {"FLEXLB_SMOKE_CHECK_QUEUE": "true"}, "FLEXLB_SMOKE_CHECK_QUEUE"
            )
        )
        self.assertFalse(flexlb_check_enabled({}, "FLEXLB_SMOKE_CHECK_QUEUE"))
        self.assertTrue(
            flexlb_check_enabled({}, "FLEXLB_SMOKE_CHECK_QUEUE", default=True)
        )

    def test_flexlb_smoke_positive_int_list_prefers_role_env(self):
        os.environ["FLEXLB_SMOKE_CACHE_PROMPT_REPEAT_CANDIDATES"] = "1,2"
        self.assertEqual(
            [260, 320],
            flexlb_smoke_positive_int_list(
                {"FLEXLB_SMOKE_CACHE_PROMPT_REPEAT_CANDIDATES": "260,320"},
                "FLEXLB_SMOKE_CACHE_PROMPT_REPEAT_CANDIDATES",
                "8",
            ),
        )
        self.assertEqual(
            [1, 2],
            flexlb_smoke_positive_int_list(
                {},
                "FLEXLB_SMOKE_CACHE_PROMPT_REPEAT_CANDIDATES",
                "8",
            ),
        )
        os.environ.pop("FLEXLB_SMOKE_CACHE_PROMPT_REPEAT_CANDIDATES", None)
        self.assertEqual(
            [8],
            flexlb_smoke_positive_int_list(
                {},
                "FLEXLB_SMOKE_CACHE_PROMPT_REPEAT_CANDIDATES",
                "8",
            ),
        )

    def test_queue_snapshot_count(self):
        self.assertEqual(0, queue_snapshot_count(None))
        self.assertEqual(0, queue_snapshot_count({"count": "bad"}))
        self.assertEqual(8, queue_snapshot_count({"count": "8"}))

    def test_role_hit_cache_len_from_schedule_debug_info(self):
        response = {
            "server_status": [
                {
                    "role": "PREFILL",
                    "http_port": 10100,
                    "debug_info": {"hit_cache_len": "64"},
                },
            ]
        }
        self.assertEqual(64, role_hit_cache_len(response, "prefill"))
        self.assertIsNone(role_hit_cache_len(response, "decode"))
        response["server_status"][0]["debug_info"] = {
            "hit_cache_len": "bad",
            "hitCacheLen": 32,
        }
        self.assertEqual(32, role_hit_cache_len(response, "PREFILL"))


if __name__ == "__main__":
    unittest.main()
