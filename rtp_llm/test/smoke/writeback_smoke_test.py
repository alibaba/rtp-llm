import copy
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from google.protobuf.json_format import MessageToDict
from smoke.base_comparer import BaseComparer
from smoke.cache_status_comparer import CacheStatusComparer
from smoke.case_runner import CaseRunner
from smoke.common_def import SmokeException, Tracer
from smoke.multi_inst_case_runner import DpSeperationCaseRunner
from smoke.normal_comparer import NormalComparer, SmokeResponse
from smoke.task_info import TaskInfo, TaskStates

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import CacheStatusPB


class CacheWaitTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        clock = patch("smoke.cache_status_comparer.time.monotonic", lambda: self.now)
        sleep = patch("smoke.cache_status_comparer.time.sleep", self.advance)
        self.post = patch("smoke.cache_status_comparer.requests.post").start()
        clock.start()
        sleep.start()
        self.addCleanup(patch.stopall)
        self.server = SimpleNamespace(port=12345, visit=Mock())
        self.comparer = CacheStatusComparer(
            self.server,
            "/rtp_llm/cache_status",
            {
                "query": {"latest_cache_version": 7},
                "result": {"block_size": 8},
                "wait_for_cache": {"min_cached_blocks": 2, "timeout_seconds": 0.3},
            },
            Tracer(),
            False,
        )

    def advance(self, seconds):
        self.now += seconds

    def response(self, keys, block_size=8):
        return Mock(json=Mock(return_value={
            "cached_keys": keys, "block_size": block_size, "version": 2
        }))

    def test_waits_for_prefill_keys_without_generating_requests(self):
        self.post.side_effect = [self.response([100]), self.response([100, 200])]
        self.comparer.run()
        self.assertEqual(self.post.call_count, 2)
        for call in self.post.call_args_list:
            self.assertEqual(call.args, ("http://127.0.0.1:12345/rtp_llm/cache_status",))
            self.assertEqual(call.kwargs["json"], {
                "latest_cache_version": -1, "need_cache_keys": True
            })
            self.assertGreater(call.kwargs["timeout"], 0)
            self.assertLessEqual(call.kwargs["timeout"], 0.3)
        self.server.visit.assert_not_called()
        self.assertEqual(self.comparer.tracer.actual_result.cached_keys, [100, 200])

    def test_original_prompt_cache_alone_times_out(self):
        self.post.return_value = self.response([100])
        with self.assertRaisesRegex(SmokeException, "cached_blocks=1"):
            self.comparer.run()
        self.assertAlmostEqual(self.now, 0.3)

    def test_waits_for_keys_from_real_rpc_serialization(self):
        keys = [6858790234522785586, -6858790234522785586]
        responses = [
            CacheStatusPB(block_size=8, version=1, cache_keys={keys[0]: True}),
            CacheStatusPB(block_size=8, version=2, cache_keys=dict.fromkeys(keys, True)),
        ]
        self.post.side_effect = [
            Mock(json=Mock(return_value=MessageToDict(
                response,
                preserving_proto_field_name=True,
                including_default_value_fields=True,
            )))
            for response in responses
        ]
        self.comparer.run()
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(set(self.comparer.tracer.actual_result.cached_keys), set(keys))

    def test_empty_or_removed_rpc_keys_do_not_satisfy_wait(self):
        for keys, count in [({}, 0), ({100: True, 200: False}, 1)]:
            with self.subTest(keys=keys):
                self.now = 0.0
                response = CacheStatusPB(block_size=8, version=2, cache_keys=keys)
                self.post.return_value = Mock(json=Mock(return_value=MessageToDict(
                    response,
                    preserving_proto_field_name=True,
                    including_default_value_fields=True,
                )))
                with self.assertRaisesRegex(SmokeException, f"cached_blocks={count}"):
                    self.comparer.run()

    def test_duplicate_keys_do_not_count_as_written_back_blocks(self):
        self.post.return_value = self.response([100, 100])
        with self.assertRaisesRegex(SmokeException, "cached_blocks=1"):
            self.comparer.run()

    def test_wrong_block_size_does_not_satisfy_wait(self):
        self.post.return_value = self.response([100, 200], block_size=16)
        with self.assertRaisesRegex(SmokeException, "Cache wait timed out"):
            self.comparer.run()

    def test_transient_http_error_can_recover(self):
        self.post.side_effect = [requests.Timeout("slow status"), self.response([100, 200])]
        self.comparer.run()
        self.assertEqual(self.post.call_count, 2)

    def test_http_errors_are_bounded_and_reported(self):
        self.post.side_effect = requests.Timeout("slow status")
        with self.assertRaisesRegex(SmokeException, "slow status"):
            self.comparer.run()
        self.assertAlmostEqual(self.now, 0.3)

    def test_invalid_wait_limits_fail_before_http(self):
        for value in [0, -1, float("inf"), float("nan")]:
            with self.subTest(timeout=value):
                self.comparer.qr_info["wait_for_cache"]["timeout_seconds"] = value
                with self.assertRaises(SmokeException):
                    self.comparer.run()
        self.post.assert_not_called()

    def test_ordinary_cache_queries_keep_existing_comparison(self):
        del self.comparer.qr_info["wait_for_cache"]
        with patch.object(BaseComparer, "run") as run:
            self.comparer.run()
            run.assert_called_once_with()
        self.post.assert_not_called()


class WritebackSmokeTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parent / "data/model/qwen25/q_r_dp_sep_p2p_writeback.json"
        with path.open() as source:
            self.fixture = json.load(source)
        self.task_info = TaskInfo(**self.fixture)
        self.decode = Mock(port=12345)
        self.prefill = Mock(port=12346)

    def test_query_routing_uses_prefill_only_for_cache_status(self):
        runner = CaseRunner(self.task_info, [], "")
        runner.server_managers = {"prefill": self.prefill, "decode": self.decode}
        with patch.object(runner, "_get_comparer_cls", return_value=Mock()) as factory:
            with patch.dict(os.environ, {"STABILITY_REPEAT": "1", "SAVE_RESPONSE": "False"}):
                states = runner.curl_server(self.decode)
        self.assertTrue(states.ret)
        comparer = factory.return_value
        self.assertEqual(
            [call.args[0] for call in comparer.call_args_list],
            [self.decode, self.prefill, self.decode] * 2,
        )

    def test_unknown_role_is_a_failure_instead_of_using_decode(self):
        runner = CaseRunner(self.task_info, [], "")
        with self.assertRaisesRegex(SmokeException, "Unknown server_role"):
            runner._resolve_server({"server_role": "prefill"}, self.decode)

    def test_decode_entrance_runner_registers_both_servers(self):
        runner = DpSeperationCaseRunner(
            self.task_info,
            {"prefill": ["WORLD_SIZE=2"], "decode": ["WORLD_SIZE=2"]},
            "",
        )
        runner._keepalive_enabled = Mock(return_value=False)
        runner.start_servers_parallel = Mock(return_value=(
            [self.decode, self.prefill], [TaskStates(), TaskStates()]
        ))
        runner.curl_server = Mock(return_value=TaskStates())
        with patch("smoke.multi_inst_case_runner.get_gpu_ids", return_value=[0, 1, 2, 3]):
            with patch("smoke.multi_inst_case_runner.MagaServerManager.get_free_port",
                       side_effect=["12346", "12345"]):
                self.assertTrue(runner.run().ret)
        self.assertEqual(runner.server_managers, {
            "prefill": self.prefill, "decode": self.decode
        })
        runner.curl_server.assert_called_once_with(self.decode)
        self.prefill.stop_server.assert_called_once_with()
        self.decode.stop_server.assert_called_once_with()

    def test_fixture_reuses_generated_tokens_and_excludes_last_sample(self):
        first, wait, followup = self.fixture["query_result"]
        source = first["result"]["input_ids"][0] + first["result"]["output_ids"][0]
        target = followup["result"]["input_ids"][0]
        self.assertEqual(target, source[:len(target)])
        self.assertEqual(followup["result"]["output_ids"][0], source[len(target):])
        reuse = followup["result"]["aux_info"]["prefill_local_reuse_len"]
        self.assertGreater(reuse, first["result"]["aux_info"]["input_len"])
        self.assertEqual(reuse, ((len(source) - 1) // 8) * 8)
        self.assertEqual(wait["server_role"], "prefill")
        self.assertEqual(wait["wait_for_cache"]["min_cached_blocks"] * 8, reuse)

    def test_decode_local_hit_cannot_hide_missing_prefill_writeback(self):
        expected = self.fixture["query_result"][2]["result"]
        actual = copy.deepcopy(expected)
        actual["aux_info"]["prefill_local_reuse_len"] = 8
        actual["aux_info"]["prefill_total_reuse_len"] = 8
        comparer = NormalComparer(self.decode, "/", {}, Tracer(), False)
        with self.assertRaisesRegex(SmokeException, "prefill_local_reuse_len"):
            comparer.compare_result(SmokeResponse(**expected), SmokeResponse(**actual))

    def test_missing_aux_info_cannot_skip_writeback_assertions(self):
        expected = self.fixture["query_result"][2]["result"]
        actual = copy.deepcopy(expected)
        del actual["aux_info"]
        comparer = NormalComparer(self.decode, "/", {}, Tracer(), False)
        with self.assertRaisesRegex(SmokeException, "aux_info: expected but missing"):
            comparer.compare_result(SmokeResponse(**expected), SmokeResponse(**actual))


if __name__ == "__main__":
    unittest.main()
