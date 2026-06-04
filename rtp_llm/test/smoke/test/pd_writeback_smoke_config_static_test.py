import json
import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SMOKE_ROOT = REPO_ROOT / "rtp_llm" / "test" / "smoke"
TASK_INFO = SMOKE_ROOT / "data" / "model" / "qwen3" / "q_r_h20_pd_writeback_reuse.json"
ROCM_TASK_INFO = (
    SMOKE_ROOT / "data" / "model" / "qwen3" / "q_r_rocm_pd_writeback_reuse.json"
)
ROCM_BOUNDARY_TASK_INFO = (
    SMOKE_ROOT / "data" / "model" / "qwen3" / "q_r_rocm_pd_writeback_boundary.json"
)
ROCM_TOKEN_DIFF_TASK_INFO = (
    SMOKE_ROOT / "data" / "model" / "qwen3" / "q_r_rocm_pd_writeback_token_diff.json"
)
SUITE = SMOKE_ROOT / "suites_h20_oss.bzl"
ROCM_SUITE = SMOKE_ROOT / "suites_rocm_oss.bzl"
DEFS = SMOKE_ROOT / "defs.bzl"
ENTRY = SMOKE_ROOT / "entry.py"
CASE_RUNNER = SMOKE_ROOT / "case_runner.py"
NORMAL_COMPARER = SMOKE_ROOT / "normal_comparer.py"
P2P_WRITEBACK_DEBUG_UTIL_CC = (
    REPO_ROOT
    / "rtp_llm"
    / "cpp"
    / "cache"
    / "connector"
    / "p2p"
    / "P2PWritebackDebugUtil.cc"
)
P2P_WORKER_PREFILL_CC = (
    REPO_ROOT
    / "rtp_llm"
    / "cpp"
    / "cache"
    / "connector"
    / "p2p"
    / "P2PConnectorWorkerPrefill.cc"
)
P2P_WORKER_DECODE_CC = (
    REPO_ROOT
    / "rtp_llm"
    / "cpp"
    / "cache"
    / "connector"
    / "p2p"
    / "P2PConnectorWorkerDecode.cc"
)


class PdWritebackSmokeConfigStaticTest(unittest.TestCase):

    def test_qwen3_pd_writeback_case_asserts_prefill_reuse_from_aux_info(self):
        task = json.loads(TASK_INFO.read_text())

        self.assertEqual(task["model_type"], "qwen_3")
        self.assertNotIn("FP8", task["model_path"])
        self.assertEqual(len(task["query_result"]), 2)

        for query_result in task["query_result"]:
            self.assertNotIn("tools", query_result["query"])
            self.assertNotIn("messages", query_result["query"])

        first_query = task["query_result"][0]["query"]
        second_query = task["query_result"][1]["query"]
        self.assertIn("<|im_start|>assistant", first_query["prompt"])
        self.assertGreaterEqual(first_query["generate_config"]["max_new_tokens"], 128)
        self.assertIn("1到120", first_query["prompt"])
        self.assertTrue(second_query["prompt"].startswith(first_query["prompt"]))
        self.assertIn("继续输出121到130", second_query["prompt"])

        second_assertions = task["query_result"][1]["aux_info_assertions"]
        self.assertEqual(second_assertions["mode"], "aux_info_only")
        self.assertGreaterEqual(
            second_assertions["fields"]["aux_info.prefill_local_reuse_len"]["ge"],
            64,
        )

    def test_suite_enables_feature_by_env_for_prefill_and_decode(self):
        suite = SUITE.read_text()

        self.assertIn('name="dense_pd_writeback_reuse"', suite)
        self.assertIn(
            '"prefill": ["ENABLE_PD_KV_CACHE_WRITEBACK=1", "CACHE_STORE_RDMA_MODE=1"]',
            suite,
        )
        self.assertIn(
            '"decode": ["ENABLE_PD_KV_CACHE_WRITEBACK=1", "CACHE_STORE_RDMA_MODE=1"]',
            suite,
        )
        case_block = _case_block(suite, "dense_pd_writeback_reuse")
        self.assertIn("--tp_size 1 --world_size 1", case_block)
        self.assertIn("--cache_store_rdma_mode 1", case_block)
        self.assertIn("sleep_time_qr=10", case_block)
        self.assertNotIn("--disable_flash_infer", case_block)
        self.assertNotIn("--enable_remote_cache", case_block)

    def test_qwen3_pd_writeback_tp2_case_exists(self):
        suite = SUITE.read_text()

        self.assertIn('name="dense_pd_writeback_reuse_tp2"', suite)
        case_block = _case_block(suite, "dense_pd_writeback_reuse_tp2")
        self.assertIn("--tp_size 2 --world_size 2", case_block)
        self.assertIn(
            '"prefill": ["ENABLE_PD_KV_CACHE_WRITEBACK=1", "CACHE_STORE_RDMA_MODE=1"]',
            case_block,
        )
        self.assertIn(
            '"decode": ["ENABLE_PD_KV_CACHE_WRITEBACK=1", "CACHE_STORE_RDMA_MODE=1"]',
            case_block,
        )
        self.assertIn("--cache_store_rdma_mode 1", case_block)
        self.assertNotIn("--enable_remote_cache", case_block)

        task = json.loads(TASK_INFO.read_text())
        second_assertions = task["query_result"][1]["aux_info_assertions"]
        self.assertEqual(second_assertions["mode"], "aux_info_only")
        self.assertIn("aux_info.prefill_local_reuse_len", second_assertions["fields"])

    def test_rocm_pd_writeback_stress_case_repeats_concurrent_reuse_queries(self):
        suite = ROCM_SUITE.read_text()
        task = json.loads(ROCM_TASK_INFO.read_text())

        self.assertIn('name="rocm_pd_qwen3_8b_tp1_to_tp1_tcp_writeback_reuse"', suite)
        self.assertIn(
            'name="rocm_pd_qwen3_8b_tp2_to_tp2_tcp_writeback_reuse_single"', suite
        )
        self.assertIn('name="rocm_pd_qwen3_8b_tp2_to_tp2_tcp_writeback_reuse"', suite)
        self.assertIn('name="rocm_pd_qwen3_8b_tp2_to_tp2_rdma_stress"', suite)
        self.assertIn('name = "smoke_rocm_pd_stress"', suite)
        pd_suite = suite[
            suite.index('name = "smoke_rocm_pd"') : suite.index(
                'name = "smoke_rocm_pd_stress"'
            )
        ]
        self.assertNotIn('name="rocm_pd_qwen3_8b_tp2_to_tp2_rdma_stress"', pd_suite)

        tcp_case_block = _case_block(
            suite, "rocm_pd_qwen3_8b_tp1_to_tp1_tcp_writeback_reuse"
        )
        self.assertIn(
            'task_info="data/model/qwen3/q_r_rocm_pd_writeback_reuse.json"',
            tcp_case_block,
        )
        self.assertIn("--tp_size 1 --world_size 1", tcp_case_block)
        self.assertIn("--cache_store_rdma_mode 0", tcp_case_block)
        self.assertIn("concurrency_request_count=4", tcp_case_block)
        self.assertIn("stability_repeat=2", tcp_case_block)

        tcp_tp2_single_case_block = _case_block(
            suite, "rocm_pd_qwen3_8b_tp2_to_tp2_tcp_writeback_reuse_single"
        )
        self.assertIn(
            'task_info="data/model/qwen3/q_r_rocm_pd_writeback_reuse.json"',
            tcp_tp2_single_case_block,
        )
        self.assertIn("--tp_size 2 --world_size 2", tcp_tp2_single_case_block)
        self.assertIn("--cache_store_rdma_mode 0", tcp_tp2_single_case_block)
        self.assertNotIn("concurrency_test=True", tcp_tp2_single_case_block)
        self.assertIn("read_transfer_not_done", tcp_tp2_single_case_block)

        tcp_tp2_case_block = _case_block(
            suite, "rocm_pd_qwen3_8b_tp2_to_tp2_tcp_writeback_reuse"
        )
        self.assertIn(
            'task_info="data/model/qwen3/q_r_rocm_pd_writeback_reuse.json"',
            tcp_tp2_case_block,
        )
        self.assertIn("--tp_size 2 --world_size 2", tcp_tp2_case_block)
        self.assertIn("--cache_store_rdma_mode 0", tcp_tp2_case_block)
        self.assertIn("concurrency_request_count=4", tcp_tp2_case_block)
        self.assertIn("stability_repeat=2", tcp_tp2_case_block)

        case_block = _case_block(suite, "rocm_pd_qwen3_8b_tp2_to_tp2_rdma_stress")
        self.assertIn(
            'task_info="data/model/qwen3/q_r_rocm_pd_writeback_reuse.json"', case_block
        )
        self.assertIn("concurrency_test=True", case_block)
        self.assertIn("concurrency_request_count=16", case_block)
        self.assertIn("concurrency_workers=16", case_block)
        self.assertIn("stability_repeat=10", case_block)
        self.assertIn("read_transfer_not_done", case_block)
        self.assertIn("P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE", case_block)
        self.assertIn("--tp_size 2 --world_size 2", case_block)
        self.assertIn("--cache_store_rdma_mode 1", case_block)

        self.assertEqual(task["model_type"], "qwen_3")
        self.assertEqual(len(task["query_result"]), 2)
        first_assertions = task["query_result"][0]["aux_info_assertions"]
        self.assertEqual(first_assertions["mode"], "aux_info_only")
        self.assertEqual(
            first_assertions["fields"],
            {"aux_info.pd_sep": {"eq": True}},
        )
        second_assertions = task["query_result"][1]["aux_info_assertions"]
        self.assertEqual(second_assertions["mode"], "aux_info_only")
        self.assertGreaterEqual(
            second_assertions["fields"]["aux_info.prefill_local_reuse_len"]["ge"],
            64,
        )

    def test_smoke_runner_exposes_stress_knobs_and_log_failure_assertions(self):
        defs = DEFS.read_text()
        entry = ENTRY.read_text()
        case_runner = CASE_RUNNER.read_text()

        for token in [
            "concurrency_request_count",
            "concurrency_workers",
            "stability_repeat",
            "assert_no_log_patterns",
        ]:
            self.assertIn(token, defs)
            self.assertIn(token, entry)
            self.assertIn(token, case_runner)

        self.assertIn("STABILITY_REPEAT", case_runner)
        self.assertIn("ThreadPoolExecutor(", case_runner)
        self.assertIn("max_workers=self.concurrency_workers", case_runner)
        self.assertIn("range(self.concurrency_request_count)", case_runner)
        self.assertIn("read_transfer_not_done", case_runner)
        self.assertIn("P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE", case_runner)
        self.assertIn("TEST_UNDECLARED_OUTPUTS_DIR", case_runner)

    def test_rocm_pd_writeback_boundary_cases_compare_output_with_and_without_writeback(
        self,
    ):
        suite = ROCM_SUITE.read_text()
        task = json.loads(ROCM_BOUNDARY_TASK_INFO.read_text())

        enabled_case = _case_block(
            suite, "rocm_pd_qwen3_8b_tp2_to_tp2_tcp_writeback_boundary"
        )
        disabled_case = _case_block(
            suite, "rocm_pd_qwen3_8b_tp2_to_tp2_tcp_writeback_boundary_disabled"
        )
        rdma_enabled_case = _case_block(
            suite, "rocm_pd_qwen3_8b_tp2_to_tp2_rdma_writeback_boundary"
        )
        rdma_disabled_case = _case_block(
            suite, "rocm_pd_qwen3_8b_tp2_to_tp2_rdma_writeback_boundary_disabled"
        )
        self.assertIn(
            'task_info="data/model/qwen3/q_r_rocm_pd_writeback_boundary.json"',
            enabled_case,
        )
        self.assertIn(
            'task_info="data/model/qwen3/q_r_rocm_pd_writeback_boundary.json"',
            disabled_case,
        )
        self.assertIn(
            'task_info="data/model/qwen3/q_r_rocm_pd_writeback_boundary.json"',
            rdma_enabled_case,
        )
        self.assertIn(
            'task_info="data/model/qwen3/q_r_rocm_pd_writeback_boundary.json"',
            rdma_disabled_case,
        )
        self.assertIn("ENABLE_PD_KV_CACHE_WRITEBACK=1", enabled_case)
        self.assertNotIn("ENABLE_PD_KV_CACHE_WRITEBACK=1", disabled_case)
        self.assertIn("ENABLE_PD_KV_CACHE_WRITEBACK=1", rdma_enabled_case)
        self.assertNotIn("ENABLE_PD_KV_CACHE_WRITEBACK=1", rdma_disabled_case)
        self.assertIn("PD_KV_WRITEBACK_DEBUG_CHECKSUM=1", enabled_case)
        self.assertIn("PD_KV_WRITEBACK_DEBUG_CHECKSUM=1", disabled_case)
        self.assertIn("PD_KV_WRITEBACK_DEBUG_CHECKSUM=1", rdma_enabled_case)
        self.assertIn("PD_KV_WRITEBACK_DEBUG_CHECKSUM=1", rdma_disabled_case)
        self.assertIn("--tp_size 2 --world_size 2", enabled_case)
        self.assertIn("--cache_store_rdma_mode 0", enabled_case)
        self.assertIn("--cache_store_rdma_mode 1", rdma_enabled_case)
        self.assertIn("CACHE_STORE_RDMA_MODE=1", rdma_enabled_case)
        self.assertIn("RDMA_CONNECT_RETRY_TIMES=2", rdma_enabled_case)
        self.assertIn("--seq_size_per_block 16", enabled_case)

        self.assertEqual(task["model_type"], "qwen_3")
        self.assertEqual(task["model_path"], "/mnt/data3/zhenyun.yzy/hf/Qwen3-8B")
        self.assertEqual(len(task["query_result"]), 6)
        case_names = [item["case_name"] for item in task["query_result"]]
        self.assertEqual(
            case_names,
            [
                "partial_tail_no_full_seed",
                "partial_tail_no_full_probe",
                "mixed_block_seed",
                "mixed_block_probe",
                "mixed_plus_pure_blocks_seed",
                "mixed_plus_pure_blocks_probe",
            ],
        )

        for index in [1, 3, 5]:
            query_result = task["query_result"][index]
            self.assertNotEqual(
                query_result["aux_info_assertions"].get("mode"), "aux_info_only"
            )
            self.assertIn(
                "aux_info.pd_sep", query_result["aux_info_assertions"]["fields"]
            )
            self.assertIn("response", query_result["result"])

    def test_rocm_pd_writeback_token_diff_case_compares_tcp_writeback_on_off(self):
        suite = ROCM_SUITE.read_text()
        defs = DEFS.read_text()
        entry = ENTRY.read_text()
        case_runner = CASE_RUNNER.read_text()
        multi_inst_runner = (SMOKE_ROOT / "multi_inst_case_runner.py").read_text()
        normal_comparer = NORMAL_COMPARER.read_text()
        task = json.loads(ROCM_TOKEN_DIFF_TASK_INFO.read_text())

        case_block = _case_block(
            suite, "rocm_pd_qwen3_8b_tp1_to_tp1_tcp_writeback_token_diff"
        )
        self.assertIn(
            'task_info="data/model/qwen3/q_r_rocm_pd_writeback_token_diff.json"',
            case_block,
        )
        self.assertIn("--tp_size 1 --world_size 1", case_block)
        self.assertIn("--cache_store_rdma_mode 0", case_block)
        self.assertIn("--seq_size_per_block 16", case_block)
        self.assertIn("ENABLE_PD_KV_CACHE_WRITEBACK=1", case_block)
        self.assertIn("paired_baseline_envs", case_block)
        self.assertIn("USE_CACHE_STORE=1", case_block)
        self.assertIn("CACHE_STORE_RDMA_MODE=0", case_block)
        baseline_block = case_block[case_block.index("paired_baseline_envs") :]
        self.assertNotIn("ENABLE_PD_KV_CACHE_WRITEBACK=1", baseline_block)
        self.assertIn("concurrency_test=True", case_block)
        self.assertIn("concurrency_request_count=8", case_block)
        self.assertIn("concurrency_workers=8", case_block)

        self.assertIn("paired_baseline_envs=[]", defs)
        self.assertIn("--paired_baseline_envs", defs)
        self.assertIn("--paired_baseline_envs", entry)
        self.assertIn("paired_baseline_envs", case_runner)
        self.assertIn("collect_actual_results", case_runner)
        self.assertIn("task_info: Optional[TaskInfo] = None", case_runner)
        self.assertIn("task_info = task_info or self.task_info", case_runner)
        self.assertIn("self.curl_server(curl_server_mgr, task_info)", multi_inst_runner)
        self.assertIn("token index", normal_comparer)

        self.assertEqual(task["model_type"], "qwen_3")
        self.assertEqual(task["model_path"], "/mnt/data3/zhenyun.yzy/hf/Qwen3-8B")
        self.assertEqual(len(task["query_result"]), 3)
        self.assertIn("1到300", task["query_result"][0]["query"]["prompt"])
        self.assertIn("继续输出421到900", task["query_result"][2]["query"]["prompt"])
        self.assertGreaterEqual(
            task["query_result"][2]["query"]["generate_config"]["max_new_tokens"],
            768,
        )
        for query_result in task["query_result"]:
            generate_config = query_result["query"]["generate_config"]
            self.assertGreaterEqual(generate_config["max_new_tokens"], 256)
            self.assertEqual(generate_config["temperature"], 0.0)
            self.assertEqual(generate_config["top_k"], 1)
            self.assertEqual(generate_config["top_p"], 0.1)
            self.assertTrue(generate_config["return_output_ids"])
            self.assertNotEqual(
                query_result.get("aux_info_assertions", {}).get("mode"),
                "aux_info_only",
            )
            self.assertIn("response", query_result["result"])
            self.assertIn("output_ids", query_result["result"])

    def test_rocm_pd_writeback_token_diff_has_tp2_tcp_case(self):
        suite = ROCM_SUITE.read_text()

        case_block = _case_block(
            suite, "rocm_pd_qwen3_8b_tp2_to_tp2_tcp_writeback_token_diff"
        )
        self.assertIn(
            'task_info="data/model/qwen3/q_r_rocm_pd_writeback_token_diff.json"',
            case_block,
        )
        self.assertIn("--tp_size 2 --world_size 2", case_block)
        self.assertIn("--cache_store_rdma_mode 0", case_block)
        self.assertIn("--seq_size_per_block 16", case_block)
        self.assertIn("ENABLE_PD_KV_CACHE_WRITEBACK=1", case_block)
        self.assertIn("paired_baseline_envs", case_block)
        baseline_block = case_block[case_block.index("paired_baseline_envs") :]
        self.assertNotIn("ENABLE_PD_KV_CACHE_WRITEBACK=1", baseline_block)
        self.assertNotIn("concurrency_test=True", case_block)
        self.assertNotIn("concurrency_request_count=8", case_block)
        self.assertNotIn("concurrency_workers=8", case_block)

    def test_writeback_checksum_debug_logging_is_env_gated(self):
        debug_util_cc = P2P_WRITEBACK_DEBUG_UTIL_CC.read_text()
        prefill_cc = P2P_WORKER_PREFILL_CC.read_text()
        decode_cc = P2P_WORKER_DECODE_CC.read_text()

        self.assertIn("PD_KV_WRITEBACK_DEBUG_CHECKSUM", debug_util_cc)
        self.assertIn("PD KV writeback checksum", debug_util_cc)
        self.assertIn("sampled_bytes", debug_util_cc)
        self.assertIn("decode_send_source", prefill_cc)
        self.assertIn("prefill_receive_destination", decode_cc)


def _case_block(suite: str, case_name: str) -> str:
    start = suite.index(f'name="{case_name}"')
    end = suite.index("            smoke_test(", start + 1)
    return suite[start:end]


if __name__ == "__main__":
    unittest.main()
