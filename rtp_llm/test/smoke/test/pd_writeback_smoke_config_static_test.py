import json
import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SMOKE_ROOT = REPO_ROOT / "rtp_llm" / "test" / "smoke"
TASK_INFO = SMOKE_ROOT / "data" / "model" / "qwen3" / "q_r_h20_pd_writeback_reuse.json"
SUITE = SMOKE_ROOT / "suites_h20_oss.bzl"


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


def _case_block(suite: str, case_name: str) -> str:
    start = suite.index(f'name="{case_name}"')
    end = suite.index("            smoke_test(", start + 1)
    return suite[start:end]


if __name__ == "__main__":
    unittest.main()
