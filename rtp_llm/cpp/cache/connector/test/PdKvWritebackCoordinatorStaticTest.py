import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]


class PdKvWritebackCoordinatorStaticTest(unittest.TestCase):
    def test_writeback_manager_is_initialized_outside_generic_connectors(self):
        coordinator_h = (
            REPO_ROOT / "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
        ).read_text()
        coordinator_cc = (
            REPO_ROOT / "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.cc"
        ).read_text()
        kv_cache_manager_h = (
            REPO_ROOT / "rtp_llm/cpp/cache/KVCacheManager.h"
        ).read_text()
        kv_cache_manager_cc = (
            REPO_ROOT / "rtp_llm/cpp/cache/KVCacheManager.cc"
        ).read_text()
        prefill_cc = (
            REPO_ROOT / "rtp_llm/cpp/model_rpc/PrefillRpcServer.cc"
        ).read_text()
        normal_engine_cc = (
            REPO_ROOT / "rtp_llm/cpp/normal_engine/NormalEngine.cc"
        ).read_text()

        self.assertIn("PdKvWritebackManagerPtr", coordinator_h)
        self.assertIn("pd_kv_writeback_manager_", coordinator_h)
        self.assertIn("writebackManager() const", coordinator_h)
        self.assertIn("connectorCountForTest() const", coordinator_h)
        self.assertIn("initPdKvWriteback", coordinator_cc)
        self.assertIn("enable_pd_kv_cache_writeback", coordinator_cc)
        self.assertIn("pd_sep_config_.decode_entrance", coordinator_cc)
        self.assertIn("cache_store_listen_port + 1", coordinator_cc)
        self.assertIn("cache_store_rdma_mode = false", coordinator_cc)
        self.assertNotIn(
            "connectors_.emplace_back(pd_kv_writeback_manager_", coordinator_cc
        )

        self.assertIn("writebackManager() const", kv_cache_manager_h)
        self.assertIn("coordinator_->writebackManager()", kv_cache_manager_cc)
        self.assertIn(
            "writeback_manager_ = engine_->resourceContext().cache_manager->writebackManager()",
            prefill_cc,
        )
        self.assertIn(
            "pd_kv_writeback_launcher = resource_context_.cache_manager->writebackManager()",
            normal_engine_cc,
        )
        self.assertIn("pd_sep_config", normal_engine_cc)
        self.assertIn("cache_store_config", normal_engine_cc)

    def test_enabled_writeback_init_failure_fails_service_startup(self):
        coordinator_h = (
            REPO_ROOT / "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
        ).read_text()
        coordinator_cc = (
            REPO_ROOT / "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.cc"
        ).read_text()

        self.assertRegex(coordinator_h, r"bool\s+initPdKvWriteback\(\);")
        self.assertIn("if (!initPdKvWriteback())", coordinator_cc)
        self.assertIn("init PD KV writeback worker failed", coordinator_cc)
        self.assertIn("return false;", coordinator_cc)
        self.assertNotIn("writeback disabled", coordinator_cc)


if __name__ == "__main__":
    unittest.main()
