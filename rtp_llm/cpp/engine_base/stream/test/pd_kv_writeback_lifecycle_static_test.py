import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]


class PdKvWritebackLifecycleStaticTest(unittest.TestCase):
    def test_decode_release_launches_writeback_before_free(self):
        resource_context_h = (
            REPO_ROOT / "rtp_llm/cpp/engine_base/stream/ResourceContext.h"
        ).read_text()
        stream_h = (
            REPO_ROOT / "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
        ).read_text()
        stream_cc = (
            REPO_ROOT / "rtp_llm/cpp/engine_base/stream/StreamCacheResource.cc"
        ).read_text()
        normal_engine_cc = (
            REPO_ROOT / "rtp_llm/cpp/normal_engine/NormalEngine.cc"
        ).read_text()

        self.assertIn("enable_pd_kv_cache_writeback", resource_context_h)
        self.assertIn("pd_kv_writeback_launcher", resource_context_h)
        self.assertIn("pd_kv_writeback_partition_count", resource_context_h)
        self.assertIn("pd_kv_writeback_tp_rank", resource_context_h)

        self.assertIn("maybeLaunchPdKvWriteback", stream_h)
        self.assertIn("buildPdKvWritebackManifest", stream_cc)
        self.assertIn("launchFromDecode", stream_cc)
        self.assertIn("RoleType::DECODE", stream_cc)
        self.assertIn(
            "request.local_tp_rank             = resource_context_.pd_kv_writeback_tp_rank",
            stream_cc,
        )
        self.assertNotIn("pd_kv_writeback_tp_rank != 0", stream_cc)
        self.assertIn("incrKVCacheRef", stream_cc)
        self.assertIn("request.held_resource", stream_cc)
        self.assertIn("rtp_llm/cpp/cache/KVCacheHashUtil.h", stream_cc)
        self.assertIn("updateCacheKeys(batch_kv_cache_resource_", stream_cc)
        self.assertLess(
            stream_cc.index("updateCacheKeys(batch_kv_cache_resource_"),
            stream_cc.index("buildPdKvWritebackManifest(snapshot)"),
        )
        self.assertLess(
            stream_cc.index("maybeLaunchPdKvWriteback();"),
            stream_cc.index("resource_context_.cache_manager->free(free_info);"),
        )

        self.assertIn("pd_sep_config.enable_pd_kv_cache_writeback", normal_engine_cc)
        self.assertIn("parallelism_config.tp_size", normal_engine_cc)
        self.assertIn("parallelism_config.tp_rank", normal_engine_cc)


if __name__ == "__main__":
    unittest.main()
