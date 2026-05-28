import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


class PDSepWritebackConfigStaticTest(unittest.TestCase):
    def test_pd_kv_writeback_config_is_wired_across_layers(self):
        pd_args = (
            REPO_ROOT / "rtp_llm/server/server_args/pd_separation_group_args.py"
        ).read_text()
        config_h = (REPO_ROOT / "rtp_llm/cpp/config/ConfigModules.h").read_text()
        config_cc = (REPO_ROOT / "rtp_llm/cpp/config/ConfigModules.cc").read_text()
        pybind = (REPO_ROOT / "rtp_llm/cpp/pybind/ConfigInit.cc").read_text()
        pyi = (REPO_ROOT / "rtp_llm/ops/libth_transformer_config.pyi").read_text()

        self.assertIn("--enable_pd_kv_cache_writeback", pd_args)
        self.assertIn("ENABLE_PD_KV_CACHE_WRITEBACK", pd_args)
        self.assertIn('"enable_pd_kv_cache_writeback"', pd_args)

        self.assertIn("enable_pd_kv_cache_writeback", config_h)
        self.assertIn("enable_pd_kv_cache_writeback", config_cc)

        self.assertIn(
            '.def_readwrite("enable_pd_kv_cache_writeback"',
            pybind,
        )
        self.assertIn("t.size() != 21", pybind)
        self.assertIn("t[20].cast<bool>()", pybind)

        self.assertIn("enable_pd_kv_cache_writeback: bool", pyi)


if __name__ == "__main__":
    unittest.main()
