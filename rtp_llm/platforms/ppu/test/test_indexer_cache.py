"""Check PPU layout policy without loading the native descriptor bindings."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


class IndexerCacheTest(unittest.TestCase):
    def test_only_indexer_changes_at_final_mtp_depth(self):
        spec = importlib.util.spec_from_file_location(
            "ppu_cache", Path(__file__).parents[1] / "models/dsv4/cache.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "rtp_llm.models.dsv4_kv_cache": NS(
                INDEXER_KV_TAG="indexer_kv", INDEXER_STATE_TAG="indexer_state"),
            "rtp_llm.ops": NS(OpaqueBlockEntryCountMode=NS(EXPLICIT="explicit")),
        }):
            spec.loader.exec_module(module)
        for gamma, expected in ((0, 8), (1, 12), (2, 12), (3, 12)):
            indexer = NS(tag="indexer_kv", entry_elems=132)
            state = NS(tag="indexer_state", state_ring_overlap=1,
                       compression_ratio=4, state_ring_include_gen_num_per_cycle=True)
            other = NS(tag="csa_state", entry_elems=2048)
            config = NS(gen_num_per_cycle=gamma, kv_cache_spec_descs=[[indexer, state, other]])
            module.configure_indexer_cache(config)
            self.assertEqual(indexer.entry_elems, 68)
            self.assertEqual(state.entry_count_mode, "explicit")
            self.assertEqual(state.explicit_entry_count, expected)
            self.assertEqual(vars(other), {"tag": "csa_state", "entry_elems": 2048})
            module.configure_indexer_cache(config)
            self.assertEqual(state.explicit_entry_count, expected)
        config.gen_num_per_cycle = -1
        with self.assertRaises(ValueError):
            module.configure_indexer_cache(config)


if __name__ == "__main__":
    unittest.main()
