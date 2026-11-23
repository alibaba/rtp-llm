"""Keep offline branch analysis out of native node mutation and eviction ordering."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
BTC = ROOT / "rtp_llm/cpp/cache/block_tree_cache"
BASELINE = "0798492cdc5bdf85954f014fee71c9e16cb9e9fd"


class AnalysisBoundary(unittest.TestCase):
    def test_native_tree_and_victim_order_are_upstream(self):
        for relative in ("TreeNode.h", "BlockTree.cc", "evict/EvictionHeap.cc"):
            with self.subTest(file=relative):
                path = BTC / relative
                expected = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                                    BASELINE + ":" + str(path.relative_to(ROOT))])
                self.assertEqual(path.read_bytes(), expected)

    def test_analysis_state_is_not_in_native_headers(self):
        for relative in ("BlockTree.h", "BlockTreeCache.h"):
            text = (BTC / relative).read_text()
            for name in ("MatchHistory", "branch_hits", "observation_timestamp_ns_", "enable_match_history"):
                with self.subTest(file=relative, name=name):
                    self.assertNotIn(name, text)

    def test_legacy_patch_cannot_reintroduce_analysis(self):
        self.assertFalse((BTC / "skills/btc-offline-replay/adapters/metadata-replay.patch").exists())
        self.assertFalse((BTC / "skills/btc-offline-replay/scripts/native_reader.py").exists())


if __name__ == "__main__":
    unittest.main()
