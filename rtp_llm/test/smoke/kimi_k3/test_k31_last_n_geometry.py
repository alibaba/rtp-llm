"""Readiness must prove the actual CP pages and P-only retention policy."""
from pathlib import Path
import tempfile
import unittest

from k31_smoke_capacity import cache_geometry


def runtime_log(role="prefill", n=5, physical=1032, linear=None, page=128):
    linear = linear or (86 if role == "prefill" else 4)
    return "".join(
        f"CP sharded KV cache enabled: cp_rank={rank}, cp_size=8, "
        f"block_size={page}, virtual_block_size={8 * page}\n"
        f"[RANK {rank}] cache manager initialized with config CacheConfig{{\n"
        "  layer_num=93\n  layer_all_num=94\n"
        f"  seq_size_per_block={page}\n  local_kv_cache_sharded=true\n"
        f"  reuse_cache_tail_blocks={n if role == 'prefill' else 2147483647}\n"
        f"  group_block_nums={physical},{physical},{linear}\n"
        f"    kernel_seq_size_per_block={page}\n"
        "  group_tags=[full,indexer_kv,linear]\n"
        "  layers_by_group=[[3,93],[3],[0]]\n"
        f"  LinearKVCacheSpec{{\n    cache_key_token_stride={page}\n}}\n"
        for rank in range(8)
    )


class LastNGeometryTest(unittest.TestCase):
    def test_calibration_checks_unbounded_policy_at_the_same_cp_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'engine.log'
            path.write_text(runtime_log(n=2147483647, physical=256, linear=256))
            proof = cache_geometry(path, 'prefill', 8, long_case=False,
                                   last_n_checkpoints=5, calibration=True)
            self.assertTrue(proof['last_n_calibration'])
            self.assertEqual((proof['page_size'], proof['checkpoint_span']), (128, 1024))
            with self.assertRaisesRegex(ValueError, 'runtime configuration'):
                cache_geometry(path, 'prefill', 8, long_case=False, last_n_checkpoints=5)

    def check_log(self, text, role="prefill", n=5, page=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"engine.log"
            path.write_text(text)
            return cache_geometry(path, role, 8, long_case=False, last_n_checkpoints=n,
                                  last_n_page_tokens=page)

    def test_b64_requires_both_page_sizes_and_capacity_for_one_million(self):
        text = runtime_log(physical=2056, linear=96, page=64)
        proof = self.check_log(text, page=64)
        self.assertEqual((proof['page_size'], proof['kernel_page_size'], proof['checkpoint_span']),
                         (64, 64, 512))
        with self.assertRaisesRegex(ValueError, '64-token kernel pages'):
            self.check_log(text.replace('kernel_seq_size_per_block=64',
                                        'kernel_seq_size_per_block=128'), page=64)
        with self.assertRaisesRegex(ValueError, 'capacity is insufficient'):
            self.check_log(runtime_log(linear=96, page=64), page=64)

    def test_runtime_geometry_and_decode_policy(self):
        for n in range(1, 6):
            for role in ("prefill", "decode"):
                with self.subTest(n=n, role=role):
                    proof = self.check_log(runtime_log(role, n), role, n)
                    self.assertEqual((proof["page_size"], proof["checkpoint_span"]), (128, 1024))
                    self.assertEqual(proof["reuse_cache_tail_blocks"], n if role == "prefill" else 2147483647)

    def test_rejects_runtime_mismatch_and_insufficient_capacity(self):
        original = runtime_log()
        for text in (
            original.replace("virtual_block_size=1024", "virtual_block_size=8192"),
            original.replace("reuse_cache_tail_blocks=5", "reuse_cache_tail_blocks=3"),
            original.replace("cache_key_token_stride=128", "cache_key_token_stride=1024"),
            runtime_log(physical=1026),
            runtime_log(linear=85),
        ):
            with self.subTest(text=text[:80]), self.assertRaises(ValueError):
                self.check_log(text)
        with self.assertRaisesRegex(ValueError, "runtime configuration"):
            self.check_log(runtime_log("decode").replace("2147483647", "5"), "decode")


if __name__ == "__main__":
    unittest.main()
