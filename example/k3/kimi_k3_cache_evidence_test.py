from __future__ import annotations

import datetime
import json
import pathlib
import tempfile
import unittest

from example.k3.kimi_k3_cache_evidence import (
    CacheEvidence,
    main,
)


def log_line(message, rank=0, epoch_s=1_788_900_000):
    timestamp = datetime.datetime.fromtimestamp(epoch_s).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )
    return f"[{timestamp}] [INFO] [100:101] [RANK {rank}][engine.cc:1] {message}\n"


def startup_log(tp_size=8):
    lines = []
    for rank in range(tp_size):
        for group, count in enumerate((4, 4, 9)):
            lines.append(
                log_line(
                    f"BlockPool backing selected: pool_name=group_{group} allocation_type=DEVICE "
                    f"requested_backing=CUDA actual_backing=GPU is_cuda=1 is_pinned=0 "
                    f"ptr=0x123 total_size={count * 4096} bytes block_num={count} memory_layouts=1",
                    rank,
                )
            )
        lines.append(
            log_line(
                "KVCacheAllocator set reserve blocks: ratio=5% reserve_blocks=1 available_blocks=14",
                rank,
            )
        )
        lines.append(
            log_line("init connector coordinator, cache config: [CacheConfig{", rank)
        )
        lines.extend(
            [
                "  seq_size_per_block=4096\n  kernel_seq_size_per_block=128\n",
                "  linear_step=1\n  group_block_nums=4,4,9\n",
                "  cache_specs[0] {\n    layer_num=1\n    seq_size_per_block=4096\n  }\n",
                "  cache_specs[1] {\n    layer_num=2\n    seq_size_per_block=32768\n  }\n",
                "  cache_specs[2] {\n    layer_num=1\n    seq_size_per_block=4096\n  }\n",
                "  global_layer_ids=[[3], [0,1], [4]]\n  group_types=[1,0,2]\n",
                "  mtp_sub_configs[0]:\n    CacheConfig{\n      group_block_nums=9\n    }\n",
                "}\n], kv cache config: [reuse_cache: 1\nreserve_block_ratio: 5\n]\n",
            ]
        )
    return "".join(lines)


def read(text):
    evidence = CacheEvidence()
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        evidence.feed(line, f"engine.log:{number}")
    evidence.finish_record()
    return evidence


class CacheEvidenceTest(unittest.TestCase):

    def test_real_merged_format_preserves_group_units_and_reserved_zero(self):
        evidence = read(startup_log())
        inventory = evidence.inventory("prefill", 8, 8)
        self.assertEqual(len(inventory), 24)
        self.assertEqual([r["usable_blocks"] for r in inventory[:3]], [3, 3, 8])
        self.assertEqual(
            [r["logical_block_tokens"] for r in inventory[:3]], [32768, 32768, 4096]
        )
        self.assertEqual(
            [r["spec_seq_size_per_block"] for r in inventory[:3]], [4096, 32768, 4096]
        )
        self.assertEqual(inventory[1]["layer_ids"], [0, 1])
        self.assertEqual(inventory[0]["allocator_reserve_blocks"], 1)

    def test_interleaved_long_config_uses_single_line_allocator_reserve(self):
        text = startup_log(2)
        # Reproduce a foreign log record inserted mid-line in the long MTP
        # sub-config, before the trailing reserve_block_ratio is printed.
        text = text.replace(
            "  mtp_sub_configs[0]:\n    CacheConfig{\n      group_block_nums=9\n    }\n}\n], kv cache config: [reuse_cache: 1\nreserve_block_ratio: 5\n]\n",
            "  mtp_sub_configs[0]:\n    CacheConfig{\n      group_block_nums=9" + log_line("unrelated startup message"),
        )
        text = text.replace(
            log_line("KVCacheAllocator set reserve blocks: ratio=5% reserve_blocks=1 available_blocks=14", 1),
            "continuation" + log_line("KVCacheAllocator set reserve blocks: ratio=5% reserve_blocks=1 available_blocks=14", 1),
        )
        inventory = read(text).inventory("prefill", 2, 8)
        self.assertEqual(len(inventory), 6)
        self.assertTrue(all(row["reserve_ratio_percent"] == 5 for row in inventory))
        self.assertTrue(all(row["allocator_reserve_blocks"] == 1 for row in inventory))
        # Missing evidence must remain an error, not an inferred zero reserve.
        broken = text.replace("KVCacheAllocator set reserve blocks:", "irrelevant:")
        with self.assertRaisesRegex(ValueError, "missing reserve budget evidence"):
            read(broken).inventory("prefill", 2, 8)

    def test_missing_rank_and_mismatched_backing_are_not_inventory(self):
        evidence = read(startup_log(1))
        with self.assertRaisesRegex(ValueError, "missing merged configs"):
            evidence.inventory("prefill", 2, 8)
        evidence.backings[(0, "group_0")]["block_num"] = "5"
        with self.assertRaisesRegex(ValueError, "mismatched allocated pool"):
            evidence.inventory("prefill", 1, 8)

    def test_cli_saves_snapshots_without_waiting_for_reclamation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            engine = root / "engine.log"
            engine.write_text(startup_log())
            args = [
                "--engine-log",
                str(engine),
                "--output-dir",
                str(root / "cache"),
                "--role",
                "prefill",
                "--tp-size",
                "8",
                "--cp-size",
                "8",
            ]
            self.assertEqual(main(args + ["--phase", "startup"]), 0)
            self.assertTrue((root / "cache/pool_inventory.csv").is_file())
            self.assertEqual(main(args + ["--phase", "final"]), 0)
            final = json.loads((root / "cache/final.json").read_text())
            self.assertNotIn("drain", final)
            self.assertEqual(len(final["inventory"]), 24)


if __name__ == "__main__":
    unittest.main()
