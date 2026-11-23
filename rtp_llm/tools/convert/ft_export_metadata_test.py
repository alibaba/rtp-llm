"""Disk-backed tests for public FT artifact index publication (no native model)."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index


class FTExportIndexTest(unittest.TestCase):
    def test_head_source_must_agree_across_ranks(self):
        for other in (None, "lm_head"):
            with self.subTest(other=other), tempfile.TemporaryDirectory() as tmp:
                for rank, source in enumerate(("embedding", other)):
                    meta = {"ft_lm_head_source": source} if source else None
                    save_file(
                        {f"rank_0{rank}_00_00.global.lm_head": torch.ones(2, 4)},
                        str(Path(tmp) / f"rank-{rank}.safetensors"),
                        metadata=meta,
                    )
                with self.assertRaisesRegex(ValueError, "provenance"):
                    generate_ft_index(tmp, {})
                self.assertFalse((Path(tmp) / "model.safetensors.index.json").exists())

    def test_rank_coordinates_and_quantized_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            ranks = ["rank_00_00_00", "rank_01_00_01"]
            wanted = {}
            for i, rank in enumerate(ranks):
                path = dest / f"model-0{i}-00-part-00000.safetensors"
                key = rank + ".layers.0.ffn_weights.intermediate_weight13.kernel"
                save_file({key: torch.tensor([[1, 2]], dtype=torch.int8)}, str(path))
                wanted[key] = path.name
            env = {"TP_SIZE": "2", "EP_SIZE": "0"}
            generate_ft_index(tmp, env, expected_ranks=ranks)
            index = json.loads((dest / "model.safetensors.index.json").read_text())
            self.assertEqual(index["weight_map"], wanted)
            self.assertEqual(index["__env__params__"], env)
            self.assertNotIn("module_export_metadata", index)

    def test_bad_shards_or_missing_rank_cannot_publish_an_index(self):
        for problem in ("empty", "corrupt", "duplicate", "missing", "duplicate_rank"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp)
                key = "rank_00_00_00.global.embedding"
                records = {key: torch.ones(1)}
                ranks = ["rank_00_00_00"]
                if problem != "empty":
                    save_file(records, str(dest / "part-0.safetensors"))
                if problem == "duplicate":
                    save_file(records, str(dest / "part-1.safetensors"))
                elif problem == "corrupt":
                    (dest / "part-1.safetensors").write_bytes(b"truncated header")
                elif problem == "missing":
                    ranks.append("rank_01_00_01")
                elif problem == "duplicate_rank":
                    ranks += ranks
                with self.assertRaises(Exception):
                    generate_ft_index(tmp, {}, expected_ranks=ranks)
                self.assertFalse((dest / "model.safetensors.index.json").exists())


if __name__ == "__main__":
    unittest.main()
