"""Exercise the PPU request-boundary type without importing native kernels."""

import ast
import __future__
from pathlib import Path
from typing import NamedTuple
import unittest


class IsolatedMetadataTest(unittest.TestCase):
    def test_cache_validator_import_and_storage_contract(self):
        import torch
        from rtp_llm.platforms.ppu.models.dsv4.attention_contract import _validate_model1_cache_tensor

        pool = torch.empty((2, 128, 584), dtype=torch.uint8)
        _validate_model1_cache_tensor(pool, block_size=128)
        with self.assertRaises(ValueError):
            _validate_model1_cache_tensor(pool.float(), block_size=128)
        with self.assertRaises(ValueError):
            _validate_model1_cache_tensor(pool.transpose(0, 1), block_size=128)
        with self.assertRaises(ValueError):
            _validate_model1_cache_tensor(pool.as_strided(pool.shape, (1, 584, 1)), block_size=128)

    def test_indexer_constructor_retains_request_score_boundaries(self):
        path = Path(__file__).parents[1] / "models/dsv4/impl/fp8/indexer.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == "_IndexerFP8PrefillMeta")
        scope = {"NamedTuple": NamedTuple, "__name__": __name__}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                     flags=__future__.annotations.compiler_flag), scope)
        cls = scope[node.name]
        boundaries = ((0, 3, 0, 2), (3, 8, 2, 6))
        values = dict.fromkeys(cls._fields)
        values["request_score_slices"] = boundaries
        metadata = cls(**values)
        self.assertIs(metadata.request_score_slices, boundaries)
        self.assertIs(metadata._replace(M=8).request_score_slices, boundaries)


if __name__ == "__main__":
    unittest.main()
