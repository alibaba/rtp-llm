"""Exercise the PPU request-boundary type without importing native kernels."""

import ast
import __future__
from pathlib import Path
from typing import NamedTuple
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


class IsolatedMetadataTest(unittest.TestCase):
    def test_ppu_state_slot_update_replaces_reference_and_keeps_storage(self):
        import sys
        import torch
        from rtp_llm.models_py.modules.dsv4.fp8.decode import decode_attn_metadata as shared
        from rtp_llm.platforms.ppu.models.dsv4 import ppu_metadata_update as ppu

        def allocate():
            return shared.allocate_decode_metadata_fp8(
                2, 2, 128, 512, 1024, [], 512, torch.device("cpu"),
                {"csa_state": (8, 256, 2)},
            )

        reference, candidate = allocate(), allocate()
        positions = torch.tensor([255, 256], dtype=torch.int32)
        tables = {"csa_state": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)}
        kwargs = dict(
            forbid_realloc=True, paged_block_tables=tables,
            paged_pool_entries_per_block={"csa_state": 8},
            paged_pool_tokens_per_block={"csa_state": 256},
        )
        # Only GPU kernels are stubbed; state-slot arithmetic and storage
        # validation execute on real tensors through both orchestration paths.
        prefix = "rtp_llm.models_py.modules.dsv4.fp8.decode."
        modules = {
            prefix + "_fused_prepare_meta_triton": SimpleNamespace(
                fused_update_decode_meta_pure=Mock(),
                fused_phase2b_pool_slot_mapping=Mock(),
            ),
            prefix + "paged_topk_translator": SimpleNamespace(
                translate_local_to_global_slots=Mock(),
            ),
        }
        pointer = candidate.compressor_state_slot_mappings["csa_state"].data_ptr()
        original = shared._update_compressor_state_slot_mappings
        with patch.dict(sys.modules, modules):
            for offset in (0, 1, 256):
                positions.add_(offset)
                tables["csa_state"].add_(1)
                shared.update_decode_metadata_in_place_fp8(reference, positions, **kwargs)
                fused = Mock(wraps=original)
                with patch.object(ppu, "_update_compressor_state_slot_mappings") as fallback:
                    ppu.update_decode_metadata_in_place_ppu(
                        candidate, positions, compressor_state_slot_updater=fused, **kwargs
                    )
                    fallback.assert_not_called()
                fused.assert_called_once_with(candidate, 2, {"csa_state": 8})
                self.assertTrue(torch.equal(
                    reference.compressor_state_slot_mappings["csa_state"],
                    candidate.compressor_state_slot_mappings["csa_state"],
                ))
                self.assertEqual(pointer, candidate.compressor_state_slot_mappings["csa_state"].data_ptr())

            with patch.object(ppu, "_update_compressor_state_slot_mappings", wraps=original) as fallback:
                ppu.update_decode_metadata_in_place_ppu(candidate, positions, **kwargs)
                fallback.assert_called_once()

            def realloc(meta, batch, entries):
                meta.compressor_state_slot_mappings["csa_state"] = torch.zeros_like(
                    meta.compressor_state_slot_mappings["csa_state"]
                )

            with self.assertRaisesRegex(AssertionError, "reallocated buffer"):
                ppu.update_decode_metadata_in_place_ppu(
                    candidate, positions, compressor_state_slot_updater=realloc, **kwargs
                )

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
