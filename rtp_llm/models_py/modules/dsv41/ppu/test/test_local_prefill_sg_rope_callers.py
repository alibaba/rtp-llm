"""Full/compact position provenance and host bounds before direct lookup."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import attention_layer
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_sg_cache_rows import _state
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_segments import FLAGS, compact_plan


class SGRopeCallersCPU(unittest.TestCase):
    def setUp(self):
        changes = {**FLAGS, 'RTP_V41_PPU_LOCAL_PREFILL_SG_ALIGN': '1',
                   'RTP_V41_PPU_LOCAL_PREFILL_SOURCE_CACHE': '1',
                   'RTP_V41_PPU_LOCAL_PREFILL_SOURCE_NATIVE_READ': '1',
                   'TP_SIZE': '1', 'PREFILL_CP_SIZE': '1'}
        for item in (
            patch.dict(os.environ, changes),
            patch.object(SharedNativeCache, '_local_source_native_read_device', return_value=True),
            patch('rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata._upload',
                  new=lambda rows, device, dtype: torch.tensor(rows, dtype=dtype, device=device)),
        ):
            item.start()
            self.addCleanup(item.stop)

    @torch.no_grad()
    def test_original_and_padded_compact_positions_share_host_range_proof(self):
        state = _state('cpu', 256, 4096)
        full = state.rows(None)
        frequencies = torch.empty((4352, 32), dtype=torch.complex64)
        self.assertIs(attention_layer._local_sg_rope_positions(state, None, full, frequencies), full[0])
        compact = compact_plan(state)
        rows = state.rows(compact)
        self.assertIs(attention_layer._local_sg_rope_positions(state, compact, rows, frequencies), rows[0])
        self.assertEqual(rows[0].numel(), 140)
        self.assertIsNone(attention_layer._local_sg_rope_positions(state, compact, rows, frequencies[:-1]))

    @torch.no_grad()
    def test_derived_mutation_foreign_owner_or_host_changes_decline(self):
        frequencies = torch.empty((4352, 32), dtype=torch.complex64)
        for change in ('position', 'prefix'):
            state = _state('cpu', 256, 4096)
            rows = state.rows(None)
            self.assertIsNone(attention_layer._local_sg_rope_positions(
                state, None, (rows[0].clone(), *rows[1:]), frequencies))
            if change == 'position':
                rows[0].fill_(-1)
            else:
                state.query_prefixes = (-1,)
            self.assertIsNone(attention_layer._local_sg_rope_positions(state, None, rows, frequencies))

    @torch.no_grad()
    def test_phase_table_layout_and_no_gpu_value_download(self):
        state = _state('cpu', 256, 4096)
        rows = state.rows(None)
        frequencies = torch.empty((4352, 32), dtype=torch.complex64)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('download')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar download')):
            self.assertIs(attention_layer._local_sg_rope_positions(state, None, rows, frequencies), rows[0])
        for table in (frequencies.conj(), frequencies[:, ::2], frequencies.real):
            self.assertIsNone(attention_layer._local_sg_rope_positions(state, None, rows, table))
        state.local_prefill_sg_align = False
        self.assertIsNone(attention_layer._local_sg_rope_positions(state, None, rows, frequencies))


if __name__ == '__main__':
    unittest.main()
