import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel


class KimiK3MtpCpHiddenTest(unittest.TestCase):
    def model(self, cls=KimiK3Model):
        model = object.__new__(cls)
        nn.Module.__init__(model)
        model.tp_size, model.tp_rank = 8, 2
        model.prefill_query_cp = True
        model.config = SimpleNamespace(
            moe_prefill_max_tokens_per_rank=32, max_seq_len=32, hidden_size=4
        )
        model.norm = nn.LayerNorm(4, dtype=torch.bfloat16)
        model.register_buffer('_mtp_prefill_hidden_buffer', None, persistent=False)
        model._mtp_prefill_hidden_valid_rows = 0
        model.register_buffer('_mtp_prefill_last_hidden_buffer', None, persistent=False)
        model._mtp_prefill_last_hidden_valid_rows = 0
        model._max_generate_batch_size = 4
        model._mtp_target_hidden_states = None
        model._query_layout = None
        return model

    def initialize(self, model, *, speculative=True, requests=2):
        model._initialize_mtp_hidden_handoff(SimpleNamespace(
            is_speculative=speculative, max_context_batch_size=requests
        ))

    def test_capability_requires_real_storage(self):
        model = self.model()
        self.assertFalse(model.has_mtp_hidden_buffer())
        self.assertIsNone(model.get_mtp_target_hidden_states(-1))
        self.initialize(model, speculative=False)
        self.assertFalse(model.has_mtp_hidden_buffer())
        model.prefill_query_cp = False
        self.initialize(model)
        self.assertFalse(model.has_mtp_hidden_buffer())
        model.prefill_query_cp = True
        self.initialize(model)
        self.assertTrue(model.has_mtp_hidden_buffer())
        self.assertEqual(tuple(model._mtp_prefill_hidden_buffer.shape), (8, 4))

    def test_same_capacity_reuses_address_and_growth_is_rejected(self):
        model = self.model()
        self.initialize(model)
        pointer = model._mtp_prefill_hidden_buffer.data_ptr()
        self.initialize(model)
        self.assertEqual(pointer, model._mtp_prefill_hidden_buffer.data_ptr())
        with self.assertRaisesRegex(ValueError, 'Cannot replace'):
            self.initialize(model, requests=3)
        self.assertEqual(pointer, model._mtp_prefill_hidden_buffer.data_ptr())

    def test_rank_local_pre_norm_padding_and_stride_are_preserved(self):
        model = self.model()
        self.initialize(model)
        model._query_layout = SimpleNamespace(local_rows=4)
        hidden = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)[:, ::2]
        hidden[-1].zero_()
        expected = hidden.clone()
        published = model._publish_mtp_prefill_hidden(hidden)
        hidden.zero_()
        torch.testing.assert_close(published, expected, rtol=0, atol=0)
        torch.testing.assert_close(model.get_mtp_target_hidden_states(-1), expected, rtol=0, atol=0)
        torch.testing.assert_close(model.get_mtp_target_hidden_states(2), expected[:2], rtol=0, atol=0)
        self.assertEqual(published.data_ptr(), model._mtp_prefill_hidden_buffer.data_ptr())
        self.assertFalse(torch.equal(model.norm(expected), published))
        with self.assertRaisesRegex(ValueError, 'requested=5, written=4'):
            model.get_mtp_target_hidden_states(5)

    def test_empty_unwritten_overflow_and_wrong_layout_are_rejected(self):
        model = self.model()
        self.initialize(model)
        with self.assertRaisesRegex(ValueError, 'written=0'):
            model.get_mtp_target_hidden_states(-1)
        model._query_layout = SimpleNamespace(local_rows=9)
        with self.assertRaisesRegex(ValueError, 'rank-local CP storage'):
            model._publish_mtp_prefill_hidden(torch.ones(9, 4, dtype=torch.bfloat16))
        model._query_layout.local_rows = 4
        with self.assertRaisesRegex(ValueError, 'rank-local CP storage'):
            model._publish_mtp_prefill_hidden(torch.ones(3, 4, dtype=torch.bfloat16))

    def test_target_and_draft_storage_are_independent(self):
        target, draft = self.model(), self.model(KimiK3MtpModel)
        for model in (target, draft):
            self.initialize(model)
            model._query_layout = SimpleNamespace(local_rows=4)
        expected = torch.full((4, 4), 2, dtype=torch.bfloat16)
        target._publish_mtp_prefill_hidden(expected)
        draft._publish_mtp_prefill_hidden(torch.full_like(expected, 3))
        self.assertNotEqual(target._mtp_prefill_hidden_buffer.data_ptr(), draft._mtp_prefill_hidden_buffer.data_ptr())
        torch.testing.assert_close(target.get_mtp_target_hidden_states(-1), expected, rtol=0, atol=0)

    def test_cp_projection_uses_all_local_rows_without_gather_or_second_split(self):
        torch.manual_seed(3101)
        model = self.model(KimiK3MtpModel)
        model._query_layout = SimpleNamespace(local_rows=4, local_valid=torch.tensor([True, True, True, False]))
        model.embed_tokens = nn.Embedding(32, 4)
        model.enorm = nn.LayerNorm(4)
        model.hnorm = nn.LayerNorm(4)
        model.eh_proj = nn.Linear(8, 4, bias=False)
        inputs = SimpleNamespace(input_ids=torch.tensor([1, 2, 3, 0]),
            combo_position_ids=torch.tensor([0, 1, 9, 0]), input_hiddens=torch.randn(4, 4))
        embedded = model.embed_tokens(inputs.input_ids) * torch.tensor([False, True, True, False])[:, None]
        expected = model.eh_proj(torch.cat((model.enorm(embedded), model.hnorm(inputs.input_hiddens)), -1))
        with patch.object(model, '_gather_modeling_boundary', side_effect=AssertionError('unexpected gather')):
            actual = model._project_local_mtp_input(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tuple(actual.shape), (4, 4))
        inputs.input_hiddens = inputs.input_hiddens[:1]
        with self.assertRaisesRegex(ValueError, 'rank-local rows'):
            model._project_local_mtp_input(inputs)

    def test_cp_projection_without_explicit_positions_uses_global_layout(self):
        torch.manual_seed(3101)
        model = self.model(KimiK3MtpModel)
        model._query_layout = SimpleNamespace(local_rows=4,
            local_positions=torch.tensor([0, 1, 65535, -1]),
            local_valid=torch.tensor([True, True, True, False]))
        model.embed_tokens = nn.Embedding(32, 4)
        model.enorm, model.hnorm = nn.LayerNorm(4), nn.LayerNorm(4)
        model.eh_proj = nn.Linear(8, 4, bias=False)
        inputs = SimpleNamespace(input_ids=torch.tensor([1,2,3,0]),
            combo_position_ids=torch.empty(0, dtype=torch.int32),
            input_hiddens=torch.randn(4,4))
        embedded = model.embed_tokens(inputs.input_ids) * torch.tensor([False,True,True,False])[:,None]
        expected = model.eh_proj(torch.cat((model.enorm(embedded),model.hnorm(inputs.input_hiddens)),-1))
        for missing in (None, torch.empty(0, dtype=torch.int32)):
            inputs.combo_position_ids = missing
            torch.testing.assert_close(model._project_local_mtp_input(inputs),expected,rtol=0,atol=0)
        # Cached non-contiguous global positions must not become local row IDs.
        model._query_layout.local_positions = torch.tensor([8192,8193,65535,-1])
        embedded = model.embed_tokens(inputs.input_ids) * torch.tensor([True,True,True,False])[:,None]
        expected = model.eh_proj(torch.cat((model.enorm(embedded),model.hnorm(inputs.input_hiddens)),-1))
        torch.testing.assert_close(model._project_local_mtp_input(inputs),expected,rtol=0,atol=0)

    def test_cp_model_output_keeps_local_pre_norm_recurrence(self):
        model = self.model(KimiK3MtpModel)
        self.initialize(model)
        model._query_layout = SimpleNamespace(local_rows=4)
        hidden = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
        with (
            patch.object(model, '_project_local_mtp_input', return_value=hidden),
            patch.object(model, '_forward_layers', return_value=hidden),
            patch.object(model, '_gather_modeling_boundary', side_effect=AssertionError('unexpected gather')),
        ):
            output = model._forward_single(SimpleNamespace())
        torch.testing.assert_close(output.mtp_target_hidden_states, hidden, rtol=0, atol=0)
        torch.testing.assert_close(output.hidden_states, model.norm(hidden), rtol=0, atol=0)

    def ragged_layouts(self, lengths=(3, 17, 2), prefixes=(0, 2048, 4095)):
        from rtp_llm.models_py.modules.hybrid.query_layout import QueryLayout
        size = 8
        padded = [((length + 2 * size - 1) // (2 * size)) * 2 * size for length in lengths]
        rankmajor = []
        for rank in range(size):
            offset = 0
            for length in padded:
                pair = length // (2 * size)
                rankmajor.extend(range(offset + rank * pair, offset + (rank + 1) * pair))
                rankmajor.extend(range(offset + length - (rank + 1) * pair, offset + length - rank * pair))
                offset += length
        rankmajor = torch.tensor(rankmajor)
        restore = torch.empty(sum(padded), dtype=torch.int64)
        restore[rankmajor] = torch.arange(sum(padded))
        valid = torch.cat([torch.arange(physical) < real for physical, real in zip(padded, lengths)])
        info = SimpleNamespace(prefill_actual_input_lengths_cpu=torch.tensor(lengths, dtype=torch.int32),
            prefill_prefix_lengths_cpu=torch.tensor(prefixes, dtype=torch.int32),
            prefill_qkv_restore_indice=restore, prefill_qkv_padding_mask=valid)
        layouts = [QueryLayout(info, size, rank, 'cpu') for rank in range(size)]
        natural = torch.arange(sum(lengths) * 4, dtype=torch.bfloat16).reshape(-1, 4)
        hidden = torch.full((sum(padded), 4), float('nan'), dtype=torch.bfloat16)
        hidden.index_copy_(0, layouts[0].natural_to_rankmajor, natural)
        local = list(hidden.chunk(size))
        expected = natural[torch.tensor(lengths).cumsum(0) - 1]
        return layouts, local, expected

    def test_ragged_cp_selects_one_real_last_row_per_request(self):
        layouts, local, expected = self.ragged_layouts()
        values = []
        owners = []
        for layout, hidden in zip(layouts, local):
            layout.prepare_last_hidden_rows()
            values.append(layout.local_last_hidden(hidden))
            owners.append(layout.last_is_mine)
        torch.testing.assert_close(torch.stack(owners).sum(0), torch.ones(3, dtype=torch.int64), rtol=0, atol=0)
        torch.testing.assert_close(torch.stack(values).sum(0), expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(torch.stack(values)).all())
        self.assertEqual(torch.stack(owners).nonzero()[:, 0].tolist(), [1, 2, 7])

    def test_last_hidden_handoff_gathers_only_request_rows_and_keeps_pre_norm(self):
        layouts, local, expected = self.ragged_layouts()
        for layout in layouts:
            layout.prepare_last_hidden_rows()
        wire = torch.cat([layout.local_last_hidden(hidden) for layout, hidden in zip(layouts, local)])
        model = self.model(KimiK3MtpModel)
        self.initialize(model)
        model._initialize_mtp_last_handoff(SimpleNamespace(max_context_batch_size=3))
        pointer = model._mtp_prefill_last_hidden_buffer.data_ptr()
        model._initialize_mtp_last_handoff(SimpleNamespace(max_context_batch_size=3))
        self.assertEqual(pointer, model._mtp_prefill_last_hidden_buffer.data_ptr())
        with self.assertRaisesRegex(ValueError, 'written=0'):
            model.get_mtp_last_hidden_states(-1)
        model._query_layout = layouts[2]
        seen = []
        def gather(value, group):
            seen.append(tuple(value.shape))
            torch.testing.assert_close(value, wire.reshape(8, 3, 4)[2], rtol=0, atol=0)
            return wire
        with patch('rtp_llm.models_py.model_desc.kimi_k3_mtp.all_gather', side_effect=gather):
            model._publish_mtp_last_hidden(local[2])
        self.assertEqual(seen, [(3, 4)])
        torch.testing.assert_close(model.get_mtp_last_hidden_states(3), expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(model.norm(expected), model.get_mtp_last_hidden_states(-1)))
        self.assertEqual(pointer, model._mtp_prefill_last_hidden_buffer.data_ptr())
        with self.assertRaisesRegex(ValueError, 'written=3'):
            model.get_mtp_last_hidden_states(4)

    def test_last_hidden_small_ambiguous_global_and_local_row_counts(self):
        layouts, local, expected = self.ragged_layouts((2,), (4095,))
        self.assertEqual(layouts[0].true_rows, layouts[0].local_rows)
        values = []
        for layout, hidden in zip(layouts, local):
            layout.prepare_last_hidden_rows()
            values.append(layout.local_last_hidden(hidden))
        torch.testing.assert_close(torch.stack(values).sum(0), expected, rtol=0, atol=0)

    def test_last_hidden_unavailable_on_decode_or_target_only(self):
        model = self.model(KimiK3MtpModel)
        model.prefill_query_cp = False
        self.initialize(model)
        model._initialize_mtp_last_handoff(SimpleNamespace(max_context_batch_size=3))
        self.assertIsNone(model.get_mtp_last_hidden_states(-1))


if __name__ == '__main__':
    unittest.main()
