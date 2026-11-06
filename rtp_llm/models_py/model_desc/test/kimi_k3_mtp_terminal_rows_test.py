import unittest
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel


class MtpTerminalRowsTest(unittest.TestCase):
    def test_all_rank_terminal_selection_matches_full_gather_exactly(self):
        torch.manual_seed(3101)
        # Ragged packed request ends, including boundaries between SP owners.
        full = torch.randn(32, 12, dtype=torch.bfloat16)
        indexes = torch.tensor([2, 7, 8, 21, 31])
        pieces = []
        for rank in range(8):
            offset = indexes - rank * 4
            value = full[rank * 4:(rank + 1) * 4].index_select(0, offset.clamp(0, 3))
            pieces.append(torch.where(((offset >= 0) & (offset < 4))[:, None], value, 0))
        for rank in range(8):
            model = object.__new__(KimiK3MtpModel)
            nn.Module.__init__(model)
            model.tp_size, model.tp_rank = 8, rank
            with patch('rtp_llm.models_py.model_desc.kimi_k3_mtp.all_gather', return_value=torch.cat(pieces)) as gather:
                result = model._gather_terminal_rows(full[rank * 4:(rank + 1) * 4], indexes)
            torch.testing.assert_close(result, full[indexes], rtol=0, atol=0)
            self.assertEqual(tuple(gather.call_args.args[0].shape), (5, 12))
            # Recurrent H is pre-norm; normalization is exclusively the logits path.
            self.assertFalse(torch.equal(result, nn.functional.layer_norm(result, (12,))))

    def test_internal_round_has_no_output_gather(self):
        model = object.__new__(KimiK3MtpModel)
        nn.Module.__init__(model)
        model.tp_size, model.tp_rank = 8, 0
        with patch('rtp_llm.models_py.model_desc.kimi_k3_mtp.all_gather') as gather:
            result = model._gather_terminal_rows(torch.ones(8, 12), torch.empty(0, dtype=torch.long))
        self.assertEqual(tuple(result.shape), (0, 12))
        gather.assert_not_called()


if __name__ == '__main__':
    unittest.main()
