import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3DecoderLayer


class RecordingResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.write_indices = []

    def forward(self, hidden, anchors, *, block_write_idx=-1, **kwargs):
        self.write_indices.append(block_write_idx)
        if block_write_idx >= 0:
            anchors[:, block_write_idx].copy_(hidden)
        return hidden


class ZeroAttention(nn.Module):
    def forward(self, hidden, *args):
        return torch.zeros_like(hidden)


class ZeroMLP(nn.Module):
    def forward(self, hidden, valid_mask):
        return torch.zeros_like(hidden)


class KimiK3AnchorWriteTest(unittest.TestCase):
    def test_first_layer_writes_anchor_inside_residual(self):
        layer = object.__new__(KimiK3DecoderLayer)
        nn.Module.__init__(layer)
        layer.index = 0
        layer.block_size = 12
        layer.attention_norm = SimpleNamespace(weight=None, variance_epsilon=1e-6)
        layer.mlp_norm = SimpleNamespace(weight=None, variance_epsilon=1e-6)
        layer.attention_residual = RecordingResidual()
        layer.mlp_residual = RecordingResidual()
        layer.attention = ZeroAttention()
        layer.mlp = ZeroMLP()

        hidden = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
        anchors = torch.full((2, 1, 8), -1, dtype=torch.bfloat16)
        result = layer.forward(hidden, anchors, None, None, None, None, None)

        self.assertEqual(layer.attention_residual.write_indices, [0])
        torch.testing.assert_close(anchors[:, 0], hidden, rtol=0, atol=0)
        torch.testing.assert_close(result, torch.zeros_like(hidden), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
