import unittest
from unittest.mock import Mock

import torch
from torch import nn

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer
from rtp_llm.models_py.model_desc.qwen3 import Qwen3DecoderLayer


class DecoderNanCheckTest(unittest.TestCase):
    def _layer(self, layer_type):
        layer = object.__new__(layer_type)
        nn.Module.__init__(layer)
        layer.layer_idx = 5
        attention = torch.zeros((3, 4))
        attention[0, 2] = float("inf")
        ffn = torch.zeros_like(attention)
        ffn[0, 0] = float("nan")
        ffn[1, 3] = float("-inf")
        layer.self_attn = Mock(return_value=attention)
        layer.mlp = Mock(return_value=ffn)
        norm = (
            (lambda hidden: hidden)
            if layer_type is Qwen3DecoderLayer
            else (lambda hidden, residual: (hidden, residual))
        )
        layer.input_layernorm = Mock(side_effect=norm)
        layer.post_attention_layernorm = Mock(side_effect=norm)
        return layer

    def test_attention_precedes_ffn_and_tokens_are_independent(self):
        for layer_type in (Qwen3DecoderLayer, GenericMoeDecoderLayer):
            with self.subTest(layer=layer_type.__name__):
                layer = self._layer(layer_type)
                records = torch.zeros((3, 4), dtype=torch.int64)
                hidden = torch.zeros((3, 4))
                if layer_type is Qwen3DecoderLayer:
                    layer(hidden, None, nan_metadata=records)
                else:
                    layer(hidden, hidden.clone(), None, nan_metadata=records)
                self.assertEqual(
                    records.tolist(),
                    [[5, 1, 2, 2], [5, 2, 3, 3], [0, 0, 0, 0]],
                )

    def test_disabled_keeps_existing_layer_outputs(self):
        layer = self._layer(Qwen3DecoderLayer)
        hidden = torch.zeros((3, 4))
        actual = layer(hidden, None)
        expected = layer.self_attn.return_value + layer.mlp.return_value
        torch.testing.assert_close(actual, expected, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
