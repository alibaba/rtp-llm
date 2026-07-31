import unittest

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


def _capture_harness(num_layers: int = 4) -> V4Transformer:
    transformer = V4Transformer.__new__(V4Transformer)
    nn.Module.__init__(transformer)
    transformer.layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
    transformer.capture_aux_hidden_layer_ids = ()
    transformer._aux_hidden_states = None
    return transformer


class AuxHiddenCaptureTest(unittest.TestCase):
    def test_capture_reduces_hc_and_preserves_configured_layer_order(self) -> None:
        transformer = _capture_harness()
        transformer.set_aux_hidden_capture_layer_ids((2, 0))
        captured = transformer.begin_aux_hidden_capture()

        layer0 = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        layer2 = layer0 + 100
        transformer.capture_aux_hidden(captured, 0, layer0)
        transformer.capture_aux_hidden(captured, 2, layer2)
        aux = transformer.finish_aux_hidden_capture(captured)

        self.assertEqual(aux.shape, (2, 2, 3))
        torch.testing.assert_close(aux[:, 0], layer2.mean(dim=-2))
        torch.testing.assert_close(aux[:, 1], layer0.mean(dim=-2))

    def test_decode_shape_keeps_batch_and_query_axes(self) -> None:
        transformer = _capture_harness()
        transformer.set_aux_hidden_capture_layer_ids((1,))
        captured = transformer.begin_aux_hidden_capture()

        hidden = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(
            2, 3, 4, 5
        )
        transformer.capture_aux_hidden(captured, 1, hidden)
        aux = transformer.finish_aux_hidden_capture(captured)

        self.assertEqual(aux.shape, (2, 3, 1, 5))
        torch.testing.assert_close(aux[..., 0, :], hidden.mean(dim=-2))

    def test_empty_configuration_has_no_capture(self) -> None:
        transformer = _capture_harness()
        self.assertIsNone(transformer.begin_aux_hidden_capture())
        self.assertIsNone(transformer._aux_hidden_states)

    def test_invalid_layer_configuration_is_rejected(self) -> None:
        transformer = _capture_harness()
        with self.assertRaises(ValueError):
            transformer.set_aux_hidden_capture_layer_ids((1, 1))
        with self.assertRaises(ValueError):
            transformer.set_aux_hidden_capture_layer_ids((4,))


if __name__ == "__main__":
    unittest.main()
