import unittest

import torch

from rtp_llm.models_py.modules.dsv4.aux_hidden_states import (
    AuxHiddenStatesCapture,
    make_aux_hidden_states_layers_tensor,
    resolve_aux_hidden_states_layers,
)


class AuxHiddenStatesTest(unittest.TestCase):
    def test_default_layers_match_eagle3_layout(self):
        self.assertEqual(resolve_aux_hidden_states_layers(None, 61), [1, 29, 57])

    def test_explicit_layers_from_tensor(self):
        layer_ids = torch.tensor([1, 3, 5], dtype=torch.int32)
        self.assertEqual(resolve_aux_hidden_states_layers(layer_ids, 8), [1, 3, 5])

    def test_invalid_layers_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            resolve_aux_hidden_states_layers(torch.tensor([1, 1, 2]), 8)
        with self.assertRaisesRegex(ValueError, "out of range"):
            resolve_aux_hidden_states_layers(torch.tensor([1, 8, 2]), 8)

    def test_capture_flattens_selected_layers_in_requested_order(self):
        template = torch.zeros((2, 3, 4), dtype=torch.float32)
        capture = AuxHiddenStatesCapture([3, 1], template)

        layer1 = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        layer3 = layer1 + 100
        ignored = layer1 + 200

        capture.maybe_capture(1, layer1)
        capture.maybe_capture(2, ignored)
        capture.maybe_capture(3, layer3)

        expected = torch.cat(
            [layer3.reshape(2, 12), layer1.reshape(2, 12)],
            dim=-1,
        )
        torch.testing.assert_close(capture.tensor, expected)

    def test_layer_tensor_dtype_and_device(self):
        layers = make_aux_hidden_states_layers_tensor([1, 3, 5], torch.device("cpu"))
        self.assertEqual(layers.dtype, torch.int32)
        self.assertEqual(layers.tolist(), [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
