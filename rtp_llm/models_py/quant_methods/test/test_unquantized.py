import unittest

import torch

from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod


class TestUnquantizedLinearMethod(unittest.TestCase):
    def test_post_load_preserves_values_and_uses_transpose_view(self):
        layer = torch.nn.Module()
        method = UnquantizedLinearMethod()
        method.create_weights(
            layer,
            input_size=5,
            output_size=3,
            params_dtype=torch.float32,
        )
        source = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        with torch.no_grad():
            layer.weight.copy_(source)

        original_parameter = layer.weight
        method.process_weights_after_loading(layer)

        self.assertIs(layer.weight, original_parameter)
        self.assertEqual(layer.weight.shape, (3, 5))
        self.assertEqual(layer.weight.stride(), (1, 3))
        torch.testing.assert_close(layer.weight, source, rtol=0, atol=0)

        inputs = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        torch.testing.assert_close(
            method.apply(layer, inputs),
            torch.nn.functional.linear(inputs, source),
            rtol=0,
            atol=0,
        )

    def test_post_load_is_idempotent(self):
        layer = torch.nn.Module()
        method = UnquantizedLinearMethod()
        method.create_weights(
            layer,
            input_size=5,
            output_size=3,
            params_dtype=torch.float32,
        )
        method.process_weights_after_loading(layer)
        data_ptr = layer.weight.data_ptr()

        method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.stride(), (1, 3))
        self.assertEqual(layer.weight.data_ptr(), data_ptr)


if __name__ == "__main__":
    unittest.main()
