import pathlib
import unittest


class WeightsConvertOrderTest(unittest.TestCase):
    def test_hack_layer_num_is_applied_before_post_build(self):
        source = pathlib.Path(__file__).with_name("weights_convert.py").read_text()
        hack_pos = source.index("model_config.num_layers = int(")
        post_pos = source.index("self.model_cls._post_build_model_config(model_config)", hack_pos)
        self.assertLess(hack_pos, post_pos)


if __name__ == "__main__":
    unittest.main()
