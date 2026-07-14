import unittest
from types import SimpleNamespace

from rtp_llm.tools.convert.weights_convert_utils import (
    apply_layer_override_and_post_build,
)


class FakeModelClass:
    @staticmethod
    def _post_build_model_config(model_config):
        model_config.kv_cache_spec_descs = [
            [SimpleNamespace(tag=f"full-{layer_id}")]
            for layer_id in range(model_config.num_layers)
        ]


class WeightsConvertLayerOverrideTest(unittest.TestCase):
    def test_layer_override_is_applied_before_cache_descriptors_are_built(self):
        model_config = SimpleNamespace(
            num_layers=4,
            kv_cache_spec_descs=[[SimpleNamespace(tag="stale")]],
        )

        result = apply_layer_override_and_post_build(
            model_config, FakeModelClass, {"HACK_LAYER_NUM": "2"}
        )

        self.assertIs(result, model_config)
        self.assertEqual(result.num_layers, 2)
        self.assertEqual(
            [[desc.tag for desc in layer] for layer in result.kv_cache_spec_descs],
            [["full-0"], ["full-1"]],
        )


if __name__ == "__main__":
    unittest.main()
