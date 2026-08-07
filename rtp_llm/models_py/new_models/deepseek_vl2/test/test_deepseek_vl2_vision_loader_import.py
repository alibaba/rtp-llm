import unittest


class DeepSeekVLV2VisionLoaderImportTest(unittest.TestCase):
    def test_vision_loader_import_is_independent(self):
        from rtp_llm.models_py.new_models.deepseek_vl2.vision import (
            DeepSeekVLV2ForVisionEmbedding,
            DeepSeekVLV2VisionModel,
            load_deepseek_vl2_vision,
        )

        self.assertTrue(
            issubclass(DeepSeekVLV2ForVisionEmbedding, DeepSeekVLV2VisionModel)
        )
        self.assertTrue(callable(load_deepseek_vl2_vision))


if __name__ == "__main__":
    unittest.main()
