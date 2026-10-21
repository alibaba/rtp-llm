import unittest
from frontend_assets import asset_name


class FrontendAssetsTest(unittest.TestCase):
    def test_metadata_only_and_bounded_paths(self):
        self.assertEqual("config.json", asset_name("model/config.json", "model/"))
        self.assertEqual("tokenizer/tokenizer.model", asset_name("model/tokenizer/tokenizer.model", "model/"))
        for key in ("model/weights.safetensors", "model/pytorch_model.bin", "other/config.json",
                    "model/../config.json", "model//config.json", "model/"):
            self.assertIsNone(asset_name(key, "model/"), key)


if __name__ == "__main__":
    unittest.main()
