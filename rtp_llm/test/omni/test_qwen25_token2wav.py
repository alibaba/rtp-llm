import json
import os
import tempfile
import unittest

from rtp_llm.omni.models.qwen2_5_omni.token2wav import Qwen2_5OmniToken2Wav


class TestQwen25OmniToken2Wav(unittest.TestCase):
    def _make_config_dir(self):
        tmpdir = tempfile.mkdtemp()
        config = {
            "architectures": ["Qwen2_5OmniModel"],
            "model_type": "qwen2_5_omni",
            "token2wav_config": {
                "model_type": "qwen2_5_omni_token2wav",
                "dit_config": {
                    "model_type": "qwen2_5_omni_dit",
                    "depth": 22,
                    "dim": 1024,
                    "heads": 16,
                    "head_dim": 64,
                    "mel_dim": 80,
                    "num_embeds": 8193,
                    "ff_mult": 2,
                },
                "bigvgan_config": {
                    "model_type": "qwen2_5_omni_bigvgan",
                    "input_mel_dim": 80,
                    "upsample_rates": [5, 3, 2, 2, 2, 2],
                    "upsample_initial_channel": 1536,
                    "resblock_kernel_sizes": [3, 7, 11],
                },
            },
        }
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        return tmpdir

    def test_create_config_reads_token2wav_config(self):
        tmpdir = self._make_config_dir()
        config = Qwen2_5OmniToken2Wav._create_config(tmpdir)
        self.assertIsNotNone(config)
        self.assertEqual(config.ckpt_path, tmpdir)

    def test_create_config_stores_dit_params(self):
        tmpdir = self._make_config_dir()
        config = Qwen2_5OmniToken2Wav._create_config(tmpdir)
        self.assertEqual(config.dit_depth, 22)
        self.assertEqual(config.dit_dim, 1024)
        self.assertEqual(config.dit_heads, 16)
        self.assertEqual(config.mel_dim, 80)

    def test_create_config_stores_bigvgan_params(self):
        tmpdir = self._make_config_dir()
        config = Qwen2_5OmniToken2Wav._create_config(tmpdir)
        self.assertEqual(config.upsample_rates, [5, 3, 2, 2, 2, 2])
        self.assertEqual(config.upsample_initial_channel, 1536)


if __name__ == "__main__":
    unittest.main()
