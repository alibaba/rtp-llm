import json
import os
import tempfile
import unittest

import torch

from rtp_llm.model_factory_register import _model_factory


class TestOmniDiTConfig(unittest.TestCase):
    def test_dit_config_from_dict(self):
        from rtp_llm.omni.models.qwen2_5_omni.token2wav_dit import OmniDiTConfig

        config = OmniDiTConfig.from_dict(
            {
                "depth": 22,
                "dim": 1024,
                "heads": 16,
                "head_dim": 64,
                "ff_mult": 2,
                "mel_dim": 80,
                "num_embeds": 8193,
                "emb_dim": 512,
                "dropout": 0.1,
            }
        )
        self.assertEqual(config.depth, 22)
        self.assertEqual(config.dim, 1024)
        self.assertEqual(config.heads, 16)


class TestOmniDiTForward(unittest.TestCase):
    def test_dit_forward_shape(self):
        from rtp_llm.omni.models.qwen2_5_omni.token2wav_dit import (
            OmniDiT,
            OmniDiTConfig,
        )

        config = OmniDiTConfig(
            depth=2,
            dim=64,
            heads=4,
            head_dim=16,
            ff_mult=2,
            mel_dim=80,
            num_embeds=100,
            emb_dim=32,
            dropout=0.0,
        )
        model = OmniDiT(config)
        x = torch.randn(1, 10, 80)
        t = torch.tensor([0.5])
        codec_ids = torch.randint(0, 100, (1, 10))
        out = model(x, t, codec_ids)
        self.assertEqual(out.shape, (1, 10, 80))


class TestBigVGANConfig(unittest.TestCase):
    def test_bigvgan_config_from_dict(self):
        from rtp_llm.omni.models.qwen2_5_omni.token2wav_bigvgan import BigVGANConfig

        config = BigVGANConfig.from_dict(
            {
                "mel_dim": 80,
                "upsample_rates": [5, 3, 2, 2, 2, 2],
                "upsample_initial_channel": 1536,
            }
        )
        self.assertEqual(config.mel_dim, 80)
        self.assertEqual(config.upsample_rates, [5, 3, 2, 2, 2, 2])


class TestBigVGANForward(unittest.TestCase):
    def test_bigvgan_forward_shape(self):
        from rtp_llm.omni.models.qwen2_5_omni.token2wav_bigvgan import (
            BigVGAN,
            BigVGANConfig,
        )

        config = BigVGANConfig(
            mel_dim=80,
            upsample_rates=[2, 2],
            upsample_initial_channel=64,
            upsample_kernel_sizes=[4, 4],
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 2]],
        )
        model = BigVGAN(config)
        mel = torch.randn(1, 80, 10)
        wav = model(mel)
        self.assertEqual(wav.shape[0], 1)
        self.assertEqual(wav.shape[1], 1)
        self.assertEqual(wav.shape[2], 40)


class TestQwen25OmniToken2WavRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rtp_llm.omni.models.qwen2_5_omni.token2wav  # noqa: F401

    def test_model_registered(self):
        self.assertIn("qwen2_5_omni_token2wav", _model_factory)


class TestQwen25OmniToken2WavConfig(unittest.TestCase):
    def test_create_config_from_omni_checkpoint(self):
        from rtp_llm.omni.models.qwen2_5_omni.token2wav import Qwen25OmniToken2Wav

        config_json = {
            "token2wav_config": {
                "dit_config": {
                    "depth": 22,
                    "dim": 1024,
                    "heads": 16,
                    "head_dim": 64,
                    "ff_mult": 2,
                    "mel_dim": 80,
                    "num_embeds": 8193,
                    "emb_dim": 512,
                },
                "bigvgan_config": {
                    "mel_dim": 80,
                    "upsample_rates": [5, 3, 2, 2, 2, 2],
                    "upsample_initial_channel": 1536,
                    "upsample_kernel_sizes": [11, 7, 4, 4, 4, 4],
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_json, f)
            config = Qwen25OmniToken2Wav._create_config(tmpdir)
            self.assertIsNotNone(config)
            self.assertEqual(config.hidden_size, 1024)
            self.assertEqual(config.num_layers, 22)


if __name__ == "__main__":
    unittest.main()
