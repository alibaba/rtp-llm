"""End-to-end test for Qwen2.5-Omni pipeline — no mocks.

Uses real model classes, real config parsing from a HuggingFace-format
config.json, real ModelFactory pipeline detection, real OmniEngine,
real stage processors, and real output assembly. The only thing not
exercised is the GPU forward pass (no weights loaded).

Run on mateng04:
    cd /root/mateng/rtp-llm
    python -m pytest rtp_llm/test/omni/test_e2e_qwen25_omni_no_mock.py -v
"""

import json
import os
import shutil
import tempfile
import unittest

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import _model_factory
from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.engine.omni_engine import OmniEngine
from rtp_llm.omni.engine.output_processor import OmniOutputProcessor
from rtp_llm.omni.engine.stage_connector import SharedMemoryConnector, StageOutput
from rtp_llm.omni.models.qwen2_5_omni.pipeline import QWEN2_5_OMNI_PIPELINE
from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker
from rtp_llm.omni.models.qwen2_5_omni.talker import Qwen2_5OmniTalker
from rtp_llm.omni.models.qwen2_5_omni.token2wav import Qwen2_5OmniToken2Wav
from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
    thinker2talker,
    talker2code2wav,
)

# Full HuggingFace Qwen2.5-Omni config.json matching the real model structure.
QWEN25_OMNI_HF_CONFIG = {
    "architectures": ["Qwen2_5OmniModel"],
    "model_type": "qwen2_5_omni",
    "thinker_config": {
        "model_type": "qwen2_5_omni_thinker",
        "text_config": {
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000,
        },
        "audio_config": {
            "model_type": "qwen2_5_omni_audio_encoder",
            "d_model": 1280,
            "encoder_attention_heads": 20,
            "encoder_ffn_dim": 5120,
            "encoder_layers": 32,
            "num_mel_bins": 128,
            "max_source_positions": 1500,
        },
    },
    "talker_config": {
        "model_type": "qwen2_5_omni_talker",
        "hidden_size": 896,
        "intermediate_size": 18944,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "num_hidden_layers": 24,
        "vocab_size": 8448,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000,
        "thinker_hidden_size": 3584,
    },
    "token2wav_config": {
        "model_type": "qwen2_5_omni_token2wav",
        "dit_config": {
            "depth": 22,
            "dim": 1024,
            "heads": 16,
            "head_dim": 64,
            "mel_dim": 80,
            "num_embeds": 8193,
        },
        "bigvgan_config": {
            "upsample_rates": [5, 3, 2, 2, 2, 2],
            "upsample_initial_channel": 1536,
            "resblock_kernel_sizes": [3, 7, 11],
        },
    },
}


class TestE2EQwen25OmniNoMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="rtp_omni_test_")
        config_path = os.path.join(cls.tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(QWEN25_OMNI_HF_CONFIG, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    # ── 1. Model registration ──

    def test_model_classes_registered(self):
        self.assertIn("qwen2_5_omni", _model_factory)
        self.assertIn("qwen2_5_omni_thinker", _model_factory)
        self.assertIn("qwen2_5_omni_talker", _model_factory)
        self.assertIn("qwen2_5_omni_token2wav", _model_factory)

    def test_pipeline_registered(self):
        config = OmniPipelineRegistry.get("qwen2_5_omni")
        self.assertIsNotNone(config)
        self.assertEqual(config.model_arch, "Qwen2_5OmniModel")
        self.assertEqual(len(config.stages), 3)

    def test_stage_processor_functions_importable(self):
        self.assertTrue(callable(thinker2talker))
        self.assertTrue(callable(talker2code2wav))

    # ── 2. Real config parsing from HF config.json ──

    def test_thinker_config_from_real_hf_json(self):
        config = Qwen2_5OmniThinker._create_config(self.tmpdir)
        self.assertIsInstance(config, ModelConfig)
        self.assertEqual(config.hidden_size, 3584)
        self.assertEqual(config.num_layers, 28)
        self.assertEqual(config.attn_config.head_num, 28)
        self.assertEqual(config.attn_config.kv_head_num, 4)
        self.assertEqual(config.inter_size, 18944)
        self.assertEqual(config.vocab_size, 152064)
        self.assertEqual(config.attn_config.rope_config.base, 1000000)
        self.assertEqual(config.attn_config.size_per_head, 3584 // 28)

    def test_talker_config_from_real_hf_json(self):
        config = Qwen2_5OmniTalker._create_config(self.tmpdir)
        self.assertIsInstance(config, ModelConfig)
        self.assertEqual(config.hidden_size, 896)
        self.assertEqual(config.num_layers, 24)
        self.assertEqual(config.attn_config.head_num, 14)
        self.assertEqual(config.attn_config.kv_head_num, 2)
        self.assertEqual(config.inter_size, 18944)
        self.assertEqual(config.vocab_size, 8448)
        self.assertEqual(config.embedding_size, 3584)
        self.assertEqual(config.attn_config.size_per_head, 896 // 14)

    def test_token2wav_config_from_real_hf_json(self):
        config = Qwen2_5OmniToken2Wav._create_config(self.tmpdir)
        self.assertEqual(config.dit_depth, 22)
        self.assertEqual(config.dit_dim, 1024)
        self.assertEqual(config.dit_heads, 16)
        self.assertEqual(config.dit_head_dim, 64)
        self.assertEqual(config.mel_dim, 80)
        self.assertEqual(config.dit_num_embeds, 8193)
        self.assertEqual(config.upsample_rates, [5, 3, 2, 2, 2, 2])
        self.assertEqual(config.upsample_initial_channel, 1536)
        self.assertEqual(config.resblock_kernel_sizes, [3, 7, 11])

    # ── 3. OmniEngine creation from pipeline config ──

    def test_omni_engine_from_pipeline_config(self):
        model_config = ModelConfig()
        model_config.model_type = "qwen2_5_omni"
        model_config.ckpt_path = self.tmpdir

        engine = OmniEngine.from_pipeline_config(
            QWEN2_5_OMNI_PIPELINE, model_config=model_config
        )
        self.assertIsInstance(engine, OmniEngine)
        self.assertEqual(engine.num_stages, 3)
        self.assertIs(engine.model_config, model_config)
        self.assertEqual(
            engine.get_final_output_types(), {"text": "thinker", "audio": "token2wav"}
        )

    # ── 4. Full pipeline data flow with real processors ──

    def test_full_pipeline_data_flow(self):
        connector = SharedMemoryConnector()
        engine = OmniEngine(
            pipeline_config=QWEN2_5_OMNI_PIPELINE,
            connector=connector,
        )

        request_id = "e2e-no-mock-001"

        # Submit
        state = engine.orchestrator.submit(request_id)
        self.assertEqual(
            engine.orchestrator.get_execution_order(),
            ["thinker", "talker", "token2wav"],
        )

        # ── Stage "thinker": output ──
        thinker_embeddings = torch.randn(5, 3584)
        thinker_output = StageOutput(
            token_ids=[151644, 1001, 1002, 1003, 151645],
            embeddings=thinker_embeddings,
            metadata={"text": "The weather is nice today."},
        )
        connector.put(request_id, "thinker", thinker_output)
        state.mark_complete("thinker")

        # ── thinker -> talker processor ──
        talker_input = thinker2talker(connector.get(request_id, "thinker"))

        self.assertTrue(torch.equal(talker_input.embeddings, thinker_embeddings))
        self.assertEqual(
            talker_input.metadata["source_token_ids"],
            [151644, 1001, 1002, 1003, 151645],
        )
        self.assertEqual(
            talker_input.metadata["source_text"], "The weather is nice today."
        )

        # ── Stage "talker": output (codec tokens) ──
        codec_tokens = [101, 202, 303, 404, 505, 606, 707, 808, 8294]
        talker_output = StageOutput(
            token_ids=codec_tokens,
            metadata={"codec_format": "tts_v1"},
        )
        connector.put(request_id, "talker", talker_output)
        state.mark_complete("talker")

        # ── talker -> token2wav processor ──
        t2w_input = talker2code2wav(connector.get(request_id, "talker"))

        self.assertEqual(
            t2w_input.token_ids, [101, 202, 303, 404, 505, 606, 707, 808]
        )
        self.assertEqual(t2w_input.metadata["codec_token_count"], 8)

        # ── Stage "token2wav": output (audio waveform) ──
        sample_rate = 24000
        duration_s = 1.0
        num_samples = int(sample_rate * duration_s)
        waveform = torch.randn(1, num_samples)

        token2wav_output = StageOutput(
            audio_waveform=waveform,
            metadata={"sample_rate": sample_rate, "duration_s": duration_s},
        )
        connector.put(request_id, "token2wav", token2wav_output)
        state.mark_complete("token2wav")
        self.assertTrue(state.is_complete)

        # ── Output assembly ──
        final_output_types = engine.get_final_output_types()
        self.assertEqual(final_output_types, {"text": "thinker", "audio": "token2wav"})

        stage_outputs = {
            name: connector.get(request_id, name)
            for name in engine.orchestrator.get_execution_order()
        }

        output_processor = OmniOutputProcessor()
        result = output_processor.assemble(stage_outputs, final_output_types)

        self.assertEqual(result["text"], "The weather is nice today.")
        self.assertTrue(torch.equal(result["audio"]["waveform"], waveform))
        self.assertEqual(result["audio"]["metadata"]["sample_rate"], 24000)
        self.assertEqual(result["audio"]["metadata"]["duration_s"], 1.0)

        # ── Audio encoding ──
        audio_b64 = OmniOutputProcessor.encode_audio_base64(
            waveform, sample_rate=sample_rate
        )
        self.assertIsInstance(audio_b64, str)
        self.assertGreater(len(audio_b64), 100)

        # ── Cleanup ──
        engine.orchestrator.cleanup(request_id)
        self.assertIsNone(engine.orchestrator.get_request_state(request_id))
        for name in ["thinker", "talker", "token2wav"]:
            self.assertIsNone(connector.get(request_id, name))

    # ── 5. Cross-stage config consistency ──

    def test_thinker_talker_embedding_size_match(self):
        thinker_cfg = Qwen2_5OmniThinker._create_config(self.tmpdir)
        talker_cfg = Qwen2_5OmniTalker._create_config(self.tmpdir)
        self.assertEqual(
            thinker_cfg.hidden_size,
            talker_cfg.embedding_size,
            "Thinker hidden_size must equal talker embedding_size for cross-attention",
        )

    def test_talker_and_token2wav_configs_are_independent(self):
        talker_cfg = Qwen2_5OmniTalker._create_config(self.tmpdir)
        t2w_cfg = Qwen2_5OmniToken2Wav._create_config(self.tmpdir)
        self.assertEqual(talker_cfg.vocab_size, 8448)
        self.assertEqual(t2w_cfg.dit_num_embeds, 8193)

    # ── 6. Weight class identity ──

    def test_thinker_weight_cls(self):
        from rtp_llm.omni.models.qwen2_5_omni.thinker import (
            Qwen2_5OmniThinkerWeight,
        )
        weight_cls = Qwen2_5OmniThinker.get_weight_cls()
        self.assertIs(weight_cls, Qwen2_5OmniThinkerWeight)

    def test_talker_weight_cls(self):
        from rtp_llm.omni.models.qwen2_5_omni.talker import (
            Qwen2_5OmniTalkerWeight,
        )
        weight_cls = Qwen2_5OmniTalker.get_weight_cls()
        self.assertIs(weight_cls, Qwen2_5OmniTalkerWeight)

    def test_token2wav_weight_cls_is_none(self):
        self.assertIsNone(Qwen2_5OmniToken2Wav.get_weight_cls())

    # ── 7. Multiple requests isolation ──

    def test_concurrent_requests_isolation(self):
        connector = SharedMemoryConnector()
        engine = OmniEngine(
            pipeline_config=QWEN2_5_OMNI_PIPELINE,
            connector=connector,
        )

        state_a = engine.orchestrator.submit("iso-a")
        state_b = engine.orchestrator.submit("iso-b")

        connector.put("iso-a", "thinker", StageOutput(
            token_ids=[1], embeddings=torch.randn(1, 3584),
            metadata={"text": "Request A"},
        ))
        connector.put("iso-b", "thinker", StageOutput(
            token_ids=[2], embeddings=torch.randn(1, 3584),
            metadata={"text": "Request B"},
        ))

        result_a = thinker2talker(connector.get("iso-a", "thinker"))
        result_b = thinker2talker(connector.get("iso-b", "thinker"))

        self.assertEqual(result_a.metadata["source_text"], "Request A")
        self.assertEqual(result_b.metadata["source_text"], "Request B")
        self.assertEqual(result_a.metadata["source_token_ids"], [1])
        self.assertEqual(result_b.metadata["source_token_ids"], [2])

        state_a.mark_complete("thinker")
        self.assertTrue(state_a.is_stage_complete("thinker"))
        self.assertFalse(state_b.is_stage_complete("thinker"))

        engine.orchestrator.cleanup("iso-a")
        self.assertIsNone(connector.get("iso-a", "thinker"))
        self.assertIsNotNone(connector.get("iso-b", "thinker"))


if __name__ == "__main__":
    unittest.main()
