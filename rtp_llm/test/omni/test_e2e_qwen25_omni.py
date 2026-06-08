import unittest

import torch

from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.engine.omni_engine import OmniEngine
from rtp_llm.omni.engine.output_processor import OmniOutputProcessor
from rtp_llm.omni.engine.stage_connector import SharedMemoryConnector, StageOutput
from rtp_llm.omni.models.qwen2_5_omni.pipeline import QWEN2_5_OMNI_PIPELINE
from rtp_llm.omni.models.qwen2_5_omni.stage_processors import (
    thinker2talker,
    talker2code2wav,
)


class TestE2EQwen25Omni(unittest.TestCase):
    """End-to-end test simulating a full Qwen2.5-Omni pipeline run.

    Exercises: engine creation -> request submission -> stage-by-stage
    data flow through connector and processors -> output assembly.
    """

    def setUp(self):
        if OmniPipelineRegistry.get("qwen2_5_omni") is None:
            OmniPipelineRegistry.register(QWEN2_5_OMNI_PIPELINE)

        self.connector = SharedMemoryConnector()
        self.engine = OmniEngine(
            pipeline_config=QWEN2_5_OMNI_PIPELINE,
            connector=self.connector,
        )

    def test_engine_has_three_stages(self):
        self.assertEqual(self.engine.num_stages, 3)

    def test_engine_final_output_types(self):
        final = self.engine.get_final_output_types()
        self.assertEqual(final, {"text": "thinker", "audio": "token2wav"})

    def test_full_pipeline_flow(self):
        request_id = "e2e-test-001"

        # --- Submit request ---
        state = self.engine.orchestrator.submit(request_id)
        self.assertFalse(state.is_complete)

        execution_order = self.engine.orchestrator.get_execution_order()
        self.assertEqual(execution_order, ["thinker", "talker", "token2wav"])

        # --- Stage "thinker": produces text tokens + hidden embeddings ---
        thinker_output = StageOutput(
            token_ids=[1001, 1002, 1003],
            embeddings=torch.randn(3, 3584),
            metadata={"text": "Hello, how can I help you?"},
        )
        self.connector.put(request_id, "thinker", thinker_output)
        state.mark_complete("thinker")
        self.assertFalse(state.is_complete)

        # --- thinker -> talker: processor transforms data ---
        thinker_raw = self.connector.get(request_id, "thinker")
        self.assertIsNotNone(thinker_raw)

        talker_input = thinker2talker(thinker_raw)

        self.assertIsNotNone(talker_input.embeddings)
        self.assertEqual(talker_input.embeddings.shape, (3, 3584))
        self.assertEqual(talker_input.metadata["source_token_ids"], [1001, 1002, 1003])
        self.assertEqual(
            talker_input.metadata["source_text"], "Hello, how can I help you?"
        )

        # --- Stage "talker": produces codec tokens ---
        talker_output = StageOutput(
            token_ids=[100, 200, 300, 400, 500, 8294],
            metadata={"codec_format": "tts_v1"},
        )
        self.connector.put(request_id, "talker", talker_output)
        state.mark_complete("talker")
        self.assertFalse(state.is_complete)

        # --- talker -> token2wav: processor filters codec end token ---
        talker_raw = self.connector.get(request_id, "talker")
        self.assertIsNotNone(talker_raw)

        token2wav_input = talker2code2wav(talker_raw)

        self.assertEqual(token2wav_input.token_ids, [100, 200, 300, 400, 500])
        self.assertNotIn(8294, token2wav_input.token_ids)
        self.assertEqual(token2wav_input.metadata["codec_token_count"], 5)

        # --- Stage "token2wav": produces audio waveform ---
        sample_rate = 24000
        duration_s = 0.5
        num_samples = int(sample_rate * duration_s)
        waveform = torch.randn(1, num_samples)

        token2wav_output = StageOutput(
            audio_waveform=waveform,
            metadata={"sample_rate": sample_rate, "duration_s": duration_s},
        )
        self.connector.put(request_id, "token2wav", token2wav_output)
        state.mark_complete("token2wav")
        self.assertTrue(state.is_complete)

        # --- Assemble final output ---
        stage_outputs = {
            name: self.connector.get(request_id, name)
            for name in execution_order
        }
        final_output_types = self.engine.get_final_output_types()

        processor = OmniOutputProcessor()
        result = processor.assemble(stage_outputs, final_output_types)

        self.assertIn("text", result)
        self.assertEqual(result["text"], "Hello, how can I help you?")
        self.assertIn("audio", result)
        self.assertTrue(
            torch.equal(result["audio"]["waveform"], waveform)
        )
        self.assertEqual(result["audio"]["metadata"]["sample_rate"], sample_rate)

        # --- Cleanup ---
        self.engine.orchestrator.cleanup(request_id)
        self.assertIsNone(self.engine.orchestrator.get_request_state(request_id))
        self.assertIsNone(self.connector.get(request_id, "thinker"))

    def test_duplicate_request_rejected(self):
        self.engine.orchestrator.submit("dup-001")
        with self.assertRaises(ValueError):
            self.engine.orchestrator.submit("dup-001")

    def test_processor_functions_work(self):
        source = StageOutput(
            token_ids=[1, 2, 3],
            embeddings=torch.randn(3, 128),
            metadata={"text": "test"},
        )
        result = thinker2talker(source)
        self.assertIsNotNone(result.embeddings)
        self.assertIn("source_token_ids", result.metadata)

        source2 = StageOutput(token_ids=[10, 20, 8294])
        result2 = talker2code2wav(source2)
        self.assertEqual(result2.token_ids, [10, 20])

    def test_audio_encoding_roundtrip(self):
        waveform = torch.randn(1, 16000)
        encoded = OmniOutputProcessor.encode_audio_base64(
            waveform, sample_rate=16000, audio_format="wav"
        )
        self.assertIsInstance(encoded, str)
        self.assertTrue(len(encoded) > 0)

    def test_multiple_concurrent_requests(self):
        state_a = self.engine.orchestrator.submit("req-a")
        state_b = self.engine.orchestrator.submit("req-b")

        out_a = StageOutput(token_ids=[10], metadata={"text": "A"})
        out_b = StageOutput(token_ids=[20], metadata={"text": "B"})

        self.connector.put("req-a", "thinker", out_a)
        self.connector.put("req-b", "thinker", out_b)

        got_a = self.connector.get("req-a", "thinker")
        got_b = self.connector.get("req-b", "thinker")
        self.assertEqual(got_a.metadata["text"], "A")
        self.assertEqual(got_b.metadata["text"], "B")

        state_a.mark_complete("thinker")
        self.assertFalse(state_b.is_complete)
        self.assertTrue(state_a.is_stage_complete("thinker"))
        self.assertFalse(state_b.is_stage_complete("thinker"))

        self.engine.orchestrator.cleanup("req-a")
        self.assertIsNone(self.connector.get("req-a", "thinker"))
        self.assertIsNotNone(self.connector.get("req-b", "thinker"))


if __name__ == "__main__":
    unittest.main()
