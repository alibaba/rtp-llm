import base64
import unittest

import torch

from rtp_llm.omni.engine.output_processor import OmniOutputProcessor
from rtp_llm.omni.engine.stage_connector import StageOutput


class TestOmniOutputProcessor(unittest.TestCase):
    def test_assemble_text_only(self):
        processor = OmniOutputProcessor()
        stage_outputs = {
            0: StageOutput(token_ids=[1, 2, 3], metadata={"text": "Hello world"}),
        }
        result = processor.assemble(stage_outputs, final_output_types={"text": 0})
        self.assertEqual(result["text"], "Hello world")
        self.assertNotIn("audio", result)

    def test_assemble_text_and_audio(self):
        processor = OmniOutputProcessor()
        waveform = torch.randn(1, 16000)
        stage_outputs = {
            0: StageOutput(metadata={"text": "Hello"}),
            2: StageOutput(audio_waveform=waveform),
        }
        result = processor.assemble(
            stage_outputs,
            final_output_types={"text": 0, "audio": 2},
        )
        self.assertEqual(result["text"], "Hello")
        self.assertIn("audio", result)
        self.assertIsInstance(result["audio"]["waveform"], torch.Tensor)

    def test_assemble_empty_outputs(self):
        processor = OmniOutputProcessor()
        result = processor.assemble({}, final_output_types={})
        self.assertEqual(result, {})

    def test_encode_audio_wav_base64(self):
        processor = OmniOutputProcessor()
        waveform = torch.zeros(1, 100, dtype=torch.float32)
        encoded = processor.encode_audio_base64(waveform, sample_rate=16000)
        self.assertIsInstance(encoded, str)
        decoded_bytes = base64.b64decode(encoded)
        self.assertGreater(len(decoded_bytes), 0)


if __name__ == "__main__":
    unittest.main()
