"""Reference transcription with transformers on real whisper-large-v3.

Produces golden transcripts for the smoke case and the WER/CER baseline.
Run with --test_env WHISPER_CKPT=<ckpt> --test_env WHISPER_AUDIO_DIR=<dir>.
Outputs one "<name>.transcript.txt" per .flac/.wav in the audio dir, plus a
combined JSON at <audio_dir>/reference_transcripts.json.
"""

import glob
import json
import os
import sys
import unittest


class ReferenceTranscribeTest(unittest.TestCase):
    def test_transcribe(self):
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration
        import librosa

        ckpt = os.environ.get("WHISPER_CKPT")
        audio_dir = os.environ.get("WHISPER_AUDIO_DIR")
        self.assertTrue(ckpt, "WHISPER_CKPT required")
        self.assertTrue(audio_dir, "WHISPER_AUDIO_DIR required")
        files = sorted(
            f for f in glob.glob(os.path.join(audio_dir, "*"))
            if f.endswith((".flac", ".wav", ".mp3"))
        )
        self.assertTrue(files, f"no audio files in {audio_dir}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperForConditionalGeneration.from_pretrained(
            ckpt, torch_dtype=torch.float16
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(ckpt)
        results = {}
        for path in files:
            audio, _ = librosa.load(path, sr=16000)
            inputs = processor(
                audio, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs.input_features.to(device=device, dtype=torch.float16)
            with torch.inference_mode():
                predicted_ids = model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                    return_timestamps=False,
                )
            text = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            name = os.path.splitext(os.path.basename(path))[0]
            results[name] = text.strip()
            with open(os.path.join(audio_dir, f"{name}.transcript.txt"), "w") as f:
                f.write(text.strip())
            print(f"REFERENCE {name}: {text.strip()}", flush=True)
        # audio-tower (encoder) latency: steady-state over the first sample
        import time

        audio0, _ = librosa.load(files[0], sr=16000)
        inputs0 = processor(audio0, sampling_rate=16000, return_tensors="pt")
        feats = inputs0.input_features.to(device=device, dtype=torch.float16)
        for _ in range(3):
            with torch.inference_mode():
                model.model.encoder(feats)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            with torch.inference_mode():
                model.model.encoder(feats)
        torch.cuda.synchronize()
        per = (time.time() - t0) / 10
        print(f"ENCODER_TOWER_LATENCY: {per*1000:.2f} ms/forward", flush=True)
        with open(os.path.join(audio_dir, "reference_transcripts.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"reference transcripts written: {len(results)}", flush=True)


if __name__ == "__main__":
    unittest.main()
