import os
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.standalone.auto_model import AutoModel
from rtp_llm.ops.compute_ops import PyModelInputs

# MODEL_INPUTS_DUMP_PATH=/tmp/model-inputs/model_inputs MODEL_INPUTS_REPLAY_MODEL_PATH=/path/to/Qwen2-0.5B bazelisk test --config=cuda12_9 //rtp_llm/cpp/models/test:model_inputs_replay_test --test_env=MODEL_INPUTS_DUMP_PATH --test_env=MODEL_INPUTS_REPLAY_MODEL_PATH --test_output=all
DUMP_PATH = os.environ.get("MODEL_INPUTS_DUMP_PATH")
MODEL_PATH = os.environ.get("MODEL_INPUTS_REPLAY_MODEL_PATH")


class ReplayAutoModel(AutoModel):
    def _set_configs(self) -> None:
        super()._set_configs()
        self.py_env_configs.load_config.load_method = "scratch"
        self.py_env_configs.fmha_config.enable_xqa = False


@unittest.skipUnless(DUMP_PATH and MODEL_PATH, "set replay dump and model paths")
class ModelInputsReplayTest(unittest.TestCase):
    def test_forward_replay(self) -> None:
        paths = sorted(Path(DUMP_PATH).glob("*.pt"))
        chunks = [
            torch.load(path, map_location="cpu", weights_only=False) for path in paths
        ]
        records = [
            record
            for chunk in chunks
            for record in (
                chunk["records"]
                if chunk["record_type"] == "model_inputs_chunk"
                else [chunk]
            )
        ]
        gaps = [r for r in records if r["record_type"] == "model_inputs_gap"]
        self.assertFalse(gaps, f"dump contains dropped records: {gaps}")
        snapshots = [
            r for r in records if r["record_type"] == "gpt_model_inputs_snapshot"
        ]
        self.assertGreater(len(snapshots), 1, f"no replay chain under {DUMP_PATH}")
        snapshots.sort(key=lambda snapshot: snapshot["record_sequence"])
        seqs = [snapshot["record_sequence"] for snapshot in snapshots]
        self.assertEqual(seqs, list(range(seqs[0], seqs[0] + len(seqs))))
        stages = [snapshot["execution_stage"] for snapshot in snapshots]
        self.assertEqual(stages, ["prefill"] + ["decode"] * (len(snapshots) - 1))
        self.assertEqual({snapshot["model_role"] for snapshot in snapshots}, {"normal"})
        self.assertTrue(all(snapshot["dropped_before"] == 0 for snapshot in snapshots))

        prompt_length = int(snapshots[0]["input_lengths"].sum().item())
        model = ReplayAutoModel.from_pretrained(
            MODEL_PATH,
            max_total_tokens=prompt_length + len(snapshots) + 1,
            tokens_per_block=int(snapshots[0]["seq_size_per_block"]),
        )
        attention_inputs = None
        for index, snapshot in enumerate(snapshots):
            if snapshot["execution_stage"] == "prefill":
                attention_inputs = model._prepare_prefill_attention_inputs(
                    prompt_length
                )
            else:
                sequence_length = int(snapshot["sequence_lengths"].item()) + 1
                attention_inputs = model._prepare_decode_attention_inputs(
                    attention_inputs, sequence_length
                )
            replay_inputs = PyModelInputs(
                input_ids=snapshot["combo_tokens"].to(model.device),
                attention_inputs=attention_inputs,
            )
            outputs = model.model.forward(replay_inputs)
            self.assertTrue(torch.isfinite(outputs.hidden_states).all().item())
            if index + 1 < len(snapshots):
                # Sampler configuration and RNG state are outside ModelInputs.
                next_token = int(snapshots[index + 1]["combo_tokens"][0].item())
                self.assertTrue(0 <= next_token < model.model_config.vocab_size)


if __name__ == "__main__":
    unittest.main()
