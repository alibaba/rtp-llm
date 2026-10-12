import base64
import io
import json
import logging
import os
import shlex
import sys
import unittest
import wave
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

SMOKE_ROOT = Path(__file__).absolute().parent
sys.path.insert(0, str(SMOKE_ROOT.parent))

from smoke.case_runner import CaseRunner
from smoke.task_info import TaskInfo


def audio_url():
    time = np.arange(16000, dtype=np.float32) / 16000
    signal = (
        0.2 * np.sin(2 * np.pi * 220 * time) + 0.05 * np.sin(2 * np.pi * 430 * time)
    ).astype(np.float32)
    pcm = np.round(signal * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()


def task_info(model_type, checkpoint_env):
    path = SMOKE_ROOT / "data/model/gemma" / f"{model_type}.json"
    payload = json.loads(path.read_text())
    payload["model_path"] = os.environ.get(checkpoint_env, payload["model_path"])

    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if value == "$GEMMA_SMOKE_AUDIO":
            return audio_url()
        if isinstance(value, str) and value.startswith("data/model/"):
            return str(SMOKE_ROOT / value)
        return value

    payload["query_result"] = resolve(payload["query_result"])
    return TaskInfo(**payload, taskinfo_rel_path=str(path))


class GemmaMultimodalSmoke(unittest.TestCase):
    def test_all_modalities(self):
        assistant = shlex.quote(
            os.environ.get("GEMMA4_ASSISTANT_PATH", "/mnt/nas1/hf/gemma4-assistant")
        )
        configs = [
            (
                "gemma4",
                "GEMMA4_MODEL_PATH",
                "--warm_up 0 --act_type BF16 --tp_size 1 --world_size 1 "
                "--seq_size_per_block 128 --reserver_runtime_mem_mb 16384 "
                f"--sp_type mtp --sp_checkpoint_path {assistant} "
                "--gen_num_per_cycle 6 --enable_cuda_graph 1 "
                "--decode_capture_config 1",
            ),
            (
                "embedding_gemma2",
                "EMBEDDING_GEMMA2_MODEL_PATH",
                "--act_type BF16 --enable_cuda_graph 0 --embedding_model 1 "
                "--max_seq_len 8192 --tp_size 1 --world_size 1 --warm_up 0 "
                "--model_warm_up 0 --reserver_runtime_mem_mb 4096",
            ),
        ]
        for model, checkpoint_env, args in configs:
            with self.subTest(model=model):
                managers = []
                runner = CaseRunner(
                    task_info(model, checkpoint_env),
                    env_args=[
                        "PYTHONNOUSERSITE=1",
                        "RTP_LLM_STREAM_ASYNC=1",
                        "SHUTDOWN_TIMEOUT=30",
                        "RTP_LLM_BACKEND_POST_FRONTEND_DRAIN_SECONDS=0",
                        "WORLD_SIZE=1",
                    ],
                    gpu_card="SM100_ARM_CU13",
                    smoke_args=args,
                )
                runner.start_server = partial(
                    runner.start_server, server_manager_callback=managers.append
                )
                try:
                    states = runner.run()
                    self.assertTrue(states.ret, str(states))
                    self.assertEqual(states.total_count, 1)
                    logging.info("%s: %s", model, states)
                    if model == "embedding_gemma2":
                        tracer = states.query_status[0][2]
                        self.assertEqual(
                            tracer.actual_result.usage, tracer.expect_result.usage
                        )
                        expected = torch.as_tensor(
                            tracer.expect_result.data[0].embedding
                        )
                        actual = torch.as_tensor(tracer.actual_result.data[0].embedding)
                        self.assertGreater(
                            F.cosine_similarity(actual, expected, dim=0).item(), 0.999
                        )
                finally:
                    for manager in managers:
                        manager.stop_server()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
