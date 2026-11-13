"""Real-model resource smoke with a test Epsilon barrier, not native dump/restore."""

import json
import os
import unittest
import uuid
from pathlib import Path

import requests
from rtp_llm.test.smoke.scr_checkpoint_probe import (
    CheckpointBarrier,
    assert_clean_checkpoint,
    kv_backing_snapshot,
)
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class ScrCheckpointSteadyStateTest(unittest.TestCase):
    def test_checkpoint_resources_and_release(self):
        model_path = os.environ.get(
            "CHECKPOINT_PATH",
            "/mnt/nas1/hf/models--Qwen--Qwen3-1.7B/snapshots/"
            "0060bc56d46589041c1048efd1a397421b1142b5",
        )
        self.assertTrue(Path(model_path, "config.json").is_file(), model_path)
        output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        output.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        barrier = CheckpointBarrier(generation)
        self.addCleanup(barrier.close)
        server = MagaServerManager(
            env_args={
                "SCR_ENABLE": "1",
                "SCR_PHASE": "checkpoint",
                "RTPLLM_ENABLE_SCR": "1",
                "HIPPO_ROLE_SHORT_NAME": "scr_kmonitor_negative",
                "HIPPO_SLAVE_IP": "203.0.113.1",  # Reserved example IP, not a real Flume endpoint.
                "RTP_LLM_SCR_GENERATION": generation,
                "RTP_LLM_SCR_TIMEOUT": "900",
                "SCR_TEST_BARRIER_SOCKET": barrier.path,
            },
            role_name="scr_checkpoint",
            smoke_args_str=os.environ["SMOKE_ARGS"],
        )
        self.addCleanup(server.stop_server)
        try:
            self.assertTrue(
                server.start_server(
                    model_path=model_path,
                    model_type=os.environ.get("MODEL_TYPE", "qwen_3"),
                    wait_ready=False,
                    server_module="rtp_llm.test.smoke.scr_checkpoint_worker",
                )
            )
            arrivals = barrier.wait_ready(server.server_pid, timeout=600)
            (output / "scr_arrivals.json").write_text(
                json.dumps(
                    {"barrier": "test-epsilon-not-native-scr", "arrivals": arrivals},
                    indent=2,
                )
            )
            with (output / "scr_steady_state.jsonl").open("w") as evidence:
                for snapshot in barrier.observe(server.server_pid, seconds=5):
                    backing = kv_backing_snapshot(
                        snapshot["participants"],
                        (output / "scr_checkpoint_logs" / "process.log").read_text(),
                    )
                    for participant in snapshot["participants"]:
                        participant["host_kv"] = backing[participant["pid"]]
                    evidence.write(json.dumps(snapshot) + "\n")
                    evidence.flush()
                    assert_clean_checkpoint(snapshot, expected_tp_size=4)
            barrier.release()
            self.assertTrue(server.wait_sever_done(timeout=180))
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(
                    f"http://127.0.0.1:{server.port}/",
                    json={
                        "prompt": "The capital of France is",
                        "generate_config": {
                            "max_new_tokens": 4,
                            "top_k": 1,
                            "top_p": 1.0,
                        },
                    },
                    timeout=60,
                )
                (output / "scr_release_response.txt").write_text(response.text)
                response.raise_for_status()
                actual = response.json()
                self.assertNotIn("error_code", actual)
                self.assertIs(actual["finished"], True)
                self.assertTrue(actual["response"].strip())
        except BaseException:
            server.print_process_log()
            raise


if __name__ == "__main__":
    unittest.main()
