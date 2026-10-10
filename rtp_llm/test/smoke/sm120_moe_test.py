"""Check MoE golden responses and runtime decode graph replay on one server."""

import json
import os
import re
import sys
import unittest
from pathlib import Path

import requests

# The shared smoke framework imports its modules through the smoke package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoke.case_runner import CaseRunner
from smoke.task_info import TaskInfo
from smoke.utils import resolve_prompt_refs

FIXTURE = Path(__file__).parent / "data/model/qwen35/q_r_35b_fp8pb_tp2_sm120.json"


class GraphReplayRunner(CaseRunner):
    def curl_server(self, server_manager):
        log_path = Path(server_manager.log_file_path)
        try:
            # C++ logger configuration can reset the environment-selected level
            # during startup. Enable runtime evidence after the engine is ready.
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(
                    f"http://127.0.0.1:{server_manager.port}/set_log_level",
                    json={"log_level": "DEBUG"},
                    timeout=10,
                )
                response.raise_for_status()
                assert response.json() == {"status": "ok"}, response.text
            # Startup capture and replayAndSyncCheck do not count as serving traffic.
            offset = log_path.stat().st_size
            result = super().curl_server(server_manager)
            if not result.ret:
                server_manager.stop_server()
                return result
            with log_path.open("rb") as log:
                log.seek(offset)
                runtime = log.read().decode(errors="replace")
            batches = re.findall(
                r"\[PyWrappedModel\] using CUDA graph forward, is_target_verify=0, is_prefill=0, graph_bs=(\d+)",
                runtime,
            )
            output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
            (output / "decode_graph_coverage.json").write_text(
                json.dumps({"runtime_decode_graph_batches": sorted(set(batches))})
            )
            assert {"1", "2"}.issubset(
                batches
            ), f"Missing runtime graph replay: {batches}"
            return result
        except BaseException:
            server_manager.stop_server()
            raise


class SM120MoeTest(unittest.TestCase):
    def test_golden_and_decode_graph(self):
        fixture = json.loads(FIXTURE.read_text())
        fixture["query_result"] = [
            resolve_prompt_refs(qr) for qr in fixture["query_result"]
        ]
        runner = GraphReplayRunner(
            task_info=TaskInfo(
                **fixture,
                taskinfo_rel_path="rtp_llm/test/smoke/data/model/qwen35/q_r_35b_fp8pb_tp2_sm120.json",
            ),
            env_args=[
                "LOAD_PYTHON_MODEL=1",
                "LOG_LEVEL=DEBUG",
                "FT_SERVER_TEST=1",
                "WORLD_SIZE=2",
            ],
            gpu_card="RTX_5000_PRO_CU13",
            smoke_args=os.environ["SMOKE_ARGS"],
        )
        result = runner.run()
        self.assertTrue(result.ret, str(result))


if __name__ == "__main__":
    unittest.main()
