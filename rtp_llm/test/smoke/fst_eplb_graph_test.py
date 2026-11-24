"""Do not count an unchanged golden from static weights as a dynamic EPLB pass."""

import json
import os
import re
import sys
import unittest
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoke.case_runner import CaseRunner
from smoke.task_info import TaskInfo
from smoke.utils import resolve_prompt_refs

FIXTURE = "data/model/qwen3_moe/q_r_30b_py_tp2.json"


class EplbGraphRunner(CaseRunner):
    def curl_server(self, server_manager):
        # C++ startup may reset the log level. Only count real request traffic,
        # not capture, autotuning, or startup replayAndSyncCheck.
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                f"http://127.0.0.1:{server_manager.port}/set_log_level",
                json={"log_level": "DEBUG"},
                timeout=10,
            )
            response.raise_for_status()
            assert response.json() == {"status": "ok"}, response.text
        with Path(server_manager.log_file_path).open("rb") as log:
            log.seek(0, os.SEEK_END)
            result = super().curl_server(server_manager)
            if not result.ret:
                return result
            saw_commit = saw_post_commit_graph = False
            for line in log:
                if b"[NewLoader EPLB] stage_finalize committed=0" in line:
                    raise AssertionError("EPLB rolled back during the golden request")
                if re.search(
                    rb"\[NewLoader EPLB\] stage_finalize committed=1 layer=\d+ ranks=2\b",
                    line,
                ):
                    saw_commit = True
                elif (
                    saw_commit
                    and b"using CUDA graph forward, is_target_verify=0, is_prefill=0, graph_bs=1"
                    in line
                ):
                    saw_post_commit_graph = True
            assert saw_commit, "No request-time stage-wide two-rank EPLB commit"
            assert saw_post_commit_graph, "No decode graph replay after the EPLB commit"
        return result


class FstEplbGoldenTest(unittest.TestCase):
    def test_qwen3_golden_after_balancing(self):
        fixture = json.loads((Path(__file__).parent / FIXTURE).read_text())
        fixture["query_result"] = [
            resolve_prompt_refs(qr) for qr in fixture["query_result"]
        ]
        runner = EplbGraphRunner(
            task_info=TaskInfo(
                **fixture, taskinfo_rel_path="rtp_llm/test/smoke/" + FIXTURE
            ),
            env_args=["WORLD_SIZE=2", "LOG_LEVEL=DEBUG", "FT_SERVER_TEST=1"],
            gpu_card="H20_CU13",
            smoke_args=os.environ["SMOKE_ARGS"],
        )
        result = runner.run()
        self.assertTrue(result.ret, str(result))


if __name__ == "__main__":
    unittest.main()
