"""GB200 CP2/DP2 P/D smoke with CED, DSpark, Engram, verify graph and resident weights."""

import json
import logging
import os
import re
import unittest
import uuid
from pathlib import Path

from rtp_llm.resident_weights import store
from rtp_llm.test.smoke import deepseek_v41_dspark_contract_test as dspark
from rtp_llm.test.smoke import deepseek_v41_pd_contract_test as pd

EXPECTED_SLOTS = 8  # 4 GPUs (2 prefill + 2 decode) x 2 roles (target + draft)


class DeepSeekV41FullContractTest(
    dspark.DeepSeekV41DSparkContractTest, pd.DeepSeekV41PDContractTest
):
    prefill_cp_size = 2
    decode_dp_size = 2
    short_prompt = (
        "Write the numbers from one to twenty, separated by commas: one, two,"
    )
    # Main's speculative output reports acceptance, not generation scores.
    # Keep the PD token/text repeatability and CED checks, and the DSpark
    # acceptance and verify-graph checks inherited from the two runners.
    compare_generation_logits = False

    def setUp(self):
        self.store_name = f"ci_dsv41_{uuid.uuid4().hex}"
        previous = os.environ.get("RTP_LLM_RESIDENT_STORE")
        os.environ["RTP_LLM_RESIDENT_STORE"] = self.store_name

        def restore_store_name():
            if previous is None:
                os.environ.pop("RTP_LLM_RESIDENT_STORE", None)
            else:
                os.environ["RTP_LLM_RESIDENT_STORE"] = previous

        self.addCleanup(restore_store_name)
        # Keep the store outside both server process trees. It outlives server restarts,
        # but its owner PID bounds its lifetime if the test is killed before cleanup.
        self.addCleanup(self._stop_store)
        store.ensure_running(self.store_name, owner_pid=os.getpid())
        super().setUp()

    def _stop_store(self):
        try:
            store.stop_store(self.store_name)
        except (OSError, store.StoreError):
            pass

    def _all_role_logs(self) -> str:
        chunks = []
        for manager in (self.server, self.graph_server):
            log_dir = Path(manager.log_file_path).parent
            chunks.extend(
                path.read_text(errors="replace")
                for path in sorted(log_dir.glob("*.log"))
            )
        return "\n".join(chunks)

    def _assert_resident_weights_restart(self, cold_short: dict) -> None:
        slots = dict(store.list_store(self.store_name).slots or {})
        self.assertEqual(len(slots), EXPECTED_SLOTS, sorted(slots))
        for key, summary in slots.items():
            self.assertGreater(summary.total_bytes, 0, key)
            self.assertGreater(summary.allocations, 0, key)

        self.server.stop_server()
        self.graph_server.stop_server()
        self.assertEqual(set(store.list_store(self.store_name).slots or {}), set(slots))

        self.start_servers(role_suffix="_warm")
        for manager in (self.server, self.graph_server):
            level = self.session.post(
                f"http://127.0.0.1:{manager.port}/set_log_level",
                json={"log_level": "DEBUG"},
                timeout=10,
            )
            level.raise_for_status()
        warm_log_offset = Path(self.graph_server.log_file_path).stat().st_size

        warm_logs = self._all_role_logs()
        attached = set(re.findall(r"attached resident weights (\S+):", warm_logs))
        self.assertEqual(
            attached, set(slots), "Not every rank/role attached from store"
        )
        self.assertNotIn("load weights took", warm_logs)
        self.assertIn("DeepSeek V4.1 Engram initialized on layers", warm_logs)
        self.assertIn("capture success for batch size", warm_logs)

        warm_short = self.request("warm_short_uncached", self.short_prompt, False)
        self.assertEqual(warm_short["output_ids"], cold_short["output_ids"])
        self.assertEqual(warm_short["response"], cold_short["response"])
        with Path(self.graph_server.log_file_path).open("rb") as reader:
            reader.seek(warm_log_offset)
            runtime = reader.read().decode(errors="replace")
        self.assertRegex(
            runtime,
            r"\[PyWrappedModel\] using CUDA graph forward, is_target_verify=1, is_prefill=1, graph_bs=1",
        )

        self.server.stop_server()
        self.graph_server.stop_server()
        store.stop_store(self.store_name)
        with self.assertRaises((OSError, store.StoreError)):
            store.list_store(self.store_name)

    def test_graph_engram_and_prefix_reuse(self):
        super().test_graph_engram_and_prefix_reuse()
        actual = {}
        for record in self.observed:
            label = record["label"]
            result = json.loads((self.output_dir / f"{label}.json").read_text())
            actual[label] = {
                "query": result["query"],
                "output_ids": result["actual"]["output_ids"],
                "response": result["actual"]["response"],
                "input_len": record["input_len"],
                "prefill_reuse_len": record["measured_reuse_len"],
            }
        (self.output_dir / "full_feature_golden_actual.json").write_text(
            json.dumps(actual, ensure_ascii=False, indent=2) + "\n"
        )
        if os.environ.get("SAVE_RESPONSE", "False") != "True":
            golden = (
                Path(__file__).parent
                / "data/model/deepseek_v41/full_feature_cp2_dp2.json"
            )
            self.assertEqual(actual, json.loads(golden.read_text()))

        self._assert_resident_weights_restart(actual["short_uncached"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
