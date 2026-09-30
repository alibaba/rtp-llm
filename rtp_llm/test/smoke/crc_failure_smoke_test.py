"""Exercise real CRC failure, dump, zero-reuse fallback and subsequent reuse."""

import json
import logging
import os
import re
import struct
import time
import unittest
from pathlib import Path

import requests

from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class CrcFailureSmokeTest(unittest.TestCase):
    def setUp(self):
        self.output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]).resolve()
        self.arm = self.output / "corrupt_next_load"
        workspace = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        injector = workspace / "rtp_llm/test/smoke/libcrc_failure_injector.so"
        self.assertTrue(injector.is_file())
        self.server = MagaServerManager(
            env_args={
                "LOAD_PYTHON_MODEL": "1",
                "LOG_LEVEL": "DEBUG",
                "FT_SERVER_TEST": "1",
                "LD_PRELOAD": str(injector),
                "CRC_SMOKE_ARM_FILE": str(self.arm),
                "DSV4_USE_FRAMEWORK_KV": "1",
                "DG_JIT_CPP_STANDARD": "20",
            },
            role_name="crc_failure",
            smoke_args_str=os.environ["SMOKE_ARGS"],
        )
        self.addCleanup(self.server.stop_server)
        self.assertTrue(
            self.server.start_server(
                model_path=os.environ.get(
                    "CHECKPOINT_PATH", "/mnt/nas1/hf/DeepSeek-V4-Flash-DSpark"
                ),
                model_type="deepseek_v4",
                timeout=3600,
            )
        )
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        level = self.session.post(
            f"http://127.0.0.1:{self.server.port}/set_log_level",
            json={"log_level": "DEBUG"},
            timeout=10,
        )
        level.raise_for_status()

    def request(self, label):
        query = {
            "prompt": "<｜begin▁of▁sentence｜><｜User｜>请忽略以下缓存测试文字，并且只回答OK："
            + "缓存压力测试。" * 160
            + "<｜Assistant｜></think>",
            "generate_config": {
                "max_new_tokens": 1,
                "top_k": 1,
                "top_p": 1.0,
                "random_seed": 42,
                "reuse_cache": True,
                "enable_device_cache": False,
                "enable_memory_cache": True,
                "enable_disk_cache": False,
                "enable_remote_cache": False,
                "return_output_ids": True,
                "aux_info": True,
            },
        }
        response = self.session.post(
            f"http://127.0.0.1:{self.server.port}/", json=query, timeout=300
        )
        (self.output / f"{label}.json").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        self.assertNotIn("error_code", actual)
        self.assertIs(actual["finished"], True)
        self.assertTrue(actual["output_ids"][0])
        logging.info("%s: %s", label, actual)
        return actual

    @staticmethod
    def crc32c(data):
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        return crc ^ 0xFFFFFFFF

    def assert_dump(self):
        injected = json.loads(Path(str(self.arm) + ".injected.json").read_text())
        manifests = list(self.output.glob("**/kv_cache_crc/rank_*/crc_*/manifest.json"))
        self.assertEqual(len(manifests), 1, manifests)
        manifest = manifests[0]
        data = json.loads(manifest.read_text())
        self.assertEqual(int(data["cpu_address"]), injected["address"])
        self.assertEqual(data["failure_stage"], "source_crc")
        self.assertEqual(data["gpu_crc_status"], "0")
        self.assertEqual(data["cpu_captured"], "true")
        self.assertEqual(data["staging_captured"], "true")
        payload = int(data["payload_bytes"])
        self.assertEqual(payload, injected["payload_bytes"])
        cpu = (manifest.parent / "cpu.bin").read_bytes()
        gpu = (manifest.parent / "gpu_staging.bin").read_bytes()
        self.assertEqual(len(cpu), int(data["storage_bytes"]))
        self.assertEqual(len(gpu), payload)
        self.assertEqual(cpu[:payload], gpu)
        self.assertEqual(cpu[0], injected["after"])
        actual = self.crc32c(gpu)
        expected = int(data["expected_crc32c"])
        self.assertEqual(actual, int(data["actual_crc32c"]))
        self.assertNotEqual(actual, expected)
        original = bytearray(gpu)
        original[0] = injected["before"]
        self.assertEqual(self.crc32c(original), expected)
        for name in ("cpu_footer.bin", "gpu_staging_footer.bin", "checked_footer.bin"):
            self.assertEqual(
                struct.unpack("<I", (manifest.parent / name).read_bytes())[0], expected
            )

    def test_crc_failure_falls_back_and_recovers(self):
        cold = self.request("cold")
        self.assertEqual(cold["aux_info"]["reuse_len"], 0)
        # Cache storage completes asynchronously after a response.
        time.sleep(10)
        healthy = self.request("healthy_reuse")
        self.assertGreater(healthy["aux_info"]["memory_reuse_len"], 0)
        time.sleep(10)
        self.arm.touch(exist_ok=False)
        fallback = self.request("crc_fallback")
        self.assertFalse(self.arm.exists(), "The real CRC load was never intercepted")
        self.assertEqual(fallback["aux_info"]["reuse_len"], 0)
        self.assertEqual(fallback["aux_info"]["memory_reuse_len"], 0)
        self.assertEqual(fallback["output_ids"], cold["output_ids"])
        self.assertEqual(fallback["response"], cold["response"])
        self.assert_dump()
        self.assert_detached_reclaimed()
        time.sleep(10)
        recovered = self.request("recovered_reuse")
        self.assertGreater(recovered["aux_info"]["memory_reuse_len"], 0)
        self.assertEqual(recovered["output_ids"], healthy["output_ids"])
        self.assertEqual(
            len(list(self.output.glob("**/kv_cache_crc/rank_*/crc_*/manifest.json"))), 1
        )

    def assert_detached_reclaimed(self):
        deadline = time.monotonic() + 30
        while True:
            log = Path(self.server.log_file_path).read_text(errors="replace")
            events = re.findall(
                r"(detached subtree|reclaimed detached nodes): tree=(0x[0-9a-f]+) "
                r"(?:added|before)=(\d+) remaining=(\d+)",
                log,
            )
            detached = {
                tree
                for kind, tree, count, _ in events
                if kind == "detached subtree" and int(count) > 0
            }
            latest = {tree: int(remaining) for _, tree, _, remaining in events}
            if detached and all(latest[tree] == 0 for tree in detached):
                (self.output / "detached_nodes.json").write_text(
                    json.dumps({"events": events, "remaining": latest}, indent=2)
                )
                return
            self.assertLess(
                time.monotonic(),
                deadline,
                f"Detached nodes did not drain: events={events}",
            )
            time.sleep(0.1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
