"""Exercise cache recovery after real transfer deadlines on a TP2 model."""

import logging
import os
import re
import time
import unittest
from pathlib import Path

import requests

from rtp_llm.test.smoke.remote_kvcm_server import RemoteKVCMServer
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


_FROZEN_EVENT = re.compile(
    r"cache block (freeze|unfreeze) pool=(.*?) block=(\d+) frozen_ref_count=(\d+)"
)
_FROZEN_METRIC = re.compile(
    rb"\brtp_llm_kv_cache_frozen_ref_blocks\s+\d+\s+(\S+)[^\r\n]*\btier=(DEVICE|HOST)\b"
)


class CacheRecoverySmokeTest(unittest.TestCase):
    def setUp(self):
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.engine_log = output / "logs" / "engine.log"
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)

    def start_kvcm(self):
        runfiles = Path(os.environ["TEST_SRCDIR"])
        workspace = runfiles / os.environ["TEST_WORKSPACE"]
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.kvcm_log = workspace / "logs" / "kv_cache_manager.log"
        self.kvcm = RemoteKVCMServer(
            str(workspace / "3rdparty/remote_kv_cache_manager"),
            {"SEQ_SIZE_PER_BLOCK": "8"},
            str(self.kvcm_log.parent),
            str(output / "kvcm_logs"),
        )
        self.addCleanup(self.kvcm.copy_logs)
        self.addCleanup(self.kvcm.stop_server)
        self.assertTrue(self.kvcm.start_server(), "KVCM did not start")

    def start_model(self, role_name, *, local_transfer=False):
        env_args = {
            **({} if local_transfer else self.kvcm.client_env()),
            "FT_SERVER_TEST": "0" if local_transfer else "1",
            "LOG_LEVEL": "DEBUG",
            "RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH": "0",
            "kmonitorEnableLogFileSink": "true",
        }
        server = MagaServerManager(
            env_args=env_args,
            role_name=role_name,
            smoke_args_str=os.environ["SMOKE_ARGS"],
        )
        self.addCleanup(server.stop_server)
        self.assertTrue(
            server.start_server(
                model_path="/mnt/nas1/hf/Qwen2.5-0.5B-Instruct",
                model_type="qwen_tool",
            )
        )
        response = self.session.post(
            f"http://127.0.0.1:{server.port}/set_log_level",
            json={"log_level": "DEBUG"},
            timeout=10,
        )
        response.raise_for_status()
        self.assertEqual(response.json(), {"status": "ok"})
        return server

    def generate(self, server, prompt, *, max_new_tokens=4):
        response = self.session.post(
            f"http://127.0.0.1:{server.port}/",
            json={
                "prompt": prompt,
                "generate_config": {
                    "max_new_tokens": max_new_tokens,
                    "top_k": 1,
                    "top_p": 1.0,
                    "random_seed": 42,
                    "reuse_cache": True,
                    "return_output_ids": True,
                    "aux_info": True,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        actual = response.json()
        self.assertIs(actual.get("finished"), True, actual)
        self.assertTrue(actual["output_ids"][0], actual)
        return actual

    def wait_for_freeze_metric(self, offset, metric, tags):
        deadline = time.monotonic() + 30
        expected_tags = set(tags)
        with self.engine_log.open("rb") as log:
            log.seek(offset)
            while time.monotonic() < deadline:
                for line in iter(log.readline, b""):
                    _, separator, sample = line.partition(metric + b" ")
                    if separator:
                        _, value, *actual_tags = sample.split()
                        if expected_tags.issubset(actual_tags) and float(value) > 0:
                            return log.tell()
                time.sleep(0.2)
        self.fail(f"No positive {metric!r} sample with {tags!r}")

    def wait_for_zero_frozen_metric(self, offset, *, tiers):
        remaining = set(tiers)
        deadline = time.monotonic() + 30
        with self.engine_log.open("rb") as log:
            log.seek(offset)
            while time.monotonic() < deadline:
                for line in iter(log.readline, b""):
                    sample = _FROZEN_METRIC.search(line)
                    if sample:
                        # Gauges export avg_sum_max_min_count for each interval.
                        maximum = float(sample[1].split(b"_")[2])
                        if maximum == 0:
                            remaining.discard(sample[2])
                            if not remaining:
                                return
                time.sleep(0.2)
        self.fail(f"No zero frozen-block metric sample for {remaining}")

    def wait_for_unfrozen_blocks(self, transfer_log, timeout_pattern):
        deadline = time.monotonic() + 40
        frozen_refs = {}
        while True:
            logs = transfer_log.read_text(errors="replace")
            for event, pool, block, refs in _FROZEN_EVENT.findall(logs):
                if event == "freeze" or (pool, block) in frozen_refs:
                    frozen_refs[pool, block] = int(refs)
            if (frozen_refs and not any(frozen_refs.values())) or time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        self.assertTrue(re.search(timeout_pattern, logs), f"No timeout log matching {timeout_pattern}")
        self.assertTrue(frozen_refs, "The timed-out write did not freeze any blocks")
        self.assertFalse(any(frozen_refs.values()), frozen_refs)
        return frozen_refs

    def test_remote_write_timeout_releases_frozen_blocks(self):
        self.start_kvcm()
        server = self.start_model("remote")
        metric_start = self.engine_log.stat().st_size
        for index, size in enumerate((96, 256, 512, 768, 1024, 1536, 2048, 2048)):
            prompt = f"Remote workload {index}:" + " cache" * size
            cold = self.generate(server, prompt)
            self.assertEqual(cold["aux_info"]["remote_reuse_len"], 0)

        frozen_refs = self.wait_for_unfrozen_blocks(
            Path(server.log_file_path),
            r"KVCM broadcast timed out: broadcast rpc timeout[^\n]*\bdirection=D2REMOTE\b",
        )
        logging.info("Recovered %d frozen source blocks", len(frozen_refs))
        event_end = self.wait_for_freeze_metric(
            metric_start, b"rtp_llm_remote_cache_quarantine_qps", (b"op=write", b"reason=rpc_timeout")
        )
        self.wait_for_zero_frozen_metric(event_end, tiers=(b"DEVICE",))

        # Use a fresh prefix so the recovered server has to publish another write.
        prompt = "Berlin is the capital of Germany. " * 12
        log_start = self.kvcm_log.stat().st_size
        published = self.generate(server, prompt)
        deadline = time.monotonic() + 30
        with self.kvcm_log.open("rb") as log:
            while time.monotonic() < deadline:
                log.seek(log_start)
                if re.search(rb"\[traceId: block_tree_finish_write_\d+\] FinishWriteCache succeeded", log.read()):
                    break
                time.sleep(0.2)
            else:
                self.fail("KVCM did not publish a successful write after recovery")

        # Restart only the model. KVCM retains its published payload, and the
        # second model cannot satisfy the request from the first model's HBM.
        server.stop_server()
        restarted = self.start_model("restarted")
        warm = self.generate(restarted, prompt)
        self.assertEqual(warm["output_ids"], published["output_ids"])
        self.assertEqual(warm["aux_info"]["local_reuse_len"], 0)
        self.assertGreater(warm["aux_info"]["remote_reuse_len"], 0)
        logging.info("Reused %d remote tokens after restart", warm["aux_info"]["remote_reuse_len"])

    def test_local_transfer_timeout_releases_both_tiers(self):
        server = self.start_model("local", local_transfer=True)
        metric_start = self.engine_log.stat().st_size
        # Build several independent, evictable prefixes before crossing the
        # high watermark so the first D2H broadcast has a batch to copy.
        sizes = (96, 192, 256, 384, 512, 512, 768, 896, 1024, 1280, 1536, 2048)
        for index, size in enumerate(sizes):
            label = chr(ord("A") + index) * 16
            self.generate(server, f"{label} local workload {index} " + " cache" * size, max_new_tokens=2)
            # Space requests so each completed prefix can become evictable.
            time.sleep(0.025)

        frozen_refs = self.wait_for_unfrozen_blocks(
            self.engine_log, r"multi-rank transfer RPC failed: broadcast rpc timeout[^\n]*\bdirection=D2H\b"
        )
        pools = {pool for pool, _ in frozen_refs}
        self.assertTrue(any(pool.startswith("block_tree_host_") for pool in pools), pools)
        self.assertIn("default", pools)
        logging.info("Recovered %d frozen DEVICE and HOST blocks", len(frozen_refs))
        event_end = self.wait_for_freeze_metric(
            metric_start,
            b"rtp_llm_kv_cache_transfer_stage_qps",
            (b"operation=evict", b"source_tier=DEVICE", b"target_tier=HOST", b"stage=freeze_blocks"),
        )
        self.wait_for_zero_frozen_metric(event_end, tiers=(b"DEVICE", b"HOST"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
