"""Exercise physical-window sampling through a real native Agent and model."""

import ctypes
import os
import re
import select
import tempfile
import time
import unittest
from pathlib import Path

from navi_sched import NaviSchedAgent
from navi_sched import _navi_sched_core as core


class VLLMExecutionWindowIntegrationTest(unittest.TestCase):
    def test_observed_windows_train_without_request_lifecycle_events(self):
        self.assertEqual(1, core.VLLM_SHARED_EXECUTION_SAMPLES_VERSION)
        original = os.getcwd()
        agent = None
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        with tempfile.TemporaryDirectory(prefix="navi-execution-window-") as directory:
            root = Path(directory)
            watched = set()
            try:
                os.chdir(directory)
                policy = core.TPUPrefillExecutionPolicyConfig()
                policy.pcp_size = 8
                policy.interleave_size = 256
                policy.compile_sizes = [512, 1024]
                policy.max_scheduled_tokens = 8192
                policy.long_prefill_token_threshold = 8192
                policy.hash_block_size = 16
                policy.scheduler_block_size = 16
                agent = NaviSchedAgent(
                    "execution_window_test",
                    server_ip="127.0.0.1",
                    server_port=17093,
                    total_blocks=1024,
                    block_size=16,
                    linear_step=1,
                    hybrid_tail_required=False,
                    execution_geometry_enabled=True,
                    execution_cache_source_enabled=True,
                    prefill_execution_policy=policy,
                    runtime_config={"thread_num": 2},
                    name="execution_window_agent",
                    enable_kmon=False,
                    enable_snapshot=False,
                    wait_no_query=False,
                    latency_validation_enabled=True,
                    latency_fit_batch_size=1,
                    latency_fit_interval=0,
                    latency_warmup_sample_count=0,
                    latency_cold_start_fit_epochs=1,
                )
                agent.init()
                bridge = core._VLLMAgentBridge(agent, pd_separated=True)

                def record(requests):
                    return bridge.record_observed_execution_sample(
                        requests, 2000.0, 4, 32768, time.time_ns() // 1000
                    )

                self.assertFalse(record([(10, 32768, 0, 0)]))
                self.assertTrue(bridge.enable_observed_execution_window_training())
                self.assertFalse(record([(10, 32768, 0, -1)]))
                for identity in range(1, 13):
                    self.assertTrue(record([(identity, 32768, 0, 0)]))

                deadline = time.monotonic() + 20
                text = ""
                while True:
                    for path in [root, *root.rglob("*")]:
                        if path.is_dir() and path not in watched:
                            result = libc.inotify_add_watch(
                                fd, os.fsencode(path), 0x100 | 0x80 | 0x8
                            )
                            if result < 0:
                                raise OSError(ctypes.get_errno(), str(path))
                            watched.add(path)
                    text = "\n".join(
                        p.read_text(errors="replace")
                        for p in root.rglob("*.log")
                        if p.is_file()
                    )
                    samples = [
                        line
                        for line in text.splitlines()
                        if "[navi-latency-training-sample]" in line
                        and "type=observed_execution_sample " in line
                    ]
                    revisions = [
                        (int(before), int(after))
                        for line in text.splitlines()
                        if "[navi-latency-training-update]" in line
                        for before, after in re.findall(
                            r"revision_before=(\d+) revision_after=(\d+)", line
                        )
                    ]
                    if len(samples) == 12 and any(b > a for a, b in revisions):
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                        self.fail(
                            "native window samples did not train:\n" + text[-4000:]
                        )
                    try:
                        os.read(fd, 65536)
                    except BlockingIOError:
                        pass
            finally:
                if agent is not None:
                    agent.stop()
                os.chdir(original)
                os.close(fd)


if __name__ == "__main__":
    unittest.main()
