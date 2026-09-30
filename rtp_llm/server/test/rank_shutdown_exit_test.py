"""Main's rank entry propagates failures from the shared shutdown owner.

Runs SIGTERM, local_rank_start and BackendManager.stop in a real subprocess.
Model startup, coordination transport, engine resources and FUSE are replaced
with event-recording fakes. This does not qualify native destructor behavior.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def child_main(scenario):
    import rtp_llm.server.backend_manager as managers
    import rtp_llm.start_backend_server as entry

    def event(name):
        print(json.dumps({"event": name}), flush=True)

    def coordinate(*args):
        event("coordinated")
        if scenario == "coordination_failure":
            raise RuntimeError("injected all-rank quiesce failure")

    def unmount():
        event("unmount")
        if scenario == "unmount_failure":
            raise RuntimeError("injected unmount failure")

    class FakeEngine:
        started = True

        def stop(self):
            event("engine_stop")
            if scenario == "engine_stop_failure":
                raise RuntimeError("injected engine stop failure")

    class TestBackend(managers.BackendManager):
        def __init__(self, config):
            self.py_env_configs = config
            self.engine = FakeEngine()
            self.thread_lock_ = threading.Lock()
            self._stopped = False
            self._stop_error = None
            self._shutdown_requested = threading.Event()
            self._shutdown_control = object()
            self._shutdown_incarnation = "test-worker"
            self._distributed_server = SimpleNamespace(store=object())

        def start(self):
            event("start")
            if scenario == "startup_failure":
                raise RuntimeError("injected backend startup failure")

        def request_shutdown(self):
            event("shutdown_requested")
            super().request_shutdown()

        def serve_forever(self):
            os.kill(os.getpid(), signal.SIGTERM)
            if not self._shutdown_requested.wait(5):
                raise AssertionError("SIGTERM handler did not request shutdown")
            event("signal_handled")
            super().serve_forever()

    config = SimpleNamespace(
        parallelism_config=SimpleNamespace(local_rank=0, world_rank=0, world_size=1),
        server_config=SimpleNamespace(set_local_rank=Mock(), shutdown_timeout=1),
        distribute_config=SimpleNamespace(set_local_rank=Mock()),
        ffn_disaggregate_config=SimpleNamespace(),
        prefill_cp_config=SimpleNamespace(),
    )
    no_ops = (
        "_install_hot_hook_runtime",
        "copy_gemm_config",
        "set_parallelism_config",
        "configure_kv_cache_event_host_ip_port",
        "setup_cuda_device_and_accl_env",
        "prepare_expandable_coexistence",
        "limit_init_segment_splitting",
        "set_global_controller",
        "install_oom_dump",
    )
    with contextlib.ExitStack() as stack:
        for name in no_ops:
            stack.enter_context(patch.object(entry, name, return_value=None))
        stack.enter_context(patch.object(managers, "BackendManager", TestBackend))
        stack.enter_context(
            patch(
                "rtp_llm.utils.lifecycle.shutdown.graceful_backend_shutdown", coordinate
            )
        )
        stack.enter_context(
            patch(
                "rtp_llm.utils.fuser._nfs_manager", SimpleNamespace(unmount_all=unmount)
            )
        )
        entry.local_rank_start(global_controller=None, py_env_configs=config)


class RankShutdownExitTest(unittest.TestCase):
    def run_child(self, scenario):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", scenario],
            capture_output=True,
            text=True,
            timeout=45,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        )
        events = []
        for line in result.stdout.splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and "event" in value:
                events.append(value["event"])
        return result, events

    def test_coordination_failure_exits_one_without_resource_cleanup(self):
        result, events = self.run_child("coordination_failure")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(
            events, ["start", "shutdown_requested", "signal_handled", "coordinated"]
        )
        self.assertIn("injected all-rank quiesce failure", result.stderr)

    def test_startup_failure_does_not_claim_shutdown_success(self):
        result, events = self.run_child("startup_failure")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(events, ["start"])
        self.assertIn("injected backend startup failure", result.stderr)

    def test_success_exits_zero_after_coordination_stop_and_unmount(self):
        result, events = self.run_child("success")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            events,
            [
                "start",
                "shutdown_requested",
                "signal_handled",
                "coordinated",
                "engine_stop",
                "unmount",
            ],
        )

    def test_cleanup_failures_do_not_claim_success(self):
        for scenario in ("engine_stop_failure", "unmount_failure"):
            with self.subTest(scenario=scenario):
                result, events = self.run_child(scenario)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(events[-3:], ["coordinated", "engine_stop", "unmount"])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child_main(sys.argv[2])
    else:
        unittest.main()
