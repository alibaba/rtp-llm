"""Host-only subprocess coverage of the real rank entry-point exit contract.

BackendManager and startup helpers (configuration, CUDA, reporting, hot hooks,
OOM dumps and JIT cache setup) are stubbed; native comm cleanup and FUSE unmount
are recorded. This does not exercise model, reporting or GPU resource behavior.
SIGTERM, local_rank_start, its handler/finally control flow and os._exit are real.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def child_main(scenario):
    import rtp_llm.server.backend_manager as managers
    import rtp_llm.start_backend_server as entry
    import rtp_llm.utils.fuser as fuser

    os.environ.pop(entry.DEFER_FIRST_SIGTERM_ENV, None)

    def event(name):
        print(json.dumps({"event": name}), flush=True)

    class FakeBackend:
        def __init__(self, config):
            self.shutdown_received = False

        def start(self):
            event("start")
            if scenario == "startup_failure":
                raise RuntimeError("injected backend startup failure")

        def request_shutdown(self):
            event("shutdown_requested")
            self.shutdown_received = True

        def serve_forever(self):
            os.kill(os.getpid(), signal.SIGTERM)
            # Observe the real handler's callback before returning or raising.
            # A lock-free flag avoids taking a lock from a Python signal handler.
            deadline = time.monotonic() + 5
            while not self.shutdown_received:
                if time.monotonic() >= deadline:
                    raise AssertionError("SIGTERM handler did not request shutdown")
                time.sleep(0.001)
            event("signal_handled")
            if scenario == "shutdown_loop_failure":
                raise RuntimeError("injected shutdown service loop failure")

    def unmount():
        event("unmount")
        if scenario == "unmount_failure":
            raise RuntimeError("injected unmount failure")

    config = SimpleNamespace(
        parallelism_config=SimpleNamespace(local_rank=0, world_size=1),
        server_config=SimpleNamespace(set_local_rank=Mock(), shutdown_timeout=1),
        distribute_config=SimpleNamespace(set_local_rank=Mock()),
        ffn_disaggregate_config=SimpleNamespace(),
        prefill_cp_config=SimpleNamespace(),
        jit_config=SimpleNamespace(remote_jit_dir=""),
    )
    no_ops = (
        "_install_hot_hook_runtime",
        "copy_gemm_config",
        "set_parallelism_config",
        "setup_cuda_device_and_accl_env",
        "prepare_expandable_coexistence",
        "limit_init_segment_splitting",
        "set_global_controller",
        "install_oom_dump",
        "_setup_jit_cache",
    )
    with contextlib.ExitStack() as stack:
        for name in no_ops:
            stack.enter_context(patch.object(entry, name, return_value=None))
        stack.enter_context(patch.object(managers, "BackendManager", FakeBackend))
        stack.enter_context(patch.object(fuser, "umount_all", unmount))
        # clear_cpp_comm_ops is local to local_rank_start and lazily imports this
        # module. Replace that native dependency, keeping the real cleanup wrapper
        # and finally branch under test (there is no entry.clear_cpp_comm_ops).
        stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "librtp_compute_ops": SimpleNamespace(
                        clear_comm_ops=lambda: event("clear_comm_ops")
                    ),
                },
            )
        )
        entry.local_rank_start(global_controller=None, py_env_configs=config)
    raise AssertionError("shutdown entry-point returned instead of hard-exiting")


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

    def test_shutdown_loop_failure_exits_one_without_resource_destructors(self):
        result, events = self.run_child("shutdown_loop_failure")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(events, ["start", "shutdown_requested", "signal_handled"])

    def test_startup_failure_cleans_comm_without_unmount_or_shutdown(self):
        result, events = self.run_child("startup_failure")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(events, ["start", "clear_comm_ops"])
        self.assertIn("injected backend startup failure", result.stderr)

    def test_success_exits_zero_after_comm_cleanup_and_unmount(self):
        result, events = self.run_child("success")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            events,
            [
                "start",
                "shutdown_requested",
                "signal_handled",
                "clear_comm_ops",
                "unmount",
            ],
        )

    def test_unmount_failure_does_not_claim_success(self):
        result, events = self.run_child("unmount_failure")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(
            events,
            [
                "start",
                "shutdown_requested",
                "signal_handled",
                "clear_comm_ops",
                "unmount",
            ],
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child_main(sys.argv[2])
    else:
        unittest.main()
