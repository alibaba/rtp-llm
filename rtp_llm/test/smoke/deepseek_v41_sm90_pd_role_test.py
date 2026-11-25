import fcntl
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.test.smoke import deepseek_v41_sm90_pd_role as role


def options(**updates):
    values = dict(role="decode", local_concurrency=4, gamma=0, verify_budget=3,
                  graph_max_rows=None, max_seq_len=32768, load_method="scratch",
                  kv_cache_mem_mb=1024, checkpoint="/checkpoint", log_level="INFO",
                  peer_port=33120, peer_ip="10.0.0.1")
    values.update(updates)
    return SimpleNamespace(**values)


class Sm90PDRoleTest(unittest.TestCase):
    def test_proc_identity_is_observed_instead_of_inferred_from_poll(self):
        if not Path("/proc/self/stat").exists():
            self.skipTest("Linux /proc identity observation")
        identity = role.process_identity(os.getpid())
        self.assertTrue(identity["exists"])
        self.assertEqual(identity["uid"], os.getuid())
        self.assertTrue(identity["starttime"].isdigit())
        self.assertFalse(role.process_identity(None)["exists"])

    def test_build_proxy_is_removed_from_runtime_and_both_peers_bypass(self):
        env = dict(HTTP_PROXY="build-only", https_proxy="build-only", ALL_PROXY="build-only",
                   NO_PROXY="existing.internal", no_proxy="127.0.0.1")
        removed = role.prepare_runtime_network_env(env, "10.0.0.1", "10.0.0.2")
        self.assertEqual(set(removed), {"HTTP_PROXY", "https_proxy", "ALL_PROXY"})
        self.assertFalse(any(name.lower() in ("http_proxy", "https_proxy", "all_proxy") for name in env))
        self.assertEqual(env["NO_PROXY"], env["no_proxy"])
        self.assertTrue({"existing.internal", "localhost", "127.0.0.1", "10.0.0.1", "10.0.0.2"}.issubset(env["NO_PROXY"].split(",")))

    def test_prefill_matches_speculative_cache_mode_without_packed_verify(self):
        args, env = role.recipe(options(role="prefill", gamma=5))
        self.assertEqual((args["tp_size"], args["dp_size"], args["ep_size"]), (8, 1, 8))
        self.assertEqual(args["sp_type"], "dspark")
        self.assertEqual(args["sp_model_type"], "deepseek_v41_dspark")
        self.assertEqual(args["sp_checkpoint_path"], options().checkpoint)
        self.assertEqual(args["max_context_batch_size"], 1)
        self.assertEqual(args["concurrency_limit"], 128)
        self.assertFalse(any(name.startswith("SP_DSPARK_") for name in env))
        self.assertNotIn("RTP_LLM_DROP_BROAD_SYNC", env)
        self.assertEqual(env["REMOTE_RPC_SERVER_IP"], "10.0.0.1")
        baseline, _ = role.recipe(options(role="prefill", gamma=0, verify_budget=0))
        self.assertEqual(baseline["sp_type"], "none")
        self.assertNotIn("sp_checkpoint_path", baseline)

    def test_real_decode_baseline_and_gamma5_geometry(self):
        args, env = role.recipe(options(verify_budget=0))
        self.assertEqual((args["tp_size"], args["dp_size"], args["ep_size"]), (1, 8, 8))
        self.assertEqual(args["sp_type"], "none")
        self.assertEqual(args["use_batch_decode_scheduler"], 0)
        self.assertEqual(args["cache_store_rdma_mode"], 0)
        self.assertFalse(any(name.startswith("SP_DSPARK_") for name in env))
        self.assertEqual(env["RTP_LLM_STREAM_ASYNC"], "1")
        self.assertEqual(env["RTP_LLM_DROP_BROAD_SYNC"], "1")
        self.assertEqual(env["RTP_LLM_DEVICE_INPUT"], "1")
        args, env = role.recipe(options(gamma=5, local_concurrency=16))
        self.assertEqual(args["sp_type"], "dspark")
        self.assertEqual(args["decode_capture_config"], "1,2,4,16")
        self.assertEqual(env["DSV41_SM90_DECODE_GRAPH_MAX_ROWS"], "96")
        self.assertEqual(env["SP_DSPARK_VERIFY_BUDGET"], "3")
        self.assertEqual(env["SP_DSPARK_VERIFY_MODE"], "packed")
        self.assertEqual(env["SP_DSPARK_CONFIDENCE_MODE"], "bounded")
        with self.assertRaises(ValueError):
            role.recipe(options(gamma=5, local_concurrency=16, graph_max_rows=64))

    def test_same_inode_readonly_locks_interoperate_across_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            global_path, gpu_root = root / "global", root / "gpus"
            gpu_root.mkdir()
            paths = [global_path, gpu_root / "0", gpu_root / "1"]
            for path in paths:
                path.touch()
                path.chmod(0o444)
            before = [(p.stat().st_dev, p.stat().st_ino, p.stat().st_mode) for p in paths]
            probe = "import os,fcntl,sys; f=os.open(sys.argv[1],os.O_RDONLY); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
            with role.canonical_lease([0, 1], 0, global_path, gpu_root) as lease:
                self.assertEqual(len(lease["records"]), 3)
                for path in paths[1:]:
                    result = subprocess.run([sys.executable, "-c", probe, str(path)], capture_output=True)
                    self.assertNotEqual(result.returncode, 0)
                self.assertEqual(subprocess.run([sys.executable, "-c", probe, str(global_path)]).returncode, 0)
            for path in paths:
                self.assertEqual(subprocess.run([sys.executable, "-c", probe, str(path)]).returncode, 0)
            self.assertEqual(before, [(p.stat().st_dev, p.stat().st_ino, p.stat().st_mode) for p in paths])

    def test_missing_canonical_lock_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing"
            with self.assertRaises(FileNotFoundError):
                with role.canonical_lease([0], 0, path, Path(directory)):
                    self.fail("missing lock accepted")
            self.assertFalse(path.exists())

    def test_done_file_requires_current_run_and_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            done = Path(directory) / "done.json"
            self.assertFalse(role.done_requested(done, "current"))
            done.write_text(json.dumps(dict(run_id="old", action="stop")))
            with self.assertRaisesRegex(ValueError, "wrong run-id"):
                role.done_requested(done, "current")
            done.write_text(json.dumps(dict(run_id="current", action="stop")))
            self.assertTrue(role.done_requested(done, "current"))

    def test_occupied_ports_are_rejected(self):
        with socket.socket() as sock:
            sock.bind(("0.0.0.0", 0))
            with self.assertRaises(OSError):
                role.port_precheck(sock.getsockname()[1])

    def test_ready_requires_actual_all_dp_and_startup_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "process.log"
            log.write_text("\n".join(["Backend server initialized successfully, sending ready status"] * 8 + ["SM90 Engram UVA prefetch enabled", "capture success for batch size1", "parallelism: test"]) + "\n")
            manager = SimpleNamespace(log_file_path=str(log), _server_process=SimpleNamespace(poll=lambda: None))
            args = SimpleNamespace(port=33240, role="decode", host_ip="10.0.0.2")
            status = dict(results=[dict(dp_rank=rank, alive=True, address=f"10.0.0.2:{33241 + 10 * rank}") for rank in range(8)])
            with mock.patch.object(role, "get_status", return_value=status):
                actual, markers = role.wait_fixture_ready(manager, args, time.monotonic() + 1, lambda text, name: None)
            self.assertEqual(actual, status)
            self.assertIn("capture success", markers)


if __name__ == "__main__":
    unittest.main()
