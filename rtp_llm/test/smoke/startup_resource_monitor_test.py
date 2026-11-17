import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import startup_resource_monitor as monitor


class StartupResourceMonitorTest(unittest.TestCase):
    def test_cgroup_mount_roots_and_escaped_mountpoints(self) -> None:
        mountinfo = (
            "1 0 0:1 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
            "2 0 0:2 /tenant /cpu\\040group rw - cgroup cgroup rw,cpu,cpuacct\n"
            "3 0 0:3 /elsewhere /wrong rw - cgroup2 cgroup rw\n"
        )
        self.assertEqual(
            monitor._cgroup_directories(
                "0::/tenant/job/rank\n2:cpu,cpuacct:/tenant/job", mountinfo
            ),
            [
                Path(p)
                for p in (
                    "/sys/fs/cgroup/job/rank",
                    "/sys/fs/cgroup/job",
                    "/sys/fs/cgroup",
                    "/cpu group/job",
                    "/cpu group",
                )
            ],
        )

    def test_sample_observes_real_waiting_child_without_terminating_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(
            max_workers=1
        ) as launcher:
            shard = Path(directory) / "source.safetensors"
            shard.write_bytes(b"x" * 4096)
            with launcher.submit(
                subprocess.Popen,
                [
                    sys.executable,
                    "-c",
                    (
                        "import mmap,sys,threading\n"
                        "stopped = threading.Event()\n"
                        "def wait():\n"
                        " print(threading.get_native_id(), flush=True)\n"
                        " stopped.wait()\n"
                        "with open(sys.argv[1], 'rb') as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as view:\n"
                        " worker = threading.Thread(target=wait)\n"
                        " worker.start()\n"
                        " sys.stdin.read()\n"
                        " stopped.set()\n"
                        " worker.join()\n"
                    ),
                    str(shard),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            ).result() as child:
                assert child.stdout is not None
                assert child.stdin is not None
                native_tid = int(child.stdout.readline().strip())
                output = Path(directory) / "resources.jsonl"
                sampled = threading.Event()
                snapshot = monitor._snapshot

                def observe(pid: int) -> monitor.ResourceSample:
                    result = snapshot(pid)
                    if str(child.pid) in result["processes"]:
                        sampled.set()
                    return result

                try:
                    with patch.object(monitor, "_snapshot", side_effect=observe):
                        with monitor.monitor_startup_resources(output):
                            self.assertTrue(sampled.wait(timeout=5))
                            self.assertIsNone(child.poll())
                finally:
                    child.stdin.close()
                    self.assertEqual(child.wait(timeout=5), 0)
                sample = json.loads(output.read_text().splitlines()[0])
                record = sample["processes"][str(child.pid)]
                self.assertEqual(int(record["stat"].split()[0]), child.pid)
                self.assertIn("read_bytes:", record["io"])
                self.assertIn("NSpid:", record["status"])
                self.assertIn(str(shard), record["maps"])
                self.assertIn(str(shard.stat().st_ino), record["maps"])
                threads = json.loads(record["threads"])
                waiting_thread = threads["threads"][str(native_tid)]
                self.assertEqual(int(waiting_thread["stat"].split()[0]), native_tid)
                self.assertIn("read_bytes:", waiting_thread["io"])
                self.assertTrue(waiting_thread["wchan"])
                self.assertEqual(
                    waiting_thread["stat"].rsplit(")", 1)[1].split()[19],
                    waiting_thread["stat_after"].rsplit(")", 1)[1].split()[19],
                )
                self.assertFalse(threads["thread_limit_reached"])
                self.assertEqual(
                    record["cgroup"], Path("/proc/self/cgroup").read_text()
                )
                self.assertIn("MemAvailable:", sample["host"]["meminfo"])
                self.assertGreaterEqual(int(sample["host"]["tmpfs_free_bytes"]), 0)

    def test_artifact_is_flushed_and_closed_after_sampler_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "resources.jsonl"
            with monitor.monitor_startup_resources(output):
                pass
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["root_pid"], os.getpid())
            self.assertIn(str(os.getpid()), records[0]["processes"])

    def test_missing_and_oversized_files_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            self.assertIn("ERROR errno=2:", monitor._read(path))
            path.write_text("x" * (monitor._FILE_LIMIT + 100))
            self.assertEqual(
                monitor._read(path), "x" * monitor._FILE_LIMIT + "\nTRUNCATED"
            )


if __name__ == "__main__":
    unittest.main()
