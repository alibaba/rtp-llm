import subprocess
import unittest
from unittest.mock import patch

from gpu_diagnostics import dump_dmesg_errors


class DmesgDiagnosticsTest(unittest.TestCase):
    def collect(self, lines: list[str], baseline: int = 0) -> str:
        output = subprocess.CompletedProcess(
            args=["dmesg", "-T"], returncode=0, stdout="\n".join(lines) + "\n"
        )
        with patch("gpu_diagnostics.subprocess.run", return_value=output):
            return dump_dmesg_errors(baseline)

    def test_oom_charge_and_task_rows_survive_with_fault(self):
        lines = [
            "[1] Memory cgroup stats for /kubepods/pod-example:",
            "[1] memory: usage 262144000kB, limit 262144000kB, failcnt 3",
            "[1] anon 209715200000",
            "[1] shmem 32212254720",
            "[1] [  pid  ]   uid  tgid total_vm      rss pgtables_bytes swapents oom_score_adj name",
            "[1] [ 728231]   500 728231 52428800 40000000  1048576        0             0 rtp_llm_rank-0",
            "[1] oom-kill:constraint=CONSTRAINT_MEMCG,task=rtp_llm_rank-0,pid=728231",
        ]

        report = self.collect(lines)

        self.assertEqual(report.splitlines(), lines)

    def test_multiple_faults_preserve_initial_charge_context(self):
        lines = ["[1] memory: usage 1024kB, limit 1024kB, failcnt 1"] + [
            f"[{index + 2}] NVRM: Xid event {index}" for index in range(40)
        ]

        report = self.collect(lines)

        self.assertEqual(report.splitlines(), lines)

    def test_baseline_excludes_historical_fault_and_context(self):
        report = self.collect(
            ["[1] oom-kill: previous case", "[2] memory: usage 32kB"], baseline=1
        )

        self.assertEqual(report, "(no crash/OOM messages in 1 new dmesg lines)")

    def test_raw_delta_limit_reports_omitted_context(self):
        lines = ["[1] oom-kill:constraint=CONSTRAINT_MEMCG"] + [
            f"[{index + 2}] task row {index}" for index in range(4100)
        ]

        report = self.collect(lines)

        self.assertEqual(report.splitlines()[1:], lines[-4096:])
        self.assertEqual(int(report.splitlines()[0].split()[0].lstrip("(")), 5)
        self.assertIn("omitted", report.splitlines()[0])

    def test_current_fault_excludes_prebaseline_charge_rows(self):
        lines = [
            "[1] memory: historical charge",
            "[2] memory: current charge",
            "[2] oom-kill: current case",
        ]

        report = self.collect(lines, baseline=1)

        self.assertEqual(report.splitlines(), lines[1:])

    def test_command_failures_remain_diagnostic_results(self):
        for error, expected in (
            (FileNotFoundError(), "dmesg not found"),
            (PermissionError(), "dmesg permission denied"),
            (subprocess.TimeoutExpired("dmesg", 10), "dmesg timeout"),
        ):
            with self.subTest(error=type(error).__name__):
                with patch("gpu_diagnostics.subprocess.run", side_effect=error):
                    self.assertEqual(dump_dmesg_errors(), expected)

    def test_nonzero_command_reports_failure(self):
        output = subprocess.CompletedProcess(args=["dmesg", "-T"], returncode=1)
        with patch("gpu_diagnostics.subprocess.run", return_value=output):
            self.assertEqual(dump_dmesg_errors(), "dmesg failed (code 1)")


if __name__ == "__main__":
    unittest.main()
