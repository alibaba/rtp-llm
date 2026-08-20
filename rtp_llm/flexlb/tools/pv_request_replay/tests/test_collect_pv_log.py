from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collect_pv_log as collector  # noqa: E402


def log_line(timestamp: str, request_id: str) -> str:
    return (
        f'{timestamp} [routing-queue-worker] INFO pvLogger - '
        f'{{"requestId":"{request_id}"}}\n'
    )


class FakeDashctl:
    def __init__(self, files: dict[str, list[str]], response_limit: int = 220):
        self.files = files
        self.response_limit = response_limit
        self.commands: list[list[str]] = []
        self.stat_calls: dict[str, int] = {}

    def __call__(self, command: Sequence[str]) -> str:
        command = list(command)
        self.commands.append(command)
        if "get" in command:
            return json.dumps(
                [
                    {
                        "metadata": {"name": "inst/running"},
                        "status": {"name": "running"},
                    },
                    {
                        "metadata": {"name": "inst-stopped"},
                        "status": {"name": "stopped"},
                    },
                ]
            )

        remote = command[command.index("--") + 1 :]
        if remote[:2] == ["ls", "-1"]:
            names = [Path(path).name for path in self.files]
            return "\n".join(names + ["application.log", "pv.log.1"]) + "\n"
        if remote[:2] == ["wc", "-l"]:
            path = remote[2]
            return f"{len(self.files[path])} {path}\n"
        if remote[:2] == ["head", "-n"]:
            path = remote[3]
            return self.files[path][0] if self.files[path] else ""
        if remote[:2] == ["tail", "-n"]:
            path = remote[3]
            if remote[2] == "1":
                return self.files[path][-1] if self.files[path] else ""
            start = int(remote[2][1:])
            output = "".join(self.files[path][start - 1 :])
            return output[: self.response_limit]
        if remote[:2] == ["stat", "-c"]:
            path = remote[3]
            self.stat_calls[path] = self.stat_calls.get(path, 0) + 1
            inode = list(self.files).index(path) + 100
            size = sum(len(line.encode("utf-8")) for line in self.files[path])
            return f"{inode} {size}\n"
        raise AssertionError(f"unexpected command: {command}")


class ChangingStatDashctl(FakeDashctl):
    def __init__(
        self,
        files: dict[str, list[str]],
        stat_values: list[tuple[int, int]],
    ) -> None:
        super().__init__(files, response_limit=10_000)
        self.stat_values = list(stat_values)

    def __call__(self, command: Sequence[str]) -> str:
        command = list(command)
        if "--" in command:
            remote = command[command.index("--") + 1 :]
            if remote[:2] == ["stat", "-c"]:
                self.commands.append(command)
                inode, size = self.stat_values.pop(0)
                return f"{inode} {size}\n"
        return super().__call__(command)


class NewlineTerminatingTruncationDashctl(FakeDashctl):
    """Emulate a transport that adds a newline after a partial log record."""

    def __call__(self, command: Sequence[str]) -> str:
        command = list(command)
        if "--" in command:
            remote = command[command.index("--") + 1 :]
            if remote[:2] == ["tail", "-n"] and remote[2].startswith("+"):
                self.commands.append(command)
                path = remote[3]
                start = int(remote[2][1:])
                output = "".join(self.files[path][start - 1 :])
                if len(output) > self.response_limit:
                    return output[: self.response_limit].rstrip("\n") + "\n"
                return output
        return super().__call__(command)


class CollectPvLogTest(unittest.TestCase):
    def test_local_multi_instance_filters_extended_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            instance_a = root / "instance-a.log"
            instance_a.write_text(
                "not a timestamp\n"
                + log_line("2026-08-11 01:59:59.999", "too-early")
                + log_line("2026-08-11 02:00:00.000", "lead-edge")
                + log_line("2026-08-11 02:05:00.000", "inside")
                + log_line("2026-08-11 02:10:00.000", "tail-edge")
                + log_line("2026-08-11 02:10:00.001", "too-late"),
                encoding="utf-8",
            )
            instance_b = root / "instance-b.log"
            instance_b.write_text(
                log_line("2026-08-11 02:00:00.000", "b-start")
                + log_line("2026-08-11 02:10:00.000", "b-end"),
                encoding="utf-8",
            )
            output = root / "output"

            manifest = collector.collect_logs(
                workspace="unused",
                deployment=None,
                instances=None,
                start="2026-08-11 02:01:00",
                end="2026-08-11 02:09:00",
                output_dir=output,
                lead_grace=timedelta(minutes=1),
                tail_grace=timedelta(minutes=1),
                local_inputs={
                    "instance/a": instance_a,
                    "instance-b": instance_b,
                },
            )

            self.assertEqual("complete", manifest["status"])
            self.assertEqual("Asia/Shanghai", manifest["timezone"])
            self.assertEqual(2, len(manifest["snapshots"]))
            snapshot = next(
                item for item in manifest["snapshots"] if item["instance"] == "instance/a"
            )
            self.assertEqual(3, snapshot["line_count"])
            self.assertIn("instance_a", snapshot["path"])
            contents = (output / snapshot["path"]).read_text(encoding="utf-8")
            self.assertIn("lead-edge", contents)
            self.assertIn("inside", contents)
            self.assertIn("tail-edge", contents)
            self.assertNotIn("too-early", contents)
            self.assertNotIn("too-late", contents)
            source_file = next(
                item
                for item in manifest["instances"]
                if item["instance"] == "instance/a"
            )["source_files"][0]
            self.assertEqual(1, source_file["unparsable_line_count"])
            self.assertTrue((output / "collect_manifest.json").is_file())

    def test_aware_request_times_are_converted_to_shanghai_wall_clock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "pv.log"
            source.write_text(
                log_line("2026-08-11 01:59:59.999", "before")
                + log_line("2026-08-11 02:00:00.000", "selected")
                + log_line("2026-08-11 02:00:00.001", "after"),
                encoding="utf-8",
            )
            manifest = collector.collect_logs(
                workspace="unused",
                deployment=None,
                instances=None,
                start="2026-08-10T18:00:00+00:00",
                end="2026-08-10T18:00:00+00:00",
                output_dir=root / "output",
                lead_grace=timedelta(0),
                tail_grace=timedelta(0),
                local_inputs={"local": source},
            )

            self.assertEqual("2026-08-11 02:00:00.000", manifest["requested_window"]["start"])
            self.assertEqual(1, manifest["snapshots"][0]["line_count"])

    def test_remote_collection_resolves_running_and_pages_truncated_tail(self) -> None:
        log_dir = "/home/admin/logs"
        rotated = f"{log_dir}/pv.log.2026-08-11.0.log"
        current = f"{log_dir}/pv.log"
        files = {
            rotated: [
                log_line("2026-08-11 02:00:10.000", "r1"),
                log_line("2026-08-11 02:00:11.000", "r2"),
                log_line("2026-08-11 02:00:12.000", "r3"),
                log_line("2026-08-11 02:00:13.000", "r4"),
            ],
            current: [
                log_line("2026-08-11 02:00:14.000", "c1"),
                log_line("2026-08-11 02:00:20.000", "c2"),
            ],
        }
        fake = FakeDashctl(files, response_limit=220)
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = collector.collect_logs(
                workspace="ai-lab-test",
                deployment="flexlb-deployment",
                instances=None,
                start="2026-08-11 02:00:10",
                end="2026-08-11 02:00:20",
                output_dir=temp_dir,
                lead_grace=timedelta(0),
                tail_grace=timedelta(0),
                page_lines=1,
                command_runner=fake,
            )

            self.assertEqual(["running"], manifest["source"]["resolved_instances"])
            self.assertEqual("complete", manifest["status"])
            self.assertEqual(6, manifest["snapshots"][0]["line_count"])
            self.assertFalse(manifest["snapshot_truncated"])
            rotated_manifest = manifest["instances"][0]["source_files"][0]
            self.assertGreater(len(rotated_manifest["pages"]), 1)
            self.assertGreaterEqual(rotated_manifest["pages"][0]["line_count"], 1)
            self.assertTrue(
                any(
                    page["transport_response_truncated"]
                    for page in rotated_manifest["pages"]
                )
            )
            snapshot_text = (
                Path(temp_dir) / manifest["snapshots"][0]["path"]
            ).read_text()
            self.assertNotIn("application.log", snapshot_text)
            self.assertNotIn("pv.log.1", [Path(path).name for path in files])
            exec_commands = [command for command in fake.commands if "exec" in command]
            self.assertTrue(exec_commands)
            self.assertTrue(
                all(
                    command[command.index("exec") + 2 : command.index("--")]
                    == ["-c", "worker0"]
                    for command in exec_commands
                )
            )

    def test_remote_collection_recovers_newline_terminated_partial_boundary(self) -> None:
        path = "/home/admin/logs/pv.log"
        files = {
            path: [
                log_line(
                    f"2026-08-11 02:00:0{index}.000",
                    f"request-{index}" + "x" * 80,
                )
                for index in range(4)
            ]
        }
        fake = NewlineTerminatingTruncationDashctl(
            files, response_limit=len(files[path][0]) + 20
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = collector.collect_logs(
                workspace="ai-lab-test",
                deployment="flexlb-deployment",
                instances=None,
                start="2026-08-11 02:00:00",
                end="2026-08-11 02:00:03",
                output_dir=temp_dir,
                lead_grace=timedelta(0),
                tail_grace=timedelta(0),
                command_runner=fake,
            )

            self.assertEqual("complete", manifest["status"])
            source = manifest["instances"][0]["source_files"][0]
            self.assertFalse(source["snapshot_truncated"])
            self.assertTrue(
                any(
                    page["boundary_verified"] is False
                    for page in source["pages"]
                )
            )
            snapshot = Path(temp_dir) / manifest["snapshots"][0]["path"]
            self.assertEqual("".join(files[path]), snapshot.read_text())

    def test_remote_skips_files_whose_probed_bounds_do_not_overlap(self) -> None:
        log_dir = "/home/admin/logs"
        old = f"{log_dir}/pv.log.2026-08-11.0.log"
        selected = f"{log_dir}/pv.log.2026-08-11.1.log"
        current = f"{log_dir}/pv.log"
        files = {
            old: [
                log_line("2026-08-11 01:00:00.000", "old-first"),
                log_line("2026-08-11 01:01:00.000", "old-last"),
            ],
            selected: [
                log_line("2026-08-11 02:00:10.000", "selected-first"),
                log_line("2026-08-11 02:00:20.000", "selected-last"),
            ],
            current: [
                log_line("2026-08-11 03:00:00.000", "future-first"),
                log_line("2026-08-11 03:01:00.000", "future-last"),
            ],
        }
        fake = FakeDashctl(files, response_limit=10_000)
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = collector.collect_logs(
                workspace="ai-lab-test",
                deployment="flexlb-deployment",
                instances=None,
                start="2026-08-11 02:00:10",
                end="2026-08-11 02:00:20",
                output_dir=temp_dir,
                lead_grace=timedelta(0),
                tail_grace=timedelta(0),
                command_runner=fake,
            )

            self.assertEqual("complete", manifest["status"])
            self.assertEqual(2, manifest["snapshots"][0]["line_count"])
            source_files = manifest["instances"][0]["source_files"]
            self.assertEqual(
                ["skipped_outside_window", "complete", "skipped_outside_window"],
                [item["status"] for item in source_files],
            )
            fetched_paths = []
            for command in fake.commands:
                if "--" not in command:
                    continue
                remote = command[command.index("--") + 1 :]
                if (
                    remote[:2] == ["tail", "-n"]
                    and remote[2].startswith("+")
                ):
                    fetched_paths.append(remote[3])
            # The collector re-reads the last apparent line of a response to
            # verify that the transport did not add a newline to a partial
            # record, so a small one-file snapshot needs an overlap request.
            self.assertGreaterEqual(len(fetched_paths), 2)
            self.assertTrue(all(path == selected for path in fetched_paths))
            self.assertEqual(2, fake.stat_calls[current])

    def test_active_log_rollover_or_shrink_is_never_complete(self) -> None:
        path = "/home/admin/logs/pv.log"
        lines = [
            log_line("2026-08-11 02:00:00.000", "first"),
            log_line("2026-08-11 02:00:01.000", "last"),
        ]
        original_size = sum(len(line.encode("utf-8")) for line in lines)
        scenarios = {
            "rollover": [(101, original_size), (202, original_size)],
            "shrink": [(101, original_size), (101, original_size - 1)],
        }
        for name, stat_values in scenarios.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                fake = ChangingStatDashctl({path: lines}, stat_values)
                manifest = collector.collect_logs(
                    workspace="ai-lab-test",
                    deployment="flexlb-deployment",
                    instances=None,
                    start="2026-08-11 02:00:00",
                    end="2026-08-11 02:00:01",
                    output_dir=temp_dir,
                    lead_grace=timedelta(0),
                    tail_grace=timedelta(0),
                    command_runner=fake,
                )

                self.assertEqual("failed", manifest["status"])
                self.assertEqual([], manifest["snapshots"])
                self.assertIn("active log changed", manifest["errors"][0])

    def test_rotation_index_gap_marks_collection_partial(self) -> None:
        log_dir = "/home/admin/logs"
        files = {
            f"{log_dir}/pv.log.2026-08-11.0.log": [
                log_line("2026-08-11 02:00:00.000", "first")
            ],
            f"{log_dir}/pv.log.2026-08-11.2.log": [
                log_line("2026-08-11 02:00:10.000", "middle")
            ],
            f"{log_dir}/pv.log": [
                log_line("2026-08-11 02:00:20.000", "last")
            ],
        }
        fake = FakeDashctl(files, response_limit=10_000)
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = collector.collect_logs(
                workspace="ai-lab-test",
                deployment="flexlb-deployment",
                instances=None,
                start="2026-08-11 02:00:00",
                end="2026-08-11 02:00:20",
                output_dir=temp_dir,
                lead_grace=timedelta(0),
                tail_grace=timedelta(0),
                command_runner=fake,
            )

            self.assertEqual("partial", manifest["status"])
            self.assertEqual(1, len(manifest["snapshots"]))
            self.assertIn("rotation index gap", manifest["errors"][0])
            self.assertIn("missing 1", manifest["errors"][0])
            snapshot_path = Path(manifest["snapshots"][0]["path"])
            self.assertFalse(snapshot_path.is_absolute())
            self.assertTrue((Path(temp_dir) / snapshot_path).is_file())

    def test_non_strict_failure_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output"
            manifest = collector.collect_logs(
                workspace="unused",
                deployment=None,
                instances=None,
                start="2026-08-11 02:00:00",
                end="2026-08-11 02:01:00",
                output_dir=output,
                local_inputs={"missing": Path(temp_dir) / "does-not-exist.log"},
            )

            self.assertEqual("failed", manifest["status"])
            self.assertTrue(manifest["snapshot_truncated"])
            self.assertIn("does-not-exist.log", manifest["errors"][0])
            stored = json.loads((output / "collect_manifest.json").read_text())
            self.assertEqual("failed", stored["status"])

    def test_remote_mid_line_truncation_records_failed_file_range(self) -> None:
        path = "/home/admin/logs/pv.log"
        fake = FakeDashctl(
            {path: [log_line("2026-08-11 02:00:00.000", "request")]},
            response_limit=20,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = collector.collect_logs(
                workspace="ai-lab-test",
                deployment="flexlb-deployment",
                instances=None,
                start="2026-08-11 02:00:00",
                end="2026-08-11 02:00:01",
                output_dir=temp_dir,
                command_runner=fake,
            )

            self.assertEqual("failed", manifest["status"])
            source = manifest["instances"][0]["source_files"][0]
            self.assertEqual("failed", source["status"])
            self.assertEqual(1, source["reported_line_count"])
            self.assertEqual(0, source["fetched_line_count"])
            self.assertTrue(source["snapshot_truncated"])
            self.assertIn("truncated inside the first line", source["errors"][0])

    def test_strict_failure_writes_manifest_then_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output"
            with self.assertRaises(collector.CollectionError) as raised:
                collector.collect_logs(
                    workspace="unused",
                    deployment=None,
                    instances=None,
                    start="2026-08-11 02:00:00",
                    end="2026-08-11 02:01:00",
                    output_dir=output,
                    strict=True,
                    local_inputs={"missing": Path(temp_dir) / "missing.log"},
                )

            self.assertEqual("failed", raised.exception.manifest["status"])
            self.assertTrue((output / "collect_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
