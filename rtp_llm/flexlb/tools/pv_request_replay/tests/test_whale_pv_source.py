from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))

import whale_pv_source as source  # noqa: E402

MODULE_SPEC = importlib.util.spec_from_file_location(
    "pv_request_replay_generate_whale", TOOL_DIR / "generate_replay_whale.py"
)
assert MODULE_SPEC and MODULE_SPEC.loader
WHALE_CLI = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(WHALE_CLI)


POD = "dash-pd-test.master-part-c8fa517c-a-affa"
NAMESPACE = "hippo-c2-infer-spectrum"
CLUSTER = "asi_bj_ai_infra_egs_01"
LOG_DIR = "/home/admin/ai-whale/logs"


def line_at(stamp: str, request_id: str) -> str:
    body = json.dumps({"requestId": request_id})
    return f"{stamp} [worker] INFO pvLogger - {body}\n"


def log_line(second: int, request_id: str, pad: int = 0) -> str:
    body = json.dumps({"requestId": request_id}) + ("x" * pad)
    return f"2026-09-19 20:00:{second:02d}.000 [worker] INFO pvLogger - {body}\n"


def describe_payload(pod_name: str = POD, role: str = "master_part") -> dict[str, Any]:
    return {
        "carbon_status": {
            "hippo_id": CLUSTER,
            "carbon_status": {
                "roles": {
                    role: {
                        "nodes": [
                            {
                                "cur_worker_node_status": {
                                    "ip": "10.68.129.10",
                                    "health_info": {"health_status": "HT_ALIVE"},
                                    "service_info": {"status": "SVT_AVAILABLE"},
                                    "labels": {
                                        "app#dot#c2#dot#io/pod-name": pod_name,
                                        "namespace": NAMESPACE,
                                        "sigma#dot#ali/sn": "sn-1",
                                    },
                                }
                            }
                        ]
                    },
                    "kvcm_part": {"nodes": []},
                }
            },
        }
    }


def list_payload() -> dict[str, Any]:
    return {
        "data": {
            "deployment_status": [
                {"deployment_id": "dep-1", "deployment_name": "beijing_a"},
                {"deployment_id": "dep-2", "deployment_name": "beijing_b"},
            ]
        }
    }


AWK_BOUNDS = re.compile(r"-v s='([^']+)' -v e='([^']+)'")
SED_RANGE = re.compile(r'\| sed -n "(\d+),(\d+)p"')


class FakePlatform:
    """Emulate whale JSON plus the read-only asicli ``sh -c`` scripts."""

    def __init__(
        self,
        files: dict[str, list[str]],
        deployment: dict[str, Any] | None = None,
        corrupt_chunk: int | None = None,
        truncate: bool = False,
    ) -> None:
        self.files = files
        self.deployment = deployment if deployment is not None else describe_payload()
        self.corrupt_chunk = corrupt_chunk
        self.truncate = truncate
        self.commands: list[list[str]] = []
        self.chunk_reads = 0

    def __call__(self, command: Sequence[str]) -> str:
        argv = list(command)
        self.commands.append(argv)
        if argv[0] == "whale":
            if "list" in argv:
                return json.dumps(list_payload())
            return json.dumps(self.deployment)
        return self._script(argv[-1])

    def _path_in(self, script: str) -> str:
        # Matched with its quotes so that pv.log does not also match the rotated
        # pv.log.<date>.<n>.log path that has it as a prefix.
        for path in self.files:
            if f"'{path}'" in script:
                return path
        raise AssertionError(f"no known log path in script: {script}")

    def _selected(self, script: str) -> list[str]:
        bounds = AWK_BOUNDS.search(script)
        assert bounds, script
        start, end = bounds.group(1), bounds.group(2)
        lines = self.files[self._path_in(script)]
        return [line for line in lines if start <= line[:19] <= end]

    def _capped(self, output: str) -> str:
        encoded = output.encode("utf-8")
        if self.truncate and len(encoded) >= source.EXEC_STDOUT_LIMIT:
            return encoded[: source.EXEC_STDOUT_LIMIT].decode("utf-8", "ignore") + "\n"
        return output

    def _script(self, script: str) -> str:
        if script.startswith("ls -1"):
            names = [Path(path).name for path in self.files]
            return "\n".join(sorted(names + ["application.log"])) + "\n"
        if script.startswith("head -n 1"):
            path = script.split("'", 2)[1]
            lines = self.files.get(path, [])
            if not lines:
                return ""
            return lines[0] + (lines[-1] if len(lines) > 1 else "")
        if script.endswith("| wc -l"):
            return f"{len(self._selected(script))}\n"
        if script.startswith("a=1; i=1; while"):
            return self._digests(script)
        match = SED_RANGE.search(script)
        if match:
            return self._chunk(script, int(match.group(1)), int(match.group(2)))
        raise AssertionError(f"unexpected remote script: {script}")

    def _digests(self, script: str) -> str:
        page = int(re.search(r"b=\$\(\(a\+(\d+)\)\)", script).group(1)) + 1
        total = len(self._selected(script))
        out = []
        index = 1
        first = 1
        while first <= total:
            last = min(first + page - 1, total)
            payload = self._chunk_body(script, first, last)
            out.append(f"{index} {hashlib.md5(payload.encode()).hexdigest()}")
            index += 1
            first = last + 1
        return "\n".join(out) + "\n" if out else ""

    def _chunk_body(self, script: str, first: int, last: int) -> str:
        return "".join(self._selected(script)[first - 1 : last])

    def _chunk(self, script: str, first: int, last: int) -> str:
        self.chunk_reads += 1
        body = self._chunk_body(script, first, last)
        if self.corrupt_chunk == self.chunk_reads:
            body = body[:-2] + "\n"
        return self._capped(body)


def fetch(files: dict[str, list[str]], page_lines: int = 2, **kwargs: Any) -> list[str]:
    fake = FakePlatform(files, **kwargs)
    pod = source.WhalePod(
        role="master_part",
        pod_name=POD,
        namespace=NAMESPACE,
        cluster=CLUSTER,
        ip="10.68.129.10",
        health_status="HT_ALIVE",
        service_status="SVT_AVAILABLE",
    )
    lines, _ = source.fetch_pod_window(
        fake,
        pod,
        "load-balancer",
        LOG_DIR,
        "pv.log",
        datetime.fromisoformat("2026-09-19T19:59:00+08:00"),
        datetime.fromisoformat("2026-09-19T20:01:00+08:00"),
        page_lines,
    )
    return lines


def request_ids(lines: list[str]) -> list[str]:
    return [json.loads(line.split(" - ", 1)[1])["requestId"] for line in lines]


class WindowFilterTest(unittest.TestCase):
    def test_both_bounds_use_their_own_v_flag(self) -> None:
        script = source._window_filter(
            datetime(2026, 9, 19, 19, 50), datetime(2026, 9, 19, 20, 31), "/x/pv.log"
        )
        self.assertIn("-v s='2026-09-19 19:50:00'", script)
        self.assertIn("-v e='2026-09-19 20:31:00'", script)
        self.assertIn("substr($0,1,19)>=s && substr($0,1,19)<=e", script)

    def test_window_is_inclusive_on_both_ends(self) -> None:
        files = {
            f"{LOG_DIR}/pv.log": [
                line_at("2026-09-19 19:58:59.000", "early"),
                line_at("2026-09-19 19:59:00.000", "start"),
                line_at("2026-09-19 20:00:30.000", "middle"),
                line_at("2026-09-19 20:01:00.000", "end"),
                line_at("2026-09-19 20:01:01.000", "late"),
            ]
        }
        self.assertEqual(
            ["start", "middle", "end"], request_ids(fetch(files, page_lines=2))
        )


class ChunkedReadTest(unittest.TestCase):
    def test_chunks_reassemble_in_order(self) -> None:
        files = {f"{LOG_DIR}/pv.log": [log_line(i % 60, f"r-{i}") for i in range(7)]}
        lines = fetch(files, page_lines=2)
        self.assertEqual(7, len(lines))
        self.assertEqual([f"r-{i}" for i in range(7)], request_ids(lines))

    def test_corrupt_chunk_is_rejected(self) -> None:
        files = {f"{LOG_DIR}/pv.log": [log_line(i % 60, f"r-{i}") for i in range(6)]}
        with self.assertRaisesRegex(source.WhaleSourceError, "failed md5 verification"):
            fetch(files, page_lines=2, corrupt_chunk=2)

    def test_stdout_cap_is_reported_not_trusted(self) -> None:
        pad = source.EXEC_STDOUT_LIMIT // 2
        files = {
            f"{LOG_DIR}/pv.log": [log_line(i % 60, f"r-{i}", pad=pad) for i in range(4)]
        }
        with self.assertRaisesRegex(source.WhaleSourceError, "reduce --page-lines"):
            fetch(files, page_lines=3, truncate=True)

    def test_rotated_and_empty_files_are_skipped_by_window(self) -> None:
        files = {
            f"{LOG_DIR}/pv.log": [log_line(30, "current")],
            f"{LOG_DIR}/pv.log.2026-09-18.0.log": [],
            f"{LOG_DIR}/pv.log.2026-09-19.0.log": [
                "2026-09-19 09:00:00.000 [worker] INFO pvLogger - {}\n"
            ],
        }
        lines = fetch(files, page_lines=5)
        self.assertEqual(1, len(lines))
        self.assertIn("current", lines[0])


class RoleResolutionTest(unittest.TestCase):
    def test_deployment_is_matched_by_name(self) -> None:
        fake = FakePlatform({})
        self.assertEqual(
            "dep-2", source.resolve_deployment_id(fake, "dash_pd", "beijing_b")
        )

    def test_unknown_deployment_lists_alternatives(self) -> None:
        with self.assertRaisesRegex(
            source.WhaleSourceError, "known: beijing_a, beijing_b"
        ):
            source.resolve_deployment_id(FakePlatform({}), "dash_pd", "nope")

    def test_role_suffix_pattern_matches_numbered_parts(self) -> None:
        payload = describe_payload(pod_name="p0", role="prefill_part0")
        pods = source.resolve_role_pods(
            lambda command: json.dumps(payload), "dep-1", "prefill"
        )
        self.assertEqual(["p0"], [pod.pod_name for pod in pods])
        self.assertEqual(CLUSTER, pods[0].cluster)

    def test_pod_name_label_is_resolved_by_suffix(self) -> None:
        payload = describe_payload()
        labels = payload["carbon_status"]["carbon_status"]["roles"]["master_part"][
            "nodes"
        ][0]["cur_worker_node_status"]["labels"]
        labels["app#dot#c2#dot.io/pod-name"] = labels.pop("app#dot#c2#dot#io/pod-name")
        pods = source.resolve_role_pods(
            lambda command: json.dumps(payload), "dep-1", "master"
        )
        self.assertEqual([POD], [pod.pod_name for pod in pods])
        self.assertEqual(NAMESPACE, pods[0].namespace)

    def test_missing_cluster_is_reported(self) -> None:
        payload = describe_payload()
        payload["carbon_status"]["hippo_id"] = None
        with self.assertRaisesRegex(source.WhaleSourceError, "hippo_id"):
            source.resolve_role_pods(
                lambda command: json.dumps(payload), "dep-1", "master"
            )

    def test_unknown_role_lists_known_roles(self) -> None:
        payload = describe_payload()
        with self.assertRaisesRegex(
            source.WhaleSourceError, "known roles: kvcm_part, master_part"
        ):
            source.resolve_role_pods(
                lambda command: json.dumps(payload), "dep-1", "router"
            )

    def test_describe_cooldown_is_surfaced(self) -> None:
        def runner(command: Sequence[str]) -> str:
            return "请等待 7 秒后重试\n"

        with self.assertRaisesRegex(source.WhaleSourceError, "rate-limited"):
            source.resolve_role_pods(runner, "dep-1", "master")


class OutputPathTest(unittest.TestCase):
    def test_output_dir_defaults_under_tool_outputs(self) -> None:
        path = WHALE_CLI.default_output_dir(
            "dash_pd-master",
            datetime.fromisoformat("2026-09-19T19:55:00+08:00"),
            datetime.fromisoformat("2026-09-19T20:21:00+08:00"),
        )
        self.assertEqual(
            TOOL_DIR / "outputs" / "dash_pd-master-20260919-195500_202100", path
        )

    def test_snapshot_is_staged_for_local_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pod = source.WhalePod(
                role="master_part",
                pod_name=POD,
                namespace=NAMESPACE,
                cluster=CLUSTER,
                ip=None,
                health_status=None,
                service_status=None,
            )
            path = source.write_pod_snapshot(root, pod, ["a\n", "b\n"])
            self.assertEqual(root / "whale_raw" / POD / "pv.log", path)
            self.assertEqual("a\nb\n", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
