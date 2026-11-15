import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "collect_bazel_testlogs", Path(__file__).with_name("collect_bazel_testlogs.py")
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectBazelTestlogsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.link = self.workspace / "bazel-testlogs"
        self.link.symlink_to(self.cache, target_is_directory=True)
        self.events = self.root / "events.json"

    def output(self, name, text="<testsuite/>"):
        path = self.cache / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def result(self, path, status="PASSED", shard=0):
        return {
            "id": {"testResult": {"label": "//test:current", "shard": shard}},
            "testResult": {
                "status": status,
                "testActionOutput": [{"name": path.name, "uri": path.as_uri()}],
            },
        }

    def publish(self, results, code=0, finished=True):
        if finished:
            results = [*results, {"finished": {"exitCode": {"code": code}}}]
        self.events.write_text("\n".join(map(json.dumps, results)) + "\n")
        return collector.publish(self.workspace, self.events, code)

    def test_excludes_stale_failure_without_deleting_cached_reports(self):
        stale = self.output("stale/test.xml", "<testsuite><error/></testsuite>")
        current = self.output("current/test.xml")
        # A valid cached result remains valid regardless of its timestamp.
        os.utime(current, (1, 1))
        result = self.publish([self.result(current)])
        self.assertEqual(
            [p.relative_to(result) for p in result.rglob("test.xml")],
            [Path("current/test.xml")],
        )
        self.assertTrue(stale.exists())
        self.assertEqual(self.link.resolve(), result)

    def test_preserves_current_failure_and_bazel_exit_code(self):
        error = (
            "<testsuite><testcase><error>current failure</error></testcase></testsuite>"
        )
        current = self.output("failed/test.xml", error)
        result = self.publish([self.result(current, "FAILED")], code=3)
        self.assertEqual((result / "failed/test.xml").read_text(), error)
        manifest = json.loads((result / "invocation-results.json").read_text())
        self.assertEqual(manifest["bazel_exit_code"], 3)
        self.assertEqual(manifest["tests"][0]["status"], "FAILED")

    def test_accepts_protobuf_json_omitting_zero_exit_code(self):
        current = self.output("current/test.xml")
        result = self.publish(
            [
                self.result(current),
                {"finished": {"exitCode": {"name": "SUCCESS"}}},
            ],
            finished=False,
        )
        self.assertTrue((result / "current/test.xml").is_file())

    def test_keeps_each_shard_and_declared_artifacts(self):
        shards = [self.output(f"current/shard_{i}/test.xml") for i in range(2)]
        archive = self.output("current/shard_0/test.outputs/outputs.zip", "artifact")
        events = [self.result(path, shard=i) for i, path in enumerate(shards)]
        events[0]["testResult"]["testActionOutput"].append(
            {"name": "test.outputs__outputs.zip", "uri": archive.as_uri()}
        )
        result = self.publish(events)
        self.assertEqual(len(list(result.rglob("test.xml"))), 2)
        self.assertEqual(
            (result / archive.relative_to(self.cache)).read_text(), "artifact"
        )

    def test_nested_junit_requires_its_own_bep_declaration(self):
        current = self.output("current/test.xml")
        payload = self.output("current/test.outputs/payload.txt", "diagnostic")
        nested = self.output("current/test.outputs/test.xml", "old failure")
        event = self.result(current)
        event["testResult"]["testActionOutput"].append(
            {"name": "test.outputs", "uri": payload.parent.as_uri()}
        )
        result = self.publish([event])
        self.assertEqual(
            (result / payload.relative_to(self.cache)).read_text(), "diagnostic"
        )
        self.assertFalse((result / nested.relative_to(self.cache)).exists())
        self.assertTrue(nested.exists())

    def test_rejects_incomplete_or_empty_success_without_replacing_link(self):
        current = self.output("current/test.xml")
        for events, finished in [([], True), ([self.result(current)], False)]:
            with self.subTest(events=events, finished=finished), self.assertRaises(
                ValueError
            ):
                self.publish(events, finished=finished)
            self.assertEqual(self.link.resolve(), self.cache)

    def test_rejects_missing_or_external_declared_output(self):
        missing = self.cache / "missing/test.xml"
        outside = self.root / "outside.xml"
        outside.write_text("outside")
        for path in [missing, outside]:
            with self.subTest(path=path), self.assertRaises((OSError, ValueError)):
                self.publish([self.result(path)])
            self.assertEqual(self.link.resolve(), self.cache)

    def test_does_not_replace_real_testlogs_directory(self):
        self.link.unlink()
        self.link.mkdir()
        marker = self.link / "keep"
        marker.write_text("owned elsewhere")
        with self.assertRaises(ValueError):
            self.publish([])
        self.assertEqual(marker.read_text(), "owned elsewhere")

    def test_keeps_all_invocations_in_the_same_job(self):
        reports = self.root / "job-outputs" / "bazel-testlogs"
        self.output("stale/test.xml", "old failure")
        for name in ["first", "second"]:
            if self.link.resolve() != self.cache:
                # Bazel restores its output-tree symlink on each invocation.
                self.link.unlink()
                self.link.symlink_to(self.cache, target_is_directory=True)
            current = self.output(f"{name}/test.xml")
            self.events.write_text(
                "\n".join(
                    map(
                        json.dumps,
                        [self.result(current), {"finished": {"exitCode": {}}}],
                    )
                )
                + "\n"
            )
            collector.publish(self.workspace, self.events, 0, reports)
        self.assertEqual(self.link.resolve(), reports)
        self.assertEqual(len(list(reports.rglob("invocation-results.json"))), 2)
        self.assertEqual(
            sorted(path.parent.name for path in reports.rglob("test.xml")),
            ["first", "second"],
        )


if __name__ == "__main__":
    unittest.main()
