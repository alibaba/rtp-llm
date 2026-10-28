import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.smoke.artifact_path import smoke_actual_path


class SmokeActualArtifactPathTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root / "paired/github-opensource"
        self.smoke = self.workspace / "rtp_llm/test/smoke"
        self.smoke.mkdir(parents=True)
        self.output = self.root / "uploaded"
        self.output.mkdir()

    def path(self, source, index=1, output=None):
        return smoke_actual_path(
            str(source), index, str(output or self.output), str(self.smoke)
        )

    def test_absolute_oss_task_stays_under_upload_root_and_preserves_input(self):
        task = self.smoke / "data/model/qwen3/generation.json"
        task.parent.mkdir(parents=True)
        golden = b'{"query_result": [{"result": "original golden"}]}\n'
        task.write_bytes(golden)
        task_identity = str(task)
        actual = {"response": "actual response", "aux_info": {"iter_count": 44}}
        output = self.path(task_identity)
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps(actual))
        self.assertEqual(
            output,
            self.output
            / "smoke_actual/rtp_llm/test/smoke/data/model/qwen3/generation.query_1.json",
        )
        self.assertEqual(json.loads(output.read_text()), actual)
        self.assertEqual(task.read_bytes(), golden)
        self.assertEqual(str(task), task_identity)
        self.assertFalse(task.with_suffix(".query_1.json").exists())

    def test_internal_sibling_and_workspace_link_share_stable_internal_identity(self):
        internal = self.workspace.parent / "internal_source"
        task = internal / "rtp_llm/test/smoke/data/model/qwen3/generation.json"
        task.parent.mkdir(parents=True)
        task.write_text("{}")
        (self.workspace / "internal_source").symlink_to(
            internal, target_is_directory=True
        )
        linked = self.workspace / "internal_source" / task.relative_to(internal)
        expected = self.output / (
            "smoke_actual/internal_source/rtp_llm/test/smoke/"
            "data/model/qwen3/generation.query_1.json"
        )
        for source in (task, linked):
            with self.subTest(source=str(source)):
                self.assertEqual(self.path(source), expected)

    def test_nested_internal_tree_is_supported(self):
        task = self.workspace / "internal_source/rtp_llm/test/smoke/data/a.json"
        task.parent.mkdir(parents=True)
        task.write_text("{}")
        self.assertEqual(
            self.path(task, 0),
            self.output
            / "smoke_actual/internal_source/rtp_llm/test/smoke/data/a.query_0.json",
        )

    def test_relative_legacy_layout_is_preserved(self):
        for source in (
            "data/model/a.json",
            "rtp_llm/test/smoke/data/model/a.json",
            "internal_source/rtp_llm/test/smoke/data/model/a.json",
            "rtp_llm/test/smoke/data/模型/多轮.json",
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    self.path(source, 0),
                    self.output
                    / "smoke_actual"
                    / Path(source).with_suffix(".query_0.json"),
                )

    def test_equal_basenames_and_oss_internal_tasks_do_not_collide(self):
        internal = self.workspace.parent / "internal_source/rtp_llm/test/smoke"
        internal.mkdir(parents=True)
        sources = (
            self.smoke / "data/one/task.json",
            self.smoke / "data/two/task.json",
            internal / "data/one/task.json",
        )
        outputs = [self.path(source) for source in sources]
        self.assertEqual(len(set(outputs)), len(sources))
        for index, output in enumerate(outputs):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(str(index))
        self.assertEqual([output.read_text() for output in outputs], ["0", "1", "2"])

    def test_relative_parent_traversal_and_empty_identity_are_rejected(self):
        for source in (
            "",
            ".",
            "../task.json",
            "data/../../task.json",
            self.smoke / "../task.json",
        ):
            with self.subTest(source=str(source)):
                with self.assertRaises(ValueError):
                    self.path(source)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_unknown_absolute_roots_and_source_symlink_escape_are_rejected(self):
        outside = self.root / "outside.json"
        outside.write_text("unchanged")
        linked = self.smoke / "linked.json"
        linked.symlink_to(outside)
        for source in (outside, linked, self.root / "rtp_llm/test/smoke/fake.json"):
            with self.subTest(source=str(source)):
                with self.assertRaises(ValueError):
                    self.path(source)
        self.assertEqual(outside.read_text(), "unchanged")
        self.assertEqual(list(self.output.iterdir()), [])

    def test_invalid_query_indices_cannot_inject_output_path(self):
        for index in (-1, True, None, "../../escape", "1/x", 1.0):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    self.path("data/task.json", index)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_artifact_root_directory_and_file_symlinks_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "task.query_1.json"
        sentinel.write_text("unchanged")
        for part in (
            "smoke_actual",
            "smoke_actual/data",
            "smoke_actual/data/task.query_1.json",
        ):
            with self.subTest(part=part):
                with tempfile.TemporaryDirectory(dir=self.root) as output_dir:
                    output = Path(output_dir)
                    link = output / part
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(sentinel if part.endswith(".json") else outside)
                    with self.assertRaises(ValueError):
                        self.path("data/task.json", output=output)
        self.assertEqual(sentinel.read_text(), "unchanged")
        self.assertEqual(list(outside.iterdir()), [sentinel])

    def test_in_root_output_alias_is_rejected_to_prevent_identity_collision(self):
        data = self.output / "smoke_actual/data"
        data.mkdir(parents=True)
        (self.output / "smoke_actual/alias").symlink_to(data, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.path("alias/task.json")
        self.assertEqual(list(data.iterdir()), [])

    def test_explicit_output_root_symlink_uses_its_real_directory(self):
        linked_output = self.root / "output-link"
        linked_output.symlink_to(self.output, target_is_directory=True)
        self.assertEqual(
            self.path("data/task.json", output=linked_output),
            self.output / "smoke_actual/data/task.query_1.json",
        )


if __name__ == "__main__":
    unittest.main()
