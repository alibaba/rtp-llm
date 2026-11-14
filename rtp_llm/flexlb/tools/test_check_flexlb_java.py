"""Exercise changed-class documentation gates and their Git diff boundary."""

import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import check_flexlb_java as checks


class DocumentationGateTest(unittest.TestCase):
    def test_new_class_requires_documentation(self):
        self.assertTrue(
            checks.documentation_errors(b"class Example { void run() {} }", None)
        )

    def test_simple_carriers_do_not_require_redundant_comments(self):
        for source in [
            b"record Point(int x, int y) {}",
            b"class Point { int x; int y; }",
            b"enum Color { RED, GREEN; }",
        ]:
            self.assertEqual([], checks.documentation_errors(source, None))

    def test_record_with_validation_or_behavior_needs_contract(self):
        for source in [
            b"record Point(int x) { Point { if (x < 0) throw new IllegalArgumentException(); } }",
            b"record Point(int x) { int doubled() { return x * 2; } }",
            b"class Owner { Object handle = acquire(); }",
        ]:
            self.assertTrue(checks.documentation_errors(source, None))

    def test_one_sentence_multiline_summary_is_sufficient(self):
        source = b"/**\n * Holds one request result.\n */\n@Data\nclass Example {}"
        self.assertEqual([], checks.documentation_errors(source, None))

    def test_unrelated_nested_class_is_not_forced_into_the_change(self):
        source = b"/**\n * Runs work.\n */\nclass Outer {\n void run() {}\n class Untouched { void work() {} }\n}"
        self.assertEqual([], checks.documentation_errors(source, [(5, 5)]))
        self.assertTrue(checks.documentation_errors(source, [(6, 6)]))

    def test_tags_only_or_single_line_javadoc_is_rejected(self):
        for source in [
            b"/** Purpose. */\nclass Example {}",
            b"/**\n * @since 1.0\n */\nclass Example {}",
        ]:
            self.assertTrue(checks.documentation_errors(source, None))

    def test_comment_only_change_checks_the_documented_class(self):
        source = b"/** Summary. */\nclass Example {}"
        self.assertTrue(checks.documentation_errors(source, [(1, 1)]))

    def test_clean_historical_class_does_not_require_backfill(self):
        self.assertEqual([], checks.documentation_errors(b"class Existing {}", []))

    def test_staged_method_edit_requires_existing_class_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            path = root / "rtp_llm/flexlb/Example.java"
            path.parent.mkdir(parents=True)
            path.write_text("class Example {\n    void run() {}\n}\n")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                cwd=root,
                check=True,
            )
            self.assertEqual([], checks.changed_lines(path, root))
            path.write_text(
                "class Example {\n    void run() { System.nanoTime(); }\n}\n"
            )
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            previous = Path.cwd()
            try:
                os.chdir(root)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(1, checks.main([str(path)]))
                path.write_text(
                    "/**\n * Runs one work item.\n */\nclass Example {\n    void run() { System.nanoTime(); }\n}\n"
                )
                subprocess.run(["git", "add", "."], cwd=root, check=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(0, checks.main([str(path)]))
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
