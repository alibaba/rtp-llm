"""Regression tests for changed-line formatting and JUnit assertion imports."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


TOOLS = Path(__file__).resolve().parent
PROFILE = TOOLS.parent.parent / "eclipse-java-formatter.xml"
JUNIT = "org.junit.jupiter.api.Assertions"


class FormatChangedLinesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        target = TOOLS / "target"
        target.mkdir(exist_ok=True)
        classpath_file = target / "classpath.txt"
        if not classpath_file.exists():
            subprocess.run(
                [str(TOOLS.parent.parent / "mvnw"), "-q", "-f", str(TOOLS / "pom.xml"),
                 "dependency:build-classpath", f"-Dmdep.outputFile={classpath_file}"],
                cwd=TOOLS.parent.parent, check=True, timeout=120,
            )
        cls.workspace = tempfile.TemporaryDirectory(
            prefix="formatter-tests-", dir=target
        )
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.root = Path(cls.workspace.name)
        classes = cls.root / "classes"
        classes.mkdir()
        dependencies = classpath_file.read_text().strip()
        compilation = subprocess.run(
            [
                "javac", "-cp", dependencies, "-d", str(classes),
                *map(str, sorted(TOOLS.glob("*.java"))),
            ],
            capture_output=True, text=True, timeout=120,
        )
        if compilation.returncode:
            raise AssertionError(compilation.stdout + compilation.stderr)
        cls.classpath = os.pathsep.join((str(classes), dependencies))

    def fixture(self, source: str, source_set: str = "test") -> Path:
        path = self.root / self.id() / "src" / source_set / "java/Example.java"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
        return path

    def format_file(self, path: Path, *selected: str) -> str:
        source = path.read_text(encoding="utf-8")
        ranges = "all"
        if selected:
            lines = source.splitlines()
            numbers = []
            for marker in selected:
                matches = [i for i, line in enumerate(lines, 1) if marker in line]
                self.assertEqual(len(matches), 1, f"Ambiguous marker: {marker}")
                numbers.append(matches[0])
            ranges = ",".join(f"{number}-{number}" for number in numbers)
        result = subprocess.run(
            [
                "java", "-cp", self.classpath, "FormatChangedLines",
                str(PROFILE), f"{path}:{ranges}",
            ],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return path.read_text(encoding="utf-8")

    def static_imports(self, source: str) -> list[str]:
        return [line.strip() for line in source.splitlines()
                if line.strip().startswith("import static ")]

    def assert_compiles(self, path: Path) -> None:
        assertions = path.parent / "org/junit/jupiter/api/Assertions.java"
        assertions.parent.mkdir(parents=True, exist_ok=True)
        assertions.write_text(
            "package org.junit.jupiter.api; public class Assertions { "
            "public static void assertTrue(boolean value) {} }\n"
        )
        result = subprocess.run(
            ["javac", "-d", str(path.parent / "classes"), str(assertions), str(path)],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_fully_qualified_calls_add_sorted_imports_without_initial_imports(self) -> None:
        path = self.fixture("""
            package example;

            class Example {
                void check() {
                    org.junit.jupiter.api.Assertions.assertTrue(true);
                    org.junit.jupiter.api.Assertions.assertEquals(1, 1);
                }
            }
        """)
        output = self.format_file(path)
        self.assertEqual(self.static_imports(output), [
            f"import static {JUNIT}.assertEquals;",
            f"import static {JUNIT}.assertTrue;",
        ])
        self.assertIn("assertTrue(true);", output)
        self.assertIn("assertEquals(1, 1);", output)
        self.assertNotIn(f"{JUNIT}.assertTrue(", output)
        self.assertNotIn(f"{JUNIT}.assertEquals(", output)

    def test_short_receiver_removes_unused_ordinary_import(self) -> None:
        path = self.fixture("""
            import org.junit.jupiter.api.Assertions;

            class Example {
                void check() {
                    Assertions.assertEquals(1, 1);
                }
            }
        """)
        output = self.format_file(path)
        self.assertIn(f"import static {JUNIT}.assertEquals;", output)
        self.assertNotIn(f"import {JUNIT};", output)
        self.assertNotIn("Assertions.assertEquals(", output)

    def test_existing_explicit_and_wildcard_imports_are_reused(self) -> None:
        for imported in ("assertTrue", "*"):
            with self.subTest(imported=imported):
                path = self.fixture(f"""
                    import static {JUNIT}.{imported};

                    class Example {{
                        void check() {{
                            {JUNIT}.assertTrue(true);
                            {JUNIT}.assertTrue(false);
                        }}
                    }}
                """)
                output = self.format_file(path)
                self.assertEqual(self.static_imports(output), [
                    f"import static {JUNIT}.{imported};"
                ])
                self.assertNotIn(f"{JUNIT}.assertTrue(", output)

    def test_ranges_survive_import_insertion_and_preserve_unchanged_calls(self) -> None:
        path = self.fixture("""
            import org.junit.jupiter.api.Assertions;

            class Example {
                void check() {
                    Assertions.assertTrue( true );
                    Assertions.assertEquals( 2,2 );
                    int   untouched=7;
                    Assertions.assertEquals( 3,3 );
                }
            }
        """)
        original = path.read_text()
        output = self.format_file(path, "assertTrue", "3,3")
        self.assertIn(f"import {JUNIT};", output)
        self.assertEqual(self.static_imports(output), [
            f"import static {JUNIT}.assertEquals;",
            f"import static {JUNIT}.assertTrue;",
        ])
        self.assertIn("assertTrue(true);", output)
        self.assertIn("assertEquals(3, 3);", output)
        self.assertNotIn("Assertions.assertTrue(", output)
        for line in original.splitlines(keepends=True):
            if "2,2" in line or "untouched" in line:
                self.assertIn(line, output)

    def test_class_literals_and_method_references_keep_ordinary_import(self) -> None:
        for use in ("Class<?> type = Assertions.class;",
                    "java.util.function.Consumer<Boolean> check = Assertions::assertTrue;"):
            with self.subTest(use=use):
                path = self.fixture(f"""
                    import {JUNIT};

                    class Example {{
                        {use}
                        void check() {{
                            Assertions.assertTrue(true);
                        }}
                    }}
                """)
                output = self.format_file(path)
                self.assertIn(f"import {JUNIT};", output)
                self.assertIn(f"import static {JUNIT}.assertTrue;", output)
                self.assertIn(use, output)
                self.assertNotIn("Assertions.assertTrue(", output)

    def test_literals_and_comments_are_not_rewritten(self) -> None:
        path = self.fixture("""
            class Example {
                void check() {
                    // Assertions.assertTrue and org.junit.jupiter.api.Assertions.assertEquals
                    String text = "org.junit.jupiter.api.Assertions.assertTrue(true)";
                    String shortText = "Assertions.assertEquals(1, 1)";
                    org.junit.jupiter.api.Assertions.assertTrue(true);
                }
            }
        """)
        original = path.read_text()
        output = self.format_file(path)
        for line in original.splitlines(keepends=True):
            if "//" in line or "String " in line:
                self.assertIn(line, output)
        self.assertIn(f"import static {JUNIT}.assertTrue;", output)
        self.assertNotIn(f"{JUNIT}.assertTrue(true);", output)

    def test_competing_imports_local_methods_and_inheritance_keep_qualifier(self) -> None:
        cases = (
            ("import static other.Checks.assertTrue;", "", ""),
            ("import static other.Checks.*;", "", ""),
            ("", "", "void assertTrue(boolean value) {}"),
            ("", "extends BaseTest", ""),
            ("", "implements Checks", ""),
        )
        for imports, inheritance, member in cases:
            with self.subTest(imports=imports, inheritance=inheritance, member=member):
                path = self.fixture(f"""
                    {imports}
                    class Example {inheritance} {{
                        {member}
                        void check() {{
                            {JUNIT}.assertTrue(true);
                        }}
                    }}
                """)
                output = self.format_file(path)
                self.assertIn(f"{JUNIT}.assertTrue(true);", output)
                self.assertNotIn(f"import static {JUNIT}.assertTrue;", output)

    def test_shadowed_short_receiver_keeps_qualifier(self) -> None:
        cases = (
            ("", "Object Assertions", ""),
            ("Object Assertions;", "", ""),
            ("", "", "Object Assertions = null;"),
            ("static class Assertions {}", "", ""),
        )
        for member, parameter, local in cases:
            with self.subTest(member=member, parameter=parameter, local=local):
                path = self.fixture(f"""
                    import {JUNIT};

                    class Example {{
                        {member}
                        void check({parameter}) {{
                            {local}
                            Assertions.assertTrue(true);
                        }}
                    }}
                """)
                output = self.format_file(path)
                self.assertIn("Assertions.assertTrue(true);", output)
                self.assertNotIn(f"import static {JUNIT}.assertTrue;", output)

    def test_other_assertions_import_overrides_junit_package_wildcard(self) -> None:
        path = self.fixture("""
            import org.junit.jupiter.api.*;
            import other.Assertions;

            class Example {
                void check() {
                    Assertions.assertTrue(true);
                }
            }
        """)
        output = self.format_file(path)
        self.assertIn("Assertions.assertTrue(true);", output)
        self.assertEqual(self.static_imports(output), [])

    def test_static_imported_receiver_fields_keep_qualifier(self) -> None:
        for receiver in ("Assertions", "org"):
            with self.subTest(receiver=receiver):
                call = "Assertions.assertTrue(true)" if receiver == "Assertions" else f"{JUNIT}.assertTrue(true)"
                path = self.fixture(f"""
                    import {JUNIT};
                    import static other.Checks.{receiver};

                    class Example {{
                        void check() {{
                            {call};
                        }}
                    }}
                """)
                output = self.format_file(path)
                self.assertIn(call, output)
                self.assertNotIn(f"import static {JUNIT}.assertTrue;", output)

    def test_trailing_multiline_comments_leave_generated_imports_as_code(self) -> None:
        comment = "/* explanation\n * continued */"
        for declaration in ("package example;", f"import {JUNIT};"):
            with self.subTest(declaration=declaration):
                path = self.fixture(
                    f"{declaration} {comment}\n\n"
                    f"class Example {{ void check() {{ {JUNIT}.assertTrue(true); }} }}\n"
                )
                output = self.format_file(path, "void check")
                self.assertIn(comment, output)
                self.assertNotIn(f"{JUNIT}.assertTrue(", output)
                self.assert_compiles(path)

    def test_one_line_package_and_import_keep_imports_before_class(self) -> None:
        for header in ("package example;", f"package example; import {JUNIT};"):
            with self.subTest(header=header):
                path = self.fixture(
                    f"{header} class Example {{ void check() {{ {JUNIT}.assertTrue(true); }} }}\n"
                )
                output = self.format_file(path)
                self.assertLess(output.index("package example;"), output.index("import static"))
                self.assertLess(output.index("import static"), output.index("class Example"))
                self.assertNotIn(f"{JUNIT}.assertTrue(", output)
                self.assert_compiles(path)

    def test_main_sources_only_receive_formatting(self) -> None:
        path = self.fixture("""
            import org.junit.jupiter.api.Assertions;

            class Example {
                void check() {
                    Assertions.assertTrue( true );
                    org.junit.jupiter.api.Assertions.assertEquals( 1,1 );
                }
            }
        """, source_set="main")
        output = self.format_file(path)
        self.assertIn(f"import {JUNIT};", output)
        self.assertIn("Assertions.assertTrue(true);", output)
        self.assertIn(f"{JUNIT}.assertEquals(1, 1);", output)
        self.assertEqual(self.static_imports(output), [])

    def test_javadoc_and_declaration_formatting_preserve_untouched_body(self) -> None:
        path = self.fixture("""
            class Example {
                /** Checks values. */
                void check( int first,
                     int second ) {
                    int   untouched=7;
                    org.junit.jupiter.api.Assertions.assertEquals( first,second );
                }
            }
        """)
        output = self.format_file(path, "Checks values", "int second", "assertEquals")
        self.assertIn("    /**\n     * Checks values.\n     */", output)
        self.assertIn("void check(int first, int second)", output)
        self.assertIn("        int   untouched=7;\n", output)
        self.assertIn("assertEquals(first, second);", output)
        self.assertNotIn(f"{JUNIT}.assertEquals(", output)

    def test_all_is_idempotent(self) -> None:
        path = self.fixture("""
            import org.junit.jupiter.api.Assertions;

            class Example {
                /** Checks values. */
                void check() {
                    Assertions.assertTrue( true );
                    org.junit.jupiter.api.Assertions.assertEquals( 1,1 );
                }
            }
        """)
        first = self.format_file(path)
        self.assertEqual(self.static_imports(first), [
            f"import static {JUNIT}.assertEquals;",
            f"import static {JUNIT}.assertTrue;",
        ])
        self.assertEqual(self.format_file(path), first)

    def test_pre_commit_formats_and_restages_changed_assertions(self) -> None:
        repo = self.root / self.id() / "repo"
        flexlb = repo / "rtp_llm/flexlb"
        tools = flexlb / "tools/format-changed-lines"
        (tools / "target").mkdir(parents=True)
        for name in ("FormatChangedLines.java", "TestAssertionImports.java",
                     "format-changed-lines.sh", "git-pre-commit.sh"):
            shutil.copy2(TOOLS / name, tools / name)
        shutil.copy2(TOOLS / "target/classpath.txt", tools / "target/classpath.txt")
        shutil.copy2(PROFILE, flexlb / PROFILE.name)
        path = flexlb / "module/src/test/java/Example.java"
        path.parent.mkdir(parents=True)
        path.write_text("class Example {\n    void check() {}\n}\n")

        def git(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                 "-c", "user.name=Formatter Test", "-c", "user.email=formatter@example.invalid", *args],
                cwd=repo, capture_output=True, text=True, check=True, timeout=30,
            )

        git("init", "-q")
        git("add", str(path))
        git("commit", "-qm", "fixture")
        path.write_text(
            "class Example {\n    void check() {\n"
            f"        {JUNIT}.assertTrue( true );\n    }}\n}}\n"
        )
        git("add", str(path))
        result = subprocess.run(
            ["bash", str(tools / "git-pre-commit.sh")], cwd=repo,
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"+import static {JUNIT}.assertTrue;", git("diff", "--cached").stdout)
        self.assertIn("assertTrue(true);", path.read_text())
        git("diff", "--exit-code", "--", str(path))


if __name__ == "__main__":
    unittest.main()
