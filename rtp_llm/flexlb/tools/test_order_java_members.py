"""Regression cases for source ordering and preservation of Java semantics."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import order_java_members as formatter


class MemberOrderTest(unittest.TestCase):
    def ordered(self, source):
        result = formatter.reorder(source.encode())
        self.assertEqual(result, formatter.reorder(result))
        return result.decode()

    def test_visibility_and_call_order(self):
        result = self.ordered(
            """class Example {
    private void helper() {}
    private void wrapper() { helper(); }
    void packageMethod() {}
    protected void inherited() {}
    public void entry() { helper(); }
    private void unused() {}
}"""
        )
        self.assertLess(result.index("public void"), result.index("protected void"))
        self.assertLess(
            result.index("protected void"), result.index("void packageMethod")
        )
        self.assertLess(
            result.index("void packageMethod"), result.index("private void")
        )
        self.assertLess(result.index("void wrapper"), result.index("void helper"))

    def test_initializer_and_lombok_field_order(self):
        source = """@RequiredArgsConstructor class Example {
    final Object first;
    public void entry() {}
    static Object a = create();
    { touch(); }
    static { touch(); }
    final Object second;
    Object value = create();
    static Object b = create();
}"""
        result = self.ordered(source)
        for sequence in [
            ["static Object a", "static {", "static Object b"],
            ["Object first", "{ touch(); }", "Object second", "Object value"],
        ]:
            positions = [result.index(text) for text in sequence]
            # The unqualified block is the instance initializer; exclude the static block.
            if sequence[0] == "Object first":
                positions[1] = result.index("\n    { touch(); }")
            self.assertEqual(positions, sorted(positions))

    def test_comments_follow_their_methods(self):
        result = self.ordered(
            """class Example {
    /** private documentation */
    private void helper() {} // trailing helper comment
    /** public documentation */
    public void entry() {}
}"""
        )
        self.assertIn(
            "/** private documentation */\n    private void helper() {} // trailing helper comment",
            result,
        )
        self.assertIn("/** public documentation */\n    public void entry()", result)

    def test_enum_record_and_nested_class(self):
        result = self.ordered(
            """enum Example {
    A, B;
    private void helper() {}
    public void entry() {}
    record Value(int id) {
        private void helper() {}
        public int value() { return id; }
    }
}"""
        )
        self.assertIn("A, B;", result)
        self.assertIn("record Value(int id)", result)
        self.assertLess(
            result.index("public int value"), result.rindex("private void helper")
        )

    def test_overloads_and_recursive_calls_converge(self):
        self.ordered(
            """class Example {
    public void first(int n) { second(); }
    public void second() { first(1); }
    public void first() { first(0); }
}"""
        )

    def test_parse_errors_do_not_produce_output(self):
        with self.assertRaises(ValueError):
            formatter.reorder(b"class Broken { void method( }")

    def test_check_is_read_only_and_apply_validates_whole_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "A.java"
            original = b"class A { private void helper() {} public void entry() {} }"
            good.write_bytes(original)
            bad = Path(directory) / "Z.java"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(1, formatter.main(["--check", str(good)]))
                self.assertEqual(original, good.read_bytes())
                bad.write_bytes(b"class Z {")
                self.assertEqual(2, formatter.main(["--apply", directory]))
                self.assertEqual(original, good.read_bytes())
                bad.unlink()
                self.assertEqual(0, formatter.main(["--apply", directory]))
                self.assertEqual(0, formatter.main(["--check", directory]))


if __name__ == "__main__":
    unittest.main()
