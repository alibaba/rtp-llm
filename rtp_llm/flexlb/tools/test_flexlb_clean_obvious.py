"""Regression tests for the read-only unused Java source audit."""

import importlib.util
import sys
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "flexlb_clean_obvious", Path(__file__).with_name("flexlb-clean-obvious.py")
)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


class UnusedJavaTest(unittest.TestCase):
    def findings(self, main, test="", other=""):
        sources = {"a/src/main/java/A.java": main.encode()}
        if test:
            sources["a/src/test/java/ATest.java"] = test.encode()
        if other:
            sources["b/src/main/java/B.java"] = other.encode()
        return audit.audit(sources)

    def test_unused_import_and_field(self):
        findings = self.findings(
            "import java.util.List; class A { private int unused; }"
        )
        self.assertEqual(
            {f.rule for f in findings}, {"unused-import", "unused-private-field"}
        )

    def test_import_used_by_annotation_and_javadoc(self):
        self.assertEqual(
            [],
            self.findings(
                "import x.Bean; import java.util.List; /** {@link List} */ @Bean class A {}"
            ),
        )

    def test_test_reference_does_not_keep_production_method(self):
        findings = self.findings(
            "class A { void helper() {} }",
            "class ATest { void run() { new A().helper(); } }",
        )
        self.assertEqual([f.rule for f in findings], ["test-only-production-method"])

    def test_cross_module_reference_keeps_method(self):
        self.assertEqual(
            [],
            self.findings(
                "class A { void helper() {} }",
                other="class B { @Override public void run() { new A().helper(); } }",
            ),
        )

    def test_method_reference_keeps_method(self):
        self.assertEqual(
            [], self.findings("class A { void helper() {} Runnable r = this::helper; }")
        )

    def test_comments_are_not_calls(self):
        self.assertEqual(
            [
                f.rule
                for f in self.findings("class A { /* helper() */ void helper() {} }")
            ],
            ["unused-production-method"],
        )

    def test_recursion_does_not_make_a_dead_method_live(self):
        self.assertEqual(
            [f.rule for f in self.findings("class A { void helper() { helper(); } }")],
            ["unused-production-method"],
        )

    def test_same_named_delegation_is_a_real_call(self):
        self.assertEqual(
            [],
            self.findings(
                "class A { void run() {} }",
                other="class B { @Override public void run() { a.run(); } }",
            ),
        )

    def test_callbacks_and_main_are_not_dead(self):
        self.assertEqual(
            [],
            self.findings(
                """class A {
            @Override public void run() {}
            @org.springframework.context.annotation.Bean Object bean() { return null; }
            @JsonCreator static A from(String value) { return new A(); }
            public static void main(String[] args) {}
        }"""
            ),
        )

    def test_reflection_and_lombok_fields_are_retained(self):
        self.assertEqual(
            [],
            self.findings(
                'class A { private int state; @Override public void run() { reflect("state"); } }'
            ),
        )
        self.assertEqual([], self.findings("@Getter class A { private int state; }"))

    def test_test_fixture_counts_as_test(self):
        findings = audit.audit(
            {
                "a/src/main/java/A.java": b"class A { void helper() {} }",
                "a/src/testFixtures/java/Fixture.java": b"class Fixture { void run() { new A().helper(); } }",
            }
        )
        self.assertEqual([f.rule for f in findings], ["test-only-production-method"])

    def test_same_name_requires_semantic_review(self):
        # The parser does not resolve receiver types; ambiguous names cannot justify deletion.
        self.assertEqual(
            [],
            self.findings(
                "class A { void helper() {} }",
                other="class B { void helper() {} @Override public void run() { helper(); } }",
            ),
        )

    def test_unused_overloads_are_all_reported(self):
        self.assertEqual(
            2, len(self.findings("class A { void helper() {} void helper(int n) {} }"))
        )

    def test_invalid_syntax_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "invalid Java syntax"):
            self.findings("class A {")

    def test_audit_does_not_mutate_inputs(self):
        sources = {"a/src/main/java/A.java": b"class A { void unused() {} }"}
        before = sources.copy()
        audit.audit(sources)
        self.assertEqual(before, sources)


if __name__ == "__main__":
    unittest.main()
