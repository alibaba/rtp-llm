import importlib.metadata
import importlib.util
import unittest
from pathlib import Path


class XGrammarLoaderTest(unittest.TestCase):
    def test_distribution_record_locates_native_library(self):
        distribution = importlib.metadata.distribution("xgrammar")
        record = distribution.read_text("RECORD")
        self.assertIsNotNone(
            record, "xgrammar RECORD must be included in Bazel runfiles"
        )
        libraries = [
            entry
            for entry in distribution.files
            if entry.name
            in (
                "libxgrammar_bindings.so",
                "libxgrammar_bindings.dylib",
                "xgrammar_bindings.dll",
            )
        ]
        self.assertEqual(len(libraries), 1)
        library = Path(distribution.locate_file(libraries[0]))
        self.assertTrue(library.is_file(), str(library))
        package_dir = Path(importlib.util.find_spec("xgrammar").origin).parent
        self.assertEqual(library.resolve(), (package_dir / library.name).resolve())

    def test_native_binding_loads_and_parses_grammar(self):
        # Exercise the real tvm_ffi loader and native API without a model or GPU.
        import xgrammar

        grammar = xgrammar.Grammar.from_ebnf('root ::= "hello"')
        self.assertIn('"hello"', str(grammar))


if __name__ == "__main__":
    unittest.main()
