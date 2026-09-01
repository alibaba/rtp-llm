import ast
import os
import re
import unittest

_PYTHON_ATTRIBUTES = "rtp_llm/telemetry/attributes.py"
_CPP_ATTRIBUTES = "rtp_llm/cpp/telemetry/TraceAttributes.h"


def _source_path(relative_path):
    candidates = []
    test_srcdir = os.environ.get("TEST_SRCDIR")
    test_workspace = os.environ.get("TEST_WORKSPACE")
    if test_srcdir and test_workspace:
        candidates.append(os.path.join(test_srcdir, test_workspace, relative_path))
    candidates.append(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", relative_path)
        )
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(relative_path)


def _python_attribute_values(path):
    with open(path, encoding="utf-8") as source_file:
        tree = ast.parse(source_file.read(), filename=path)
    values = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not isinstance(node.targets[0], ast.Name):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            values.add(node.value.value)
    return values


def _cpp_attribute_values(path):
    with open(path, encoding="utf-8") as source_file:
        source = source_file.read()
    return set(re.findall(r"\bkAttr[A-Za-z0-9_]+\s*=\s*\"([^\"]+)\"", source))


class TraceAttributesParityTest(unittest.TestCase):
    def test_cpp_keys_are_registered_in_python_schema(self):
        python_keys = _python_attribute_values(_source_path(_PYTHON_ATTRIBUTES))
        cpp_keys = _cpp_attribute_values(_source_path(_CPP_ATTRIBUTES))

        self.assertTrue(cpp_keys)
        self.assertTrue(
            cpp_keys.issubset(python_keys),
            "C++ trace keys missing from Python schema: "
            + ", ".join(sorted(cpp_keys - python_keys)),
        )


if __name__ == "__main__":
    unittest.main()
