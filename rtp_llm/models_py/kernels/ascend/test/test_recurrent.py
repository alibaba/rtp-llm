import ast
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "recurrent.py"


def _tree():
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))


class _FakeTensor:
    def __init__(self, values):
        self.values = values
        self.ndim = 2 if values and isinstance(values[0], list) else 1
        self.shape = (len(values), len(values[0])) if self.ndim == 2 else (len(values),)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class TestRecurrentWithoutRuntimeDependencies(unittest.TestCase):
    def test_module_is_valid_python(self):
        compile(_tree(), str(MODULE_PATH), "exec")

    def test_accelerator_dependencies_are_lazy(self):
        imports = [
            node
            for node in _tree().body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        names = []
        for node in imports:
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            else:
                names.append(node.module or "")
        self.assertFalse(any("triton" in name or "fla_npu" in name for name in names))

    def test_state_page_resolution_tracks_cross_block_tokens(self):
        selected = [
            node
            for node in _tree().body
            if isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            or isinstance(node, ast.FunctionDef)
            and node.name in ("_to_int_list", "_resolve_state_pages")
        ]
        namespace = {}
        module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
        exec(compile(module, str(MODULE_PATH), "exec"), namespace)
        pages = namespace["_resolve_state_pages"](
            _FakeTensor([[10, 11, 12, 13]]),
            _FakeTensor([3]),
            batch=1,
            token_count=3,
            seq_size_per_block=2,
        )
        self.assertEqual(pages, ([10], [[11, 12, 13]]))


if __name__ == "__main__":
    unittest.main()
