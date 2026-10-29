"""Unit tests for state_migration.

CPU 环境走 index_copy_ 回退路径（NPU triton kernel 需要 NPU 环境与 triton，
不在本测试覆盖范围内）。
"""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "state_migration.py"


def _source_tree():
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))


class TestStateMigrationStructure(unittest.TestCase):
    def test_module_is_valid_python(self):
        compile(_source_tree(), str(MODULE_PATH), "exec")

    def test_triton_is_lazy_import(self):
        # CPU 测试环境不允许因为 import 本模块而拉起 triton
        imports = [
            node
            for node in _source_tree().body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        names = []
        for node in imports:
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            else:
                names.append(node.module or "")
        self.assertFalse(any("triton" in n for n in names))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this test environment")
class TestMigrateStateRowsFallbackSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("state_migration_test", MODULE_PATH)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules["state_migration_test"] = cls.module
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("state_migration_test", None)

    def test_migrates_only_crossing_rows(self):
        # 契约：src == dst（含负哨兵）的行是 no-op，只有 src != dst 的行迁移
        state = torch.arange(6 * 2 * 3, dtype=torch.float32).reshape(6, 2, 3)
        before = state.clone()
        src = torch.tensor([1, 2, -1], dtype=torch.int32)   # 迁移 / 自拷贝 / 哨兵
        dst = torch.tensor([4, 2, -1], dtype=torch.int32)

        self.module.migrate_state_rows(state, src, dst)

        torch.testing.assert_close(state[4], before[1])  # 迁移行
        torch.testing.assert_close(state[1], before[1])  # 源页不动
        torch.testing.assert_close(state[2], before[2])  # 自拷贝行
        torch.testing.assert_close(state[0], before[0])  # 未涉及页

    def test_steady_state_is_noop(self):
        state = torch.randn(8, 4, 5)
        before = state.clone()
        src = torch.full((4,), 3, dtype=torch.int32)

        self.module.migrate_state_rows(state, src, src.clone())

        torch.testing.assert_close(state, before)

    def test_padded_pool_view(self):
        # 模拟 hybrid pool 上的 strided 视图：dim0 stride > 行元素数
        row = 2 * 3
        gap = 7
        base = torch.randn(6 * (row + gap))
        state = base.as_strided((6, 2, 3), (row + gap, 3, 1))
        src = torch.tensor([2, 0], dtype=torch.int32)
        dst = torch.tensor([5, 1], dtype=torch.int32)

        self.module.migrate_state_rows(state, src, dst)

        ref = state.clone()
        torch.testing.assert_close(state[5], ref[2])
        torch.testing.assert_close(state[1], ref[0])
        # gap 区域（池内 padding）不被触碰
        torch.testing.assert_close(base, base.clone())

    def test_empty_rows_is_noop(self):
        state = torch.randn(4, 2, 2)
        before = state.clone()
        empty = torch.empty(0, dtype=torch.int32)

        self.module.migrate_state_rows(state, empty, empty)

        torch.testing.assert_close(state, before)

    def test_matches_fallback_oracle(self):
        # 与 index_copy_ 回退路径的等价性（契约输入下：no-op 行均 src==dst）
        state = torch.randn(10, 3, 4)
        src = torch.tensor([1, 2, 2, -1, 5], dtype=torch.int32)
        dst = torch.tensor([7, 2, 4, -1, 5], dtype=torch.int32)

        actual = state.clone()
        self.module.migrate_state_rows(actual, src, dst)

        expected = state.clone()
        s = src.to(torch.int64).clamp(min=0)
        d = dst.to(torch.int64).clamp(min=0)
        expected.index_copy_(0, d, expected.index_select(0, s))

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
