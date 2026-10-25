"""基础设施与现场适配不能反向依赖业务、步骤调度或展示。"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"


@pytest.mark.parametrize("package", ["runtime", "adapters"])
def test_infrastructure_has_no_orchestration_or_presentation_dependency(package):
    forbidden = {"scenario", "cases", "workload", "reporting"}
    violations = []
    for path in (ROOT / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                if name.split(".")[0] in forbidden:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "\n".join(violations)
