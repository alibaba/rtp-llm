import ast
from pathlib import Path


ATTENTION_DIR = (
    Path(__file__).resolve().parents[1] / "models_py/modules/factory/attention"
)
TRT_MODULE = "rtp_llm.models_py.modules.factory.attention.cuda_impl.trt"


def _class_backend_names(tree: ast.Module):
    backend_names = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "NAME"
                    for target in statement.targets
                )
                and isinstance(statement.value, ast.Constant)
            ):
                backend_names[node.name] = statement.value.value
    return backend_names


def test_trt_backend_imports_are_defined():
    """Keep CUDA backend registration aligned with its implementation module."""
    registry_tree = ast.parse((ATTENTION_DIR / "__init__.py").read_text())
    trt_imports = {
        alias.name
        for node in ast.walk(registry_tree)
        if isinstance(node, ast.ImportFrom) and node.module == TRT_MODULE
        for alias in node.names
    }

    implementation_tree = ast.parse((ATTENTION_DIR / "cuda_impl/trt.py").read_text())
    definitions = {
        node.name
        for node in implementation_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert trt_imports <= definitions, (
        "attention registry imports missing trt.py symbols: "
        f"{sorted(trt_imports - definitions)}"
    )


def test_trt_backend_names_match_public_config():
    implementation_tree = ast.parse((ATTENTION_DIR / "cuda_impl/trt.py").read_text())
    backend_names = _class_backend_names(implementation_tree)

    assert backend_names["FlashInferTRTLLMFMHAv2PrefillImpl"] == "trt"
    assert backend_names["FlashInferTRTLLMFMHAv2PagedPrefillImpl"] == "trt_paged"
