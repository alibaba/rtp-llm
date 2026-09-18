import ast
import unittest
from pathlib import Path
from typing import Iterable, Set

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    _tokenizer_factory,
    ensure_tokenizer_registered,
)
from rtp_llm.model_factory_register import (
    ModelDict,
    _model_factory,
    _model_type_to_module,
    ensure_model_registered,
)
from rtp_llm.openai.renderer_factory_register import (
    _renderer_factory,
    _renderer_type_to_module,
    ensure_renderer_registered,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _iter_existing_python_files(paths: Iterable[Path]):
    for path in paths:
        if path.exists():
            yield from path.rglob("*.py")


def _find_literal_registration_names(
    paths: Iterable[Path], function_name: str
) -> Set[str]:
    names: Set[str] = set()
    for path in _iter_existing_python_files(paths):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", None) != function_name:
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            value = node.args[0].value
            if isinstance(value, str):
                names.add(value)
    return names


class LazyRegistryTest(unittest.TestCase):
    def test_lazy_model_registry_covers_source_registrations(self):
        source_names = _find_literal_registration_names(
            (REPO_ROOT / "rtp_llm" / "models",),
            "register_model",
        )
        registered_names = {
            name
            for name, module_path in _model_type_to_module.items()
            if module_path.startswith("rtp_llm.models.")
        }

        self.assertEqual(source_names, registered_names)

    def test_lazy_renderer_registry_covers_source_registrations(self):
        source_names = _find_literal_registration_names(
            (REPO_ROOT / "rtp_llm" / "openai" / "renderers",),
            "register_renderer",
        )
        registered_names = {
            name
            for name, module_path in _renderer_type_to_module.items()
            if module_path.startswith("rtp_llm.openai.renderers.")
        }

        self.assertEqual(source_names, registered_names)

    def test_qwen4_exp_lazy_renderer_registration(self):
        self.assertTrue(ensure_renderer_registered("qwen4_exp"))
        renderer_type = _renderer_factory["qwen4_exp"]
        self.assertEqual(renderer_type.__name__, "Qwen35Renderer")
        self.assertEqual(
            renderer_type.__module__, "rtp_llm.openai.renderers.qwen35_renderer"
        )

    def test_qwen4_exp_mtp_lazy_registration(self):
        self.assertEqual(
            _model_type_to_module["qwen4_exp_mtp"],
            "rtp_llm.models.qwen4_exp.qwen4_exp_mtp",
        )
        self.assertTrue(ensure_renderer_registered("qwen4_exp_mtp"))
        self.assertEqual(_renderer_factory["qwen4_exp_mtp"].__name__, "Qwen35Renderer")
        self.assertTrue(ensure_model_registered("qwen4_exp_mtp"))
        self.assertEqual(_model_factory["qwen4_exp_mtp"].__name__, "Qwen4ExpMTP")
        self.assertTrue(ensure_tokenizer_registered("qwen4_exp_mtp"))
        self.assertEqual(
            _tokenizer_factory["qwen4_exp_mtp"].__name__, "QWenV2Tokenizer"
        )

    def test_dsv4_architecture_inference_is_available_without_model_import(self):
        self.assertEqual(
            "deepseek_v4",
            ModelDict.get_ft_model_type_by_config(
                {"architectures": ["DeepseekV4ForCausalLM"]}
            ),
        )
        self.assertEqual(
            "deepseek_v4_mtp",
            ModelDict.get_ft_model_type_by_config(
                {"architectures": ["DeepseekV4ForCausalLMNextN"]}
            ),
        )


if __name__ == "__main__":
    unittest.main()
