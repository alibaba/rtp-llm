import importlib.util
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.utils import module_util

SPEC = importlib.util.spec_from_file_location(
    "deepgemm_recipe_wrapper",
    Path(__file__).resolve().parents[1] / "deepgemm_wrapper.py",
)
wrapper = importlib.util.module_from_spec(SPEC)
with mock.patch.object(module_util, "has_module", return_value=False):
    SPEC.loader.exec_module(wrapper)


class DeepGemmRecipeTest(unittest.TestCase):
    def test_default_preserves_the_legacy_call(self):
        with mock.patch.object(wrapper, "_fp8_gemm_nt_impl") as implementation:
            wrapper.fp8_gemm_nt("a", "b", "output")
        implementation.assert_called_once_with(
            "a", "b", "output", None, compiled_dims="nk", disable_ue8m0_cast=True
        )

    def test_explicit_raw_and_packed_recipes_reach_deepgemm(self):
        for recipe in ((1, 32, 32), (1, 1, 32), (1, 128, 128), (1, 1, 128)):
            with self.subTest(recipe=recipe), mock.patch.object(
                wrapper, "_fp8_gemm_nt_impl"
            ) as implementation:
                wrapper.fp8_gemm_nt(
                    "a", "b", "output", disable_ue8m0_cast=False, recipe=recipe
                )
            implementation.assert_called_once_with(
                "a",
                "b",
                "output",
                None,
                compiled_dims="nk",
                disable_ue8m0_cast=False,
                recipe=recipe,
            )


if __name__ == "__main__":
    unittest.main()
