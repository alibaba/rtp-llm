import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Load only the pure policy module: importing the modules package eagerly
# imports CUDA-dependent attention implementations, which this test does not use.
_spec = importlib.util.spec_from_file_location(
    "glm53_fp8_capacity_policy", Path(__file__).with_name("fp8_capacity.py")
)
assert _spec is not None and _spec.loader is not None
_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_policy)
glm53_fp8_capacity = _policy.glm53_fp8_capacity


class Glm53FP8CapacityTest(unittest.TestCase):
    def test_unknown_and_unsafe_backends_keep_legacy_limit(self):
        backends = [
            SimpleNamespace(),
            SimpleNamespace(get_block_m_candidates_for_mega_moe_fp8=lambda: [192]),
        ]
        backends.extend(
            SimpleNamespace(
                get_block_m_candidates_for_mega_moe_fp8=lambda values=values: values,
                _C=SimpleNamespace(
                    get_token_alignment_for_mega_moe_fp8=lambda alignment=alignment: alignment
                ),
            )
            for values, alignment in (
                ([16, 192, 224], 384),
                ([], 384),
                ([0, 192], 384),
                ([-192, 192], 384),
                ([192], 0),
                ([192], -384),
            )
        )
        for backend in backends:
            with patch.dict(sys.modules, {"deep_gemm": backend}):
                self.assertEqual(glm53_fp8_capacity(131072, 288, 8, 8), 2880)
                self.assertEqual(glm53_fp8_capacity(48, 288, 8, 8), 48)

    def test_ring_aligned_backend_preserves_configured_budget(self):
        backend = SimpleNamespace(
            get_block_m_candidates_for_mega_moe_fp8=lambda: [
                8,
                16,
                32,
                64,
                96,
                128,
                192,
            ],
            _C=SimpleNamespace(get_token_alignment_for_mega_moe_fp8=lambda: 384),
        )
        with patch.dict(sys.modules, {"deep_gemm": backend}):
            for tokens in (1, 48, 2880, 2883, 16384, 65536, 131072):
                self.assertEqual(glm53_fp8_capacity(tokens, 288, 8, 8), tokens)


if __name__ == "__main__":
    unittest.main()
