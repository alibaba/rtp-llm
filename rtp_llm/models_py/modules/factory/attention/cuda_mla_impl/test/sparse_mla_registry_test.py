import unittest
from types import SimpleNamespace

# Importing the package performs device-specific implementation registration.
import rtp_llm.models_py.modules.factory.attention  # noqa: F401, E402
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MLA_IMPS,
    PREFILL_MLA_IMPS,
    _supports_sparse_prefill_dense_fast_path,
)
from rtp_llm.ops import KvCacheDataType


class SparseMlaRegistryTest(unittest.TestCase):
    def test_base_sparse_mla_registration_is_not_coupled_to_cp(self) -> None:
        prefill_names = {impl.__name__ for impl in PREFILL_MLA_IMPS}
        decode_names = {impl.__name__ for impl in DECODE_MLA_IMPS}

        self.assertIn("SparseMlaImpl", prefill_names)
        self.assertIn("SparseMlaImpl", decode_names)

    def test_fp8_nope_cache_stays_on_sparse_prefill(self) -> None:
        glm53_fp8 = SimpleNamespace(
            kv_cache_dtype=KvCacheDataType.FP8,
            rope_head_dim=0,
        )
        ds_fp8 = SimpleNamespace(
            kv_cache_dtype=KvCacheDataType.FP8,
            rope_head_dim=64,
        )
        glm53_bf16 = SimpleNamespace(
            kv_cache_dtype=KvCacheDataType.BASE,
            rope_head_dim=0,
        )

        self.assertFalse(_supports_sparse_prefill_dense_fast_path(glm53_fp8))
        self.assertTrue(_supports_sparse_prefill_dense_fast_path(ds_fp8))
        self.assertTrue(_supports_sparse_prefill_dense_fast_path(glm53_bf16))


if __name__ == "__main__":
    unittest.main()
