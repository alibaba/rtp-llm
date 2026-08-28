import unittest

# Importing the package performs device-specific implementation registration.
import rtp_llm.models_py.modules.factory.attention  # noqa: F401, E402
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MLA_IMPS,
    PREFILL_MLA_IMPS,
)


class SparseMlaRegistryTest(unittest.TestCase):
    def test_base_sparse_mla_registration_is_not_coupled_to_cp(self) -> None:
        prefill_names = {impl.__name__ for impl in PREFILL_MLA_IMPS}
        decode_names = {impl.__name__ for impl in DECODE_MLA_IMPS}

        self.assertIn("SparseMlaImpl", prefill_names)
        self.assertIn("SparseMlaImpl", decode_names)


if __name__ == "__main__":
    unittest.main()
