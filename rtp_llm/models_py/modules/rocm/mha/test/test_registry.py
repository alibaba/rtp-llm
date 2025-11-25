import unittest

# Import the module to trigger registration
import rtp_llm.models_py.modules.rocm.mha
from rtp_llm.models_py.modules.common.mha import DECODE_MHA_IMPS, PREFILL_MHA_IMPS
from rtp_llm.models_py.modules.rocm.mha.aiter import AiterDecodeImpl, AiterPrefillImpl


class TestRocmFMHARegistry(unittest.TestCase):
    def test_aiter_registered(self):
        self.assertIn(AiterPrefillImpl, PREFILL_MHA_IMPS)
        self.assertIn(AiterDecodeImpl, DECODE_MHA_IMPS)


if __name__ == "__main__":
    unittest.main()
