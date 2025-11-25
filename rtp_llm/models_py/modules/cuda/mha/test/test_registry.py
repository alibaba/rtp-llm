import unittest

# Import the module to trigger registration
import rtp_llm.models_py.modules.cuda.mha
from rtp_llm.models_py.modules.common.mha import DECODE_MHA_IMPS, PREFILL_MHA_IMPS
from rtp_llm.models_py.modules.cuda.mha.flash_infer import (
    FlashInferDecodeImpl,
    FlashInferPrefillImpl,
)
from rtp_llm.models_py.modules.cuda.mha.trt import TRTMHAImpl
from rtp_llm.models_py.modules.cuda.mha.xqa import XQAImpl


class TestCudaFMHARegistry(unittest.TestCase):

    def test_flashinfer_registered(self):
        self.assertIn(FlashInferPrefillImpl, PREFILL_MHA_IMPS)

        self.assertIn(FlashInferDecodeImpl, DECODE_MHA_IMPS)

    def test_trt_registered(self):
        self.assertIn(TRTMHAImpl, PREFILL_MHA_IMPS)

    def test_xqa_registered(self):
        self.assertIn(XQAImpl, DECODE_MHA_IMPS)


if __name__ == "__main__":
    unittest.main()
