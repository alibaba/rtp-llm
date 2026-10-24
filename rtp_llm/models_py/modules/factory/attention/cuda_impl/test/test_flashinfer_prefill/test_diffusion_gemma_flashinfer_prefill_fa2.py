"""fa2-backend variant of the DiffusionGemma flashinfer prefill regression.

Sets DGM_FI_TEST_BACKEND=fa2 before loading the shared suite and runs it in
this separate test process: an illegal-access crash in the auto (fa3 on
sm90) variant cannot kill the fa2 run, so one CI round yields both signals
— whether the crash reproduces, and whether the fa2 kernel template is a
working fallback for the sliding layers.
"""

import os
import unittest

os.environ["DGM_FI_TEST_BACKEND"] = "fa2"

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.test_flashinfer_prefill import (  # noqa: E402
    test_diffusion_gemma_flashinfer_prefill as suite,
)

if __name__ == "__main__":
    unittest.main(module=suite, exit=True)
