"""fa2-backend variant of the DiffusionGemma 4c-attribution window A/B.

Sets DGM_WAB_BACKEND=fa2 before loading the shared suite and runs it in
this separate test process (the same crash-isolation convention as
test_diffusion_gemma_flashinfer_prefill_fa2): an illegal-access crash of
one backend cannot kill the other's rows. The fa2 numbers feed the
fa2/fa3/blockwise comparison table of the KERNEL_PATH_EFFICIENCY branch
of the 4c attribution contract.
"""

import os
import unittest

os.environ["DGM_WAB_BACKEND"] = "fa2"

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.test_flashinfer_prefill import (  # noqa: E402
    test_diffusion_gemm_window_ab_prefill as suite,
)

if __name__ == "__main__":
    unittest.main(module=suite, exit=True)
