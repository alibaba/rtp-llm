"""CPU contract tests: unavailable GPU backend constructors are fail-fast sentinels.
No torch.cuda availability spoofing; no CUDA kernels or model initialization.
"""

import unittest
from rtp_llm.ops import compute_ops


class UnavailableGpuBackend:
    def __init__(self, *args, **kwargs):
        raise AssertionError("CPU contract test unexpectedly constructed a GPU backend")


for name in (
    "FusedRopeKVCacheDecodeOp",
    "FusedRopeKVCachePrefillOpQKVOut",
    "FusedRopeKVCachePrefillOpQOut",
):
    if not hasattr(compute_ops, name):
        setattr(compute_ops, name, UnavailableGpuBackend)
suite = unittest.defaultTestLoader.loadTestsFromNames(
    [
        "rtp_llm.test.kimi_k3_mla_workspace_config_test",
        "rtp_llm.models_py.modules.hybrid.test.kimi_k3_fp8_weight_test",
        "rtp_llm.models_py.modules.hybrid.test.kimi_k3_mtp_contract_test",
        "rtp_llm.models_py.modules.hybrid.test.kimi_k3_mtp_chunk_prefill_unit_test",
    ]
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
