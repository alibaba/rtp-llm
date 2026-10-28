import pytest

# Guard the directory on whether the CUDA compute_ops .so loads. The tests here
# don't import any specific compute_ops symbol directly — they exercise the
# Python-side `FusedMoe` class — but they transitively pull in modules that need
# `librtp_compute_ops.so` (e.g. via `rtp_llm.models_py.modules`). On envs where
# libcuda.so.1 is missing or compute_ops failed to build, importing tests would
# fail at collection with a hard error.
#
# Previous version of this guard checked `from rtp_llm.ops.compute_ops import
# FusedMoEOp`, but that symbol doesn't exist anywhere in the codebase. The guard
# therefore always tripped, silently skipping the entire directory — including
# 3 tests (fused_moe_test, moe_ep_reorder_test, triton_fused_executor_test)
# that should run on H20 workers.
try:
    from rtp_llm.ops.compute_ops import rtp_llm_ops  # noqa: F401
except ImportError as e:
    pytest.skip(
        f"rtp_llm compute_ops unavailable (likely missing libcuda or unbuilt .so): {e}",
        allow_module_level=True,
    )
