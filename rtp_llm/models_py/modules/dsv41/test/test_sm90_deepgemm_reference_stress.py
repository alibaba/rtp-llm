"""Strict IEEE-reference diagnostic, separate from native FP8 equivalence.

Hopper FP8 WGMMA can fail the original .015/.02 reference under amplified
cancellation even when DeepGEMM and Humming are bitwise identical. This
manual diagnostic keeps those failures visible; it never relaxes tolerance.
"""

import pytest
from rtp_llm.models_py.modules.dsv41.test.test_sm90_deepgemm_dense import (
    _native_backend,
    _run_actual_shape,
    _run_graph_case,
)


@pytest.mark.parametrize(
    "shape", ["merged_qkv", "cp8_q_b", "cp8_o_b", "shared_w13", "shared_w2"]
)
def test_large_m_strict_reference(shape):
    _run_actual_shape(shape, 512, strict_reference=True)


@pytest.mark.parametrize(
    "rows,n,k,leading", [(15, 320, 640, (3, 5)), (129, 256, 512, (129,))]
)
def test_amplified_changed_input_strict_reference(rows, n, k, leading):
    _run_graph_case(rows, n, k, leading, strict_reference=True)
