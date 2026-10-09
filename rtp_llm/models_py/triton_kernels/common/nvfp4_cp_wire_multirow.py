"""Compatibility imports for the consolidated CP wire implementation."""
from rtp_llm.models_py.triton_kernels.common.nvfp4_cp_wire import (
    WIRE_BYTES,
    _scatter_wire_multirow,
    pack_cp_nvfp4_wire,
    scatter_cp_nvfp4_wire,
)
