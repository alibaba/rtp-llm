"""PPU-first shared compute for V4.1: kernels only, no private NVIDIA imports.

Modules under this package implement the ops declared in
``rtp_llm.models_py.modules.dsv41.backend`` for the
``bf16_dense_w4a16_moe_sparse_mla_host_enp_v1`` strategy. Kernel sources adapted
from sglang-for-sail (commit 512b03d, Apache-2.0) keep a per-file attribution
note. Heavy accelerator libraries (triton, deep_gemm, deep_ep, flash_mla) are
imported lazily inside functions so the package imports on a CPU host.

The backend binds the inherited V4.1 target and draft models to PPU
operators, framework weights and native cache adapters.
"""
