# DSv4 mHC Optimization Notes

## Scope

This note tracks the mHC-only optimization pass. TRTLLM-Gen and MoE work are
intentionally out of scope.

## Reference Baseline

Before implementation, `~/oss-references` was inspected for vLLM, SGLang,
TileKernels, DeepGEMM, and FlashInfer mHC references. The implementation uses
these relevant open-source shapes:

- vLLM `model_executor/layers/mhc.py`: `compute_num_split`,
  `tf32_hc_prenorm_gemm`, `mhc_pre_big_fuse_tilelang`, and
  `hc_head_fuse_tilelang`.
- DeepGEMM: optional `tf32_hc_prenorm_gemm` helper for the mHC pre GEMM and
  split-K sqrsum.
- TileKernels vendored in RTP: existing `_mhc_pre_big_fuse` remains the
  Sinkhorn/residual-mixing stage.

## Implemented Choices

- `hc_head` now has a fused TileLang fast path controlled by
  `DSV4_MHC_HEAD_FUSED`. When enabled, fused head is required and raises if it
  cannot run; set `DSV4_MHC_HEAD_FUSED=0` to use the older TileLang
  composition.
- `mhc_pre_big_fuse` now selects its pre-GEMM backend via
  `DSV4_MHC_PRE_GEMM_BACKEND`.
  - Default/`auto`: try DeepGEMM split-K, then fall back to TileLang single-K
    if DeepGEMM is unavailable.
  - `deepgemm`: require DeepGEMM split-K and raise on failure.
  - `tilelang_single`: use the previous single-split TileLang path.
  - `tilelang_splitk`: reserved until the corresponding TileLang split-K kernel
    is wired in this vendored TileKernels snapshot.

## Rationale

Full mHC fusion is not implemented. The pre GEMM is a large K-dimensional
reduction that benefits from split-K CTA parallelism, while Sinkhorn and
residual apply have different parallel shapes. Keeping the pre GEMM separate
lets DeepGEMM handle small-batch utilization while `_mhc_pre_big_fuse` still
removes the intermediate normalization, Sinkhorn, and residual-mixing launches.
