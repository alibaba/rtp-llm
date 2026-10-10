# Kimi K3 dependency selection

K3 deployment package versions and private artifact URLs are maintained by the
enclosing internal RTP-LLM repository on its matching `feat/merge_k3_dev` branch,
under `internal_source/deps/kimi_k3`. Its pinned Git submodule commit defines the
paired source version.

The internal profile supplies the CUDA 13 Bazel lock and SM10x wheel requirements
through `--override_repository=rtp_deps=...` and
`--override_repository=arch_config=...`. Use its checked-in preparation/build
entry point so Bazel dependencies and wheel installation metadata agree.

K3 requires DeepGEMM with `fp8_gemm_rs_nt` and `GemmRSBuffer`, TokenSpeed paged MLA,
cuLA and patched FLA, plus the independent `k3_native_deep_gemm` extension. The
extension's upstream revisions and patches remain in
`rtp_llm/models_py/modules/kimi_k3/native_deep_gemm/UPSTREAM.json`.
The deployment profile also records the isolated FlashInfer/CUTLASS runtimes and
the required ABI-compatible vLLM stable operator library.

The generic CUDA 13 dependency set remains the main baseline. It does not install
the private K3 linear-attention wheels or claim to be a complete K3 environment.
Source support for K3 does not imply that its deployment artifacts are installed.
