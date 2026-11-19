# SM120 grouped GEMM provenance

- DeepGEMM operation and launcher from `522942a9cd6b4aaa6b6e7000eae12983f17302ec`, with source paths and hashes in `EXTRACTION_SOURCES.json`. RTP adapts the host binding, runtime selection and packaging. The contiguous and masked 1D1D launchers use the same extracted device implementation.
- DeepJIT headers from `2efdab421e1cfb17fe8bc20e11ca72aa6d0e6c43`. Its license is in `third-party/deep_jit/LICENSE`.
- CUTLASS/CuTe headers from the declared `deep_gemm==2.8.0+122e18b.cu132` package. The compiled subset matches CUTLASS `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` (BSD-3-Clause). RTP does not redistribute those headers in its own wheel.
- The C++20 host binding uses the existing image GCC 12 and binutils. The action compares its compiler, implicit headers and link inputs with its declared toolchain identity before executing. The normal Bazel compiler and NVCC host settings are independent.
- The extracted host format calls use Torch 2.11's header-only fmt 12.1.0 (MIT with the compiled-in exception). CUDA hexadecimal literals retain upstream spelling for both signs of zero. The binding links the Torch C++11 ABI and CUDA 13.
- `RTP_DG120_ENABLE_LIBDW=0` omits elfutils headers and libdw symbolization. Exceptions retain the upstream conditions and message text, with backtrace module names and offsets from the system loader. The hidden symbols can be resolved offline from the unstripped extension.
- Cold device compilation uses CUDA 13.2 NVCC and its configured GCC host compiler, with `compute_120f` / `sm_120f`. The standard CUDA 13 image already supplies this runtime JIT chain.
