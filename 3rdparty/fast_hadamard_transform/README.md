# Fast Hadamard Transform (SM120)

Provides the block-128 Hadamard transform used by the SM120 W4A16 FFN backend.

## Build

- `repositories.bzl` pins upstream v1.0.4.post1 at commit
  `4ea722e434e3d4f2a14522341959ebdbe62be2de`, with SHA256 verification and
  fallback download URLs.
- `fast_hadamard_transform.BUILD` compiles the C++/CUDA sources for SM120 using
  RTP's `torch_deps()` and CUDA toolchain.
- `embedded_binding.patch` disables the standalone Python module registration
  through `RTP_LLM_EMBEDDED_FHT`; the computation code remains unchanged.

## Integration

FHT is statically linked into `librtp_compute_ops.so` and exposed through
`rtp_llm_ops.w4a16_sm120_hadamard`. The interface accepts BF16 or FP32 tensors
of shape `[rows, 128]` and a normalization scale.

The RTP wheel includes the implementation and the static upstream BSD-3-Clause
`LICENSE` and `AUTHORS` files under `.dist-info/licenses/fast_hadamard_transform/`.
Keep these files in sync when updating the upstream version. No separate FHT
package or runtime compilation is required for this backend.
