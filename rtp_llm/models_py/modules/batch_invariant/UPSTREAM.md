# Upstream provenance

The persistent GEMM, 2D unified attention, RMSNorm and logprob reduction are adapted from the installed
**vLLM 0.29.0** source, under Apache-2.0. Copyright/SPDX notices are retained.
There is no vLLM runtime dependency and no process-wide aten/BLAS monkey patch.

| Upstream file | SHA256 of the source read from the development environment |
|---|---|
| [v1/attention/ops/triton_attention_helpers.py](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/attention/ops/triton_attention_helpers.py) | `8c730611e7b3c5fb7579ec7846d56a2ab7e348ce06b39136da22072ecc363c95` |
| [v1/attention/ops/triton_unified_attention.py](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/attention/ops/triton_unified_attention.py) | `992a2bc892e2e2b43fbd3c8163816ccf7e97ced56cb542bd827adb0ddb2df9fa` |
| [model_executor/determinism/batch_invariant.py](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/determinism/batch_invariant.py) | `347cdf3deb9825ec6f9123674f98f3010d842d83791db323857da8ba9b1f3fd7` |
| [model_executor/layers/rotary_embedding/base.py](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/layers/rotary_embedding/base.py) | `c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5` |
| [csrc/libtorch_stable/activation_kernels.cu](https://github.com/vllm-project/vllm/blob/v0.29.0/csrc/libtorch_stable/activation_kernels.cu) | `d6510ee0271fb6c41b14aae501da53bcd6d5ec3707288c8644c4d3c120f0f4d0` |
| [csrc/libtorch_stable/layernorm_kernels.cu](https://github.com/vllm-project/vllm/blob/v0.29.0/csrc/libtorch_stable/layernorm_kernels.cu) | `2041b5da8fa2e2868c1ef6133a46d4ae85ffdbc599a8a5da0c5e66cb91c573f6` |

## Adaptation boundaries

* BF16 GEMM with N/K divisible by eight on Hopper/Blackwell follows vLLM's
  cuBLASLt strategy. Exceptional sliced inputs with less than 16-byte alignment
  are copied to aligned storage; ordinary model buffers require no copy. RTP's native
  `batch_invariant_cublaslt_gemm` scopes zero workspace, FP32 compute and
  no-reduction preferences to one operation, and rejects any heuristic with
  split-K > 1, a reduction scheme, or nonzero workspace. It preserves ordinary
  model instances' global Torch/BLAS settings. Plans are cached per host thread,
  device, shape, strides, output dtype and buffer alignment, with at most 256
  entries; weight pointers are never cached.
* The Blackwell M=1 algorithm query follows the GEMV-avoidance workaround in
  [PyTorch v2.11.0 CUDABlas.cpp](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/cuda/CUDABlas.cpp):
  query a two-column RHS, then execute the chosen GEMM algorithm with the actual
  one-column RHS. Without this, an unsplit GEMV may still use a different
  reduction order from batched GEMM. No extra rows are read or written.
* cuBLASLt descriptors use the documented
  [no-reduction scheme](https://docs.nvidia.com/cuda/archive/13.0.2/cublas/index.html#cublasltreduction-scheme-t).
  This configuration still requires qualification on the actual GPU/library
  build; it is not a universal guarantee for every future cuBLAS release.
* The explicit odd-feature-width/FP32/other-architecture fallback `persistent_matmul.py` retains vLLM's persistent SM scheduling, grouped tile
  ordering, strided operands, FP32 accumulator and optional bias. RTP's wrapper
  supports its transposed weight view and FP32 LM-head output. FP32 input dot
  explicitly requests IEEE precision. Launch configurations are selected before
  capture; K partitions must remain fixed for every batch of the same operator.
* `unified_attention.py` retains the causal BF16 2D algorithm: grouped query
  heads, Tensor Core QK and PV, FP32 online softmax, and no split-KV/3D mode.
  TILE_SIZE=32 and BLOCK_M=max(16,next_power_of_2(GQA)) are unchanged between
  prefill, decode and batch sizes. Unlike upstream's phase-dependent large-head
  tuning, this profile also keeps those tiles fixed for head dimension 256.
* The attention adapter supplies RTP HND cache strides and existing device-side
  positions. The adapter computes packed query offsets on device from effective
  query lengths, since RTP's MHA device planner leaves the MLA query-offset
  buffer untouched. Offset/table allocations remain stable on Graph replay;
  capture and replay retain the same metadata planner. Ragged sequence lookup follows upstream's query-block
  mapping. Indexed queries use the same arithmetic with masked query rows.
  Table and KV tail loads are masked; unused lanes for non-power-of-two GQA
  ratios are masked so blocks cannot overlap output writes.
* FP8/INT8 KV, tensor descriptors, sinks, ALiBi, sliding windows, bidirectional
  attention, speculative decoding and TP collectives are not imported or enabled.
  The existing capability checks remain in force.
* `normalization.py` follows the pinned vLLM RMSNorm and log-softmax reduction:
  fixed 1024-element blocks, FP32 accumulation, `1 / sqrt` and sequential
  accumulation of exponentials after the row maximum. The RTP adapter returns
  `(row_max, log_shifted_sum)` so selected-token and prompt scoring retain
  `(logit - row_max) - log_shifted_sum` through the native dispatcher. Combining
  the two normalizer values first can change the final FP32 bits.
* `batch_invariant_norm.cu` retains vLLM's scalar, fixed-block CUB reduction
  and reciprocal square root for norms after residual addition. RTP already
  materializes the same BF16-rounded sum, so the adapter reads that sum without
  repeating the addition. Qwen's shared execution class selects this algorithm
  for post-attention, later-layer input and final norms; its first input norm
  and Q/K norms continue to use the upstream Triton RMSNorm.
* `activation.py` follows the CUDA SwiGLU operation's BF16 rounding boundary:
  SiLU rounds to BF16 before multiplying by the up operand. One Triton launch
  preserves that boundary without materializing the intermediate tensor. The
  source is checksum-verified from the vLLM 0.29.0 PyPI source distribution.
* `rope.py` follows the unscaled base RoPE cache construction. Values round to
  BF16 before promotion to the FP32 storage required by RTP's FlashInfer
  adapter. Cache keys include device, dimension, theta and maximum position;
  ordinary execution's global cache is not mutated. Scaled RoPE is rejected.
* Raw FP32 logprob transport and the immutable per-model policy remain RTP
  integrations. This is not a port of vLLM's entire runtime, sampler or BLAS
  global-state management.

The v1 implementation is retained by Git commit
`08bd92d904da14d074ec08b261735f457058e132` for reproducible before/after comparison.
Batch invariance applies within one software build; changing arithmetic kernels
can change values relative to the old build or to vLLM. Independent numerical
accuracy checks and bitwise checks across batches are both required.
