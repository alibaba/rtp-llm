# MegaMoE JIT warmup

This warmup contract targets DeepGEMM **83961ec**, without adding backend APIs.
It applies to FP8xFP4 and FP8xFP8, including their shared-expert executors.
Re-audit the C++ launchers and run the coverage check when updating DeepGEMM.

RTP scans every token count from zero through the logical runtime capacity. It
queries the existing backend `get_block_m_for_mega_moe[_fp8]` with the allocated
buffer's aligned capacity; it does not reproduce the non-monotonic tile heuristic.
The deduplication key also retains these token-dependent C++ branches:

- Store tile and epilogue threads: FP8's low-latency and throughput branches differ.
- FP8 single-pass dispatch: `tokens * topk <= 32768`, for both implementations.
- FP4 has fixed epilogue/dispatch threads and no single-pass template argument.

At fixed model shape, symmetric buffer, SM count, launch flags and compiler
environment, these fields determine the remaining configuration: load/SF tiles,
ring sizes, swizzles, pipeline stages, shared memory and dispatch threads.
BLOCK_M alone is **not** the coverage criterion. For EP4/E512/K10, the FP8
representatives include BLOCK_M=192 with single-pass dispatch both enabled and
disabled (first encountered at 1500 and 3277 tokens). FP4 also needs the 192 tile.

The ordinary and gate packers add representatives from their actual tiling
selectors, including ordinary BLOCK_M=2/8 at 2048 tokens and gate BLOCK_M=2/4/8
at 1024/2048. Warmup uses contiguous BF16 activations, FP32 top-k weights, int64
indices, and the actual buffer strides. Both ordinary forward and supported
gate payloads run with the DeepGEMM launch suppressed first. CUDA synchronization
and an EP-group rendezvous finish this packer phase. The second phase performs
real DeepGEMM calls for the representative tokens, with a rendezvous before each
token count and at completion.

The existing DeepGEMM API compiles and launches in one call. Thus this isolates
Triton compilation, but **does not isolate nvcc compilation from collectives**;
a slow rank's first DeepGEMM compilation can still delay collective entry.
A warmup-complete log means these representative calls completed, not that all
possible future strides, dtypes, launch flags or dependency versions are covered.
`MEGA_MOE_JIT_WARMUP_TOKENS` is an explicit partial-coverage override.

## Verification

Unit tests: `utils/test/test_mega_moe_jit_warmup.py`,
`test_mega_moe_fp8_impl.py` and `test_mega_moe_fp8_sm_reserve.py`.

The opt-in audit compiles `utils/test/mega_moe_config_probe.cc` directly against
the unchanged pinned DeepGEMM headers. With a matching package on `PYTHONPATH`,
CUDA headers and a C++20 compiler selected by `CXX`, run from the RTP repository:

```bash
python rtp_llm/models_py/modules/factory/fused_moe/utils/test/verify_mega_moe_jit_coverage.py \
  --deep-gemm-source /path/to/DeepGEMM
```

It checks every token against the real backend BLOCK_M API, verifies each RTP
bucket maps to one **complete C++ heuristic config plus dispatch mode**, and
verifies the representatives cover all configurations. Cases vary EP, expert
count, hidden/intermediate dimensions, and include the few-local-experts branch.
It also checks the single-pass condition in both FP8 C++ launchers. This source
audit complements multi-rank cold-cache runs; it does not replace a new source
review when the pinned version changes.
