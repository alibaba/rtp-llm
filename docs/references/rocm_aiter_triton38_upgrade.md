# ROCm AITER / Triton 3.8 upgrade

## Pinned stack

| Package | Version |
| --- | --- |
| AITER | 0.1.22.dev59+gc6677e075.d20260909 |
| Triton | 3.8.0+amd.rocm7.2.0.git111ff227 |
| triton-kernels | 1.0.0+amd.rocm7.2.0.git111ff227 |
| FlyDSL | 0.3.2 |

The ROCm requirements, lockfile, archive SHA256 and wheel packaging dependencies
are updated together. The official AITER wheel is used, not a locally repaired
PA wheel. NVIDIA dependency pins are unchanged.

## Compatibility changes

- Replace removed block-pointer operations in RTP FLA and Kimi KDA kernels with
  explicit pointer offsets and masked loads/stores; preserve existing strides,
  triangular boundaries and zero-padding behavior.
- Adapt the local FlyDSL compatibility shim to 0.3.2.
- The pinned AITER decode GDN now writes zero output for inactive rows natively.
  Remove the old archive patch. Verify native kernel/wrapper hashes and the
  packaged source copy before allowing uninitialized output allocation; unknown
  sources retain the safe RTP zero-initialization fallback.

## Framework-side PA workaround

Gluon PA can read physically allocated V padding outside the logical sequence.
Zero attention probability does not prevent `0 * NaN/Inf` propagation.

ROCm `BlockPool` initializes its backing bytes to zero before publishing the pool.
This covers V, K, scale and state storage, including host backing that can later
be copied to GPU. GPU initialization is synchronized once before other streams
can access it. No clear or synchronization is added to the per-token path;
ordinary free/reallocation does not erase cached prefixes. NVIDIA behavior is
unchanged by this policy.

The invariant requires subsequent valid KV writes and imported cache contents
to remain finite. This is not a repair for non-finite live activations or poisoned
external cache payloads. Such failures must not be silently sanitized.

## Validation (MI308X, 2026-09-11)

- RTP ROCm wheel built and installed with the pinned stack.
- Pointer-boundary tests: 4 passed.
- Prefill adapter: 14 passed.
- Decode adapter: 30 passed, 1 NVIDIA-only test skipped.
- Focused `block_pool_zero_init_test`: passed for host/device initialization
  and preservation of finite data across free/reallocation.
- Replayed the saved real PA failure: original output had 252 non-finite
  elements; clearing only invalid V padding reduced this to zero. Independent
  FP32 attention reference maximum absolute error: 0.00597954.
- Qwen3.5-4B, TP=1, Graph-on: baseline and both-GDN modes completed an initial
  8K request, eight fixed requests each at 2K/8K, then six 2K requests at
  concurrency 3, with 100 output tokens for the fixed/mixed sets.
- Comparing each mode against its own pre-upgrade recorded outputs: 8/8 exact
  at 2K and 8/8 at 8K for both modes. Between baseline and optimized modes:
  8/8 at 2K and 5/8 at 8K, unchanged from the recorded pre-upgrade comparison.
  Token comparisons re-encode returned text, not raw server token IDs.
- Both-GDN Graph-off completed the same long/short and mixed-batch scenario;
  outputs matched Graph-on 8/8 at both 2K and 8K.

The existing broad `block_pool_test` target cannot build in this ROCm environment
because it depends on CUDA-only FlashMLA. The focused pool target avoids those
unrelated dependencies. This is not full-CI, full KDA, or NVIDIA GPU validation.
Startup initialization cost and steady-state
performance should be measured separately.

## End-to-end performance

Qwen3.5-4B, TP=1, 8192 input tokens, concurrency 1, Graph-on. Both modes use
the upgraded dependencies; baseline disables both AITER GDN paths and sets
`USE_FLYDSL=0`. Each mode has two rounds of 3 warmups + 50 measured requests
per phase, without profiling. All 400 measured requests succeeded.

| Mean latency | Triton GDN baseline | AITER GDN enabled | Reduction |
| --- | ---: | ---: | ---: |
| TTFT, 1 output token | 602.879 ms | 580.623 ms | 3.69% |
| TPOT, 100 output tokens | 6.6882 ms | 6.6152 ms | 1.09% |
| E2E, 100 output tokens | 1265.239 ms | 1236.273 ms | 2.29% |

These are within-stack comparisons, not proof that the dependency upgrade alone
caused the absolute latency change from previous runs. Small gains remain
sensitive to run conditions.

```bash
bazelisk build //rtp_llm:rtp_llm --config=rocm --jobs=8
bazelisk test \
  //rtp_llm/models_py/triton_kernels/fla/test:test_pointer_boundaries \
  //rtp_llm/models_py/triton_kernels/fla/test:test_aiter_flydsl_gdn_prefill \
  //rtp_llm/models_py/triton_kernels/fla/test:test_aiter_flydsl_gdn_decode_rocm \
  //rtp_llm/cpp/cache/test:block_pool_zero_init_test \
  --config=rocm --jobs=8 --nocache_test_results --test_output=errors \
  --run_under=//rtp_llm/test/utils:gpu_lock
```
