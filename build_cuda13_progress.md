# CUDA 13 x86 Wheel Build Progress

## Environment
- **Machine**: Alibaba Cloud Linux 3 (x86_64)
- **CUDA**: 13.2.1 (nvcc 13.2.78)
- **GPU**: NVIDIA L20D x5
- **GCC**: 10.2.1 (system default)
- **Python**: 3.10.9 (/opt/conda310/bin/python3)
- **PyTorch**: 2.11.0
- **Bazel**: bazelisk

## Branches
- **External (github-opensource)**: `image-cuda13_x86` @ `1f27b4c80` (fix(cuda13): finish x86 build compatibility)
- **Internal (RTP-LLM)**: `feature/image-cuda13_x86` @ `d4408a9c3` (fix(cuda13): finish internal x86 dependency wiring)

## Build Config
- `--config=cuda13`
- Target: `//rtp_llm:rtp_llm_whl`
- SM targets: 8.0, 8.6, 8.9, 9.0, 10.0
- Jobs: 32 parallel
- Spawn strategy: local

## Build Result

### Attempt 1 — SUCCESS
- **Command**: `bazelisk build //rtp_llm:rtp_llm_whl --config=cuda13 --spawn_strategy=local --jobs=32`
- **Total actions**: 24,325 (17,219 internal, 7,106 local)
- **Elapsed time**: 1497s (~25 min)
- **Critical path**: 492s
- **Status**: Build completed successfully

### Output
- **Wheel path**: `bazel-bin/rtp_llm/rtp_llm-0.2.0-py3-none-any.whl`
- **Wheel size**: 3.0 GB
- **Files in wheel**: 670

### Shared Libraries
| Library | Size |
|---------|------|
| `libth_transformer.so` | 1.65 GB |
| `librtp_compute_ops.so` | 1.39 GB |
| `libaccl_ep.so` | 58 MB |
| `libth_transformer_config.so` | 58 MB |

### Warnings (non-fatal)
- SparseMlaParams.h(67): warning #997-D — virtual function override hidden by derived class method (cosmetic, does not affect correctness)

## Smoke Test Results (CUDA 13)

### `mla_kernel_block_size` — PASSED
- Single-GPU GLM-5 FP8 test with kernel block size config
- Required fix: added `fast-hadamard-transform` to cuda13 deps (built from source)
- Required fix: updated golden data for cuda13 GPU output
- Golden: `"acherideraHashTableMJellerhaupterdechu_mkhots"`

### `mla_cp_pd` — BLOCKED (requires deep_ep)
- P/D disaggregation test with DeepEP-based MoE routing
- Root cause: `deep_ep` has no CUDA 13-compatible wheel
  - SONAME issue: existing `.so` links to `libcudart.so.12`
  - NCCL GIN APIs unavailable in current NCCL version for CUDA 13
- Both DeepEP routers call `DeepEPWrapper.supported()` which returns False
- Cannot fix without rebuilding `deep_ep` for CUDA 13

### `mla_allgather_pd` — PASSED (P/D with CUDA graph on decode)
- Tests P/D disaggregation using allgather-based MoE routing
- **Decode side runs with `--enable_cuda_graph 1`** (the requested feature)
- Prefill: `PureCpRouter` with `fp8_per_block_pure_cp` strategy
- Decode: DeepEP low-latency router with CUDA graph capture/replay
- Golden: `"acherideraHashTableMJ Treaty Minutessf56korphan"`
- Uses GLM-5-FP8-4layer model with P/D cache store infrastructure
- Elapsed: 263.2s

### `mla_cp_pd` with CUDA graph — PASSED (after deep_ep wheel install)
- After installing pre-built deep_ep CUDA 13.2 wheel (`/home/zw193905/tmp/ACCL-EP/dist/deep_ep-1.2.1.12+37fda1c.base-cp310-cp310-linux_x86_64.whl`)
- Decode side with `--enable_cuda_graph 1` + deepep_low_latency
- Required fix: `indexer_op.py` — `unsqueeze(0)` → `unsqueeze(1)` for deep_gemm 2.5.0 API change
- deep_gemm 2.5.0 expects `context_lens` as `[batch_size, next_n]`, not `[next_n, batch_size]`

## Key Bug Fix: deep_gemm 2.5.0 API Change

**File**: `rtp_llm/models_py/modules/base/cuda/indexer_op.py` (line ~390)

**Problem**: `deep_gemm.fp8_paged_mqa_logits` and `get_paged_mqa_logits_metadata` changed their `context_lens` parameter from `[next_n, batch_size]` to `[batch_size, next_n]` in deep_gemm >= 2.5.0.

**Error**: `RuntimeError: Assertion error (csrc/apis/attention.hpp:355): batch_size == __batch_size and next_n == _next_n`

**Fix**:
```python
# Before (deep_gemm 2.1.1, cuda12.9 / H20):
kvlen_2d = fmha_params.kvlen_d.unsqueeze(0)  # [1, B] — wrong for 2.5.0

# After (deep_gemm 2.5.0, cuda13 / L20D):
kvlen_2d = fmha_params.kvlen_d.unsqueeze(1)  # [B, 1] — correct
```

**Impact**: Only affects decode-phase indexer TopK with CUDA graph enabled (DSA sparse attention). The old format happened to work for batch_size=1 (warmup) but fails for batch_size>=2 (actual capture). Already committed in `1f27b4c80`.

## Files Modified for Smoke Tests

| File | Change |
|------|--------|
| `rtp_llm/BUILD` (line 249) | Added `fast-hadamard-transform` to cuda13_x86 deps; changed cuda13 from `["deep_gemm"]` to `deep` (includes deep_ep) |
| `internal_source/bazel/arch_select.bzl` (whl_deps cuda13) | Added fast-hadamard-transform wheel |
| `internal_source/deps/requirements_torch_gpu_cuda13.txt` | Added fast_hadamard_transform |
| `internal_source/deps/requirements_lock_torch_gpu_cuda13.txt` | Added lock entry |
| `rtp_llm/test/smoke/data/model/glm5/glm_5_fp8_q_r_h20.json` | Updated golden for cuda13 |
| `rtp_llm/test/smoke/data/model/glm5/glm_5_fp8_q_r_h20_allgather_pd.json` | New golden for allgather P/D test |
| `rtp_llm/test/smoke/suites_h20_oss.bzl` | `mla_allgather_pd` decode: added `--enable_cuda_graph 1` + deepep_low_latency |
| `rtp_llm/models_py/modules/base/cuda/indexer_op.py` | `unsqueeze(0)` → `unsqueeze(1)` for deep_gemm 2.5.0 |
| `rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc` | Added try/catch for better error diagnostics during initCapture |

## Uncommitted Changes (ready to commit)

1. `rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc` — try/catch around `py_forward_method_` in `initCapture()` for better Python exception reporting
2. `rtp_llm/test/smoke/suites_h20_oss.bzl` — `mla_allgather_pd` decode args: `--use_deepep_moe 1 --use_deepep_low_latency 1 --enable_cuda_graph 1`

## GLM-5 Performance Test Results (8DP8EP)

### Configuration
- Model: GLM-5-FP8 (full 78 layers, 256 experts)
- GPU: 8x NVIDIA L20D (275GB, SM 10.3)
- Deployment: 8DP8EP with DeepEP low-latency + CUDA graph
- Transport: nvshmem NVLink P2P (IB disabled)

### Results at 50ms TPOT Target

| Input Length | Max Batch Size | TPOT (ms) | Throughput (TPS) |
|---|---|---|---|
| 128 | 64 | 48.4 | **1,321** |
| 1024 | 64 | 48.7 | **1,313** |
| 2048 | 64 | 49.5 | **1,294** |
| 4096 | 56 | 50.0 | **1,121** |

### Key Findings
- CUDA graph provides **5.3x speedup** (150ms → 28.5ms at bs=1)
- DeepEP low-latency stable on NVLink-only nodes with `NVSHMEM_DISABLE_IB=1`
- Performance consistent across input lengths (MoE-compute dominated, not attention-bound)

### Full Report
See `/home/zw193905/RTP-LLM/glm5_perf_test_report.md`

## Remaining Work

1. Upload `fast_hadamard_transform-1.1.0` wheel to OSS and replace `file://` path with proper URL
2. Consider platform-specific golden data if tests need to pass on both H20 and L20D GPUs
3. Rebuild deep_ep for CUDA 13 and add to OSS deps (currently using local wheel install)
