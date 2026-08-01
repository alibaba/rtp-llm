# Kimi K3 4-layer 64K Prefill timeline

This directory is the self-contained launcher for the K3 performance-only path:

- TP8 / EP8, batch 1, 65,536 input tokens, one generated token.
- `KDA -> KDA -> KDA -> MLA`, with three MoE layers.
- cuLA `4db9fb97b791ace6b8c7709b9ead8016b9c0c72a` is the default KDA
  backend. Its SM100 kernel publishes exact FP32 recurrent-state checkpoints
  every 4K tokens without splitting the 64K recurrence. This revision also
  supports packed varlen checkpoints, accepts non-contiguous beta logits and
  uses 64-bit flattened offsets beyond the H64 256K-token boundary; the
  current K3 optimized path uses its fixed-length checkpoint API.
- FlashKDA `fa7eb894824a` remains selectable for same-branch A/B profiling.
- cuLA uses FLA `3a9ce1c83a13994d824dbb3421e2989d330bb38b` plus the pinned
  Python 3.10 / Triton 3.6 compatibility and varlen 64-bit-offset patches.
- DeepGEMM `f5a76426fa084087169693fd0cd815223576d6e9` with K3 SiTU
  and the CUDA13 float-NTTP fix for MegaMoE.
- RTP native MLA.
- Sequence Parallel Attention/MoE using Attention-side AllGather +
  ReduceScatter; MegaMoE owns expert dispatch/combine.
- Accuracy/reference/trace-comparison paths are rejected by
  `KIMI_K3_PERF_MODE=1`.

The checked-in operator wheels are intentionally limited to the reference
environment: CPython 3.10, PyTorch 2.11.0+cu130, CUDA 13.0 and L20D/B300
`sm_103a`. All four artifacts are verified through `wheels/SHA256SUMS` and
installed into an isolated runtime overlay; they do not change RTP's shared
Bazel/Python dependency set.

## One command per KDA communication backend

Run inside the CUDA13 `lhc_GPU` container as `luohaocheng.lhc`:

```bash
cd /path/to/RTP-LLM
CHECKPOINT_PATH=/data0/luohaocheng.lhc/Kimi-K3-4layers-preflight \
KIMI_K3_KDA_COMM_BACKEND=rs_ag \
  ./example/kimi_k3_prefill_perf/run_64k_timeline.sh

CHECKPOINT_PATH=/data0/luohaocheng.lhc/Kimi-K3-4layers-preflight \
KIMI_K3_KDA_COMM_BACKEND=a2a \
  ./example/kimi_k3_prefill_perf/run_64k_timeline.sh
```

`rs_ag` is the safe default.  `a2a` is deliberately restricted to four-layer
Prefill TP8/EP8 experiments: it pre-packs replicated KDA projection weights,
uses SP→TP and TP→SP AllToAll around FlashKDA, and keeps only the low-rank
forget-gate AllGather.  A startup memory guard includes an 8 GiB safety margin
and rejects a full-model A2A configuration before weight replication.

The inner `github-opensource` checkout still uses RTP-LLM's normal outer-repo
layout: its `stub_source -> ../internal_source` link must resolve before Bazel
is invoked.

The script:

1. validates and installs the pinned cuLA, FLA, FlashKDA and DeepGEMM wheels
   into the run artifact;
2. builds the server with `--config=cuda13 --config=sm10x`;
3. starts only a standalone `PDFUSION` service, with no remote Decode;
4. performs one materialization request;
5. performs at least three full 64K warmups, requiring the latest three to be
   within median ±5%;
6. arms all eight ranks and captures exactly one profiled 64K request;
7. writes all eight traces, logs, environment snapshots and the input hash
   below `${HOME}/kimi_k3_perf_runs/`.

The launcher only stops the process group it created. Set `KEEP_SERVER=1` to
leave that service running. Override `RUN_ROOT`, `START_PORT` or
`CUDA_VISIBLE_DEVICES` when needed. On an inode-constrained host, set
`KIMI_K3_BAZEL_OUTPUT_BASE` to an existing compatible CUDA13/SM10x Bazel
output base so the incremental build reuses its external repositories.
Set `KIMI_K3_KDA_BACKEND=flash_kda` for an A/B run; the default is `cula`.

## Rebuild the operators

The pinned wheels make the timeline reproducible without network access or
rebuilding operators.
To rebuild from source:

```bash
./example/kimi_k3_prefill_perf/build_operator_wheels.sh
```

This clones the pinned FlashKDA, internal cuLA and DeepGEMM revisions,
initializes their submodules, builds the patched FLA runtime, performs a fresh
SM103-only cuLA native build, applies
`patches/deepgemm_cuda13_float_nttp.patch` and the cuLA dynamic checkpoint
pointer fix, and builds all operator wheels. Rebuilding the native cuLA
extension is required because the latest long-sequence fix changes C++ W/U
address calculation; overlaying Python on an older `.so` is not valid. The
cuLA pointer patch preserves the kernel math while making FP32 page-boundary
state publication valid for CUTLASS DSL dynamic indices. The builder checks
the pinned Python/PyTorch/CUDA environment and publishes only the four exact
wheel filenames after all builds complete successfully. Set
`K3_PERF_OPS_BUILD_ROOT` to a scratch parent directory when needed; the
builder creates and removes only its own unique child directory.

## Scope

This launcher enables timeline-only validation and output shortcuts through
`KIMI_K3_PERF_MODE=1`. The same optimized modeling path is available to the
PD launcher through `KIMI_K3_EXECUTION_MODE=optimized`, but PD keeps
`KIMI_K3_PERF_MODE=0` so profiling-only output shortcuts cannot change
serving math.

Sequence Parallel ReduceScatter, cuLA/FlashKDA and FP8-activation MXFP4
MegaMoE change accumulation order and datatype behavior relative to the
accuracy fallback. Use the four-layer Golden and full-model PD regression
before claiming numerical acceptance for a new environment.
