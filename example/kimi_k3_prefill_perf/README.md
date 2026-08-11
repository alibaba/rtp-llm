# Kimi K3 4-layer 64K Prefill timeline

This directory contains the launcher for the K3 performance-only path:

- TP8 / EP8, batch 1, 65,536 input tokens, one generated token.
- `KDA -> KDA -> KDA -> MLA`, with three MoE layers.
- cuLA is the default KDA backend. cuLA and its matching FLA runtime are
  supplied by the outer repository's CUDA13 Bazel dependency set; their wheels
  are not checked into the open-source tree.
- FlashKDA `fa7eb894824a` remains selectable for same-branch A/B profiling.
- DeepGEMM `f5a76426fa084087169693fd0cd815223576d6e9` with K3 SiTU
  and the CUDA13 float-NTTP fix for MegaMoE.
- RTP native MLA.
- Sequence Parallel Attention/MoE using Attention-side AllGather +
  ReduceScatter; MegaMoE owns expert dispatch/combine.

The checked-in auxiliary operator bundle contains only FlashKDA and DeepGEMM.
It is validated for CPython 3.10, PyTorch 2.11.0+cu130, CUDA 13.0 and
L20D/B300 `sm_103a`. Both artifacts are verified through
`wheels/SHA256SUMS` and installed into an isolated runtime overlay.

## Run the rs_ag timeline

Run inside the CUDA13 `lhc_GPU` container as `luohaocheng.lhc`:

```bash
cd /path/to/RTP-LLM
CHECKPOINT_PATH=/data0/luohaocheng.lhc/Kimi-K3-4layers-preflight \
  ./example/kimi_k3_prefill_perf/run_64k_timeline.sh
```

The K3 fused projection prefix uses the production `rs_ag` weight and
communication layout. The former four-layer A2A experiment requires a
different replicated weight layout and is rejected rather than silently
running with incompatible weights.

The inner `github-opensource` checkout still uses RTP-LLM's normal outer-repo
layout: its `stub_source -> ../internal_source` link must resolve before Bazel
is invoked.

The script:

1. validates and installs the pinned FlashKDA and DeepGEMM wheels into the run
   artifact; cuLA/FLA come from the Bazel dependency set;
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
The KDA prefill backend is always `cula`.

## Refresh or rebuild the operators

The pinned auxiliary wheels make the timeline reproducible without rebuilding
FlashKDA or DeepGEMM. To rebuild those two native operators:

```bash
./example/kimi_k3_prefill_perf/build_operator_wheels.sh
```

This clones and builds the pinned FlashKDA and DeepGEMM revisions. DeepGEMM
uses `patches/deepgemm_cuda13_float_nttp.patch`. The builder checks the pinned
Python/PyTorch/CUDA environment and publishes only the two expected wheel
filenames after both builds complete successfully. Set
`K3_PERF_OPS_BUILD_ROOT` to a scratch parent directory when needed; the
builder creates and removes only its own unique child directory.

## Scope

This launcher exercises the same production modeling path as the PD launcher.

Sequence Parallel ReduceScatter, cuLA/FlashKDA and FP8-activation MXFP4
MegaMoE change accumulation order and datatype behavior relative to the
accuracy fallback. Use the four-layer Golden and full-model PD regression
before claiming numerical acceptance for a new environment.
