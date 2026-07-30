# Kimi K3 4-layer 64K Prefill timeline

This directory is the self-contained launcher for the K3 performance-only path:

- TP8 / EP8, batch 1, 65,536 input tokens, one generated token.
- `KDA -> KDA -> KDA -> MLA`, with three MoE layers.
- FlashKDA `fa7eb894824a` for KDA.
- DeepGEMM `f5a76426fa084087169693fd0cd815223576d6e9` with K3 SiTU
  and the CUDA13 float-NTTP fix for MegaMoE.
- RTP native MLA.
- Sequence Parallel Attention/MoE using Attention-side AllGather +
  ReduceScatter; MegaMoE owns expert dispatch/combine.
- Accuracy/reference/trace-comparison paths are rejected by
  `KIMI_K3_PERF_MODE=1`.

The checked-in wheels are intentionally limited to the environment used for
the reference timeline: CPython 3.10, PyTorch 2.11.0+cu130, CUDA 13.0 and
L20D/B300 `sm_103a`. Their hashes are pinned in `wheels/SHA256SUMS`.

## One command

Run inside the CUDA13 `lhc_GPU` container as `luohaocheng.lhc`:

```bash
cd /path/to/RTP-LLM
CHECKPOINT_PATH=/data0/luohaocheng.lhc/Kimi-K3-4layers-preflight \
  ./example/kimi_k3_prefill_perf/run_64k_timeline.sh
```

The inner `github-opensource` checkout still uses RTP-LLM's normal outer-repo
layout: its `stub_source -> ../internal_source` link must resolve before Bazel
is invoked.

The script:

1. validates and installs the bundled operator wheels into the run artifact;
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

## Rebuild the operators

The prebuilt wheels make the timeline reproducible without network access.
To rebuild from source:

```bash
./example/kimi_k3_prefill_perf/build_operator_wheels.sh
```

This clones the pinned FlashKDA and DeepGEMM revisions, initializes their
submodules, applies `patches/deepgemm_cuda13_float_nttp.patch`, and builds an
`sm_103a` FlashKDA wheel plus the K3 SiTU DeepGEMM wheel.

## Scope

This launcher enables timeline-only validation and output shortcuts through
`KIMI_K3_PERF_MODE=1`. The same optimized modeling path is available to the
PD launcher through `KIMI_K3_EXECUTION_MODE=optimized`, but PD keeps
`KIMI_K3_PERF_MODE=0` so profiling-only output shortcuts cannot change
serving math.

Sequence Parallel ReduceScatter, FlashKDA and FP8-activation MXFP4 MegaMoE
change accumulation order and datatype behavior relative to the accuracy
fallback. Use the four-layer Golden and full-model PD regression before
claiming numerical acceptance for a new environment.
