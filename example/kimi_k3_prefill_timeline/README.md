# Kimi K3 4-layer pure-PREFILL 64K timeline

This directory contains a timeline launcher for the real K3 `PREFILL` role:

- TP8 / EP8, batch 1, 65,536 input tokens, one generated token.
- `KDA -> KDA -> KDA -> MLA`, with three MoE layers.
- cuLA is the default KDA backend. cuLA and its matching FLA runtime are
  supplied by the outer repository's CUDA13 Bazel dependency set; their wheels
  are not checked into the open-source tree.
- DeepGEMM with K3 SiTU and the CUDA13 float-NTTP fix for MegaMoE, resolved
  from RTP-LLM's pinned CUDA13 dependency set and Bazel runfiles.
- RTP native MLA.
- Sequence Parallel Attention/MoE using Attention-side AllGather +
  ReduceScatter; MegaMoE owns expert dispatch/combine.
- The request fixes `max_new_tokens=1` and `can_use_pd_separation=false`.
  `PrefillRpcServer` therefore executes it locally without starting or
  contacting Decode, while the model still selects its production PREFILL
  backends. PDFUSION is not used.

No operator wheel is stored or installed from this example. FlashKDA is not
part of this timeline or the production model path.

## Run the rs_ag timeline

Run inside the CUDA13 `lhc_GPU` container as `luohaocheng.lhc`:

```bash
cd /path/to/RTP-LLM
CHECKPOINT_PATH=/data0/luohaocheng.lhc/Kimi-K3-4layers-preflight \
  ./example/kimi_k3_prefill_timeline/run_64k_timeline.sh
```

The K3 fused projection prefix uses the production `rs_ag` weight and
communication layout. This launcher does not support the incompatible
replicated A2A weight layout.

The inner `github-opensource` checkout still uses RTP-LLM's normal outer-repo
layout: its `stub_source -> ../internal_source` link must resolve before Bazel
is invoked.

The script:

1. verifies that the checkpoint is on a local disk and uses FastSafetensors;
2. builds `//rtp_llm:rtp_llm_server` on L20-dev-115 with
   `--config=cuda13 --config=sm10x`;
3. verifies that the pinned DeepGEMM package resolves from that target's Bazel
   runfiles;
4. starts a standalone `PREFILL` service, with no remote Decode;
5. performs one materialization request and at least ten full 64K warmups,
   requiring the latest five to be
   within median ±3%;
6. captures one separate profiler-warmup trace, then re-arms all eight ranks
   and captures exactly one measured 64K request in the final trace;
7. writes eight measured traces plus eight separately labelled profiler-warmup
   traces, logs, environment snapshots and the input hash below
   `${HOME}/kimi_k3_perf_runs/`.

The launcher only stops the process group it created. Set `KEEP_SERVER=1` to
leave that service running. Override `RUN_ROOT`, `START_PORT` or
`CUDA_VISIBLE_DEVICES` when needed. On an inode-constrained host, set
`KIMI_K3_BAZEL_OUTPUT_BASE` to an existing compatible CUDA13/SM10x Bazel
output base so the incremental build reuses its external repositories.
The KDA prefill backend is always `cula`.

## Rebuild DeepGEMM

The runtime wheel is stored in the dependency artifact service. Its source
recipe lives with the RTP-LLM third-party dependency:

```bash
./3rdparty/deep_gemm/build_cuda13_b300_wheel.sh
```

The builder records the source commit and applies the K3 CUDA13 patch. Publish
the resulting wheel outside the checkout and update the CUDA13 requirement and
lock hash if the artifact changes.

## Scope

This launcher uses the same role-derived K3 model backends as production:
PREFILL selects cuLA, FlashMLA, fused projection paths and DeepGEMM MegaMoE.
It changes only the service topology and measurement protocol: one local
PREFILL role, one 64K request at a time, no Decode and no cache transfer.

Sequence Parallel ReduceScatter, cuLA and FP8-activation MXFP4
MegaMoE change accumulation order and datatype behavior relative to accuracy
mode. Use the four-layer Golden and full-model PD regression
before claiming numerical acceptance for a new environment.
