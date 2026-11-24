# Qwen4 M11 timeline

Use one 8-GPU B300 host and a CUDA 13 RTP-LLM runtime built from this checkout
with its pinned CUDA 13 dependencies. The checkpoint
argument points to the mounted `qwen4-max-m11` release `MXFP4-1m-2026-11-02-1`
with its tokenizer, config, safetensors index, and shards. The driver generates
prompts using that checkpoint's tokenizer.

The default run uses one TP1/DP8/EP8 `PDFUSION` server with MTP enabled. It
sends eight 16381-token requests, one per DP rank, and generates 64 tokens per
request. The driver captures the target and draft prefill once per rank and
the first six MTP decode steps. Each decode step proposes three tokens and
replays four CUDA graphs for draft, verification, and draft update. The
decode view contains no fake prefill.

The driver sets `RTP_LLM_STREAM_ASYNC=1`, `RTP_LLM_DEVICE_INPUT=1`,
`RTP_LLM_DROP_BROAD_SYNC=1`, and `GEN_TIMELINE_SYNC=0` in the server launch
environment. `RTP_BATCH_INVARIANT=0` selects the optimized MTP and CUDA Graph
paths. The model uses its default LM head precision. The target and
draft run with online quantization disabled. Dense projections and shared
experts keep their BF16 checkpoint weights, and `mega_moe` runs the MXFP4
routed experts through DeepGEMM. The driver aligns target and draft token
capacities so they share the MegaMoE buffer. The driver requests FlashInfer GDN
and Indexer and enables single-launch Indexer preparation. The GPU trace records
the resolved backend for each phase. Prefill runs without chunked prefill, with
4096-token FlashInfer Indexer scoring windows for cold requests. Prefix-hit
diagnostics score directly from paged KV.

## Run from the B300 runtime

From the repository root, with the runtime and native libraries available to the
same Python interpreter used to launch RTP-LLM:

```bash
checkpoint=/absolute/path/to/qwen4-m11-checkpoint
output=$(mktemp -d /tmp/qwen4-m11-timeline.XXXXXX)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
GPU_COUNT=8 RTP_TEST_WATCHDOG_SECONDS=10800 python -m rtp_llm.test.utils.device_resource \
    python -m rtp_llm.test.perf_test.qwen4_m11_timeline \
    --checkpoint_path "$checkpoint" --output_dir "$output"
```

The corresponding manual Bazel target is
`//rtp_llm/test/perf_test:qwen4_m11_mxfp4_sm10x_timeline`. Pass
`--test_arg=--checkpoint_path=/absolute/path/to/qwen4-m11-checkpoint`, use the
CUDA 13 configuration and `--test_timeout=10800`, and make the checkpoint
accessible to the test worker.
The target uses the Bazel test GPU lock and writes to the undeclared outputs.

`--mtp` is an alias for the default run. For a separate diagnostic capture
with MTP disabled, pass one of these options (Bazel `--test_arg`):

- `--decode_lookahead` checks that the server submits the next Decode CUDA
  graph during the previous replay, with batch 32 and a 32K input.
- `--prefix_cache` adds a prefill with 14336 cached and 2048 new tokens per
  rank after the ordinary decode and cold prefill cases.
- `--short_prefix` adds eight prefill requests per rank, each with 32768 cached
  and eight new tokens. The raw trace retains up to three decode forwards used
  for DP alignment.

Set `QWEN4_PROFILE_VERIFY_GRAPH_EVENT=1` (Bazel `--test_env`) to add an external
event after the first 12 MTP target verify layers.

Set `QWEN4_PLE_SHARED_TABLE` (Bazel `--test_env`) to reuse
a matching shared PLE manifest whose backing remains readable throughout the
test. If unset, the driver stages the roughly 305 GiB PLE table from the checkpoint
in a Linux memfd and keeps it alive until the server stops. Allow an additional
96 GiB of available host RAM for the rest of the run.

The default run writes one raw trace and separate `-mtp-prefill-` and
`-mtp-decode-` traces per rank in `timelines/`. The phase traces preserve GPU
kernel timestamps and launch correlations. The driver checks the CUDA Graph
count, FlashInfer Prefill QSA and Indexer, fused prefill GR on each rank, and
tiled PLE prefill on rank 0. It writes
metrics for the profiled requests to `mtp_metrics.json`. Each run records its
server arguments and backend environment in `launch.json`.

The manual smoke target `//rtp_llm/test/smoke:qwen4_m11_mxfp4_sm10x` also checks
that a changed continuation hits the seeded page-aligned prefix and generates
the same token IDs as a cold request. Give it the same checkpoint through
`QWEN4_M11_CHECKPOINT` and the shared PLE manifest through `QWEN4_PLE_SHARED_TABLE`.
