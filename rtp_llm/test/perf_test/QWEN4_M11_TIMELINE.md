# Qwen4 M11 Indexer timeline

Use one 8-GPU B300 host and a CUDA 13 RTP-LLM runtime built from this checkout
with its pinned CUDA 13 dependencies, including FlashInfer 0.7.4. The checkpoint
argument points to the mounted `qwen4-max-m11` release `MXFP4-1m-2026-11-02-1`
with its tokenizer, config, safetensors index, and shards. The driver generates
prompts using that checkpoint's tokenizer.

| Phase | Global batch | Batch per DP rank | Input tokens | Output tokens | Profiled forwards per rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode | 32 | 4 | 32768 | 16 | 3 CUDA Graph replays |
| Cold prefill | 8 | 1 | 16384 | 1 | 1 eager forward |
| Prefix-hit prefill (optional) | 8 | 1 | 16384 (14336 cached + 2048 new) | 1 | 1 eager forward |

Both phases run on one TP1/DP8/EP8 `PDFUSION` server, with MXFP8 online quantization,
FlashInfer GDN and Indexer, native QSA, and single-launch Indexer preparation.
Prefill runs without chunked prefill. Cold prefill scores in 4096-token windows,
and prefix-hit prefill scores directly from paged KV in one window. The driver
runs an unprofiled warmup and measurement before arming all eight ranks for each
phase.

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

Add `--prefix_cache` to the direct command (Bazel `--test_arg=--prefix_cache`)
to append the prefix-hit case. The driver seeds the cache on each DP rank before
each prefix-hit batch, then routes the continuation to the same rank. It checks
that each continuation reports exactly 14336 reused tokens. Decode and cold
prefill bypass the cache in both modes.

For either invocation, set `QWEN4_PLE_SHARED_TABLE` (Bazel `--test_env`) to reuse
a matching shared PLE manifest whose backing remains readable throughout the
test. If unset, the driver stages the roughly 305 GiB PLE table from the checkpoint
in a Linux memfd and keeps it alive until the server stops. Allow an additional
96 GiB of available host RAM for the rest of the run.

The output contains 16 raw Chrome Trace JSON files in `timelines/`, unprofiled
`decode_metrics.json` and `prefill_metrics.json`, and server logs. With
`--prefix_cache`, it adds eight `-prefix14336` timelines and
`prefill_prefix_cache_metrics.json`. The driver checks each rank for the expected
CUDA Graph replays, fused Indexer preparation, and FlashInfer score and TopK launches.

The manual smoke target `//rtp_llm/test/smoke:qwen4_m11_mxfp4_sm10x` also checks
that a changed continuation hits the seeded page-aligned prefix and generates
the same token IDs as a cold request. Give it the same checkpoint through
`QWEN4_M11_CHECKPOINT` and the shared PLE manifest through `QWEN4_PLE_SHARED_TABLE`.
