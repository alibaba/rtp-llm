# Qwen4 M11 MTP timeline

The combined Qwen4 M11 target uses the native `offline_bench_test.py` runner
to capture prefill and decode from the same requests in a single `PDFUSION`
server launch. The standalone phase targets use `batch_decode_test.py`.

| Phase | Bazel target | Profiled steps |
| --- | --- | ---: |
| Prefill | `//rtp_llm/test/perf_test:qwen4_m11_mxfp4_sm10x_mtp_prefill_timeline` | 1 |
| Decode | `//rtp_llm/test/perf_test:qwen4_m11_mxfp4_sm10x_mtp_decode_timeline` | 6 |
| Combined | `//rtp_llm/test/perf_test:qwen4_m11_mxfp4_sm10x_timeline` | 1 prefill + 6 decode |

Use CUDA 13 on an eight-GPU B300 worker. The combined target uses
`/mnt/fuse/qwen4-m11-zhongwei` for the target checkpoint, tokenizer, and
draft checkpoint, and sets `ENABLE_FP32_LM_HEAD=0`. Run it directly on a
worker with that path mounted.

For the standalone phase targets, pass the same absolute checkpoint path through
`--test_arg=--checkpoint_path=/path/to/checkpoint`,
`--test_arg=--tokenizer_path=/path/to/checkpoint`, and
`--test_arg=--sp_checkpoint_path=/path/to/checkpoint`. Each launch stages the
roughly 305 GiB PLE table once for all ranks. Allow an additional 96 GiB of
available host RAM.

The combined target sends eight 16,381-token prompts and requests 64 output
tokens each. The offline runner keeps the real context stream and uses the
normal MTP sampler. The native MTP test scheduling mode expects one real
request on each DP rank, then keeps the ranks in the decode phase for the
rest of the batch. Each rank's startup profiler targets the first seven
model steps. The raw `qwen4_m11_mtp_pd_wr*.json` files and
`offline_benchmark_result.json` go directly in `TEST_UNDECLARED_OUTPUTS_DIR`.

All targets use TP1/DP8/EP8. The standalone grid targets use batch size one
per DP rank and request one output token for prefill or 64 for decode. The
standalone decode case uses synthetic KV inputs and forced speculative
acceptance. Its trace includes fake prefill for DP alignment. The BUILD
enables three MTP proposals per decode step, CUDA Graphs, single-launch QSA
Indexer preparation, FlashInfer GDN, and 4,096-token Indexer scoring windows. Set
`QWEN4_PROFILE_VERIFY_GRAPH_EVENT=1` through `--test_env` to add an event
after the first 12 MTP target verify layers.

For the standalone targets, the runner also pre-arms profiling on DP rank 0
for the measured grid point. Look in `TEST_UNDECLARED_OUTPUTS_DIR` for
`timelines/`, the prefill or decode results JSON, and `test_info.json`.

For prefix-cache performance, use `cache_grid_perf_test` with
`--cache_grid_json`. The manual smoke target
`//rtp_llm/test/smoke:qwen4_m11_mxfp4_sm10x` also compares token IDs from a
prefix-cache hit with a cold request. Give it the same checkpoint through
`QWEN4_M11_CHECKPOINT`.
