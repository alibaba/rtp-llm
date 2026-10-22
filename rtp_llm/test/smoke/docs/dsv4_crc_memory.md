# Flash DSpark CRC memory-cache smoke

Target: `//rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_dspark_cprr_crc_memory_sm103`.

This CUDA13 x86 test uses four GPUs: CP2/EP2 prefill and DP2/EP2 decode. CUDA13 x86 and ARM builds automatically protect HOST/DISK cache transfers with CRC; there is no runtime CRC flag. Builds without CRC support, including PPU, ROCm and CUDA12, retain the existing copy path. In this test, the prefill server stores and reloads HOST cache, with DEVICE/DISK/remote reuse disabled. Decode receives the P/D transfer and runs DSpark with CUDA graphs. Both roles use CP page-RR (`seq_size_per_block=256`, `kernel_seq_size_per_block=128`).

The test uses the CI checkpoint `/mnt/nas1/hf/DeepSeek-V4-Flash-DSpark` for both target and draft, matching the existing DSpark smoke. It must include DSpark `mtp.*.markov_head` and confidence-head weights; the ordinary MTP-only checkpoint is not equivalent. Verify that this directory, all indexed weight shards, and the tokenizer are available in the test container before running.

The new fixture combines three requests without changing existing goldens or comparer behavior:

1. A known deterministic one-token request populates HOST cache.
2. The identical request requires `memory_reuse_len=512`, `reuse_len=512`, and `disk_reuse_len=0`, using the existing x86 HOST-only fixture's prompt and assertions. A cold-cache or no-reuse run must fail this assertion. The 10-second inter-request interval allows asynchronous stores to settle.
3. A separate generation must emit at least 16 tokens and contain the requested number endpoints. This exercises DSpark decode rather than claiming that a one-token prefill response proves speculative decoding ran. It uses existing DashSc semantic checks and no grammar constraint or exact token golden.

The HOST-cache hit must exercise the default CRC path on this CUDA13 build. Retain the `BlockTree CRC32C enabled:` startup records and the HOST-cache reuse assertions below to confirm that the test stores and reloads cache with CRC enabled. Require no unexpected checksum/compute failures and DSpark initialization in the server logs. Do not bypass CRC or replace DSpark with MTP to obtain a pass.

## Evidence required for the third request

Keep `SAVE_RESPONSE=False`, `SAVE_LOGITS=False`, and `SAVE_HIDDEN_STATES=False` in the test environment. Each of these, when `True`, makes the comparers skip assertions; do not use `rewrite_smoke` for this regression.

The third request checks `min_generated_ids >= 16` and the required `content_contains` strings through `DashScGrpcComparer`. Those output checks do not by themselves prove speculative proposal execution. This target enables `LOG_LEVEL=DEBUG` on decode: within the third request's execution interval, require both `decode dspark propose model model_input:` and a subsequent `[MTP decode] target model verify forward end`. Match the `trace_ids` / `request_id` in the model-input record to the live request where available. Exclude server startup, CUDA-graph capture, and warmup records; the initialization message `[DeepSeekV4DSparkModel] fixed gamma=3` alone is insufficient.

Keep the `smoke_actual/...query_0.json`, `query_1.json`, and `query_2.json` artifacts, the test log, and complete `prefill_logs/` and `decode_logs/` directories. For request 2, require top-level `reuse_len=local_reuse_len=memory_reuse_len=512` and `disk_reuse_len=0`; the expected phase-specific prefill fields are zero because this one-token request can finish on prefill without running the remote-generate response-merging stage. They are not a sum over CP ranks. Check the CRC-enabled startup records in the prefill worker logs together with these response reuse assertions; successful CRC copies do not emit completion logs.

## Execution

Use a CUDA13 build with four SM103 GPUs and the checkpoint described above. From the repository root, run:

```bash
bazel test //rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_dspark_cprr_crc_memory_sm103 \
  --config=cuda13 \
  --run_under=//rtp_llm/test/utils:gpu_lock \
  --nocache_test_results \
  --test_env=SMOKE_KEEP_SERVER_ALIVE=False \
  --test_timeout=3600
```

The target is also registered in `//rtp_llm/test/smoke:smoke_cuda13_x86`. Keep golden comparisons enabled and avoid running the smoke alongside a GPU benchmark.

CRC failure dumps are written under `${LOG_PATH:-logs}/kv_cache_crc/rank_<rank>/`. Retain that directory, the test log, server logs, and response artifacts when investigating a failure. After the test, confirm that its server processes have stopped and released their GPUs.
