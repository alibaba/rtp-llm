# Flash DSpark CRC memory-cache smoke

Target: `//rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_dspark_cprr_crc_memory_sm103`.

This CUDA13 x86 test uses four GPUs: CP2/EP2 prefill and DP2/EP2 decode. CUDA13 x86 and ARM builds automatically protect HOST/DISK cache transfers with CRC; there is no runtime CRC flag. Builds without CRC support, including PPU, ROCm and CUDA12, retain the existing copy path. In this test, the prefill server stores and reloads HOST cache, with DEVICE/DISK/remote reuse disabled. Decode receives the P/D transfer and runs DSpark with CUDA graphs. Both roles use CP page-RR (`seq_size_per_block=256`, `kernel_seq_size_per_block=128`).

The checkpoint is `/mnt/hf3fs/3fs/models/DeepSeek-V4-Flash-0731` for both target and draft. It must include DSpark `mtp.*.markov_head` and confidence-head weights; the ordinary MTP-only checkpoint is not equivalent. The host and container must both expose the actual 3FS mount at `/mnt/hf3fs/3fs/`; a parent `/mnt` bind alone may hide its later submount. Verify all indexed weight shards and the tokenizer before running.

The new fixture combines three requests without changing existing goldens or comparer behavior:

1. A known deterministic one-token request populates HOST cache.
2. The identical request requires `memory_reuse_len=512`, `reuse_len=512`, and `disk_reuse_len=0`, using the existing x86 HOST-only fixture's prompt and assertions. A cold-cache or no-reuse run must fail this assertion. The 10-second inter-request interval allows asynchronous stores to settle.
3. A separate generation must emit at least 16 tokens and contain the requested number endpoints. This exercises DSpark decode rather than claiming that a one-token prefill response proves speculative decoding ran. It uses existing DashSc semantic checks and no grammar constraint or exact token golden.

The HOST-cache hit must exercise the default CRC path on this CUDA13 build. Retain server startup/config logs and CRC transfer evidence (the implementation's counters or diagnostics) to confirm that HOST loads and stores use CRC. Require `BlockTree CRC32C completed: operation=store` and `operation=load` success records (logged once per operation per transfer service), no unexpected checksum/compute failures, and DSpark initialization in the server logs. Do not bypass CRC or replace DSpark with MTP to obtain a pass.

## Evidence required for the third request

Keep `SAVE_RESPONSE=False`, `SAVE_LOGITS=False`, and `SAVE_HIDDEN_STATES=False` in the test environment. Each of these, when `True`, makes the comparers skip assertions; do not use `rewrite_smoke` for this regression.

The third request checks `min_generated_ids >= 16` and the required `content_contains` strings through `DashScGrpcComparer`. Those output checks do not by themselves prove speculative proposal execution. This target enables `LOG_LEVEL=DEBUG` on decode: within the third request's execution interval, require both `decode dspark propose model model_input:` and a subsequent `[MTP decode] target model verify forward end`. Match the `trace_ids` / `request_id` in the model-input record to the live request where available. Exclude server startup, CUDA-graph capture, and warmup records; the initialization message `[DeepSeekV4DSparkModel] fixed gamma=3` alone is insufficient.

Keep the `smoke_actual/...query_0.json`, `query_1.json`, and `query_2.json` artifacts, the test log, and complete `prefill_logs/` and `decode_logs/` directories. For request 2, require top-level `reuse_len=local_reuse_len=memory_reuse_len=512` and `disk_reuse_len=0`; the expected phase-specific prefill fields are zero because this one-token request can finish on prefill without running the remote-generate response-merging stage. They are not a sum over CP ranks. Require successful CRC `operation=store` and `operation=load` records in the prefill worker logs, in addition to the response reuse assertions.

## Execution

Run unit tests first, then this smoke; do not run it alongside another GPU benchmark. Follow `bbdetector-run-distributed-test` and `test-execution`: refresh live GPU availability, reserve four completely empty GPUs, run the scoped precheck, use the GPU lock, and keep smoke cleanup enabled. The local checkout may contain uncommitted changes under the task's confirmed plan; the remote RTP runner requires clean published public/internal commits and an exact matching gitlink. Do not launch Bazel through the runner's generic `--command` or `--script` mode.

After the branch plan and remote source revisions are settled, a remote invocation is:

```bash
python3 "$BBDETECTOR_ROOT/.codex/skills/bbdetector-run-distributed-test/scripts/run.py" \
  --rtp-root "$RTP_LLM_ROOT" \
  --host "$GPU_HOST" --local-host "$LOCAL_HOST" --gpu-count 4 --confirmed-plan \
  --rtp-test //rtp_llm/test/smoke:smoke_v4_flash_pd_cp2ep2_dp2ep2_dspark_cprr_crc_memory_sm103 \
  --bazel-arg=--run_under=//rtp_llm/test/utils:gpu_lock \
  --bazel-arg=--config=daily_aone_bazel_cache \
  --bazel-arg=--remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --bazel-arg=--test_timeout=3600
```

The runner adds CUDA13, disables cached test results and sets `SMOKE_KEEP_SERVER_ALIVE=False`. Run the required scoped precheck separately before this command; it is not built into the runner. Execute all container commands as the real user. Do not pass `--fix` to terminate unrelated or active work. If a runtime comparison fails, investigate the actual response and reuse metrics instead of rewriting the prior goldens.

Capture the test log, undeclared outputs/server logs, selected host/GPU indexes, both source SHAs and build flags. On success, failure or timeout verify that this run's server/worker processes exited and released their GPUs. A passing startup alone is not a smoke pass.
