# ROCm real-request startup warmup

ROCm kernels can compile on their first actual inference shape even after model
loading and decode graph capture have completed. For latency-sensitive services,
a real-request warmup keeps readiness closed until a bounded set of
prefill, prefix-cache, and decode requests has finished.

## Scope and configuration

This option currently supports single-rank, non-speculative PDFUSION deployments
on ROCm with device prefix caching enabled. For `model_type=qwen_3` in this
topology it is now **enabled by default** (`auto`); no additional environment
variable is required. Other models/topologies retain their previous behavior.
Explicit `1` validates the supported topology and fails closed when unsupported;
`0` is an emergency opt-out. This does not enable decode graphs automatically.

```sh
RTP_LLM_ROCM_STARTUP_WARMUP=auto
RTP_LLM_ROCM_STARTUP_WARMUP_TIMEOUT_S=600
```

`WARM_UP=1`, `MODEL_WARM_UP=1`, and `REUSE_CACHE=1` must also be enabled. The first
two remain master switches; disabling either skips this warmup. The timeout is
per request and must be finite and positive. Allow the startup probe enough time
for compilation; `/liveness` is intentionally not held closed.

The warmup sends ten sequential, synthetic-token requests directly to the
backend RPC endpoint: one uncached prefill (up to 1024 tokens), a 256-token cache
seed, and cached requests with query lengths 1, 2, 3, 4, 8, 14, 15, and 16. Each
generates two tokens to exercise decode. No business prompt or external request
is used.
The 256-token prefix assumes a compatible cache block size and sufficient cache
capacity; models must accept the synthetic token IDs 100 through 108.

The runner verifies final output length and exact prefix reuse. RPC failure,
early termination, or an unexpected cache hit/miss fails startup rather than
marking the process ready. `/`, `/health`, `/status`, and CM2 health endpoints
remain unavailable until completion. `/frontend_health` is an internal bootstrap
probe and must not be used as the external traffic-readiness signal.

This is bounded shape coverage, not a promise that every possible production
shape is compiled. Validate the model, cache configuration, startup probe budget,
and actual request shapes in a canary before wider rollout. A cold-cache restart
must complete all ten cases before external readiness becomes successful.

Each request has both a backend RPC deadline and a client-side asyncio deadline.
Timeout/cancellation propagates out of warmup, leaves the gate closed, and takes
the existing failed-startup cleanup path. `/liveness` remains healthy during
compilation; do not use it or `/frontend_health` as the traffic readiness probe.

## Steady-state cached-prefill optimizations

The eager ROCm path now borrows the runner's existing cumulative Q/KV lengths
(after its staged copies are flushed), rather than recomputing them on GPU.
CK block-table padding, KV lengths and dummy index arrays are prepared in one
batch-owned pinned host buffer and transferred together. Valid block IDs and
the extra 128-token speculative-prefetch padding are unchanged. Host sources
are retained with that batch; there is no cross-request mutable cache.

Uniform short-query Triton prefill also skips building a compaction index that
its forward never uses. Ragged batches still compact normally. Graph capture
and replay keep their original owned buffers and refresh semantics; these
optimizations do not enable unsupported prefix-cache graph replay.

These changes remove redundant setup work, but are not a measured TTFT claim.
Before rollout, run the ROCm operator tests and compare the same prompts/cache
hit lengths, including q=1..4 and q=14..17 with an 816-token prefix. Report both
steady-state TTFT/RT and cold-start readiness; exclude compilation from the
steady-state window. Do not infer model-wide TTFT gains from a metadata-only
microbenchmark or CPU unit tests.

## Regression checks

Tensor-only and warmup-plan tests run without a GPU:

```sh
PYTHONPATH=. python rtp_llm/models_py/modules/factory/attention/rocm_impl/test/test_paged_prefill_metadata.py
PYTHONPATH=. python rtp_llm/utils/test/rocm_startup_warmup_test.py
```

In a compiled ROCm environment, also run `start_server_warmup_test`,
`frontend_shutdown_test`, and the ROCm `test_aiter_prefill_op` target. The latter
checks real CK output against a reference and repeats forward calls using one
batch's metadata. Include graph replay, prefix lengths, BF16/FP8 KV-cache cases,
and a cold-JIT end-to-end startup before deploying a new image. CPU tests alone
do not establish numerical correctness or latency improvement on MI308X.
