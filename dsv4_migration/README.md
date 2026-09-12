# DeepSeek V4 Pro CSA Offload

Based on main at `1f57ca1b5b` (2026-09-11), developed on
`rym/feat/dsv4_dsa_kvoffload`. Read [RESULTS.md](RESULTS.md)
for real-model measurements and their limits, and [REPRODUCE.md](REPRODUCE.md)
for commands. Synthetic cache timings are separate from model TPOT.

中文入口：[设计与显存口径](DESIGN.zh-CN.md)、[实测结果](SUMMARY.zh-CN.md)。

## Design

- CSA Indexer Key, HCA KV, SWA KV and all compressor states remain on GPU.
- CSA KV has a full pinned CPU backing. A fixed prefix of physical CSA slots is
  mirrored on GPU, sized from the configured GPU budget.
  The native BlockPool allocator takes the lowest free block IDs first, so freed
  resident slots are reused before the CPU-only suffix.
- Each request owns 2048 compressed CSA entries per CSA layer for cross-step
  reuse. Pro selects 1024 compressed entries per step. This is a private hot
  cache, separate from the GPU resident prefix.
- Current TopK hits are protected. Misses use a circular scan over unprotected
  slots in that request's partition. This is CLOCK-like replacement, not exact
  LRU. Tokens selected by another request never evict this partition.
- Request registration runs once per model step. GPU epochs invalidate cached
  entries when a request's first allocator page is reused after prefill.
- TopK translation starts fetch on a side stream before the main CSA compressor.
  The current compressed boundary entry is deferred until the compressor writes
  it. Stream events join the fetch before attention. Storage and metadata have
  fixed addresses for CUDA Graph replay.
- Packed MODEL1 payload and scale bytes are copied exactly, including the split
  per-block payload/scale layout. No changed TopK selection or quantization.

Current scope: FP8 KV, CP1, no prefix reuse, no PD, no MTP, single-token decode.
Baseline uses the same main code with offload disabled. There is no GLM indexer
sharing assumption in this implementation.

## Configuration

| Variable | Meaning | Default |
| --- | --- | --- |
| `DSV4_CSA_OFFLOAD` | Explicit opt-in | 0 |
| `DSV4_CSA_GPU_CACHE_MIB` | All CSA resident, hot and metadata GPU allocations | 32768 |
| `DSV4_CSA_LOGICAL_BLOCKS` | CPU CSA pool blocks, each covers 256 raw tokens in this harness | 65537 |
| `DSV4_CSA_FETCH_CTAS` | Concurrent CUDA thread blocks for CPU fetch | 64 |
| `DSV4_CSA_VALIDATE_BYTES` | Diagnostic: assert every selected GPU CSA entry matches CPU bytes on every decode/graph replay; exclude from timing | 0 |

The C++ memory planner deducts the GPU cache reservation from the overall
`--kv_cache_mem_mb` budget before sizing mandatory native pools. Do not subtract
it a second time in the launcher. The 42 GiB budget used in the earlier GLM work
leaves insufficient prefill room for Pro on these four cards. Real 128K
calibration measured about 20.2 GiB of additional PyTorch allocations. The final
comparison uses 12 GiB overall per GPU; the offload split is 6 GiB for CSA
residency/hot/metadata and 6 GiB for mandatory native pools. Allocation and
capacity are verified in runtime logs.

The local experiment launcher defaults to the tested 12 GiB total KV budget,
6 GiB CSA GPU reservation, maximum B32 and 256 fetch CTAs. These launcher
overrides differ from the generic module defaults in the table above. The
benchmark defaults to a 45 ms SLO; older raw results retain their original
100 ms setting and the summary recomputes goodput at the stated thresholds.

## Files

- `rtp_llm/models_py/modules/dsv4/fp8/csa_cache.py`: residency, private cache,
  request epochs and asynchronous fetch.
- `rtp_llm/models_py/modules/dsv4/fp8/kv_offload.py`: minimal gather primitive and
  MODEL1 transfer validation.
- `rtp_llm/models/dsv4_kv_cache.py`: CSA CPU placement descriptor.
- `rtp_llm/cpp/cache/MemoryEvaluationHelper.cc`: GPU budget reservation.
- DSV4 model, decode, attention and compressor modules contain integration hooks.
- `harness/build.sh`: native build for this machine.
- `harness/start.sh`: TP4, EP4, DP1, CP1, MegaMoE SE, CUDA Graph server.
- `harness/corpus.py`: select original trace contexts; prompts stay in `/tmp`.
- `harness/bench.py`: real-prefill fixed-cohort long-context comparisons.
- `harness/cache_probe.py`: synthetic cache-only graph latency bounds.
- `harness/summarize.py`: common-budget tables, SLO goodput and output parity.

## Pro TP4 Bring-Up

The unmodified main execution path needed three corrections on this machine:

- The installed FlashMLA x86 wheel `1.0.0+9241ae3` rejects 32 heads in both
  prefill and decode. `flash_mla_compat.py` pads independent heads to 64, then
  returns the original 32 outputs. Both schemes use this adapter. GPU numerical
  and graph replay tests cover it. This is a compatibility cost in both results.
- The loader shards the embedding's hidden dimension under TP, whereas DSV4's
  `EmbeddingTorch` returned only the local 1792 dimensions. The model now uses
  the existing TP `Embedding` implementation for sharded weights, restoring
  the 7168-dimensional hidden state with the existing collective API.
- The prefill Q workspace used the global head count. It now uses the actual
  layer's local head count, fixing a reshape failure and avoiding approximately
  12 GiB of excess Q storage at 128K/TP4. TP and CP workspace regressions cover
  the shape contract.

The launcher materializes weights before allocating the full KV cache, avoiding
the weight-repacking startup OOM observed with the old 42 GiB configuration.
The framework's no-cache prefill warmup returns placeholder hidden states, so
it is not a valid runtime-memory probe. `BENCH_REAL_PREFILL_MEMORY` instead
records allocator peaks on real input requests.

For a maximum batch of B, SWA/CSA-state/indexer-state pools use `2*B+2` physical
blocks and HCA state uses `B+2`. The fixed-cohort harness disables the default
5% admission reserve for both schemes; it still allocates real decode KV pages.
An undersized fixed state pool or admission reserve must not be misreported as
the full attention-KV capacity limit.

## Measurement

`RTP_BENCH_REAL_PREFILL=1` makes the existing batch decode scheduler perform one
real prefill per request before decoding the complete cohort. The default
microbenchmark behavior skips prefill and must not be used for this comparison.

Decode measurements omit the first 16 output tokens and then time 128 tokens.
Report actual model input lengths, observed decode batch sizes, graph replay
evidence, per-GPU memory, TPOT and output tokens/second. Failure rate is actual
request errors divided by attempts. Exceeding the TPOT SLO does not count as a
request failure. Decode goodput counts measured tokens from successful requests
meeting the explicitly recorded TPOT SLO, divided by decode wall time. This
fixed-cohort metric excludes prefill and is not end-to-end serving goodput.

Capacity rejection applies to the whole fixed cohort in this harness. Ordinary
serving can queue requests or reduce its active batch instead. A response error
from this explicit fixed-cohort rule is not a physical CUDA OOM.

Keep KV budget admission failures distinct from physical CUDA OOM. At 128K,
offload B16 measured 4.3% higher TPOT than vanilla B16. Offload B32 fits and
increases decode goodput by 52.1% over vanilla B16 at a 45 ms SLO, with 31.4%
higher TPOT. It does not establish unchanged latency when doubling the batch.
The finer 8/4 GiB split gives B18 at 30.82 ms and 583.49 token/s, or +5.1%
TPOT and +7.0% goodput relative to vanilla B16. B20 gives 32.36 ms and
616.03 token/s (+10.4% TPOT, +13.0% goodput). All are single-pass measurements.

## Validation So Far

- Native build succeeds.
- Existing 28 DSV4 cache configuration tests pass.
- 27 configuration and attention orchestration checks pass.
- GPU transfer layout, FlashMLA parity, graph replay, request reuse and private
  cache tests pass, including the real compressor and B64/TopK1024/Hot2048 case.
- The final focused suite passes 18 tests, including detection of intentionally
  corrupted payload bytes during graph replay.
- Real-model byte validation passes at B16/128K for 40 decode steps, all 30 CSA
  layers and all four ranks. This diagnostic is excluded from performance data.
- `results/cache-probe.json` contains synthetic cache latency, not model TPOT.
