# Heterogeneous PD for DeepSeek-V4-Flash: prefill KV-transfer fixes and ring MoE

Branch `dev/pd-opt-xiaoweiw`, based on `dev/pd-cache-store-prefill-publish` (`3ffc6f3`).

This branch changes the **prefill** side of heterogeneous prefill/decode (PD) serving:
RTX PRO 5000 (SM120) prefill, PPU decode, KV cache sent over eRDMA. It contains:

1. Three fixes that make every prefill rank publish its KV once per request instead of after every chunk.
2. An opt-in switch that removes the 4x duplicated KV transfer under replicated context parallelism (CP).
3. A benchmark-only switch that sends the whole request's KV at the end (to measure transport drain rate).
4. A copy-engine ring all-gather / reduce-scatter transport for the DSV4 MoE expert-parallel path.
5. Optional timestamps on the prefill side of the PD handoff.
6. Tools that measure the KV transfer directly from the NIC's hardware byte counter.

Server TTFT went from 1189 ms to 1125 ms (client TTFT 1247 ms to 1183 ms) for a 32K-token request, and the
bytes on the wire dropped from 561 MiB to 140 MiB per request. Details and caveats are below.

## 1. Setup

| Role | Hardware | Parallelism | Code |
|---|---|---|---|
| Prefill | 8x RTX PRO 5000 72GB (SM120), one eRDMA NIC | CEP4PP2: 2 pipeline stages (22 + 21 layers), each stage CP4 with EP4 MoE; KV replicated across CP ranks (`--prefill_cp_kv_cache_sharded 0`) | this branch + the internal Barex RDMA cache-store overlay (`internal_source`, branch `dev/hetero-pd-cache-store`) |
| Decode | 8x PPU | DP8, EP8, pipeline-stage-aware KV load | `dev/pd-load-stage-scoping` (`8c8a6c5`), unchanged |

Workload: DeepSeek-V4-Flash, ShareGPT prompts at exactly 32,768 tokens, 64 greedy output tokens, one request at
a time, prefix caching off on both sides (`--reuse_cache 0`, every request reported `reuse_len=0`).

Prefill writes KV into decode memory with RDMA WRITE through the internal Barex cache store. The open-source
`rtp_llm/cpp/disaggregate/cache_store/` code is the generic part; the RDMA transport is in the overlay.

### How KV reaches decode

There are two publish paths on prefill:

- **Per-layer streaming** (`CacheStoreAsyncWriter::write`): during every chunk, each layer's KV for that
  chunk's tokens is handed to the cache store as soon as the layer finishes. This overlaps the transfer with
  prefill compute. This path carries the data in normal operation.
- **End-of-request publish** (`PPExecutor::publishPdPlanRequests`): after the final chunk, each PP rank
  registers its stage's full block table with the cache store. Decode drops keys it already received, so in
  normal operation this sends no extra data; it is a safety net, and the only path in backlog mode (below).

Rank 0 also registers the request at the first token (`StreamCacheResource::publishKVCacheForPDSep`).

## 2. Changes

### 2.1 Publish once per rank (default on)

Before this branch the end-of-request publish ran after **every** chunk on most ranks (8 times for a 32K
request with 4K-token chunks), and each run walked the whole block table on the CPU. It added no network bytes
(decode reads each key once) but cost CPU time on the prefill critical path. Three bugs were involved:

- `PPExecutor::process`: CP rewrites `input_lengths` to the local shard before the "last chunk" check.
  The global lengths are now saved before `forwardPP` and passed to `publishPdPlanRequests`.
- `publishPdPlanRequests`: on PP worker ranks the plan's `seq_size_per_block` arrives as 1, which made every
  chunk look final. The check now uses the local `cache_config.seq_size_per_block` (identical on all ranks).
  This was the main cause.
- `publishPdPlanRequests`: stage 1 orders its cache groups differently from stage 0 (`swa_kv` first), while
  the plan's block table is laid out in PP-canonical order. Blocks are now looked up by
  `groupById(gid).canonical_idx`. Without this, backlog mode published the wrong blocks and decode timed out
  (`CACHE_STORE_LOAD_BUFFER_TIMEOUT`).

### 2.2 `RTP_LLM_PD_SERVING_CP_RANK=<r>` (opt-in, recommended)

With `prefill_cp_kv_cache_sharded=0` all 4 CP ranks of a stage hold the same KV. Decode loads each stage's
layers from every peer prefill advertises for that stage, so every block crossed the wire 4 times (all four
copies land at the same decode address). When set, prefill:

- advertises only CP rank `r` of each stage to decode (`PrefillRpcServer::buildAllocateRequest`,
  `fillStagePeerGroups`);
- publishes and streams only from that rank (`PPExecutor::publishPdPlanRequests`,
  `CacheStoreAsyncWriter::write`, using the new `KVCacheManager::stageCpRank()`).

Unset keeps the old behaviour. Decode needs no change. Any `r` in `[0, cp_size)` works; we used 0.

### 2.3 `RTP_LLM_PD_PUBLISH_AT_END=1` (benchmark only)

Skips per-layer streaming so the whole request's KV is published at the end. This reproduces the backlog a
prefix-cache hit or a busy prefill would create, and lets us measure how fast the transport drains it. Do not
use it in production (see the results: it delays token 2).

Both switches are read once in `rtp_llm/cpp/disaggregate/cache_store/PdTransferSwitches.h`.

### 2.4 Debug switches

- `RTP_LLM_PD_PUBLISH_DEBUG=1`: logs the last-chunk decision (`PD_PUBLISH_CHECK`) and per-tag block coverage
  of the final layer (`PD_PUBLISH_TAG`).
- `PREFILL_TRACE_LOG_ENABLE=1` (existing switch): also logs `PD_HANDOFF_TRACE` timestamps in
  `PrefillRpcServer` at first token written, decode "load done" received, GENERATE sent, and first decode
  output received.

### 2.5 Ring MoE transport (opt-in)

`rtp_llm/models_py/modules/dsv4/moe/_ring_ep_transport.py`, used by the `fork_nccl_mxfp8` EP strategy
(`strategies/nccl_ep_mxfp8.py`). On PCIe-only hosts it replaces the NCCL collectives inside each pipeline
stage with a copy-engine ring: rank r writes only into rank r+1's CUDA-IPC buffer with `cuMemcpyAsync`, and
flags are written and polled with `cuStreamWriteValue32` / `cuStreamWaitValue32` (no host sync per call).

| Env var | Values | Meaning |
|---|---|---|
| `DSV4_MOE_RING_DISPATCH` | `0` (default), `1` | ring all-gather of the MXFP8 dispatch packet |
| `DSV4_MOE_RING_COMBINE` | `nccl` (default), `mxfp8`, `bf16` | `nccl`: existing all-to-all return; `mxfp8`: ring reduce-scatter, MXFP8 requantized at each hop; `bf16`: ring reduce-scatter with BF16 running sums |
| `DSV4_MOE_RING_MAX_ROWS` | default `2048` | per-rank row cap of the ring buffers |

A call falls back to NCCL if ranks have different row counts, a chunk exceeds the row cap, or a CUDA graph is
being captured. Every rank decides from the same stage-wide counts, so all ranks take the same path.

## 3. Results

Hardware NIC counter for bytes, prefill server logs for timing. "Server TTFT" is the prefill server's
`first_token_cost_time`; "client TTFT" adds HTTP and tokenization. 6 requests per run, 2 runs per variant,
72 requests in total, all succeeded, and all produced the identical 64-token output.

| Variant | Server TTFT (ms) | Client TTFT (ms) | Token 1-2 gap (ms) | Bytes on wire per request | KV finished |
|---|---|---|---|---|---|
| Base (`3ffc6f3`) | 1189 | 1247 | 34.5 | 561 MiB | 9 ms before 1st token |
| 2.1 publish once | 1137 | 1193 | 35.1 | 561 MiB | 10 ms before |
| 2.1 + 2.2 (`SERVING_CP_RANK=0`) | **1125** | **1183** | **26.9** | **140 MiB** | 10 ms before |
| 2.1 + 2.3 (backlog) | 1118 | 1174 | 61.6 | 561 MiB | 24 ms after |
| 2.1 + 2.2 + 2.3 | 1109 | 1167 | 57.4 | 140 MiB | 24 ms after |

Notes:

- Runs of the same variant agree within about 1 ms (server TTFT).
- The base run used 16 output tokens instead of 64; this does not affect TTFT.
- 140 MiB on the wire is one copy of the 131 MiB of KV plus about 7% packet headers. The KV size matches a
  first-principles count for a 32K request (129 MiB) within 1.3%.
- Backlog mode lowers server TTFT by ~16 ms because prefill no longer does per-layer streaming work. But the
  KV then arrives after the first token, and decode cannot start until it has it, so token 2 is later.
  Keep streaming in production. The ~16 ms is likely the CPU cost of per-layer streaming on the prefill
  critical path (not yet proven).

Ring MoE (separate experiment, same PD setup, before the publish fixes, 16 requests per variant):

| Variant | Server TTFT (ms) | Change |
|---|---|---|
| NCCL (default) | 1189.6 | |
| ring dispatch + NCCL combine | 1183.1 | -6.4 (noise) |
| ring dispatch + ring MXFP8 combine | 1166.8 | **-22.8** |
| ring dispatch + ring BF16 combine | 1213.1 | +23.6 |

Ring MXFP8 combine is the only clear win. It requantizes at every hop, so it is slightly less accurate: on
captured real inputs the error vs an exact reference rises from 0.0435 to 0.0472.

### 3.1 Is the eRDMA link saturated?

No. Link limit from a VRAM-to-VRAM RDMA WRITE benchmark between the two hosts: 23.5 GB/s payload with 1 MiB
messages and 32 QPs (about 25 GB/s on the wire). It falls quickly for small messages and few QPs: 32 KiB
with 8 QPs gives 2.8 GB/s.

- **Streaming mode:** with duplicates removed the NIC averages 0.13 GB/s (0.3 GB/s peak over 10 ms). The
  send rate follows prefill compute, and the transfer is hidden before the first token.
- **Backlog mode (drain rate per stage, on the wire):**

  | Senders per stage | QPs per connection | Drain rate |
  |---|---|---|
  | 1 (2.2 on) | 8 | 3.1-3.5 GB/s |
  | 1, `CACHE_STORE_RDMA_WRITE_BLOCK_CONCAT=1` | 8 | 2.8-3.0 GB/s (no gain) |
  | 1 | 32 | 4.0-5.2 GB/s |
  | 4 (2.2 off) | 8 | 10.4-12.0 GB/s |
  | 4 (2.2 off) | 32 | 14.2-16.1 GB/s, 22 GB/s peak |

  The limit is per sender, not the link: four senders get about four times the rate. Each RDMA write is
  about 14 KiB (one block of one layer), which sits in the small-message regime of the link benchmark.
  Block concatenation cannot merge them because neighbouring blocks are not contiguous on both sides.

### 3.2 The token 1-2 gap

The gap is not KV transfer: the NIC sent 0 bytes in the 40 ms after the first token. With
`PD_HANDOFF_TRACE` on (2.1 + 2.2), prefill receives decode's "load done" 0.1 ms after it writes the first
token and sends GENERATE 0.1 ms later; all of the ~26 ms is between GENERATE leaving prefill and decode's
first token coming back. A normal decode step is 16.5 ms, so about 10 ms is extra work on the decode side for
the first step. It happens once per request. It has not been broken down further (needs decode-side tracing).

## 4. Measuring the KV transfer directly

`benchmark/hetero_pd/` contains the tools used above. All timestamps come from the prefill host, so the NIC
samples and the engine log share one clock.

| Tool | What it does |
|---|---|
| `nic_sampler.py` | Samples the eRDMA port's hardware `hw_tx_bytes_cnt` / `hw_rx_bytes_cnt` (`rdma -j statistic show link <dev>/<port>`) every 2 ms. Writes `<realtime> <monotonic> <tx> <rx> <call_ms>` per sample. |
| `kv_bw.py` | Per request: bytes sent, busy window, mean and 10-ms peak rate, bytes sent after the first token. Anchored on rank 0's `publish kv cache for pd sep` log line. Needs `RTP_LLM_PP_OBJ_LOG=1` for the round count. |
| `kv_rounds.py` | Bytes per prefill round (chunk) for one request, plus 10-ms buckets. |
| `burst.py` | Backlog mode: per-stage burst timeline and drain rate. |
| `handoff.py` | Prefill-side handoff breakdown from `PD_HANDOFF_TRACE` lines. |
| `runstat.py` | One line per run directory: TTFT, gap, TPOT, NIC bytes, drain rates, output agreement. |
| `pd_bench.py` | Streams one request at a time to the prefill server's native endpoint and records per-token arrival times and `output_ids`. |

Counter calibration: on a pure RDMA WRITE benchmark, wire bytes / payload bytes = 1.069 (headers), and the
counter-derived rate matched the benchmark's own rate within 0.3%. Time resolution is the 2-ms sample period,
so windows under ~20 ms are flagged.

Typical run (prefill host):

```bash
python3 nic_sampler.py run/nic.txt --dev erdma_0/1 --period-ms 2 --seconds 600 --stop-file run/stop &
for k in 1 2 3 4 5 6; do
  python3 pd_bench.py --endpoint 127.0.0.1:<prefill_http_port> --prompts sharegpt32k.jsonl \
      --out run/m$k.json --max-new 64 --warmup 0 --n 1
  sleep 3
done
touch run/stop; wait
python3 kv_bw.py run/nic.txt <prefill_log_dir>/engine.log --json-out run/kv_bw.json
```

Start prefill with `RTP_LLM_PP_OBJ_LOG=1` for `kv_bw.py`'s round count and `kv_rounds.py`, and with
`PREFILL_TRACE_LOG_ENABLE=1` for `handoff.py`. Warm up with 2 requests before sampling.

## 5. Reproducing

1. Check out this branch and add the internal Barex RDMA cache-store overlay as `internal_source`
   (branch `dev/hetero-pd-cache-store`; the repo's `.bazelrc` picks it up automatically).
2. Build for SM120 with CUDA 13:

   ```bash
   bazelisk build --config=cuda13_sm120 --define=use_remote_kv_cache=false \
       //:th_transformer //:th_transformer_config //:rtp_compute_ops //:th_grammar_tokenizer_info
   ```

   A clean build takes about 16 minutes on 64 cores; an incremental C++ change about 1-2 minutes.
3. Start decode from `dev/pd-load-stage-scoping` with `DSV4_PP_STAGE_LAYER_COUNTS=22,21` and
   `PREFILL_CP_SIZE=1` (with `PREFILL_CP_SIZE=4` the first request hangs with no error on either side).
4. Start prefill (CEP4PP2) with the arguments below, pointing `MODEL_SERVICE_CONFIG` at the decode endpoint.
5. Warm up with 2 requests, then run the measurement in section 4.

Prefill arguments that matter here (the rest follow the CEP4PP2 SM120 prefill recipe):

```bash
python3 -m rtp_llm.start_server --model_type deepseek_v4 --role_type PREFILL \
  --pp_size 2 --tp_size 4 --dp_size 1 --ep_size 4 --world_size 8 \
  --cp_rotate_method PREFILL_CP --prefill_cp_size 4 --prefill_cp_kv_cache_sharded 0 \
  --seq_size_per_block 256 --max_seq_len 32832 --fp8_kv_cache 1 \
  --cache_store_rdma_mode 1 --cache_store_rdma_connect_timeout_ms 800 \
  --reuse_cache 0 --enable_device_cache 0 --enable_memory_cache 0 --enable_remote_cache 0 \
  --concurrency_limit 1 --max_context_batch_size 1 --enable_cuda_graph 0
```

```bash
export DSV4_PP_EP_ENABLE=1 DSV4_PP_EP_BACKEND=fork_nccl_mxfp8 DSV4_PP_STAGE_LAYER_COUNTS=22,21
export DSV4_CHUNK_TOKENS=12288 RTP_LLM_PP_CHUNK_OVERLAP=2
export CACHE_STORE_RDMA_MODE=1 CACHE_STORE_RDMA_QP_COUNT_PER_CONNECTION=8   # must match decode
export ACCL_SET_ERDMA=1 ACCL_USE_NICS=erdma_0 RTP_CACHE_STORE_RDMA_DEVICES=erdma_0 RTP_CACHE_STORE_RDMA_ROLE=prefill
export RTP_LLM_PD_SERVING_CP_RANK=0                                          # 2.2
# optional: DSV4_MOE_RING_DISPATCH=1 DSV4_MOE_RING_COMBINE=mxfp8               # 2.5
```

## 6. Next steps

1. **Split each stage's KV across its 4 CP ranks.** Instead of one serving rank, each CP rank of a stage sends
   a different quarter of the blocks. The bytes stay at 1x, but there are four senders per stage, the setup
   that measured 14-16 GB/s with 32 QPs (vs 4-5 GB/s for one sender). A stage's backlog would drain in about
   5 ms instead of about 23 ms. This needs a decode-side change in `dev/pd-load-stage-scoping`: decode
   currently loads every block from every advertised peer, so it must instead request a disjoint key subset
   from each peer. It only helps when there is a backlog (prefix-cache hits, high prefill concurrency).
2. **Use 32 QPs per connection** (`CACHE_STORE_RDMA_QP_COUNT_PER_CONNECTION=32` on both prefill and decode;
   the two sides must match or the RDMA handshake fails). It raised the single-sender drain rate from
   3.1-3.5 to 4.0-5.2 GB/s and did not change streaming TTFT (1124.5 vs 1125.0 ms).
3. Make `RTP_LLM_PD_SERVING_CP_RANK=0` the default when KV is replicated across CP ranks.
4. Reduce the per-layer streaming cost (~16 ms), for example one cache-store call per chunk instead of one
   per layer.
5. Trace decode's first step to explain the ~10 ms in the token 1-2 gap.

## 7. Not verified

- Accuracy on a benchmark such as GSM8K. Correctness was checked by identical greedy outputs across all
  variants on one prompt, plus a short sanity prompt.
- Concurrency above 1.
- `hca_kv` is 1.4 MiB larger per request than the first-principles estimate; not explained.
