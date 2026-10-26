# M890P decode projection tiles

The candidate passes same-input numerical checks, but its end-to-end benefit
did not reproduce. Keep `DSV4_PPU_DECODE_PROJECTION_TILES` disabled by default.
The implementation is retained as an opt-in experiment for further profiling.

It selects dense FP8 GEMM tiles for merged QKV, shared-expert W13/W2, Indexer Q,
and MTP E/H projections at M=8/24/32 on the 39-SM M890P. Other shapes and SM
counts retain the existing attention-tile or library fallback. Quantization,
weight formats, K traversal, and the BF16 output boundary are unchanged.

## Numerical checks

The isolated sweep uses real checkpoint weights, six input magnitudes, and the
same E4M3 payload / FP32 scales for candidate and baseline. The 18 selected
operator/shape combinations passed all 108 byte-equality checks. Their measured
operator latency improvements range from 3.2% to 10.4%. Measurements use a
256 MiB cache flush outside each one-operation CUDA graph, three alternating
rounds of 30 samples; these are not service speedups.

The instrumented service then compared each changed projection against the
library kernel on identical live input and weight tensors. The 64-request,
32768-token-input, 384-token-output precision run completed successfully:

| Projection shape (N, K) | Module instances across 8 ranks | Comparisons | Unequal bytes |
| --- | ---: | ---: | ---: |
| 1536, 4096 | 352 | 61468 | 0 |
| 4096, 4096 | 368 | 64294 | 0 |
| 4096, 2048 | 352 | 61468 | 0 |
| 8192, 1024 | 168 | 29337 | 0 |

Total: **1240 module instances, 216567 comparisons, 40265498624 compared bytes,
zero mismatches**, with no instrumentation errors on any rank. The live target
path used M=24; draft execution also exercised M=8 and M=32. This validates the
changed operator boundary on the exercised workload, not full model quality or
precision parity with NVIDIA.

## End-to-end results

All runs use DeepSeek-V4-Flash, eight M890P decode devices, TP1/DP8/EP8, MTP
depth 2 on both P and D, and the retained two-stage MegaMoE configuration.
The workload and metric definitions are those in `mtp_depth2.md`: 64 unique
32768-token inputs, 12288 output tokens, no KV reuse, a two-second D64 guard,
and a fixed 30-second measurement window. Shadow hooks are off. Profiling is
outside the measured window. All three runs completed 64/64 requests and
passed the full-concurrency window checks.

| Run | Tiles | TPOT p50 (ms/token) | Output tokens/s | Chunk interval p50 (ms) |
| --- | --- | ---: | ---: | ---: |
| r2_gemm_b1, earlier session | On | 14.088061 | 4566.867 | 29.691949 |
| r2_gemm_a_return, fresh control | Off | 14.279804 | 4498.833 | 30.328136 |
| r2_gemm_b2, fresh candidate | On | 14.403774 | 4489.800 | 30.339566 |

The fresh A/B comparison is **+0.87% TPOT / -0.20% output throughput**, with
essentially unchanged chunk interval. The earlier candidate run is retained
as a separate observation, not pooled with the fresh pair across an overnight
interruption. The data does not establish a repeatable service improvement or
a statistically significant regression. The candidate is therefore not adopted
as the next performance baseline.

## Provenance

RTP-LLM source parent: `c2a8029f7512f06d2e1dc72be6cb91f07f987817`.
This parent adds reports on top of runtime source
`7dda4bb2602ffd063e658e2de639793f7003f111`; the tested candidate adds the three
projection-tile implementation files in this commit. The serving overlay also
includes the matching shared-expert / EP / MegaMoE interfaces from that runtime
source, with staged shared experts disabled. The existing installed DeepGEMM
dense-GEMM kernel is unchanged by this item.

`decode_projection_tiles_results.json` contains selected operator measurements,
aggregated precision counters, service metrics, and SHA-256 hashes of the
underlying evidence. The committed implementation was compared to the tested
serving overlay using Python AST equality; formatting changes do not change
its behavior. Setup failures before successful service startup are excluded
from performance results and remain in the local experiment record.
