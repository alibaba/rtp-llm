# M890P D64: MTP depth 2

For this workload, use depth 2 instead of depth 3. Fresh alternating runs
reproduced the benefit. Both P and D were restarted with matching speculative
KV geometry; no quantization, model weights, sampler, or rejection kernel changed.
The deployment override is `d64_mtp_depth2.env`.

| Run | Depth | TPOT p50 (ms/token) | Output tokens/s | Successful requests |
| --- | ---: | ---: | ---: | ---: |
| r2_mtp_d3_a1 | 3 | 15.625471 | 4086.333 | 64/64 |
| r2_mtp_d2_b1r1 | 2 | 14.381173 | 4472.933 | 64/64 |
| r2_mtp_d3_a2 | 3 | 15.569523 | 4137.833 | 64/64 |
| r2_mtp_d2_b2 | 2 | 14.436468 | 4458.233 | 64/64 |

Mean TPOT: **15.597497 -> 14.408821 ms/token (-7.62%)**.
Mean output throughput: **4112.083 -> 4465.583 tokens/s (+8.60%)**.
These are means of two independent run metrics, not a pooled token statistic.

## Reproduction and boundaries

- Source baseline: RTP-LLM `7dda4bb2602ffd063e658e2de639793f7003f111`,
  DeepGEMM `f94750b2f765875e58dbfcb5d088e00930dfe58c`.
- Model: DeepSeek-V4-Flash. Decode: eight M890P devices, TP1/DP8/EP8,
  eight requests per rank, retained two-stage MegaMoE.
- The existing decode fusions, qualified attention tiles, C4 ordered queries,
  and MTP input fusion were enabled. HC POST tiling and staged shared experts
  remained disabled in every run.
- Identical 64 unique inputs of 32768 tokens, no KV reuse, 12288 output tokens
  per request. Pause decode until all 64 inputs are prefetched, then release.
- Measure a fixed 30-second window after a two-second D64 guard. Normalize each
  request's streaming interval by the actual number of emitted tokens, then take
  the median across requests. Output throughput counts actual arrivals in that
  window. Profiling happens after measurement.
- Context ceiling is 43008. Every run passed the D64 window and all-request
  completion checks. Return to depth 3 reproduced the earlier result, reducing
  concern about simple time drift.
- The first depth-2 attempt (`r2_mtp_d2_b1`) failed in warmup with KV-connection
  error 8302, before the main workload. It is excluded, not reported as a timing
  result. A retry on the same depth-2 service passed. The helper now permits four
  warmup attempts for that specific connection error; main-workload policy is
  unchanged.

This qualifies a workload-specific speculation setting, not a new numerical
kernel. Functional completion is not a full model-quality evaluation. Generated
text identity across separate stochastic service runs was not used as a precision
oracle. Neither numerical precision versus NVIDIA nor a universal best depth for
other inputs/concurrency is established here.
