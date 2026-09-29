# Eager MLA metadata reservation candidate, 2026-09-29

Status: **diagnostic candidate only**. The patch is stored here for recovery and is not applied to the integration branch. It does not constitute the Prefill performance anchor.

`KimiK3MlaVerifyImpl` reserves metadata capacity for the longest CUDA Graph replay during construction, then calls `prepare(inputs)` for the actual request. An eager Prefill has no replay and pays for both `fill_params` calls. The candidate keeps the reservation for CUDA Graph and omits it for eager Prefill. On 113, 100 paired isolated metadata measurements after ten warmups per path had median host times of 0.508 ms with reservation and 0.266 ms without it. This is a metadata microbenchmark, not a model latency result.

The candidate passed its two focused tests, Python compilation, and an independently checked 11-case four-layer FP8+Native MTP PD flow on 114 Prefill and 115 Decode. The 64K timeline used the same hosts, target checkpoint, request token IDs, 12 stable warmups, 16 profiled HTTP 200 responses, and all eight Prefill rank traces as the baseline. Each request ran target and draft, used no Prefill cache reuse, and handed 61,440 tokens to Decode. All correlation audits had zero missing or ambiguous launches. Both candidate profile passes used the same live service; only the 114 Prefill source differed from the baseline. Communication remained BF16 NCCL. Weights were loaded directly from 3FS through FastSafetensors with 64 task-local pread workers.

| Run | Prefill target + draft GPU span | Target GPU span | Draft GPU span |
| --- | ---: | ---: | ---: |
| Baseline `r2` | 134.847 ms | 71.611 ms | 63.387 ms |
| Candidate `r8` | 134.544 ms | 71.460 ms | 63.222 ms |
| Candidate repeat `r8b` | 137.900 ms | 71.686 ms | 63.597 ms |

Values are medians of eight profiled requests, taking the slowest rank for each request. The first candidate pass improved the complete Prefill span by 0.303 ms; the repeat did not confirm that small gain. Two requests in `r8b` took about 196 ms. Their draft embedding BF16 AllGather had about 63–64 ms GPU duration on several ranks versus a normal 1–3 ms. On rank 5, the corresponding `c10d::_allgather_base_` CPU scope lasted 62.6 ms before its GPU kernel launched. Other ranks waited in the collective. This is a measured rank delay; its cause remains unknown, and the MLA metadata change does not touch that collective. Two further `r8b` target requests were around 76.6–76.9 ms, versus the usual 71–72 ms. The baseline also contained smaller outliers. No samples were removed from the table.

The data supports a local metadata saving but does not yet establish a reproducible model-level win. Recheck under monitored, exclusive conditions before selecting this patch. The existing `r2` baseline raw trace is in `../integrated-r2-114115.tar.gz`; the two archives here contain all candidate traces, responses, and audits. `eager-reserve-114115-phase-ab-20260929.json` contains per-rank, per-request timing and launch attribution. Four-layer generated text is not a semantic correctness check for the full 93-layer model.
