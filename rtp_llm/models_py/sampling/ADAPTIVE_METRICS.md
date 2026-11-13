# Adaptive verification monitoring

Metrics use the existing `RtpLLMSpeculativeEngineMetrics` / kmonitor path and
existing instance tags. Only TP rank 0 reports. All names below have the prefix
`rtp_llm_sp_`. They are per-round gauges, not cumulative counters.

| Suffix | Meaning |
| --- | --- |
| `verify_batch_size` | Requests in the measured verification round |
| `verify_target_rows` | Actual logical target verification rows, including one mandatory row per request |
| `verify_rows_per_request` | Target rows / requests; subtract 1 for average verified draft tokens |
| `verify_saved_row_ratio` | 1 - target rows / full-cap rows; logical row reduction, **not a measured speedup** |
| `verify_draft_accept_rate` | (Accepted output tokens - requests) / (target rows - requests); excludes mandatory/bonus token |
| `adaptive_enabled` | Whether this round used adaptive verification (includes fixed-cap diagnostics) |
| `adaptive_calibration` | 1 for cost calibration rounds, 0 otherwise |
| `adaptive_learned` | 1 for learned-budget decisions, 0 otherwise |
| `adaptive_confidence_fallback` | 1 when learned mode falls back to full cap because a request lacks fresh, valid confidence |
| `adaptive_zero_cap_ratio` | Fraction of requests verifying no draft tokens (mandatory target row still runs) |
| `adaptive_full_cap_ratio` | Fraction of requests verifying all configured draft tokens |
| `adaptive_plan_us` | Host plan construction/enqueue time in microseconds; excludes prediction kernels, feedback collection and TP broadcasts |

Verification metadata is staged alongside the same round's accepted-token
counts and reported on their next consumption, matching the existing acceptance
pipeline. Batch reorder/shrink does not mix current capacities with previous
acceptance. The last pending round may remain unreported when decode becomes
idle, as with the existing acceptance gauges. No new CUDA synchronization,
GPU reductions, or device-to-host copies are introduced for these metrics.
For CAP=0, draft acceptance is undefined and is not reported; do not fill it
forward as a current valid sample. Zero-cap ratio and target rows remain valid.
Fixed verification also reports row counts/reduction (zero reduction) with
`adaptive_enabled=0`. Fake/warmup paths do not produce decision samples.

## Evaluating benefit

Compare matching model, traffic, batch size, KV length and hardware windows:

1. Check `verify_rows_per_request`, `verify_saved_row_ratio`, and zero/full-cap
   ratios to see whether the policy actually reduces verification work.
2. Compare existing `rtp_llm_sp_avg_accept_token_num` with the new
   `verify_draft_accept_rate`. A higher rate alone can simply mean fewer tokens
   were attempted; it does not establish better output throughput.
3. Check existing `rtp_llm_generate_tps`, `rtp_llm_sp_step_latency_us`,
   `rtp_llm_sp_estimate_tpot_us` and request latency together. The existing step
   latency is host-side and TPOT is an estimate, not an exact GPU/client timer.
4. High calibration fraction suggests changing batch sizes prevent cost history
   from stabilizing. High confidence fallback suggests cold requests, stale or
   invalid prediction feedback. High `adaptive_plan_us` identifies host planning
   overhead; use a timeline to measure predictor, packing and graph-padding cost.

Gauges average rounds, not requests. Do not use `rate()` on these values or
interpret an unweighted average of ratios as a traffic-weighted acceptance
rate. Compare at instance level first; preserve batch-size context when
aggregating. Logical target rows can be padded by CUDA graph replay, so fewer
rows may not save GPU work. These metrics deliberately do not publish an
unverified predicted-speedup number.
