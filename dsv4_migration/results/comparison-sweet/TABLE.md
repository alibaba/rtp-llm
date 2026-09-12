# Measured Decode Results

Reference: largest successful vanilla batch B16, TPOT 29.32 ms.
Goodput thresholds: {"45_ms": 45.0, "baseline_plus_10pct": 32.25343757585506, "baseline_plus_20pct": 35.185568264569156}

| Scheme | Batch | TPOT ms | Token/s | Goodput @45ms | Goodput @ref+10% | Goodput @ref+20% | Error rate | Peak GPU GiB | Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla | 1 | 18.48 | 54.10 | 54.10 | 54.10 | 54.10 | 0% | 263.55 | verified |
| vanilla | 8 | 24.05 | 331.86 | 331.86 | 331.86 | 331.86 | 0% | 263.55 | verified |
| vanilla | 16 | 29.32 | 545.18 | 545.18 | 545.18 | 545.18 | 0% | 263.55 | verified |
| vanilla | 24 | - | 0.00 | 0.00 | 0.00 | 0.00 | 100% | 263.55 | - |
| vanilla | 32 | - | 0.00 | 0.00 | 0.00 | 0.00 | 100% | 263.55 | - |
| offload | 18 | 30.82 | 583.49 | 583.49 | 583.49 | 583.49 | 0% | 263.34 | verified |
| offload | 20 | 32.36 | 616.03 | 616.03 | 184.81 | 616.03 | 0% | 263.34 | verified |

Errors count benchmark response failures, never SLO misses. Fixed-cohort capacity rejection is an admission error imposed by this benchmark; normal serving can queue or shrink a batch. It is not physical CUDA OOM.

Goodput excludes prefill: successful measured output tokens from requests meeting the per-request mean TPOT SLO, divided by decode wall time. These results do not measure mixed prefill/decode interference or end-to-end serving goodput.

## Output Parity
