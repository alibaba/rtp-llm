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
| offload | 1 | 18.74 | 53.35 | 53.35 | 53.35 | 53.35 | 0% | 263.46 | verified |
| offload | 8 | 24.38 | 325.92 | 325.92 | 325.92 | 325.92 | 0% | 263.46 | verified |
| offload | 16 | 30.57 | 522.92 | 522.92 | 522.92 | 522.92 | 0% | 263.46 | verified |
| offload | 24 | 34.76 | 689.59 | 689.59 | 0.00 | 689.59 | 0% | 263.46 | verified |
| offload | 32 | 38.53 | 829.36 | 829.36 | 0.00 | 0.00 | 0% | 263.46 | verified |

Errors count benchmark response failures, never SLO misses. Fixed-cohort capacity rejection is an admission error imposed by this benchmark; normal serving can queue or shrink a batch. It is not physical CUDA OOM.

Goodput excludes prefill: successful measured output tokens from requests meeting the per-request mean TPOT SLO, divided by decode wall time. These results do not measure mixed prefill/decode interference or end-to-end serving goodput.

## Output Parity

- B1: identical complete outputs for 1/1 requests; 145 tokens compared.
- B8: identical complete outputs for 3/8 requests; 1160 tokens compared.
- B16: identical complete outputs for 7/16 requests; 2320 tokens compared.

Vanilla-versus-vanilla output reference (not additional performance samples):

- B1: 1/1 identical complete outputs.
- B8: 3/8 identical complete outputs.

Free-running greedy outputs also diverge between vanilla runs. This reference used different fixed-pool/reserve settings and uncontrolled request admission order; it does not isolate a numerical cause. Byte-preserving cache tests are independent evidence, not proof of whole-model quality equivalence.
