# TokenSpeed MLA backend setup cache: four-layer candidate

This branch is a recoverable performance candidate, **not** the final performance anchor. The only model change caches the TokenSpeed import, CUTLASS path check, and installed package version once per process. It preserves the existing FP8 and BF16 operand selection and adds no precision switch. The separate eager MLA reservation candidate was restored to baseline before this run.

The focused test failed on the old repeated setup, then passed after the change. In the same 115 runtime container, 100 paired hot constructor calls had median host time of 390.49 µs before and 1.91 µs after the cache. This small constructor probe does not establish model speed. The change also passed Python compilation and an independently audited 11-case four-layer FP8+Native MTP PD flow on 114 Prefill / 115 Decode. The flow audit confirmed PD cache handoff and draft execution. It counted one Unicode replacement character in one random four-layer output, so the flow is **not** a semantic or clean-text acceptance result. An earlier baseline four-layer flow also generated different text between a deterministic cache miss/hit pair. Full 93-layer answer checks remain necessary.

The same-host 64K timeline used TP8/EP8, direct 3FS FastSafetensors, task-local 64-thread pread, BF16 NCCL, FP8 target, BF16 MTP, and TokenSpeed Prefill. Each pass had 12 stable warmups, 16 HTTP 200 profiled requests, eight rank traces, no Prefill cache reuse, and a 61,440-token PD handoff. Independent audits checked actual submitted token IDs, response hashes, MTP draft rounds, and zero replacement characters in these fixed eight-token responses. Launch correlation found zero missing or ambiguous kernels. The two pass selector snapshots recorded no external GPU process on either host immediately before profiling.

| Four-layer run | Target + draft Prefill GPU span | Target GPU span | Draft GPU span |
| --- | ---: | ---: | ---: |
| Same-host integrated baseline `r2` | 134.847 ms | 71.611 ms | 63.387 ms |
| Backend cache `r9` | 133.811 ms | 70.911 ms | 62.894 ms |
| Backend cache repeat `r9b` | 133.293 ms | 70.804 ms | 62.628 ms |

Each number is the median of eight profiled requests after taking the slowest rank for each request. `r9` had three NCCL outliers, including one target span near 961 ms; no samples were removed. `r9b` ranged from 132.990 to 134.125 ms for the complete Prefill span and had no such large spike. The repeat supports a roughly 1.0–1.6 ms same-host Prefill improvement over the earlier baseline. The fixed `feat/k3_dev@a9bf762e` same-host record reported 132.082 ms for target+draft, but it computes 65,533 rather than 65,536 tokens and uses some FP8 value/scale communication, whereas this candidate keeps BF16 NCCL. The integrated version has **not** established a whole-Prefill win against `feat`.

`backend-cache-r9-114115-raw-20260929.tar.gz` contains the first full-rank timeline, all warmup/profiled responses, and four-layer flow audit. `backend-cache-r9b-114115-raw-20260929.tar.gz` contains the repeat. The phase JSON preserves every rank and request. The exact baseline trace is in `../20260929_3fs_prefill_ab/integrated-r2-114115.tar.gz`. A full 93-layer FP8 PD smoke and final three-way full-model timeline have not been run on this candidate.
