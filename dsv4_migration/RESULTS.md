# DeepSeek V4 Pro KV Offload Results

## Conclusion

The model download completed and the model runs locally on four GPUs. The CSA
offload implementation retains Indexer Key, HCA, SWA and compressor state on GPU,
uses the remaining configured GPU space for CSA residency, and reserves a
private 2048-entry cross-step CSA cache for every request and CSA layer.

At the same 12 GiB/GPU KV budget and 128K input length, vanilla fits the tested
B16 cohort but rejects B24/B32. Offload runs B32. Same-batch TPOT overhead is
1.4% at B1/B8 and 4.3% at B16. Relative to vanilla B16, offload B32 increases
decode goodput by 52.1% at a 45 ms TPOT SLO, while increasing TPOT by 31.4%.
The batch doubles, but latency is not unchanged.

## Equal-Budget Matrix

All inputs are distinct natural traces truncated to exactly 131072 tokens.
Each request generates 145 tokens. Exclude the first token and 16 warmup tokens;
measure 128 output tokens. This is one pass, not a confidence interval.

| Batch | Vanilla TPOT ms | Offload TPOT ms | Vanilla goodput token/s | Offload goodput token/s | Vanilla / offload response errors |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 18.48 | 18.74 | 54.10 | 53.35 | 0/1 / 0/1 |
| 8 | 24.05 | 24.38 | 331.86 | 325.92 | 0/8 / 0/8 |
| 16 | 29.32 | 30.57 | 545.18 | 522.92 | 0/16 / 0/16 |
| 24 | Capacity rejected | 34.76 | N/A | 689.59 | 24/24 / 0/24 |
| 32 | Capacity rejected | 38.53 | N/A | 829.36 | 32/32 / 0/32 |

The table uses **45 ms** as the per-request mean TPOT SLO. All completed requests
pass that SLO, so their decode goodput equals measured decode throughput.
Goodput is successful measured output tokens from SLO-passing requests divided
by the cohort's decode wall time. Prefill, queuing and TTFT are excluded.
The raw benchmark's 100 ms field is retained; the comparison script recomputes
45 ms and relative thresholds from individual request measurements.

If the SLO is vanilla B16 TPOT +20% (35.19 ms), offload B24 gives 689.59 token/s,
26.5% above vanilla B16. At +10% (32.25 ms), the original five-point matrix
does not show a goodput gain: B24/B32 miss that threshold, and offload B16 is
slightly slower than vanilla B16. A finer capacity split is evaluated separately.

Full machine-readable rows, per-request timing, token IDs, input manifests,
actual decode batch sizes and graph replay evidence are in
`results/vanilla-128k-12g-final` and `results/offload-128k-12g-final`.
The generated multi-SLO table is [results/comparison/TABLE.md](results/comparison/TABLE.md).

## Finer Capacity Split

An additional run uses 8 GiB for CSA GPU storage and 4 GiB for native mandatory
pools, still 12 GiB/GPU total. Maximum batch capacity, private 2K allocations,
fixed-state pools and input/output settings are unchanged. CSA residency grows
to 391808 entries per layer, approximately twelve 128K requests. Capture sizes
are exactly B18/B20. This split does not target B32 capacity.

| Scheme and split | Batch | TPOT ms | Goodput at 45 ms token/s | TPOT vs vanilla B16 | Goodput vs vanilla B16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vanilla | 16 | 29.32 | 545.18 | Reference | Reference |
| Offload 8/4 GiB | 18 | 30.82 | 583.49 | +5.1% | +7.0% |
| Offload 8/4 GiB | 20 | 32.36 | 616.03 | +10.4% | +13.0% |
| Offload 6/6 GiB | 24 | 34.76 | 689.59 | +18.5% | +26.5% |
| Offload 6/6 GiB | 32 | 38.53 | 829.36 | +31.4% | +52.1% |

At the strict reference +10% SLO (32.2534 ms), all B18 requests pass, giving
583.49 goodput (+7.0%). Only 6/20 B20 requests pass that threshold in this one
run, giving 184.81 goodput despite its higher total throughput. B20 is close to
the boundary; do not turn one run into a robust threshold guarantee. Under
+20%, B24 has the highest measured qualifying goodput, 689.59 token/s.

The input hashes agree with the original matrix. This is a configuration sweep,
not a same-batch causal comparison isolating the effect of the larger resident
prefix. Results and per-request SLO calculations are in
`results/offload-128k-12g-sweet` and
[results/comparison-sweet/TABLE.md](results/comparison-sweet/TABLE.md).

## Capacity and Memory

- Four NVIDIA L20D GPUs, approximately 267.69 GiB usable memory per GPU.
- TP4, EP4, DP1, CP1, MegaMoE SE, FP8 KV, CUDA Graph in both schemes.
- Model materialization consumes roughly 219 GiB/GPU. Actual 128K prefill adds
  about 20.2 GiB of PyTorch allocations. Both schemes reach about 263.5 GiB/GPU
  in sampled NVML memory, including reserved allocations and runtime overhead.
- KV budget: 12 GiB/GPU in both schemes. Offload reserves 6 GiB for CSA GPU
  resident/hot/metadata storage, leaving 6 GiB for native mandatory pools.
- Offload allocates 269440 resident compressed CSA entries per layer, enough
  for slightly over eight 128K requests. B16 and above exercise CPU-backed CSA
  requests. The private 2048 entries per request are additional to residency.
- Each CSA entry summarizes four original tokens. Pro selects 1024 compressed
  entries per decode step. The 2K hot capacity means compressed entries per CSA
  layer, not 2K raw input tokens and not a shared 2K capacity across all layers.
- CPU CSA backing has 65537 physical blocks per layer, approximately 68.6 GiB
  per GPU rank across the 30 CSA layers. CPU quota is not the tested limit.

Vanilla has 8297 usable paged blocks at this budget. Each block covers 256 input
tokens. A 128K request plus this decode needs 513 blocks: B16 needs 8208 and
B24 needs 12312. Fixed-state pools have separately been sized for B32.

**The B24/B32 vanilla errors are fixed-cohort admission rejections, not physical
CUDA OOM.** The benchmark requires the entire cohort to enter decode together
and returns an explicit error when that is impossible. An ordinary serving
scheduler can queue or reduce the batch instead. Error rate counts actual error
responses divided by attempts; a TPOT SLO miss never counts as a response error.
This experiment establishes the limit under the stated KV budget, not the
absolute maximum possible batch on this hardware.

A 42 GiB startup configuration did produce a real CUDA OOM during weight
repacking, but that is a bring-up failure and is not evidence for the final
runtime capacity comparison. The launcher now materializes weights before KV
allocation. No deliberate unhandled CUDA OOM is required to show insufficient
KV capacity.

## Where the Gain Comes From

At 128K, the paged attention caches per request per GPU use approximately
548.44 MiB for CSA, 123.75 MiB for Indexer Key and 26.16 MiB for HCA, excluding
the fixed SWA/compressor pools. CSA is about 78.5% of this paged footprint.
Moving its capacity to CPU is therefore enough to make substantially larger
decode cohorts fit, while the sparse attention still reads the same TopK.

The private 2K cache reuses entries selected on previous steps. Resident entries
need no fetch, and new misses start fetching once the layer's TopK is ready.
CUDA Graph removes repeated host launch work; it does not remove CPU traffic,
cache planning or larger-batch compute. The same-batch B16 difference of about
1.25 ms includes all these offload effects, and is not an isolated memcpy cost.

This design has a capacity and performance ceiling. Resident Indexer/HCA storage
still grows with context and batch, and private hot caches grow with batch. A
step with low cross-step hit rate needs more CPU traffic. As batch increases,
attention, Indexer, MoE and communication work also increase. The current data
supports capacity-driven throughput improvement, not unlimited batch growth at
unchanged TPOT. No real-model per-layer timing or hit-rate attribution is claimed.

## Correctness and Validation Limits

Native build and the GPU transfer/cache/TP4 regression suites pass. Coverage
includes MODEL1 payload/scales and padding, exact selected KV bytes, real CSA
compressor writes, private eviction, request reordering and allocator-page reuse,
deferred boundary writes, real FlashMLA parity and CUDA Graph cross-stream replay.
The final focused suite contains 18 passing tests, including corruption detection
and rejection of both separated prefill and decode roles.

B1 generates all 145 tokens identically between vanilla and both offload resident
splits. Whole-output exact matches are 3/8 at B8 and 7/16 at B16. A second vanilla
B8 run also matches only 3/8 complete outputs against the final vanilla B8 run.
The first generated token matches for every compared request; divergence appears during
free-running decode. The repeated vanilla run has different fixed-pool/reserve
settings and uncontrolled admission order, so the numerical cause is not isolated.
These observations do not establish whole-model quality equivalence.

An opt-in `DSV4_CSA_VALIDATE_BYTES=1` diagnostic compares every selected GPU
resident/hot entry against its authoritative CPU entry after compressor writes
and before attention, including during graph replay. Its real-model run is
stored separately in `results/offload-128k-byte-validation` and must not be used
as a performance sample. **The B16 real-model diagnostic passes:** all 16 inputs
have 131072 tokens, all generate 41 output tokens, and all 40 batched decode
steps run with validation enabled on all 30 CSA layers and all four GPU ranks.
No selected-byte assertion fails. This includes requests outside the resident
prefix, hot hits, new CPU fetches and compressed-boundary updates.

This checks 16 * 40 * 30 * 1024 = 19660800 selected entries per GPU during real
decode, excluding capture/warmup. Each entry checks 576 payload and eight scale
bytes. The check catches injected corruption in a separate regression test.
It provides direct cache-integrity evidence on this workload; it does not
resolve the independent source of free-running whole-output divergence.

The installed FlashMLA wheel does not support 32 local heads on this machine.
Both schemes pad to 64 independent heads and slice back to 32. TP embedding and
Q-workspace shape corrections are also shared by both schemes. These results
therefore compare the two cache policies on this functioning TP4 implementation;
they do not represent an optimized native H32 attention kernel.

No quality benchmark, mixed prefill/decode serving, prefix reuse, CP, PD or MTP
validation is claimed. Current offload explicitly requires FP8 KV, CP1, no PD,
no MTP, no prefix reuse and single-token decode.
