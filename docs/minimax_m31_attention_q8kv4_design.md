# MiniMax-M3.1 Attention Q8KV4 phase-1 design (historical)

Status: superseded by
`minimax_m31_nvfp4_native_kernel_design.md`. This document records the retired
FP4-persistent/BF16-working-page phase and its historical evidence; it does not
describe the current runtime. The current MiniMax-M3.1 path requires native
packed-FP4 prefill, decode, and target verification, with no BF16 working-page
service fallback. Requirements below that say otherwise are intentionally kept
only as phase-1 history.

## 1. Scope and delivery boundary

Phase 1 adds MiniMax-M3.1 persistent KV4 storage for main K/V and MSA
indexer-K, while retaining the existing BF16 attention kernels through transient
working pages. It covers standalone and PD-separated execution, including scale
sidecar transfer, and integrates the same storage contract into DSpARK commit and
query-block writes.

This phase does not make the experimental native Q8 x KV4 score/value kernel
part of the default path and does not claim an attention speedup. Routed-MoE
NVFP4 is supplied independently by the `mega_moe_nvfp4` implementation already
present on the M3.1 branch; both BF16-KV and KV4 validation use that identical
MoE path so the cache A/B changes only KV/indexer-K storage. The expected first
KV4 benefit is larger persistent KV capacity; quantize/dequantize and BF16
working pages add runtime cost that must be measured separately.

The real M3.1 checkpoint has 60 sparse-MSA layers (`sparse_attention_freq` is
all ones). Phase 1 is deliberately limited to that all-sparse model. Enabling
NVFP4 for a model containing `CausalAttention` or a dense-attention layer fails
during configuration; it must not allocate a full-pool BF16 mirror.

The Q path remains unchanged: the main-attention query and indexer query use the
existing scale-1 E4M3 cast. The new code owns only persistent K/V/indexer-K
representation and its adapters.

## 2. Numeric contract

For every contiguous group of 16 BF16 values:

1. Compute `amax = max(abs(x[0:16]))` in FP32.
2. Compute `scale_fp32 = clamp(amax / 6, 1/512, 448)`.
3. Store `scale_e4m3 = cast_e4m3(scale_fp32)`.
4. Normalize with the value that is actually stored:
   `normalized = x / bf16_or_fp32(scale_e4m3)`.
5. Encode E2M1 using the explicit RNE ladder. The CUDA/Triton implementation
   compares `abs(x)` against `threshold * stored_scale` in the original value
   domain; it must not divide first, because reciprocal/division rounding can
   move an exact midpoint such as 2.5 slightly above the strict threshold.
6. Suppress negative zero: when the magnitude code is zero, the sign bit is
   always zero.
7. Pack the lower-dimension even value into the low nibble and the following odd
   value into the high nibble.

The RNE decision ladder is:

| Test, evaluated top-down | Magnitude code | Decoded value |
| --- | ---: | ---: |
| `a > 5.0` | 7 | 6.0 |
| `a >= 3.5` | 6 | 4.0 |
| `a > 2.5` | 5 | 3.0 |
| `a >= 1.75` | 4 | 2.0 |
| `a > 1.25` | 3 | 1.5 |
| `a >= 0.75` | 2 | 1.0 |
| `a > 0.25` | 1 | 0.5 |
| otherwise | 0 | 0.0 |

There is no tensor-level/outer scale. Dequantization is exactly
`decode_e2m1(code) * scale_e4m3`.

Clamping before the E4M3 cast is intentional. Casting an out-of-range FP32 scale
before clamping can create a non-finite or zero encoded value. Quantization must
divide by the rounded E4M3 value, not the pre-cast FP32 candidate, so CPU,
Triton, CUDA and checkpoint/QAT semantics agree.

## Runtime validation contract and known PD trap

The first production-like validation topology is Prefill CP2 + KV-sharded +
prefix-cache enabled, Decode DP2 + CUDA Graph capture. `tp_size` is normally
1 in this topology: CP owns the prefill workers and is not represented by the
attention TP width. Therefore the PD allocation protobuf must carry the
logical `prefill_cp_size` from `prefill_cp_config`; deriving it from
`parallelism_config.tp_size` silently sends zero for the common CP2/TP1 case.
Decode then cannot reconstruct page-RR ownership for the two opaque KV/scale
planes. This propagation is a correctness requirement, not an optimization.

For every PD run, validate these boundaries independently: (1) Prefill sends
`prefill_cp_size=2` and both peer cache-store addresses, (2) Decode receives
the same CP size and builds page-RR load requests, (3) the opaque block has
`kv` plus `kv_scale` parts with nonzero sizes, and (4) the Decode first-token
response is written back to the Prefill stream. A `worker_status` entry marked
`finished` only proves an RPC stage was retired; it is not proof that the HTTP
stream received output.

The phase-1 fallback intentionally uses BF16 working pages. It must not be
described as a native FP4 attention implementation. The implemented phase-1
adapter gathers every page referenced by the bounded request block table for
both indexer-K and main K/V, then runs the existing sparse decode kernels over
the remapped BF16 pages. This is bounded by request capacity rather than global
cache-pool capacity, but it still performs more main-K/V conversion than the
performance target. The follow-up target is a two-stage reader: gather all
visible indexer-K rows for top-k selection, then gather only main K/V blocks
selected by that top-k result. Scratch capacity in both stages must remain
graph-stable and bucketed.

## 3. Persistent per-layer block layout

Both value and side planes are raw byte-addressed tensors. Dimensions must be
positive multiples of 16.

`kv_cache_base[physical_block]` contains:

```text
[packed K for all heads/pages][packed V for all heads/pages]
```

`kv_scale_base[physical_block]` contains:

```text
[K E4M3 scales][V E4M3 scales]
[packed indexer-K values][indexer-K E4M3 scales]
```

For `H` local KV heads, physical page size `P`, main head dimension `D`, and
indexer dimension `I`:

```text
main packed bytes/block = 2 * H * P * D / 2
main scale bytes/block  = 2 * H * P * D / 16
index packed bytes      = P * I / 2
index scale bytes       = P * I / 16
side bytes/block        = main scale bytes + index packed bytes + index scale bytes
```

Main K/V plus their scales therefore consume 9/16 byte per original scalar,
28.125% of BF16 storage. Indexer-K has the same ratio. C++ cache sizing is the
single source of truth; Python validates exact byte extents before creating
views and must never infer the side offsets from tensor dtype.

The layout uses a dedicated `nvfp4_kv_cache` capability flag and raw byte dtype.
It must not masquerade as `KvCacheDataType::FP8`, because the byte geometry,
scale addressing and supported readers differ.

## 4. Configuration and compatibility

- Add `NVFP4_KV_CACHE=0/1` to `KVCacheConfig`, `AttentionConfigs`, pybind and
  server arguments.
- `INT8_KV_CACHE`, `FP8_KV_CACHE` and `NVFP4_KV_CACHE` are mutually exclusive.
- NVFP4 persistent storage keeps `kv_cache_dtype=BASE` so existing attention
  selection chooses BF16 operators after materialization.
- MiniMax MSA requires paged indexer-K. Enabling NVFP4 without a positive
  `indexer_head_dim` fails during configuration.
- MSA main K/V and indexer-K must switch together; mixed KV4/indexer-BF16 or
  BF16/indexer-K4 layouts fail before allocation.
- Phase 1 accepts only MiniMax-M3.1 all-sparse MSA with attention TP size 1.
  Prefill CP4 still satisfies this because CP owns the raw TP dimension while
  `get_attn_tp_size()==1`; decode DP4 also has attention TP 1 per DP rank.
- An attention-TP value greater than one fails before cache allocation. Supporting
  it requires a segmented sidecar descriptor because main scales are per head
  while indexer values/scales are replicated.
- Legacy BF16/FP8/INT8 paths remain byte-for-byte unchanged when the flag is off.
- DSpARK inherits target NVFP4 when `SP_FP8_KV_CACHE=-1`. The existing explicit
  draft override can select BF16 (`0`) or FP8 (`1`). Phase 1 does not add another
  speculative flag unless the production launch requires target BF16 with draft
  KV4, which the requested M3.1 deployment does not require.
- Ordinary decode and target verification both materialize BF16 working pages.
  Native packed-cache readers are intentionally excluded from the phase-1
  delivery and must land as a separately reviewed and validated change.

## 5. Cache allocation and Python views

C++ changes:

1. `MHAKVCacheSpec` reports half-byte packed K/V sizes and one E4M3 scale byte
   per 16 values.
2. `IndexerCacheLayout` adds internal mode 3:
   `I/2 + I/16` bytes per token.
3. `SingleConfigCreator` allocates the main packed plane plus one opaque side
   plane using the formulas in section 3.
4. `MemoryLayoutStrategy` exposes both planes as `uint8` without attempting a
   FP32 scale reshape.
5. `LayerKVCache.nvfp4` prevents the raw value plane from being reshaped as a
   normal `[block, 2, head, page, dim]` tensor.

Python `cache_layout()` validates dtype, rank, contiguity, block counts,
group-16 dimensions and exact minimum strides. It returns typed views but does
not allocate persistent storage.

## 6. Writer paths

One slot-based writer is shared by ordinary target attention and DSpARK:

- `quantize_main_rows(k, v, physical_slots, layout)` writes packed K/V and main
  scales.
- `quantize_index_rows(index_k, physical_slots, layout)` writes packed index-K
  and index scales.
- Invalid/padded slot `-1` is masked and must not modify persistent storage.
- Rewriting an existing slot overwrites both packed values and its scale in the
  same invocation contract. This is required for speculative reject/rollback
  self-healing.

Call sites:

- regular prefill suffix writes;
- regular decode writes;
- M3.1 DSpARK `commit_feature_rows()`;
- DSpARK non-causal query-block writes before index selection and sparse
  attention.

Tests cover first/last slot, cross-page rows, unordered slots, duplicate-slot
last-writer behavior where supported, padded slots and overwrite after reject.

## 7. BF16 reader and working-page lifecycle

There is intentionally no native NVFP4 attention reader in phase 1.

- Row gather unpacks nibbles, loads the matching E4M3 scale and writes BF16.
- Page conversion materializes the bounded request block table, including its
  graph-bucket padding, into BF16 HND working pages. It never mirrors the global
  cache pool, but phase 1 does not yet delay main-K/V conversion until after
  sparse top-k selection.
- Indexer-K gather materializes only slots visible to the current request into
  BF16 scratch before score/top-k.
- Existing dense, sparse prefill and sparse decode kernels then run unchanged.

Working buffers are transient, never published through PD, and are not counted
as reusable prefix cache. Their addresses and maximum capacities must be stable
for CUDA Graph replay; replay updates live slot/page metadata in place. Tail rows
outside the live sequence are cleared or masked before an attention kernel can
observe them.

For CP4 page-RR/sharded prefill, physical-slot mapping happens before the KV4
writer. Each rank quantizes only locally owned slots. Restoring a prefix uses the
following explicit path instead of applying token-major gather code to packed
bytes:

1. gather the rank-owned raw `kv_cache_base` block rows according to the existing
   CP prefix gather plan;
2. gather the matching complete `kv_scale_base` rows with the identical plan;
3. restore logical page order using the plan's restore indices;
4. create a temporary validated KV4 layout over the gathered raw value/side
   rows;
5. dequantize those rows into the existing logical BF16 HND working pages and
   BF16 indexer scratch.

The suffix path continues to write only locally owned physical slots and fills
working pages from the already all-gathered suffix activations. Compact CP
prefill remains fail-closed until it has an explicit KV4 reader; silently falling
back inside a capture is not allowed.

Decode and target verify use a bounded compact working-page
adapter rather than a BF16 mirror of the global cache pool. For a request block
table shaped `[B, M]`, it allocates/prewarms at most `B_bucket * M` BF16 pages,
dequantizes each referenced physical page into its deterministic row-major
working page, and replaces valid table entries with `0..B_bucket*M-1`. Both the
main-attention table and index-score table use the remapped IDs. Invalid/padded
entries are sourced from the framework's immutable zero/sentinel page and are
also assigned deterministic working-page IDs; no `unique()`, host sync or
shape-dependent allocation is permitted during graph replay. Warmup/capture
uses exact-shape, per-bucket workspace objects so growing a later bucket cannot
invalidate pointers captured by an earlier CUDA Graph.

The current M3.1 DSpARK draft implementation has the same storage writer hooks,
but target-NVFP4 plus mock-draft PD is a separate E2E gate. Until that gate
passes on the final built source, this document does not claim that DSpARK
query-block attention has validated the NVFP4 reader lifecycle.

## 8. PD-separated transfer contract

PD transports persistent bytes, never BF16 working pages. MiniMax MSA uses
opaque whole-block cache-store transfer so the same block key carries:

```text
kv_       = packed K + packed V for all participating layers
kv_scale_ = main K/V E4M3 scales + packed indexer-K + indexer-K E4M3 scales
```

Required invariants:

1. Prefill and decode derive identical `kv_block_stride_bytes` and
   `kv_scale_stride_bytes` from the same model/page/head/index dimensions.
2. `use_opaque_kv_cache_store=true`; neither value nor side plane is split by
   K/V byte halves.
3. `scale_region_is_head_partitioned=false`; the opaque side plane cannot be
   sliced using main-KV head geometry.
4. Cache-store descriptors and RPC metadata carry the full side-plane byte
   count. A missing/short scale block is a hard load failure.
5. Publication completes only after both planes are visible. Decode may build
   BF16 pages only after the cache-load completion event.
6. Target and DSpARK draft cache publications use their own cache configs and
   strides; target and draft precision are never inferred from one another.
7. Phase 1 requires attention TP 1 on both P and D. Raw Prefill CP may be 4 and
   Decode DP may be 4; page-RR chooses which P peer owns a complete opaque page,
   and Decode loads that complete value+side page. Arbitrary attention-TP
   resharding is rejected rather than partially copying the mixed sidecar.

Add a byte-pattern integration test that writes distinct sentinels into packed
K, packed V, main scales, packed index-K and index scales on the prefill side,
passes them through the real cache-store block descriptors, and checks every
region on the decode side. Add a second test with corrupted/short side bytes and
require deterministic failure rather than partial decode.

## 9. DSpARK-specific behavior

- `MiniMaxM31DSparkModel.commit_feature_rows()` reuses the normal KV4 slot
  writer after K/V/index-K projection; it does not implement a second codec.
- When the draft cache itself is configured as KV4, query-block attention must
  persist generated K/V/index-K in KV4 and consume a remapped working table for
  the visible prefix plus the entire non-causal query block. The first final
  validation uses target KV4 with a BF16 mock draft, so this draft-KV4 branch is
  a design requirement rather than evidence already established by that smoke.
- Proposal rejection may rewrite the same physical slots on the next round;
  packed bytes and scales must both be replaced.
- PREFILL-role commit-only publication includes draft value and side planes.
- The first implementation remains static-width and retains the current
  role-aware CUDA Graph policy.

## 10. Migration plan

Use `origin/feat/m3_nvfp4@e473ad313c` as the codec/layout baseline because the
current branch shares its parent. Apply the feature commit as a three-way
starting point, then review every hunk against M3.1/DSpARK. Remove the generic
`CausalAttention` full-pool BF16 workspace path from this delivery; retain only
the all-sparse M3.1 path. Add CP-sharded, compact decode/DSpARK working-page and
PD tests rather than treating a clean cherry-pick as completion.

Implementation order:

1. codec reference and GPU pack/depack tests;
2. config, C++ sizing and raw-byte view plumbing;
3. ordinary M3.1 target writer/reader;
4. CP4 non-compact working-page path;
5. DSpARK commit/query-block integration;
6. PD opaque transfer including side-plane tests;
7. CUDA Graph exact/padded working-buffer tests;
8. build/install, path proof, PD E2E, quality, stability and performance.

## 11. Requirement-to-test matrix

| Requirement | Minimum proof |
| --- | --- |
| RNE/tie/zero/nibble contract | CPU golden plus Triton equality at every midpoint, adjacent values, zeros, saturation and random groups |
| Layout sizing | C++ tests for D=128/256, I=128, page 64/128, invalid non-multiple-of-16 dimensions |
| Slot writer | first/last/cross-page/unordered/padded/overwrite cases for main and indexer planes |
| BF16 reader | dequantized rows/pages against CPU reference with explicit error tolerance |
| Legacy compatibility | feature-off BF16 and FP8 unit/E2E paths unchanged |
| CP4 | local ownership, page-RR slots, ragged suffix, padding, long prefix; compact mode rejects explicitly |
| DSpARK | commit, query-block future visibility, reject overwrite, full accept, padding and draft-prefill publication |
| PD bytes | sentinel transfer for all five regions and short-side failure test |
| CUDA Graph | eager, exact bucket, live 3/capture 4, live 5/capture 8, drain to batch 1, stable workspace addresses |
| Quality | same IDs and settings, BF16 versus KV4 deterministic smoke plus requested task/long-context evaluation |
| Stability | three production-topology pressure rounds, two cold starts, overload/recovery, clean idle/stop |
| Performance | quantize/dequantize kernels, TTFT/decode throughput, KV capacity and peak BF16 workspace memory |

The production topology gate is the current MiniMax-M3 PD configuration:
prefill CP4 with KV sharding and decode DP4. Results must prove active per-DP
load, not only frontend-global concurrency. Q8KV4 performance is reported as a
matched A/B against BF16 using the same source, weights, prompts, topology,
cache flags and graph settings.

## 12. Release criteria

The phase is functionally complete only when standalone and PD-separated
requests prove the KV4 writer, scale transfer and BF16 reader paths from logs or
trace, with no transport/runtime errors. It is not production-ready while CP4,
DSpARK reject/overwrite, graph padding, same-ID quality, peak stability or PD
scale transfer remains untested.

Even after those functional gates pass, the current all-request-page main-K/V
materialization is not the performance end state. Production performance
readiness additionally requires the post-index-top-k selected-block reader (or
a native packed reader), matched throughput/TPOT evidence and an acceptable
peak BF16 workspace envelope.

## 13. Design review record

## 14. 2026-09-22 PD2+2 validation evidence

The validated topology is Prefill `TP2/EP2/WORLD2/DP1/CP2` on GPUs 4-5 and
Decode `TP1/EP2/WORLD2/DP2` on GPUs 6-7. Decode uses CUDA Graph capture buckets
`1,2`; both roles use native `mega_moe_nvfp4`; PD uses TCP cache-store with
`load_cache_timeout_ms=120000`.

The NVFP4 path completed PD requests with `pd_sep=true` and no transport errors.
The first successful request returned two output tokens. Ten subsequent greedy
requests all completed successfully; the hot request `cost_time` was
231.8--246.0 ms (median 242.7 ms), with no runtime or cache-store failures.
After warm-up, GPUs 4-7 reported approximately 245--247 GB used and 27--29 GB
free; this is the current peak-memory envelope for the tested 4K-context
configuration, not a capacity guarantee for longer contexts.

Prefix cache was verified with a 514-token prompt: first request had
`reuse_len=0`, second identical request had `reuse_len=512` and
`prefill_local_reuse_len=512`. Prompts shorter than the Prefill block size 256
correctly did not hit prefix cache.

For a historical matched greedy arithmetic probe (`temperature=0`,
`do_sample=false`, `top_k=1`), BF16 persistent-cache and the then-current
NVFP4 persistent-cache + BF16 working-page prototype both returned `84` on
repeated requests. This records old phase-1 evidence only: the BF16
working-page runtime has since been deleted and cannot be used as a service
fallback. A task-level native-FP4 accuracy campaign remains a separate gate.

The historical BF16 hot reference for this small probe was about 185--191 ms,
while that prototype's NVFP4 hot median was about 243 ms. The difference was
dominated by the removed BF16 working-page gathers and is not evidence about
the current native FP4 attention performance. The first FP4 request after
restart was excluded because CUDA graph/JIT and cache-store warm-up dominated it.

### Strengths

- The numeric contract fixes stored-scale rounding, RNE midpoint behavior,
  nibble order and positive zero rather than relying on hardware FP4 defaults.
- Persistent value and side planes have explicit byte formulas and PD ownership.
- Legacy modes are opt-in and DSpARK overwrite semantics are part of the first
  implementation, not deferred as an unrelated optimization.

### Critical findings resolved in this revision

1. **CP-sharded prefix restore was unspecified.** The reference branch rejects
   this topology. Section 7 now requires paired raw value/side CP gather followed
   by logical-order restore and BF16 materialization.
2. **The reference dense adapter mirrors the full cache pool in BF16.** This can
   erase the KV4 capacity gain. Phase 1 is now restricted to the real M3.1
   all-sparse checkpoint and uses bounded request-table working pages for decode.
3. **DSpARK target/query cannot consume raw packed pages.** Sections 7 and 9 now
   require one remapped BF16 page adapter shared by score and value attention.
4. **The mixed sidecar cannot represent arbitrary TP resharding with one boolean.**
   Phase 1 explicitly requires attention TP 1 on P and D, which covers the
   requested CP4-to-DP4 deployment, and fails other layouts before allocation.

### Important findings retained as delivery gates

- PD needs byte-pattern value/scale/index round-trip tests; layout wiring alone
  is not evidence.
- CUDA Graph needs exact and padded live-batch tests with stable working-buffer
  addresses.
- Q8 is unchanged and needs path evidence; KV4 codec tests do not prove Q-side
  behavior.
- Native Q8 x KV4 kernels, compact CP and arbitrary P/D attention-TP resharding
  remain separate follow-up work.

### Assessment

**Ready to implement: yes, with the scoped constraints above.**

**Ready to merge or call production-ready: no.** That verdict requires the
complete requirement-to-test matrix, especially CP4 KV-sharded PD scale transfer,
DSpARK overwrite, graph padding, quality and repeated stability evidence.
