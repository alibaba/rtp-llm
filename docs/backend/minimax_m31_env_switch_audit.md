# MiniMax-M3.1 recent environment-switch audit

Audited on2026-10-04 against `69be6060368b246939b67396872d36bfb6ff18bb` in the existing DSpark worktree. Uncommitted performance candidates are identified separately below. Current source, not a different worktree's historical configuration, determines whether a switch is live.

## Scope and counting

Examined the M3.1 commits `5cff7481dc`, `f1701e2688`, `e2bc3169c7`, `eff9080c40`, `36c5883242`, `fbc5ca1c33`, and `69be606036`, plus shared memory-cache commit `f901e0ea71` and GCC13 commit `af4403eb17`. Compared added/removed production-source names with each introducing commit's parent. Tests, kernel constexpr names and existing shared launch settings are not counted as new feature switches.

Seven newly introduced switches remain live in M3.1/DSpark/MegaMoE; including three shared memory-cache thread controls gives ten. These counts are not a count of every variable needed to deploy PD. The original provisional-math and MoE-mock selectors were introduced and subsequently removed.

## Live new feature switches

| Switch | Introduced | Meaning/default | Deployment guidance |
| --- | --- | --- | --- |
| `NVFP4_KV_CACHE` | e2bc3169c7 | Default0. Enables packed main KV and idxK, E2M1 plus per16 E4M3 scales, native Q8K4 readers/writers. | Set1 on both M3.1 PD roles. Mutually exclusive with target FP8/INT8 KV selectors. Keep it as a real cache-format choice. |
| `RTP_LLM_DSPARK_CUDA_GRAPH` | 5cff7481dc | Defaultoff, only literal1 opts into separate exact-shape DSpark propose/commit graphs. | Set1 on Decode along with target graph enable/capture buckets. Prefill remains eager and may use0. This does not enable the target graph by itself. |
| `GLM5_MEGA_MOE_NVFP4_INPUT_PACKER` | f1701e2688 | Default `fused`; `auto` currently selects the same fused implementation; `torch` selects reference packing. | Omit for production. Keep explicit torch as a correctness/rollback tool; consider deprecating redundant auto alias later. This chooses activation/routing packing, not the MegaMoE GEMM or weight format. |
| `GLM5_MEGA_MOE_NVFP4_PACK_BLOCK_M` | f1701e2688 | Optional override1/2/4/8/16. Auto policy selects16 for tokens>=1024, and H6144/topk4 at80/96/112/128 rows; otherwise4. | Omit to retain shape-aware tuning. Do not globally force16 for every small Decode shape. Retain override for measured tuning/debug. |
| `SP_DSPARK_VERIFY_TOKENS` | fbc5ca1c33 | Default0 means gamma. Meaning depends on mode: full static; adaptive initial average extra-row budget; legacy fixed-prefix budget when no mode and boolfalse. | Keep. It is not the draft block size and not an acceptance guarantee. |
| `SP_DSPARK_VERIFY_MODE` | 69be606036 | `static`, `adaptive`, or defaultempty for older bool/budget semantics. | Prefer an explicit mode in new deployments. This replaces the need to configure the legacy bool, not the token budget. |
| `SP_DSPARK_ADAPTIVE_VERIFY` | 69be606036 | Defaultfalse. Compatibility bool enabling confidence-ranked compact verification. | Omit in new deployments and use VERIFY_MODE. Keep code compatibility for existing configs/pickles for now; static plus true is rejected. |

### Verification modes: precise semantics

For released gamma7 weights, keep `GEN_NUM_PER_CIRCLE=7`; the draft backbone still produces7 rows irrespective of the target verification budget.

* `SP_DSPARK_VERIFY_MODE=static`: confidence planning is disabled; verifies all gamma draft candidates plus anchor. VERIFY_TOKENS must be0 or7. Zero here means use gamma, not verify zero tokens.
* `SP_DSPARK_VERIFY_MODE=adaptive`: the confidence planner compacts real target rows. VERIFY_TOKENS controls initial total extra-row budget `min(B * budget, B * gamma)`, not a fixed per-request length or per-request cap. Each request still has its anchor; selected candidate counts may differ across requests.
* Empty mode plus legacy boolfalse preserves the older fixed-prefix behavior: e.g. VERIFY_TOKENS4 means each request verifies that prefix plus anchor. Do not confuse this with explicit static verify-all.

Thus adaptive with budget4 and B16 supplies64 extra-row budget, with16 anchors, while individual request lengths are confidence-dependent. Adaptive with0 supplies the full gamma budget and is not a useful way to reduce target compute.

New adaptive deployment fragment, applied consistently to both PD roles:

```text
GEN_NUM_PER_CIRCLE=7
SP_DSPARK_VERIFY_MODE=adaptive
SP_DSPARK_VERIFY_TOKENS=4
```

Budget4 is the current tested configuration, not a universally optimal production value; tune against normal-rejection latency/MAL and batch size. Forced acceptance is a separate diagnostic.

## New shared memory-cache controls

| Switch | Default/priority | Guidance |
| --- | --- | --- |
| `RTP_LLM_MEMORY_CACHE_H2D_WAIT_DONE_THREADS` |16; overrides common control | Controls host threads waiting for H2D completion, not GPU copy streams or PD RDMA bandwidth. |
| `RTP_LLM_MEMORY_CACHE_D2H_WAIT_DONE_THREADS` |24; overrides common control | Separate D2H completion-wait pool; increasing may consume CPU without improving GPU copy bandwidth. |
| `RTP_LLM_MEMORY_CACHE_WAIT_DONE_THREADS` | Common fallback when direction-specific value absent/empty | Avoid setting both common and directional knobs in new configs. Retain common as compatibility fallback. |

All three were introduced by f901e0ea71; accepted thread counts1..64, invalid values fall back to direction defaults. They matter when Memory Cache is enabled, not pure device-cache-hit GPU computation. `RTP_LLM_MEMORY_CACHE_WAIT_DONE_QUEUE_SIZE` is an existing shared knob, default1000, not a new M3.1 feature switch.

## Removed or ineffective settings

The following have no production reader in the audited worktree and should be removed from deployment environment lists:

* `M31_DSPARK_CANDIDATE_MATH`: removed by69be606036. No longer selects provisional causal math; released DSpark uses checkpoint-validated Gemma norms, non-causal query block, checkpoint sliding window.
* `M3_M31_MOCK_NVFP4_MOE`: removed byf1701e2688. Real NVFP4 MegaMoE replaces the old mock path.
* `M3_IDX_PAGED`: removed by e2bc3169c7; paged indexer cache is not an optional flat-cache mode.
* `M3_LOAD_MSA_INDEX`, `M3_MSA_RAW_IDX_MXFP8`: removed by36c5883242; sparse-layer/weight handling no longer needs manual switches.
* `M3_MSA_FUSED_KV_GATHER`, `M3_MSA_KV_GATHER_BLOCK_T`, `M3_MSA_IDX_GATHER_BLOCK_T`, `M3_DISABLE_TRTLLM_GEN`: removed with legacy flat-scratch routing by36c5883242.
* Removed verbose diagnostic selectors in69be606036 include `RTP_LLM_ASYNC_DEBUG`, `RTP_LLM_COMPARE_SP_PREFILL`, `RTP_LLM_DEBUG_MTP_ACCEPT`, `RTP_LLM_DEBUG_MTP_DECODE_DATA`, `RTP_LLM_DEBUG_MTP_PREFILL_DATA`, `RTP_LLM_DEBUG_TARGET_VERIFY_INPUT`, `RTP_LLM_PD_DEBUG`.

`M3_MSA_CP_COMPACT_PREFILL` is different: still live for old M3, but explicitly excluded when NVFP4 is enabled. Omit from M3.1 configs; do not remove the legacy M3 code merely because this M3.1 path does not use it.

## Existing settings still relevant, not newly introduced here

* `SP_TYPE=dspark`, `SP_MODEL_TYPE=minimax_m31_dspark`, `SP_CHECKPOINT_PATH`, `SP_ACT_TYPE=BF16`, `SP_FP8_KV_CACHE=0`: existing speculative config infrastructure. SP_CHECKPOINT_PATH remains required by the current ModelFactory; using a target root/nested draft checkpoint is a loader choice, not automatic omission of this field. Target embedding/LM head are shared, not independently recomputed/copied draft weights.
* `MOE_STRATEGY=mega_moe_nvfp4`, model type `minimax_m31_vl` for VL traffic, fastsafetensors and `FORCE_CPU_LOAD_WEIGHTS=0` remain required operational choices for this deployment.
* `PREFILL_CP_SIZE=4`, `PREFILL_CP_KV_CACHE_SHARDED=1`, role-specific CP rotate mode, Decode DP topology and CUDA Graph capture buckets remain important. They are shared settings, not the seven new feature switches above.
* `M3_MSA_INDEX_SCORE_CHUNK_ROWS`, `M3_SPARSE_ATTN_CHUNK_ENABLE/SIZE`, `DSV4_MOE_CHUNK_PREFILL/TOKENS` remain separate workspace/chunk policies. Do not combine them into one number: attention partial workspace, index scores and EP MoE collective scheduling have different geometry.
* `M3_ROUTER_BF16` remains supported:0 selects FP32 router and1 selects BF16 router. The current2026-10-06 measured candidate uses1 on both PD roles, following the approved BF16 policy. FP32 Prefill owns its expansion buffers; BF16 retains the ordinary linear path. Do not silently change precision between comparisons or alter old M3 semantics.
* `M3_SPARSE_ATTN_FP8_PARTIAL` remains a live legacy/global plan selector. Keep unset/0 for the measured BF16-partial contract. BF16 partial O is not historical BF16 working-page KV and does not mean the Q8K4 attention path was reverted.
* `SP_DETERMINISTIC_DRAFT_EXACT_MATCH=0` belongs to normal rejection configuration; do not turn on exact-match/force-accept diagnostics to report production stochastic accuracy or MAL.

## Recommended cleanup order

1. Remove dead environment names above and fix stale DSpark docs. No runtime semantics change is needed for these deletions.
2. New configs use only VERIFY_MODE for mode selection; keep VERIFY_TOKENS as an independent budget. Deprecate the compatibility bool later, with migration and serialized-config tests.
3. Omit packer mode/tile overrides and common memory thread override unless profiling proves a need. Keep measured directional overrides where H2D/D2H throughput needs them.
4. Retain the supported router precision selector and explicitly record its measured value. The current candidate uses BF16; removing the selector or forcing FP32 is not part of this release.
5. Current uncommitted norm/RoPE grouping adds no environment variable. TopK32 flat-combine capability is negotiated from the dependency callable, not another operator environment flag. Old dependencies remain on the native unflattened path; shipping that optimization requires the validated dependency update as well as RTP source changes.

This document describes source semantics. It does not certify a different installed wheel, online environment, new verify budget, or newly added dependency package.
