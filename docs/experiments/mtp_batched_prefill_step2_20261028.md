# Step 2 — batched MTP fastgen prefill

October 28, 2026. Local branch: `dev/pd-opt-xiaoweiw-mtp`. No push. Decode unchanged.

## Outcome

MTP fastgen prefill supports **up to four requests per pass**, including mixed intermediate/final rows. Draft KV is written on every chunk; intermediate rows emit no sampled/proposed output and do not publish PD state.

**Leave `CP_SINGLE=1` in service.** Batching currently improves the first-token arrival-span rate but worsens end-to-end hit16/hit32 wall time. Draft + sampling is only **4.2% of the four-request last-stage pass**; draft tuning is stopped. Per-request indexer or cheaper attention work is outside this task and was not started.

## Final revision comparison: three full suites per revision

The frozen **6c82da5c** worktree and current serving-code HEAD **e1fadfd2** each ran the same full 32K suite **three times**, with `CP_SINGLE=1 MTP_CHUNKED=1 MAX_CTX_BS=4 MAX_INITED=8 PP_OBJ_LOG=0`.

| Revision | Run 1 | Run 2 | Run 3 | Wrong answers | Transport errors |
|---|---:|---:|---:|---:|---:|
| Snapshot `6c82da5c` | 86/86 correct | 86/86 | 86/86 | **0/258 (0%)** | **0** |
| Current code `e1fadfd2` | 86/86 correct | 86/86 | 86/86 | **0/258 (0%)** | **0** |

**516/516 correct in the requested comparison. The observed wrong-answer rates are identical.** Earlier sporadic wrong-city outputs did not reproduce in these runs. They are recorded as known baseline/nondeterministic behavior for the requested close-out, rather than an established Step-2 regression. The zero-failure comparison does not independently prove their mechanism.

Both arms used the snapshot's unchanged `fgreuse_smoke2.py`, identical prompt contents, one disjoint warmup request excluded from the counts, and the same running decode instance. Each suite covered `cold1,seed,hit4,hit16,hit32,cold32`. The standard checker searches for the expected city in the generated text. No runs were discarded or repeated after seeing their results.

The selected source revisions, clean serving-source diffs, loaded library paths and SHA-256 hashes were verified before each arm. The frozen snapshot was not edited or rebuilt. Raw records, manifests, script and aggregate counts are in the accompanying `final_baseline` directory.

## Implementation and local commits

| Commit | Change |
|---|---|
| `6d10e4da` | Draft KV on every chunk; prompt lookahead and positions; model-owned async head builders |
| `744b8f0b` | Per-request phase mask, PP serialization/TP broadcast, mixed-row sampling and batched CP |
| `bde403bd` | Skip unused intermediate LM/output work; KV-only intermediate draft; native fast paths |
| `148f72c9` | Fix allocation-stream ordering for eager MoE count buffers |
| `a867b947` | Fence only newly allocated count buffers, preserving steady-state overlap |
| `e1fadfd2` | Gated PPSTEP stage-wall/component timings and root handoff timing |

The serving code measured in the final revision comparison is **e1fadfd2**. The close-out documentation commit does not change serving code.

Main changed areas:

- PP scheduler, execution plan, serialization, sampling/dispatch and executor.
- CP row/layout validation, zigzag hidden-state handling and Python model bindings.
- MTP prefill, per-model async metadata builders, fast-path eligibility and KV-only draft execution.
- Eager MoE count-buffer ordering and regression tests.

The accompanying `changed_files.txt` contains the full absolute file list; `changes.patch` contains the serving-code diff from `6c82da5c`.

## Validation

- Single and four-concurrent mixed-length cold/cache-hit tests passed, including intermediate/final rows in the same pass.
- Logs demonstrate **`PPSTEP streams=4`** and **`PPPHASE rows=4 intermediate=3 final=1`**.
- Earlier full 32K suites passed 86/86; the final repeated revision comparison above records the actual wrong-answer rate rather than selecting only passing runs.
- Existing CP tests, 40 actual C++ CP layout cases and five batched last-hidden merges passed. Phase-mask serialization/selection/bounds, prompt-tail shifting and empty CP output selection were checked.
- Async-head, fast-path, KV-only and count-buffer ordering tests passed. Production builds and `git diff --check` passed. No Bazel test targets were built because their torch-wheel dependency returns HTTP403; existing production libraries/manual wrappers and CPU tests were used.
- Identical-input KV-only versus full draft projection: **140 exact matches**.
- Checked first proposals and first-step output lengths matched the non-chunked reference on the tested prompts. This is not a claim of exhaustive first-step acceptance coverage.

### Explicit verification limit

**Exact full-vs-chunked whole-pipeline draft-KV equality was not established.** Captured target hidden states already differ before the draft; exact-aligned 32K also differed (draft-KV relative L2 approximately 0.229). The identical-input projection checks validate the KV-only optimization, not full-pipeline parity. This limitation is recorded; no further investigation is part of the requested close-out.

## Before/after measurements

| Metric | Singleton | Batched cap 4 |
|---|---:|---:|
| hit16 wall, including output completion | **2.902 s** | **3.185 s** |
| hit16 first-token arrival-span rate¹ | **10.71 req/s** | **12.55 req/s** |
| hit16 correctness / transport errors | 16/16 / 0 | 16/16 / 0 |
| hit32 wall² | **4.587 s** | **5.017 s** |
| hit32 first-token arrival-span rate² | **10.56 req/s** | **10.60 req/s** |

¹ `(N−1)/(last first-token arrival − first first-token arrival)`, excluding pipeline fill and output completion. The current-code hit16 pair used disjoint warmups; all requests reused 30,720 of 32,768 prompt tokens. Batching improved this rate 17.2%, but total wall was 9.8% worse.

² Last matched cap-4 hit32 pair, measured at `bde403bd` before the count-buffer fence. It is not a fresh batched hit32 measurement at the final HEAD. Both legs passed their city checks. The final close-out runs only the requested singleton revision comparison.

Cold TTFT: latest eight-unique-nonce `pd_bench.py` measurement was **1,214.22 ms server mean**, against the requested **1,216 ms** limit. That run used cap 3, concurrency 1; it is not a profile-enabled or startup/JIT measurement. Earlier singleton baseline was 1,205.50 ms.

Final cap-4 32 × 1,024 decode load: **2,020.7 tok/s**, **15.892 ms TPOT p50**, acceptance **2.913**, observed batch **32**, rank counts **7/8/9/8**, zero errors. A subsequent cap-3 load returned 2,019.7 tok/s, 15.924 ms and acceptance 2.895, also batch 32/zero errors; some of its prefills were cold, so it is not a cache-hit timing comparison.

## Profile breakdown

Timing-only instrumentation preserves production synchronization and execution order. `PPSTEP wall_us` starts after incoming activation reception and ends before waiting for a previous batch's result. These are stage service times, not client TTFT.

| Pass | Samples | PP0 mean [range] | PP1 mean [range] |
|---|---:|---:|---:|
| One request, CP_SINGLE=1 | 16 | **95.33 [92.97–96.88] ms** | **97.61 [96.85–99.59] ms** |
| Four requests, batching | 2 | **363.38 [304.03–422.72] ms** | **317.40 [317.11–317.69] ms** |

PP0 has 22 target layers; PP1 has 21 target layers plus draft/sampling. Matching PP0-start → PP1-result-enqueue latency averaged **198.28 ms** for singleton passes and **682.90 ms** for four-request passes. The slow four-request PP0 pass is included, not discarded.

### One clean four-request cache-hit pass

| Component, both stages | Wall |
|---|---:|
| Target forward/completion, excluding host publication | **538.093 ms** |
| Draft step | **12.742 ms** |
| Sampling/length handling | **0.647 ms** |
| Host PD publication | **69.837 ms** |
| Activation/result-send enqueue | 0.298 ms |
| Other stage overhead | 0.101 ms |
| Interstage gap | 2.085 ms |
| **PP0 start → PP1 result enqueue** | **623.803 ms** |

Subsequent root result dispatch: **18.431 ms**, outside that endpoint. Publication/send/dispatch are host measurements; **RDMA wire transfer was not independently timed**. Publication remains before the original GPU-completion wait, so its host work can overlap pending GPU work. The table is non-overlapping host-wall accounting, not independently synchronized GPU phases.

Draft + sampling: **13.389 ms for this pass**, 4.21% of PP1 wall; across both four-request passes the mean is **13.29 ms / 4.19%**. No further draft tuning.

### Target GPU attribution

A separate matching four-request cache-hit pass was captured on stage roots 0 and 4 with the existing fast path retained. Each rank processed four 512-token CP shards, all with prefix length 30,720. Temporary Python capture hooks were removed. CUPTI/export host overhead is excluded from the clean timings above.

| Exclusive target GPU timeline component | Both stages |
|---|---:|
| Attention, excluding indexer | **257.03 ms** |
| MoE | **198.22 ms** |
| Indexer | **32.66 ms** |
| Cross-category overlap | 2.97 ms |
| Other kernels/copies, including mHC | 17.86 ms |
| Gaps with no captured GPU activity | 32.07 ms |
| **Target GPU span** | **540.81 ms** |

Indexer is removed from attention; overlap is not double-counted. Attention includes projections, sparse attention, compression/cache work and CP communication. MoE includes routing, expert/shared-expert computation and EP communication. **Attention + MoE occupy 84.2% of the target GPU span.**

## Previously requested 09:17 checks

These used prefill `6c82da5c`, CP_SINGLE=1, and the same unchanged decode instance. Every input reused 30,720/32,768 tokens, so both modes executed one 2K prefill pass.

| Prefill | TPOT p50 | Full-batch tok/s | Acceptance³ | Requests/rank |
|---|---:|---:|---:|---|
| MTP fastgen | **16.19 ms** | **1,995.1** | **2.851** | **7/9/8/8** |
| Non-fastgen MTP | **16.79 ms** | **1,916.4** | **2.819** | **8/10/7/7** |

Both completed 32 × 1,024 tokens, zero errors, observed decode batch 32. This does not attribute the historical 855 → ~1,940 jump to fastgen. The October 27, 17:31 run used decode PIDs 898322–898325; the current instance started October 27, 20:36:02 with PIDs 1040956–1040959. The exact historical slowdown was not isolated.

Natural output: 32 city reports, `max_new_tokens=2048`, **no minimum length**. All completed, zero errors, ranks **8/8/8/8**. Acceptance **2.192**, versus supplied 8-PPU reference **2.15**; full-batch throughput **1,582.3 tok/s**, end-to-end **1,425.9 tok/s**. Mean output length **2,014.69**, median **2,048**: 26 outputs reached 2,048; the other six were **1,670; 1,828; 1,855; 1,900; 1,963; 2,006**.

³ Round-weighted first-to-last `MTP_ACCEPT_PROBE` intervals emitted inside each run. Counters flush every 512 observations; unflushed partial intervals are not covered exactly. Accepted length includes the retained target token, range 1–4.

## Final state and scope

- Prefill is healthy on **10.65.32.187:14000**, running the current serving build, with **CP_SINGLE=1**, MTP_CHUNKED=1, MAX_CTX_BS=4, MAX_INITED=8 and PP_OBJ_LOG=0. The normal target fast path is restored. The final HEAD arm leaves the server warmed.
- Decode remains healthy on ports **23000/23009/23018/23027**, with the same PIDs **1040956–1040959**, start ticks and configuration SHA-256 as before testing. No decode restart, code/configuration change or manual cache reset occurred.
- All implementation commits are local. The final documentation commit records this close-out; no push.
- The two pre-existing untracked generated protobuf files were preserved. No temporary Python capture hooks remain in serving source.
- GPU0–7 remain claimed because prefill is left running. `/mnt/fuse/scratch_admin/GPU_CLAIMS.txt` is updated at close-out.
- Investigation and optimization stop at this state, as requested; the verification limit above remains documented.

No further wrong-city investigation, draft tuning, indexer changes or attention optimization will run in this task. The remaining performance work is a separate task and has not been started.
