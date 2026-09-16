# Unified SM120 DSV4 development line

Source integration date: 2026-09-16. This is a **development candidate**, not a
production, numerical-quality, performance or PD-compatibility acceptance.

## Shared source, separate explicit profiles

Develop CP2PP4 prefill, CP4EP4PP2 prefill and SM120 decode on this source line.
Do not fork model implementations to change a launch topology. Existing measured
worktrees remain immutable evidence, not the destination for new development.

The foundation is official `feat/support_pp` at
`41b3d61088f012d92ac1ac699cbfb9372780d99a`, reconciled with CEP
`1ce7577f71a9edf4506796b7b908a6aa9f7acc82` and its previously uncommitted four-file
runtime delta (`68fa175c139386e386a2e5f9f5338e93c64bad2734bd5aa00ecce1dcf711f9ea`).
The decode donor is `feat/dsv4_sm120_rtx5000_cp2pp4` at
`94f53cd964ab6cd16b23fa6552b893abf4d28d25`.

This is a forward integration, **not a replay of the whole donor history** and
not a force-rewrite of its shared branch. That donor tip does not contain the
PPExecutor/RankLayout framework needed by CEP. Both the official PP and measured
CEP histories remain ancestors of the new line; donor semantic ports have the
mapping below. Do not mark the donor tip as merged with an `ours` merge: that would
hide unported differences from the next review.

### Profile contracts (not launch scripts)

| Profile | Logical topology | Physical/kernel cache page | Initial arithmetic |
|---|---|---|---|
| CP2PP4 prefill | PP4, CP2, TP group2, DP1, EP-off, world8; layers11/10/11/11 | 256/128 | Retain its recorded incumbent recipe; native prefill indexer/tile64 require its pinned provider |
| CEP4PP2 prefill | PP2, CP4, TP group4, DP1, EP4, world8; layers22/21 | 256/128 | Retain the frozen CEP recipe; native flags introduced here OFF |
| Native decode candidate | PP-off, DP4, TP1, EP4, world4; speculation chosen explicitly | Donor native paged-score recipe 256/256 | Opt-in native kernel choices below; reproduce donor workload/dependencies before comparing speed |

The first two profiles are warm-engine 32K/chunk4096 prefill experiments. A
PDFUSION, one-output-token benchmark does not qualify a real PREFILL->DECODE
service. In particular, **256/128 and 256/256 are NOT automatically PD-compatible**.
Resolve group/layer/global cache keys and the actual page layout before connecting
the instances. Speculative CEP remains outside the narrow PP+EP opt-in.

The launcher must explicitly set every topology, page size, KV dtype, role,
checkpoint, chunk/overlap, speculative mode and numerical flag. These profile
notes do not provide those full launchers or authorize a launch.

## Native kernel selections

All new flags default to **0**; set them before importing the runtime. They are
numerical/backend variants and must not silently replace the frozen CEP recipe.

| Setting | Selected mechanism / condition |
|---|---|
| `DSV4_SM120_WOA_EINSUM=1` | Native `deep_gemm.fp8_einsum`; the matching einsum warmup gate is enabled too |
| `DSV4_SM120_NATIVE_FP8=1` | Exact SM120 native dense FP8 factory, packed UE8M0 scales, no cached-scale FlashInfer override |
| `DSV4_SM120_SHARED_EXPERT_FUSED=1` | Shared-expert fusion; on SM120 requires the native FP8 flag/packed-scale path |
| `DSV4_INDEXER_FP8_DEEPGEMM_PAGED=1` | Native **decode** paged score; rejects anything other than64 indexer entries/block and missing API |
| `DSV4_SM120_PACK_DECODE_SLOTS=1` | Optional compact footer-cache packing for native sparse decode widths; direct-paged/wide-window fallback remains available |
| `DSV4_SM120_BF16_PRETRANSPOSE=1` | Decode-only weights projection via pretransposed nn-layout GEMM |
| `DSV4_MHC_PRE_GEMM_BACKEND=deepgemm` | Existing explicit native mHC selection; no global default change |
| `DSV4_INDEXER_FP8_DEEPGEMM=1` | Existing **prefill/nonpaged** score flag; not the paged flag above |

`DSV4_DEEPGEMM_SHADOW_PATH` carries the donor's provider override. Pin the provider
source, binary digest and JIT configuration, then prove the imported module and
actual kernels. Presence of a symbol, an environment variable or a successful
source build is **not** evidence that a bundled wheel supports SM120. Do not
substitute a colleague's unrecorded wheel or quietly fall back during an A/B.

## Donor accounting

| Donor | Treatment |
|---|---|
| `dbfc93d2d` | Carried: rank-uniform decode padding/admission options and ALL-fake semantics; options remain off by default |
| `103fda00d` | Already semantically covered by official PP's `get_global_rank(pg, root)` mapping; no duplicate root conversion |
| `d223abc5c` | Carried provider-path override |
| `b8d5ff639` | Native wo_a route + matching warmup, behind independent opt-in |
| `415ec80be` | Native dense intent adapted to the newer SM120 factory and scale contract; not just deletion of a call-site shim |
| `3050b3319` | Singleton packed-scale dtype-view repair retained in the baseline helper; CPU regression covers it |
| `17a5aed4b` / `f4cd982fe` | Shared-expert native path retained behind explicit flag, preserving baseline fallback |
| `a7d55209b` / `71b8f3960` | Native paged score with a rejecting geometry gate; destructive ablation and global fallback deletion omitted |
| `fbc8ade2c` | Use existing explicit mHC backend override rather than changing auto/default arithmetic |
| `d9e0fd50f` | Packing primitive and tests ported; retain newer direct-paged path, generic wide-width handling and bounded gather. Add pool upper bounds and keep int64 slot IDs until validation |
| `960194925` | Carried decode-only pretranspose option; remove the unsupported claim that subsequent quantization proves equivalence |
| `94f53cd96` | **Not ported:** stale-writer recovery and catch-all retry need a separate failure-lifecycle review against the newer PP request-error path. No PP retry/recovery equivalence claimed |
| Donor diagnostics / ablations | Not automatically enabled or bulk-ported |

Qoder's B7 reconstruction, MoE prequantized-input candidate, and PR4720 fusion are
NOT included. The indexer/wo_a/MoE fixture status is not promoted by consolidation.

## PP merge invariants and tests

- `pdSeparation` appears once in the shape-hint enum, sender and receiver.
- The new request-scoped upstream error protocol is retained.
- PPExecutor always supplies its dispatch-time `PPStreamRoundSnapshot` to result
  dispatch. The two-argument overload is for synchronous callers/tests only.
- Delayed intermediate chunks never append sampled tokens merely because the live
  stream cursor has advanced to the final chunk.
- Official RankLayout is authoritative; stage-local EP and narrow CEP opt-in stay.
- The C++ regression cases are in `PPBatchStreamProcessorTest.cc`; policy tests are
  in `tests/sm120_unified/`. They are not replacements for GPU/service regression.

CPU-only checks (Python3.10+):

```bash
CUDA_VISIBLE_DEVICES= python tests/sm120_unified/test_integration_policy.py
CUDA_VISIBLE_DEVICES= python tests/sm120_unified/test_scale_cpu.py
```

Use the repository's worktree/test-execution skills for Bazel. The SM120 native
build config is **`cuda13_sm120`**, not separate cuda13+sm12x configs. Rebuild the
paired internal runfiles after source changes; never launch this code against an
old worktree's native libraries and call that an integration test.

## Collaborator handoff / gates

1. Work on the new unified source branch. Make reviewable commits; do not rebase
   or force-push a colleague's already published history.
2. Keep the same compatible internal build revision and exact dependency pins.
   Internal smoke/build support is separate from this public source tree.
3. First reproduce each original profile with unchanged arithmetic, then enable
   candidate kernels separately. Preserve failed runs and raw measurements.
4. B7 first-causal model work stays on its frozen reference until explicitly
   migrated. No goldens, acceptable-ID sets or reference-freeze waiver is created.
5. Production readiness still requires correctness/repeatability, native-kernel
   capability/binding, performance, decode, PD, cancellation/drain and fault tests.
6. Publication needs explicit owner approval. This source integration itself does
   not authorize a push to either official or fork remotes.

## Bounded unified smoke qualification (2026-09-16 UTC)

Host artifacts use `20260917` (Asia/Shanghai, UTC+8). **Runtime revision
`0e2e9b632a4701924a864f6bc901011630221c1f` passed the two profiles below.**
This is development smoke qualification, not full CI or production acceptance.

| Validated profile | Evidence / exact scope |
|---|---|
| CEP4PP2 baseline | PP2/CP4/EP4, 8 GPUs, 32K input, chunk4096, split22/21, overlap2, PDFUSION, FP8 KV/BF16, pages256/128, bundled DeepGEMM, numerical candidates OFF. Two warmups + two ordered8-prompt passes:18/18 completed;16/16 measured screening IDs exact. Final warm server TTFT mean1576.826ms (range1569.103–1590.107), not a vLLM-parity or speedup claim. |
| DP4EP4 DSpark decode through real PD | Four CP4EP4 producer GPUs + four DP4EP4 decoder GPUs; PP-off; DSpark gamma3; decode graphs1/2/4/8; both sides FP8 KV/BF16/pages256/128. Original4-query smoke passes unchanged comparer/goldens. First2 produce exact Paris answer,9tokens each with `pd_sep=true`; final2 request1token and intentionally finish on prefill, with2048cached input tokens on the repeat. |

The second profile is **not** an8-GPU CEP producer plus4-GPU decoder: that
combined topology needs12GPUs and has not been measured here. The original
one-token cases have `skip_choices`/`skip_usage`; they are cache/transport smoke,
not a new semantic golden or long-decode quality test.

### Required decode integration details

- Use SM120's NCCL/FusedMoE EP strategy: `--use_deepep_moe 0` and
  `--use_deepep_low_latency 0`. Do not disable EP4 to evade a startup failure.
- With the pinned provider and256/128 cache geometry, explicitly set
  `DSV4_SM120_PACK_DECODE_SLOTS=1` on **decode only**. It supplies the native
  sparse-decode64-entry page ABI without changing the PD wire/cache geometry.
  This is not a source default change or permission to enable other native flags.
- PREFILL DSpark uses its pruned commit-only weights and rejects PROPOSE;
  DECODE retains full proposals. The warmup flag crosses native C++ `setenv`,
  so its Python consumer reads native `getenv`, not a stale `os.environ` snapshot.
- Remote producer CP metadata must not enable local CP execution on TP1 DECODE.
- CP target-forward mutates input geometry: saved draft inputs must own cloned
  storage before asynchronous H2D, not merely keep the original pinned storage
  alive. The old race was reproduced with a paused DMA and regression-tested.
- Always use absolute `LOG_PATH`. Also **unset** `RTP_LLM_LOG_TPSYNC` to disable
  it: assigning `0` still enables the C++ getenv-presence guard.
- The internal manual smoke recipe needs its separately supplied BUILD patch;
  the public source commit does not implicitly update the parent submodule pin.

Selected tests:47 C++ (9ModelData+38PP),89 Python attention/decode/cache,
32 DSpark contract,42 warmup/strategy,5 decode-page adapter hardware tests,
plus the unified CPU policy/scale checks. Counts are suite-local, not a claim
that every repository test ran. All failed model/fixture attempts are retained.

**Remaining gates:** held-out numerical repeatability/acc011/acc022; cold-first-use
readiness and latency; cancellation, fault recovery and graceful shutdown;
long/concurrent decode and performance acceptance. A newly introduced random
native-attention-vs-FP32 probe failed its4e-2 envelope (max absolute0.09765625).
That failure is not waived: the passing adapter test establishes bitwise equality
to the same native kernel on independently reblocked legal pages, not native
precision acceptance. Post-TERM peer-loss errors also remain unqualified.

Machine evidence: `~/rtp_cp2pp4/UNIFIED_OVERNIGHT_RETURN_20260917.md` and
`investigations/unified_overnight_20260917/`. Final-chunk stage0/rank0 and
stage1/rank4 Chrome traces are in `~/rtp-cep4ep4pp2-32k-unified-traces.tar.gz`.
They were captured on `a40db533e`, in separate launches, and are **not** a
simultaneous two-stage or full-request timeline. The later final-source smoke
revalidates CEP after the decode repairs; it does not relabel those old traces.
