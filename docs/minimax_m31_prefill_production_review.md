# MiniMax-M3.1 Prefill optimization production review — 2026-10-10

Review baseline: `399db00d611eddf52263a5ca3e2c09fae6ef01de`, scoped Prefill worktree changes. The unrelated Decode threshold, C++ sampler/state, compact-vocabmask, and Decode kernel changes are excluded from the submission artifact.

## Findings and fixes

- **P2 / maintainability, fixed:** `common/nvfp4_cp_wire.py` retained an unused R1 receiver while production used the R16 receiver from a second module. Consolidated sender and R8/R16 receiver into one implementation. The old multirow module remains a compatibility import. Sender/kernel arithmetic and the sender wrapper have identical ASTs to the model-validated version.
- **P2 / invalid-call guard, fixed:** CP destination validation accepted aliasing K/V planes, which could race writes if a caller supplied overlapping views. Added metadata-only overlap validation. It supports the legitimate FI and persistent common-buffer page pitches, skips empty views, and performs no device synchronization. Validated with an independent finite-interval oracle, legal FI views, same-plane aliases, next-page overlap, and empty planes. This was not an observed model corruption.
- **P2 / regression coverage, fixed:** Repository tests lacked direct FI producer/consumer and FI prefix-restore coverage. Added production page pitch with retained capacity, exact converted-page and attention comparison, FP32 attention reference, R1/R8 writer variants, repeated source IDs, untouched tails, changed contents/maps during CUDA Graph replay, and malformed direct-layout rejection.

No confirmed P0/P1 defect was found in the successful CP4 path. Review covered fused norm/RoPE outputs and BF16 rounding, working/persistent layout separation, mapped dual writes, immutable per-forward index metadata and layer-key refresh, native score/TopK workspace epochs, CP leases/events/grow-shrink retirement, and explicit feature routing.

## Validation

Operator validation passed: 70 CPU workspace/wire/native-index tests, 14 GPU restore/attention tests, and the 66-case wire GPU byte gate for MMA and FI working layouts. The wire invalid-source assertion was exercised in an isolated child process. Zero skipped GPU tests.

Post-review model gate completed from the immutable `production_review_source` / `production_review_cp4` package: all 20 formal rounds / 960 requests passed; all 660 same-ID quality requests passed with no transport/runtime/scoring errors. Four rank traces each contain 60 wire senders and R16 scatters, no old writer or page converter, and the expected 71.875% suffix payload reduction. Restore/suffix-AG overlap remains zero.

| Scenario | Warm native model s | Per-card cache TPM | Per-card cache TPMS |
| --- | ---: | ---: | ---: |
| pd_bs40_80k_reuse95_h32 | 1.900557 | 25255754 | 420.929 |
| pd_bs60_80k_reuse95_h32 | 2.729859 | 26374989 | 439.583 |
| pure_bs40_80k_reuse95 | 1.683691 | 28508794 | 475.147 |
| pure_bs40_shrink_80k_reuse95 | 1.704716 | 28157183 | 469.286 |
| pure_bs60_80k_reuse95 | 2.486263 | 28959125 | 482.652 |

TPM is tokens/minute; TPMS is tokens/millisecond. Count all 80,000 input IDs/request, including 75,776 reused IDs, divide by four P cards, and use the median of three unprofiled `NormalExecutor.model_forward_us` rounds. PD sums both native context steps. Matched differences against the prior final-wire arm range from about -0.87% to +0.31%; this refactor does not establish a new performance gain.

Fresh quality: GSM8K 438/500 vs 436/500; LongBench 66.4556623 vs 66.2377381. Inputs and IDs match for all 660 samples. Full responses match for 307/500 GSM8K and 154/160 LongBench. Current GSM8K has one length-limited response, prior arm none; LongBench has five in both arms. Preserve these completion differences; neither aggregate scores nor unchanged wire arithmetic establish complete-text equivalence.

**Additional stability results:** two cold starts, three BS60 peak rounds, and 17 mixed-length PD32 requests (129 / 4,097 / 32,771 / 80,000 tokens) passed: 198 positive requests in the second cycle. The first cycle was stopped before the second launch. All owned services, ten HTTP/gRPC ports, and GPU processes are now stopped/clean.

**P1 pre-existing native RPC boundary risk; unresolved outside this Prefill patch:** a direct native RPC with 1,048,577 input tokens against `max_seq_len=1,048,576` aborted rank 0 at `CompleteTokenIds.cc:48`, then the service exited. First failure is in request construction, before any model/Prefill kernel. Both arms use identical native libraries, and this C++ file is unchanged by the Prefill patch. The negative probe failed; subsequent cancellation and recovery probes did not execute. This is not a successful overload/recovery validation. Fix native input rejection and rerun the failed boundary/cancellation/recovery probes before unrestricted production sign-off.

All failures and positive results are retained under `/data0/ruixuan.zrx/minimax-m31-dev/1010_prefill_production_review`; full model output is under `/data0/ruixuan.zrx/minimax-m31-dev/1009_prefill_model_integration/production_review_cp4`. In particular, see `lifecycle_cycle2/failure.json`, `lifecycle_terminal.json`, and `production_review_cycle2/prefill/logs/engine.log:11011`.

## Submission and rollout scope

The submission preserves `RTP_LLM_CP_SUFFIX_NVFP4_WIRE=0` by default. Enable wire explicitly only for the validated MiniMax-M3.1 BF16-carrier / native KV4 / TP4-CP4-EP4 / same-layer / Zero-CTA / FI configuration. Persistent and PD byte ABIs are unchanged. No new AG-overlap schedule, R1 model writer, FP8 working pages, or unstable norm/index/score candidate is included.

The original LongBench FI→wire change (66.8171467→66.2377381) remains recorded; the user prioritized performance and deferred causal accuracy expansion. Aggregate scores do not prove exact generation equivalence. Non-CP, different CP sizes, different NCCL scheduling, the failed native RPC boundary and unexecuted cancellation/recovery probes, a longer soak, and a rebuilt release wheel must receive their own gates if they become rollout requirements. The current validation package overlays scoped Python on identical frozen native libraries; it is not a build of unrelated current C++ WIP.

## Assessment

The scoped Prefill patch can be submitted with the fixed P2 findings and the explicit default-off wire gate. Unrestricted production release is not signed off: the fatal native RPC negative case and skipped recovery probes remain open, and release CI must build the scoped commit.
