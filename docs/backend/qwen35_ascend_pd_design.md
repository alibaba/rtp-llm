# Qwen3.5 prefill and decode disaggregation on Ascend

This proposal adds single-host prefill/decode (PD) serving for
Qwen3.5-35B-A3B in BF16 on Ascend 950. It reuses RTP-LLM's model definitions,
hybrid cache managers, PD lifecycle and routing. New work is concentrated in
Ascend execution, device-aware cache transport, UB transfer and graph replay.
This document is a design proposal, not a declaration of Ascend support or a
performance result.

The first hardware milestone uses two cards, one prefill instance and one
decode instance, both TP=1. The implementation must also permit 2P1D, 1P2D and
2P2D. Those configurations require three or four available cards for acceptance;
multiple processes sharing two cards do not establish their performance.

## Audited baseline

The source audit was performed on 2026-10-21 against upstream main commit
[`bcd02fdb056bdf42714b697c1aab0260501f3aa6`](https://github.com/alibaba/rtp-llm/commit/bcd02fdb056bdf42714b697c1aab0260501f3aa6).
No target machine, runtime image or model checkpoint was available during the
audit. Source coverage below is not a substitute for running the target model
on the target device.

| Area | Verified upstream evidence | Reuse and remaining work |
| --- | --- | --- |
| Qwen3.5 model | `Qwen35Moe` and `qwen35_moe` registration in `rtp_llm/models/qwen3_next/qwen3_next.py`; checkpoint mapping in `qwen3_next_weight.py` | Reuse configuration, weight mapping and model structure. The Python model allowlist currently admits CUDA, ROCm and PPU, not Ascend. |
| Qwen3.5 PD | `rtp_llm/test/smoke/data/model/qwen35/qwen35_bf16_tp2_dp2_pd.json` asserts `pd_sep=true`; `suites_h20_oss.bzl` wires the fixture | Reuse the smoke harness and golden comparisons. This is not evidence for Ascend TP=1, 64K or UB. |
| Hybrid state | `CoordinatorCacheManager`, `LinearCacheManager`, `KVCacheTransferPlanner` and tagged cache groups under `rtp_llm/cpp/cache/` | Reuse full-attention and linear-state ownership, sparse checkpoints and transfer planning. Validate the actual Ascend layouts and state dtypes. |
| PD lifecycle | `PrefillRpcServer`, `DecodeRpcServer`, `P2PConnectorScheduler*`, `P2PConnectorWorker*` | Reuse admission, per-layer publication, asynchronous completion and cancellation; extend their existing protocol instead of adding a second router. |
| Transfer abstraction | `IKVCacheSender`, `IKVCacheReceiver` and `TransferBackendFactory` under `rtp_llm/cpp/cache/connector/p2p/transfer/` | Add an Ascend device transport here. The public factory implements TCP and explicitly rejects Barex RDMA in this build; neither establishes UB support. |
| Prefix reuse | Local block-tree cache, `LinearCacheManager`, reuse lengths and cache-event publication | Reuse local prefix matching and cache ownership. Verify P/D checkpoint agreement and skipped bytes independently. Shared remote cache storage is a separate feature. |
| Routing and APIs | FlexLB, OpenAI endpoints, request cancellation and PD tracing | Reuse their service contracts. Add only missing state/transport metrics and verify prefix affinity against live cache availability. |
| Graph execution | CUDA/ROCm graph paths; merged full-prefill graph work in PR #1360 | Reuse capture policy concepts and tests. Ascend capture/replay requires its own implementation and runtime evidence. |

Relevant merged work includes [#1316](https://github.com/alibaba/rtp-llm/pull/1316)
(Qwen3.5 PD/MTP smoke and fixes),
[#1248](https://github.com/alibaba/rtp-llm/pull/1248) (PD reuse block size),
[#747](https://github.com/alibaba/rtp-llm/pull/747) (PD cancellation leaks),
[#1382](https://github.com/alibaba/rtp-llm/pull/1382) (block-tree cache),
[#1423](https://github.com/alibaba/rtp-llm/pull/1423) (independent cache pools and
stable group identity),
[#1398](https://github.com/alibaba/rtp-llm/pull/1398) (PD tracing), and
[#1360](https://github.com/alibaba/rtp-llm/pull/1360) (full-prefill CUDA graphs).
MTP remains outside this proposal; its tests provide integration references only.

### Ascend dependency status

On the audit date, [#1349](https://github.com/alibaba/rtp-llm/pull/1349) and its
follow-up [#1481](https://github.com/alibaba/rtp-llm/pull/1481) were open and
unmerged. PR #1481 targets the official `feature/ascend` branch. That branch,
at `003437913662bc4dfdd329f004ae0c43a2c0d5d5`, was 18 commits behind main with
no additional commits. The branch name alone therefore does not establish an
available Ascend implementation.

The PR descriptions report torch_npu/CANN device and build support, MHA and
BF16 operator paths. They explicitly describe graph capture as unavailable and
several operators as functional fallbacks. Their changed-file list does not
establish an Ascend Gated DeltaNet implementation or a UB transport.

Unmerged patches are research references, not implementation dependencies.
Use merged upstream commits or original changes. Recheck these PRs and their
successors before changing shared interfaces. A supplied working Ascend image
must be audited and preserved with its exact source provenance before use.

## Scope and admission gates

The initial scope is text-only requests, BF16 weights, TP=1 per instance,
single-host UB connectivity, 8192/65536 input tokens and 1024 generated tokens.
The service context limit must be at least 66560 tokens. Multimodal requests,
EP, asymmetric TP, multi-host transfer, quantization and speculative decoding
are excluded from the first implementation.

Before serving, record the exact card SKU, HBM capacity, device IDs, UB
topology, driver, firmware, CANN, torch_npu, framework commit, image digest,
model revision, launch command and ports. Confirm that the assigned devices,
ports and worker processes are available. Preserve an existing working runtime;
source changes must not silently replace its packaging or dependencies.

Single-card admission must account for the complete model weights, all cache
groups, graph workspace, activations, communication buffers and an explicit
reserve. Approximately 35 billion BF16 parameters already require about
70 GB before these additions. If TP=1 cannot fit, report the failed deployment
constraint; do not silently substitute TP=2 or quantization.

## Hybrid state contract

The [official model configuration](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)
has 40 text layers: 10 full-attention layers and 30 linear-attention layers.
Full attention uses two KV heads with head dimension 256. With BF16 KV, the
full-attention payload is 20480 bytes per token, or 160 MiB for 8192 tokens and
1.25 GiB for 65536 tokens. These figures exclude linear state, alignment and
metadata; they are not total device-memory estimates.

Transfer and cache reuse must preserve:

1. Full-attention K/V blocks and their position information.
2. Linear-attention convolution state and recurrent state at the committed
   token boundary, including their real storage dtypes.
3. Stable cache-group tags, layer identity, physical block geometry, model
   revision and the token boundary represented by each state checkpoint.

Use the existing typed cache layout and group tags to derive buffer descriptors.
Do not infer a linear state from the ordinary KV/token formula or assume all
groups use BF16. Reuse the current tail-state transfer policy; skip full KV
prefix blocks only when the destination holds a compatible committed prefix.
The source's final recurrent state still needs transfer after processing a
nonempty suffix, even when the destination has the original shared prefix.

P/D negotiation must reject incompatible models, layouts, dtypes, block sizes,
protocol versions or token boundaries before decode admission. Checksums in
test mode cover every state type, not only full-attention KV.

## Execution and transport design

### Reuse the existing PD control flow

Keep FlexLB and the existing P/D RPC roles. The decode side reserves cache
capacity before prefill work is committed. Waiting for state transfer must not
place a request in the runnable decode batch. Publish each completed layer
through `P2PConnectorWorker::writeByLayer`; decode becomes runnable only after
all required groups are ready and their device completion events are visible.

The existing sender/receiver interfaces remain the transport boundary. Add an
explicit Ascend UB backend with a capability check and actionable startup
error. Selecting UB must never silently select TCP or host staging. Any TCP
bring-up mode must be named in logs and excluded from UB acceptance results.

### Device identity and memory lifetime

The current `BlockInfo` describes storage with `is_cuda`, and TCP copy helpers
construct `torch::kCUDA` views. These assumptions cannot represent NPU memory
safely. Preserve the device kind and device index from the originating tensor
through buffer registration and the copy/transfer boundary. Unsupported device
kinds must fail before host dereference or incorrect tensor wrapping.

Ascend memory registration, peer reachability and asynchronous completion must
use an API verified against the provided CANN/communication SDK. Do not infer
UB capability from a library name or an enabled HCCL option. The backend must
demonstrate the actual device-to-device path and report host-staged bytes.

Keep the source cache lease until device transfer completion, including when
the RPC is cancelled. A timeout is not proof that a DMA operation has stopped.
Invalidate an attempt's destination buffers until completion is acknowledged
or safely drained. Bound in-flight bytes and buffer allocations independently
from the decode batch size. Registration failure is fatal for a direct-device
backend rather than a warning inherited from TCP behavior.

### Identity, retries and cancellation

Extend existing request/layer keys with an attempt or generation identity when
needed so a delayed completion cannot satisfy a retried request. An attempt
may commit its state only once. Repeated messages are idempotent; conflicting
metadata and unexpected blocks are errors. Retry only before releasing the
source state needed by that retry.

Reuse the existing cancellation and resource-store lifecycle. Fault injection
must cover cancellation during prefill, transfer and decode; peer termination;
late completion; duplicate delivery; and timeout while transfer is in flight.
For a stream that already emitted tokens, the initial recovery policy is an
explicit terminal error. Transparent restart is out of scope until token replay
and generation-state continuity are specified and tested.

## Prefix reuse and graphs

Reuse local immutable prefix entries and linear-state checkpoints. A lookup
must return a boundary valid across all required groups. Keep prefix entries
pinned while an admitted request or transfer uses them. Report P and D reuse
separately, together with actual bytes omitted from transfer. Prefix affinity
must not override admission, stale-cache detection or overload limits.

Ascend execution needs full-attention, Gated DeltaNet prefill/decode,
causal convolution, gated normalization, MRoPE and BF16 MoE implementations
compatible with RTP-LLM's layout and factory conventions. Existing GPU kernel
imports and model allowlists must be isolated before admitting Ascend; adding
only a device-enum value is insufficient.

Graph replay follows eager correctness. Capture compute with stable cache
addresses and runtime-supported shape buckets. Run transport and admission
outside capture and establish device event dependencies before replay. P and
D have independent graph switches. A requested but unsupported graph mode must
fail clearly instead of running eager while reporting graph success.

## Delivery sequence

| Stage | Change | Exit evidence |
| --- | --- | --- |
| D0 | Design and upstream dependency audit | This proposal, source references and reviewed interfaces |
| D1 | Device-aware transfer boundary and Ascend execution prerequisites | Portable tests plus Linux build; preserved single-card eager baseline |
| D2 | TP=1 1P1D complete-state handoff | Two-card short greedy request, state comparison, cancellation and repeated-output stability |
| D3 | Direct UB backend and layer overlap | Verified device transport, buffer lifetime fault tests and measured transfer decomposition |
| D4 | Prefix reuse and independent graph modes | 8K/64K correctness across the two-card feature matrix; eager/graph and cold/warm comparisons |
| D5 | Multi-instance deployment and acceptance | Three/four-card runs, agreed performance SLOs, fault recovery and 72-hour mixed load |

Development proceeds while the design PR is reviewed, but hardware-dependent
stages are not marked complete by static checks, mocks or documentation alone.
Changes should remain small enough to review independently, with each PR
stating its dependency and the exact validation completed.

## Validation and performance contract

Use existing unit, smoke and benchmark tooling before introducing new runners.
Run checks in this order: static/package checks, minimal startup, short
correctness request, repeated-output stability, then performance. Save the
baseline and candidate commits, runtime identity, inputs and raw measurements.

For two cards, test two input lengths, prefix on/off and all four independent
P/D graph combinations: 16 feature configurations. If only both-graphs-off and
both-graphs-on are initially supported, label that reduced matrix explicitly
as eight configurations. Across four PD ratios, complete independent graph
coverage is 64 configurations. Single and concurrent requests, streaming,
cancellation and fault recovery are additional test dimensions.

Compare 1P1D against two TP=1 colocated replicas using the same total cards,
checkpoint, dtype, request trace, enabled features and output length. Report
latency at controlled common load and peak goodput under agreed absolute
TTFT/ITL SLOs separately. A TP=2 colocated baseline may be reported as a separate
experiment, not substituted silently. Warm-prefix comparisons use identical
request tokens and arrival patterns with cache enabled and disabled.

Measure prefill-end to decode-ready latency separately from total transfer
duration and exposed transfer tail. Report full KV bytes, linear-state bytes,
reused bytes, retries, host-staged bytes and buffer peak. Evaluate UB efficiency
against a matched device-pair/payload/concurrency reference; zero-byte cache
hits are excluded from bandwidth efficiency statistics.

The requested targets (30% TTFT P99 reduction, 20% output-throughput increase,
TPOT P99 within 1.1x baseline, 50/300 ms handoff and 90% link efficiency) remain
acceptance targets pending a frozen measurement contract and device evidence.
No result from two cards establishes acceptance for 2P1D, 1P2D or 2P2D.

## Items to resolve with maintainers and hardware provision

- Ascend foundation: merge status, supported integration branch and available
  known-good image, without depending on unmerged community changes.
- Exact 950 SKU, HBM, peer topology and supported UB API/library versions.
- Whether text-only use may skip loading vision weights and how that setting
  is represented without changing the source checkpoint.
- Independent P/D graph support, required shape coverage and graph-memory budget.
- Absolute SLOs, concurrency/load sweeps, benchmark tool and dataset revisions,
  accuracy comparison rules and three/four-card acceptance availability.
- Whether final delivery requires merged code or an explicitly recorded
  maintainer acceptance of the implementation and its test report.
