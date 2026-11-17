# ROCm ATOM full MTP draft-loop graph

Opt-in port of the full draft-loop graph from alibaba/rtp-llm PR #1697.
This is not the deferred-tail-wait experiment. The captured loop includes
both draft forwards, LM heads, TP gathers, greedy sampling and inter-step
metadata updates for `gen_num_per_cycle=3`.

## Installation and activation

Build RTP-LLM from the accompanying C++ changes. Install the pinned ATOM
`0.1.8+g9af739efc` wheel and existing Qwen compatibility/HC/TopK patches first,
then run:

```sh
python patches/atom/draft_loop/apply.py
python patches/atom/draft_loop/apply.py --verify-only
export RTP_LLM_MTP_DRAFT_LOOP_GRAPH=1
```

Set the variable only on the decode service; restart workers after changing it.
It defaults off. NVIDIA, other plugins, prefill,
fake/warmup requests and requests ineligible for CUDA graph keep the old path.
The hook checks shared device sequence-length storage and captured metadata.
Padded rows remain at zero; device lengths and all three token-major MRoPE
axes advance together between steps and are restored after each loop.
The current end-to-end validation uses text requests on an MRoPE model;
image requests are not part of this performance claim.
The recurrent input is the pre-collapse HC hidden tensor, not the LM-head input.
This branch's `computeLogits` already gathers TP logits: no second gather is added.

The installer checks the wheel version, exact source/patch/output hashes,
Python syntax and supports repeat application. It rejects unknown modifications.
The original wheel hash is unchanged. Apply base patches before this overlay.
Do not rerun the base patch verifier after overlays that change its tracked files.

## Validation contract

PD: one TP2/EP2 prefill and one TP2/EP2 decode; P reuse enabled, D reuse disabled;
MTP gen=3, async and async prepare enabled, post-rejection prepare disabled.
Fixed requests: input6656, actual cache hit5120, output347, concurrency1.
Run the existing 36-question correctness suite before unprofiled comparisons.
Use the same fixed request JSON, three warmups and 20 measured requests per round.
Report every round; acceptance outside 58–60% is flagged, not hidden.
Trace must show `executor.mtp.draft_loop_graph` with LM head, TP gather,
argmax and scatter nodes correlated to its graph launch.

## Validated fixed-workload results

TP2 P + TP2 D, gen=3, input6656 / actual reused5120 / output347 / C1;
P cache on, D cache off. Both modes use the same rebuilt RTP wheel and the
same installed ATOM patches. Only the D-side loop-graph flag changes.
Both modes passed 36/36 answers and 36/36 feature checks; all answer strings
were identical. Three candidate rounds used the same frozen 20 requests.

| Mode / round | TTFT ms | TPOT ms | Mean request acceptance |
| --- | ---: | ---: | ---: |
| Baseline 1 | 282.79 | 4.790 | 58.21% |
| Graph 1 | 285.80 | 4.622 | 59.67% |
| Graph 2 | 282.13 | 4.665 | 59.16% |
| Baseline 2 | 285.80 | 4.656 | 60.71% * |
| Baseline 3 | 281.25 | 4.800 | 58.61% |
| Graph 3 | 285.15 | 4.651 | 59.41% |

Baseline round 2 exceeded the preregistered 58–60% acceptance gate; it is
retained and flagged, not discarded. Across all three rounds per mode,
mean TPOT was 4.749ms baseline vs 4.646ms graph (-2.17%); mean TTFT changed
from 283.28ms to 284.36ms (+0.38%). Candidate acceptance was 59.16–59.67%,
and each candidate round mean TPOT was below 4.74ms. This is a per-round
mean result, not a per-request latency bound or an online-production result.
Decode elapsed time divided by actual draft rounds fell by about 1.89%;
this is a supporting execution-cost proxy, not a replacement for measured TPOT.
The original wheel bytes remain pinned; this feature is a hashed overlay.

Final trace validation: each TP rank showed 20 full-loop scopes and 20 loop
replays, zero old loop-iteration scopes, and 60 total graph launches (three
per cycle, versus four with separate draft steps). A completely correlated
GPU replay contains two LM-head GEMMs, two TP collectives and two argmax /
scatter sampling sequences. Kineto did not expose complete GPU-node
correlations for the other rank; no complete per-node claim is made for it.
