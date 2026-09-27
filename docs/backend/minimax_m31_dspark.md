# MiniMax-M3.1 DSpark

This integration uses real preview2 DSpark weights and real execution. There is
no mock model path. The draft shares the target embedding and LM head, has five
dense-FFN sliding-window transformer layers, and stores BF16 draft KV separately
from the target's packed NVFP4 main KV and indexer K.

## Explicit support boundary

This MiniMax-M3.1 integration requires CUDA. Other model families retain their
existing HIP sampler routing; strict sampled rejection is not claimed for HIP.

Checkpoint metadata alone does not specify all training forward semantics.
`M31_DSPARK_CANDIDATE_MATH=gemma_causal_v1` is an explicit **provisional** opt-in,
not certification of training/reference alignment. It uses weight-plus-one RMS
normalization, projection followed by hidden normalization, causal query blocks,
and a 4095-token left window. Target feature capture uses the existing residual
boundaries for layers 3, 17, 31, 45, and 59. The loaded confidence head is not used
by fixed-width verification. Do not remove this opt-in based on benchmark
accuracy or acceptance length alone.

## Configuration

Set both PD roles consistently:

```text
MODEL_TYPE=minimax_m31
LOAD_METHOD=fastsafetensors
FORCE_CPU_LOAD_WEIGHTS=0
MOE_STRATEGY=mega_moe_nvfp4
NVFP4_KV_CACHE=1
FP8_KV_CACHE=0
SP_TYPE=dspark
SP_MODEL_TYPE=minimax_m31_dspark
SP_CHECKPOINT_PATH=<preview2-checkpoint>/dspark
SP_ACT_TYPE=BF16
SP_FP8_KV_CACHE=0
GEN_NUM_PER_CIRCLE=7
M31_DSPARK_CANDIDATE_MATH=gemma_causal_v1
SEQ_SIZE_PER_BLOCK=128
KERNEL_SEQ_SIZE_PER_BLOCK=128
```

`SP_DSPARK_VERIFY_TOKENS=0` verifies all seven candidates. An explicit value
1 through 7 verifies that prefix without changing the five-layer checkpoint or
seven-row draft backbone. Target verification and the subsequent commit use
`verify_tokens + 1` rows. Proposal width is not MTP depth or request batch size.
Keep this setting equal across PD roles. Report the verification budget alongside
MAL: existing fixed-acceptance metrics retain the generated width denominator.

Prefill uses eager CP4 with `PREFILL_CP_KV_CACHE_SHARDED=1`; Decode may use DP4
and target CUDA Graph. `RTP_LLM_DSPARK_CUDA_GRAPH=1` additionally opts into the
draft's separate exact-shape proposal/commit graphs; prompt seeding remains eager.
Target native FP4 readers and writers do not require BF16 history working pages.
The draft's BF16 KV is intentional, not an FP4 target fallback.

## Sampling and state

DSpark applies the Markov bias to each sampled predecessor and preserves the
actual proposal distribution for rejection sampling. Strict sampled rejection is
explicitly selected for DSpark; legacy MTP behavior is not silently redefined.
History-dependent verification uses committed history plus only the preceding
proposal tokens at each position. Rejected KV slots are overwritten on reuse.
Terminal output publication prevents late asynchronous updates from appending
more tokens before the scheduler consumes the completion event.

Force-accept settings are performance diagnostics only. They must be disabled for
functional validation, dataset accuracy, and acceptance measurements.

## Validation requirements

Run checkpoint/schema, draft SWA/KV, native FP4 writer/reader, prefix restoration,
MoE chunk/packing, sampler, verify-budget, graph and terminal-update regressions.
GPU tests must execute on the required architecture, not silently skip.

Release evidence must identify the exact source, compiled libraries, checkpoint,
PD topology, graph buckets, sampling fields and dataset IDs. Require normal
rejection PD smoke, GSM8K and long-context evaluation, complete response/route
accounting, and healthy idle ranks. Historical results and forced-accept timing
do not certify a newly changed revision. TCP PD validation does not establish
RDMA readiness; provisional training math remains a separate open gate.

### Local validation snapshot (2026-09-28)

CUDA13 build and 128 executed C++/CUDA regressions passed; four pre-existing
disabled tests were not executed. Native draft/FP4/MoE Python tests and CP4 mixed
cache synchronous/asynchronous TCP transfer checks also passed.

Real-weight PD4+4 used the configuration above with verification budget 5,
Prefill CP4/EP4, Decode DP4/EP4, target/draft graphs, and normal rejection:

| Check | Result |
| --- | --- |
| Greedy and sampled smoke across all four Decode ranks | 8/8 passed |
| Repetition, presence/frequency, and ngram history smoke | 12/12 passed |
| GSM8K, 500 requests, temperature 0.7 | Raw scorer 432/500; separate strict numeric/boxed-answer audit 465/500 |
| LongBench, 160 requests, greedy | Raw score 68.19; three length-limited responses retained in this score |
| Native accepted tokens per round | GSM8K 3.794; LongBench 2.997 |

Both dataset runs had zero request/integrity errors. GSM8K reached 16 active
client requests per Decode rank; LongBench used two. The numeric GSM8K audit is
a separately identified scoring contract, not a replacement of raw evidence.
These checks are not a full no-DSpark equivalence test or a prolonged production
soak, and do not remove the provisional math gate above.
