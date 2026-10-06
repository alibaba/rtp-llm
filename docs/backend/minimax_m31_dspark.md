# MiniMax-M3.1 DSpark

This integration uses real preview2 DSpark weights and real execution. There is
no mock model path. The draft shares the target embedding and LM head, has five
dense-FFN sliding-window transformer layers, and stores BF16 draft KV separately
from the target's packed NVFP4 main KV and indexer K.

## Explicit support boundary

This MiniMax-M3.1 integration requires CUDA. Other model families retain their
existing HIP sampler routing; strict sampled rejection is not claimed for HIP.

The released checkpoint contract uses weight-plus-one Gemma RMS normalization,
projection followed by hidden normalization, non-causal query blocks and a
checkpoint-defined sliding window. Target feature capture uses the configured
layers3,17,31,45,59. The old `M31_DSPARK_CANDIDATE_MATH` provisional selector
is no longer read. The confidence head is used only by adaptive verification;
static verification does not use it. See the [switch audit](minimax_m31_env_switch_audit.md)
for exact current mode/budget semantics and compatibility settings.

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
SP_DSPARK_VERIFY_MODE=adaptive
SP_DSPARK_VERIFY_TOKENS=4
SEQ_SIZE_PER_BLOCK=128
KERNEL_SEQ_SIZE_PER_BLOCK=128
```

Explicit `SP_DSPARK_VERIFY_MODE=static` verifies all seven candidates plus anchor
and requires VERIFY_TOKENS0 or7. Adaptive mode uses confidence to distribute a
batch-wide extra-row budget: VERIFY_TOKENS4 initially budgets4 extra rows per
request on average, not exactly4 candidates for each request. Zero selects the
full gamma budget. The seven-row draft backbone does not change. Keep mode and
budget consistent across PD roles and report both alongside MAL. An empty mode
retains legacy bool/fixed-prefix behavior; new deployments should set a mode.

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
RDMA readiness; source alignment and runtime dataset quality remain distinct gates.

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
soak. This historical snapshot predates the reference-alignment release and is
not validation of the current revision or a different verification mode.
