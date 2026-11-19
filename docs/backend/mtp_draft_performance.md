# Qwen4 MTP draft execution

For Qwen4 MTP with CUDA Graph enabled, set
`RTP_LLM_MTP_DRAFT_LOOP_GRAPH=1` before starting the server to capture the
successive draft forwards together with their native LM heads, TP logits
gathers, greedy sampling and recurrent input updates. The default is disabled.
With `--gen_num_per_cycle 3`, extend produces the first proposal and this graph
produces the remaining two. Target verification and extend keep their own
execution paths.

The initial supported model is `Qwen4ExpMTPModel`. Graph capture uses the
existing batch buckets and scratch cache page. Unsupported calls, warmup and
fake streams retain the ordinary forward path. The graph refreshes the live
device lengths used by QSA and RoPE between steps, including page crossings,
and restores the shared lengths before returning.

Two synchronization changes apply automatically to their supported paths:

- The asynchronous non-root MTP rank retains the completed round's tensor
  owners and defers its CPU wait until model staging buffers are reused.
  GPU completion is still required before reuse or teardown.
- QSA CUDA Graph uses device cumulative KV lengths. Its unused host total is
  computed only when preparation or forward falls back to an eager backend.
  Eager consumers continue to receive the exact total, including prefix lengths.

## Logits precision

BF16 native LM-head weights now produce BF16 logits and retain that dtype
through the TP gather. The public sampler interface receives FP32 logits after
communication. Set `RTP_LLM_BF16_LOGITS=0` before server startup to retain the
previous FP32-output BF16 GEMM and FP32 gather; the default is `1`. The policy
is fixed at wrapper construction, before graph capture.

BF16 output rounding can change token choices and MTP acceptance. This setting
does not promise output equivalence to FP32 logits. Compare accepted/proposed
counts and output token IDs separately; a similar mean acceptance rate alone
does not establish numerical equivalence.

## Sharded draft top-1

Supported ordinary MTP draft models select tokens from local vocabulary
shards by default. Each TP rank sends one 8-byte
candidate per row instead of its full logits shard, then selects the global
top-1 token. The LM-head GEMM and logits precision follow the existing policy.
Set `RTP_LLM_MTP_SHARDED_TOP1=0` before server startup to gather full logits
for debugging or performance comparisons. The policy is fixed before graph capture.

This path requires CUDA, TP greater than one, PP equal to one, and a draft model
with the MTP graph-step contract. Target execution, DSpARK, NaN-checking mode,
and calls requesting all logits retain the existing path. Compact outputs
with vocabulary padding also retain that path to preserve their output contract.
Draft-to-target token mapping and top-1 tie-breaking follow the existing
selection behavior.

## Implicit top-1 proposal probabilities

Set `RTP_LLM_MTP_POINT_MASS=1` before server startup to represent ordinary MTP
proposal distributions with token IDs instead of dense one-hot tensors. The
selected token has probability one and all other tokens have probability zero.
The setting applies independently of sharded top-1 and defaults to disabled.
It covers the first proposal, single/multistep drafting, whole-draft graph
replay, stream/device state, and P/D handoff. DSpARK retains its actual
probability distributions.

The rejection sampler consumes this point-mass distribution for both greedy
and stochastic target sampling. The proposal count is still configured when
creating the engine through `sp_config.gen_num_per_cycle`, exposed by
`--gen_num_per_cycle` or `GEN_NUM_PER_CIRCLE` at server startup.

Token-only P/D handoff sets the protobuf field
`propose_token_ids_are_point_mass`. Enable it on P only after D has been
updated to understand this field. Updated D accepts dense handoffs from
existing senders and can batch dense and token-only proposals together. P and
D may use different point-mass settings when both binaries support the field.

## RTX transport selection

On SM120 Blackwell RTX devices, RTP leaves transport selection to NCCL when
every visible device pair supports direct peer access. Older RTX devices,
missing peer access and failed capability probes retain the compatibility
fallback. An explicit `NCCL_P2P_DISABLE` value takes precedence. Confirm the
selected transport in NCCL logs for the actual deployment.

## Validation

Relevant regression coverage includes:

- `rtp_llm.cpp.models.test.native_lm_head_test`, run with each logits policy:
  dtype, actual two-rank gather, and CUDA Graph replay with changing inputs.
- `rtp_llm.cpp.models.test.mtp_draft_top1_test`: two-rank draft selection,
  BF16/FP32 logits, ties, non-finite values, vocabulary padding and mapping,
  fallback paths, and CUDA Graph replay with changing inputs.
- `rtp_llm.cpp.models.test.pywrapped_model_cache_store_integration_test`:
  device-length metadata and exact eager fallback after deferred totals.
- `rtp_llm.models_py.modules.qwen4_exp.test.qsa.decode_rope_stride_test` and
  `qsa_decode_graph_test`: changing lengths, page crossings and FP8 KV writes
  across successive draft steps.
- `rtp_llm.config.test.server_config_setup_test`: peer capability fallback,
  explicit transport overrides and frontend behavior.
- `SpeculativeSamplerTest`: fixed-input, fixed-random-draw comparisons of
  dense and implicit proposals for greedy/stochastic acceptance, rejection,
  residual sampling, token mapping, and forced acceptance.
- `MtpBatchStreamProcessorTest`, `PrefillRpcServerTest`, and
  `DecodeRpcServerTest`: mixed dense/token-only proposals, proposal widths,
  asynchronous device state, P/D handoff, and clearing stale dense buffers.

Measure unprofiled service throughput separately from profiler timings. For
before/after acceptance measurements, hold checkpoint, topology, quantization,
KV dtype, proposal count, request corpus and batching constant, and retain
repeated runs and raw acceptance counters.
