# Batch invariant execution (experimental)

Enable with `RTP_BATCH_INVARIANT=1` or `--batch_invariant true` when starting
the server. Standalone Python accepts
`AutoModel.from_pretrained(checkpoint, batch_invariant=True)`.
The default is disabled. Rebuild the native libraries and regenerate the Python
RPC bindings together with the Python source; mixing old native libraries with
this mode is rejected at model initialization.

## Contract

For a fixed checkpoint, GPU architecture, software build, dtype and parallel
topology, the raw FP32 logprob of a fixed target token conditioned on the same
token prefix must retain its exact bits across supported batch compositions,
request orderings, prefill chunks, prefill/decode transitions and KV page
placements. There is no rounding or tolerance in the invariance assertion.
Numerical accuracy against a separate implementation is checked separately.

This does not imply bitwise equality with vLLM, a different GPU/software build,
a different parallel topology, or a different model weight version. Sampling
RNG independence is a separate contract: use fixed token continuations to test
this feature. It also does not establish RL training quality or throughput.

## One configuration across models

`BatchInvariantPolicy` is an immutable per-model policy. A scoped construction
context lets existing shared factories select their implementations, then each
module retains the policy; changing another model's setting cannot modify an
existing instance. There is no process-wide monkey patch.

The initial `dense-gqa-v1` execution capability is declared by `Qwen3Model`, the
shared dense execution class used by Qwen family loaders. The policy checks the
actual architecture/configuration, rather than matching checkpoint names.
Models using this execution class and the supported operations use the same
switch. A different execution class must explicitly declare a reviewed
capability and pass the same contract checks before it can be enabled. New
operation families (such as MoE or recurrent attention) require additional
implementations; a model registration alone is insufficient.

Initial scope: single rank, BF16 dense causal MHA/GQA, BF16 KV cache, unscaled base RoPE,
head dimension 64/128/256, SwiGLU and ordinary RMSNorm. Quantized linear layers,
zero-centered norms, speculative decoding, beam expansion/reordering,
layer micro batching, disaggregated FFN and context parallelism are rejected. TP/DP/PP/EP/SP larger
than one and model execution classes without the capability are rejected.

## Arithmetic and transport

* On Hopper/Blackwell, BF16 Linear with N/K divisible by eight uses cuBLASLt with operation-local FP32
  compute, zero workspace and explicit rejection of split-K/reduction algorithms,
  following vLLM's BLAS strategy. The Blackwell M=1 heuristic follows PyTorch
  2.11's GEMV avoidance. A bounded descriptor cache contains no weight pointers.
  Odd feature widths, other architectures and FP32 operator tests use the adapted vLLM persistent
  Triton GEMM with fixed K partitions. LM head uses these same paths with FP32
  output, including when called through C++. No global BLAS flags are changed.
* SwiGLU preserves vLLM's intermediate BF16 rounding after SiLU and before
  multiplication, within a single fused kernel. Ordinary execution is unchanged.
* The first input RMSNorm and Q/K norm follow vLLM's Triton reduction: fixed
  1024-element blocks, FP32 accumulation and `1 / sqrt`. Norms after residual
  addition use vLLM's scalar CUB reduction and `rsqrt`, with a block size that
  depends only on feature width. RTP supplies its already BF16-rounded sum.
* Prefill and decode share the causal BF16 subset of vLLM's 2D unified
  attention. Tensor Core QK/PV and grouped query tiles share KV loads. Logical KV
  blocks of 32 retain the same traversal/softmax partition in both phases; no
  split-KV path is enabled. The adapter computes ragged query offsets on device
  and supplies absolute positions and HND cache strides. RoPE uses a configuration-keyed cache with BF16 rounding before
  promotion to the FP32 storage required by the native adapter.
* Vocabulary normalization follows vLLM's fixed 1024-element reduction. The
  Python/native contract carries two FP32 values per row: maximum and the log
  of the shifted exponential sum. Prompt scoring and generation preserve
  `(logit - maximum) - log_shifted_sum`; precombining the normalizers changes
  rounding. Native contract version 2 rejects a version 1 library.
* RPC carries the selected FP32 raw logprob directly. The token client preserves
  its bits instead of reconstructing it from a rounded/underflowed probability.
  Older servers without this field retain the existing probability fallback;
  that fallback does not provide this feature's strict transport contract.

See [UPSTREAM.md](UPSTREAM.md) for source versions, hashes and adaptations.
Performance changes must pass the same zero-byte-difference checks and an
independent numerical reference. Kernel speedups alone do not establish
end-to-end rollout or RL training throughput.

## Validation

Operator tests (CUDA required):

```bash
python -m pytest -q rtp_llm/models_py/modules/batch_invariant/test_kernels.py
```

Full model, fixed-token validation:

```bash
python -m rtp_llm.models_py.modules.batch_invariant.validate_model \
  --model /path/to/checkpoint --output batch-invariant-model.json
```

The full-model checker compares hidden states, full-vocabulary logits and
selected logprobs as byte arrays, including changed metadata on actual Graph
replay. It never starts training, evaluation datasets or checkpoint creation.
RPC transport tests live in `rtp_llm/RL/test/token_rpc_client_test.py`.

The arithmetic alignment was qualified on 2026-10-27 using Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, L20B / SM 10.3, single rank,
RTP Torch 2.11.0 / Triton 3.6.0 and vLLM 0.29.0 Torch 2.13.0 / Triton 3.7.1,
CUDA 13. The vLLM reference enables `VLLM_BATCH_INVARIANT=1`, selects
`TRITON_ATTN`, uses BF16 LM-head operands and FP32 head output, and disables
prefix caching. Torch compilation is disabled (`compilation_config.mode=0`);
the timing run enables decode CUDA Graph. Ordinary vLLM AUTO attention and
other compilation modes are outside this comparison.

* A passive 65-token trace matched all 36 layers' inspected boundaries, final
  hidden states, 9,875,840 full-vocabulary FP32 logits and selected logprobs
  with zero unequal bytes. RoPE cache storage has different dtypes; promoting
  the reference BF16 cache to FP32 gives exactly the same operand bytes.
* Cross-runtime operator checks matched vLLM exactly: 64 normalization checks
  in each runtime, six RoPE cache configurations through position 4096,
  four SiLU input sweeps and 40 residual RMSNorm checks. The SiLU sweeps cover
  every finite BF16 gate encoding with four fixed up operands.
* Qwen3-8B passed 26 full-model batch invariance checks. They compare hidden
  states, complete vocabulary logits and selected logprobs, including 1025-token
  prefixes, chunk boundaries, KV page placement and live Graph metadata changes.
* Native RPC matched the reference for all 93 fixed continuation tokens and
  four greedy sequences of 32 tokens each, including every returned FP32 raw
  logprob. Serial, concurrent and reversed-order scoring matched exactly.
* Native FIFO/eager and ratio/decode-Graph modes passed real multi-request
  prefill/decode batches, reordering, decode-versus-prefill rescoring and
  cross-mode equality. Graph buckets were 1/2/4/8; execution logs verified
  that actual Graph replay and batches larger than one occurred.
* A seeded small Qwen2 fixture with QKV bias and without Q/K norm passed the
  same 20 standard checks and an independent Hugging Face eager reference.
  Maximum absolute errors were 0.012001 for logits and 0.003486 for logprobs.
  This is structural coverage, not pretrained Qwen2 acceptance or cross-engine
  bitwise qualification of that checkpoint.
* Native dispatcher tests include probability underflow and a large-logit
  cancellation case that distinguishes `(x - max) - log_sum` from
  `x - (max + log_sum)` for prompt targets, prompt top-k and decode outputs.

The prior implementations are preserved at `08bd92d9` (initial feature) and
`e5e28b98` (vLLM-derived GEMM/attention). Their historical measurements do not
qualify this updated arithmetic. These are bounded conformance checks, not
exhaustive qualification of every sequence length, checkpoint, architecture or
software release. TP greater than one is unsupported. No RL quality or training
throughput acceptance is implied.
