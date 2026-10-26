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

Initial scope: single rank, BF16 dense causal MHA/GQA, BF16 KV cache, base RoPE,
head dimension 64/128/256, SwiGLU and ordinary RMSNorm. Quantized linear layers,
zero-centered norms, speculative decoding, beam expansion/reordering,
layer micro batching, disaggregated FFN and context parallelism are rejected. TP/DP/PP/EP/SP larger
than one and model execution classes without the capability are rejected.

## Arithmetic and transport

* Linear uses a fixed 16x64x32 GEMM tiling with FP32 accumulation and no split-K
  or shape-dependent autotuning. LM head uses the same implementation, including
  when called through C++.
* RMSNorm and Q/K norm use a fixed per-row reduction.
* Prefill and decode share one attention implementation. Every query visits
  logical causal KV blocks of 32 in the same order, independent of other queries,
  physical page addresses and future tokens. RoPE uses a fixed cache.
* Vocabulary logsumexp uses fixed 1024-element partitions and a fixed merge.
  Native prompt scoring and generation consume these same normalizers.
* RPC carries the selected FP32 raw logprob directly. The token client preserves
  its bits instead of reconstructing it from a rounded/underflowed probability.
  Older servers without this field retain the existing probability fallback;
  that fallback does not provide this feature's strict transport contract.

These kernels prioritize an auditable numerical path. They run batched work on
the GPU, but are not a throughput optimization. Further tuning must retain the
same reduction topology or introduce a separately validated profile.

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

The initial validation on L20B / SM 10.3, Torch 2.11 / CUDA 13 includes:

* 16 CUDA/operator-policy checks and 17 token RPC client tests.
* 18 full-model checks on Qwen3-8B, comparing hidden states, full-vocabulary
  logits and selected logprobs. Every comparison has zero unequal bytes.
* The same 18 checks on a seeded small Qwen2 fixture with QKV bias and without
  Q/K norm, plus a separate Hugging Face numerical reference. This validates
  an additional execution structure, not a pretrained Qwen2 checkpoint.
* 27 native batch dispatcher tests, including raw logprob transport when the
  corresponding probability underflows to zero.
* A bounded native RPC check with FIFO/eager and ratio/decode-Graph modes,
  serial/concurrent scoring, request reordering and decode-versus-prefill
  rescoring. The Graph capture buckets are 1/2/4/8.

These are bounded conformance checks, not exhaustive qualification of all
sequence lengths, checkpoints or architectures. The initial model checker uses
a 65-token target sequence; attention operator checks extend to 97 tokens.
