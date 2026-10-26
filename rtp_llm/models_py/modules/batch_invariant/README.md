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

* On Hopper/Blackwell, BF16 Linear with N/K divisible by eight uses cuBLASLt with operation-local FP32
  compute, zero workspace and explicit rejection of split-K/reduction algorithms,
  following vLLM's BLAS strategy. The Blackwell M=1 heuristic follows PyTorch
  2.11's GEMV avoidance. A bounded descriptor cache contains no weight pointers.
  Odd feature widths, other architectures and FP32 operator tests use the adapted vLLM persistent
  Triton GEMM with fixed K partitions. LM head uses these same paths with FP32
  output, including when called through C++. No global BLAS flags are changed.
* RMSNorm and Q/K norm use a fixed per-row reduction.
* Prefill and decode share the causal BF16 subset of vLLM's 2D unified
  attention. Tensor Core QK/PV and grouped query tiles share KV loads. Logical KV
  blocks of 32 retain the same traversal/softmax partition in both phases; no
  split-KV path is enabled. The adapter computes ragged query offsets on device
  and supplies absolute positions and HND cache strides. RoPE uses a fixed cache.
* Vocabulary logsumexp uses fixed 1024-element partitions and a fixed merge.
  Native prompt scoring and generation consume these same normalizers.
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

The vLLM-derived revision was validated on L20B / SM 10.3, Torch 2.11 / CUDA 13:

* 24 CUDA/operator-policy checks and 17 token RPC client tests passed. Coverage
  includes GEMM tile boundaries, sliced-buffer alignment, row/column-major
  weights, live weight updates, unchanged global BLAS settings, ragged attention,
  GQA ratios including seven, head dimensions 64/128/256 and KV lengths to 1025.
* Qwen3-8B passed 26 full-model checks with zero unequal bytes in hidden states,
  full-vocabulary logits and selected FP32 raw logprobs. The 20 standard checks
  use 65 tokens; six extended checks compare the final position of a 1025-token
  sequence across peers, page placement and chunk boundaries.
* A seeded small Qwen2 fixture with QKV bias and without Q/K norm passed the
  same 20 standard checks and a separate Hugging Face eager numerical reference.
  Maximum absolute differences were 0.01163 for logits and 0.005094 for logprobs.
  This is structural coverage, not acceptance of a pretrained Qwen2 checkpoint.
* Native RPC passed FIFO/eager and ratio/decode-Graph checks, including real
  multi-request batches, Graph padding, serial/concurrent scoring, reordering,
  decode-versus-prefill rescoring and equality across both server modes.
* Host and device metadata planners each passed full-model Graph capture/replay
  with changing positions. A live graph must retain its capture-time planner;
  switching between the planners' buffer-capacity conventions is unsupported.

The original implementation at `08bd92d9` was validated on L20B / SM 10.3,
Torch 2.11 / CUDA 13 as follows (historical results, not a substitute for
revalidating the updated kernels):

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
sequence lengths, checkpoints or architectures. The v1 attention operator checks
extended to 97 tokens; the updated checks extend to 1025 tokens. Other GPUs and
cuBLAS releases must be qualified independently. No RL effect or end-to-end
rollout performance acceptance is implied.
