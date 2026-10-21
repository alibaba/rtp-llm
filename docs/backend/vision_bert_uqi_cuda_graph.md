# Embedding CUDA/HIP Graph

Embedding graphs capture the Transformer backbone. Token embedding, model-supported
visual preparation, pooling, classifier heads and response processing execute
outside capture. BERT/UQI is a model adaptation, not the execution contract for
all embedding models. This file retains its historical
name to preserve existing links.

## Configuration and support boundaries

```bash
export ENABLE_CUDA_GRAPH=1
```

This is the only embedding Graph enable switch. The executor selects
`EMBEDDING_PREFILL`; neither a BERT environment variable nor the presence of an
optional Python method selects capture scope. Decode, MTP and generative
full-prefill retain their separate existing execution paths.

`BaseModel.load()` retains the existing `support_cuda_graph()` precheck for
language-model tasks. For embedding, `PyWrappedModel` checks
`embedding_cuda_graph_unsupported_reason()` after Python model initialization,
before constructing the Graph runner. This check validates the split interface
and backend requirements for embedding prefill, excluding MTP. An unsupported
model/configuration uses eager with a diagnostic before Graph allocation/capture.
This capability query is not another configuration switch. Unexpected
capture/kernel failures are not swallowed as capability failures.

The common cache-free backend supports FP16/BF16 activations, MHA/GQA, configured
causal or noncausal attention, and either no RoPE or base RoPE with zero offset
and a finite scale of at least one.
MLA, sparse/logn attention, custom attention scaling, other RoPE styles, context
parallelism and headwise attention require additional adaptations. A model-level
`support_cuda_graph()` flag alone does not establish embedding backend support.

The CUDA ragged Graph path uses `attn_configs.dtype` for Q/K/V. Eager dtype
selection can instead select FP8 based on KV-cache settings; enabling Graph does
not preserve that eager FP8 selection. Validate numerical behavior with the
actual precision used by each path.

Non-empty request `input_embeddings` overrides are ineligible for embedding
Graph replay. Eager support remains model-specific; BERT rejects these overrides.
Model-owned visual injection is a separate supported BERT adaptation.

| Model | Preparation outside capture | Captured backbone |
| --- | --- | --- |
| Ordinary BERT | Word/position/type embedding and embedding LayerNorm | BERT layers |
| VisionBert/UQI | BERT preparation and visual injection | BERT layers with model-specific user-segment attention |
| Other models | Must implement the split contract | Eager until the model and backend pass capability checks |

The execution contract is model-independent. A new embedding model must implement
the split interface and validate its attention semantics before enabling replay.
Backend capabilities alone do not establish end-to-end model support.

## Model and attention contracts

The interfaces are defined on `GptModelBase` and implemented/overridden by models:

| Interface | Responsibility |
| --- | --- |
| `prepare_transformer_inputs(inputs)` | Return real-token hidden states `[tokens, hidden_size]` outside capture |
| `forward(inputs, attention)` | Reuse the eager backbone; nonempty `inputs.input_hiddens` supplies prepared embeddings and skips embedding computation |
| `prepare_embedding_attention_inputs(attention_inputs, host_tokens, host_text_mask)` | Reuse supplied metadata or derive missing model metadata from host views, without reading deferred H2D storage |
| `validate_embedding_graph_inputs(inputs)` | Side-effect-free eligibility check before replay planning |
| `prepare_embedding_graph_attention(inputs, is_cuda_graph)` | Create bucket-owned Attention buffers and its startup plan |
| `prepare_embedding_graph_replay(attention, graph_inputs, request_inputs)` | Replan outside capture using request metadata and fixed Graph storage |

The Runner binds the split interface for every `EMBEDDING_PREFILL` model. A model
without that contract never silently falls back to capturing its entire forward.
`CudaGraphRunner` does not inspect user-profile settings, segment token IDs or
UQI bounds. BERT validates and passes its own segment metadata through these
hooks. UQI-specific Graph masks and layouts remain in the backend adapters;
the retained C++ host metadata producer is described below.

Replay follows this sequence:

1. Gather host int32 tokens and lengths. On CUDA, preserve the existing device
   conversion unless BERT UQI two-pass is enabled. For enabled cache-free embedding
   graphs, `PyWrappedModel` obtains host planning views without changing the
   model-facing inputs; eager fallback can reuse the resulting metadata.
2. Submit queued metadata H2D copies before the Runner reads device buffers.
3. Validate the request and select a token bucket. Ineligible requests use eager.
4. Wait for prior replay before reusing bucket-owned input/metadata buffers.
5. Prepare real-token embeddings outside capture, copy into the fixed hidden-state
   buffer and clear the unused token tail. Prepare the bucket's Attention plan.
6. Replay the captured Transformer backbone, return only real-token rows, then
   run the existing pooling or classifier. CLS, last-token and mean pooling are model/head choices.

Warmup and capture use the common call `py_forward_method_(inputs, attn_pyobj)`.
For embedding, the Runner reuses `forward` and aliases the local
capture inputs' `input_hiddens` to the existing bucket-owned `transformer_input_`
buffer. Request preparation continues writing into that buffer before replay;
its address and ownership remain unchanged.

Embedding attention replanning currently goes through the model's
`prepare_embedding_graph_replay` hook. BERT first updates request segment bounds,
then the base implementation calls each attention implementation's
`prepare_cuda_graph()`. Other Graph roles use C++ `callPrepareCudaGraph()`.

Planning and input preparation are both outside capture; their relative order
can differ when the caller pre-plans Attention. Repeated eligibility checks must
not invalidate already prepared metadata for the same request.

## Fixed buffers, positions and capacities

Configure existing prefill token buckets and `CONCURRENCY_LIMIT`. A bucket is an
aggregate token capacity, while `max_seq_len` limits each real sequence. The
request capacity must cover all sequences in a merged model batch. Exceeding
request or token capacity falls back to eager.

Each bucket owns a fixed hidden-state buffer, output hold, Attention workspace
and cumulative-length buffers. Real sequences and padding are distinct ragged
sequences, preventing padding from participating in real-token attention.

Embedding Graph support extends the existing device Attention implementations:
`PyFlashinferPrefillImpl` and its ragged op on CUDA, and
`AiterPrefillImplNonAsm` and its varlen op on HIP. The factory selects their
Graph capability from the registered backends. Shared host-length and RoPE
helpers live in `common.py`; `fmha_impl_base.py` only defines interfaces.
The device implementations preserve the model's causal setting. For base RoPE, fixed position buffers
reset at each real sequence; padding positions are zero. Requests with explicit
position IDs fall back to eager unless the model provides a position adapter.
Each backend owns a RoPE table generated by the engine's canonical implementation,
preserving configured scaling and partial rotary dimensions without sharing a
process-wide cache across models. QK norm
stays in the model layer before Attention; final normalization stays in the
model backbone. There is no paged KV cache requirement.

Startup profiles distribute aggregate bucket tokens across legal request slots.
Fixed padding slots split the unused tail into sequences bounded by
`max_seq_len`, avoiding bucket-sized padding Attention and out-of-range RoPE
positions. Actual requests are validated against the same per-sequence limit.
Startup profiles are not inference requests or accuracy references.

The Runner remains serialized. Stable input addresses and synchronization do
not enable concurrent replay or extend returned output lifetimes.

## BERT-specific user-segment adaptation

Only a BERT service using user-profile semantics should set:

```bash
export USE_VISION_BERT_UQI_BLOCK_MASK=1
```

The legacy host scan in `PyWrappedModel` requires block mask enabled,
`VISION_BERT_UQI_TWO_PASS` unset or `1`, contiguous CPU int32
`inputs.combo_tokens`, and a CPU text-token mask when supplied. It finds the first
text `CLS_UQI` (default token 2) through the first following `SEP` (102), inclusive,
ignoring visual positions. The model hook reuses native bounds and derives them
only when usable host bounds were not supplied. Without SEP,
the segment extends to sequence end. `VISION_BERT_CLS_UQI_TOKEN_ID` overrides the
start marker. Tokens outside that segment form A; user-profile tokens form B.

| Query | A keys | B keys |
| --- | --- | --- |
| A | Visible | Hidden |
| B | Visible | Visible |

Pure QI has no B tokens and reduces to full bidirectional attention. Different
requests never attend to one another. The BERT validator checks contiguous CPU
int32 bounds, batch cardinality, nonnegative lengths and valid sequence ranges
before Graph planning. Whenever the legacy scan runs, C++ also precomputes the
permutation, inverse permutation, B rows and indptr arrays, including when Graph
is enabled. Pure-QI batches still get indptr metadata but no token permutation arrays.
Graph mode prepares its own mask/layout from the bounds; precomputing the eager
schedule does not select the Graph execution algorithm. An eager fallback reuses
the native schedule, or builds a missing schedule from bounds in Python. This
retained optimization is a BERT-specific exception in the shared wrapper; it does not select Graph capture scope or add a Graph enable switch.

CUDA uses `BertUqiEncoderGraphImpl` in the existing CUDA BERT Attention module:
one FlashInfer custom-mask call per layer, without token permutation. Every
plan supplies a mask, including startup and pure-QI plans, so the captured path
is stable. Dense mask construction has transient quadratic cost; packed device
storage is bounded conservatively by `T*T/8` plus per-sequence rounding. The
installed FlashInfer must support graph mask buffers and replanning.

HIP uses `BertUqiRocmEncoderGraphImpl` in the existing ROCm BERT Attention module:
stable A/B permutations, separate-segment Attention and full-sequence Attention,
then selection of A/B outputs and padding zeroing. The model restores original
token order after the backbone. Padding sequences are bounded by the model's
sequence profile.
Pure QI still executes both Attention calls on this UQI path.

`VISION_BERT_UQI_TWO_PASS` selects eager BERT Attention and gates the legacy C++
scan/schedule producer. Disabling it selects the CUDA custom-mask eager path;
that eager configuration is unsupported on HIP. When native bounds are absent,
the model metadata hook derives Graph bounds from host planning views. This
variable does not enable Graph or select the Graph backend.

## Tests and validation

Existing test files cover host layouts/bounds, common causal planning and RoPE,
model-owned eligibility, and fixed-buffer Runner
replay. Tests import actual modules instead of extracting AST definitions.
GPU tests cover causal/noncausal replay through the existing device prefill
implementations, BERT backbone replay and model-specific user-segment attention,
including GQA and FP16/BF16, compared against dense or eager oracles.

The standalone Runner test models use the same `forward(inputs, attention)`
entry as eager execution. Their `forward` consumes prepared `input_hiddens`;
embedding preparation remains outside capture. Models that implement
`prepare_transformer_inputs` must honor this prepared-input contract in `forward`;
models without that preparation contract remain on eager execution.
The target list describes test coverage, not a successful native GPU test run.

Relevant targets:

- `//rtp_llm/models_py/model_desc/test:bert_encoder_graph_metadata_test`
- `//rtp_llm/models_py/model_desc/test:bert_uqi_graph_layout_test`
- `//rtp_llm/models_py/model_desc/test:bert_uqi_encoder_graph_test`
- `//rtp_llm/models_py/model_desc/test:bert_uqi_encoder_graph_test_rocm`
- `//rtp_llm/cpp/cuda_graph/tests:cuda_graph_tagged_cache_test`
- `//rtp_llm/cpp/embedding_engine/test:embedding_input_metadata_test`

Required deployment validation includes normal BERT, VisionBert QI/UQI/mixed
batches, changing request lengths/buckets, capacity fallback,
visual input changes, repeated replay, and model-specific pooling. Compare eager
and Graph outputs with identical checkpoint, tokenizer, dtype and quantization.
Operator tests and code-level adaptation do not establish checkpoint score
accuracy or a performance gain. Previously observed score differences and the
reported NVIDIA illegal access have not been diagnosed by this refactor.

Pooling and classification remain in postprocessing. First-token indices are
computed there from input lengths; the executor does not pre-upload CLS indices.
HTTP profile parsing and RPC field compatibility are independent features;
their semantics are not part of the Graph contract.
