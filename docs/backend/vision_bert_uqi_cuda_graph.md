# Embedding CUDA/HIP Graph: current implementation and support boundaries

The goal is CUDA/HIP Graph execution for embedding workloads. Embedding is a
service task, not a particular attention architecture: a BERT encoder and a
Qwen3 model used for embedding can require different masks, positional encoding,
normalization and cache behavior. Enabling the embedding executor's Graph entry
point does not establish Graph support for every model or attention backend.

## Shared execution entry and current coverage

`EmbeddingExecutor` forwards the existing engine Graph configuration to
`PyWrappedModel` in embedding-prefill mode. `ENABLE_CUDA_GRAPH` controls whether
Graph capture is attempted. Token buckets and request capacities determine the
captured input shapes. Pooling, classifier heads and response processing run
outside the model Graph.

The current Transformer-only adaptation is implemented by `BertModel`:
`prepare_transformer_inputs` computes text/visual embeddings outside capture,
and `forward_transformer` runs the layer stack inside capture. `CudaGraphRunner`
selects this path through the model interfaces. Models without those interfaces
continue through their existing `forward` and attention-factory paths; they do
not automatically inherit BERT's Graph attention implementation.

| Model/path | Current implementation | Validation boundary |
| --- | --- | --- |
| Ordinary BERT | Transformer-only Graph with noncausal, cache-free attention | GPU/checkpoint validation is still required |
| VisionBert with UQI mask | BERT Graph with the CUDA or ROCm segmented-attention adaptation described below | CPU layout tests do not establish end-to-end GPU accuracy |
| Qwen3-0.6B used for embedding | Existing Qwen3 model/attention-factory path; no adaptation added here | Graph execution is not established by this change |

In particular, `Qwen3Model` has no `prepare_transformer_inputs` or
`forward_transformer` interface. Its normal ragged FlashInfer prefill backend,
`PyFlashinferPrefillImpl`, reports `support_cuda_graph() == False`; Graph
selection rejects that implementation. Paged alternatives require a suitable
KV-cache contract, while `EmbeddingExecutor` supplies no KV cache. Backend
selection depends on the device and configuration, so capture can fail before
request-time eager fallback is available. The model-level
`support_cuda_graph()` flag inherited by Qwen3 is not an embedding-Graph test.

Supporting Qwen3 embedding requires checking its actual attention configuration
and adapting a cache-free, Graph-capable backend while preserving RoPE, QK norm,
causal/noncausal semantics and final normalization. It also requires startup
capture, repeated replay with changing sequence lengths, and output comparison
against eager execution on the target GPU. This document does not claim those
checks have passed.

Do not set `USE_VISION_BERT_UQI_BLOCK_MASK` for Qwen3. It configures BERT user
segment semantics; shared C++ metadata code currently also reads this variable,
which is a remaining model-specific dependency in the common path.

## BERT-specific adaptation: VisionBert UQI

VisionBert is the embedding/reranking model used for the segmented-attention
adaptation below. It captures the Transformer stack and prepares request-owned
embeddings outside the graph. UQI attention preserves these visibility rules within each
request:

| Query | A keys: QI and vision | B keys: user profile |
| --- | --- | --- |
| A | Visible | Hidden |
| B | Visible | Visible |

B starts at the first `CLS_UQI` token (default ID 2) and ends at the first
following `SEP` (ID 102), including both markers. Without a following SEP, B
extends to the sequence end. Tokens after the closing SEP belong to A again.
A request without `CLS_UQI` is pure QI and uses full bidirectional attention.
Different requests and synthetic padding sequences never attend to each other.

## Configuration

For an embedding model with a compatible Graph backend, enable the shared switch:

```bash
export ENABLE_CUDA_GRAPH=1
```

For the VisionBert UQI example only, also configure the model's user-segment mask:

```bash
export USE_VISION_BERT_UQI_BLOCK_MASK=1
```

Configure the existing prefill token buckets and `CONCURRENCY_LIMIT` for the
workload. The Graph sequence capacity follows `CONCURRENCY_LIMIT`; it must cover
the total number of item sequences in a model batch, including any merged RPCs.
Requests exceeding that sequence capacity fall back to eager execution. Token
bucket capacity is an aggregate token limit; `max_seq_len` remains the
per-request sequence limit.
The ROCm startup profile also requires bucket capacity to fit within
`request_capacity * max_seq_len`.

`VISION_BERT_CLS_UQI_TOKEN_ID` overrides the default B start token ID.
`VISION_BERT_UQI_TWO_PASS` defaults to `1` and selects the eager implementation;
it does not select the Graph backend:

| Execution | CUDA | ROCm |
| --- | --- | --- |
| Graph | FlashInfer FA2 custom mask, one attention call per layer | AITER fixed-shape two-pass, two attention calls per layer |
| Eager, two-pass enabled | FlashInfer two-pass | AITER two-pass |
| Eager, two-pass disabled | FlashInfer custom mask | Unsupported; raises an error |

`enable_cuda_graph_` is the single Graph execution switch, configured by the
business at service startup. Embedding models exposing `prepare_transformer_inputs`
and `forward_transformer` automatically use Transformer-only capture, including
ordinary BERT. The model's `use_user_profile_mask` selects UQI attention semantics
and required segment validation; it does not independently enable Graph.
With `USE_VISION_BERT_UQI_BLOCK_MASK=1`, QI and UQI requests share the UQI Graph;
QI requests must supply zero B lengths. ROCm still performs two attention calls
per layer for pure QI requests.
Rebuild after C++ changes and restart the service to apply environment settings.

## Execution boundary

Outside capture and before each replay:

1. Gather tokens, sequence lengths and optional visual features.
2. Scan host tokens for B starts/lengths, ignoring non-text positions.
3. Prepare attention metadata in the selected bucket's fixed device buffers.
4. Compute text embeddings and embedding LayerNorm, inject visual features,
   copy hidden states into the fixed Transformer input buffer, and clear its tail.

Inside capture:

- QKV projection, attention and output projection in each BERT layer.
- Residual/LayerNorm and FFN operations, including configured quantized paths.
- ROCm UQI token permutation before the layer loop and inverse permutation after it.

The classifier/Mainse heads and response handling remain outside the graph.
The adapters require noncausal encoder attention without KV cache or nonzero
prefix lengths. Pure QI, UQI and mixed requests can reuse a token bucket's graph.

## CUDA backend

`BertUqiEncoderGraphImpl` extends the shared encoder implementation with a
FlashInfer ragged custom mask. Each sequence contributes a flattened logical
mask with `visible[q, k] = query_in_B or not key_in_B` in original token order.
No A/B permutation is needed on CUDA.

Every plan supplies a custom mask, including synthetic startup inputs and
pure-QI requests, so capture and replay use the same masked execution path.
Mask construction, packing and `wrapper.plan()` run outside capture;
`wrapper.run()` is captured. Each bucket owns its workspace, Q/KV indptr buffers,
packed mask buffer and mask-indptr buffer, preserving their device addresses.

For token capacity T, packed mask storage is conservatively bounded by about
T*T/8 bytes plus per-sequence rounding. Constructing the logical mask also has
transient quadratic cost. The installed FlashInfer must support
`custom_mask_buf`, `mask_indptr_buf` and custom-mask replanning with graph-enabled
ragged wrappers. Backend and bucket differences can change both performance
and numerical results relative to eager execution.

## ROCm backend

`BertUqiRocmEncoderGraphImpl` uses `aiter.flash_attn_varlen_func`; it does not
introduce a Triton attention kernel or allocate a dense attention mask.

Before the Transformer loop, stable indices rearrange each sequence into
`[A, B]`, preserving the original order inside each group. Every layer then runs:

1. Separate attention over A and B segments. Keep the A outputs.
2. Full attention over each complete A+B sequence. Keep the B outputs.
3. Select outputs using the B mask and clear padding outputs.

The two passes use the same QKV projection. After all layers, inverse indices
restore the original token order. Unlike eager two-pass, the Graph second pass
computes every query rather than gathering only the current B rows. This keeps
operation shapes fixed as B lengths change. Both calls remain captured even for
pure-QI requests. Zero-length request and A/B slots remain in the fixed indptrs;
the installed AITER must support these layouts.

Padding is split into independent synthetic sequences no longer than
`max_seq_len`. Startup planning similarly splits the synthetic bucket across
legal request slots instead of using a bucket-sized AITER sequence.

Permutation, inverse permutation, two indptr arrays, B mask and valid-token mask
share one reusable host byte buffer per bucket. Updates copy index ranges from
a cached identity array and perform one blocking H2D upload into persistent
device views. Completion before return makes host-buffer reuse safe. The ROCm
adapter does not allocate or upload the base class's unused cumulative lengths.

## Host metadata and eager fallback

On CUDA, `EmbeddingExecutor` preserves host tokens and lengths whenever UQI is
enabled, including `VISION_BERT_UQI_TWO_PASS=0`. Its final conversion step is
shared with the metadata regression tests in `EmbeddingInputMetadata.h`.
Ordinary QI retains the existing CUDA int32 conversion path.

`PyWrappedModel` always scans eligible host inputs when the UQI mask is enabled.
It builds the additional eager two-pass schedule only when two-pass is enabled
and prefill graphs are disabled. Graph planning uses B bounds directly. Eager
fallback can also construct a schedule from those bounds when the prebuilt
schedule was omitted.

The runner checks for correctly sized contiguous CPU int32 B-bound tensors and
rejects invalid counts/host-visible ranges before UQI Graph planning. It forwards
the current request's B bounds when the Transformer model uses the UQI mask; these
host tensor references are not captured GPU pointers. Requests missing required
bounds are ineligible for Graph. The supported eager path can derive missing
bounds from tokens; malformed requests and unsupported eager configurations can
still raise errors.

## Tests and validation status

| Target | Coverage |
| --- | --- |
| `//rtp_llm/models_py/model_desc/test:bert_uqi_graph_layout_test` | CPU segment bounds, mask semantics, permutation/inverse, padding, repeated packed-buffer updates and invalid-input rejection |
| `//rtp_llm/models_py/model_desc/test:bert_uqi_encoder_graph_test` | CUDA FP16/BF16 attention capture/replay, dense oracle, full buckets, changing segments, mixed/pure QI and stable mask pointers |
| `//rtp_llm/models_py/model_desc/test:bert_uqi_encoder_graph_test_rocm` | HIP equivalents, padding poisoned with NaN, zero padding output, stable metadata pointers and comparison with the eager AITER two-pass core |
| `//rtp_llm/cpp/embedding_engine/test:uqi_graph_without_two_pass_metadata_test` | Production metadata conversion with UQI and Graph enabled, two-pass disabled; verifies host residency, values and allocation preservation |
| `//rtp_llm/cpp/embedding_engine/test:qi_device_metadata_test` | Ordinary QI conversion to CUDA int32, including empty sequence lengths |

Both adapter targets include a 6144-token bucket with a 512-token per-sequence
limit. They also include a real two-layer BERT Transformer test comparing
Graph output with an independent dense masked eager attention oracle using the
same weights. This exercises projections, residual LayerNorm and FFN, but does
not cover checkpoint-level classifier scores or the full HTTP service.

The runner regression `test_uqi_graph_requires_valid_host_segment_bounds` in
`cpp/cuda_graph/tests/cuda_graph_tagged_cache_test.py` checks missing, invalid and
device-resident bounds, plus recovery to a valid replay. Its tagged-input setter
updates every tag and reassigns the mapping to refresh the binding's primary
input. The executor metadata targets run in separate processes because the
production environment decision is cached on first use. They test the conversion
stage called by `gatherModelInput`, without constructing a service or model.

The seven CPU layout tests have passed locally. Native C++ compilation and the
CUDA/HIP tests have not been run in the current host environment, which lacks
PyTorch/GPU test access. CPU coverage alone does not establish GPU correctness,
zero-length AITER support or checkpoint-level score equivalence.

## Diagnosing score differences

Use identical requests, checkpoint weights, tokenizer and quantization settings
for Graph-enabled and Graph-disabled service runs. Verify the intended paths
actually execute; directory names alone do not establish runtime configuration.
Check both repeated-request stability and consistency across candidate batches.

For deeper diagnosis, first compare replay with ordinary execution of the same
padded backend, then compare that backend with the original unpadded eager path.
This separates capture/replay effects from attention scheduling, maximum-length
profiles and GEMM shape changes. The current Transformer path does not execute
extra diagnostic forwards automatically or emit `UQI_GRAPH_CHECK` metrics.

Run numerical checks separately from latency/timeline measurements. Reusing
AITER and passing operator tests do not guarantee bitwise equality or classifier
score equivalence. Previously observed Graph/eager score differences remain
unresolved; the metadata and test fixes above do not establish their root cause.

## Source layout

| File | Responsibility |
| --- | --- |
| `model_desc/bert.py` | Shared `BertEncoderGraphImpl` and CUDA `BertUqiEncoderGraphImpl` |
| `modules/factory/attention/rocm_impl/bert_uqi_two_pass.py` | Fixed-buffer AITER Graph adapter |
| `modules/factory/attention/bert_uqi_schedule.py` | Host-only segment validation and ROCm layout generation |
| `cpp/embedding_engine/EmbeddingInputMetadata.h` | Model-facing metadata conversion shared by executor and regression tests |
| `cpp/cuda_graph/cuda_graph_runner.{h,cc}` | Graph eligibility, request metadata handoff and replay |
