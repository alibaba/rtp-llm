# Optional output during generation

The normal generation engine can invoke a deployment-defined CustomModule handler
once per prefill batch that contains selected, uncached token rows. The handler may
perform tensor operations or use learned weights; it need not be an MLP or scoring head.
A deployment registers it through the existing module factory. Optional weights use
CustomModule's checkpoint loader. Without a generation module, the factory returns
`None`; failures constructing a configured module propagate instead of disabling it.
Embedding and reranking keep their existing EmbeddingCppEngine / RtpEmbeddingOp path.

The handler implements:

- `select_token_position(input_ids, text_tokens_mask=None)`: a CPU token tensor and
  optional text-token mask to a zero-based position in that tensor, or -1 to skip.
  Called once per request in the model process, after RPC deserialization and
  multimodal expansion, before system-prefix insertion.
  The selection is computed locally and is not an RPC request field. Raise `ValueError`
  to reject prompt input; other selector exceptions are reported as execution failures.
- `extend_forward_args()`: `["selected_hidden_states"]`.
- `extend_forward(selected_hidden_states=...)`: receives `[rows, hidden_size]` and
  returns a nonempty `[rows]` or `[rows, width]` tensor on the same GPU, preserving row
  order and count. The selected hidden states come from the model's existing output
  after its final normalization, with the model activation dtype; the handler must
  be compatible with these features.

`rows` counts selected context sequences in the engine's expanded batch. A request
with multiple return sequences contributes one row per sequence, using the same
selected token position. The handler processes all selected rows in one batched call.

The engine selects only rows computed in the current prefill. A position covered by
prefix-cache reuse produces no custom output; caching and generation continue normally.
It compacts indexes on CPU, batches the handler invocation and reuses the existing output
transfer. Float32, float16, bfloat16 and int32 outputs retain their dtype through RPC;
HTTP responses serialize them as numeric arrays. The engine applies no additional
sigmoid or softmax to the handler's result.
Results appear as native `custom_output` or OpenAI `extra_outputs.custom_output`.
The prefill result stays on the request for subsequent responses, including the final
response; decode does not recompute it. Multi-sequence native responses preserve each
sequence's result. Explicit `num_return_sequences > 0`, including 1, adds an outer
sequence dimension; `/batch_infer` retains its first-completion-per-prompt response format.
OpenAI's request-level `extra_outputs` follows the existing last-choice convention.
The handler runs locally on TP rank 0 over replicated hidden states; it must not perform
TP collectives. The existing backbone handles TP reductions (for example, Qwen's
attention output projection and DenseMLP); the handler needs no additional all-gather.
Other ranks follow the normal model forward without custom-output indexes.

Python generation models with an LM head are supported in PDFUSION (prefill and
decode together), without speculative decoding or prefill CP. P/D separation is
not supported. The integration uses the shared model wrapper, not a Qwen-specific hook.
Single-device Qwen3 generation-prefill graphs reuse their normal model output;
row selection and the handler run after graph replay. Decode does not invoke the handler.
TP uses the existing eager prefill and optional decode graph; the current main branch
does not support generation-prefill graphs with TP greater than one.
Business token rules, postprocessing and deployment configuration belong to the module.
