# Optional output during generation

The normal generation engine can invoke an existing CustomModule handler once per
prefill batch. A deployment registers that module through the existing module factory;
head weights continue to use CustomModule's checkpoint loader.
The factory returns `None` when no generation head is configured; failures while
constructing a configured head propagate instead of silently disabling its output.
Embedding and reranking keep their existing EmbeddingCppEngine / RtpEmbeddingOp path.

The handler implements:

- `select_token_position(input_ids, text_tokens_mask)`: expanded CPU prompt to a
  token position, or -1 to skip. Called once before system-prefix insertion.
- `hidden_state_stage()`: `post_final_norm` (default) or `pre_final_norm`, matching training.
- `extend_forward_args()`: `["selected_hidden_states"]`.
- `extend_forward(selected_hidden_states=...)`: receives `[rows, hidden_size]` and
  returns `[rows]` or `[rows, width]` on the same device.

The engine selects only rows computed in the current prefill. A position covered by
prefix-cache reuse produces no custom output; caching and generation continue normally.
It compacts indexes on CPU, batches the handler invocation and reuses the existing output
transfer. Float32, float16, bfloat16 and int32 outputs retain their dtype through RPC;
HTTP responses serialize them as numeric arrays. No sigmoid or softmax is applied.
Results appear as native `custom_output` or OpenAI `extra_outputs.custom_output`.
The prefill result stays on the request for subsequent responses, including the final
response; decode does not recompute it. Native responses preserve each sequence's score.
OpenAI's request-level `extra_outputs` follows the existing last-choice convention.
The handler runs locally on TP rank 0 over replicated hidden states; it must not perform
TP collectives. Other ranks follow the normal model forward without scoring indexes.

Python models in PDFUSION are supported without speculative decoding or prefill CP.
Pre-final-norm requires model support (Qwen3 or the C++ layer-microbatch path).
Single-device Qwen3 generation-prefill graphs retain selected rows through their normal
graph lifetime; decode does not invoke the handler. Unsupported feature stages fail at initialization.
TP uses the existing eager prefill and optional decode graph; the current main branch
does not support generation-prefill graphs with TP greater than one.
Business token rules, head definitions and deployment configuration belong to the module.
