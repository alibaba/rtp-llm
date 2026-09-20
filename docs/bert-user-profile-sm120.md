# BERT user profiles on SM120

Enable `USE_VISION_BERT_UQI_BLOCK_MASK=1` before loading the model. The checkpoint
must contain `w_out_uqi.weight` and `w_out_uqi.bias` as well as the existing
`w_out` head. Configure `output_num` to use the Mainse `MULTI_OUTPUT` path.
The output contains `2 * output_num` probabilities: QI first, UQI second,
with a separate softmax for each head. A four-class checkpoint returns eight values.

The assembled input is Query + Item + `[CLS_UQI] User [SEP]` + Vision.
`VISION_BERT_CLS_UQI_TOKEN_ID` defaults to `2`; SEP is `102`.
The user span, including its markers, can attend to every token. Query, Item and
Vision cannot attend to the user span. Vision positions are excluded when searching
for user markers. The mask uses FlashInfer ragged prefill `custom_mask` and retains
the existing BERT GEMM, residual/LayerNorm, GELU and FP8 fusion paths.

With the flag enabled, requests without a user span also use FlashInfer native
ragged prefill, with `custom_mask=None` and non-causal attention, rather than the
TRT-LLM FMHA path. The UQI head still runs and pools the first token if its marker
is absent. With the flag disabled, attention retains the original factory selection.
With the flag disabled, the original single head runs; supplying a user profile to
the renderer/ARPC handler is rejected to avoid unmasked personalized inputs.
VisionBert continues to disable CUDA graphs, and regular BERT also disables them
when the user-profile flag is enabled.

## Input interfaces

JSON input adds `user_profile_bert_ids` (including CLS_UQI and SEP) and
`user_profile_bert_mask_len: [length]`. The shared profile is appended after each
candidate item and before its vision tokens. It has token type 0; block-mask
segments are independent of token-type embeddings.

Internal ARPC input adds raw text `qui_user_bert_input`. The engine tokenizes it
and replaces its leading CLS with CLS_UQI. The new field number is **12**.
The mainline profiling fields stay unchanged: `gen_timeline=9`, `profile_step=10`,
`profile_trace_name=11`. Existing profiling clients remain wire-compatible.
User-profile clients (including clients based on the feature branch's field-9
profile protocol) must regenerate their protobuf definitions to send
`qui_user_bert_input` at field 12. Do not send profile requests to an old server,
which would ignore the new field. No submodule-pointer update is required for the
source changes, but both repositories must be built together.

## FP8 GEMM selection

Set `RTP_LLM_SM120_FP8_BACKEND=deepgemm` or `cutlass` before loading weights.
`auto` still resolves to DeepGEMM. BF16 and explicitly unquantized layers are
unaffected by this FP8 backend switch.

CUTLASS dense weights use only 128 × 128 blocks in physical `(N, K)` layout,
with physical scale shape `(N / 128, K / 128)`. K and N must be multiples of
128. Per-output-channel weight scales are rejected. DeepGEMM retains its
UE8M0 weight/activation contracts.

CUTLASS applies bias in the GEMM epilogue, including attention output and MLP
down projections. The GELU up projection also applies exact (erf) GELU in that
epilogue and writes BF16 output. The next linear separately quantizes this BF16
tensor to FP8. CUTLASS does not fuse GELU or LayerNorm with activation
quantization, and does not defer output bias to residual LayerNorm.

Changing weight block size changes FP8 numerics. Restart and reload weights when
changing these settings; validate both heads against the same checkpoint/reference.
Mask isolation alone does not establish full-model accuracy or latency.
