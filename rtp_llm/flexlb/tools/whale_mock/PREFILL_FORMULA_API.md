# Mock prefill formula updates

The mock control HTTP port exposes `GET /prefill_formula` and `POST /prefill_formula`.
This modifies mock execution timing only. It does not modify master routing estimates.

Read first, save the current expression for rollback, then POST:

```json
{"expression":"200 + sum(computeTokens / 1024.)"}
```

Omitting `engine` applies to all current P engines. Use `"engine":"prefill-0"` to target one.
The expression above is an API example, not a calibrated production formula.

Expressions use the existing mathematical formula parser. Validation is performed before
any engine update; malformed expressions and invalid targets return 400. Blank/oversized
expressions and formulas returning negative/nonfinite results on validation samples are rejected.
Validation samples do not prove a formula is valid over every workload; callers must verify their domain.
Each engine replaces its formula reference atomically; updating multiple engines is not a global batch barrier.
Already scheduled batch delays are unchanged. Subsequent evaluations use the new formula
with prefill scale fixed at 1.0. This override takes precedence over fixed-ms settings.
Readback lists each engine's expression and scale. Runtime changes do not survive restart;
persist the expression in the deployment configuration separately. Restoring a saved expression
uses the same POST operation, still with scale 1.0. Cache is not cleared by this API.

`rtp_llm_context_batch_size` reports one sample per started prefill execution batch.
Idle polling does not add zero samples; use running-stream metrics for idle occupancy.
`rtp_llm_device_reuse_length` aliases the same device-only token value as
`rtp_llm_stream_cache_device_reuse_length`; memory reuse is excluded.
`rtp_llm_input_token_length` retains its existing reporting point; no duplicate event is added.
