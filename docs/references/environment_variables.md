# Environment variables

## ROCm Qwen3.5 GDN

`DISABLE_AITER_FLYDSL_GDN_DECODE=1` disables automatic AITER FlyDSL GDN
decode dispatch and falls back to the Triton implementation. This is an
emergency process-start rollback setting: restart the serving process after
changing it because RTP-LLM caches the value on first use.

Prefill FlyDSL dispatch remains independently controlled by `USE_FLYDSL=1`.
It is valid to use the default prefill backend together with automatic FlyDSL
decode dispatch.
