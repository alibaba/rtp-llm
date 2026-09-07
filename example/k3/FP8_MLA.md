# K3 dense FP8 MLA

This opt-in path uses ordinary E4M3 cache and FP8 attention operands. It is independent of attention weight FP8 and native MoE quantization. Full-model and PD validation is required before production use; operator tests alone do not establish model quality or performance.

## Configuration

- `KIMI_K3_MLA_FP8=1`: target MLA cache and attention compute. Default `0`; Eagle3 draft retains its existing policy.
- `KIMI_K3_MLA_FP8_Q_SCALE=1`, `KIMI_K3_MLA_FP8_KV_SCALE=1`: fixed dequantization factors (`real = fp8 * scale`). These are not dynamically calibrated group scales.
- `KIMI_K3_ATTENTION_QUANTIZATION=fp8_per_block`: separate existing online weight quantization switch.
- Use the existing dense FlashMLA Prefill adapter and TokenSpeed Decode adapter. With MLA FP8 enabled, the Prefill adapter invokes TokenSpeed FP8 Prefill; unsupported backends fail rather than selecting BF16 attention.
- All model loads require local checkpoint data and `LOAD_METHOD=fastsafetensors`.

## Precision and cache contract

| Stage | Operands and computation | Output |
|---|---|---|
| Prefill attention | Expanded Q/K/V quantized to E4M3; FP8 QK and PV; high precision softmax/accumulation | BF16 attention output and FP32 LSE |
| Decode/Verify attention | BF16 absorbed Q from kc BMM, then fixed-scale E4M3 Q; ordinary E4M3 latent cache | BF16 compressed output, then BF16 vc BMM |
| Cache write | Quantize latent512 and suffix64 with fixed KV scale | 576 bytes/token, scales in configuration metadata |
| Prefix expansion | Dequantize historical cache with KV scale, project kv_b, quantize expanded K/V for FP8 attention | FP8 attention operands |
| KDA recurrent state | Existing FP32/BF16 representation | Unchanged |

The new format does not reinterpret the old 656-byte mixed MLA format. PD allocation exchanges format version and fixed scales and rejects incompatible peers. Page size 128 is supported by the pinned dependency in operator tests; physical shared-pool strides remain governed by the cache allocator.

Decode query and output buffers are reserved before CUDA Graph capture. Workspace and kernel geometry use the existing TokenSpeed capability checks. FP8 Prefill outputs must be compared with a reference that accounts for the kernel’s FP8 probability conversion before PV, as well as a full FP32 attention reference when reporting numerical error.

## Verification targets

Build on 115 inside `lhc_GPU`, as `luohaocheng.lhc`, using `--config=cuda13 --config=sm10x`:

- `//rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test:mla_fp8_test`: actual GPU conversion, cache writes and reads, causal/noncausal Prefill, Decode/Verify and graph replay.
- `//rtp_llm/test:kimi_k3_mla_workspace_config_test`: default-off behavior, independent weight/draft policies and scale validation.
- `//rtp_llm/cpp/cache/test:mla_fp8_cache_spec_test`: physical byte counts for ordinary FP8, mixed FP8 and BF16.

GPU tests require SM100/SM103 and do not pass through skips. Use the host-selection and warmup protocol before performance measurements. The PD benchmark repeats identical input tokens with `reuse_cache=False` and verifies zero reuse in every response.

Reference source snapshots audited for this implementation: vLLM K3 `38e7f533d8b2b964393e4b07b55b66a57f67d5f6`, SGLang K3 `578edb240a6d6f6f2fa4c31497276955d7f73432`. Runtime geometry support is established by the locally built TokenSpeed dependency, not by another framework’s page-size restriction.

`KIMI_K3_MLA_FP8_DIAGNOSTICS=1` emits operand shape, fixed scale, finite absolute maximum, nonfinite counts and clipping fractions for compressed cache input and attention Q/K/V. It synchronizes to the CPU and is strictly for separate diagnostic runs. CUDA Graph capture skips observations and replay does not run Python; use eager target-only requests to inspect real Decode operands. Do not report these diagnostic request latencies as performance samples.

## Independent switches

Both switches default to off. Set the same MLA cache policy on both PD roles.

| Mode | `KIMI_K3_ATTENTION_QUANTIZATION` | `KIMI_K3_MLA_FP8` |
|---|---|---|
| BF16 attention baseline | `none` | `0` |
| Weight FP8 only | `fp8_per_block` | `0` |
| MLA FP8 only | `none` | `1` |
| Weight FP8 and MLA FP8 | `fp8_per_block` | `1` |

Native MoE quantization and the standalone Eagle3 draft precision are unchanged.
The rebased implementation uses the upstream expanded-KV byte budget and forward
planner. FP8 historical chunks restore the cache into bounded BF16 latent/RoPE
buffers before projection and FP8 attention; BF16 retains the upstream fused
gather and FlashMLA path. Historical performance results from the previous
branch do not validate this rebased implementation.
