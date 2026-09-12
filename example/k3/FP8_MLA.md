# K3 dense FP8 MLA

This opt-in path uses ordinary E4M3 cache and FP8 attention operands. It is independent of attention weight FP8 and native MoE quantization. Full-model and PD validation is required before production use; operator tests alone do not establish model quality or performance.

## Configuration

- `KIMI_K3_MLA_FP8=1`: target MLA cache and attention compute. Default `0`; Eagle3 draft retains its existing policy. K3 MTP attention and KV cache stay native BF16 even when this switch or global FP8/INT8 cache flags are enabled. Native MXFP4 experts and MegaMoE internal FP8 activation compute are unchanged.
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

Both model switches default to off. The two-host smoke overrides these defaults
to enable both. Set the same MLA cache policy on both PD roles.

| Mode | `KIMI_K3_ATTENTION_QUANTIZATION` | `KIMI_K3_MLA_FP8` |
|---|---|---|
| BF16 attention baseline | `none` | `0` |
| Weight FP8 only | `fp8_per_block` | `0` |
| MLA FP8 only | `none` | `1` |
| Weight FP8 and MLA FP8 | `fp8_per_block` | `1` |

Native MoE quantization and the standalone Eagle3 draft precision are unchanged.

## Two-host PD smoke

The role script defaults to Weight FP8, MLA FP8, unit scales, FP8 collective GEMM
and a 4 GiB historical KV expansion budget per rank. The controller forwards
explicit overrides to both roles. After
configuring the existing host, endpoint, local checkpoint and deployed-launcher
settings, launch the full suite with:

```bash
export KIMI_K3_ATTENTION_QUANTIZATION=fp8_per_block
export KIMI_K3_MLA_FP8=1
export KIMI_K3_MLA_FP8_Q_SCALE=1
export KIMI_K3_MLA_FP8_KV_SCALE=1
export KIMI_K3_FP8_COLLECTIVE_GEMM=1
export KIMI_K3_MLA_FP8_DIAGNOSTICS=0
export LOAD_METHOD=fastsafetensors
export RTP_LLM_SKIP_BUILD=1
export SMOKE_SUITE=all
# Default smoke prefix-expansion budget: 4 GiB per rank.
export KIMI_K3_MLA_PREFILL_EXPANDED_KV_BUDGET_BYTES=4294967296
python3 example/k3/kimi_k3_full_model_two_host_pd_smoke_driver.py
```

Use the independent-switch table above for the other three precision modes.
Explicitly use `none` and `0` for a BF16 comparison so an earlier shell export
does not carry over. Add `--dry-run` to inspect both remote launch commands
without starting services. Each role records and checks the supplied settings
against its service process in `service.env`; these environment checks alone do
not prove FP8 kernel execution.

The smoke prefix budget defaults to `4294967296`. The full `all` suite requires
a positive budget so its long-prefix case can exercise multiple historical
blocks. Outside this smoke the model default remains `0` (disabled). The budget splits historical KV
expansion into page-aligned blocks, runs attention on each block, and merges
output/LSE. It does not cap the current chunk's expanded KV or FP8 temporary
buffers; use `SMOKE_CHUNK_TOKENS` to control the current input chunk. A positive
budget only exercises the split route when the request exceeds its capacity
and has a historical prefix.

The `all` suite now includes `long_prefix_seed` and `long_prefix_hit`. It uses the
service tokenizer to construct a roughly 600k-token archive with records near
20k, 300k and 580k, stores it with an acknowledgment-only reply, then appends the
actual assistant reply and a new retrieval question. It requires correct record
values and `37² = 1369`, PD separation, a long common token prefix, page-aligned
cache reuse larger than one expansion buffer, and nonzero uncached input.

At TP8 with the default 4 GiB budget, 4096-token cache pages and 128-token MLA
kernel pages, the expansion capacity is 559232 tokens. A reuse length of 598016
therefore plans historical blocks of 559232 and 38784 tokens. The test reads
checkpoint dimensions and the configured TP size, budget and page sizes; it
fails if the selected configuration cannot exercise multiple historical blocks.
It also works with BF16 precision switches, and does not change model execution.
The four-layer `flow` preflight does not run this full-model case.

Both requests count toward `accuracy.json`; a failed answer or missing prefix
coverage fails the suite. Full request/response payloads, token IDs, hashes and
`RESULT.json` are retained in `prefill/long-prefix/`, including on failure.
Ordinary smoke does not arm a profiler or perform timed warmups. Its
`planned_prefix_blocks` describes coverage computed from cache metadata; actual
FP8 kernel launches and output/LSE merges require the separate timeline audit.

The rebased implementation uses the upstream expanded-KV byte budget and forward
planner. FP8 historical chunks restore the cache into bounded BF16 latent/RoPE
buffers before projection and FP8 attention; BF16 retains the upstream fused
gather and FlashMLA path. Historical performance results from the previous
branch do not validate this rebased implementation.
