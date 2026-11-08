# K3.1 NVFP4 projection ablations

`KDA_NVFP4` and `DSA_NVFP4` independently select online NVFP4 W4A4 for the
target model's external attention projections. Both accept `0` or `1` and
default to `0` in an ordinary server launch. They are resolved when the model
configuration is constructed. Restart the service to change a selection.

Use the same original K3.1 checkpoint for all four ablations:

| KDA_NVFP4 | DSA_NVFP4 | KDA input and F_b | DSA input and Q-B |
| --- | --- | --- | --- |
| 0 | 0 | Original FP8 policy | Original FP8 policy |
| 1 | 0 | NVFP4 W4A4 | Original FP8 policy |
| 0 | 1 | Original FP8 policy | NVFP4 W4A4 |
| 1 | 1 | NVFP4 W4A4 | NVFP4 W4A4 |

The FP8 control uses `QUANTIZATION=FP8_PER_BLOCK`. With quantization unset,
unselected projections use the original BF16 policy.

```bash
export KDA_NVFP4=1
export DSA_NVFP4=0
export QUANTIZATION=FP8_PER_BLOCK
export RTP_LLM_FP4_GEMM_BACKEND=cutlass
export LOAD_METHOD=fastsafetensors
# Start the normal kimi_k31 service with its original target and MTP checkpoints.
```

KDA selection covers the fused `[Q,K,V,G,F_a,beta]` input projection and `F_b`.
DSA selection covers fused `[Q-A,KV-A,G]` and `Q-B`. With either switch
enabled, that group's output projection uses grouped FP8 in both Prefill and
Decode. For a TP-sharded output in eager Prefill, the existing FP8
GEMM/ReduceScatter path uses DeepGEMM at 512 rows and above. Natural Query-CP
DSA uses the FP8 full-head output projection and its usual reverse row exchange.
On a dedicated Decode worker, target verify uses an FP8 GEMM that produces
BF16 partials for push ReduceScatter.
Indexer, KC/VC BMMs, KV storage, KDA state, convolution, routed/shared experts,
MoE latent projections and native MLA MTP retain their existing precision policies.
Target flags do not propagate to the draft model.

## Online quantization

The native loader retains ownership of fusion order, TP head slicing,
replicated F_a/Q/KV latents and preserved-layout query CP. After that layout
is prepared, selected original BF16 weights are quantized directly into packed
E2M1 bytes. No intermediate FP8 weight is used and no checkpoint file is written.

Each GEMM uses an FP32 weight global scale and E4M3 local scales for blocks of
16 elements. Its BF16 activation is dynamically quantized on each forward;
the activation global scale is recomputed by device operations during CUDA
Graph replay as well. FlashInfer CUTLASS `mm_fp4` produces BF16 output.
The existing static-scale FP4 Linear mode remains available to other models.

Rank-local K must align to 32. KDA's short beta tail is padded at the end of
the fused N dimension and trimmed by the existing consumer; its logical width
is retained and checked against F_b.

In eager TP Prefill, each KDA rank quantizes its own input shard. One AllGather
packet carries the packed E2M1 values, E4M3 block scales, and that shard's FP32
global scale. On SM10x, the native AttnRes kernel collects partial maxima as it
writes the normalized BF16 activation. The quantizer combines those maxima
and writes directly into the packet.

`KDA_NVFP4_AG_MODE=serial` is the default. `direct` uses Direct AllGather when
its staging buffer can hold the packet. `overlap` pipelines the AllGather with
the shard GEMMs. `auto` selects overlap for TP8 with at least 8192 local rows
and serial otherwise.

KDA's `F_b` reads the original row stride of the BF16 `F_a` slice. The CUDA
quantizer reduces the partial maxima and writes the packed FP4 values and E4M3
scales for the existing scalar-alpha CUTLASS GEMM.

Supported execution is CUDA13 on SM100/SM103 with the repository-pinned
FlashInfer CUTLASS backend. Missing operands, unsupported layouts, another
backend and load-time LoRA merging fail explicitly.
Natural-order Prefill keeps its input projections in NVFP4. Q/gate are packed
from the existing FP4 codes, and KV uses a separate row slice. Local scales
are reinterleaved for each slice, including boundaries inside a 128-row tile.
On the native AttnRes path, local KV-A and the Natural-to-Zigzag row exchange
share one NVFP4 activation. Indexer K consumes local FP8, while the FP32 head
weights are computed from local BF16 and exchanged after the FP4 payload.
Startup/execution evidence requires both Q/gate
and KV fragments on every natural-order Prefill rank.
The removed `k31_nvfp4_modules` JSON override produces a migration error.
Prepacked/calibrated NVFP4 exports are not inputs to this online mode.

## Two-host smoke

The manual and Bazel K3.1 two-host smoke tests enable both NVFP4 switches by
default. The manual runner resolves each switch from the CLI, then the JSON
config (`kda_nvfp4`, `dsa_nvfp4`), then the controller environment
(`KDA_NVFP4`, `DSA_NVFP4`). It saves the resolved values and sends them to both
peers.

```bash
# Full regular + long smoke with both NVFP4 switches enabled.
example/k3/kimi_k31_full_model_two_host_pd_smoke.sh \
  --config /path/to/pair.json --output /path/to/new-result

# FP8 baseline with both NVFP4 switches disabled.
example/k3/kimi_k31_full_model_two_host_pd_smoke.sh \
  --config /path/to/pair.json --output /path/to/new-fp8-result \
  --no-kda-nvfp4 --no-dsa-nvfp4

# KDA-only ablation; alternatively set the two JSON booleans or environment.
example/k3/kimi_k31_full_model_two_host_pd_smoke.sh \
  --config /path/to/pair.json --output /path/to/new-result \
  --kda-nvfp4 --no-dsa-nvfp4
```

The complete smoke retains the 93-layer target, trained native MTP, regular
and 1M long-context stages and all existing semantic/native execution audits.
Projection evidence additionally requires every rank/layer to construct the
expected precision, execute every selected NVFP4 projection, and retain BF16
draft projections. Readiness or flag values alone are insufficient.

The long-stage Prefill fixture uses 1032 physical LINEAR blocks, including
reserved block zero; its full/indexer pools remain at 140 blocks. The current
allocator budgets LINEAR states by 1024-token physical pages. The 8192-token
canonical CP storage key span does not reduce this allocation. Startup checks
require enough physical states for the 1M input, generation and MTP reserve.
The long fixture admits one request per peer and uses 8K Prefill rounds to
bound MoE staging beside the dense LINEAR pool. Prefill uses one BlockTree
transfer worker per direction, which reserves two CRC workspaces instead of
eight. The cache format and CRC checks are unchanged, and startup evidence
must show both CRC slots on every Prefill rank.
The complete continuation still contains exactly 1,048,576 tokens. Both peers'
maximum sequence lengths include 4096 tokens of headroom. Workspace receipts
must match the configured budget on every rank.

Run CPU switch/model-policy/launcher tests before GPU validation. On a selected
CUDA host with the native build closure, run
`python -m unittest rtp_llm.test.kimi_k31_nvfp4_cuda_test` for dynamic/static
Linear, padding, zero/strided/empty inputs and changed-input Graph replay.
The numerical reference decodes the actual E2M1 codes/scales into FP32.
Decode side-stream projections also replay with changed input scales through
the upstream `StreamDag` used by the multi-stream prologue.

This mode uses online dynamic quantization. Precision results from the
GDN/QSA document's offline static calibration do not establish its model
accuracy; evaluate the four configurations separately on identical inputs.
