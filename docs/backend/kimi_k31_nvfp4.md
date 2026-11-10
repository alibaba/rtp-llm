# K3.1 NVFP4 projection ablations

`KDA_NVFP4` and `DSA_NVFP4` independently select online NVFP4 W4A4 for the
target model's external attention projections. Both accept `0` or `1` and
default to `0` in an ordinary server launch. They are resolved when the model
configuration is constructed; restart the service to change a selection.

Use the same original K3.1 checkpoint for all four ablations:

| KDA_NVFP4 | DSA_NVFP4 | KDA projections | DSA projections |
| --- | --- | --- | --- |
| 0 | 0 | Original FP8 policy | Original FP8 policy |
| 1 | 0 | NVFP4 W4A4 | Original FP8 policy |
| 0 | 1 | Original FP8 policy | NVFP4 W4A4 |
| 1 | 1 | NVFP4 W4A4 | NVFP4 W4A4 |

The FP8 control uses `QUANTIZATION=FP8_PER_BLOCK`. With quantization unset,
unselected projections retain the original BF16 policy instead.

```bash
export KDA_NVFP4=1
export DSA_NVFP4=0
export QUANTIZATION=FP8_PER_BLOCK
export RTP_LLM_FP4_GEMM_BACKEND=cutlass
export LOAD_METHOD=fastsafetensors
# Start the normal kimi_k31 service with its original target and MTP checkpoints.
```

KDA selection covers the fused `[Q,K,V,G,F_a,beta]` input projection, `F_b`,
and output projection. DSA selection covers fused `[Q-A,KV-A,G]`, `Q-B`, and
the output projection. The selection applies to every corresponding target layer.
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
is retained and checked against F_b. Mixed FP8/NVFP4 uses the matching BF16
activation and TP communication path instead of feeding FP8 operands to FP4.

Supported execution is CUDA13 on SM100/SM103 with the repository-pinned
FlashInfer CUTLASS backend. Missing operands, unsupported layouts, another
backend and load-time LoRA merging fail explicitly.
Natural-order Prefill retains NVFP4: Q/gate are packed from the existing FP4
codes, while KV uses a separate row slice. Local scales are reinterleaved for
each slice, including boundaries inside a 128-row tile; weights are neither
dequantized nor requantized. Startup/execution evidence requires both Q/gate
and KV fragments on every natural-order Prefill rank.
The removed `k31_nvfp4_modules` JSON override produces a migration error.
Prepacked/calibrated NVFP4 exports are not inputs to this online mode.

## Two-host smoke

The manual and Bazel K3.1 two-host smoke tests default to FP8 with both NVFP4
switches off. The manual runner resolves each switch from the CLI, then the JSON
config (`kda_nvfp4`, `dsa_nvfp4`), then the controller environment
(`KDA_NVFP4`, `DSA_NVFP4`). It saves the resolved values and sends them to both
peers.

```bash
# Default: full regular + long smoke with FP8 and native MTP.
example/k3/kimi_k31_full_model_two_host_pd_smoke.sh \
  --config /path/to/pair.json --output /path/to/new-result

# Opt in to NVFP4 for both KDA and DSA projections.
example/k3/kimi_k31_full_model_two_host_pd_smoke.sh \
  --config /path/to/pair.json --output /path/to/new-fp4-result \
  --kda-nvfp4 --dsa-nvfp4

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
