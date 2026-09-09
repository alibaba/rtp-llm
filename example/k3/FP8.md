# Kimi K3 attention block FP8

Enable with `KIMI_K3_ATTENTION_QUANTIZATION=fp8_per_block` alongside the existing
K3 launcher and explicit `LOAD_METHOD=fastsafetensors`. The default is `none`.
The setting is target-attention-only: global quantization remains disabled,
native MoE and Eagle3 draft retain their existing precision policies. This is
128 x 128 block E4M3, not MXFP8. KV caches, attention cores and KDA states are
unchanged.

## Weight and execution contract

All FP8 projections dynamically quantize BF16 input by token/group128 and emit
BF16 output. On B300, weight scales are rounded to the final power-of-two value
before the only FP8 cast, then packed for DeepGEMM; packing does not requantize.

| Projection | TP8 logical [out,in] | Storage / execution |
|---|---|---|
| KDA Q/K/V/G + F_A + beta | [6368,7168] | FP8 / fused FP8 GEMM |
| KDA F_B | [1536,128] | FP8 / FP8 GEMM |
| KDA O | [7168,1536] | FP8 / FP8 GEMM, BF16 partial reduction |
| MLA Q_A + KV_A | [2112,7168] | replicated FP8 / FP8 GEMM |
| MLA gate | [1536,7168] | FP8 / independent FP8 GEMM, shared input quantization |
| MLA Q_B | [2304,1536] | FP8 / FP8 GEMM |
| MLA KV_B | [3072,512] | FP8 / Prefill FP8 GEMM |
| MLA O | [7168,1536] | FP8 / FP8 GEMM, BF16 partial reduction |
| MLA KC / VC | head-specific views of KV_B | BF16 / Decode BMM |

Q_B uses 12 local heads times (128 non-RoPE + 64 RoPE) dimensions at TP8.
KC/VC are derived from dequantized **final FP8 KV_B**, so Prefill and Decode
use the same quantized-weight values.

KDA Q/K/V/G are sharded on block-aligned head boundaries. The complete 128-row
F_A and 96-row beta remain replicated; execution slices beta to local heads.
MLA gate is separated from Q_A/KV_A so the 64-row partial KV_A block is the last
block. Scales for partial blocks use valid rows only. Temporary zero padding
inside the quantizer is cropped before persistent weight storage. No physical
weight padding is used by this implementation.

The loader subclasses the existing `LoadQuantPerBlockFp8Weight`, but is only
selected by the K3 attention manifest. LoRA merging and TP sizes outside
1/2/4/8/16 fail explicitly. It does not join the global quantizer registry.
`K3_FP8_WEIGHT` log entries describe logical shapes, packed scales, TP rank,
derivation and weight padding; retain them with each run.

## Communication

The existing `KIMI_K3_GEMM_REDUCE_SCATTER_BACKEND` control remains available.
`KIMI_K3_FP8_COLLECTIVE_GEMM=0` selects explicit AG/RS for the FP8 correctness
reference; the default `1` enables both fused paths above the existing 32K
physical-token threshold. Precision never silently falls back to BF16 weights.

FP8 AllGather/GEMM uses PyTorch's pipelined BF16 shard consumer and shares input
quantization between compatible projections. Each consumer owns its temporary
scales. FP8 GEMM/RS writes destination-sized GEMM outputs directly into the
existing DeepGEMM symmetric BF16 source slots. It reuses the existing reduction
and two-barrier protocol, including stream handoff. The reducer accumulates
BF16 partials in FP32 before BF16 output. This requires the custom GemmRSBuffer
ABI with 128 control bytes and peer mappings, not an arbitrary PyPI DeepGEMM.
It uses multiple GEMM launches; speedup over the original fused BF16 kernel is
not assumed. No new DeepGEMM binary is introduced.

Historical MLA prefix projection uses FP8 KV_B GEMM plus BF16 placement into
`[K(128), RoPE(64), V(128)]` per head. The existing RoPE region is preserved.
This adds an unpadded temporary BF16 GEMM output and two copies; it does not
pad weights or compute KV_B in BF16.

Warm each projection and collective on the actual capture stream before CUDA
Graph capture. Cross-stream waits on uncaptured work cannot be introduced
inside capture. The graph owns captured temporary allocations.

## Validation and reproducibility

Initial baselines: parent RTP-LLM `8cc7ac61ec24841a024c5b4bd538a8163a06e94d`,
child `2d12375ab3c696dac86770a7356073513dfeae0a`.
Build on 115 in lhc_GPU as the normal user with
`--config=cuda13 --config=sm10x`.

* `//rtp_llm/models_py/modules/hybrid/test:kimi_k3_fp8_weight_test` covers tails,
  TP1/2/4/8/16 KDA layout, final-quantized KV_B derivation and policy isolation.
* `fp8_shape_probe.py` checks logical dimensions against independently
  dequantized operands on the installed DeepGEMM.
* `//example/k3:fp8_runtime_probe` exercises the actual loader, packing, Linear factory,
  dynamic activation quantization and CUDA Graph.
* `fp8_collective_probe.py` checks eager and graph AG/RS against explicit
  communication. Build the `//example/k3:fp8_collective_probe` binary, then run with
  `torchrun --nproc_per_node=8 --no-python bazel-bin/example/k3/fp8_collective_probe`. `--source-root` can point at the
  deployed server runfiles/rtp_llm directory.
* The existing two-host smoke driver forwards the new FP8 flag and loader
  setting. Use flow only for four-layer connectivity and all for 93-layer
  semantics/cache validation.

Development artifacts are recorded at
`/data1/luohaocheng.lhc/k3_fp8_runs/20260905` on 115. These logs distinguish
kernel correctness from model quality and performance. Standalone kernel and
Graph passes do not constitute full-model acceptance. Full PD, 93-layer,
110k-context and measured performance results must be recorded before claiming
complete delivery.
