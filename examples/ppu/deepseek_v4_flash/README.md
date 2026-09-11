# DeepSeek V4 Flash on M890P

These launch scripts describe the measured PPU module configurations. Install
the PPU distribution and its matching SDK before using them; set `CHECKPOINT_PATH`
to the Flash checkpoint directory and optionally set `START_PORT` and
`CUDA_VISIBLE_DEVICES`.

```bash
CHECKPOINT_PATH=/models/deepseek-v4-flash bash examples/ppu/deepseek_v4_flash/prefill.sh
CHECKPOINT_PATH=/models/deepseek-v4-flash bash examples/ppu/deepseek_v4_flash/decode.sh
```

Run one service at a time on the assigned devices. Prefill uses TP4/EP1/DP1,
FP4 Indexer and FP8 KV, with Graph and prefix reuse disabled. It exposes the
prefill phase only: requests must use one generated token. Decode uses
TP1/DP8/EP8, FP4 Indexer, block256 FP8 KV, engine-owned DeepEP low-latency and
Graph batch sizes 1–128. It is a Decode worker configuration; it does not
provide a Prefill worker or qualify a complete PD service.

The scripts set the existing execution options explicitly. Module selection
validates them before constructing kernels; no configuration framework or
automatic option rewriting is involved. These options are frozen with each
model instance and included in the cross-rank protocol digest. Other option
combinations and the retired FP8 Indexer module IDs are outside this package's
published PPU module scope. Shared public/CUDA primitives retain their APIs.

## Validation boundary

The prior Prefill configuration was accepted for its measured runtime and
performance scope; it was not a strict all-logit accuracy pass. The prior Decode
fixed-history run matched 576/576 last-position top-1 tokens and produced finite
logits. Some logit/probability distances exceeded the observed same-batch SGLang
repeat range. This is not a bitwise, free-generation, PD or MTP qualification.
Those results belong to the previous implementation and must not be reused as
proof of a later source revision without its own targeted regression.

## Release dependencies still to resolve

The tested environment supplied TileLang `0.1.9+v0.1.0.ppu2.1.0` and TVM FFI
`0.1.11` from a development directory. That TileLang artifact declares
`apache-tvm-ffi <0.1.10`, while the current PPU lock declares TVM FFI `0.1.12`.
A supported, resolvable SDK release must settle this discrepancy and supply
TileLang through the release dependency set. Do not bypass dependency resolution
with task-directory `PYTHONPATH` overlays for release qualification.

The locked flash-attn, flash-attn-3 and Triton artifacts also need a reachable
distribution source. Keep their version and hash checks until matching artifacts
are available. A clean wheel install/startup and affected CUDA HC/TP regression
remain release conditions; these examples do not establish production readiness.

## Focused PPU regression entry

`//rtp_llm/platforms:ppu_decode_regression_tests` groups the generated-input HC,
FP8/QKV, state-slot, metadata Graph, MoE output/scheduling, stream-error cleanup,
and TP shared-weight partition tests. Run it with the PPU configuration and the
repository GPU-lock wrapper on one M890P with the matching SDK. Its constituent
targets are PPU-only; checkpoint and multirank tests retain their explicit manual
entries. Successful generated-input tests do not qualify a clean SDK install or
replace the fixed-history whole-model comparison.
