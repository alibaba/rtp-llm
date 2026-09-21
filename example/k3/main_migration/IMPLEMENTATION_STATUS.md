# K3 main migration — work in progress

This branch is not ready to merge or deploy. Neither full-model smoke gate has
passed. FP8 implementation has not started: it is gated on complete BF16
validation.

## Source pins

- RTP main: `6aebdf0a659ad7b3b0520834ff5112e48b41c7a2`
- K3 dev reference: `64c6aff3666402228950f1f09031e228c3734277`
- vLLM code reference: `04c1f4a407962a37d01b573f3a2aba91dd2ed3a9`
- Earlier vLLM deployment reference: `c3b48446349569512749db7f6e2164aa8a33437d`

## Implemented, awaiting integration validation

- K3 text configuration and RTP weight manifests; head-wise TP projection
  splitting, EP-native MXFP4 bytes/scales, standalone nextn source mapping.
- Decoder, bounded KDA, NoPE gated MLA, SiTU latent/shared MoE, AttnRes, native
  recurrent nextn module. Target and draft return logits inputs separately from
  prenorm recurrence features.
- Opt-in single-module MTP reuse and default-preserving framework changes.
- K3 SP physical padding, logical output restoration, request-aligned decode
  geometry, phase-specific Graph bucket alignment and persistent validity masks.
- Position IDs independent of positional embeddings, including Graph storage.
- K3 rectangular paged MLA verification adapter using main's planner.
- DeepGEMM SiTU parameter mapping (`activation_alpha` = gate scale,
  `activation_beta` = up scale); backend capability failure is explicit.

No DCP, PageRR, KTP, EAGLE3/DSpark or multimodal implementation was imported.
No checkpoint was modified. RDMA remains main's implementation; correct K3
state transfer is still unproven.

## Current evidence

On macOS, a separate Python 3.9 / PyTorch 2.8 CPU environment ran:

```sh
python -m pytest -q tests/kimi_k3/test_model_math.py
```

Result: 30 passed. This checks AttnRes against a scalar float64 reference,
unused residual-bank capacity, bank update/output normalization and MTP source
layer validation, plus TP1/2/4/8 fused KDA projection equivalence with replicated F_a. It does not validate CUDA kernels, weight loading, full-model
outputs, distributed execution, cache state, Graph or RDMA.

New C++ checks cover the opt-in single-module policy and TP control flags;
they have not run yet. Python syntax and `git diff --check` passed during
implementation and must be rerun after further edits.

The first B300 build was started on L20D-dev-145 inside `lhc_GPU` as
`luohaocheng.lhc`, with `--config=cuda13 --config=sm10x --jobs=8`.
Source and output are on local ext4 `/ssd/5`. Dependency resolution is ongoing;
there is no successful compiler/linker result yet. A source revision must be
frozen and synchronized again before the final build.

## Remaining gates

1. Compile and fix ABI/runtime integration. Verify weight manifest/shapes and
   bounded KDA/SiTU/MLA against the pinned reference.
2. Validate actual state commit for accept counts 0/1/2/3; cached decode versus
   full-prefix recomputation; batch 1/2/3/7/8/9 and consecutive Graph replay.
3. Port the dev driver/cases into an explicit TP8/EP8/SP text profile. Preserve
   formal requests, raw responses, exact token prefixes, reuse and checkpoint
   assertions. Mark excluded DCP/PageRR/DP-owner/multimodal checks N/A. Do not
   claim the dev `all` suite passed.
4. Run full 93-layer BF16 two-host smoke, including >64K chunk, exact budget
   +1/+7, 8-wide rolling 16 requests, cache boundaries, A/B reuse and 110K
   retrieval. Prove RDMA handoff, real MTP acceptance and Graph replay.
5. Only after gate 4 and supplementary correctness pass: target-only FP8
   online projections and Tokenspeed FP8 QK/PV, then the same full smoke.
6. BF16 recheck, affected legacy-model regression, cancellation/error/resource
   cleanup validation and final version/checkpoint/evidence manifests.

Native draft Attention, ordinary projections and cache must stay BF16 in both
profiles. The initial phase must not enable target FP8. No reduced-concurrency
or shorter-prefix substitute may be counted as the required full smoke.
