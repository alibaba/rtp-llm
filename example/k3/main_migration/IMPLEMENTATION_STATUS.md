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
python -m pytest -q tests/kimi_k3
```

Result: 55 passed: 36 model math/checkpoint checks, five launch-profile checks and 14 smoke-harness
contract checks. These cover AttnRes against a scalar float64 reference, unused
residual-bank capacity, output normalization, MTP source-layer and safetensors
shape/dtype/payload validation, and TP1/2/4/8 fused KDA projection equivalence
with replicated F_a. Harness checks reject weakened acceptance profiles,
require strict long-prefix JSON, preserve failed responses and verify that a
formal HTTP failure is sent only once. These do not validate CUDA kernels,
full weight loading, full-model outputs, distributed execution, state commit,
Graph or RDMA.

New C++ checks cover the opt-in single-module policy and TP control flags;
they have not run yet. Python syntax and `git diff --check` passed during
implementation and must be rerun after further edits. The generated
`libth_transformer_config.pyi` already contains an invalid `None:` enum annotation
in base main; it is excluded from the Python AST success count (27 files).

The native `//:th_transformer` target at commit
`2de697da70e32a0bcf9841e2f4b26198c77bce02` compiled successfully on
L20D-dev-145 (exit 0, 512 actions, 629 seconds). It ran inside `lhc_GPU`
as `luohaocheng.lhc`, with `--config=cuda13 --config=sm10x --jobs=8`.
Source and Bazel output root are on local `/ssd/5` ext4. Both 144 and 145
are building the service target at `8e78a72779`; success is not yet established.
WebTerminal access has recovered. GPU availability must be reprobed before
launch; no GPU model test has run and no external service was stopped.

On 145, the full local `/ssd/5/kimi-k3` checkpoint passed the loader guard:
96 shards, 497220 tensors. The original MTP checkpoint passed the model-specific
5400-required-tensor validation but its index total_size included header bytes.
A task-owned copy at
`/ssd/5/luohaocheng.lhc/k3-main-mtp-checkpoint-20260921` corrects only this index
metadata, hard-links unchanged shards, and records provenance. It passed the
generic loader guard. Original checkpoints remain unchanged. Repeat independent
validation on each selected host and verify actual FastSafetensors startup logs.

`launch_bf16.py` fixes TP8/EP8/SP, native MTP3, draft/target BF16 and RDMA;
Decode enables Graph. It records the command, requires new local run storage,
validates both checkpoints, and checks compute PIDs and service ports. It is
not yet runtime-validated and does not replace fleet selection or runtime evidence.

A newly confirmed integration gap is whole-model chunk prefill: main's
`max_batch_tokens_size` is a scheduler constraint, not a chunk executor, and the
current K3 target has no chunk loop. The 65536 setting in the launch profile
alone must not be described as chunk support. Port the target/draft round
coordination and verify KDA/conv/KV continuity before counting any >64K or
budget+1/+7 case as passed.

## Text smoke entry point (not yet run against services)

`text_smoke.py` ports the dev cases to an explicitly named `main-text` suite.
It preserves answer, actual token-prefix, reuse, chunk, PD and MTP assertions.
It reports DCP/PageRR-owner/DP-multi-owner/multimodal checks as not applicable.
The ordinary reuse unit is the configured cache block, never `TP * block`.
A full suite requires a 93-layer checkpoint, real MTP acceptance, at least a
64K chunk budget and at least a 110K long prefix. It adds exact distinct input
lengths to the 8-wide/16-request rolling case and batch shapes 1/2/3/7/8/9 with
repeated transitions. These are request shapes, not proof of actual Graph replay.

Example after both services are ready (replace paths, ports and block size with
the verified runtime values; use a new output directory each time):

```sh
python example/k3/main_migration/text_smoke.py \
  --suite main-text --base-url http://PREFILL_IP:HTTP_PORT \
  --decode-health-url http://DECODE_IP:HTTP_PORT/health \
  --decode-role-addr DECODE_IP:HTTP_PORT:GRPC_PORT --decode-dp-size 1 \
  --long-prefix-checkpoint /local/data/kimi-k3 \
  --block-size 4096 --chunk-tokens 65536 --require-mtp \
  --namespace unique-bf16-run --output /local/data/new-run/result.json
```

The driver, reproducible service configurations, runtime evidence collector and
numerical/state comparison runner remain incomplete. Passing this text runner
alone must not unlock FP8 or be described as complete BF16 acceptance.

## Remaining gates

1. Compile and fix ABI/runtime integration. Verify weight manifest/shapes and
   bounded KDA/SiTU/MLA against the pinned reference.
2. Validate actual state commit for accept counts 0/1/2/3; cached decode versus
   full-prefix recomputation; batch 1/2/3/7/8/9 and consecutive Graph replay.
3. Complete the two-host driver and reproducible TP8/EP8/SP service configuration;
   run the ported text cases and attach separate runtime evidence. Do not claim
   the dev `all` suite passed.
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
