# Text-only Qwen3.5 MoE policy updates

`qwen35_moe_text` uses the existing Qwen3.5 MoE weight descriptors, hybrid
GDN/MHA layers and MRoPE configuration, with the unused vision tower disabled.
`qwen36_moe_text` remains an alias for deployment compatibility. Neither alias
changes automatic architecture inference: an ordinary
`Qwen3_5MoeForConditionalGeneration` checkpoint still selects the original
multimodal `qwen35_moe` registration.

This support was developed for the Qwen3.6-35B-A3B and Qwen3.5-122B-A10B and
397B-A17B text-policy deployments. It does not establish support for a different
Qwen3.8 checkpoint, FP8 training, MoE batch invariance, or arbitrary parallelism.

## Execution and update contract

- The native RL guard requires BF16, no quantization, `moe_style == 2`, pure TP,
  and DP=EP=PP=1. Existing scheduling, speculative decoding, cache and graph
  restrictions in the native guard continue to apply.
- Both text aliases receive the existing Qwen3.5 FP32 SSM cache preset.
  Explicit user choices retain the original preset's precedence rules.
- A full source transaction contains the actual HF checkpoint parameters.
  Packed `experts.gate_up_proj` and `experts.down_proj` inputs are expanded by
  the ordinary MoE descriptors, with the same TP split and device conversion
  used during initial loading. The per-expert fast-loader aliases are not
  accepted as substitutes for those packed HF parameters.
- Streaming conversion clones each source once and shares that owned clone
  among every descriptor consuming it. The pending-source budget counts that
  clone once until its last consumer has finished. It does not bound temporary
  output tensors or all model memory.
- FP32 HF sources, including GDN `A_log` and `dt_bias`, can be transported and
  accepted alongside BF16 sources. This is not proof of FP32 native forward
  computation. The normal checkpoint descriptor selects its explicit
  `data_type`, or otherwise the loader's `compute_dtype`; the current Qwen GDN
  descriptors do not explicitly select FP32. A received FP32 source can
  therefore be rounded to BF16 before a later kernel promotes it to FP32.
  Effective forward dtype requires separate validation.
- Inference must remain paused while source tensors are received and copied
  into existing target storage. Unknown, duplicate or missing sources, shape
  mismatches, missing targets and storage replacement prevent a successful
  transaction. A partially failed transaction must not publish a policy version.
- When `RTP_WEIGHT_MANIFEST_DIR` is set in RL mode, each native worker writes
  its required HF source names and TP rank/process/GPU identity. The external
  trainer owns global source coverage, source lifetime and all-rank ACK checks.

Initial loading may enable the existing `--moe_pure_tp_preshard` option. Streamed
refit uses a `TensorCollector`, so it cannot reload a stale packed expert from
the checkpoint database. The option is not a new optimization in this change.

For multi-node TP with EP=1, each physical TP host retains the complete expert
identity mapping. Physical host count does not subdivide the single EP rank.
Only nonredundant, complete identity mappings are accepted by this new branch;
existing single-node and EP>1 algorithms remain unchanged.

## Selected-token scores and decode cache

Non-BI selected-token log probabilities are computed directly as the FP32
selected raw logit minus FP32 `logsumexp(raw_logits)`, before sampler processing.
They do not round-trip through a probability that can underflow to zero. The
existing BI normalization branch retains its original arithmetic.

The fused QK norm/RoPE/KV-cache decode dispatcher uses
`decode_cu_seqlens_device`. Native decode intentionally leaves the context
`cu_seqlens_device` array zero-filled. Prefill continues to use that context
array. A missing decode mirror falls back before the fusion writes Q/K cache.

The explicit ARM PyTorch BF16 RoPE compatibility backend is an integration
option, not a replacement for the installed `rtp_kernel` distribution. It stays
outside the default core import path. Its supported layouts, rejected FP8/YaRN
and graph paths, and activation are documented with the RL integration example.

## Regression checks

The tests below invoke the actual production methods or kernels. CUDA/C++
targets require a compatible built RTP environment; CPU-only source checks do
not establish GPU or full-model acceptance.

| Area | Test source / target |
| --- | --- |
| Decode boundaries and real paged-cache append | `rtp_llm/models_py/triton_kernels/common/test/qk_norm_rope_cache_test.py` |
| Raw FP32 scores, underflow and unprocessed logits | `//rtp_llm/cpp/normal_engine/test:normal_output_dispatcher_test` |
| Text alias, unchanged multimodal config | `//rtp_llm/models/qwen3_next/test:qwen35_moe_text_test` |
| Alias lookup and unchanged architecture inference | `rtp_llm/test/lazy_registry_test.py` |
| SSM preset | `rtp_llm/config/test/server_config_setup_test.py` |
| Shared-source ownership and bounded buffering | `//rtp_llm/model_loader/test:test_tensor_weight_stream` |
| Real transaction ACKs, coverage, dtype and storage | `//rtp_llm/model_loader/test:test_rl_weight_update` |
| Real packed MoE descriptors, initial load vs fresh refit | `//rtp_llm/model_loader/test:test_packed_moe_rl_stream` |
| Multi-node EP=1 identity validation | `//rtp_llm/model_loader/test:test_pure_tp_expert_placement` |

The packed MoE test enables the existing initial-load preshard path, compares
its outputs against a real streamed refit, forbids database reads during refit,
then changes the pushed expert values and checks the resulting weights change.
Device postshuffle still requires its GPU integration checks.

## Source provenance and acceptance boundary

The source material was persisted independently of the retired GPU machines:

- Source project commit: `61796bd4ba2d1eda9148c20e77aa69c943241820`.
- Source archive SHA256:
  `7ef3becb14a56fb56e94b250cc44c3582e1e870fa8fb303d357cb294ecb30610`.
- Source project INDEX SHA256:
  `d8349875a1a92470086b82f57ef482a853ecc858fb38f1efcf80f0a6e6fc6b17`.
- OSS source:
  `RL/projects/qwen35-397b-32gpu-rl-20261104/commits/61796bd4ba2d1eda9148c20e77aa69c943241820/`.

Core changes originate from these source objects:

| Source object | SHA256 |
| --- | --- |
| `native/patches/qwen36-qk-norm-decode-cu.patch` | `551b3cb0b1e750dba1ec5e9a5970d590cf0edacbdc41318580ca8dc8276954f7` |
| `native/patches/qwen36-native-raw-logprobs.patch` | `d82adf9be0a5d3ddfce7f50bb9385fcc62fa93c48376a5735d68ca84deaa0fe9` |
| `native/patches/qwen36-native-rl-guard.patch` | `817aa69255e683c924e5e0942de9c2611e41c2b7de57a4b4f1c868b5545f07a3` |
| `rtp/overlays/loader.py` | `f28b9c709a077f2ff0b308cfddf6d7245582203c063f310f29b24f52e05f1613` |
| `rtp/overlays/weight_manager.py` | `d80bb4947ebce4fed66a85c39f48e191a30848a7ceef614231e9105f034755df` |
| `rtp/overlays/rl_weight_update.py` | `bf7cc2c3a12a537497cb6ea4720c5891be565203a3b50463f543a6893c4606b2` |

The text registration was moved from the explicit deployment import hook into
the ordinary model/tokenizer registries, with the new preferred alias added.
The EP=1 fix was moved from an opt-in deployment hook into the corresponding
`LoadConfig` branch and a standard-library helper. The dispatcher math was
factored into a tested C++ function without changing that computation.
Deployment-specific logging and hooks were not copied wholesale into core.

The original 397B run completed two genuine 32-rank GRPO/AdamW updates and
refit/probes for weight-sync steps 0, 1 and 2. It nevertheless exited with code 1
because its outer cleanup looked up Ray owners after their normal destruction.
The unchanged 19-check diagnostic passed 17 checks and failed the final driver
cleanup and successful-exit checks. These historical facts are not a clean-exit
acceptance of this source port or a new training run.

The independent short HF/native logprob comparison had maximum absolute error
0.01473543 within its 0.05 limit. Longer real trajectories had actor/rollout
selected-probability outliers above 0.5; those are not the same metric or sample
set as the short logprob comparison. Uniform BF16 FSDP forward casting, FP32 HF
source export, native descriptor conversion and kernel promotion are separate
stages. A source dtype receipt does not establish the dtype or precision of the
values ultimately used by a model forward pass. Full long-trajectory numerical
alignment, an error bound of 0.05 for those trajectories, and MoE BI acceptance
are not claimed by these commits.

## Reusable integration and source-port checks

The [verl/rLLM example](../../example/rl/qwen_moe_rl/README.md) preserves the
32-rank FSDP2/two-TP16 integration, bounded CPU AdamW moment staging, official
checkpoint loading, actual delivered trajectory linkage and Python Tool budgets.
Its explicit ARM BF16 backend remains opt-in and deployment settings are required
rather than inherited from the retired machines.

This source port passed four EP1 helper cases, two lazy registration cases,
seven actual transaction-controller CPU fixture cases, and four groups using
the actual selected-logprob C++ helper compiled against CPU libtorch. The
example passed 39 topology/journal/recorded-evidence/trace CPU unit cases, budget
and HTTP/gateway checks, and the 35-case completed-shutdown CPU suite. The latter
patch also had an actual read-only ARM/Ray regression against both closed native
replicas before migration. These checks do not substitute for a full RTP native
build, CUDA kernel tests or a fresh full-model RL run after this source port.
