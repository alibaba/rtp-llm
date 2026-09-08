# Kimi K3 MTP

K3 supports two independent draft modes over the existing speculative executor.
Both PD endpoints must select the same mode, checkpoint revision and cache layout.

| Mode | SP_TYPE | SP_MODEL_TYPE | Target feature | Draft pool |
| --- | --- | --- | --- | --- |
| EAGLE3 | eagle3 | kimi_k3_mla_swa_eagle3 | Existing 3H auxiliary layers | Independent SWA |
| MTP | mtp | kimi_k3_mtp | Output AttnRes, before final RMSNorm (H) | Independent FULL MLA |

For MTP, set `SP_CHECKPOINT_PATH` to the local data-disk draft-only checkpoint,
`LOAD_METHOD=fastsafetensors`, `SP_ACT_TYPE=BF16` and `GEN_NUM_PER_CIRCLE=3`.
The target uses its separate `CHECKPOINT_PATH`. All shards referenced by the draft index form one
MTP layer; no checkpoint merge is needed. Candidate count does not replicate
the MTP module or its cache. The single nextn layer starts at the draft config’s `num_hidden_layers`
(93 in the delivered checkpoint); runtime local layer 0 maps to the global cache layer after target.

The two-host smoke defaults to attention weight FP8
(`KIMI_K3_ATTENTION_QUANTIZATION=fp8_per_block`) and dense MLA FP8
(`KIMI_K3_MLA_FP8=1`) for both target and MTP. MLA uses ordinary E4M3 cache
with matching fixed scales on both endpoints; KDA recurrent state retains its
native representation. MoE experts retain checkpoint-native MXFP4. BF16 remains
the activation/output and embedding/head dtype, so `SP_ACT_TYPE=BF16` does not
disable the attention FP8 paths. EAGLE3 retains its separate precision policy.
Set the two FP8 switches to `none` and `0` for a BF16 attention comparison.

The reference is vLLM **v0.28.0**, commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. References below are relative to that
repository, not to a moving main branch.

| vLLM source | Contract | RTP implementation / verification |
| --- | --- | --- |
| models/kimi_k3/nvidia/model.py:1369 | Output AttnRes followed by a separate final norm | K3 target exports H before RMSNorm; EAGLE3 retains its existing output |
| models/kimi_k3/common/mtp.py | Mask embedding at absolute position 0; concat normalized embedding then hidden | MTP layer fusion; positions reconstructed from cache lengths when NoPE inputs omit them |
| models/kimi_k3/nvidia/mtp.py:90 | Disable MTP AttnRes; residual starts at eh_proj output | Dedicated MTP layer; four known unused AttnRes keys explicitly allowed |
| models/kimi_k3/nvidia/mtp.py:144 | Return normalized logits input and pre-norm recurrent state | Forward returns z, executor hidden getter returns h; two-step CPU contract test |
| v1/spec_decode/llm_base_proposer.py:1430 | PP1 shares target embedding and LM head, retaining MTP norm | Bind target global weight tensors before creating the draft Python model |
| models/kimi_k3/nvidia/mla.py | MLA latent norm uses config epsilon; NoPE retains 512+64 latent dimensions | Explicit MTP latent epsilon override; existing RTP MLA kernels and cache format |
| model_executor/layers/logits_processor.py:_apply_head | Default unquantized BF16 head produces BF16 logits before sampler conversion | K3 MTP enables model-dtype LM-head rounding in the C++ model description |
| v1/spec_decode/llm_base_proposer.py:1362 | K3 MTP lacks supports_multimodal_embeddings; proposer uses token embeddings, with vision information in target hidden | Restore RTP media feature-hash IDs to media_placeholder_token_id before shift/lookup; never inject target visual embedding into draft |
| v1/spec_decode/llm_base_proposer.py:846 | Shift tokens, preserve initial positions, feed recurrent h | Existing executor shift/verify/update plus MTP H getter |

The chunk-Prefill terminal pass must replace its ordinary output with recurrent
H **before** cloning `draft_last_hidden_states` for RPC. Applying the getter
after this snapshot would preserve normalized Z in the snapshot and feed the
wrong state into Decode. The MTP executor now performs the override first.

Checkpoint preflight reads every indexed safetensors header and validates all
required source shapes, dtypes, global expert IDs, index membership and payload
bounds before loading. Run it without Torch or GPU initialization:

```bash
python3 rtp_llm/utils/kimi_k3_mtp_checkpoint.py "$SP_CHECKPOINT_PATH"
```

MTP uses ordinary residuals around gated MLA and LatentMoE. It never transfers
the target AttnRes bank. Shared head norm is applied exactly once for logits;
the normalized value must not be fed back to the next draft step.

Cache storage remains one manager with three physical pools: target MLA,
target KDA and draft MLA. The two FULL groups are distinct. KV uses the existing
cache-store/RDMA path; proposal tokens, probabilities and recurrent H use the
existing RPC tensors. MTP reuses this protocol; the upstream FP8 path also
checks MLA format and fixed scales when allocating PD cache.

Both endpoints default to TP8EP8. Set `KIMI_K3_TP_SIZE` and
`KIMI_K3_EP_SIZE` together to change the local topology; the current MegaMoE
path requires TP=EP=world size, and Prefill CP remains unsupported.
Only TP8EP8 has passed the full-model GPU smoke. Lower TP sizes may not fit
the full checkpoint in the available device memory.
The smoke forwards `GEN_NUM_PER_CIRCLE` (default 3) without changing the
number of recurrent modules.
For multimodal requests the target still injects visual embeddings. MTP uses
the corresponding media placeholder token embeddings, matching the official
vLLM capability check. Target chunk inputs clip and repack visual spans; the
draft restores media IDs before chunk slicing/lookahead and clears the visual
features from its own input. The original target/cache-key token stream is
unchanged. This corrects the earlier plan to copy EAGLE3 visual injection.
Set `PREFILL_SMOKE_ARTIFACT_ROOT` and `DECODE_SMOKE_ARTIFACT_ROOT` when the
endpoints use different local data disks; each overrides the common
`SMOKE_ARTIFACT_ROOT` for its own role.
The two-host smoke driver forwards SP_TYPE/SP_MODEL_TYPE; without them it keeps
the EAGLE3 default. Run four-layer flow before the full 93-layer all suite.
For the four-layer `SMOKE_SUITE=flow` fixture, set `SMOKE_CHUNK_TOKENS=4096`:
its approximately 5K-token prompt must cross the actual runtime chunk boundary.
Keep `SMOKE_CHUNK_TOKENS=65536` for the full 93-layer `SMOKE_SUITE=all` suite.
Both suites assert tokenizer-reported input length exceeds the configured
threshold; a request that never reaches a second chunk does not pass flow.

## Validation status

After rebasing onto `origin/feat/k3_dev@b0ee8c10a`, the full 93-layer
FP8 + TP8EP8 + Decode CUDA Graph + MTP smoke passed on 106 (Prefill) and
142 (Decode), at runtime commit `0986ea1fc`. All 31 cases passed. The MTP
chunk case crossed the 65,536-token boundary with a 90,135-token input and
63 accepted draft tokens; long-prefix hits reused 90,112 tokens. Both remote
roles exited 0. The earlier BF16 result is retained in the validation report.

Set `SMOKE_PARALLEL_START=1` to load both roles concurrently; readiness checks
remain mandatory. During this FP8 run, WebTerminal auth expiry interrupted the
controller. Browser-backed auth was restored and the original remote roles
were observed to completion without restarting services or replaying requests.
The report preserves the controller failure and successful remote results
separately.

The implementation includes CPU contract tests and a C++ cache configuration
regression for an independent MTP FULL pool. These do not establish full-model
numerical accuracy or PD correctness. Record actual build, checkpoint-loader,
reference numerical and GPU smoke results separately; acceptance greater than
zero alone is not sufficient evidence. Full MTP GPU validation is pending until
the complete reference/PD test matrix has passed.

Actual results and remaining prerequisites are tracked in
[the validation report](kimi_k3_mtp_validation.md). In particular, the legacy
target keeps its HF MLA latent-norm epsilon of 1e-6; MTP explicitly uses the
vLLM epsilon of 1e-5. Full-model reference comparisons must record this existing
target difference as well as checking the new MTP stages.
