# BERT Reranker — User-Profile Block-Attention Mask

## Summary

This change extends the multimodal BERT cross-encoder reranker (`vision_bert`) with a
**user-profile-aware** scoring branch, while keeping the original Query–Item (QI)
relevance score **independent of the user**. The independence is enforced by a **block
attention mask** inside the model: the Q-I segment cannot attend to the User segment. On
the engine side the mask is applied through **FlashInfer's `custom_mask`** (not a
hand-packed TRT mask). The feature is opt-in and defaults to off — when disabled the
reranker is byte-identical to the previous behavior.

This document is the single spec + design + change reference for the feature.

---

## 1. Background & Motivation

The existing relevance model judges generic Query↔Item relevance. In e-commerce search,
the same query maps to different real intents for different users (gender, age, price
tier, region, season, brand preference, ...). We add a **user-profile-aware** relevance
path on top of the existing model rather than rebuilding it, to keep training cost,
rollout risk and engineering churn low.

## 2. Requirements & Acceptance

| # | Requirement | Acceptance |
|---|-------------|-----------|
| R1 | Keep the QI path independent of User | With the flag on, the QI 4-class probabilities are the same whether or not the User segment is present (≤ fp32 tiling tol); perturbing User tokens must not change the QI score. |
| R2 | Add a user-aware branch | A second head emits a personalized UQI 4-class score from the `CLS_UQI` hidden, which depends on the User segment. |
| R3 | Reuse the existing model | Extend the existing Q-I BERT (input format, attention, heads) instead of a new model. |
| R4 | Old path unchanged | Flag off → byte-identical to the previous reranker. |
| R5 | Output | One forward → **8 floats** = `score_qi` (4) + `score_uqi` (4). |
| R6 | Golden parity | Reproduce the training golden `score_qi` / `score_uqi` on the validation samples within tiling tolerance. |

## 3. Design

### 3.1 Input format

```
old:  [CLS_QI] query [SEP] item [SEP]                       (+ vision)
new:  [CLS_QI] query [SEP] item [SEP] [CLS_UQI] user [SEP]  (+ vision)
```

`CLS_QI` token id = **101**, `CLS_UQI` token id = **2**. Keeping the Q-I prefix unchanged
means that prefix stays exactly aligned with the old model.

### 3.2 Sequence assembly (exact, validated against the training code)

| Field | Tokens | token_type | segment_ab |
|-------|--------|-----------|------------|
| query | `query_token_ids` (`101 … 102`) | 0 | A (0) |
| item  | `enrich_title_token_ids` (non-pad, `… 102`) | 1 | A (0) |
| user  | `user_profile_token_ids` (`[CLS_UQI] user [SEP]` = `2 … 102`) | 0 | B (1) |
| vision | 4 precomputed tokens (`vision_emb`, 4×128) | — | A (appended after text, segment A) |

- Vision: each 128-dim token → `Linear(128→768)` + LayerNorm, then **appended at the end**
  of the embedded sequence.
- `position_ids`: continuous `arange` over the assembled sequence.
- `cls_uqi_pos`: the index of `CLS_UQI` (the `2`) = start of the user segment; the UQI head
  reads the hidden at this position.

### 3.3 Block attention mask

Sequence split into two segments:

- **A** = `[CLS_QI] query [SEP] item [SEP]` (+ vision tokens)
- **B** = `[CLS_UQI] user [SEP]`

Desired flow — `M = [[1, 0], [1, 1]]` (row = query token, col = attended key):

```
            key/value
            C1   Q    I    C2   U
query C1     1    1    1    0    0
query Q      1    1    1    0    0
query I      1    1    1    0    0
query C2     1    1    1    1    1
query U      1    1    1    1    1
```

i.e. **A attends A, A does NOT attend B, B attends A, B attends B**. This guarantees the
`CLS_QI` head is unaffected by user-profile features. The per-pair visibility rule:

```
visible[i, j] = (seg[j] == 0) OR (seg[i] == 1)
```

No B segment (old QI-only input) → all-visible → standard full attention, byte-identical
to the old behavior.

### 3.4 Dual scoring head

One forward → **8 floats**: `softmax(QI)` (from the `CLS_QI` hidden, position 0) ++
`softmax(UQI)` (from the `CLS_UQI` hidden at `cls_uqi_pos`). The QI head (`w_out`) is
unchanged; the UQI head (`w_out_uqi`) is a separate `Linear(hidden, 4)`.

### 3.5 Loss (training reference)

Both branches distill from a teacher: `L = KL(p_teacher^qi ‖ p_qi) + λ · KL(p_teacher^uqi ‖ p_uqi)`.
(Training side — listed for completeness; not part of the inference change.)

### 3.6 Rejected alternative

Adding an extra user-aware Transformer layer *after* the frozen 4-layer BERT+QI head
(user interacts only post-encoding). Lower intrusion, but the user signal cannot fuse
through in-BERT attention, so its expressiveness is weaker. Rejected in favor of the
in-BERT block mask.

## 4. Why FlashInfer `custom_mask` (engine decision)

The block mask is a **correctness** requirement, so it must run inside the fused prefill
attention. Two options were evaluated:

- **TRT v2 fmha `CUSTOM_MASK`** — *rejected*. Its interface *is* the hardware-swizzled
  packed bitmask: the caller must pre-arrange every mask bit into the tensor-core
  `mma.m16n8k16` fragment layout, and the packing constants differ per architecture
  (`mThreadsPerCTA` = 384 on sm90 warp-specialized vs 128 on sm80/86/89/120). Hardcoding
  one architecture's constants silently produces wrong results on any other GPU, and the
  helper that derives them (`get_warps`) is not in the source tree. Not portable.
- **FlashInfer `custom_mask`** — *chosen*. The caller passes a **logical boolean mask**
  (row-major `[q, kv]`, `True` = visible) to `plan(custom_mask=..., causal=False)`;
  FlashInfer does the tensor-core swizzle / packbits **internally, per architecture**.
  Clean, hardware-portable (sm80/86/89/90/120), fused-fast, pure-Python wiring with no
  C++/CUDA changes.

## 5. Code layout & changes

`vision_bert` (`VisionBert`, registered internally) is a thin subclass of the open-source
`Bert`: it only overrides config parsing and the weight class, and does **not** override
the model forward. So the actual attention runs through the open-source
`models_py/model_desc/bert.py` `BertModel`. The feature splits by layer:

| Layer | Change | Repo |
|-------|--------|------|
| Engine / attention | block mask (derive `segment_ab`, build the FlashInfer custom mask) | **open-source** (this change) |
| Downstream / business | dual head (`w_out_uqi` → 8 floats), renderer inserting `[CLS_UQI user SEP]` | internal `mainse` |

### 5.1 Open-source changes (this commit)

- **`models_py/modules/factory/attention/block_mask.py`** (new) — production helpers:
  - `derive_segment_ab(input_ids, cu_seqlens, b_start_token_id=2, sep_token_id=102)`:
    derive the A/B (0/1) tag from combo token ids; the B span is exactly `[CLS_UQI .. SEP]`
    so vision (after user) stays segment A.
  - `build_flashinfer_block_mask(segment_ab, cu_seqlens)`: build the ragged logical boolean
    mask for FlashInfer (`True` = visible), per-request `[q, kv]` flattened.
- **`models_py/model_desc/bert.py`** — `BertModel.prepare_fmha_impl`: when
  `USE_USER_PROFILE_BLOCK_MASK=1`, prefill, and a B segment exists, derive `segment_ab`,
  build the mask, and route to `PyFlashinferPrefillImpl` with the custom mask. Default off
  → unchanged factory path.
- **`models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py`** —
  `PyFlashinferPrefillAttnOp` / `PyFlashinferPrefillImpl` accept an optional `custom_mask`
  and call `plan(custom_mask=..., causal=False)`; absent → `causal=True` (unchanged).
- **`models/bert.py`**, **`utils/tensor_utils.py`** — supporting hooks
  (`get_token_at_first_id_from_combo_tokens` extracts the `CLS_UQI` hidden for the head).

### 5.2 Internal changes (separate, `mainse`)

- `mainse_module.py` — `dual_head_forward_func`: load `w_out_uqi`, emit `softmax(QI) ++
  softmax(UQI)` (8 floats), reading the `CLS_UQI` hidden.
- `vision_mainse_module.py` — `VisionMainseRenderer.create_input` inserts
  `[CLS_UQI user_profile SEP]` between item and vision (absent → original assembly,
  byte-identical).

### 5.3 Flag gating

`USE_USER_PROFILE_BLOCK_MASK` (default `0`) gates the whole path. Off → the original
factory attention impl is selected and the reranker is byte-identical to before.

## 6. Online deployment (reference)

- **RTP**: receives Query and User-profile token ids; outputs the 8 floats (QI + UQI);
  tair cache key extended with the User profile.
- **Whale**: input extended QI → UQI sequence; the UQI tokenization moves to Whale.
- **RS**: new user-profile feature; the item-grained weighting and SKU selection consume
  the UQI score.

## 7. Verification

Validated against 20 real golden samples (token_ids + vision_emb + `score_qi` /
`score_uqi`) produced by the training model (`modeling_mbert.py`):

1. **Reference reproduces golden** — the original training code on the 20 samples
   reproduces `score_qi` **20/20** and `score_uqi` **20/20** within fp32 tiling tolerance,
   confirming the §3.2 assembly.
2. **Mask equivalence** — `build_flashinfer_block_mask` is **bit-identical** to the
   training code's `build_block_attention_mask` on all 20 samples (0 mismatched cells).
3. **Kernel equivalence** — the FlashInfer `custom_mask` op output matches the pure-torch
   eager reference on GPU, and an all-visible mask reproduces full attention.

Together: the mask the FlashInfer path feeds is identical to the one the golden was
generated with, and FlashInfer applies it identically to the reference — so the user-aware
attention reproduces the golden, and the QI score (the must-match-online correctness
item, R1/R6) is reproduced exactly on all 20 samples.

## 8. Notes

- The naive eager attention (`eager_masked_attention`) is a **test oracle only**, kept out
  of the production path.
- End-to-end RTP serving emitting the 8 floats additionally exercises the internal
  renderer / dual head (§5.2).
- Model definition / checkpoint references are internal (training repo
  `relevance_BERT_model`, branch `aop_cross_yisheng_user_profile`).
