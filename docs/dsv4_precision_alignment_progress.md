# DSV4 Precision Alignment Progress

### 2026-05-19 21:18 CST - Restore Decode Production MoE Path and Match vLLM FP8 Quant eps

- Files changed:
  - `rtp_llm/models_py/modules/dsv4/moe/strategies/grouped_fp4.py`
  - `rtp_llm/models_py/modules/dsv4/moe/strategies/local_loop.py`
  - `rtp_llm/models_py/modules/dsv4/qlinear.py`
  - `rtp_llm/models_py/modules/factory/linear/impl/cuda/fp8_deepgemm_linear.py`
  - `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode.sh`
- Evidence:
  - vLLM `DeepGemmFP4Experts` keeps DeepSeek V4 router weights until
    `deepgemm_unpermute_and_reduce(... topk_weights ...)`, so weights are
    applied after FC2. RTP grouped FP4 already follows this semantic.
  - The previous `local_loop` / shared-fast-path-off env was diagnostic only
    and is not the final parity target.
  - vLLM DeepGemm FP8 input/activation quantization uses `eps=1e-10`
    (`per_token_group_quant_fp8_packed_for_deepgemm` default and fused
    `silu_mul_quant_fp8_packed_triton` scale floor).
  - RTP DSV4 routed/shared DeepGEMM paths had explicit `eps=1e-4`, which can
    change UE8M0-packed activation scales before the first observed MoE expert
    output diff.
- Action:
  - Change DSV4 FP8 activation quant calls on routed FP4 grouped/local-loop
    paths and FP8 DeepGEMM linear path from `eps=1e-4` to `eps=1e-10`.
  - Restore decode launch script to production MoE path by unsetting
    `DSV4_MOE_STRATEGY`, `DSV4_MOE_STRICT_FUSED`, and
    `DSV4_SHARED_EXPERT_FAST_PATH`.
- Next validation:
  - Restart decode only on GPU7 from the restored script.
  - Run teacher-forced 130-token gate and check whether sampler
    `first_bad=oracle_idx=104` moves or disappears.

This document is the live progress log for aligning RTP DeepSeekV4-Flash
`top_k=1` generation with the vLLM oracle on `/data3/q` record 89.

## Rules

- Goal: RTP output token IDs must match the vLLM oracle for 1000 generated tokens.
- Short gates first: do not run 1000-token natural generation until teacher-forced and natural short gates pass.
- Keep `DSV4_GATE_FP32=1`.
- Top-k/router reference policy: use torch/reference semantics for top-k and fp32 router where applicable.
- Do not reintroduce decode FlashMLA `topk_length`.
- Every code/script/env change made for this alignment must be appended to the change log below before or when it is tested.
- Every validation run must record: service env, command intent, output path, first_bad token, and tensor evidence if available.

## Current Known Baseline

- Oracle IDs:
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/vllm_stable_nodump_ignoreeos_len1000_oracle_20260517_220909_record89_20260517_220916/vllm_run01/generated_ids.json`
- Current RTP PD scripts:
  `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_prefill.sh`
  `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode.sh`
- Current teacher-forced sampler log:
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_teacher_forced_len130_sampler_logits_nograph_l0_20260519_175058.jsonl`
- Current vLLM layer0 reference dump:
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l0_internal_pos19689_20260517_232041/rank0_pid2277834_step342.pt`

## Current Evidence

- Token/input alignment at first investigated divergence:
  - vLLM: `positions=[19669]`, `input_ids=[13660]`.
  - RTP embed at the same position matches vLLM.
- Latest stable short-gate result on normal decode path:
  - teacher-forced 130 generated ID prefix matches vLLM hash.
  - sampler first_bad remains at `oracle_idx=84`.
  - top1 `320`, teacher/vLLM `303`.
- Layer0 earliest tensor mismatch:
  - `L00_attn_decode_swa_selected_k_dequant` differs in 14 elems, only 2 elems `>1e-3`, max `0.00390625`.
  - `L00_*selected_cache_logical` differs in 14 bytes, all in the RoPE bf16 tail area.
  - Diff rows map across both prompt-tail and decode-written SWA cache positions, so prefill-only changes are not sufficient.

## Already Excluded

- Decode FlashMLA `topk_length`: removed and must stay removed.
- `DSV4_INV_ROPE_FP8_QUANT_IMPL=legacy`: did not move first_bad.
- `DSV4_HC_IMPL=fallback`: worsened first_bad to oracle_idx 2.
- `DSV4_DECODE_OUT_PROJ_EAGER=1`: worsened first_bad to oracle_idx 2.
- `DSV4_DECODE_KV_ROPE_BACKEND=vllm`: worsened first_bad to oracle_idx 78, so the existing fused+round-before decode KV path is closer than the current Python reference branch.

## vLLM-Reference Direction

The next implementation direction is not another blind env toggle. Build a clear
RTP vLLM-reference path for correctness first, then re-enable production kernels
one by one.

Observed vLLM semantics to mirror:

- DSV4 attention uses one fused op for:
  - Q side: q head RMSNorm + GPT-J RoPE.
  - KV side: GPT-J RoPE + `fp8_ds_mla` quant insert into SWA cache.
- vLLM RoPE table is fp32 packed `cos_sin_cache = cat(cos, sin)`.
- vLLM DSV4 RoPE applies to the last `rope_dim` and uses GPT-J style (`is_neox_style=False`).
- RTP still has several paths using complex `freqs_cis` and separate reference helpers; these must be audited against vLLM's packed cos/sin semantics.

## Change Log

### 2026-05-19 19:40 CST - Planned RoPE Table Generation Alignment

- Files changed:
  - `rtp_llm/models_py/modules/dsv4/rope.py`
  - `rtp_llm/models_py/modules/dsv4/fp8/attention.py`
- Purpose:
  - Match vLLM DSV4 RoPE table generation more closely.
  - Replace CPU `torch.polar(...)` complex table construction with fp32
    `cos/sin` construction.
  - Rebuild FP8 attention RoPE cache directly on the target device in
    `reset_rope_cache(device)`, matching vLLM's CUDA-side `cos_sin_cache`
    creation.
- Expected effect:
  - Reduce or eliminate the 14 RoPE bf16 byte diffs in layer0 selected SWA
    cache.
- Local validation:
  - `python3 -m py_compile rtp_llm/models_py/modules/dsv4/rope.py rtp_llm/models_py/modules/dsv4/fp8/attention.py` passed.
  - `precompute_freqs_cis(..., device="cpu")` returns `torch.complex64` and matches default CPU generation.
- Runtime validation plan:
  - Restart clean PD without `DSV4_DECODE_KV_ROPE_BACKEND=vllm`.
  - Run 130-token teacher gate.
  - Compare latest RTP layer0 selected SWA cache bytes against vLLM dump.

### 2026-05-19 19:32 CST - Progress Log Added

- Files changed:
  - `docs/dsv4_precision_alignment_progress.md`
- Purpose:
  - Create a live progress log requested by the user.
  - Establish the rule that every code/script/env change and validation result must be recorded here.
- Validation:
  - Not applicable; documentation-only change.

## Validation Log

### 2026-05-19 19:49 CST - Handoff Resource Check

- Context:
  - Continuing from the latest RoPE table generation change.
  - Decode is already up on `18880/18881`, GPU7, from this worktree.
  - Prefill is not listening on `18800/18801`.
- Resource check:
  - GPU5/GPU6 were previously occupied by an unrelated `start_port=30120` RTP
    service, which blocked the TP2 prefill process.
  - Rechecked at 19:49 and GPU5/GPU6 are now free.
  - vLLM oracle remains on `18000`, GPU4; it must not be stopped.
  - Unrelated ondev/MTP services remain on GPUs 0-3; they are left untouched.
- Action:
  - Proceed to restart only the prefill service on GPU5/GPU6 using the existing
    `start_rtp_prefill.sh`.
  - Keep current precision env: `DSV4_GATE_FP32=1`, torch top-k backend,
    `DSV4_PREFILL_KV_ROPE_ROUND_BEFORE=1`, and
    `DSV4_PREFILL_SWA_CACHE_KV_BACKEND=vllm`.

### 2026-05-19 19:51 CST - Prefill Resource Blocker Reappeared

- Resource check:
  - `start_port=30120` service restarted and reoccupied GPU5/GPU6 before the
    precision prefill could be restarted.
  - Root PID: `3087649`.
  - Rank PIDs: `3088643`, `3088644`.
  - CWD: `/data3/tanboyu.tby/RTP-LLM/github-opensource-hca-baseline`.
  - Env:
    `CUDA_VISIBLE_DEVICES=5,6`,
    `PYTHONPATH=/data3/tanboyu.tby/bazel_output_cuda13_dsv4/...`.
  - GPU usage: about 272 GB on each of GPU5/GPU6.
- Impact:
  - Current precision prefill cannot start because it requires TP2 on GPU5/GPU6.
  - Decode on `18880/18881` is still alive on GPU7.
  - vLLM oracle on `18000` remains untouched on GPU4.
- Action:
  - Do not kill the `30120` service without explicit confirmation because it is
    outside the current precision worktree.
  - Check whether an offline or already-running precision-compatible prefill
    path exists; otherwise request GPU5/GPU6 release.

### 2026-05-19 20:00 CST - Planned Standalone GPU7 Short Gate

- Reason:
  - PD prefill on GPU5/GPU6 is blocked by the unrelated `30120` service.
### 2026-05-19 20:14 CST - Standalone GPU7 CP Rotate Fix

- Files changed:
  - `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_standalone_gpu7.sh`
- Change:
  - `--cp_rotate_method ALL_GATHER` -> `--cp_rotate_method PREFILL_CP`
- Reason:
  - The previous standalone launch failed during warmup with:
    `execAllGather called but allgather callback not registered via register_comm_ops`.
  - This is a standalone script/communication setup issue, not a precision
    mismatch. `PREFILL_CP` matches the previously usable precision decode
    startup path and avoids the unregistered allgather callback in the
    temporary single-GPU diagnostic.
- Validation target:
  - Restart GPU7 standalone on `18980/18981`.
  - Run the 130-token teacher-forced gate and compare the latest sampler
    `first_bad` against the previous `oracle_idx=84`.

### 2026-05-19 20:24 CST - Standalone GPU7 Teacher Gate Result

- Service:
  - Started via `setsid` on GPU7, `18980/18981`.
  - Health returned `"ok"`.
- 130-token teacher-forced output:
  - Generated ID prefix matched vLLM for the compared 130 tokens.
  - This is expected under teacher forcing and is not by itself a precision
    proof.
- Sampler logits check:
  - File:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_teacher_forced_len130_sampler_logits_standalone_20260519_2000.jsonl`
  - Latest segment: oracle `1..130`.
  - `first_bad=(step=1, oracle_idx=2, top1=1415, teacher=3803)`.
  - `bad_count=6`.
- Conclusion:
  - Standalone TP1 is much worse than the previous PD path
    (`first_bad=oracle_idx=84`), so it must not be used as the precision
    decision path.
  - Next validation must return to the real PD layout: TP2 prefill plus
    decode, using this worktree and the same precision env.

### 2026-05-19 20:29 CST - Return To Real PD Validation

- Reason:
  - Standalone TP1 diverged at `oracle_idx=2`; it is not representative of the
    target production PD path.
- Planned service layout:
  - Prefill: GPU5/6, `18800/18801`, TP2/EP2, from this precision worktree.
  - Decode: GPU7, `18880/18881`, TP1, from this precision worktree.
- Precision env to keep:
  - `DSV4_GATE_FP32=1`
  - `DSV4_INDEXER_TOPK_BACKEND=torch`
  - `DSV4_PREFILL_KV_ROPE_ROUND_BEFORE=1`
  - `DSV4_PREFILL_SWA_CACHE_KV_BACKEND=vllm`
  - `DSV4_DECODE_KV_ROPE_ROUND_BEFORE=1`
  - no decode FlashMLA `topk_length`
  - no `DSV4_DECODE_KV_ROPE_BACKEND=vllm`
- Validation gate:
  - Run 130-token teacher-forced request.
  - Parse sampler logits and compare `first_bad` against previous PD baseline
    `oracle_idx=84`.

### 2026-05-19 20:33 CST - PD RoPE Table Result And New Dump Target

- Result:
  - Real PD 130-token teacher-forced request completed.
  - Generated ID prefix matched vLLM for 130 compared tokens, as expected
    under teacher forcing.
  - Sampler logits latest segment moved from previous best
    `first_bad=oracle_idx=84` to `first_bad=oracle_idx=104`.
  - New failing row:
    `step=102, oracle_idx=104, top1=303, teacher=478, sequence_length=19690`.
- Interpretation:
  - The RoPE table construction change improved alignment but did not solve
    the full score path.
- Files changed:
  - `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode.sh`
- Change:
  - Move layer0 debug target from old `pos19669/oracle84` to new
    `pos19689/oracle104`.
  - New dump dirs:
    - `rtp_moedbg_forced_oracle104_l0_20260519_2033`
    - `rtp_decode_dump_forced_oracle104_l0_20260519_2033`
    - `rtp_meta_dump_forced_oracle104_l0_20260519_2033`
- Next validation:
  - Restart only precision decode on GPU7.
  - Re-run 130-token teacher-forced gate.
  - Compare RTP layer0 `pos19689` dump against vLLM dump for the same logical
    position.

### 2026-05-19 20:46 CST - MoE Routed Strategy A/B Setup

- Evidence at `pos19689`:
  - Correct vLLM file is
    `vllm_l0_internal_pos19689_20260517_232041/rank0_pid2277834_step107.pt`
    (internal `positions=[19689]`, `input_ids=[13660]`).
  - RTP file is
    `rtp_moedbg_forced_oracle104_l0_20260519_2033/rtp_pos19689/rank0_pid3186868_step000.pt`.
  - Layer0 attention input/output and SWA selected cache are bit-identical.
  - MoE input and router topk indices are identical.
  - MoE topk weights differ only at ~`6e-7`.
  - First meaningful layer0 mismatch is MoE expert output:
    - routed output max diff `0.021484375`, mean `0.004836`
    - shared output max diff `0.005859375`, mean `0.001082`
    - ffn output max diff `0.0234375`, mean `0.004981`
- Files changed:
  - `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode.sh`
- Change:
  - Add `DSV4_MOE_STRATEGY=local_loop` for decode-only A/B.
- Purpose:
  - Determine whether the current `grouped_fp4` routed expert path is the
    source of the remaining logits ranking flip at `oracle_idx=104`.
  - This is a diagnostic switch, not a final production decision.

### 2026-05-19 20:49 CST - `DSV4_MOE_STRATEGY=local_loop` A/B Result

- Service:
  - Restarted decode only on GPU7 with `DSV4_MOE_STRATEGY=local_loop`.
  - Kept prefill TP2 on GPU5/6 alive.
- Result:
  - 130-token teacher-forced request completed.
  - Sampler latest segment still has
    `first_bad=(step=102, oracle_idx=104, top1=303, teacher=478)`.
  - Logit gap narrowed but did not flip:
    - grouped path top values at `oracle_idx=104`: `303=35.6611`,
      `478=34.7408`
    - local_loop path top values at `oracle_idx=104`: `303=35.309`,
      `478=34.6141`
- Conclusion:
  - The remaining mismatch is not solely caused by grouped FP4 routed expert
    dispatch. The next target should be MoE shared expert / epilogue and the
    exact vLLM decode-step/token alignment around `oracle_idx=104`.

### 2026-05-19 20:53 CST - Shared Expert Fast Path Diagnostic Switch

- Files changed:
  - `rtp_llm/models_py/modules/dsv4/moe/shared_expert.py`
  - `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode.sh`
- Code change:
  - Add env switch `DSV4_SHARED_EXPERT_FAST_PATH=0`.
  - When disabled, `get_shared_expert_executor()` passes `fast_path=None`, so
    `_run_shared_expert()` can use the generic `shared_experts(x).float()` path.
- Decode A/B env:
  - `DSV4_MOE_STRATEGY=local_loop`
  - `DSV4_MOE_STRICT_FUSED=0`
  - `DSV4_SHARED_EXPERT_FAST_PATH=0`
- Purpose:
  - Isolate whether the fused shared expert path contributes to the remaining
    `oracle_idx=104` ranking flip.
  - This is diagnostic only; not a production decision.

### 2026-05-19 20:56 CST - Shared Expert Fast Path A/B Result

- Service:
  - Restarted decode only with:
    - `DSV4_MOE_STRATEGY=local_loop`
    - `DSV4_MOE_STRICT_FUSED=0`
    - `DSV4_SHARED_EXPERT_FAST_PATH=0`
- Result:
  - 130-token teacher-forced request completed.
  - Sampler latest segment still has
    `first_bad=(step=102, oracle_idx=104, top1=303, teacher=478)`.
  - Top values are identical to the previous local_loop run:
    `303=35.309`, `478=34.6141`.
- Conclusion:
  - Fused shared expert fast path is not the deciding source of the
    `oracle_idx=104` ranking flip.
  - The remaining issue is in the MoE expert numerical semantics / vLLM branch
    selection rather than router/topk or attention.

 - Existing ondev/MTP prefill endpoints are not precision-compatible with this
   worktree and would contaminate the comparison.
- Action:
  - Stop only the current precision decode service on GPU7
    (root PID `3069783`, ports `18880/18881`) to free GPU7.
  - Start a temporary standalone RTP service from this worktree on
    `18980/18981`, GPU7, with teacher forcing enabled.
- Scope:
  - This is a short 130-token diagnostic gate, not the final PD parity proof.
  - It does not touch vLLM oracle on `18000` or the unrelated `30120` service.

### 2026-05-19 20:02 CST - Standalone GPU7 Launch Script Added

- Files changed:
  - `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_standalone_gpu7.sh`
- Purpose:
  - Start a temporary single-GPU standalone RTP service from the current
    precision worktree on `18980/18981`.
  - Keep teacher forcing and sampler logits dump enabled for a 130-token
    diagnostic run.
  - Explicitly assert that `libth_transformer_config`,
    `librtp_compute_ops`, and `libth_transformer` load from this worktree's
    `bazel-bin`.
- Precision env:
  - `DSV4_GATE_FP32=1`
  - `DSV4_INDEXER_TOPK_BACKEND=torch`
  - `DSV4_PREFILL_KV_ROPE_ROUND_BEFORE=1`
  - `DSV4_PREFILL_SWA_CACHE_KV_BACKEND=vllm`
  - `DSV4_DECODE_KV_ROPE_ROUND_BEFORE=1`
  - no `DSV4_DECODE_KV_ROPE_BACKEND=vllm`

### 2026-05-19 19:39 CST - RoPE Table Change Startup Attempt

- Service env:
  - Restarted both prefill and decode from:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_prefill.sh`
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode.sh`
  - Decode started and `/health` returned ok.
  - Prefill failed during multi-rank startup.
- Result:
  - No teacher gate was run because prefill was unavailable.
  - Prefill log:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/logs/prefill_rope_table_20260519_193904.log`
  - Failure summary:
    `Rank 0 process died unexpectedly with exit code -6`.
- Next action:
  - Inspect rank-specific crash output.
  - If the crash is caused by CUDA-side RoPE cache construction during model
    init, change the implementation to keep CPU construction for `__init__`
    but use a vLLM-style CPU cos/sin construction, then only move to device in
    `reset_rope_cache`.

### 2026-05-19 19:16 CST - Prefill `DSV4_PREFILL_SWA_CACHE_KV_BACKEND=vllm`

- Service env:
  - Prefill had `DSV4_PREFILL_SWA_CACHE_KV_BACKEND=vllm`.
  - Both sides kept `DSV4_GATE_FP32=1`.
  - Top-k backend set to torch.
- Result:
  - teacher-forced 130 generated IDs matched vLLM prefix/hash.
  - sampler latest segment first_bad stayed at `oracle_idx=84`.
  - top1 `320`, teacher `303`.
- Tensor evidence:
  - New RTP layer0 selected SWA cache still had 14 byte diffs vs vLLM.
  - Conclusion: this env alone did not eliminate the SWA RoPE byte mismatch.

### 2026-05-19 19:22 CST - Decode `DSV4_DECODE_KV_ROPE_BACKEND=vllm`

- Service env:
  - Prefill unchanged.
  - Decode restarted with `DSV4_DECODE_KV_ROPE_BACKEND=vllm`.
- Result:
  - teacher-forced 130 generated IDs matched by forced feeding.
  - sampler first_bad worsened to `oracle_idx=78`.
  - top1 `6451`, teacher `3975`.
- Tensor evidence:
  - Layer0 selected SWA cache byte diff count stayed 14 vs vLLM.
- Conclusion:
  - Current RTP Python "vllm/reference" decode KV RoPE branch is not actually closer than fused+round-before for this case.
  - Need a real vLLM-reference path based on vLLM's packed fp32 cos/sin + fused op semantics, not this existing branch.

### 2026-05-19 21:09 CST - `eps=1e-10` Teacher Gate Result

- Service:
  - Restarted decode only on GPU7 from the restored production MoE script.
  - Prefill stayed on existing TP2 GPU5/6 service.
- Validation:
  - Teacher-forced 130-token run completed and generated prefix hash matched
    vLLM for the forced output stream.
  - Sampler logits latest segment still has one mismatch:
    `step=102, oracle_idx=104, top1=303, teacher=478, sequence_length=19690`.
  - Top values stayed at grouped-path numbers:
    `303=35.6611`, `478=34.7408`.
- Conclusion:
  - Matching vLLM's FP8 quant `eps=1e-10` is correct for semantic parity but
    does not explain the current ranking flip; the active group's absmax is
    large enough that `eps=1e-4` vs `1e-10` does not change this case.
  - Continue inside L0 MoE expert internals at `pos19689`: compare FC1 output,
    SiLU/clamp output, FP8 quantized activation/scale, and FC2 output against
    vLLM dump.

### 2026-05-19 21:14 CST - Add Narrow Grouped MoE Internal Dumps

- File changed:
  - `rtp_llm/models_py/modules/dsv4/moe/strategies/grouped_fp4.py`
- Action:
  - Added MOEDBG-gated dumps for L0 grouped FP4 routed expert internals:
    `Lxx_moe_routed_x_quant`, `Lxx_moe_routed_x_scale`,
    `Lxx_moe_routed_gate_up`, `Lxx_moe_routed_hidden_quant`,
    `Lxx_moe_routed_hidden_scale`, and `Lxx_moe_routed_down_out`.
- Purpose:
  - Compare RTP against vLLM's existing `L00_moe_routed_x_quant/x_scale` dump
    first, then use gate/hidden/down stats to localize the first divergence
    inside FC1 -> activation quant -> FC2.

### 2026-05-19 21:20 CST - L0 MoE Input Quant Check

- New dump:
  - RTP: `rank0_pid3254004_step000.pt` under
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_moedbg_forced_oracle104_l0_20260519_2033/rtp_pos19689/`
  - vLLM reference remains `rank0_pid2277834_step107.pt`.
- Evidence:
  - `L00_moe_x_in`: byte-identical between RTP and vLLM.
  - `L00_moe_topk_indices`: value-identical; RTP int64 vs vLLM int32 only dtype differs.
  - `L00_moe_topk_weights`: max diff `5.96e-7`.
  - RTP `L00_moe_routed_x_quant` vs CPU reference computed from identical
    `x_in` with 128-group absmax and UE8M0 scale: `0/4096` byte diff.
  - Existing vLLM dump `L00_moe_routed_x_quant/x_scale` is not a valid packed
    DeepGEMM reference for this comparison: its scale shape is `[1,128]` fp8,
    while the actual DeepGEMM packed scale layout is `[1,8]` int32 for
    4096/128 groups packed four per int32. It differs from CPU reference by
    `768/4096` quant bytes.
- Conclusion:
  - RTP L0 MoE input, router, and RTP input FP8 quant are not the current root
    cause.
  - Do not change topk or input quant based on the existing vLLM `x_quant` dump.
  - Next focus: selected routed FP4 expert weight/scale layout and FC1/FC2
    DeepGEMM semantics.

### 2026-05-19 21:24 CST - Align Routed MoE FP8 Activation Group Size to vLLM MegaMoE

- Evidence:
  - vLLM production DSV4 FP4 routed path calls
    `torch.ops.vllm.deepseek_v4_mega_moe_experts` ->
    `deep_gemm.fp8_fp4_mega_moe`, not the hand-written
    `ep_scatter -> grouped_fp8_fp4_gemm -> silu quant -> grouped_fp8_fp4_gemm`
    sequence.
  - vLLM `_stage_deepseek_v4_mega_moe_inputs_kernel` uses `GROUP_K=32` and
    packs four block-32 scale exponents into one int32 per 128 hidden values.
  - RTP `GroupedFP4Strategy` was using `FP8_BLOCK=128` for routed input and
    post-SwiGLU activation quantization, with `recipe_a=(1, 128)`.
- Files changed:
  - `rtp_llm/models_py/kernels/cuda/fp8_kernel/fp8_kernel.py`
  - `rtp_llm/models_py/modules/dsv4/moe/strategies/grouped_fp4.py`
- Action:
  - Fixed UE8M0 scale allocation to use the requested `group_size` instead
    of hard-coding `/128`.
  - Changed grouped routed MoE activation quantization, scatter scale buffer,
    and DeepGEMM `recipe_a` to block-32 via `_MEGA_MOE_ACT_BLOCK=32`.
- Validation plan:
  - Restart decode only.
  - Re-run the focused L0 `pos19689` dump and teacher-forced 130 sampler.
  - First check `L00_moe_routed_x_quant` against vLLM's group-32 dump, then
    check whether first_bad at `oracle_idx=104` disappears.

### 2026-05-19 21:29 CST - Fix ep_scatter Scale Width for Group-32 UE8M0

- Evidence:
  - Decode restart with the group-32 routed MoE change failed during decode
    warmup with `AssertionError` in
    `rtp_llm/models_py/triton_kernels/moe/ep_kernels.py:141`.
  - That assertion used a hard-coded `BLOCK_D=128`, so UE8M0 packed scale
    width was computed as `hidden/128/4=8`.
  - vLLM MegaMoE group-32 staging needs packed scale width
    `hidden/32/4=32`.
- Files changed:
  - `rtp_llm/models_py/triton_kernels/moe/ep_kernels.py`
  - `rtp_llm/models_py/modules/dsv4/moe/strategies/grouped_fp4.py`
- Action:
  - Added `scale_group_size: int = 128` to `ep_scatter`, preserving existing
    default behavior.
  - Passed `scale_group_size=32` only from DSV4 `GroupedFP4Strategy`.
- Next:
  - Restart decode again and verify warmup reaches port `18881`.

### 2026-05-19 22:02 CST - Revert GroupedFP4 Group-32 Half-Change and Add EP1 MegaMoE Switch

- Negative validation:
  - After changing only `GroupedFP4Strategy` routed activation staging to
    group-32, teacher-forced prefix/hash still matched vLLM for 130 tokens, but
    sampler top1 regressed from first_bad `oracle_idx=104` to first_bad
    `oracle_idx=78`.
  - Focused L0 comparison showed `L00_moe_x_in`, topk ids, and routed input FP8
    bytes matched vLLM/CPU reference, while `L00_moe_routed_y` still differed by
    max `0.021484375`. Therefore the remaining mismatch is not solved by just
    changing activation group size.
- Root-cause evidence:
  - vLLM DeepSeek-V4-Flash uses `deep_gemm.fp8_fp4_mega_moe`, with L1 gate/up
    interleave and UTCCP-transposed FP4 scale layout via
    `deep_gemm.transform_weights_for_mega_moe`.
  - RTP TP1/EP1 was auto-selecting `GroupedFP4Strategy` because
    `MegaMoEStrategy.can_handle()` required `ep_size > 1`, and
    `mega_buf._mega_moe_available()` required `torch.distributed` world size
    greater than 1.
  - A single-rank NCCL process group can be initialised in this environment, and
    DeepGEMM reports a valid DSV4 real-shape MegaMoE buffer size for
    `world_size=1`, `hidden=4096`, `intermediate=2048`, `experts=256`.
- Files changed:
  - `rtp_llm/models_py/modules/dsv4/moe/strategies/grouped_fp4.py`
  - `rtp_llm/models_py/modules/dsv4/moe/mega_buf.py`
  - `rtp_llm/models_py/modules/dsv4/moe/strategies/mega.py`
  - `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode.sh`
- Action:
  - Reverted the active `GroupedFP4Strategy` routed activation group size,
    `recipe_a`, and eps back to the previous FP8 block-128 semantics; kept the
    MOEDBG dumps.
  - Added explicit EP1 MegaMoE enablement: `DSV4_MEGA_MOE_EP1=1` allows
    `MegaMoEStrategy` for `ep_size == 1` by creating a one-rank NCCL process
    group only for this opt-in path.
  - Decode start script now forces `DSV4_MOE_STRATEGY=mega`,
    `DSV4_MEGA_MOE_EP1=1`, and disables MegaMoE JIT warmup to avoid compiling
    unused buckets during precision validation.
- Next:
  - Run syntax/import checks.
  - Restart only the decode service on GPU7.
  - Run teacher-forced 130 first; only run natural/1000 if every sampler top1
    matches the vLLM teacher sequence through 130.

### 2026-05-19 22:12 CST - EP1 MegaMoE Runtime Rejected by PyTorch Symmetric Memory

- Validation:
  - `py_compile` passed for `mega_buf.py`, `strategies/mega.py`,
    `strategies/grouped_fp4.py`, `ep_kernels.py`, and `fp8_kernel.py`.
  - Decode was restarted with `DSV4_MOE_STRATEGY=mega`,
    `DSV4_MEGA_MOE_EP1=1`, and `DSV4_MEGA_MOE_JIT_WARMUP=0`.
- Failure evidence:
  - Backend failed while allocating the DeepGEMM symmetric buffer:
    `CUDASymmetricMemory.cu:552 SymmetricMemory: fail to export multicast handle`
    followed by `CUDA driver error: invalid argument`.
  - Traceback enters
    `c10d::symmetric_memory::CUDASymmetricMemoryAllocator::rendezvous`.
- Conclusion:
  - The installed `/opt/conda310` PyTorch symmetric-memory runtime cannot run
    the single-rank MegaMoE path, even though DeepGEMM can compute a valid
    real-shape buffer size for `world_size=1`.
  - EP1 MegaMoE remains behind the explicit `DSV4_MEGA_MOE_EP1=1` code switch,
    but it is not usable for the current precision validation environment.
- Action:
  - Killed the failed decode service on ports `18880/18881`.
  - Reverted the decode start script back to automatic strategy selection
    (`GroupedFP4Strategy`) so validation can continue on the runnable path.
- Next:
  - Restart decode with grouped path and the reverted block-128 semantics.
  - Re-run teacher-forced 130 to restore the previous baseline.
  - Continue tensor-level L0 routed expert comparison at FC1/activation/FC2,
    focusing on grouped kernel vs vLLM MegaMoE fused semantics and weight-scale
    layout rather than input quant/topk.

### 2026-05-19 22:18 CST - Grouped Baseline Restored

- Validation:
  - Restarted decode with automatic `GroupedFP4Strategy` after reverting the
    group-32 half-change from the active grouped path.
  - Teacher-forced 130 completed:
    `rtp_hash == vllm_hash == ac1958f580e2deaf`, `first_diff=null`.
  - Latest sampler segment has exactly one mismatch:
    `oracle_idx=104`, `sequence_length=19690`, RTP top1 `303`, teacher/vLLM
    token `478`; top candidates `[303, 478, 320, 1237, 25728, 876, 1530, 4572]`
    with logits `[35.6611, 34.7408, 34.4649, 26.2531, 23.837, 23.3206, 22.734, 22.5352]`.
- Conclusion:
  - The bad group-32 half-change has been removed from the active path.
  - Current baseline is back to the previous single mismatch at
    `oracle_idx=104`.
- Next:
  - Do not run natural 1000 yet.
  - Compare the focused L0 routed expert intermediates for `pos19689`:
    `routed_gate_up`, `hidden_quant/scale`, and `down_out`; then isolate
    whether the first mismatch is FC1 weight/scale layout, SwiGLU quant, or FC2
    fused semantics.

### 2026-05-19 22:33 CST - Move Router Weight Before L2 Quantization

- Evidence:
  - vLLM MegaMoE CUDA source
    `/opt/conda310/lib/python3.10/site-packages/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`
    applies top-k router weights in the L1 epilogue:
    `swiglu_values = silu(gate) * up * weights`, then computes amax/scale and
    casts those weighted values to FP8 for L2.
  - RTP `GroupedFP4Strategy` was applying router weights in `ep_gather` after
    L2/down projection. That changes the tensor being quantized into L2 and is
    a concrete semantic mismatch after the already-aligned `x_in/topk`.
- Files changed:
  - `rtp_llm/models_py/modules/dsv4/moe/_silu_mul_fp8_quant_triton.py`
  - `rtp_llm/models_py/triton_kernels/moe/ep_kernels.py`
  - `rtp_llm/models_py/modules/dsv4/moe/strategies/grouped_fp4.py`
- Action:
  - Added optional `row_weights` to `silu_mul_fp8_quant_packed` and its split
    variant. When provided, it multiplies each row by the router weight before
    BF16 rounding, amax, FP8 quant, and UE8M0 scale packing.
  - In `GroupedFP4Strategy`, built a per-scattered-row `routed_weights` vector
    using `output_index`, passed it into the fused SwiGLU quant kernel, and
    changed `ep_gather(..., apply_weights=False)` so weights are not applied a
    second time.
  - `py_compile` passed for the changed Python files.
- Next:
  - Restart only decode on GPU7.
  - Run teacher-forced 130 and parse latest sampler segment. Expected signal:
    `oracle_idx=104` should either disappear or the routed_y diff should shrink.

### 2026-05-19 22:14 CST - Router Weight Before L2 Rejected

- Validation:
  - Restarted decode with the router-weight-before-L2 change.
  - Teacher-forced 130 token-id prefix still matched vLLM:
    `first_diff=null`, `rtp_hash == vllm_hash == ac1958f580e2deaf`.
  - Latest sampler-logit segment still has the same real sampling mismatch:
    `oracle_idx=104`, `sequence_length=19690`, RTP top1 `303`, teacher/vLLM
    token `478`; top candidates `[303, 320, 478, 1237, 25728, 876, 4572, 1530]`
    with logits `[35.7308, 34.6877, 34.4682, 26.1601, 23.375, 23.0822, 22.8977, 22.6271]`.
- L0 diff evidence at `pos19689`:
  - `x_in`, top-k indices, and top-k weights stayed aligned with vLLM.
  - `shared_y` unchanged: max `0.005859375`, mean `0.0010820435`.
  - `routed_y` got worse versus vLLM, from old max/mean
    `0.021484375 / 0.0048361216` to new max/mean
    `0.0234375 / 0.005313491`.
- Conclusion:
  - Folding router weights before L2 quantization is not the active-path fix for
    RTP `GroupedFP4Strategy` versus the captured vLLM output. It neither removes
    `oracle_idx=104` nor shrinks the L0 routed diff.
- Action:
  - Reverted this change from the active grouped path:
    `silu_mul_fp8_quant_packed` no longer takes `row_weights`,
    `GroupedFP4Strategy` no longer builds `routed_weights`, and `ep_gather`
    applies router weights after down projection again.
- Next:
  - Restart decode with the reverted grouped baseline.
  - Continue comparing L0 routed intermediates, focusing on gate/up and down
    projection weight/scale/layout semantics rather than router-weight timing.

### 2026-05-19 22:17 CST - Grouped Baseline Revalidated After Revert

- Validation:
  - `py_compile` passed for `_silu_mul_fp8_quant_triton.py`, `ep_kernels.py`,
    and `grouped_fp4.py` after reverting the router-weight-before-L2 change.
  - Restarted decode on GPU7 and re-ran teacher-forced 130.
  - Token-id prefix still matches vLLM:
    `first_diff=null`, `rtp_hash == vllm_hash == ac1958f580e2deaf`.
  - Latest sampler-logit segment is back to the previous baseline:
    exactly one mismatch at `oracle_idx=104`, `sequence_length=19690`, RTP
    top1 `303`, teacher/vLLM token `478`; top candidates
    `[303, 478, 320, 1237, 25728, 876, 1530, 4572]` with logits
    `[35.6611, 34.7408, 34.4649, 26.2531, 23.837, 23.3206, 22.734, 22.5352]`.
- Conclusion:
  - Baseline is restored and the rejected router-weight experiment is no longer
    in the active grouped path.
- Next:
  - Compare `L00_moe_routed_gate_up`, `L00_moe_routed_hidden_quant/scale`, and
    `L00_moe_routed_down_out` against a vLLM-compatible reference for the same
    `x_in/topk` to isolate whether the remaining error is FC1, activation
    quantization, or FC2.

### 2026-05-19 22:24 CST - LocalLoop A/B Does Not Fix oracle_idx=104

- Experiment:
  - Restarted decode with `DSV4_USE_GROUPED_FP4=0` so EP1 falls back to
    `LocalLoopStrategy` instead of `GroupedFP4Strategy`.
  - Verified process env contains `DSV4_USE_GROUPED_FP4=0`.
  - Ran teacher-forced 130 on the same q89 oracle sequence.
- Result:
  - Token-id prefix still matches vLLM:
    `first_diff=null`, `rtp_hash == vllm_hash == ac1958f580e2deaf`.
  - Sampler-logit mismatch remains at the same place:
    `oracle_idx=104`, RTP top1 `303`, teacher/vLLM token `478`.
  - LocalLoop logits at the mismatch are `[303, 478, 320, 1237, 25728, 876,
    1530, 4572]` with `[35.309, 34.6141, 34.3845, 25.821, 23.4228,
    22.7977, 22.5363, 22.3237]`.
- L0 comparison versus vLLM at `pos19689`:
  - Inputs/top-k remain aligned for both strategies.
  - `shared_y` is identical between grouped and local_loop, with max/mean diff
    to vLLM `0.005859375 / 0.0010820435`.
  - `routed_y` grouped diff: max/mean `0.021484375 / 0.0048361216`.
  - `routed_y` local_loop diff: max/mean `0.0232543945 / 0.0053117541`.
- Conclusion:
  - The remaining mismatch is not uniquely caused by the grouped contiguous
    scatter/GEMM/gather path. LocalLoop does not remove `oracle_idx=104` and is
    slightly farther from vLLM on L0 routed output.
  - Focus should move to common Expert FP4 weight/scale semantics, shared expert
    parity, and accumulation/rounding into layer output, not to top-k or grouped
    scatter.
- Next:
  - Restore decode to grouped path for further validation.
  - Add or build a reference harness that computes the same selected routed
    experts from the RTP-loaded weights and compares FC1/SwiGLU/FC2 stage
    outputs against the vLLM MegaMoE dump.

### 2026-05-19 22:32 CST - Offline Routed Expert Reference Narrows Root Cause

- Experiment:
  - Used GPU0 only; left vLLM oracle on GPU4, prefill on GPU5/6, and decode on
    GPU7 untouched.
  - Loaded layer0 selected experts directly from
    `/data3/DeepSeekV4-Flash/model-00002-of-00046.safetensors`.
  - Used RTP dump `L00_moe_x_in`, top-k indices, and top-k weights at
    `pos19689`: indices `[103, 191, 186, 138, 150, 105]`, weights
    `[0.16679035, 0.19092926, 0.08385476, 0.28166676, 0.05351850,
    0.72324038]`.
  - Recomputed routed output with DeepGEMM for four variants:
    input quant eps `1e-4` vs `1e-10`, and router weight applied before vs
    after `w2`.
- Result:
  - `eps=1e-4` and `eps=1e-10` produced identical diffs for this case.
  - `weight_before_w2=True` exactly reproduces `LocalLoopStrategy`:
    diff to local_loop `0.0 / 0.0`, but diff to vLLM
    `0.0232543945 / 0.0053117541`.
  - `weight_before_w2=False` closely reproduces `GroupedFP4Strategy`:
    diff to grouped `0.0016902089 / 0.0001528796`, and diff to vLLM
    `0.0218343437 / 0.0048408825`.
- Conclusion:
  - The remaining `oracle_idx=104` mismatch is not caused by routed input FP8
    quant eps.
  - The previous router-weight-before-L2 change was correctly rejected: it
    matches local_loop exactly, but local_loop is farther from vLLM.
  - The active grouped path is the closer runnable path, but still does not
    match vLLM. Focus now shifts to vLLM MegaMoE/DeepGEMM weight-scale layout
    and fused output semantics, not top-k, router, or input quant.
- Next:
  - Inspect vLLM MegaMoE preparation/layout for FP4 scales and expert order,
    then compare against RTP `prepare_fp4_weight_scale_for_deepgemm` and
    grouped `w13/w2` packing.

### 2026-05-19 22:47 CST - Resume Check and Dump Granularity Gap

- Service check:
  - vLLM oracle remains on port `18000`.
  - RTP prefill remains on `18800/18801`.
  - RTP decode remains on `18880/18881`.
  - `curl http://127.0.0.1:18880/health` returned `ok`.
- Worktree state:
  - Current worktree: `/data3/dsv4_repeat_compare/worktrees/RTP-LLM-dsv4-precision/github-opensource`
  - Current branch: `wt-gho-dsv4-xuanche-indexer-only`.
  - The worktree is dirty; do not revert unrelated changes.
- Dump inspection:
  - Correct vLLM dump:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l0_internal_pos19689_20260517_232041/rank0_pid2277834_step107.pt`
    contains `positions=[19689]` and has L0 MoE tensors through
    `L00_moe_routed_y`, but does not contain MegaMoE internal
    `gate_up`, `hidden_quant/scale`, or `down_out`.
  - RTP grouped dump:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_moedbg_forced_oracle104_l0_20260519_2033/rtp_pos19689/rank0_pid3349467_step000.pt`
    contains grouped internals:
    `L00_moe_routed_hidden_quant`, `L00_moe_routed_hidden_scale`,
    `L00_moe_routed_x_quant`, `L00_moe_routed_x_scale`, and
    `L00_moe_routed_y`.
  - RTP local-loop dump contains only routed output, not comparable grouped
    intermediate tensors.
- Conclusion:
  - Existing vLLM dump is enough to prove the mismatch is at or before L0 MoE
    `routed_y`, but it is not fine-grained enough to localize FC1 vs activation
    quant vs FC2.
  - The next useful step is a minimal offline vLLM-compatible MegaMoE/staging
    harness or a narrowly scoped vLLM instrumentation point around
    `_stage_deepseek_v4_mega_moe_inputs` and `deep_gemm.fp8_fp4_mega_moe`.
    Re-running natural 1000-token generation before this would not add signal.

### 2026-05-19 22:52 CST - L0 Common Tensor Diff Confirms MoE Is First Numeric Split

- Compared the correct vLLM L0 dump against the RTP grouped L0 dump for
  `pos19689`.
- Exact matches:
  - `attn_in`: max/mean `0 / 0`, nonzero `0`.
  - `attn_out`: max/mean `0 / 0`, nonzero `0`.
  - `ffn_in` / `moe_x_in`: max/mean `0 / 0`, nonzero `0`.
  - `topk_indices`: `[103, 191, 186, 138, 150, 105]` exactly match.
- Small pre-MoE numeric drift:
  - `topk_weights`: max `5.960464477539062e-07`, mean
    `2.2227565921184578e-07`.
  - `router_logits`: max `5.245208740234375e-06`, mean
    `1.675100065767765e-06`.
- MoE output diffs:
  - `shared_y`: max/mean `0.005859375 / 0.0010820435`.
  - `routed_y`: max/mean `0.021484375 / 0.0048361216`.
  - `moe_y`/`ffn_out`: max/mean `0.0234375 / 0.0049808566`.
  - `layer00_out`: max/mean `0.02734375 / 0.0013799204`.
- Input quantization layout observation:
  - vLLM `L00_moe_routed_x_quant`: shape `(1,4096)`, dtype
    `float8_e4m3fn`.
  - vLLM `L00_moe_routed_x_scale`: shape `(1,128)`, dtype
    `float8_e4m3fn`.
  - RTP grouped `L00_moe_routed_x_quant`: shape `(1,4096)`, dtype
    `float8_e4m3fn`.
  - RTP grouped `L00_moe_routed_x_scale`: shape `(1,8)`, dtype `int32`
    packed UE8M0 for `group_size=128`.
- Conclusion:
  - The first hard numeric split is MoE, not attention, top-k index, or input
    hidden state.
  - The vLLM dump shows MegaMoE staged routed-input scale granularity of
    `4096 / 128 = 32` elements per scale, while the active RTP grouped path
    uses grouped-contiguous scale packing for `group_size=128`. This must be
    tested offline before changing code because grouped contiguous and MegaMoE
    consume different scale layouts.

### 2026-05-19 22:58 CST - Routed Input Quant Granularity Difference Proven

- Offline recompute on GPU0 from the exact shared `L00_moe_x_in` tensor:
  - group32 quantization reproduces vLLM `L00_moe_routed_x_quant` exactly:
    `diff_count=0`.
  - group128 quantization differs from vLLM by `768` FP8 elements.
  - group128 quantization reproduces RTP grouped `L00_moe_routed_x_quant`
    exactly: `diff_count=0`.
  - group32 quantization differs from RTP grouped by `768` FP8 elements.
  - RTP grouped `L00_moe_routed_x_scale` exactly equals the offline group128
    packed UE8M0 scale:
    first eight int32 values `[1987475318, 1987475062, 1987475062,
    1987475062, 1987475062, 1987475062, 1987475062, 1987475063]`.
- vLLM source evidence:
  - `_stage_deepseek_v4_mega_moe_inputs` launches with `BLOCK_K=128` and
    `GROUP_K=32`.
  - The Triton staging kernel packs four group32 exponents into one int32 per
    128-column block.
- Conclusion:
  - A concrete byte-level difference is now proven: RTP grouped currently
    quantizes routed expert input with group128, while vLLM MegaMoE quantizes
    with group32.
  - Next experiment should change only the active grouped path's routed input
    and hidden activation quant granularity to group32 with matching scale
    packing/DeepGEMM recipe, then rerun the existing `oracle_idx=104`
    teacher-forced sampler check. If DeepGEMM grouped-contiguous cannot consume
    group32 scales with `recipe_a=(1,32)`, the change must be rejected rather
    than papered over.

### 2026-05-19 23:04 CST - Offline MegaMoE Harness Reaches Kernel JIT

- Correction to the previous "next experiment":
  - A prior 21:24-22:02 experiment already tried the half-change of making
    only `GroupedFP4Strategy` use group32 activation quantization, and it
    regressed first_bad from `oracle_idx=104` to `oracle_idx=78`.
  - Therefore the current focus is the complete vLLM MegaMoE fused path, not
    another grouped-contiguous group32 half-change.
- Experiment:
  - Used GPU0 only; did not touch vLLM oracle, RTP prefill, or RTP decode.
  - Built a minimal 6-expert MegaMoE repro for L0 `pos19689` by remapping the
    selected experts `[103, 191, 186, 138, 150, 105]` to local ids `0..5`.
  - Loaded only those layer0 expert weights/scales from
    `/data3/DeepSeekV4-Flash/model-00002-of-00046.safetensors`.
  - Applied the vLLM MegaMoE transforms:
    `transform_sf_into_required_layout(..., (1,32), E)` and
    `transform_weights_for_mega_moe`.
  - Created a one-rank DeepGEMM symmetric buffer on GPU0 and manually staged
    the vLLM group32 input quant/topk data.
- Evidence:
  - Transformed weight shapes:
    - L1 weight `(6, 4096, 2048)`, L1 scale `(6, 4096, 32)`.
    - L2 weight `(6, 4096, 1024)`, L2 scale `(6, 4096, 16)`.
  - Staged input quant exactly matches vLLM dump:
    `xq_eq_vllm=True`.
  - Staged `x_sf` first eight int32 values:
    `[1987475062, 1970698101, 1970697846, 1987475062, 1987475062,
    1970697590, 1987475062, 1987475062]`.
- Failure:
  - `deep_gemm.fp8_fp4_mega_moe` JIT failed before execution:
    `floating-point template parameter is nonstandard`.
  - This is a compile flag issue for the MegaMoE JIT path; the perf BUILD notes
    already mention `DSV4_USE_MEGA_MOE=1 + DG_JIT_CPP_STANDARD=20`.
- Next:
  - Re-run exactly the same offline harness with `DG_JIT_CPP_STANDARD=20`.
  - If the kernel executes, compare the resulting routed output against both
    vLLM `L00_moe_routed_y` and RTP grouped `L00_moe_routed_y`.

### 2026-05-19 23:10 CST - MegaMoE JIT Still Fails With DG_JIT_CPP_STANDARD=20

- Experiment:
  - Re-ran the same 6-expert offline MegaMoE harness with
    `DG_JIT_CPP_STANDARD=20`.
- Result:
  - Staging and weight transforms still succeeded.
  - `xq_eq_vllm=True` again.
  - `deep_gemm.fp8_fp4_mega_moe` still failed at NVCC compile with the same
    error:
    `floating-point template parameter is nonstandard`.
- Additional check:
  - Searched the installed DeepGEMM Python package and vLLM third-party
    DeepGEMM package for `DG_JIT_CPP_STANDARD`, `CPP_STANDARD`, `std=c++`, and
    `c++20`; no Python-level env hook was found.
  - `/tmp/deep_gemm_jit_1428120/cache` contains successful ordinary
    `sm100_fp8_fp4_gemm_1d1d` cubins but no successful MegaMoE cubin.
- Conclusion:
  - The offline harness proves vLLM-style input staging and MegaMoE weight
    transform can be constructed from the exact token/expert subset, but the
    local JIT compile path still needs a C++20/NVCC flag fix before it can
    produce a routed output for direct comparison.
  - Do not infer from this compile failure that MegaMoE math is wrong; the
    failure is before kernel execution.
- Next:
  - Inspect the C++ JIT compiler path inside the DeepGEMM extension/source and
    find the actual way to pass `--std=c++20`.
  - In parallel, keep the active RTP service on grouped baseline and avoid
    running natural 1000 until the L0 MoE routed output cause is isolated.

### 2026-05-19 23:17 CST - MegaMoE Offline JIT Failure Details

- NVCC command probe:
  - With `DG_JIT_PRINT_COMPILER_COMMAND=1`, DeepGEMM emitted:
    `/usr/local/cuda/bin/nvcc ... -std=c++20 ... --gpu-architecture=sm_100f ...`
  - Therefore `DG_JIT_CPP_STANDARD=20` was already effective; the failing
    command still had `-std=c++20`.
  - Both `/usr/local/cuda/bin/nvcc` and `/usr/local/cuda-13.2/bin/nvcc` report
    CUDA `13.2.78`.
- Failure:
  - NVCC still rejects MegaMoE generated code at
    `sm100_fp8_fp4_mega_moe.cuh:35`:
    `float kActivationClamp` as a nonstandard floating-point template
    parameter.
  - The generated instantiation includes a float template argument:
    `0x1.cp+2f` (activation clamp `7.0`).
- NVRTC fallback:
  - Tried `DG_JIT_USE_NVRTC=1`.
  - NVRTC got past the float template parameter but failed earlier while
    compiling layout code:
    `identifier "cudaGridDependencySynchronize" is undefined`.
- Conclusion:
  - The minimal offline MegaMoE harness is blocked by DeepGEMM JIT compiler
    compatibility, not by tensor staging or weight transformation.
  - Since the running vLLM oracle is producing outputs, the next verification
    must confirm the oracle's actual MoE backend/runtime path rather than
    assuming it used the same failing JIT route.

### 2026-05-19 23:17 CST - Switched Main Alignment Target to FlashInfer TRTLLM MXFP4

- Evidence from the correct vLLM dump
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l0_internal_pos19689_20260517_232041/rank0_pid2277834_step107.pt`:
  - `extra.positions=[19689]`, `extra.input_ids=[13660]`.
  - vLLM recorded `L00_moe_routed_x_quant`, `L00_moe_routed_x_scale`,
    `L00_moe_routed_packed_topk`, `L00_moe_routed_trtllm_out`,
    `L00_moe_routed_fused_out`, `L00_moe_routed_finalize_out`,
    `L00_moe_routed_y`.
  - `L00_moe_routed_trtllm_out == L00_moe_routed_fused_out ==
    L00_moe_routed_finalize_out == L00_moe_routed_y` exactly.
  - Therefore the first routed-MoE numeric difference is inside the TRTLLM
    routed MoE expert kernel output itself, not a later finalize/reduce stage.
- Cross-dump comparison against RTP grouped baseline at the same token:
  - `L00_moe_topk_indices` exact match.
  - `L00_moe_topk_weights` max/mean diff `5.9604645e-7 / 2.2227566e-7`.
  - RTP grouped `L00_moe_routed_x_quant` is DeepGEMM FP8 group128, while vLLM
    TRTLLM uses FlashInfer MXFP8 activation quantization; shapes/scales differ.
  - `L00_moe_routed_y` max/mean diff remains
    `0.021484375 / 0.0048361216`.
- Source-level conclusion:
  - The vLLM oracle path is FlashInfer TRTLLM MXFP4/MXFP8 FusedMoE, not the
    DeepGEMM grouped-contiguous path currently used by RTP's default
    `GroupedFP4Strategy`.
  - Continuing to tune RTP grouped FP4 quantization is not a same-backend
    reference alignment problem. It is comparing different MoE kernels and
    different activation quant formats.
- Code change:
  - Added explicit opt-in strategy
    `rtp_llm/models_py/modules/dsv4/moe/strategies/flashinfer_trtllm_fp4.py`.
  - Registered it in
    `rtp_llm/models_py/modules/dsv4/moe/strategies/__init__.py`.
  - Strategy name: `flashinfer_trtllm_fp4`.
  - Selection is explicit only:
    `DSV4_MOE_STRATEGY=flashinfer_trtllm_fp4` or
    `DSV4_USE_FLASHINFER_TRTLLM_FP4=1`.
  - It does not replace the default production `grouped_fp4` path.
  - It repacks RTP contiguous routed weights/scales into vLLM/FlashInfer
    TRTLLM layout:
    contiguous `[w1, w3]` -> interleaved `[w3_0, w1_0, w3_1, w1_1, ...]`,
    then FlashInfer row permutation and `nvfp4_block_scale_interleave`.
  - Forward path uses FlashInfer `mxfp8_quantize` and
    `trtllm_fp4_block_scale_routed_moe`, with packed
    `(expert_id << 16) | bf16(router_weight).bits`, matching vLLM.
- Verification:
  - `py_compile` passed for the new strategy and registry file.
  - Import check printed `flashinfer_trtllm_fp4`.
- Next:
  - Start an isolated RTP decode instance with
    `DSV4_MOE_STRATEGY=flashinfer_trtllm_fp4` on a non-oracle GPU/port.
  - Re-run the existing teacher-forced `oracle_idx=104` MoE dump and compare
    L0 `routed_trtllm_out`/`routed_y` to the vLLM dump before running 1000
    natural tokens.

### 2026-05-19 23:24 CST - FlashInfer TRTLLM Startup Probe

- Observation:
  - Previous isolated decode attempt on `18980/18981` left an empty log:
    `/tmp/rtp_flashinfer_trtllm_decode_18980.log` size `0`.
  - No listener existed on port `18980`.
  - The baseline script
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode.sh`
    hard-codes `START_PORT=18880`, `CUDA_VISIBLE_DEVICES=7`, and unsets
    `DSV4_MOE_STRATEGY`, so it cannot be reused by only overriding env from
    the outside.
- Probe:
  - Ran a minimal import/selector check with `CUDA_VISIBLE_DEVICES=0`.
  - PyTorch reported `torch.cuda.get_device_capability() == (10, 3)`.
  - FlashInfer imports succeeded:
    `mxfp8_quantize`, `trtllm_fp4_block_scale_routed_moe`,
    `nvfp4_block_scale_interleave`, and
    `get_w2_permute_indices_with_cache`.
  - With `DSV4_MOE_STRATEGY=flashinfer_trtllm_fp4`, `select_strategy`
    selected `flashinfer_trtllm_fp4`.
- Conclusion:
  - The new strategy is selectable in this runtime.
  - The failed start was a launch/script issue, not a proven FlashInfer
    availability or `can_handle` failure.
- Next:
  - Create a dedicated isolated decode launcher that explicitly sets
    `START_PORT=18980`, uses GPU0, keeps
    `DSV4_MOE_STRATEGY=flashinfer_trtllm_fp4`, and writes fresh MoE dumps to a
    separate output directory.

### 2026-05-19 23:25 CST - Added Isolated FlashInfer TRTLLM Decode Launcher

- File added:
  - `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode_flashinfer_trtllm_18980.sh`
- Intentional differences from the baseline decode launcher:
  - `CUDA_VISIBLE_DEVICES=0`
  - `START_PORT=18980`
  - decode endpoint in `MODEL_SERVICE_CONFIG` is `127.0.0.1:18980`
  - `DSV4_MOE_STRATEGY=flashinfer_trtllm_fp4`
  - `DSV4_USE_FLASHINFER_TRTLLM_FP4=1`
  - fresh sampler/MoE/decode/meta dump directories with
    `rtp_flashinfer_trtllm_*_20260519_2324` names.
- Safety:
  - Does not touch existing vLLM oracle on `18000`.
  - Does not touch existing RTP prefill on `18800`.
  - Does not touch existing RTP baseline decode on `18880`.
  - The script includes a pre-start Python probe that asserts the loaded `.so`
    files come from this worktree's `bazel-bin`, verifies
    `cublas_gemm_bf16_bf16_fp32` for `DSV4_GATE_FP32=1`, and asserts the MoE
    selector chooses `flashinfer_trtllm_fp4`.

### 2026-05-19 23:31 CST - Foreground Startup Diagnosis

- Issue:
  - A first `nohup` launch returned a parent PID but exited within seconds,
    and `/tmp/rtp_flashinfer_trtllm_decode_18980.log` remained size `0`.
- Diagnostic command:
  - Ran `timeout 45s bash -x
    /data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode_flashinfer_trtllm_18980.sh`
    in the foreground.
- Evidence:
  - The script executed the pre-start Python probe successfully.
  - Loaded `.so` files were direct children of this worktree's `bazel-bin`.
  - `torch` was `2.11.0+cu130`.
  - `cuda_capability` was `(10, 3)`.
  - `moe_strategy_probe flashinfer_trtllm_fp4` printed before server start.
  - The RTP process entered model loading and loaded all `46/46`
    fastsafetensors shards before the `timeout` killed the diagnostic process.
  - During the diagnostic run, port `18980` briefly had a frontend listener,
    but `/health` timed out because backend initialization had not completed.
- Conclusion:
  - The new launcher and strategy selection are valid.
  - The first background attempt was a process lifetime/logging issue, not a
    strategy import or selector failure.
- Next:
  - Start the service through a detached `tmux` session so it survives the
    short-lived exec command, then wait for `18980/18981` health before running
    teacher-forced validation.

### 2026-05-19 23:36 CST - Fixed FlashInfer Permute Helper Under Meta Construction

- Experiment:
  - Started the isolated decode launcher in a detached tmux session.
- Result:
  - The service loaded all `46/46` checkpoint shards, then backend
    initialization failed.
  - `logs/main_0.log` showed:
    `NotImplementedError: Cannot copy out of meta tensor; no data!`
  - Traceback:
    `flashinfer/fused_moe/core.py:get_w2_permute_indices_with_cache()`
    called from
    `rtp_llm/models_py/modules/dsv4/moe/strategies/flashinfer_trtllm_fp4.py:_trtllm_transform_weights()`.
- Root cause:
  - RTP constructs `V4Transformer` under `with torch.device("meta")` and later
    materializes the model.
  - FlashInfer's permutation helper creates temporary tensors without an
    explicit device and then calls `.to(dst_w2_weight.device)`.
  - Under RTP's meta construction scope, those temporary tensors become meta
    tensors, and copying them to CUDA raises
    `Cannot copy out of meta tensor`.
- Code change:
  - Added local helper `_get_w2_permute_indices_on_device()` in
    `flashinfer_trtllm_fp4.py`.
  - The helper calls FlashInfer's permutation utility inside
    `with torch.device(weight.device)` so temporary tensors are created on the
    real CUDA device even when the outer model construction context is `meta`.
  - Replaced the four direct calls for `w13`, `s13`, `w2`, and `s2` with this
    wrapper.
- Verification:
  - `py_compile` passed for `flashinfer_trtllm_fp4.py` and strategy
    `__init__.py`.
  - A minimal probe created a CUDA weight, entered `with torch.device("meta")`,
    called `_get_w2_permute_indices_on_device()`, and got:
    `perm_device cuda:0 is_meta False shape (2048,) dtype torch.int64`.
- Current blocker:
  - GPU0-3 are now occupied by an unrelated Bazel smoke
    `v4_flash_pd_fusion_cp4ep4_batch_mixlen_reuse_sm100_fp8`; they are not
    occupied by the failed `18980` service.
- Next:
  - Wait until GPU0-3 are free, or pick another explicitly free GPU, then
    restart `18980` and continue to teacher-forced L0 MoE dump comparison.

### 2026-05-19 23:49 CST - FlashInfer TRTLLM Decode Startup Still In JIT

- Experiment:
  - Continued the detached tmux launch `dsv4_flashinfer_18980` for the explicit
    FlashInfer TRTLLM FP4 decode path.
- Current runtime state:
  - Frontend port `18980` is listening.
  - Dash port `18988` is listening.
  - Backend process `3742503` is alive on GPU0 with about `152.5 GiB` allocated.
  - Backend grpc port `18981` is not listening yet.
  - `GET /health` on `18980` still times out.
- Evidence:
  - `/tmp/rtp_flashinfer_trtllm_decode_18980.log` and
    `/data3/tanboyu.tby/.cache/flashinfer/0.6.11.post1/103a/flashinfer_jit.log`
    show FlashInfer cubin downloads finishing through
    `trtllm/gen/SparsityDecl.h`.
  - Process fd inspection shows the backend holds
    `/data3/tanboyu.tby/.cache/flashinfer/0.6.11.post1/103a/cached_ops/tmp/fused_moe_trtllm_sm100.lock`.
  - Child process inspection shows active FlashInfer JIT compilation:
    `ninja -v -C .../cached_ops/fused_moe_trtllm_sm100`.
  - `nvcc` is compiling
    `flashinfer/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu`
    and `ptxas` is assembling `sm_103a`.
  - The generated object files in
    `.../cached_ops/fused_moe_trtllm_sm100/` are still incomplete; no final
    `.so` exists yet.
- Conclusion:
  - This is a first-run FlashInfer TRTLLM fused MoE JIT startup delay, not yet a
    token-precision result.
  - No MoE precision logic is changed at this checkpoint.
- Next:
  - Wait for `fused_moe_trtllm_sm100.so` to finish linking and for `18981` to
    listen.
  - Then run teacher-forced validation on record 89 against `18980/18981` and
    compare the new RTP L0 MoE dump against the vLLM dump at
    `rank0_pid2277834_step107.pt`.

### 2026-05-19 23:54 CST - FlashInfer TRTLLM JIT Still Active

- Experiment:
  - Waited another 120 seconds after confirming `ptxas` was active.
- Evidence:
  - `ptxas` process `3746078` is still running at about `99.5%` CPU.
  - Its elapsed time reached about `7m29s`.
  - RSS increased to about `2.1 GiB`, indicating active compilation progress.
  - `18980` and `18988` remain listening; `18981` is still absent.
  - No `fused_moe_trtllm_sm100.so` has been linked yet.
- Conclusion:
  - Still no precision verdict. The explicit FlashInfer TRTLLM FP4 decode path
    is blocked on first-run JIT compilation.
  - Do not restart or modify precision code while this compilation is active.
- Next:
  - Continue waiting for `ptxas`/`ninja` to finish, then immediately run
    teacher-forced validation once `18981` becomes available.

### 2026-05-19 23:59 CST - FlashInfer TRTLLM FP4 Teacher-Forced Prefix Matches vLLM

- Service state:
  - `fused_moe_trtllm_sm100.so` was linked successfully at
    `/data3/tanboyu.tby/.cache/flashinfer/0.6.11.post1/103a/cached_ops/fused_moe_trtllm_sm100/fused_moe_trtllm_sm100.so`.
  - Backend grpc `18981`, frontend `18980`, and dash `18988` are all listening.
  - `GET /health` on `18980` returned `"ok"`.
  - `logs/main_0.log` reports server startup took `1021.47s`; most of this was
    first-run FlashInfer TRTLLM fused MoE JIT and cubin download.
- Teacher-forced command:
  - Ran `rtp_vllm_precision.py run-teacher` for record 89 against
    `18980/18981`, prefill `18800/18801`, `top_k=1`, `prefix_len=130`,
    `max_new_tokens=130`.
- Token result:
  - Output directory:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/rtp_teacher_forced_record89_len130_20260519_235809_record89_20260519_235821`.
  - `compare_len=130`.
  - `first_diff=null`.
  - `equal_prefix=true`.
  - RTP hash and vLLM hash are both `ac1958f580e2deaf`.
- New RTP L0 dump:
  - `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle104_l0_20260519_2324/rtp_pos19689/rank0_pid3742503_step000.pt`.
- Tensor comparison against vLLM dump
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l0_internal_pos19689_20260517_232041/rank0_pid2277834_step107.pt`:
  - `L00_moe_x_in`: exact equal.
  - `L00_moe_topk_indices`: exact equal `[103, 191, 186, 138, 150, 105]`.
  - `L00_moe_topk_weights`: max diff `5.960464477539062e-07`, mean diff
    `2.2227565921184578e-07`.
  - `L00_moe_routed_x_quant`: exact byte equal.
  - `L00_moe_routed_x_scale`: exact byte equal.
  - `L00_moe_routed_packed_topk`: exact equal.
  - `L00_moe_routed_trtllm_out`: exact equal.
  - `L00_moe_routed_y`: exact equal after dtype normalization
    (vLLM bf16, RTP dump fp32 values).
  - `L00_moe_shared_y`: still differs; max diff `0.005859375`, mean diff
    `0.00108204351272434`.
- Conclusion:
  - The explicit FlashInfer TRTLLM FP4 routed MoE path now matches vLLM at the
    L0 routed-MoE tensor level for the inspected position.
  - The prior grouped-FP4 routed MoE mismatch was removed by using the same
    FlashInfer MXFP8 + TRTLLM FP4 kernel family as the vLLM oracle.
  - Remaining observed difference at this checkpoint is in shared expert output,
    but it did not break the first 130 teacher-forced token IDs.
- Next:
  - Run a longer teacher-forced or natural validation up to 1000 generated token
    IDs using the same `18980/18981` service.
  - If a later token diverges, dump and compare the first divergent step instead
    of changing MoE code speculatively.

### 2026-05-20 00:09 CST - 1000 Teacher-Forced IDs Match, Natural Top1 Still Diverges

- Experiment:
  - Ran `rtp_vllm_precision.py run-teacher` for record 89 against the explicit
    FlashInfer TRTLLM FP4 decode service `18980/18981`, using prefill
    `18800/18801`, `top_k=1`, `prefix_len=1000`, and `max_new_tokens=1000`.
- Teacher-forced token result:
  - Output directory:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/rtp_teacher_forced_record89_len1000_flashinfer_trtllm_20260520_0000_record89_20260520_000243`.
  - `rtp_len=1001`.
  - `vllm_len=1000`.
  - `compare_len=1000`.
  - `first_diff=null`.
  - `equal_prefix=true`.
  - RTP hash and vLLM hash are both `986b77c92c844fc6`.
- Sampler natural-top1 finding:
  - Sampler dump:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_teacher_forced_len130_sampler_logits_flashinfer_trtllm_18980_20260519_2324.jsonl`.
  - Segment 0, the 130-token teacher-forced run, has `129` sampled entries and
    `2` natural-top1 mismatches before teacher override.
  - Segment 1, the 1000-token teacher-forced run, has `998` sampled entries and
    `24` natural-top1 mismatches before teacher override.
  - First mismatch in both segments:
    - `oracle_idx=78`.
    - `step=19664`.
    - Teacher/vLLM token id: `3975`.
    - RTP natural top1 before teacher override: `3003`.
    - Teacher token rank in RTP logits: `1`.
    - RTP top1 logit: `30.1747`.
    - Teacher token logit: `29.8892`.
    - Gap: about `0.2855`.
    - RTP top candidates begin with
      `[3003, 3975, 6451, 17839, 27620, 108889, 389, 78457]`.
- Conclusion:
  - The current true state is not "natural 1000-token generation is aligned".
  - What is proven is narrower: under teacher forcing, RTP can consume the vLLM
    token stream for 1000 generated tokens without produced-token mismatch.
  - Natural greedy top1 still diverges at the first observed point
    `oracle_idx=78 / global pos=19664`, before teacher forcing replaces the
    sampled token.
- Next:
  - Restart only the `18980` FlashInfer TRTLLM decode service with dump position
    changed from `19689` to `19664`.
  - Keep `DSV4_GATE_FP32=1`, `DSV4_INDEXER_TOPK_BACKEND=torch`,
    `DSV4_MOE_STRATEGY=flashinfer_trtllm_fp4`, and
    `DSV4_USE_FLASHINFER_TRTLLM_FP4=1`.
  - Run a short teacher-forced validation with enough tokens to reach
    `oracle_idx=78`, then compare the new RTP `pos19664` dump with vLLM
    `rank0_pid2277834_step082.pt`.

### 2026-05-20 00:10 CST - Retargeted 18980 Dump Instrumentation To First Natural Mismatch

- Launcher changed:
  - `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode_flashinfer_trtllm_18980.sh`.
- Dump position changed:
  - `MOEDBG_GLOBAL_POS`: `19689` -> `19664`.
  - `DSV4_DECODE_DUMP_POS`: `19689` -> `19664`.
  - `DSV4_META_DUMP_POS`: `19689` -> `19664`.
- Dump cases changed:
  - `rtp_pos19689` -> `rtp_pos19664`.
- Dump directories changed to fresh output roots:
  - `.../rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010`.
  - `.../rtp_flashinfer_trtllm_decode_dump_forced_oracle78_l0_20260520_0010`.
  - `.../rtp_flashinfer_trtllm_meta_dump_forced_oracle78_l0_20260520_0010`.
- Precision-affecting env left unchanged:
  - `DSV4_GATE_FP32=1`.
  - `DSV4_INDEXER_TOPK_BACKEND=torch`.
  - `DSV4_MOE_STRATEGY=flashinfer_trtllm_fp4`.
  - `DSV4_USE_FLASHINFER_TRTLLM_FP4=1`.
  - No decode FlashMLA `topk_length` path was reintroduced.
- Next:
  - Restart only tmux session `dsv4_flashinfer_18980`, wait for health, run a
    short teacher-forced request, and compare `pos19664` tensors against vLLM.

### 2026-05-20 00:31 CST - First Divergence Narrowed To L0 Shared Expert

- Experiment:
  - Restarted only tmux session `dsv4_flashinfer_18980` with dump target
    `pos19664`.
  - Ran a short 100-token teacher-forced request against `18980/18981`.
- Token result:
  - Output directory:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/rtp_teacher_forced_record89_len100_20260520_001233_record89_20260520_001249`.
  - `compare_len=100`.
  - `first_diff=null`.
  - `equal_prefix=true`.
  - RTP hash and vLLM hash are both `b062a9477fc5f8d1`.
- New RTP dumps:
  - MoE dump:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010/rtp_pos19664/rank0_pid3760063_step000.pt`.
  - Logits dump:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010/rtp_pos19664/rank0_pid3760063_step001.pt`.
  - Decode dump:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_decode_dump_forced_oracle78_l0_20260520_0010/rtp_pos19664/rank0_pid3760063_step000.pt`.
  - Meta dump directory did not appear for this run.
- Comparison against vLLM
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l0_internal_pos19689_20260517_232041/rank0_pid2277834_step082.pt`:
  - `input_ids=[3975]`, `positions=[19664]` match the first natural mismatch.
  - `L00_attn_residual`: exact equal after shape normalization.
  - `L00_moe_x_in`: exact equal.
  - `L00_moe_topk_indices`: exact equal `[155, 110, 87, 23, 11, 40]`.
  - `L00_moe_topk_weights`: max diff `1.5273690223693848e-07`, mean diff
    `8.257726591409664e-08`.
  - `L00_moe_routed_x_quant`: exact equal.
  - `L00_moe_routed_x_scale`: exact equal.
  - `L00_moe_routed_packed_topk`: exact equal.
  - `L00_moe_routed_trtllm_out`: exact equal.
  - `L00_moe_routed_y`: exact equal.
  - `L00_moe_shared_y`: differs, max diff `0.01953125`, mean diff
    `0.0038486008998006582`, rms `0.004879513755440712`.
  - `L00_ffn_out` / RTP `L00_decode_ffn_out`: differs, max diff
    `0.0234375`, mean diff `0.003855752293020487`.
- Conclusion:
  - The first observed natural top1 mismatch is not caused by L0 attention,
    router top-k, or routed MoE.
  - The earliest confirmed divergent tensor is L0 shared expert output.
  - vLLM has `L00_moe_shared_gate_up`, `L00_moe_shared_hidden`, and
    `L00_moe_shared_out` in the oracle dump, while RTP only had
    `L00_moe_shared_y`, so the next step is to add RTP shared-expert
    intermediate debug tensors.

### 2026-05-20 00:34 CST - Added Shared Expert Intermediate Debug Dumps

- Code changed:
  - `rtp_llm/models_py/modules/dsv4/moe/moe_layer.py`.
  - `rtp_llm/models_py/modules/dsv4/moe/shared_expert.py`.
- Debug-only changes:
  - Set `self.shared_experts._dsv4_layer_id = layer_id` in `MoeLayer` so the
    shared expert can emit layer-scoped dump names.
  - In the generic shared expert path, record:
    - `Lxx_moe_shared_gate_up`.
    - `Lxx_moe_shared_hidden`.
    - `Lxx_moe_shared_out`.
  - In the fused shared expert fast path, record:
    - `Lxx_moe_shared_gate_up`.
    - `Lxx_moe_shared_hidden` computed only for debug comparison.
    - `Lxx_moe_shared_hidden_quant`.
    - `Lxx_moe_shared_hidden_scale`.
    - `Lxx_moe_shared_out`.
- Precision-affecting changes:
  - None intended. These records are gated by `MOEDBG >= 2`.
  - No router/top-k logic changed.
  - `DSV4_GATE_FP32=1` remains required.
  - Decode FlashMLA `topk_length` remains absent.
- Verification:
  - `py_compile` passed for:
    - `rtp_llm/models_py/modules/dsv4/moe/shared_expert.py`.
    - `rtp_llm/models_py/modules/dsv4/moe/moe_layer.py`.
- Next:
  - Restart only `18980`, rerun the 100-token teacher-forced request, then
    compare RTP `shared_gate_up/hidden/out` with vLLM at `pos19664`.

### 2026-05-20 00:45 CST - Shared Expert Activation Semantics Identified

- Experiment:
  - Compared vLLM and RTP dumps at the first natural top1 mismatch checkpoint
    `record89 / oracle_idx=78 / pos=19664`.
  - Files:
    - vLLM: `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l0_internal_pos19689_20260517_232041/rank0_pid2277834_step082.pt`.
    - RTP: `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010/rtp_pos19664/rank0_pid3771706_step000.pt`.
- Confirmed shared expert input equality:
  - `L00_moe_shared_gate_up`: exact equal, shape `(1, 4096)`, dtype `bf16`.
  - Gate range: `[-2.609375, 2.53125]`; up range: `[-3.234375, 2.578125]`.
  - No clamp is active at this checkpoint: `gate > 7` count is `0`, `abs(up) > 7` count is `0`.
- Activation recompute result:
  - vLLM `L00_moe_shared_hidden` exactly matches BF16-input PyTorch semantics:
    `F.silu(gate_bf16) * up_bf16`, cast/stored as BF16.
    - max diff `0`, nonzero `0 / 2048`.
  - vLLM does **not** match FP32 activation/multiply semantics:
    `F.silu(gate.float()) * up.float()` cast to BF16.
    - max diff `0.015625`, nonzero `523 / 2048`.
  - RTP current `L00_moe_shared_hidden` exactly matches FP32 activation/multiply semantics:
    `F.silu(gate.float()) * up.float()` cast to BF16.
    - max diff `0`, nonzero `0 / 2048`.
  - RTP does **not** match BF16-input PyTorch semantics:
    - max diff `0.015625`, nonzero `523 / 2048`.
- Cross-framework tensor diffs at the same checkpoint:
  - `L00_moe_shared_gate_up`: exact equal.
  - `L00_moe_shared_hidden`: max diff `0.015625`, mean `0.000190214`, rms `0.000845982`, nonzero `523 / 2048`.
  - `L00_moe_shared_out` / `shared_y`: max diff `0.01953125`, mean `0.0038486009`, rms `0.0048795138`, nonzero `3659 / 4096`.
- Conclusion:
  - The first confirmed semantic mismatch is the shared expert SiLU-mul precision/order.
  - vLLM oracle uses BF16-input activation behavior for this op, while RTP's fused/debug path currently behaves like FP32 activation/multiply before BF16 storage.
  - This is a real model-logic mismatch and should be fixed before investigating later layers.
- Next:
  - Inspect RTP shared expert fused kernels (`_silu_mul_fp8_quant_triton.py` and related split/debug path) and change only the shared expert activation path to match vLLM/PyTorch BF16-input semantics.
  - Keep `DSV4_GATE_FP32=1` and do not touch router/top-k or decode FlashMLA `topk_length`.
  - Rebuild/restart only `dsv4_flashinfer_18980`, rerun the 100-token teacher-forced dump, and confirm `L00_moe_shared_hidden/out` align at `pos19664` before checking natural top1.

### 2026-05-20 00:55 CST - Patched Shared Expert Fast Path To Match vLLM BF16 Activation

- Code changed:
  - `rtp_llm/models_py/modules/dsv4/moe/_silu_mul_fp8_quant_triton.py`.
  - `rtp_llm/models_py/modules/dsv4/moe/shared_expert.py`.
- Implementation:
  - Added an explicit `bf16_activation` flag to `silu_mul_fp8_quant_packed` and
    `silu_mul_fp8_quant_packed_from_parts`.
  - Default remains `False`, so existing `grouped_fp4` routed MoE calls keep the
    previous FP32 activation/multiply behavior.
  - `FusedSharedExpertFastPath.run()` now calls
    `silu_mul_fp8_quant_packed(..., bf16_activation=True)`.
  - Shared expert debug `Lxx_moe_shared_hidden` now recomputes
    `(F.silu(gate_bf16) * up_bf16).to(bf16)`, matching vLLM's observed
    reference behavior.
- Scope control:
  - No router/top-k code changed.
  - No decode FlashMLA `topk_length` code added.
  - `DSV4_GATE_FP32=1` remains unchanged.
  - The production default grouped FP4 routed path was not changed by default.
- Verification:
  - `py_compile` passed for the two modified files.
  - `black` was not available in `/opt/conda310` (`No module named black`), so
    formatting was checked manually for lines longer than 120 chars; none found
    in the modified files.
  - Ran a focused CUDA smoke script on GPU0 for
    `silu_mul_fp8_quant_packed` with `bf16_activation=False/True` and
    `clamp_limit=0/7`.
  - For all four cases, FP8 output bytes and packed UE8M0 scales matched the
    corresponding PyTorch reference exactly:
    - `q_equal=True`.
    - `scale_equal=True`.
    - `q_diff_bytes=0`.
    - `scale_diff=0`.
- Next:
  - Restart only tmux session `dsv4_flashinfer_18980`.
  - Rerun the 100-token teacher-forced dump at `pos19664`.
  - Compare `L00_moe_shared_hidden/out` against vLLM, then check whether the
    natural top1 mismatch at `oracle_idx=78` is removed.

### 2026-05-20 01:00 CST - pos19664 L0 Shared Expert Diff Eliminated

- Service action:
  - Restarted only tmux session `dsv4_flashinfer_18980`.
  - vLLM oracle `18000`, prefill `18800`, and baseline decode `18880` were not touched.
  - Health passed: `http://127.0.0.1:18980/health` returned `"ok"`.
- Validation run:
  - Ran 100-token teacher-forced record 89 against RTP `18980/18981`.
  - Output directory:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/rtp_teacher_forced_record89_len100_20260520_004141_record89_20260520_004155`.
  - `compare_len=100`, `first_diff=null`, `equal_prefix=true`.
  - RTP/vLLM hashes both `b062a9477fc5f8d1`.
- New RTP dump:
  - `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010/rtp_pos19664/rank0_pid3781129_step000.pt`.
  - Logits dump:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010/rtp_pos19664/rank0_pid3781129_step001.pt`.
- Comparison against vLLM dump at `positions=[19664]`:
  - `L00_moe_x_in`: exact equal.
  - `L00_moe_topk_indices`: exact equal `[155, 110, 87, 23, 11, 40]`.
  - `L00_moe_topk_weights`: unchanged tiny diff, max `1.527369e-07`.
  - `L00_moe_routed_x_quant`: exact equal.
  - `L00_moe_routed_x_scale`: exact equal.
  - `L00_moe_routed_packed_topk`: exact equal.
  - `L00_moe_routed_trtllm_out`: exact equal.
  - `L00_moe_routed_y`: exact equal.
  - `L00_moe_shared_gate_up`: exact equal.
  - `L00_moe_shared_hidden`: exact equal, max diff `0`, nonzero `0 / 2048`.
  - `L00_moe_shared_out`: exact equal, max diff `0`, nonzero `0 / 4096`.
  - `L00_moe_shared_y`: exact equal, max diff `0`, nonzero `0 / 4096`.
  - `L00_ffn_out` vs RTP `L00_decode_ffn_out`: exact equal, max diff `0`, nonzero `0 / 4096`.
  - RTP `L00_moe_shared_hidden` also exactly matches BF16-input reference
    `(F.silu(gate_bf16) * up_bf16).to(bf16)`.
- Conclusion:
  - The previously identified first L0 shared expert divergence at
    `pos19664` is fixed.
  - This does not yet prove 1000-token natural greedy alignment; it only proves
    the first known tensor divergence at the first natural top1 mismatch has
    been eliminated.
- Next:
  - Inspect the latest sampler dump/logits at `pos19664` to verify whether RTP
    natural top1 before teacher override changed from `3003` to `3975`.
  - If fixed, run longer natural or sampler validation to find the next first
    mismatch, continuing layer-by-layer only if needed.

### 2026-05-20 01:05 CST - First Natural Top1 Mismatch Removed In Latest Sampler Segment

- Sampler dump checked:
  - `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_teacher_forced_len130_sampler_logits_flashinfer_trtllm_18980_20260519_2324.jsonl`.
- Latest appended segment after the BF16 shared activation patch:
  - Segment length: `99` sampled entries.
  - Oracle index range: `2..100`.
  - Natural top1 mismatch count before teacher override: `0`.
- Previously failing point:
  - `oracle_idx=78`, `step=19664`, teacher/vLLM token `3975`.
  - Before patch: RTP natural top1 was `3003`.
  - After patch: RTP natural top1 is `3975`.
  - Top candidates after patch:
    `[3975, 3003, 6451, 17839, 27620, 108889, 389, 78457]`.
  - Top logits after patch begin:
    `[29.7452, 29.6229, 29.6161, 29.5596]`.
- Conclusion:
  - The first known natural top1 mismatch at `pos19664` is removed.
  - Need extend validation to 1000 generated tokens to prove the user target.
- Next:
  - Run 1000-token teacher-forced sampler validation and count natural top1
    mismatches before teacher override.
  - If mismatch count is zero, run/compare natural 1000-token generation.
  - If a later mismatch appears, dump the first later mismatch and continue the
    same tensor-level RCA.

### 2026-05-20 01:10 CST - 1000 Teacher-Forced IDs Still Match; Next Natural Top1 Mismatch At pos19690

- Experiment:
  - Ran 1000-token teacher-forced record 89 against patched RTP `18980/18981`.
- Token result:
  - Output directory:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/rtp_teacher_forced_record89_len1000_20260520_004543_record89_20260520_004555`.
  - `compare_len=1000`.
  - `first_diff=null`.
  - `equal_prefix=true`.
  - RTP/vLLM hashes both `986b77c92c844fc6`.
- Sampler natural-top1 result before teacher override:
  - Summary saved to:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/rtp_teacher_forced_record89_len1000_20260520_004543_record89_20260520_004555/sampler_top1_summary.json`.
  - Latest sampler segment length: `999` entries.
  - Oracle index range: `2..1000`.
  - Natural top1 mismatch count: `32`.
  - First remaining mismatch:
    - `oracle_idx=104`.
    - `step=19690`.
    - vLLM/teacher token: `478`.
    - RTP natural top1 before teacher override: `303`.
    - Top candidates: `[303, 320, 478, 1237, 25728, 876, 1530, 4572]`.
    - Top logits: `[35.7293, 34.4886, 34.3503, 25.6628, 23.4842, 22.6578, 22.5929, 22.402]`.
- Conclusion:
  - The first old mismatch at `pos19664` is fixed.
  - The full 1000 teacher-forced stream remains token-equal, but natural greedy
    1000-token alignment is not finished.
  - The next RCA target is now `pos19690 / oracle_idx=104`.
- Next:
  - Retarget dump instrumentation from `19664` to `19690`.
  - Restart only `dsv4_flashinfer_18980` and rerun enough teacher-forced tokens
    to reach `oracle_idx=104`.
  - Compare RTP tensors at `pos19690` against the vLLM oracle dump for the same
    position, then fix only the first confirmed tensor divergence.

### 2026-05-20 01:14 CST - Retargeted Dump To Next Mismatch pos19690

- vLLM oracle dump found for next mismatch:
  - `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l0_internal_pos19689_20260517_232041/rank0_pid2277834_step108.pt`.
  - It contains `positions=[19690]`, `input_ids=[478]`.
- Launcher changed:
  - `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode_flashinfer_trtllm_18980.sh`.
- Dump target changed:
  - `MOEDBG_CASE`: `rtp_pos19664` -> `rtp_pos19690`.
  - `MOEDBG_GLOBAL_POS`: `19664` -> `19690`.
  - `DSV4_DECODE_DUMP_CASE`: `rtp_pos19664` -> `rtp_pos19690`.
  - `DSV4_DECODE_DUMP_POS`: `19664` -> `19690`.
  - `DSV4_META_DUMP_CASE`: `rtp_pos19664` -> `rtp_pos19690`.
  - `DSV4_META_DUMP_POS`: `19664` -> `19690`.
- Precision-affecting env left unchanged:
  - `DSV4_GATE_FP32=1`.
  - `DSV4_INDEXER_TOPK_BACKEND=torch`.
  - `DSV4_MOE_STRATEGY=flashinfer_trtllm_fp4`.
  - `DSV4_USE_FLASHINFER_TRTLLM_FP4=1`.
  - No decode FlashMLA `topk_length` path.
- Next:
  - Restart only `dsv4_flashinfer_18980`, run a short teacher-forced request long
    enough to include `oracle_idx=104`, and compare the new RTP `pos19690` dump
    against vLLM `step108`.

### 2026-05-20 01:18 CST - pos19690 Consumed-Token L0 Is Aligned; Mismatch Mapping Needs Previous Position

- Generated new RTP dump after retargeting to `pos19690`:
  - `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010/rtp_pos19690/rank0_pid3787733_step000.pt`.
  - Logits dump:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010/rtp_pos19690/rank0_pid3787733_step001.pt`.
- vLLM oracle used:
  - `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l0_internal_pos19689_20260517_232041/rank0_pid2277834_step108.pt`.
  - Contains `positions=[19690]`, `input_ids=[478]`.
- L0 consumed-token comparison at `positions=[19690]`:
  - `embed_hc_expanded`: exact equal.
  - `L00_attn_out`: exact equal.
  - `L00_ffn_in` / RTP `L00_decode_ffn_in`: exact equal.
  - `L00_moe_x_in`: exact equal.
  - `L00_moe_topk_indices`: exact equal.
  - `L00_moe_topk_weights`: tiny diff max `5.960464e-07`.
  - `L00_moe_routed_x_quant`: exact equal.
  - `L00_moe_routed_x_scale`: exact equal.
  - `L00_moe_routed_packed_topk`: exact equal.
  - `L00_moe_routed_trtllm_out`: exact equal.
  - `L00_moe_routed_y`: exact equal.
  - `L00_moe_shared_gate_up`: exact equal.
  - `L00_moe_shared_hidden`: exact equal.
  - `L00_moe_shared_out`: exact equal.
  - `L00_moe_shared_y`: exact equal.
  - `L00_ffn_out` / RTP `L00_decode_ffn_out`: exact equal.
- Important mapping correction:
  - The logits dump for RTP `positions=[19690], input_ids=[478]` has top1
    `5640`, which is the next vLLM token after `478`.
  - Therefore sampler mismatch `step=19690, teacher_token=478, top1=303` is the
    prediction of token `478`, produced by the previous decode input, not by the
    `positions=[19690]` consumed-token forward itself.
  - Next dump target should be checked as previous position `19689` with
    `input_ids=[13660]` before pursuing deeper layer diffs.
- Next:
  - Compare/dump `positions=[19689]`, because that forward's logits predict the
    `step=19690` teacher token `478`.

### 2026-05-20 01:21 CST - Retargeted Dump To Predictor Position pos19689

- Reason:
  - `step=19690 / teacher_token=478` is predicted from the previous decode
    forward, so the next useful tensor dump is `positions=[19689]`,
    `input_ids=[13660]`.
- Launcher changed:
  - `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode_flashinfer_trtllm_18980.sh`.
- Dump target changed:
  - `MOEDBG_CASE`: `rtp_pos19690` -> `rtp_pos19689`.
  - `MOEDBG_GLOBAL_POS`: `19690` -> `19689`.
  - `DSV4_DECODE_DUMP_CASE`: `rtp_pos19690` -> `rtp_pos19689`.
  - `DSV4_DECODE_DUMP_POS`: `19690` -> `19689`.
  - `DSV4_META_DUMP_CASE`: `rtp_pos19690` -> `rtp_pos19689`.
  - `DSV4_META_DUMP_POS`: `19690` -> `19689`.
- vLLM oracle target:
  - `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l0_internal_pos19689_20260517_232041/rank0_pid2277834_step107.pt`.
  - Contains `positions=[19689]`, `input_ids=[13660]`.

### 2026-05-20 01:25 CST - pos19689 Reproduces Next Logit Divergence, L0 Is Fully Aligned

- Generated RTP predictor-position dump:
  - `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010/rtp_pos19689/rank0_pid3792789_step000.pt`.
  - Logits dump:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010/rtp_pos19689/rank0_pid3792789_step001.pt`.
- vLLM oracle used:
  - `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l0_internal_pos19689_20260517_232041/rank0_pid2277834_step107.pt`.
  - Contains `positions=[19689]`, `input_ids=[13660]`.
- RTP logits at this predictor step:
  - Top candidates: `[303, 320, 478, 1237, 25728, 876, 1530, 4572, 14, 3975, 5640, 87208, 7831, 115725, 119316, 16734]`.
  - Top logits: `[35.75, 34.5, 34.25, 25.625, 23.5, 22.625, 22.625, 22.375, 21.25, 20.75, 20.125, 20.125, 20.125, 19.875, 19.75, 19.625]`.
  - This reproduces the next natural top1 mismatch: vLLM/teacher next token is
    `478`, but RTP natural top1 is `303`.
- L0 tensor comparison at `positions=[19689]`:
  - `embed_hc_expanded`: exact equal.
  - `L00_attn_out`: exact equal.
  - `L00_ffn_in` / RTP `L00_decode_ffn_in`: exact equal.
  - `L00_moe_x_in`: exact equal.
  - `L00_moe_router_logits`: tiny diff max `5.245209e-06`.
  - `L00_moe_topk_indices`: exact equal.
  - `L00_moe_topk_weights`: tiny diff max `5.960464e-07`.
  - `L00_moe_routed_x_quant`: exact equal.
  - `L00_moe_routed_x_scale`: exact equal.
  - `L00_moe_routed_packed_topk`: exact equal.
  - `L00_moe_routed_trtllm_out`: exact equal.
  - `L00_moe_routed_y`: exact equal.
  - `L00_moe_shared_gate_up`: exact equal.
  - `L00_moe_shared_hidden`: exact equal.
  - `L00_moe_shared_out`: exact equal.
  - `L00_moe_shared_y`: exact equal.
  - `L00_ffn_out` / RTP `L00_decode_ffn_out`: exact equal.
- Conclusion:
  - The remaining mismatch is not in L0.
  - Current vLLM/RTP dumps only contain layer 0 tensors, so the next step must
    expand instrumentation to later layer boundaries and binary-search the first
    layer where RTP diverges.
- Next:
  - Inspect record/dump controls and add or enable layer-boundary dumps beyond L0.
  - Prefer dumping compact layer outputs for all layers at `pos19689` first, then
    add internals only for the first divergent layer.

### 2026-05-20 01:31 CST - Layer Boundary Bisection Finds First Divergence At L1

- Compared current RTP `pos19689` dump against vLLM full layer-boundary oracle:
  - vLLM:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_pd_freerun_layers_20260517_1945/rank0_pid1543390_step107.pt`.
  - RTP:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010/rtp_pos19689/rank0_pid3792789_step000.pt`.
  - Summary saved:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/rtp_teacher_forced_record89_len110_20260520_005710_record89_20260520_005721/layer_boundary_compare_pos19689.json`.
- Result:
  - `embed_out`: exact equal.
  - `embed_hc_expanded`: exact equal.
  - `layer00_out`: exact equal.
  - First nonzero layer boundary diff: `layer01_out`.
    - max diff `0.0244140625`.
    - mean diff `0.0017104198`.
    - rms `0.0032757837`.
    - nonzero `10814 / 16384`.
- Conclusion:
  - The remaining `pos19689` logits mismatch is caused first inside layer 1.
  - Next RCA should dump L1 internals only, not all layers.

### 2026-05-20 01:33 CST - Retargeted MOEDBG Internals To Layer 1

- Launcher changed:
  - `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode_flashinfer_trtllm_18980.sh`.
- Debug filter changed:
  - `MOEDBG_LAYER`: `0` -> `1`.
  - `MOEDBG_NAME_REGEX`: `^(decode_|L00_)` -> `^(decode_|L01_)`.
- Dump position remains:
  - `MOEDBG_GLOBAL_POS=19689`.
  - `DSV4_DECODE_DUMP_POS=19689`.
- Purpose:
  - Dump L1 internals for the first divergent layer boundary at `pos19689`.

### 2026-05-20 01:10 CST - L1 Fine Oracle Validity Check Caveat

- Current RCA target remains the predictor forward:
  - `positions=[19689]`, `input_ids=[13660]`.
  - This forward predicts the teacher/vLLM next token `478`; RTP natural top1 is
    still `303`.
- Do not use the old vLLM fine dump
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l1fine_chunked_len6/rank0_pid146744_step213.pt`
  as a direct L1 internal oracle for current RTP.
- Reason:
  - Comparing it against the current RTP L1 dump shows `layer00_out` is already
    different, while the full layer-boundary oracle
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_pd_freerun_layers_20260517_1945/rank0_pid1543390_step107.pt`
    has `layer00_out` exact equal to current RTP.
  - Therefore `vllm_l1fine_chunked_len6` likely came from a different/older dump
    configuration and is not a valid tensor-level oracle for this step.
- Next:
  - Search existing vLLM dumps for a `positions=[19689]`, `input_ids=[13660]`
    L1-internal oracle whose `layer00_out` is exact equal to current RTP.
  - If none exists, generate a fresh vLLM L1-internal oracle without killing the
    running vLLM oracle service on `127.0.0.1:18000`.

### 2026-05-20 01:10 CST - Existing L1 Oracle Candidate Filter

- Tooling note:
  - Two first filter attempts failed because the helper script used Python
    boolean `or` with tensors. This was a script bug only; it did not produce a
    model-diff conclusion.
  - Fixed the helper to use explicit `is None` tensor checks.
- Filter input:
  - RTP reference:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010/rtp_pos19689/rank0_pid3797833_step000.pt`.
  - Required metadata: `positions=[19689]`, `input_ids=[13660]`.
  - Validity gate: candidate vLLM `layer00_out` must be exact equal to RTP
    `decode_layer00_out`.
- Candidate directory results:
  - `vllm_l1fine_ref_len6`: no matching `positions/input_ids`.
  - `vllm_l1fine_swaidx_ref_len6`: no matching `positions/input_ids`.
  - `vllm_l1_swa_cachecmp_len6`: no matching `positions/input_ids`.
  - `vllm_l1_swa_dequantcmp_len6`: no matching `positions/input_ids`.
  - `vllm_block012_pos19689_20260517_231729`:
    - one metadata match:
      `rank0_pid2271056_step107.pt`.
    - `layer00_out` exact equal to current RTP.
    - only has `layer00_out` and `layer01_out`, so it is a valid boundary oracle
      but not enough for L1 internal RCA.
  - `vllm_indexer_layers_pos19689_20260517_230819`:
    - one metadata match, but no `layer00_out`/L1 internal tensors.
  - `vllm_indexer_layers_2_10_pos19689_20260517_231136`:
    - one metadata match, but no `layer00_out`/L1 internal tensors.
  - `vllm_l1fine_chunked_len6`:
    - three metadata matches.
    - all have L1 internals, but `layer00_out` is not exact equal:
      max diff `0.02734375`, mean diff `0.0013415962`.
    - This confirms it must not be used as the oracle for current RTP.
- Conclusion:
  - Existing filtered candidates still do not provide a valid L1-internal oracle.
  - Need a broader metadata/hash scan or a fresh vLLM L1-internal dump.

### 2026-05-20 01:10 CST - Started Temporary vLLM L1 Dump Service

- Added temporary launch script outside the repo production path:
  `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_vllm_l1dump_18100.sh`.
- Purpose:
  - Generate a valid vLLM L1-internal oracle for `positions=[19689]`,
    `input_ids=[13660]` without touching the stable oracle on
    `127.0.0.1:18000` / GPU4.
- Temporary service config:
  - `CUDA_VISIBLE_DEVICES=1`.
  - Port `18100`.
  - `VLLM_DSV4_DUMP=1`.
  - `VLLM_DSV4_DUMP_CASE=vllm_l1_internal_pos19689_20260520_0110`.
  - `VLLM_DSV4_DUMP_MAX_SEQ=1`, to skip the long prefill and dump decode
    single-token forwards only.
  - `VLLM_DSV4_DUMP_LAYER=1`.
  - `VLLM_DSV4_DUMP_NAME_REGEX='^(input_ids|positions|embed_out|embed_hc_expanded|layer00_out|L01_|layer01_out|lm_top_values|lm_top_indices)'`.
- Started tmux session:
  `dsv4_vllm_l1dump_18100`.
- Next:
  - Wait for `/health` on `18100`, run a short vLLM-only request long enough to
    reach `positions=[19689]`, then filter for a dump whose `layer00_out` is
    exact equal to current RTP.

### 2026-05-20 01:19 CST - Fixed Temporary vLLM Launch Args

- First temporary `18100` launch failed during model init.
- Error evidence from `/tmp/vllm_l1dump_18100.log`:
  - `AssertionError: DeepseekV4 only supports fp8 kv-cache format for now, got auto`.
- Root cause:
  - The temporary launcher did not include the stable oracle's runtime flags,
    especially `--kv-cache-dtype fp8`.
- Updated only the temporary script
  `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_vllm_l1dump_18100.sh`:
  - Added `--tensor-parallel-size 1`.
  - Added `--dtype bfloat16`.
  - Added `--kv-cache-dtype fp8`.
  - Added `--max-num-seqs 1`.
  - Added `--gpu-memory-utilization 0.92`.
  - Added `--trust-remote-code`.
  - Added `--no-enable-flashinfer-autotune`.
- This is a temporary oracle launcher change only; production RTP code is not
  changed.

### 2026-05-20 01:20 CST - Temporary vLLM Service Ready; Corrected Request Runner Path

- Temporary vLLM L1 dump service `18100` passed `/health`.
- First request command failed before sending the request because the assumed
  runner path did not exist:
  `/data3/dsv4_repeat_compare/scripts/repeat_stability_from_q_ignore_eos.py`.
- Existing runner paths found:
  - `/tmp/repeat_stability_from_q_ignore_eos.py`.
  - `/data3/dsv4_repeat_compare/scripts/repeat_stability_from_q.py`.
- Next request will use `/tmp/repeat_stability_from_q_ignore_eos.py`.

### 2026-05-20 01:24 CST - Valid vLLM L1 Oracle Found; Compare Script Needs Float8-Safe Equality

- Ran vLLM-only request against temporary service `18100` with L1 dump enabled.
- Valid oracle found:
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l1_internal_pos19689_20260520_0110/rank0_pid3808909_step103.pt`.
- Oracle validity checks:
  - `positions=[19689]`.
  - `input_ids=[13660]`.
  - Has `90` tensors and L1 internal keys.
  - `layer00_out` exact equal to current RTP `decode_layer00_out`.
  - vLLM top candidates begin `[478, 303, 320, 1237, 25728, 876, 1530, 4572]`.
  - vLLM top values begin `[36.0, 35.75, 35.0, 25.625, 23.875, 23.375, 23.0, 22.75]`.
- First L1 compare attempt failed in the helper script, not in model inference:
  - PyTorch cannot promote mixed `float8_e4m3fn` and `bfloat16` inside
    `torch.equal`.
- Next:
  - Re-run the L1 internal comparison with float8-safe equality that casts both
    sides to float for numeric comparison before equality/diff calculation.

### 2026-05-20 01:25 CST - Valid L1 Internal Compare Narrows First Meaningful Diff To SWA Cache Read

- Compared valid vLLM L1 oracle:
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l1_internal_pos19689_20260520_0110/rank0_pid3808909_step103.pt`
  against current RTP L1 dump:
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_forced_oracle78_l0_20260520_0010/rtp_pos19689/rank0_pid3797833_step000.pt`.
- Full compare summary saved:
  `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/l1_internal_compare_valid_oracle_pos19689_20260520_0124.json`.
- Exact-equal before first meaningful diff:
  - `embed_out` / `decode_embed_out`.
  - `embed_hc_expanded` / `decode_embed_hc_expanded`.
  - `layer00_out` / `decode_layer00_out`.
  - `L01_attn_in` / `L01_decode_attn_in`.
  - `L01_attn_qr_norm` / `L01_fp8_decode_qr_norm`.
  - `L01_attn_q_linear` / `L01_fp8_decode_q_linear`.
  - `L01_attn_kv_norm` / `L01_fp8_decode_kv_norm_ref`.
  - `L01_attn_q_pre_mla` / `L01_fp8_decode_q`.
- First meaningful non-exact region:
  - `L01_attn_decode_swa_indices` vs `L01_fp8_decode_swa_topk` differ, but this
    is not directly conclusive because vLLM appears to record absolute/global
    indices while RTP records local window indices.
  - `L01_attn_decode_swa_selected_cache_logical` vs
    `L01_fp8_decode_swa_selected_cache_logical` differs in `8106 / 74752` bytes.
  - `L01_attn_decode_swa_selected_k_dequant` vs
    `L01_fp8_decode_swa_selected_k_dequant` differs with max `0.25`, mean
    `0.0033855545`, rms `0.0126710534`.
  - `L01_attn_mla_out_heads` vs `L01_fp8_decode_o_heads` differs with max
    `0.0625`, mean `0.0003146937`.
  - `L01_attn_out` vs `L01_decode_attn_out` differs with max `0.09375`, mean
    `0.0204470865`.
- Conclusion:
  - L1 input, Q projection, KV projection/norm, and q before MLA are aligned.
  - Remaining first meaningful divergence is in decode SWA selected cache / K
    dequant / attention read path, not router/top-k or shared expert.
- Next:
  - Inspect RTP and vLLM decode SWA cache selection semantics and compare whether
    the same logical tokens are being read after translating global vs local
    index representation.

### 2026-05-20 01:40 CST - L1 SWA Index Diff Is Physical-Block Mapping; Cache Byte Diff Points Back To Prefill/Write

- Code inspection:
  - vLLM `_forward_decode` passes `indices=swa_indices` and
    `topk_length=swa_lens` to `flash_mla_with_kvcache`.
  - RTP FP8 decode wrapper currently passes `topk_length=None`.
  - For the current failing sample this is not the first-order cause because
    both sides have a full SWA window (`swa_lens=[128]`) and no negative/padded
    SWA entries.
- Index evidence at predictor forward `positions=[19689]`:
  - vLLM `L01_attn_decode_swa_indices` starts at
    `437418..437503`, then `515904..515945`.
  - RTP `L01_fp8_decode_swa_topk` is `618..745`.
  - The difference is a physical block-id offset, not a semantic local-window
    difference: first 86 entries differ by `436800`, last 42 differ by
    `515200`.
- Cache-byte evidence:
  - Comparing selected logical cache rows, only `34 / 128` SWA rows differ;
    `94 / 128` rows are byte-exact even though the physical slot ids differ.
  - Diff rows map to absolute positions:
    `[19562..19586, 19588, 19590, 19597, 19606, 19607, 19647, 19648, 19659, 19676]`.
  - The first `25` diff rows are exactly the prompt tail positions before the
    first generated token (`prompt_len=19587`), so the next RCA target is L1
    prefill/write of SWA FP8 cache bytes, not decode FlashMLA indexing.
- Conclusion:
  - Do not change decode FlashMLA `topk_length` yet.
  - Continue by comparing RTP/vLLM L1 prefill SWA KV quant/insert semantics for
    the prompt tail, then the few decode-written rows that still differ.

### 2026-05-20 02:05 CST - Added Temporary L1 Prefill SWA Write Dumps

- Purpose:
  - Directly answer whether the remaining L1 cache byte diff is created during
    prefill SWA cache write, before changing any decode attention logic.
- Temporary vLLM instrumentation:
  - File:
    `/data3/vllm-dsv4-env/lib/python3.10/site-packages/vllm/model_executor/layers/deepseek_v4_attention.py`.
  - Added env-gated records after
    `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`:
    - `L01_attn_swa_write_slot_mapping`.
    - `L01_attn_swa_written_cache_logical`.
  - This is dump-only and does not alter computation inputs/outputs.
- Temporary vLLM launcher change:
  - File:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_vllm_l1dump_18100.sh`.
  - Changed case to `vllm_l1_prefill_swa_write_record89_20260520_0205`.
  - Changed `VLLM_DSV4_DUMP_MAX_SEQ=20000` so the long record-89 prefill is
    dumped.
  - Narrowed regex to `input_ids`, `positions`, `L01_attn_kv_norm`,
    `L01_attn_swa_write_slot_mapping`, and
    `L01_attn_swa_written_cache_logical`.
- Temporary RTP prefill launcher change:
  - File:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_prefill.sh`.
  - Enabled `MOEDBG=2`, `MOEDBG_LAYER=1`.
  - Dump case:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_prefill_l1_swa_write_record89_20260520_0205/prefill_record89`.
  - Regex narrowed to:
    - `L01_fp8_prefill_swa_write_slot_mapping`.
    - `L01_fp8_prefill_swa_written_cache_logical`.
- Next:
  - Restart only temporary vLLM `18100` and RTP prefill `18800`.
  - Run record-89 request once.
  - Compare vLLM/RTP L1 prefill written logical cache rows for the same
    absolute prompt-tail positions found in decode:
    `[19562..19586, 19588, 19590, 19597, 19606, 19607, 19647, 19648, 19659, 19676]`.

### 2026-05-20 02:18 CST - First Prefill Write Compare Requires Write-Input Dump

- vLLM prefill write oracle generated from temporary `18100`:
  - Dump case:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l1_prefill_swa_write_record89_20260520_0205`.
  - Three chunks:
    - `step002`: positions `0..8191`.
    - `step003`: positions `8192..16383`.
    - `step004`: positions `16384..19586`.
- RTP prefill write dump generated from restarted `18800`:
  - Dump case:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_prefill_l1_swa_write_record89_20260520_0205/prefill_record89`.
  - CP ranks:
    - `rank0_pid3831255_step015.pt`.
    - `rank1_pid3831256_step015.pt`.
  - RTP rank0/rank1 `L01_fp8_prefill_swa_written_cache_logical` are
    byte-exact equal, so this is not rank-local nondeterminism.
- Important caveat:
  - RTP SWA prefill write persists only tail slots where `slot_mapping >= 0`
    (`387 / 19587` rows, positions `19200..19586`).
  - vLLM records all positions as valid physical slots.
  - Full-row compare is therefore invalid for rows where RTP slot is `-1`.
- Valid-tail compare:
  - For RTP valid rows `19200..19586`, current RTP written logical cache differs
    from vLLM on all `387 / 387` rows.
  - Prompt-tail rows `19562..19586` also all differ.
  - Summary file:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/l1_prefill_swa_write_compare_20260520_0205.json`.
- Caveat on old decode dump:
  - Comparing the new RTP prefill dump to the old RTP decode-selected dump is
    not a valid proof because the old decode dump was captured before restarting
    prefill and may refer to an older cache allocation/service state.
- New minimal instrumentation:
  - File:
    `rtp_llm/models_py/modules/dsv4/fp8/attention.py`.
  - Added dump-only records for L1 prefill write input:
    - `L01_fp8_prefill_kv_cache_norm_pre_rope_full`.
    - `L01_fp8_prefill_kv_cache_rope_full`.
  - Launcher regex in
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_prefill.sh`
    now includes these two keys.
- Next:
  - Restart only RTP prefill `18800`.
  - Re-run record-89 one-token request.
  - Compare RTP `kv_cache_norm_pre_rope_full` against vLLM `L01_attn_kv_norm`.
  - If pre-RoPE KV is exact, root cause is KV RoPE or quant/insert.
  - If pre-RoPE KV differs, root cause is earlier L1 KV projection/RMSNorm in
    prefill path.

### 2026-05-20 02:25 CST - Root Cause Moved Before Prefill SWA Quant/Insert

- Re-ran record-89 one-token RTP request after restarting prefill `18800` with
  the new write-input dumps.
- New RTP dump:
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_prefill_l1_swa_write_record89_20260520_0205/prefill_record89`.
  - `rank0_pid3841236_step015.pt`.
  - `rank1_pid3841237_step015.pt`.
- New comparison summary:
  `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/l1_prefill_swa_write_input_compare_20260520_0220.json`.
- Rank consistency:
  - RTP rank0/rank1 `L01_fp8_prefill_kv_cache_norm_pre_rope_full` are
    byte-exact equal.
  - RTP rank0/rank1 `L01_fp8_prefill_kv_cache_rope_full` are byte-exact equal.
  - So this is not CP rank nondeterminism.
- Key evidence:
  - Comparing RTP pre-RoPE write input
    `L01_fp8_prefill_kv_cache_norm_pre_rope_full` against vLLM
    `L01_attn_kv_norm`:
    - Full valid RTP tail rows `19200..19586`: max `0.1875`, mean
      `0.0142762`, rms `0.0185590`, nonzero `183990`.
    - Decode prompt-tail rows `19562..19586`: max `0.1875`, mean
      `0.0145098`, rms `0.0191255`, nonzero `11857`.
  - RTP RoPE-after tensor differs much more from vLLM pre-RoPE, as expected;
    RoPE is not yet the first comparison target.
- Conclusion:
  - The remaining L1 SWA cache byte divergence is created before SWA
    quant/insert.
  - Current root cause target is RTP prefill L1 KV projection/RMSNorm path:
    `L01_attn_in -> L01_attn_qr_kv -> L01_attn_kv_norm`.
  - Do not change decode FlashMLA `topk_length` or SWA quant/insert yet.
- Next:
  - Dump only prompt-tail prefill tensors for both sides:
    - `L01_attn_in`.
    - `L01_attn_qr_kv`.
    - `L01_attn_kv_norm`.
  - If `L01_attn_in` differs, chase layer0 prefill boundary / CP gather order.
  - If `L01_attn_in` matches but `qr_kv` differs, compare RTP linear
    implementation/weight packing against vLLM fused `wqa_wkv`.
  - If `qr_kv` matches but `kv_norm` differs, compare RMSNorm semantics.

### 2026-05-20 02:30 CST - Added Tail-Only Prefill QKV Dumps

- Purpose:
  - Avoid full prefill tensor dumps while narrowing the first prefill-side diff
    to `attn_in`, raw `qr_kv`, or RMSNorm.
- RTP instrumentation:
  - File:
    `rtp_llm/models_py/modules/dsv4/fp8/attention.py`.
  - Added dump-only records in `_prefill_compute_qkv`:
    - `L01_fp8_prefill_attn_in`.
    - `L01_fp8_prefill_attn_qr_kv` = concat of RTP raw `wq_a(x)` and `wkv(x)`.
    - `L01_fp8_prefill_attn_qr_norm`.
    - `L01_fp8_prefill_attn_kv_linear`.
    - `L01_fp8_prefill_attn_kv_norm`.
  - These are under `_record_tensor.should_record_layer`, so they are inactive
    outside `MOEDBG`.
- RTP launcher change:
  - File:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_prefill.sh`.
  - Added `MOEDBG_TAIL_TOKENS=512`.
  - Expanded regex to include `L01_attn_in` and the new
    `L01_fp8_prefill_attn_*` keys.
- vLLM temporary launcher change:
  - File:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_vllm_l1dump_18100.sh`.
  - Changed case to `vllm_l1_prefill_tail_qkv_record89_20260520_0230`.
  - Added `VLLM_DSV4_DUMP_TAIL_TOKENS=512`.
  - Expanded regex to include:
    - `L01_attn_in`.
    - `L01_attn_qr_kv`.
    - `L01_attn_qr_norm`.
    - `L01_attn_kv_norm`.
- Next:
  - Restart temporary vLLM `18100` and RTP prefill `18800`.
  - Run record-89 one-token probes on both.
  - Compare the last 512 prompt tokens, then map tail index back to absolute
    positions `19075..19586`.

### 2026-05-20 02:42 CST - Tail QKV Compare Shows First Diff Before L1 QKV

- Ran temporary vLLM `18100` one-token probe:
  - Output run:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/vllm_l1_prefill_tail_qkv_probe_record89_20260520_021202`.
  - Dump case:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l1_prefill_tail_qkv_record89_20260520_0230`.
  - Last chunk `step004` covers positions `19075..19586`.
- Ran RTP one-token probe:
  - Output run:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/rtp_prefill_l1_tail_qkv_probe_record89_20260520_021303`.
  - Dump case:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_prefill_l1_swa_write_record89_20260520_0205/prefill_record89`.
- Compare summary:
  `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/l1_prefill_tail_qkv_compare_20260520_0230.json`.
- Key evidence on last 512 prompt tokens (`19075..19586`):
  - RTP `L01_attn_in` equals RTP `L01_fp8_prefill_attn_in` exactly
    (`512 / 512` rows exact), so RTP's local record sites agree.
  - RTP `L01_attn_in` vs vLLM `L01_attn_in`:
    - `0 / 512` rows exact.
    - max `0.27685546875`, mean `0.0375210`, rms `0.0472700`.
  - Downstream diffs are expected consequences:
    - `qr_kv`: max `1.4907227`, mean `0.0818992`.
    - `kv_norm`: max `6.1875`, mean `0.6386065`.
- Conclusion:
  - Current first proven prefill-side diff is before L1 QKV, at or before
    `L01_attn_in`.
  - Do not change L1 QKV/GEMM, RMSNorm, SWA quant/insert, or decode
    FlashMLA yet.
- Next:
  - Dump tail `layer00_out`, `L00_attn_out`, `L00_ffn_out`,
    `L01_attn_hc_pre`/`L01_attn_in` on both sides.
  - Determine whether layer0 output itself differs, or whether the
    hidden-compression/residual preparation between layer0 and L1 differs.

### 2026-05-20 02:47 CST - Expanded Tail Dump To L0/L1 Boundary

- Temporary RTP launcher update:
  - File:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_prefill.sh`.
  - Changed `MOEDBG_LAYER=0,1`.
  - Regex now includes:
    - `layer00_out`, `prefill_layer00_out`, all `L00_`.
    - `L01_attn_hc_pre`, `L01_attn_in`.
    - Existing L1 QKV/cache dump keys.
- Temporary vLLM launcher update:
  - File:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_vllm_l1dump_18100.sh`.
  - Changed case to `vllm_l0_l1_prefill_tail_record89_20260520_0245`.
  - Changed `VLLM_DSV4_DUMP_LAYER=0,1`.
  - Regex now includes:
    - `layer00_out`, all `L00_`.
    - `L01_attn_hc_pre`, `L01_attn_in`.
    - Existing L1 QKV/cache dump keys.
- Next:
  - Restart temporary vLLM `18100` and RTP prefill `18800`.
  - Run one-token probes.
  - Compare tail `layer00_out`, then L1 HC pre/norm input.

### 2026-05-20 02:55 CST - CP-Aligned L0/L1 Tail Compare Corrects The RCA

- Important correction:
  - Previous tail-only comparisons of rank-local RTP tensors were misleading
    because RTP prefill runs CP=2.
  - `Lxx_*` block tensors are rank-local; they must be mapped by
    `extra.global_positions` and filtered by `extra.local_is_real` before
    comparing to vLLM's contiguous global positions.
- CP-aligned comparison summary:
  `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/l0_l1_prefill_tail_compare_cp_aligned_20260520_0255.json`.
- Valid aligned range:
  - vLLM last prefill chunk covers positions `19075..19586`.
  - RTP rank0 tail covers `19076..19586` with `511 / 512` real rows.
  - RTP rank1 tail covers `14179..14690`, so it is outside this vLLM tail
    compare window.
- Key evidence:
  - `L00_attn_in` is bit-exact on aligned positions `19076..19586`:
    - `511 / 511` rows exact.
  - First clear diff is `L00_attn_out`:
    - `0 / 511` rows exact.
    - max `0.40625`, mean `0.0463197`, rms `0.0588693`.
  - Downstream propagation:
    - `L00_attn_residual`: max `0.0210571`, mean `0.0003626`.
    - `L00_ffn_in`: max `0.1171875`, mean `0.0070060`.
    - `L00_ffn_out`: max `0.1640625`, mean `0.0118409`.
    - `layer00_out`: max `0.1171875`, mean `0.0010707`.
    - `L01_attn_in`: max `0.0126953`, mean `0.0008164`.
- Corrected conclusion:
  - Current first proven prefill-side divergence is inside L0 attention
    prefill output, after `L00_attn_in`.
  - Do not chase L1 QKV, layer0 FFN, SWA quant/insert, or decode
    FlashMLA yet.
- Next:
  - CP-align and compare L0 attention internals on positions `19076..19586`:
    - `L00_attn_qr_kv`, `L00_attn_qr_norm`, `L00_attn_kv_norm`,
      `L00_attn_q_pre_mla`.
    - `L00_mla_prefill_selected_kv_tail` / attention output heads.
    - `L00_attn_wo_a_out` / final `L00_attn_out`.

### 2026-05-20 03:02 CST - Enabled RTP Prefill SWA FP8 Roundtrip For Verification

- CP-aligned L0 attention internal compare:
  - Summary:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/l0_attn_prefill_internal_cp_aligned_20260520_0300.json`.
  - On positions `19076..19586`:
    - `L00_attn_qr_kv`: exact (`511 / 511` rows).
    - `L00_attn_qr_norm`: exact (`511 / 511` rows).
    - `L00_attn_kv_norm`: exact (`511 / 511` rows).
    - `L00_attn_out`: differs with max `0.40625`, mean `0.0463197`.
- Interpretation:
  - L0 input and Q/KV projection/norm are aligned.
  - The remaining diff is in prefill attention numeric path.
  - RTP default cold prefill consumes BF16 `kv_full` directly, while vLLM writes
    SWA K to FP8 cache then dequant/gathers it for prefill attention.
- Launcher change:
  - File:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_prefill.sh`.
  - Added:
    `DSV4_PREFILL_SWA_FP8_ROUNDTRIP=1`.
- Next:
  - Restart only RTP prefill `18800`.
  - Re-run RTP one-token probe and compare against existing vLLM L0/L1 tail
    dump.
  - If `L00_attn_out` becomes exact or much closer, keep this switch and run
    the 110/1000-token gates.

### 2026-05-20 03:08 CST - FP8 Roundtrip Greatly Reduces L0 Attention Diff

- Re-ran RTP one-token probe after enabling
  `DSV4_PREFILL_SWA_FP8_ROUNDTRIP=1`.
- RTP dump:
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_prefill_l1_swa_write_record89_20260520_0205/prefill_record89`.
  - `rank0_pid3865913_step015.pt`.
  - `rank1_pid3865915_step015.pt`.
- CP-aligned compare summary:
  `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/l0_l1_prefill_tail_compare_roundtrip_cp_aligned_20260520_0305.json`.
- On positions `19076..19586`:
  - Still exact:
    - `L00_attn_in`: `511 / 511` rows exact.
    - `L00_attn_qr_kv`: `511 / 511` rows exact.
    - `L00_attn_qr_norm`: `511 / 511` rows exact.
    - `L00_attn_kv_norm`: `511 / 511` rows exact.
  - `L00_attn_out` improved from:
    - before: max `0.40625`, mean `0.0463197`, rms `0.0588693`.
    - after: max `0.03125`, mean `3.4600e-05`, rms `0.00057035`,
      `502 / 511` rows exact.
- Conclusion:
  - The main prefill L0 attention divergence was the BF16-KV cold-prefill path
    versus vLLM's FP8-cache roundtrip path.
  - Keep `DSV4_PREFILL_SWA_FP8_ROUNDTRIP=1` for the next token-level gates.
  - Remaining small non-exact rows may or may not affect greedy token IDs; gate
    before adding more precision changes.
- Next:
  - Run 110-token teacher-forced gate.
  - Then run 1000-token teacher-forced/natural mismatch count.

### 2026-05-20 03:18 CST - Teacher-Forced 1000 Passes; Natural Top1 Still Has 26 Mismatches

- 110-token teacher-forced gate after `DSV4_PREFILL_SWA_FP8_ROUNDTRIP=1`:
  - Run:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/rtp_teacher_forced_roundtrip_len110_record89_20260520_022834`.
  - Result:
    - `compare_len=110`.
    - `first_diff=null`.
    - hash `95fc6e2e8c14f806`.
- 1000-token teacher-forced gate:
  - Run:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/rtp_teacher_forced_roundtrip_len1000_record89_20260520_022923`.
  - Result:
    - `compare_len=1000`.
    - `first_diff=null`.
    - RTP/vLLM hash both `986b77c92c844fc6`.
- Natural sampler-log check:
  - Sampler log:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_teacher_forced_len130_sampler_logits_flashinfer_trtllm_18980_20260519_2324.jsonl`.
  - Latest 1000-token run has `26` natural-top1 mismatches.
  - New first mismatch:
    - `oracle_idx=101`.
    - sampler `step=19687`.
    - teacher/vLLM token `1644`.
    - RTP natural top1 `13660`.
    - top candidates:
      `[13660, 1644, 14604, 7790, 4741, 9968, 29329, 410]`.
    - top values:
      `[32.626, 32.5897, 32.5345, 31.7057, 31.6726, 30.6504, 29.1569, 28.7166]`.
- Conclusion:
  - Teacher-forced equality is restored for 1000 tokens, but final natural
    greedy target is not solved.
  - Continue RCA at the new first natural mismatch around `step=19687`.

### 2026-05-20 03:35 CST - Retargeted Decode Boundary Dumps To New Natural First Diff

- Reason:
  - After `DSV4_PREFILL_SWA_FP8_ROUNDTRIP=1`, 1000-token teacher-forced IDs
    match vLLM, but natural greedy still has 26 top1 mismatches.
  - Current first natural mismatch is `oracle_idx=101`, predicted by decode
    forward at `pos=19686` with input token `30869`; vLLM/teacher token is
    `1644`, RTP natural top1 is `13660`.
- Temporary RTP decode launcher update:
  - File:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_rtp_decode_flashinfer_trtllm_18980.sh`.
  - Changed `MOEDBG_GLOBAL_POS` and `DSV4_DECODE_DUMP_POS` from `19689` to
    `19686`.
  - Changed cases to `rtp_pos19686` under new `*_natural_firstdiff_20260520_0335`
    dump directories.
  - Expanded `MOEDBG_LAYER=0,1` and regex to `^(decode_|L00_|L01_)` for the
    first narrow decode RCA pass.
- Temporary vLLM dump launcher update:
  - File:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_vllm_l1dump_18100.sh`.
  - Changed case to `vllm_decode_layers_pos19686_record89_20260520_0335`.
  - Changed `VLLM_DSV4_DUMP_MAX_SEQ=1`, `VLLM_DSV4_DUMP_LAYER=` and
    `VLLM_DSV4_DUMP_TAIL_TOKENS=0` so this run captures decode single-token
    boundary tensors instead of prefill tail tensors.
  - Regex now captures only boundary keys:
    `input_ids`, `positions`, `embed_out`, `embed_hc_expanded`,
    `layerXX_out`, `final_hidden`, `final_norm`, `lm_top_values`,
    `lm_top_indices`.
- Next:
  - Restart only temporary vLLM `18100` and RTP test decode `18980`.
  - Run a 110-token request on both services.
  - Compare records where `positions=[19686]` and `input_ids=[30869]` to find
    the first divergent layer boundary.

### 2026-05-20 03:43 CST - Decode Boundary Compare Finds First Diff Inside Layer 1

- Ran RTP 110-token teacher-forced gate with dumps retargeted to `pos=19686`:
  - Run:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/rtp_teacher_forced_record89_len110_20260520_024015_record89_20260520_024025`.
  - Result:
    - `compare_len=110`.
    - `first_diff=null`.
    - hash `95fc6e2e8c14f806`.
  - RTP decode dump:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_decode_dump_natural_firstdiff_20260520_0335/rtp_pos19686/rank0_pid3880946_step000.pt`.
  - RTP dump extra confirms target alignment:
    `input_ids=[30869]`, `positions=[19686]`, `sequence_lengths=[19686]`.
- Ran temporary vLLM `18100` 110-token greedy request:
  - Run:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/vllm_decode_layers_pos19686_record89_len110_20260520_0340_record89_20260520_024114`.
  - vLLM output prefix matches the expected oracle window through the target
    token.
  - Matching vLLM dump for the same decode forward:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_decode_layers_pos19686_record89_20260520_0335/rank0_pid3880874_step100.pt`.
  - vLLM dump extra confirms target alignment:
    `input_ids=[30869]`, `positions=[19686]`.
- Layer-boundary compare at `pos=19686`, `input_id=30869`:
  - `input_ids` / `positions`: equal.
  - `layer00_out`: exact, `max=0`, `rows_exact=4/4`.
  - First divergent layer boundary: `layer01_out`:
    - max `0.0224609375`.
    - mean `0.0016165031`.
    - rms `0.0030176116`.
    - `rows_exact=0/4`.
  - Later layer diffs grow progressively; final hidden differs with max
    `0.173828125`, mean `0.01738087`.
- Conclusion:
  - The current first natural-greedy mismatch is not caused by layer0 decode
    output, prompt/cache mapping, or input token selection at this step.
  - The first proven tensor divergence is inside decode layer 1.
- Next:
  - Expand the temporary vLLM dump to `VLLM_DSV4_DUMP_LAYER=1` and regex
    `L01_` internals.
  - Reuse the existing RTP MOEDBG `L01_` dump at `pos=19686` if it contains
    matching keys; otherwise rerun RTP with a narrower L1 regex.
  - Compare `L01_attn_in`, Q/KV projection/norm, attention output, and FFN/MoE
    checkpoints to locate the first L1 internal divergence.

### 2026-05-20 03:50 CST - Expanded Temporary vLLM Dump To L1 Internals

- Inspected current RTP MOEDBG dump for `pos=19686`:
  - File:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_natural_firstdiff_20260520_0335/rtp_pos19686/rank0_pid3880946_step000.pt`.
  - Contains L0 and L1 internals, including:
    `L01_decode_attn_in`, `L01_fp8_decode_qr_norm`,
    `L01_fp8_decode_kv_norm_ref`, `L01_fp8_decode_swa_selected_k_dequant`,
    `L01_fp8_decode_wo_a_out`, and `L01_moe_*`.
- Checked logits at the target step from boundary dumps:
  - RTP top indices:
    `[13660, 14604, 1644, 7790, 4741, 9968, 29329, 410, ...]`.
  - RTP top values:
    `[32.75, 32.5, 32.5, 31.75, 31.625, 30.625, 29.125, 28.75, ...]`.
  - vLLM top indices:
    `[1644, 14604, 13660, 7790, 4741, 9968, 29246, 29329, ...]`.
  - vLLM top values:
    `[33.5, 33.25, 33.25, 32.5, 32.25, 31.125, 29.375, 29.125, ...]`.
  - This reproduces the natural-greedy first mismatch in direct logits:
    vLLM picks `1644`, RTP picks `13660`.
- Temporary vLLM launcher update:
  - File:
    `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/start_vllm_l1dump_18100.sh`.
  - Changed case to `vllm_decode_L1_internal_pos19686_record89_20260520_0350`.
  - Set `VLLM_DSV4_DUMP_LAYER=1`.
  - Regex captures L1 internals plus layer0/layer1 boundaries:
    `input_ids`, `positions`, `embed_out`, `embed_hc_expanded`,
    `layer00_out`, `layer01_out`, `L01_*`, `final_hidden`,
    `lm_top_values`, `lm_top_indices`.
- Next:
  - Restart only temporary vLLM `18100`.
  - Rerun the 110-token vLLM request and compare L1 internals against the
    existing RTP MOEDBG target dump.

### 2026-05-20 04:02 CST - Patched Temporary vLLM Dump Layer Fallback

- Problem:
  - After setting `VLLM_DSV4_DUMP_LAYER=1`, the temporary vLLM run still wrote
    only 9 boundary tensors and no `L01_*` internals.
  - Log showed `Unknown vLLM environment variable detected: VLLM_DSV4_DUMP_LAYER`.
  - Effective behavior: `VLLM_DSV4_DUMP_CASE/DIR/REGEX` reached the EngineCore,
    but layer gating did not enable `L01_*` internal dump points.
- Temporary vLLM environment patch:
  - File:
    `/data3/vllm-dsv4-env/lib/python3.10/site-packages/vllm/model_executor/models/deepseek_v4.py`.
  - Added `_DSV4_DUMP_LAYER_FALLBACK`, inferred from `L([0-9]{2})_` entries in
    `VLLM_DSV4_DUMP_NAME_REGEX`.
  - `_dsv4_dump_should_layer()` now uses explicit `VLLM_DSV4_DUMP_LAYER` first,
    then the regex-derived fallback.
- Additional temporary vLLM MoE dump patches:
  - File:
    `/data3/vllm-dsv4-env/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/runner/moe_runner.py`.
  - File:
    `/data3/vllm-dsv4-env/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py`.
  - Both now infer the target layer from `VLLM_DSV4_DUMP_NAME_REGEX` when
    `VLLM_DSV4_DUMP_LAYER` is not available in the worker.
- Scope:
  - This is only for the temporary vLLM dump service on `18100`.
  - RTP code and stable vLLM oracle `18000` were not touched.
- Next:
  - Restart temporary vLLM `18100`.
  - Rerun the 110-token request.
  - Confirm `step100` now contains `L01_*` internals and compare with RTP.

### 2026-05-20 04:11 CST - Added Temporary vLLM Dump Diagnostics

- The fallback patch did not produce `L01_*` internals; `step100` still had
  only 9 boundary tensors.
- Added temporary diagnostics to the vLLM dump instrumentation:
  - File:
    `/data3/vllm-dsv4-env/lib/python3.10/site-packages/vllm/model_executor/models/deepseek_v4.py`.
    - Prints one `[VLLM_DSV4_DUMP_DIAG]` line showing dump case, explicit
      layer env, regex-derived fallback layers, and regex.
  - File:
    `/data3/vllm-dsv4-env/lib/python3.10/site-packages/vllm/model_executor/layers/deepseek_v4_attention.py`.
    - Prints one `[VLLM_DSV4_ATTN_DUMP_DIAG]` line showing whether L1 attention
      dump is enabled and whether the model dump thread-local buffer exists,
      or the exception type/message if the helper is failing silently.
- Next:
  - Restart temporary vLLM `18100` and rerun the 110-token request.
  - Use the diagnostic line to decide whether to fix layer gating, thread-local
    buffer visibility, or an exception inside the attention dump helper.

### 2026-05-20 04:18 CST - Fixed Temporary vLLM Internal Dump Buffer Visibility

- Diagnostic result from the previous vLLM run:
  - `[VLLM_DSV4_DUMP_DIAG]` showed `layer_env='1'`, `fallback=[1]`, and the
    expected `L01_` regex.
  - `[VLLM_DSV4_ATTN_DUMP_DIAG] layer=1 name=attn_qr_kv buf_is_none=True`.
- Interpretation:
  - L1 internal dump points are reached and layer gating is enabled.
  - The failure is in instrumentation plumbing: attention custom-op execution
    cannot see the model forward's thread-local dump buffer.
- Temporary vLLM patch:
  - File:
    `/data3/vllm-dsv4-env/lib/python3.10/site-packages/vllm/model_executor/models/deepseek_v4.py`.
  - Added `_DSV4_DUMP_GLOBAL_BUF` as a fallback shared buffer.
  - `_dsv4_dump_begin()` now initializes both thread-local and global buffer.
  - `_dsv4_dump_record()` falls back to the global buffer when thread-local
    buffer is missing.
  - `_dsv4_dump_flush()` can flush the global buffer and clears it afterward.
- Scope:
  - Temporary vLLM dump service only; stable vLLM oracle `18000` and RTP code
    were not touched.
- Next:
  - Restart temporary vLLM `18100` and rerun 110-token request.
  - Confirm `step100` includes `L01_*` internals.

### 2026-05-20 04:25 CST - Added Temporary vLLM Attention Side Buffer

- Global-buffer fallback still produced only boundary tensors.
- Added a stronger temporary dump path for vLLM attention internals:
  - File:
    `/data3/vllm-dsv4-env/lib/python3.10/site-packages/vllm/model_executor/layers/deepseek_v4_attention.py`.
  - Added `_DSV4_ATTN_SIDE_BUF` and `_dsv4_take_attn_side_buf()`.
  - `_dsv4_model_dump_record()` now appends enabled decode-shaped (`shape[0] == 1`)
    `Lxx_*` tensors into the side buffer in addition to the normal record path.
  - File:
    `/data3/vllm-dsv4-env/lib/python3.10/site-packages/vllm/model_executor/models/deepseek_v4.py`.
  - `_dsv4_dump_flush()` now imports the attention module, drains the side
    buffer, merges it into the normal dump payload, then clears it.
- Rationale:
  - Diagnostics showed layer gating is correct but thread-local buffer is not
    visible inside attention custom-op (`buf_is_none=True`).
  - The side buffer decouples internal attention capture from the model-forward
    thread-local buffer while still flushing at the model output step.
- Next:
  - Restart `18100`, rerun 110-token vLLM request, confirm `step100` contains
    `L01_*`, then compare L1 internals against RTP.

### 2026-05-20 04:32 CST - L1 Internal Compare: Q Path Exact, SWA Cache Content Is First Diff

- Side-buffer vLLM run succeeded:
  - Matching vLLM dump:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_decode_L1_internal_pos19686_record89_20260520_0350/rank0_pid3909584_step103.pt`.
  - Extra confirms target alignment:
    `input_ids=[30869]`, `positions=[19686]`.
  - Top logits remain oracle ordering:
    `[1644, 14604, 13660, 7790, 4741, ...]`.
- Compared against RTP dump:
  - RTP dump:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_flashinfer_trtllm_moedbg_natural_firstdiff_20260520_0335/rtp_pos19686/rank0_pid3880946_step000.pt`.
- L1 attention exact checkpoints:
  - `L01_fp8_decode_qr_norm` vs `L01_attn_qr_norm`: exact.
  - `L01_fp8_decode_kv_norm_ref` vs `L01_attn_kv_norm`: exact.
  - `L01_fp8_decode_q_linear` vs `L01_attn_q_linear`: exact (`64/64` rows).
  - `L01_fp8_decode_q` vs `L01_attn_q_pre_mla`: exact (`64/64` rows).
- First non-exact comparable tensor:
  - `L01_fp8_decode_swa_selected_cache_logical` vs
    `L01_attn_decode_swa_selected_cache_logical`.
  - `L01_fp8_decode_swa_topk` is local-window offset style (`615..`) while
    vLLM `L01_attn_decode_swa_indices` is global slot style (`437415..`), so
    the raw index values are not directly comparable.
  - The selected cache content is comparable and differs.
- Selected SWA cache/dequant summary after squeezing to `[128,584]` and
  `[128,512]`:
  - Cache byte rows exact: `91 / 128`.
  - Dequant K rows exact: `91 / 128`.
  - Diff rows:
    `[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,31,38,47,48,88,89,100,117]`.
  - Per-row K max is mostly `0.125`, with row 16 max `0.25` and row 8/100 max
    `0.0625`.
  - No small shift in `[-8,8]` explains the rows, so this is not a simple
    off-by-one selected-window alignment issue.
- Downstream propagation:
  - `L01_fp8_decode_o_heads` vs `L01_attn_mla_out_heads` differs:
    max `0.0625`, mean `2.4448e-4`, rms `7.7938e-4`.
  - `L01_fp8_decode_wo_a_out` vs `L01_attn_wo_a_out` differs:
    max `0.015625`, mean `0.0023970`, rms `0.0032905`.
  - `layer01_out` differs with max `0.0224609375`, mean `0.0016165`.
- Conclusion:
  - At `pos=19686`, L1 decode input/Q/KV projection and Q path are aligned.
  - The first proven internal divergence is the selected L1 SWA cache content,
    not the L1 attention query computation.
  - This points backward to historical L1 SWA cache writes for selected rows,
    likely during prompt prefill and/or earlier decode writes, rather than a
    current-step FlashMLA query/kernel issue.
- Next:
  - Map the differing selected rows to semantic history positions.
  - Dump/compare L1 SWA write bytes for those positions on RTP and vLLM.
  - Start with rows `0..27` because they are contiguous and likely correspond
    to the oldest selected SWA window positions for this decode step.

### 2026-05-20 03:37 CST - Corrected L1 SWA Prefill Mapping: Decode Reads Match Prefill Writes

- Reused existing dumps only; no service restart and no code change in this
  step.
- Compared the natural first-diff decode window at `pos=19686` against prefill
  SWA writes for semantic positions `19559..19586`.
- Important correction:
  - RTP prefill `L01_fp8_prefill_swa_written_cache_logical` is produced from
    the CP all-gathered global `seq_len_full=19587` write view and was tail
    trimmed to 512 rows.
  - Therefore its row map is `global_pos = 19587 - 512 + row`, not the
    rank-local `extra.global_positions[-512 + row]`.
  - The earlier rank-local row mapping was off by one near the CP padded tail;
    do not use it for SWA write tensors.
- Correct CP-global comparison output:
  `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/l1_swa_prefill_decode_origin_pos19559_19586_cp_globalmap_20260520_0510.json`.
- Correct result:
  - RTP prefill write bytes vs RTP decode selected bytes for `19559..19586`:
    `28/28` rows exact, dequant max `0.0`.
  - vLLM prefill write bytes vs vLLM decode selected bytes for `19559..19586`:
    `28/28` rows exact, dequant max `0.0`.
  - RTP prefill vs vLLM prefill dequant for those rows:
    max row max `0.25`, first 8 row max all `0.125`.
  - RTP decode vs vLLM decode dequant shows the same numbers, proving decode
    is faithfully reading the prefill-written cache.
- Upstream L1 prefill comparison output:
  `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/l1_prefill_upstream_pos19559_19586_cp_globalmap_20260520_0516.json`.
- Upstream result for positions `19559..19586`:
  - `L01_attn_in`: max row max `0.00341796875`.
  - `L01_attn_qr_kv`: max row max `0.0089111328125`.
  - `L01_attn_qr_norm`: max row max `0.0030517578125`.
  - `L01_attn_kv_norm`: max row max `0.06640625`.
  - RTP local `attn_kv_norm` and RTP full pre-rope KV cache are exact after
    CP-global mapping, so the RTP internal write path is self-consistent.
- Conclusion:
  - Current decode topk / slot translation is not the root for the contiguous
    first 28 rows.
  - The first actionable divergence for the natural first-diff path is already
    in L1 prefill `attn_kv_norm`, before SWA FP8 quantized cache write.
- Next:
  - Trace the L1 prefill `attn_in -> qr_kv -> kv_norm` path and compare it
    with vLLM semantics.
  - Pay attention to RMSNorm precision/epsilon and whether the small L1 input
    difference from layer0 is being amplified by the KV norm.

### 2026-05-20 03:45 CST - L0 Attention Exact On Target Rows, Diff Starts After Residual/FFN Path

- Reused existing RTP/vLLM prefill dumps only; no service restart and no code change.
- Compared L0 -> L1 tensors for semantic positions `19559..19586`.
- Output:
  `/data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/l0_to_l1_pos19559_19586_20260520_0540.json`.
- Result on `19559..19586`:
  - `L00_attn_out`: `28 / 28` rows exact.
  - `L00_attn_residual`: not exact, max row max `0.0009765625`.
  - `L00_ffn_in`: max row max `0.0078125`.
  - `L00_ffn_out`: max row max `0.09375`.
  - `layer00_out`: max row max `0.01953125`.
  - `L01_attn_in`: max row max `0.00341796875`.
  - `L01_attn_kv_norm`: max row max `0.06640625`.
- Interpretation:
  - For the positions that form the first divergent decode SWA window, the
    remaining prefill divergence is not caused by same-position L0 attention
    output.
  - It starts after L0 attention output, inside the residual/HC/FFN input path,
    and is then amplified by L0 FFN and L1 KV RMSNorm.
- Next:
  - Compare RTP/vLLM block residual and HC mixing semantics after attention.
  - Inspect whether residual update uses a different HC lane, reduction order,
    dtype, or in-place update order.

### 2026-05-20 03:47 CST - Existing Prefill Dumps Lack RTP HC Window Internals

- Reused existing dumps only; no service restart and no code change.
- Actual files inspected:
  - RTP:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_prefill_l1_swa_write_record89_20260520_0205/prefill_record89/rank0_pid3865913_step018.pt`.
  - vLLM:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/dumps/vllm_l0_l1_prefill_tail_record89_20260520_0245/rank0_pid3856327_step004.pt`.
- Result:
  - RTP dump has `L00_attn_in`, `L00_attn_out`, `L00_attn_residual`,
    `L00_ffn_in`, `L00_ffn_out`, `layer00_out`, and `L01_attn_in`.
  - RTP dump does not have the full-window HC checkpoints needed to compare
    with vLLM: `L00_attn_hc_pre`, `L00_attn_post_mix`,
    `L00_attn_comb_mix`, and `L00_attn_post_residual_in`.
  - vLLM dump already has those full-window HC checkpoints, plus mHC internal
    diagnostics such as `L00_attn_hc_pre_gemm_out_mul`,
    `L00_attn_hc_pre_gemm_out_sqrsum`, and raw post/comb/layer input tensors.
- Interpretation:
  - Existing evidence proves the first target-window diff appears after
    `L00_attn_out`, but cannot yet distinguish HC post arithmetic from its
    post/comb/residual inputs.
- Next:
  - Add minimal RTP recording for full-window L0/L1 HC checkpoints already
    present on the vLLM side.
  - Rerun only the RTP prefill dump on port `18800`, then compare the new RTP
    HC tensors against vLLM.

### 2026-05-20 03:49 CST - Added RTP HC Boundary Dump Points

- Code change:
  - File:
    `/data3/dsv4_repeat_compare/worktrees/RTP-LLM-dsv4-precision/github-opensource/rtp_llm/models_py/modules/dsv4/block.py`.
  - Added level-2 `MOEDBG` records matching vLLM checkpoint names:
    - `Lxx_attn_hc_pre`
    - `Lxx_attn_post_mix`
    - `Lxx_attn_comb_mix`
    - `Lxx_attn_post_residual_in`
    - `Lxx_ffn_hc_pre`
    - `Lxx_ffn_post_mix`
    - `Lxx_ffn_comb_mix`
    - `Lxx_ffn_post_residual_in`
- Scope:
  - Recording only; no inference math, kernel path, cache path, or sampler
    behavior changed.
  - Existing `MOEDBG_NAME_REGEX` in `start_rtp_prefill.sh` already includes
    `L00_`, so L0 new tensors will be captured without script changes.
- Purpose:
  - Distinguish whether the target-window difference after `L00_attn_out`
    comes from HC post inputs (`post`, `comb`, `residual`) or from HC post
    kernel arithmetic/rounding.
- Next:
  - Restart only RTP prefill `tmux dsv4_rtp_prefill_18800`.
  - Rerun the record-89 prefill request and compare the new RTP HC tensors
    with the existing vLLM prefill dump.

### 2026-05-20 03:56 CST - Natural 1000-Token RTP/vLLM Gate Passed

- Service actions:
  - Restarted only RTP prefill:
    `tmux dsv4_rtp_prefill_18800`.
  - Did not touch stable vLLM oracle `127.0.0.1:18000`.
  - Did not touch RTP baseline decode `18880/18881`.
  - Used existing RTP test decode `127.0.0.1:18980` with prefill
    `127.0.0.1:18800`.
- Request command:
  - `docs/skills/rtp-vllm-precision-bisect/scripts/rtp_vllm_precision.py known-good-record89 --rtp-url http://127.0.0.1:18980 --prefill-url http://127.0.0.1:18800 --grpc-addr 127.0.0.1:18981 --decode-http-port 18980 --decode-grpc-port 18981 --name rtp_hc_window_dump_record89_20260520_0352 --out-root /data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs --json-out /data3/dsv4_repeat_compare/compare/forced_gate_20260519_172757/outputs/rtp_hc_window_dump_record89_20260520_0352.json`
- Result:
  - Output run dir:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_hc_window_dump_record89_20260520_0352_record89_20260520_035325`.
  - RTP generated ids:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_hc_window_dump_record89_20260520_0352_record89_20260520_035325/rtp_run01/generated_ids.json`.
  - vLLM oracle ids:
    `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/vllm_stable_nodump_ignoreeos_len1000_oracle_20260517_220909_record89_20260517_220916/vllm_run01/generated_ids.json`.
  - `compare_len=1000`.
  - `first_diff=null`.
  - `equal_prefix=true`.
  - `rtp_hash=986b77c92c844fc6`.
  - `vllm_hash=986b77c92c844fc6`.
  - `hash_matches=true`.
- Note:
  - RTP generated 1001 ids in this run; the precision gate compares the first
    1000 generated ids against the 1000-token vLLM oracle and they are exact.
  - The only RTP code change in this cycle was debug recording in `block.py`;
    no inference math was changed in this step.
- Status:
  - The user's current acceptance target, natural greedy 1000-token token-id
    equality against stable vLLM, is satisfied on this run.

### 2026-05-20 03:57 CST - Confirmed New RTP Record89 HC Prefill Dump

- Confirmed matching RTP prefill dump from the successful natural 1000-token
  run:
  `/data3/dsv4_repeat_compare/compare/hidden_dump_q89_topk1_20260516_165959/outputs/rtp_prefill_l1_swa_write_record89_20260520_0205/prefill_record89/rank0_pid3939723_step015.pt`.
- Dump metadata:
  - `seq_len_full=19587`.
  - `input_ids_shape=[9794]`.
  - `global_positions_shape=[9794]`.
  - `global_positions_tail=[19582, 19583, 19584, 19585, 19586, 19586]`.
  - `tensor_count=48`.
- New HC checkpoint presence:
  - `L00_attn_hc_pre`: present.
  - `L00_attn_post_mix`: present.
  - `L00_attn_comb_mix`: present.
  - `L00_attn_post_residual_in`: present.
  - `L00_ffn_hc_pre`: present.
  - `L00_ffn_post_residual_in`: present.
- Since the natural 1000-token acceptance gate already passed, no further
  inference-logic changes were made after this confirmation.

### 2026-05-20 10:18 CST - Updated Portable Skill and CLI Workflow

- Skill update:
  - File:
    `/data3/dsv4_repeat_compare/worktrees/RTP-LLM-dsv4-precision/github-opensource/docs/skills/rtp-vllm-precision-bisect/SKILL.md`.
  - Added a generic `run-rtp` example for cases other than the built-in
    `/data3/q` record 89 gate.
  - Added a portable reproduction checklist for another machine:
    build same-worktree RTP libraries, configure vLLM oracle stability,
    generate local launch scripts, use non-conflicting GPUs/ports, run health,
    verify loaded `.so` roots, optionally verify env dumps, prove vLLM oracle
    stability, then use natural RTP self-roll for final proof.
  - Clarified that `known-good-record89` is intentionally hard-coded for the
    original reproduction; other cases should use `run-rtp`, `run-teacher`, and
    `compare` with explicit arguments.
- CLI update:
  - File:
    `/data3/dsv4_repeat_compare/worktrees/RTP-LLM-dsv4-precision/github-opensource/docs/skills/rtp-vllm-precision-bisect/scripts/rtp_vllm_precision.py`.
  - Fixed `write-launch-scripts` so common RTP env unsets
    `DSV4_INDEXER_TOPK_BACKEND`; only the generated prefill script re-exports
    the requested prefill backend, default `torch`.
  - Decode script now keeps `DSV4_INDEXER_TOPK_BACKEND` unset while preserving
    `DSV4_INDEXER_TOPK_CANONICALIZE=1`,
    `RTP_STABLE_GREEDY_TIEBREAK=1`, and
    `RTP_CUDA_GRAPH_SYNC_REPLAY=1`.
  - Manifest now reports:
    - `RTP_PREFILL_DSV4_INDEXER_TOPK_BACKEND`
    - `RTP_DECODE_DSV4_INDEXER_TOPK_BACKEND`
    - `DSV4_INDEXER_TOPK_CANONICALIZE`
- Verification:
  - Ran syntax check:
    `python3 -m py_compile docs/skills/rtp-vllm-precision-bisect/scripts/rtp_vllm_precision.py`.
  - Ran CLI launch-bundle smoke generation to `/tmp/dsv4_precision_cli_smoke`.
  - Confirmed generated decode script lines:
    - `unset DSV4_INDEXER_TOPK_BACKEND`
    - `export DSV4_INDEXER_TOPK_CANONICALIZE=1`
    - `export RTP_STABLE_GREEDY_TIEBREAK=1`
    - `export RTP_CUDA_GRAPH_SYNC_REPLAY=1`
  - Confirmed generated prefill script re-exports:
    - `export DSV4_INDEXER_TOPK_BACKEND='torch'`
  - Confirmed generated manifest reports:
    - `RTP_PREFILL_DSV4_INDEXER_TOPK_BACKEND=torch`
    - `RTP_DECODE_DSV4_INDEXER_TOPK_BACKEND=unset`
