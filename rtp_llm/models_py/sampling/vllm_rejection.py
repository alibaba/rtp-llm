# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import triton
import triton.language as tl
from rtp_llm.models_py.sampling.draft_logits import DraftLogitsCache, draft_logits_row
from rtp_llm.models_py.sampling.vllm_gumbel import gumbel_block_argmax, tl_rand32
from triton.language.extra.cuda import libdevice as tldevice


@triton.jit
def _compute_max_and_sumexp(logits):
    max = tl.max(logits, axis=0)
    sumexp = tl.where(
        max > float("-inf"),
        tl.sum(tl.exp(logits - max)),
        0.0,
    )
    return max, sumexp


@triton.jit
def _compute_global_logsumexp(
    local_max_ptr,
    local_max_stride,
    local_sumexp_ptr,
    local_sumexp_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    maxes = tl.load(
        local_max_ptr + logit_idx * local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    sumexps = tl.load(
        local_sumexp_ptr + logit_idx * local_sumexp_stride + blocks,
        mask=blocks_mask,
        other=0.0,
    )
    global_max = tl.max(maxes, axis=0)
    global_lse = global_max + tl.log(tl.sum(sumexps * tl.exp(maxes - global_max)))
    return global_lse


@triton.jit
def _compute_global_target_argmax(
    target_local_max_ptr,
    target_local_max_stride,
    target_local_argmax_ptr,
    target_local_argmax_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    local_max = tl.load(
        target_local_max_ptr + logit_idx * target_local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    # See _insert_resampled_kernel: NaN breaks tl.argmax index bounds.
    local_max = tl.where(local_max != local_max, float("-inf"), local_max)
    max_block_idx = tl.argmax(local_max, axis=0)
    return tl.load(
        target_local_argmax_ptr + logit_idx * target_local_argmax_stride + max_block_idx
    ).to(tl.int64)


@triton.jit
def _compute_global_logprobs_and_logsumexp(
    token,
    mask,
    logit_idx,
    req_state_idx,
    draft_step,
    temp,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    DRAFT_VALUES_ARE_PROBS: tl.constexpr,
):
    target_logit = tl.load(
        target_logits_ptr + logit_idx * target_logits_stride + token,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    target_lse = _compute_global_logsumexp(
        target_local_max_ptr,
        target_local_max_stride,
        target_local_sumexp_ptr,
        target_local_sumexp_stride,
        logit_idx,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
    )
    target_log_prob = target_logit - target_lse
    if HAS_DRAFT_LOGITS:
        # draft_logits is stored pre-temperature, so apply scale first.
        draft_logit = tl.load(
            draft_logits_ptr
            + tl.maximum(
                draft_logits_row(req_state_idx, draft_logits_row_mapping_ptr), 0
            )
            * draft_logits_stride_0
            + draft_step * draft_logits_stride_1
            + token,
            mask=mask,
            other=0.0 if DRAFT_VALUES_ARE_PROBS else float("-inf"),
        ).to(tl.float32)
        if DRAFT_VALUES_ARE_PROBS:
            # Match the dtype rounding of a materialized Tensor.log().
            draft_logit = (
                tldevice.log(draft_logit)
                .to(draft_logits_ptr.dtype.element_ty)
                .to(tl.float32)
            )
        draft_logit = draft_logit / temp
        draft_lse = _compute_global_logsumexp(
            draft_local_max_ptr,
            draft_local_max_stride,
            draft_local_sumexp_ptr,
            draft_local_sumexp_stride,
            logit_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        draft_log_prob = draft_logit - draft_lse
    else:
        # One-hot draft: q(token) = 1, log_q = 0.
        draft_log_prob = 0.0
        draft_lse = 0.0
    return target_log_prob, draft_log_prob, target_lse, draft_lse


@triton.jit
def _is_bonus_logit(expanded_local_pos_ptr, logit_idx, num_logits):
    next_local_pos = tl.load(
        expanded_local_pos_ptr + logit_idx + 1,
        mask=logit_idx + 1 < num_logits,
        other=0,
    )
    return next_local_pos == 0


# num_logits only bounds the bonus-logit lookahead; one kernel serves every value.
@triton.jit(do_not_specialize=["num_logits"])
def _compute_local_logits_stats_kernel(
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    num_logits,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    DRAFT_VALUES_ARE_PROBS: tl.constexpr,
    graph_input_addresses_ptr=None,
):
    if graph_input_addresses_ptr is not None:
        target_logits_ptr = (
            tl.load(graph_input_addresses_ptr).to(tl.uint64).to(target_logits_ptr.dtype)
        )

    logit_idx = tl.program_id(0).to(tl.int64)

    if _is_bonus_logit(expanded_local_pos_ptr, logit_idx, num_logits):
        # Max/argmax and summed exponentials are not needed.
        return

    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    is_draft_cta = DRAFT_VALUES_ARE_PROBS and tl.program_id(2) == 1
    if temp == 0.0:
        if is_draft_cta:
            return
        # Greedy sampling. Only the target max/argmax are needed.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        value, idx = tl.max(target_logits, axis=0, return_indices=True)
        token_id = block_idx * BLOCK_SIZE + idx
        tl.store(
            target_local_argmax_ptr
            + logit_idx * target_local_argmax_stride
            + block_idx,
            token_id,
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            value,
        )
    else:
        if not is_draft_cta:
            # Get local target max and summed exponentials.
            target_logits = tl.load(
                target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            target_max, target_sumexp = _compute_max_and_sumexp(target_logits)
            tl.store(
                target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
                target_max,
            )
            tl.store(
                target_local_sumexp_ptr
                + logit_idx * target_local_sumexp_stride
                + block_idx,
                target_sumexp,
            )
        if DRAFT_VALUES_ARE_PROBS and not is_draft_cta:
            return
        if HAS_DRAFT_LOGITS:
            # Get local draft max and summed exponentials. draft_logits is
            # stored pre-temperature, so apply scale first.
            draft_logits = tl.load(
                draft_logits_ptr
                + tl.maximum(
                    draft_logits_row(req_state_idx, draft_logits_row_mapping_ptr), 0
                )
                * draft_logits_stride_0
                + draft_step_idx * draft_logits_stride_1
                + block_offsets,
                mask=mask,
                other=0.0 if DRAFT_VALUES_ARE_PROBS else float("-inf"),
            ).to(tl.float32)
            if DRAFT_VALUES_ARE_PROBS:
                # Match the dtype rounding of a materialized Tensor.log().
                draft_logits = (
                    tldevice.log(draft_logits)
                    .to(draft_logits_ptr.dtype.element_ty)
                    .to(tl.float32)
                )
            draft_logits = draft_logits / temp
            draft_max, draft_sumexp = _compute_max_and_sumexp(draft_logits)
            tl.store(
                draft_local_max_ptr + logit_idx * draft_local_max_stride + block_idx,
                draft_max,
            )
            tl.store(
                draft_local_sumexp_ptr
                + logit_idx * draft_local_sumexp_stride
                + block_idx,
                draft_sumexp,
            )


@triton.jit
def _rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_logits, num_blocks]
    target_local_argmax_ptr,
    target_local_argmax_stride,
    # [num_logits, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_logits, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    # [num_speculative_steps]
    synthetic_conditional_rates_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    DRAFT_VALUES_ARE_PROBS: tl.constexpr,
    graph_input_addresses_ptr=None,
):
    if graph_input_addresses_ptr is not None:
        target_logits_ptr = (
            tl.load(graph_input_addresses_ptr).to(tl.uint64).to(target_logits_ptr.dtype)
        )
        sampled_ptr = (
            tl.load(graph_input_addresses_ptr + 6).to(tl.uint64).to(sampled_ptr.dtype)
        )
        rejected_steps_ptr = (
            tl.load(graph_input_addresses_ptr + 7)
            .to(tl.uint64)
            .to(rejected_steps_ptr.dtype)
        )

    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0

    accepted_length = tl.zeros((), tl.int64)
    target_lse = 0.0
    draft_lse = 0.0
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        # -1 is used for placeholder draft token ids that should be rejected.
        is_valid_draft = draft_sampled >= 0
        # Avoid possible OOB ptr access.
        draft_sampled = tl.maximum(0, draft_sampled)
        if not is_greedy:
            # A -1 placeholder ends verification. Greedy is excluded because it
            # stores the target argmax upon first rejection, so it rejects the
            # placeholder via `accepted` instead.
            verifying &= is_valid_draft
        if verifying:
            target_argmax = tl.full((), 0, tl.int64)
            pos = tl.load(pos_ptr + logit_idx)
            u = tl_rand32(seed, pos, includes_zero=False)
            if is_greedy:
                # Greedy sampling. Accept IFF draft matches target argmax.
                # NOTE: Target argmax is stored directly so that resampling
                # can be skipped upon rejection.
                target_argmax = _compute_global_target_argmax(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_argmax_ptr,
                    target_local_argmax_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    accepted = target_argmax == draft_sampled
                accepted &= is_valid_draft
            else:
                # Speculative decoding (Leviathan et al., 2023): https://arxiv.org/abs/2211.17192
                target_logprob, draft_logprob, target_lse, draft_lse = (
                    _compute_global_logprobs_and_logsumexp(
                        draft_sampled,
                        is_valid_draft,
                        logit_idx,
                        req_state_idx,
                        i,
                        temp,
                        target_logits_ptr,
                        target_logits_stride,
                        target_local_max_ptr,
                        target_local_max_stride,
                        target_local_sumexp_ptr,
                        target_local_sumexp_stride,
                        draft_logits_ptr,
                        draft_logits_stride_0,
                        draft_logits_stride_1,
                        draft_logits_row_mapping_ptr,
                        draft_local_max_ptr,
                        draft_local_max_stride,
                        draft_local_sumexp_ptr,
                        draft_local_sumexp_stride,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                        HAS_DRAFT_LOGITS,
                        DRAFT_VALUES_ARE_PROBS,
                    )
                )
                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    accepted = target_logprob > tl.log(u) + draft_logprob

            verifying = accepted
            accepted_length += accepted
            tl.store(
                sampled_ptr + req_idx * sampled_stride + i,
                target_argmax if is_greedy and not accepted else draft_sampled,
            )

    tl.store(rejected_steps_ptr + req_idx, accepted_length)
    tl.store(target_rejected_logsumexp_ptr + req_idx, target_lse)
    tl.store(draft_rejected_logsumexp_ptr + req_idx, draft_lse)


@triton.jit
def _seeded_resample_argmax(
    residual_logits,
    block,
    mask,
    resample_token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seed_ptr,
    pos_ptr,
    vocab_size,
    USE_FP64: tl.constexpr,
):
    return gumbel_block_argmax(
        residual_logits,
        block,
        mask,
        resample_token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        None,  # logits_cache_ptr
        0,  # logits_cache_stride_0
        0,  # logits_cache_stride_1
        None,  # logits_cache_col_ptr
        None,  # logits_cache_source_ptr
        0,  # logits_cache_source_stride
        None,  # logits_cache_row_mapping_ptr
        vocab_size,
        IS_DRAFTING=False,
        APPLY_TEMPERATURE=False,
        USE_FP64=USE_FP64,
    )


@triton.jit
def _resample_kernel(
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    # [num_logits, V]
    target_logits_ptr,
    target_logits_stride,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_reqs]
    rejected_step_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_FP64: tl.constexpr,
    SKIP_MASKED_BLOCKS: tl.constexpr,
    DRAFT_VALUES_ARE_PROBS: tl.constexpr,
    graph_input_addresses_ptr=None,
):
    if graph_input_addresses_ptr is not None:
        target_logits_ptr = (
            tl.load(graph_input_addresses_ptr).to(tl.uint64).to(target_logits_ptr.dtype)
        )
        rejected_step_ptr = (
            tl.load(graph_input_addresses_ptr + 7)
            .to(tl.uint64)
            .to(rejected_step_ptr.dtype)
        )

    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx).to(tl.int64)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. No resampling needed because
        # the target argmax is already in the sampled tensor.
        return

    rejected_draft_token = tl.load(
        draft_sampled_ptr + resample_token_idx + 1,
        mask=not is_bonus,
        other=0,
    )
    is_valid_rejected_draft = rejected_draft_token >= 0

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    if SKIP_MASKED_BLOCKS and tl.max(target_logits, axis=0) == float("-inf"):
        # No target token survives in this block (e.g. top-k/top-p masked it),
        # so neither does the residual: skip the draft load and the Gumbel draw.
        tl.store(
            resampled_local_argmax_ptr
            + req_idx * resampled_local_argmax_stride
            + block_idx,
            block_idx * BLOCK_SIZE,
        )
        tl.store(
            resampled_local_max_ptr + req_idx * resampled_local_max_stride + block_idx,
            float("-inf"),
        )
        return

    # Compute the residual logits to resample the rejected token from.
    if is_bonus or not is_valid_rejected_draft:
        # Bonus token (no rejections) or -1 placeholder token. In either case,
        # directly use the target logits.
        residual_logits = target_logits
    elif HAS_DRAFT_LOGITS:
        # draft_logits is stored pre-temperature, so apply scale first.
        draft_logits = tl.load(
            draft_logits_ptr
            + tl.maximum(
                draft_logits_row(req_state_idx, draft_logits_row_mapping_ptr), 0
            )
            * draft_logits_stride_0
            + resample_idx * draft_logits_stride_1
            + block,
            mask=mask,
            other=0.0 if DRAFT_VALUES_ARE_PROBS else float("-inf"),
        ).to(tl.float32)
        if DRAFT_VALUES_ARE_PROBS:
            # Match the dtype rounding of a materialized Tensor.log().
            draft_logits = (
                tldevice.log(draft_logits)
                .to(draft_logits_ptr.dtype.element_ty)
                .to(tl.float32)
            )
        draft_logits = draft_logits / temp
        target_lse = tl.load(target_rejected_logsumexp_ptr + req_idx)
        draft_lse = tl.load(draft_rejected_logsumexp_ptr + req_idx)
        target_log_probs = target_logits - target_lse
        draft_log_probs = draft_logits - draft_lse
        # Compute the residual:
        #   r(x) = max(p(x) - q(x), 0)
        # Gumbel sampling needs logits, so we compute it in log space:
        #   log(r(x)) = log(max(exp(log_p(x)) - exp(log_q(x)), 0))
        # The more numerically stable form is:
        #   log(max(exp(a) - exp(b), 0)) = a + log(max(1 - exp(b - a), 0))
        ratio = tl.exp(draft_log_probs - target_log_probs)
        residual_logits = tl.where(
            ratio < 1.0,
            target_log_probs + tldevice.log1p(-ratio),
            float("-inf"),
        ).to(tl.float32)
    else:
        # One-hot draft. The residual is just the target distribution with
        # the rejected draft token probability zeroed out.
        residual_logits = tl.where(
            block != rejected_draft_token,
            target_logits,
            float("-inf"),
        ).to(tl.float32)

    # Resample the rejected/bonus token.
    value, idx = _seeded_resample_argmax(
        residual_logits,
        block,
        mask,
        resample_token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        vocab_size,
        USE_FP64=USE_FP64,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + block_idx,
        token_id,
    )
    tl.store(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block_idx,
        value,
    )


@triton.jit
def _insert_resampled_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    resample_num_blocks,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    PADDED_RESAMPLE_NUM_BLOCKS: tl.constexpr,
    CLEAN_OUTPUT_ROWS: tl.constexpr = 0,
    graph_input_addresses_ptr=None,
):
    if graph_input_addresses_ptr is not None:
        sampled_ptr = (
            tl.load(graph_input_addresses_ptr + 6).to(tl.uint64).to(sampled_ptr.dtype)
        )
        num_sampled_ptr = (
            tl.load(graph_input_addresses_ptr + 7)
            .to(tl.uint64)
            .to(num_sampled_ptr.dtype)
        )

    req_idx = tl.program_id(0)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + num_sampled
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    # Increment the number of sampled tokens.
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)

    if CLEAN_OUTPUT_ROWS > 0:
        columns = tl.arange(0, triton.next_power_of_2(CLEAN_OUTPUT_ROWS))
        tl.store(
            sampled_ptr + req_idx * sampled_stride + columns,
            0,
            (columns > num_sampled) & (columns < CLEAN_OUTPUT_ROWS),
        )

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. The target argmax is already
        # in the sampled tensor.
        return

    # Insert the resampled token.
    block = tl.arange(0, PADDED_RESAMPLE_NUM_BLOCKS)
    mask = block < resample_num_blocks
    resampled_local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    # NaN max values (from NaN target logits) make tl.argmax return an
    # out-of-range block index (into the padded region), causing an OOB read
    # of resampled_local_argmax. Map NaN to -inf so argmax stays in range.
    resampled_local_max = tl.where(
        resampled_local_max != resampled_local_max,
        float("-inf"),
        resampled_local_max,
    )
    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)
    resampled = tl.load(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + resampled_max_block_idx,
    )
    tl.store(
        sampled_ptr + req_idx * sampled_stride + num_sampled,
        resampled,
    )


def rejection_sample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    draft_logits_cache: DraftLogitsCache | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    # [num_speculative_steps]
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64: bool = False,
    *,
    skip_masked_vocab_blocks: bool = False,
    output_ids: torch.Tensor | None = None,
    output_lengths: torch.Tensor | None = None,
    graph_input_addresses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert target_logits.ndim == 2 and target_logits.stride(-1) == 1
    if graph_input_addresses is not None:
        # Only the point-mass Graph path redirects logits; all other consumers
        # continue to read the Tensor passed by the caller directly.
        assert draft_logits_cache is None
        assert target_logits.is_cuda and target_logits.dtype == torch.float32
        assert target_logits.is_contiguous()
        assert graph_input_addresses.device == target_logits.device
        assert graph_input_addresses.dtype in (torch.int64, torch.uint64)
        assert graph_input_addresses.shape == (8,)
        assert graph_input_addresses.is_contiguous()
        assert output_ids is not None and output_lengths is not None
        assert output_ids.is_contiguous() and output_lengths.is_contiguous()
    draft_logits = None
    draft_logits_row_mapping = None
    draft_values_are_probs = False
    if draft_logits_cache is not None:
        draft_logits = draft_logits_cache.values
        draft_logits_row_mapping = draft_logits_cache.row_mapping
        draft_values_are_probs = draft_logits_cache.values_are_probs
        if draft_values_are_probs and draft_logits.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            draft_logits = draft_logits.log()
            draft_values_are_probs = False
    assert draft_logits is None or (
        draft_logits.ndim == 3 and draft_logits.stride(-1) == 1
    )
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape

    draft_logits_stride_0 = 0
    draft_logits_stride_1 = 0
    if has_draft_logits := draft_logits is not None:
        draft_logits_stride_0 = draft_logits.stride(0)
        draft_logits_stride_1 = draft_logits.stride(1)
        # In some cases (e.g. MiMo v2.5 Pro + DFlash) the target model's
        # vocab size is larger than the draft's due to padding.
        vocab_size = min(vocab_size, draft_logits.size(-1))

    # Compute the per-vocab-block logits stats, such as target argmax
    # (for greedy requests), and target max + softmax exponential
    # (for non-greedy requests).
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    target_local_argmax = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.int64
    )
    target_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    target_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    # Separate target and draft CTAs keep probability conversion off the target path.
    stats_grid = (
        num_logits,
        vocab_num_blocks,
        2 if has_draft_logits and draft_values_are_probs else 1,
    )
    _compute_local_logits_stats_kernel[stats_grid](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_logits_row_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_logits,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        DRAFT_VALUES_ARE_PROBS=draft_values_are_probs,
        graph_input_addresses_ptr=graph_input_addresses,
    )

    # Sample up until the first rejected/bonus token, and store
    # the step.
    direct_output = output_ids is not None
    if direct_output:
        assert output_lengths is not None
        assert output_ids.shape == (num_reqs, num_speculative_steps + 1)
        assert output_ids.dtype == torch.int32 and output_ids.stride(1) == 1
        assert output_lengths.shape == (num_reqs,)
        assert output_lengths.dtype == torch.int32 and output_lengths.stride(0) == 1
        assert output_ids.device == output_lengths.device == target_logits.device
        sampled = output_ids
        num_sampled = output_lengths
    else:
        assert output_lengths is None
        sampled = draft_sampled.new_empty(
            num_reqs, num_speculative_steps + 1, dtype=torch.int64
        )
        num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    _rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        target_logits,
        target_logits.stride(0),
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_logits_row_mapping,
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        synthetic_conditional_rates,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=has_draft_logits,
        DRAFT_VALUES_ARE_PROBS=draft_values_are_probs,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        num_warps=1,
        graph_input_addresses_ptr=graph_input_addresses,
    )

    # Resample the rejected/bonus tokens.
    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.int64
    )
    resampled_local_max = target_logits.new_empty(
        num_reqs,
        resample_num_blocks,
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    _resample_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_rejected_logsumexp,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_logits_row_mapping,
        draft_rejected_logsumexp,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        DRAFT_VALUES_ARE_PROBS=draft_values_are_probs,
        USE_FP64=use_fp64,
        SKIP_MASKED_BLOCKS=skip_masked_vocab_blocks,
        graph_input_addresses_ptr=graph_input_addresses,
    )

    # Insert the resampled tokens into the output sampled.
    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
        CLEAN_OUTPUT_ROWS=num_speculative_steps + 1 if direct_output else 0,
        graph_input_addresses_ptr=graph_input_addresses,
    )
    return sampled, num_sampled
