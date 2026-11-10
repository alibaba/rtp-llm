"""Point-mass verification on compact candidates using original-vocabulary RNG keys."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.sampling.vllm_gumbel import gumbel_noised_argmax, tl_rand32


@triton.jit
def _load_row(logits, ids, dense, fallback, row, block, DENSE_VOCAB: tl.constexpr):
    overflow = tl.load(fallback + row) != 0
    if overflow:
        keys = block
        valid = block < DENSE_VOCAB
        values = tl.load(dense + row * DENSE_VOCAB + block, valid, other=float("-inf"))
    else:
        valid = block < 1024
        keys = tl.load(ids + row * 1024 + block, valid, other=-1)
        values = tl.load(logits + row * 1024 + block, valid, other=float("-inf"))
        valid &= keys >= 0
    return values, keys, valid


@triton.jit
def _compact_rejection_kernel(
    addresses,
    float_ptr,
    int_ptr,
    long_ptr,
    bool_ptr,
    ROWS: tl.constexpr,
    DENSE_VOCAB: tl.constexpr,
    HAS_CAP: tl.constexpr,
    FORCED: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    logits = tl.load(addresses + 0).to(tl.uint64).to(float_ptr.dtype)
    drafts = tl.load(addresses + 1).to(tl.uint64).to(int_ptr.dtype)
    seeds = tl.load(addresses + 2).to(tl.uint64).to(long_ptr.dtype)
    positions = tl.load(addresses + 3).to(tl.uint64).to(long_ptr.dtype)
    sample = tl.load(addresses + 4).to(tl.uint64).to(bool_ptr.dtype)
    output = tl.load(addresses + 6).to(tl.uint64).to(int_ptr.dtype)
    lengths = tl.load(addresses + 7).to(tl.uint64).to(int_ptr.dtype)
    ids = tl.load(addresses + 8).to(tl.uint64).to(int_ptr.dtype)
    dense = tl.load(addresses + 9).to(tl.uint64).to(float_ptr.dtype)
    fallback = tl.load(addresses + 10).to(tl.uint64).to(int_ptr.dtype)
    seed, position = tl.load(seeds + req), tl.load(positions + req)
    random = tl.load(sample + req)
    cap = ROWS - 1
    if HAS_CAP:
        caps = tl.load(addresses + 5).to(tl.uint64).to(int_ptr.dtype)
        cap = tl.minimum(tl.maximum(tl.load(caps + req), 0), ROWS - 1)
    lane = tl.arange(0, 1024)
    accepted = tl.full((), 0, tl.int32)
    verifying = True
    rejected_argmax = tl.full((), 0, tl.int64)
    for step in range(ROWS - 1):
        if verifying:
            row = req * ROWS + step
            draft = tl.load(drafts + req * (ROWS - 1) + step)
            draft = tl.where(step < cap, draft, -1)
            overflow = tl.load(fallback + row) != 0
            blocks = tl.where(overflow, triton.cdiv(DENSE_VOCAB, 1024), 1)
            maximum = tl.full((), float("-inf"), tl.float32)
            sumexp = tl.full((), 0.0, tl.float32)
            argmax = tl.full((), 0x7FFFFFFFFFFFFFFF, tl.int64)
            draft_logit = tl.full((), float("-inf"), tl.float32)
            u = tl_rand32(seed, position + step, includes_zero=False)
            # Forced random verification needs no target statistics. Forced
            # greedy still needs recovery argmax when a cap/invalid draft stops
            # verification, because that token bypasses the resample stage.
            if not FORCED or (not random and ((draft < 0) | (u >= 1.0))):
                for block_idx in range(blocks):
                    values, keys, valid = _load_row(
                        logits,
                        ids,
                        dense,
                        fallback,
                        row,
                        block_idx * 1024 + lane,
                        DENSE_VOCAB,
                    )
                    local_max = tl.max(values, 0)
                    new_max = tl.maximum(maximum, local_max)
                    if random:
                        sumexp = tl.where(
                            new_max > float("-inf"),
                            sumexp * tl.exp(maximum - new_max)
                            + tl.sum(tl.exp(values - new_max), 0),
                            0.0,
                        )
                        draft_logit = tl.maximum(
                            draft_logit,
                            tl.max(
                                tl.where(
                                    valid & (keys == draft), values, float("-inf")
                                ),
                                0,
                            ),
                        )
                    else:
                        local_id = tl.min(
                            tl.where(
                                valid & (values == local_max),
                                keys.to(tl.int64),
                                0x7FFFFFFFFFFFFFFF,
                            ),
                            0,
                        )
                        argmax = tl.where(
                            local_max > maximum,
                            local_id,
                            tl.where(
                                local_max == maximum,
                                tl.minimum(argmax, local_id),
                                argmax,
                            ),
                        )
                    maximum = new_max
            argmax = tl.where(maximum > float("-inf"), argmax, 0)
            if FORCED:
                keep = u < 1.0
            elif random:
                keep = draft_logit - (maximum + tl.log(sumexp)) > tl.log(u)
            else:
                keep = argmax == draft
            keep &= draft >= 0
            if keep:
                tl.store(output + req * ROWS + step, draft)
                accepted += 1
            else:
                verifying = False
                rejected_argmax = argmax
    row = req * ROWS + accepted
    if random or accepted == ROWS - 1:
        rejected = tl.load(
            drafts + req * (ROWS - 1) + accepted, mask=accepted < ROWS - 1, other=-1
        )
        rejected = tl.where(accepted < cap, rejected, -1)
        overflow = tl.load(fallback + row) != 0
        blocks = tl.where(overflow, triton.cdiv(DENSE_VOCAB, 1024), 1)
        best = tl.full((), float("-inf"), tl.float32)
        chosen = tl.full((), 0x7FFFFFFFFFFFFFFF, tl.int64)
        for block_idx in range(blocks):
            values, keys, valid = _load_row(
                logits, ids, dense, fallback, row, block_idx * 1024 + lane, DENSE_VOCAB
            )
            residual = tl.where(
                (accepted == ROWS - 1) | (rejected < 0) | (keys != rejected),
                values,
                float("-inf"),
            )
            value, token = gumbel_noised_argmax(
                residual,
                keys,
                valid,
                seed,
                position + accepted,
                random.to(tl.float32),
                IS_DRAFTING=False,
                USE_FP64=False,
                APPLY_TEMPERATURE=False,
                RETURN_KEY=True,
            )
            value = tl.where(value == value, value, float("-inf"))
            chosen = tl.where(
                value > best,
                token,
                tl.where(value == best, tl.minimum(chosen, token), chosen),
            )
            best = tl.maximum(best, value)
        rejected_argmax = tl.where(best > float("-inf"), chosen, 0)
    tl.store(output + req * ROWS + accepted, rejected_argmax)
    columns = tl.arange(0, triton.next_power_of_2(ROWS))
    tl.store(output + req * ROWS + columns, 0, (columns > accepted) & (columns < ROWS))
    tl.store(lengths + req, accepted + 1)


def materialize(logits, ids, dense, fallback):
    expanded = torch.full_like(dense, -float("inf"))
    expanded.scatter_(-1, ids.long(), logits)
    return torch.where(fallback.reshape(*dense.shape[:-1], 1).bool(), dense, expanded)
