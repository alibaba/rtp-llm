import os

import torch
from rtp_llm.models_py.sampling.vllm_gumbel import gumbel_sample


def sample(
    logits: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    batch_size = logits.shape[0]
    mapping = torch.arange(batch_size, device=logits.device, dtype=torch.int32)
    temperature = torch.ones(batch_size, device=logits.device, dtype=torch.float32)
    return gumbel_sample(
        logits=logits,
        expanded_idx_mapping=mapping,
        temperature=temperature,
        seed=seeds,
        pos=positions,
        apply_temperature=False,
        is_drafting=False,
        use_fp64=False,
    )


def rejection_sample_logits(
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor | None,
    draft_ids: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    do_sample: torch.Tensor,
    force_accept: list[bool] | None = None,
    acceptance_cap: torch.Tensor | None = None,
    cum_log_probs: torch.Tensor | None = None,
    output_ids: torch.Tensor | None = None,
    output_lengths: torch.Tensor | None = None,
    clean_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        clean_output
        and draft_probs is None
        and cum_log_probs is None
        and output_ids is not None
        and output_lengths is not None
    ):
        from rtp_llm.models_py.sampling.rejection_graph import try_replay

        if try_replay(
            _rejection_sample_logits_eager,
            target_logits,
            draft_ids,
            seeds,
            positions,
            do_sample,
            force_accept,
            acceptance_cap,
            output_ids,
            output_lengths,
        ):
            return output_ids, output_lengths
    return _rejection_sample_logits_eager(
        target_logits,
        draft_probs,
        draft_ids,
        seeds,
        positions,
        do_sample,
        force_accept,
        acceptance_cap,
        cum_log_probs,
        output_ids,
        output_lengths,
        clean_output,
    )


def _rejection_sample_logits_eager(
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor | None,
    draft_ids: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    do_sample: torch.Tensor,
    force_accept: list[bool] | None = None,
    acceptance_cap: torch.Tensor | None = None,
    cum_log_probs: torch.Tensor | None = None,
    output_ids: torch.Tensor | None = None,
    output_lengths: torch.Tensor | None = None,
    clean_output: bool = False,
    graph_input_addresses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from rtp_llm.models_py.sampling.draft_logits import DraftLogitsCache
    from rtp_llm.models_py.sampling.rejection_metadata import (
        RejectionMetadata,
        finalize_rejection_output,
    )
    from rtp_llm.models_py.sampling.vllm_rejection import rejection_sample as verify

    batch, rows, vocab = target_logits.shape
    steps = rows - 1
    if output_ids is None:
        output_ids = target_logits.new_empty((batch, rows), dtype=torch.int32)
    if output_lengths is None:
        output_lengths = target_logits.new_empty((batch,), dtype=torch.int32)
    if batch == 0:
        return output_ids, output_lengths
    forced = force_accept or [False] * batch
    assert len(forced) == batch
    if acceptance_cap is not None:
        acceptance_cap = acceptance_cap.to(device=target_logits.device)
    fuse_draft_probs_log = os.environ.get("RTP_LLM_FUSE_DRAFT_PROBS_LOG", "1") == "1"
    draft_logits = draft_probs
    if draft_probs is not None and not fuse_draft_probs_log:
        draft_logits = draft_probs.log()
    max_requests = max(1, (1 << 30) // (rows * vocab * 4))
    if graph_input_addresses is not None:
        assert draft_probs is None and cum_log_probs is None and clean_output
        assert batch <= max_requests and all(value == forced[0] for value in forced)
    cumulative_chunks: list[torch.Tensor] = []
    initial_cumulative = (
        cum_log_probs.reshape(batch, rows)[:, 0].to(target_logits.device)
        if cum_log_probs is not None
        else None
    )
    start = 0
    while start < batch:
        end = start + 1
        while (
            end < batch and end - start < max_requests and forced[end] == forced[start]
        ):
            end += 1
        count = end - start
        metadata = RejectionMetadata.prepare(
            draft_ids[start:end],
            positions[start:end],
            do_sample[start:end],
            acceptance_cap[start:end] if acceptance_cap is not None else None,
            graph_input_addresses=graph_input_addresses,
            seed_output=seeds[start:end] if graph_input_addresses is not None else None,
        )
        cache = (
            DraftLogitsCache(
                draft_logits[start:end], values_are_probs=fuse_draft_probs_log
            )
            if draft_logits is not None
            else None
        )
        output, lengths = verify(
            target_logits[start:end].reshape(count * rows, vocab),
            cache,
            metadata.packed_draft,
            metadata.cumulative,
            metadata.absolute_positions,
            metadata.mapping,
            metadata.expanded_mapping,
            metadata.local_positions,
            metadata.temperature,
            seeds[start:end],
            steps,
            metadata.rates if forced[start] else None,
            skip_masked_vocab_blocks=True,
            output_ids=output_ids[start:end] if clean_output else None,
            output_lengths=output_lengths[start:end] if clean_output else None,
            graph_input_addresses=graph_input_addresses,
        )
        if initial_cumulative is not None:
            valid = metadata.local_positions.reshape(count, rows) < lengths.unsqueeze(1)
            safe_ids = torch.where(valid, output, 0).clamp(0, vocab - 1).long()
            chosen = (
                target_logits[start:end].gather(2, safe_ids.unsqueeze(2)).squeeze(2)
            )
            log_probs = chosen - target_logits[start:end].logsumexp(dim=2)
            log_probs = torch.where(valid, log_probs, 0)
            cumulative_chunks.append(
                initial_cumulative[start:end].unsqueeze(1) + log_probs.cumsum(dim=1)
            )
        if not clean_output:
            finalize_rejection_output(
                output, lengths, output_ids[start:end], output_lengths[start:end]
            )
        start = end
    if cum_log_probs is not None:
        cum_log_probs.copy_(
            torch.cat(cumulative_chunks).reshape(-1).to(cum_log_probs.device)
        )
    return output_ids, output_lengths
