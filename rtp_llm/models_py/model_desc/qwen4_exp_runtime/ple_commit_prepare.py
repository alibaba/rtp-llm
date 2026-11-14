"""Adapt a validated PLE transaction stage to tensor preparation."""

from rtp_llm.models_py.kernels.cuda.qwen4_ple.commit_prepare import (
    prepare_commit_values as _prepare_commit_values,
)


def prepare_commit_values(stage, accepted, state_blocks, ctx_blocks):
    return _prepare_commit_values(
        stage.initial_state,
        stage.candidate_state_inputs,
        stage.initial_context,
        stage.candidate_ids,
        stage.state_pool,
        stage.ctx_pool,
        accepted,
        state_blocks,
        ctx_blocks,
    )
