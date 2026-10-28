"""Adapt backend frames without depending on the serving metrics collector."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtp_llm.utils.base_model_datatypes import GenerateOutputs


def make_backend_metric_observer(observer, context_batch_size=1, unit_id=0):
    if observer is None:
        return None

    def observe(output: GenerateOutputs, attempt: int) -> None:
        observer(
            {
                "aux_info": [item.aux_info for item in output.generate_outputs],
                "_frontend_context_batch_size": context_batch_size,
                "_frontend_output_batch_size": len(output.generate_outputs),
                "_frontend_metric_unit_id": unit_id,
                "_frontend_metric_attempt": attempt,
                "context_token_num": output.frontend_context_token_num,
                "context_token_num_with_cache": output.frontend_context_token_num_with_cache,
                "context_execute_time_us": output.frontend_context_execute_time_us,
                "context_execute_time_with_cache_us": output.frontend_context_execute_time_with_cache_us,
                "generate_token_num": output.frontend_generate_token_num,
                "generate_execute_time_us": output.frontend_generate_execute_time_us,
                "frontend_sp_tpot_samples": output.frontend_sp_tpot_samples,
            }
        )

    return observe


def metric_backend_generate_config(generate_config, enabled):
    return (
        generate_config.model_copy(
            update={"frontend_metric_streaming": True, "aux_info": True}, deep=True
        )
        if enabled
        else generate_config
    )
