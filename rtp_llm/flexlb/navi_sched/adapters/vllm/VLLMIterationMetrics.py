def record_iteration_tokens(logger, scheduler_stats, iteration_stats, engine_idx):
    if logger.engine_adapter.scheduler_iteration_metrics_enabled():
        tokens = (
            scheduler_stats.num_iteration_tokens
            if scheduler_stats is not None
            else None
        )
    else:
        tokens = (
            iteration_stats.num_prompt_tokens + iteration_stats.num_generation_tokens
            if iteration_stats is not None
            else None
        )
    if tokens is not None:
        logger.histogram_iteration_tokens[engine_idx].observe(tokens)
