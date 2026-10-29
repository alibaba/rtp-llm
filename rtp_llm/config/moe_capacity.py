"""Physical token bounds for the forwards sharing a model's MoE buffers."""

from typing import TYPE_CHECKING

from rtp_llm.device.device_type import is_cuda
from rtp_llm.ops import ParallelismConfig, RoleType, SpeculativeType, TaskType
from rtp_llm.utils.token_capacity import align_up, sp_padded_tokens

if TYPE_CHECKING:
    from rtp_llm.config.engine_config import EngineConfig
    from rtp_llm.config.kv_cache_config import KVCacheConfig
    from rtp_llm.config.model_config import ModelConfig


def _prefill_tokens(length: int, rows: int, parallelism: ParallelismConfig) -> int:
    # Query CP pads each sequence before splitting; pure-CP MoE can gather
    # those shards. Page-RR KV sharding alone does not pad queries this way.
    if parallelism.prefill_cp_config.is_enabled():
        length = align_up(length, 2 * parallelism.tp_size)
    tokens = length * rows
    # PyWrappedModel::padSequenceParallelInputs pads the packed batch for K3.
    return align_up(tokens, parallelism.tp_size) if parallelism.enable_sp else tokens


def resolve_moe_prefill_capacity(engine: "EngineConfig", model: "ModelConfig") -> int:
    """Return global physical rows; SP consumers derive their local shard size."""
    runtime = engine.runtime_config
    scheduler = runtime.fifo_scheduler_config
    chunk = int(scheduler.prefill_chunk_size)
    # Embedding/reranker engines do not execute through FIFO chunk grants.
    if chunk <= 0 or model.task_type != TaskType.LANGUAGE_MODEL:
        return min(
            int(scheduler.max_context_batch_size) * int(model.max_seq_len),
            int(scheduler.max_batch_tokens_size),
        )

    parallelism = engine.parallelism_config
    # FIFO's singleton admission bypasses max_batch_tokens_size; only the chunk
    # grant bounds execution in that case. Do not clamp this to the batch cap.
    # num_return_sequences is not capped by max_context_batch_size. A chunk of
    # one-token sequences is the worst per-sequence CP padding under this budget.
    capacity = _prefill_tokens(1, chunk, parallelism)
    speculative = engine.sp_config.type != SpeculativeType.NONE
    width = int(engine.sp_config.gen_num_per_cycle) + 1 if speculative else 1

    if (
        runtime.warm_up
        and runtime.warm_up_with_loss
        and not model.mm_model_config.is_multimodal
        and not parallelism.ffn_disaggregate_config.enable_ffn_disaggregate
        and is_cuda()
    ):
        # NormalEngine::getWarmUpInputLength / prefillWarmUp bypass chunking.
        reserve = (
            3 * int(engine.sp_config.gen_num_per_cycle)
            if engine.sp_config.type == SpeculativeType.DSPARK
            else width
        )
        length = int(model.max_seq_len) - max(reserve, 1)
        rows = int(scheduler.max_context_batch_size)
        capacity = max(capacity, _prefill_tokens(length, rows, parallelism))

    hw = engine.hw_kernel_config
    if hw.enable_cuda_graph:
        # Decode/verify capture buckets count requests; draft commit has the
        # same maximum width. SP must pad complete requests, not just tokens.
        batch = max(
            [int(runtime.max_generate_batch_size), *hw.decode_capture_batch_sizes]
        )
        widths = {1, width}
        if engine.sp_config.type == SpeculativeType.DSPARK:
            widths.add(
                int(engine.sp_config.gen_num_per_cycle)
                + int(not engine.sp_config.sp_dspark_sample_from_anchor)
            )
        for query_width in widths:
            graph_tokens = sp_padded_tokens(
                batch, query_width, parallelism.tp_size if parallelism.enable_sp else 1
            )
            # Unlike prefill queries, decode rows are replicated across CP
            # ranks; a pure-CP router gathers them before executing experts.
            if parallelism.prefill_cp_config.is_enabled():
                graph_tokens *= parallelism.tp_size
            capacity = max(capacity, graph_tokens)
        if (
            engine.pd_sep_config.role_type == RoleType.PDFUSION
            and not speculative
            and hw.generation_prefill_capture_token_buckets
        ):
            # Generation-prefill buckets already count the whole batch's tokens.
            capacity = max(capacity, max(hw.generation_prefill_capture_token_buckets))
    return capacity


def include_system_prompt_capacity(
    model: "ModelConfig", kv_cache: "KVCacheConfig", parallelism: ParallelismConfig
) -> None:
    # Called after tokenization, before Python model construction allocates MoE.
    # SystemPromptConstructor executes each resident prompt separately via preRun.
    if model.moe_prefill_max_tokens_per_rank is not None:
        longest = max(map(len, kv_cache.multi_task_prompt_tokens.values()), default=0)
        model.moe_prefill_max_tokens_per_rank = max(
            model.moe_prefill_max_tokens_per_rank,
            _prefill_tokens(longest, 1, parallelism),
        )
