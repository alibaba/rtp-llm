#include "rtp_llm/cpp/models/NanCheckRunner.h"

#include "rtp_llm/models_py/bindings/common/kernels/nan_check_torch_op.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

bool KvCacheNanCheckRunner::run(const AttentionConfigs& attention_config,
                                DataType                cache_dtype,
                                size_t                  cache_element_size,
                                size_t                  layer_num,
                                const torch::Tensor&    layer_base_addr_buffer,
                                const GptModelInputs&   inputs,
                                torch::Tensor&          nan_flag) {
#if !USING_CUDA && !USING_ROCM
    return false;
#else
    if (!inputs.kv_cache_block_id.defined()) {
        return false;
    }
    if (!inputs.input_lengths.defined()) {
        return false;
    }
    if (cache_dtype == DataType::TYPE_INT8 || cache_dtype == DataType::TYPE_INVALID || cache_element_size == 0) {
        return false;
    }

    auto block_id_sizes = inputs.kv_cache_block_id.sizes();
    if (block_id_sizes.size() != 3) {
        RTP_LLM_LOG_WARNING("KvCacheNanCheckRunner: kv_cache_block_id must be 3D [G,B,M], got %zu dims",
                            block_id_sizes.size());
        return false;
    }
    const int64_t num_groups           = block_id_sizes[0];
    const int64_t batch_dim            = block_id_sizes[1];
    const int64_t max_blocks_per_batch = block_id_sizes[2];

    int64_t decoder_batch_size = inputs.sequence_lengths.defined() ? inputs.sequence_lengths.size(0) : 0;
    int64_t context_batch_size = inputs.input_lengths.size(0) - decoder_batch_size;

    if (attention_config.use_mla
        && (attention_config.is_sparse
            || cache_dtype == DataType::TYPE_FP8_E4M3 || cache_dtype == DataType::TYPE_FP8_E8M0)) {
        RTP_LLM_LOG_DEBUG("skip KvCacheNanCheckRunner for sparse or FP8 MLA KV layout");
        return false;
    }

    const int64_t seq_size_per_block = attention_config.tokens_per_block;

    int64_t k_token_size      = 0;
    int64_t v_token_size      = 0;
    int64_t local_head_num_kv = 0;

    if (attention_config.use_mla) {
        k_token_size      = attention_config.kv_lora_rank;
        v_token_size      = attention_config.rope_head_dim;
        local_head_num_kv = 1;
    } else {
        k_token_size      = attention_config.size_per_head;
        v_token_size      = attention_config.size_per_head;
        local_head_num_kv = attention_config.kv_head_num;
    }

    const int64_t k_block_size_bytes = local_head_num_kv * k_token_size * seq_size_per_block * cache_element_size;
    const int64_t v_block_size_bytes = local_head_num_kv * v_token_size * seq_size_per_block * cache_element_size;
    const int64_t k_token_bytes      = k_token_size * cache_element_size;
    const int64_t v_token_bytes      = v_token_size * cache_element_size;
    // Use CacheConfig stride directly; for FP8 MLA it includes embedded scales that K+V alone miss.
    const int64_t block_size_bytes   = static_cast<int64_t>(inputs.kv_block_stride_bytes);

    if (!layer_base_addr_buffer.defined() || layer_base_addr_buffer.size(0) != static_cast<int64_t>(layer_num)) {
        RTP_LLM_LOG_WARNING("KvCacheNanCheckRunner: layer_base_addr_buffer invalid. Expected %zu, got %ld",
                            layer_num,
                            layer_base_addr_buffer.defined() ? layer_base_addr_buffer.size(0) : -1L);
        return false;
    }

    auto layer_to_group_opt = inputs.kv_cache_layer_to_group.defined()
                                      && inputs.kv_cache_layer_to_group.size(0) >= static_cast<int64_t>(layer_num) ?
                                  std::optional<torch::Tensor>(inputs.kv_cache_layer_to_group.cuda()) :
                                  std::nullopt;
    auto group_types_opt = inputs.kv_cache_group_types.defined() && inputs.kv_cache_group_types.size(0) >= num_groups ?
                               std::optional<torch::Tensor>(inputs.kv_cache_group_types.cuda()) :
                               std::nullopt;

    bool did_run = false;

    if (decoder_batch_size > 0) {
        check_and_reset_nan_kv_cache_decode(layer_base_addr_buffer,
                                            inputs.kv_cache_block_id,
                                            inputs.sequence_lengths.slice(0, 0, decoder_batch_size),
                                            nan_flag.slice(0, 0, decoder_batch_size),
                                            static_cast<int64_t>(cache_dtype),
                                            decoder_batch_size,
                                            static_cast<int64_t>(layer_num),
                                            num_groups,
                                            layer_to_group_opt,
                                            group_types_opt,
                                            batch_dim,
                                            0,
                                            max_blocks_per_batch,
                                            local_head_num_kv,
                                            k_token_size,
                                            v_token_size,
                                            k_block_size_bytes,
                                            v_block_size_bytes,
                                            k_token_bytes,
                                            v_token_bytes,
                                            block_size_bytes,
                                            seq_size_per_block);
        did_run = true;
    }

    if (context_batch_size > 0 && inputs.prefix_lengths.defined() && inputs.input_lengths.defined()) {
        RTP_LLM_CHECK_WITH_INFO(inputs.prefix_lengths.size(0) >= context_batch_size,
                                "prefix_lengths size %ld is smaller than context batch size %ld.",
                                inputs.prefix_lengths.size(0),
                                context_batch_size);
        RTP_LLM_CHECK_WITH_INFO(inputs.input_lengths.size(0) >= decoder_batch_size + context_batch_size,
                                "input_lengths size %ld is smaller than total batch size %ld.",
                                inputs.input_lengths.size(0),
                                decoder_batch_size + context_batch_size);

        auto prefix_lengths_t = inputs.prefix_lengths.slice(0, 0, context_batch_size).cuda();
        auto input_lengths_t =
            inputs.input_lengths.slice(0, decoder_batch_size, decoder_batch_size + context_batch_size).cuda();

        check_and_reset_nan_kv_cache_prefill(
            layer_base_addr_buffer,
            inputs.kv_cache_block_id,
            prefix_lengths_t,
            input_lengths_t,
            nan_flag.slice(0, decoder_batch_size, decoder_batch_size + context_batch_size),
            static_cast<int64_t>(cache_dtype),
            context_batch_size,
            static_cast<int64_t>(layer_num),
            num_groups,
            layer_to_group_opt,
            group_types_opt,
            batch_dim,
            decoder_batch_size,
            max_blocks_per_batch,
            local_head_num_kv,
            k_token_size,
            v_token_size,
            k_block_size_bytes,
            v_block_size_bytes,
            k_token_bytes,
            v_token_bytes,
            block_size_bytes,
            seq_size_per_block);
        did_run = true;
    }

    return did_run;
#endif  // !USING_CUDA && !USING_ROCM
}

}  // namespace rtp_llm
