#include "rtp_llm/cpp/models/NanCheckRunner.h"

#include "rtp_llm/models_py/bindings/common/kernels/nan_check_torch_op.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

KvCacheNanCheckRunner::KvCacheNanCheckRunner(DataType      cache_dtype,
                                             size_t        layer_num,
                                             torch::Tensor layer_base_addr_buffer,
                                             int64_t       seq_size_per_block,
                                             int64_t       local_head_num_kv,
                                             int64_t       k_token_size,
                                             int64_t       v_token_size,
                                             int64_t       k_block_size_bytes,
                                             int64_t       v_block_size_bytes,
                                             int64_t       k_token_bytes,
                                             int64_t       v_token_bytes,
                                             int64_t       max_batch_size):
    cache_dtype_(cache_dtype),
    layer_num_(layer_num),
    layer_base_addr_buffer_(std::move(layer_base_addr_buffer)),
    seq_size_per_block_(seq_size_per_block),
    local_head_num_kv_(local_head_num_kv),
    k_token_size_(k_token_size),
    v_token_size_(v_token_size),
    k_block_size_bytes_(k_block_size_bytes),
    v_block_size_bytes_(v_block_size_bytes),
    k_token_bytes_(k_token_bytes),
    v_token_bytes_(v_token_bytes) {
    if (max_batch_size > 0) {
        auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        prefix_lengths_d_ = torch::empty({max_batch_size}, opts);
        input_lengths_d_  = torch::empty({max_batch_size}, opts);
    }
}

std::unique_ptr<KvCacheNanCheckRunner> KvCacheNanCheckRunner::create(const AttentionConfigs& attention_config,
                                                                     DataType                cache_dtype,
                                                                     size_t                  cache_element_size,
                                                                     size_t                  layer_num,
                                                                     torch::Tensor           layer_base_addr_buffer,
                                                                     int64_t                 max_batch_size) {
    if (cache_dtype == DataType::TYPE_INT8 || cache_dtype == DataType::TYPE_INVALID || cache_element_size == 0) {
        return nullptr;
    }
    if (!layer_base_addr_buffer.defined() || layer_base_addr_buffer.size(0) != static_cast<int64_t>(layer_num)) {
        return nullptr;
    }
    if (attention_config.use_mla
        && (attention_config.is_sparse
            || cache_dtype == DataType::TYPE_FP8_E4M3 || cache_dtype == DataType::TYPE_FP8_E8M0)) {
        return nullptr;
    }

    int64_t k_token_size, v_token_size, local_head_num_kv;
    if (attention_config.use_mla) {
        k_token_size      = attention_config.kv_lora_rank;
        v_token_size      = attention_config.rope_head_dim;
        local_head_num_kv = 1;
    } else {
        k_token_size      = attention_config.size_per_head;
        v_token_size      = attention_config.size_per_head;
        local_head_num_kv = attention_config.kv_head_num;
    }

    const int64_t seq_size_per_block = attention_config.tokens_per_block;
    const int64_t k_block_size_bytes = local_head_num_kv * k_token_size * seq_size_per_block * cache_element_size;
    const int64_t v_block_size_bytes = local_head_num_kv * v_token_size * seq_size_per_block * cache_element_size;
    const int64_t k_token_bytes      = k_token_size * cache_element_size;
    const int64_t v_token_bytes      = v_token_size * cache_element_size;

    return std::unique_ptr<KvCacheNanCheckRunner>(new KvCacheNanCheckRunner(cache_dtype,
                                                                            layer_num,
                                                                            std::move(layer_base_addr_buffer),
                                                                            seq_size_per_block,
                                                                            local_head_num_kv,
                                                                            k_token_size,
                                                                            v_token_size,
                                                                            k_block_size_bytes,
                                                                            v_block_size_bytes,
                                                                            k_token_bytes,
                                                                            v_token_bytes,
                                                                            max_batch_size));
}

void KvCacheNanCheckRunner::cacheDeviceCopies(const GptModelInputs& inputs) {
    if (device_copies_cached_)
        return;

    if (inputs.kv_cache_layer_to_group.defined()
        && inputs.kv_cache_layer_to_group.size(0) >= static_cast<int64_t>(layer_num_)) {
        layer_to_group_d_ = inputs.kv_cache_layer_to_group.cuda();
    }
    if (inputs.kv_cache_group_types.defined()) {
        group_types_d_ = inputs.kv_cache_group_types.cuda();
    }

    device_copies_cached_ = true;
}

bool KvCacheNanCheckRunner::run(const GptModelInputs& inputs, torch::Tensor& nan_flag) {
#if !USING_CUDA && !USING_ROCM
    return false;
#else
    if (!inputs.kv_cache_block_id.defined() || !inputs.input_lengths.defined()) {
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

    const int64_t block_size_bytes = static_cast<int64_t>(inputs.kv_block_stride_bytes);

    cacheDeviceCopies(inputs);

    auto layer_to_group_opt =
        layer_to_group_d_.defined() ? std::optional<torch::Tensor>(layer_to_group_d_) : std::nullopt;
    auto group_types_opt =
        (group_types_d_.defined() && group_types_d_.size(0) >= num_groups) ?
            std::optional<torch::Tensor>(group_types_d_) : std::nullopt;

    bool did_run = false;

    if (decoder_batch_size > 0) {
        check_and_reset_nan_kv_cache_decode(layer_base_addr_buffer_,
                                            inputs.kv_cache_block_id,
                                            inputs.sequence_lengths.slice(0, 0, decoder_batch_size),
                                            nan_flag.slice(0, 0, decoder_batch_size),
                                            static_cast<int64_t>(cache_dtype_),
                                            decoder_batch_size,
                                            static_cast<int64_t>(layer_num_),
                                            num_groups,
                                            layer_to_group_opt,
                                            group_types_opt,
                                            batch_dim,
                                            0,
                                            max_blocks_per_batch,
                                            local_head_num_kv_,
                                            k_token_size_,
                                            v_token_size_,
                                            k_block_size_bytes_,
                                            v_block_size_bytes_,
                                            k_token_bytes_,
                                            v_token_bytes_,
                                            block_size_bytes,
                                            seq_size_per_block_);
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

        auto prefix_host = inputs.prefix_lengths.slice(0, 0, context_batch_size);
        torch::Tensor prefix_lengths_dev;
        if (prefix_lengths_d_.defined() && prefix_lengths_d_.size(0) >= context_batch_size) {
            prefix_lengths_d_.slice(0, 0, context_batch_size).copy_(prefix_host, /*non_blocking=*/true);
            prefix_lengths_dev = prefix_lengths_d_.slice(0, 0, context_batch_size);
        } else {
            prefix_lengths_dev = prefix_host.cuda();
        }

        auto input_host = inputs.input_lengths.slice(0, decoder_batch_size, decoder_batch_size + context_batch_size);
        torch::Tensor input_lengths_dev;
        if (input_lengths_d_.defined() && input_lengths_d_.size(0) >= context_batch_size) {
            input_lengths_d_.slice(0, 0, context_batch_size).copy_(input_host, /*non_blocking=*/true);
            input_lengths_dev = input_lengths_d_.slice(0, 0, context_batch_size);
        } else {
            input_lengths_dev = input_host.cuda();
        }

        check_and_reset_nan_kv_cache_prefill(
            layer_base_addr_buffer_,
            inputs.kv_cache_block_id,
            prefix_lengths_dev,
            input_lengths_dev,
            nan_flag.slice(0, decoder_batch_size, decoder_batch_size + context_batch_size),
            static_cast<int64_t>(cache_dtype_),
            context_batch_size,
            static_cast<int64_t>(layer_num_),
            num_groups,
            layer_to_group_opt,
            group_types_opt,
            batch_dim,
            decoder_batch_size,
            max_blocks_per_batch,
            local_head_num_kv_,
            k_token_size_,
            v_token_size_,
            k_block_size_bytes_,
            v_block_size_bytes_,
            k_token_bytes_,
            v_token_bytes_,
            block_size_bytes,
            seq_size_per_block_);
        did_run = true;
    }

    return did_run;
#endif  // !USING_CUDA && !USING_ROCM
}

}  // namespace rtp_llm
