#include "rtp_llm/cpp/models/NanCheckRunner.h"

#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/kernels/nan_check_torch_op.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

torch::Tensor stageInt32SliceToDevice(const BufferPtr&    buffer,
                                      size_t              offset,
                                      size_t              count,
                                      DeviceBase*         device,
                                      NanCheckStaging&    staging,
                                      const std::string&  tag) {
    if (buffer->where() == MemoryType::MEMORY_GPU) {
        return Buffer2torchTensor(*buffer, false).slice(0, offset, offset + count);
    }

    BufferPtr device_buffer =
        device->allocateBuffer({DataType::TYPE_INT32, {count}, AllocationType::DEVICE}, {tag});
    device->copy({*device_buffer, buffer->view(offset, count)});
    staging.hold(device_buffer);
    return Buffer2torchTensor(*device_buffer, false);
}

std::optional<torch::Tensor> stageOptionalInt32SliceToDevice(const BufferPtr&   buffer,
                                                             size_t             count,
                                                             DeviceBase*        device,
                                                             NanCheckStaging&   staging,
                                                             const std::string& tag) {
    if (!buffer) {
        return std::nullopt;
    }
    if (buffer->shape()[0] < count) {
        return std::nullopt;
    }
    return stageInt32SliceToDevice(buffer, 0, count, device, staging, tag);
}

}  // namespace

bool KvCacheNanCheckRunner::run(DeviceBase*             device,
                                const AttentionConfigs& attention_config,
                                DataType                cache_dtype,
                                size_t                  cache_element_size,
                                size_t                  layer_num,
                                const BufferPtr&        layer_base_addr_buffer,
                                const GptModelInputs&   inputs,
                                BufferPtr&              nan_flag,
                                NanCheckStaging&        staging) {
    if (!inputs.kv_cache_block_id) {
        return false;
    }
    if (!inputs.input_lengths) {
        return false;
    }
    if (cache_dtype == DataType::TYPE_INT8 || cache_dtype == DataType::TYPE_INVALID || cache_element_size == 0) {
        return false;
    }

    const auto& block_id_shape = inputs.kv_cache_block_id->shape();
    if (block_id_shape.size() != 3) {
        RTP_LLM_LOG_WARNING("KvCacheNanCheckRunner: kv_cache_block_id must be 3D [G,B,M], got %zu dims",
                            block_id_shape.size());
        return false;
    }
    const size_t num_groups           = block_id_shape[0];
    const size_t batch_dim            = block_id_shape[1];
    const size_t max_blocks_per_batch = block_id_shape[2];

    size_t decoder_batch_size = inputs.sequence_lengths ? inputs.sequence_lengths->shape()[0] : 0;
    size_t context_batch_size = inputs.input_lengths->shape()[0] - decoder_batch_size;

    if (attention_config.use_mla && attention_config.is_sparse) {
        RTP_LLM_LOG_DEBUG("skip KvCacheNanCheckRunner for sparse MLA KV layout");
        return false;
    }

    const size_t seq_size_per_block = attention_config.tokens_per_block;

    size_t k_block_size_bytes = 0;
    size_t v_block_size_bytes = 0;
    size_t k_token_size       = 0;
    size_t v_token_size       = 0;
    size_t local_head_num_kv  = 0;

    if (attention_config.use_mla) {
        k_token_size       = attention_config.kv_lora_rank;
        v_token_size       = attention_config.rope_head_dim;
        local_head_num_kv  = 1;
        k_block_size_bytes = local_head_num_kv * k_token_size * seq_size_per_block * cache_element_size;
        v_block_size_bytes = local_head_num_kv * v_token_size * seq_size_per_block * cache_element_size;
    } else {
        k_token_size       = attention_config.size_per_head;
        v_token_size       = attention_config.size_per_head;
        local_head_num_kv  = attention_config.kv_head_num;
        k_block_size_bytes = local_head_num_kv * k_token_size * seq_size_per_block * cache_element_size;
        v_block_size_bytes = local_head_num_kv * v_token_size * seq_size_per_block * cache_element_size;
    }

    const size_t k_token_bytes    = k_token_size * cache_element_size;
    const size_t v_token_bytes    = v_token_size * cache_element_size;
    const size_t block_size_bytes = k_block_size_bytes + v_block_size_bytes;

    if (!layer_base_addr_buffer) {
        return false;
    }
    if (layer_base_addr_buffer->shape()[0] != layer_num) {
        RTP_LLM_LOG_WARNING("KvCacheNanCheckRunner: layer_base_addr_buffer shape mismatch. Expected %zu, got %zu",
                            layer_num,
                            layer_base_addr_buffer->shape()[0]);
        return false;
    }

    torch::Tensor layer_base_addrs_t  = Buffer2torchTensor(*layer_base_addr_buffer, false);
    torch::Tensor kv_cache_block_id_t = Buffer2torchTensor(*inputs.kv_cache_block_id, false);
    torch::Tensor sequence_lengths_t  = Buffer2torchTensor(*inputs.sequence_lengths, false);
    torch::Tensor nan_flag_t          = Buffer2torchTensor(*nan_flag, false);

    auto layer_to_group_opt = stageOptionalInt32SliceToDevice(
        inputs.kv_cache_layer_to_group, layer_num, device, staging, "nan_check_layer_to_group");
    auto group_types_opt = stageOptionalInt32SliceToDevice(
        inputs.kv_cache_group_types, num_groups, device, staging, "nan_check_group_types");

    bool did_run = false;

    if (decoder_batch_size > 0) {
        check_and_reset_nan_kv_cache_decode(layer_base_addrs_t,
                                            kv_cache_block_id_t,
                                            sequence_lengths_t.slice(0, 0, decoder_batch_size),
                                            nan_flag_t.slice(0, 0, decoder_batch_size),
                                            static_cast<int64_t>(cache_dtype),
                                            static_cast<int64_t>(decoder_batch_size),
                                            static_cast<int64_t>(layer_num),
                                            static_cast<int64_t>(num_groups),
                                            layer_to_group_opt,
                                            group_types_opt,
                                            static_cast<int64_t>(batch_dim),
                                            0,
                                            static_cast<int64_t>(max_blocks_per_batch),
                                            static_cast<int64_t>(local_head_num_kv),
                                            static_cast<int64_t>(k_token_size),
                                            static_cast<int64_t>(v_token_size),
                                            static_cast<int64_t>(k_block_size_bytes),
                                            static_cast<int64_t>(v_block_size_bytes),
                                            static_cast<int64_t>(k_token_bytes),
                                            static_cast<int64_t>(v_token_bytes),
                                            static_cast<int64_t>(block_size_bytes),
                                            static_cast<int64_t>(seq_size_per_block));
        did_run = true;
    }

    if (context_batch_size > 0 && inputs.prefix_lengths && inputs.input_lengths) {
        RTP_LLM_CHECK_WITH_INFO(inputs.prefix_lengths->size() >= context_batch_size,
                                "prefix_lengths size %d is smaller than context batch size %d.",
                                inputs.prefix_lengths->size(),
                                context_batch_size);
        RTP_LLM_CHECK_WITH_INFO(inputs.input_lengths->size() >= decoder_batch_size + context_batch_size,
                                "input_lengths size %d is smaller than total batch size %d.",
                                inputs.input_lengths->size(),
                                decoder_batch_size + context_batch_size);
        RTP_LLM_CHECK_WITH_INFO(batch_dim >= decoder_batch_size + context_batch_size,
                                "kv_cache_block_id batch_dim %d is smaller than total batch size %d.",
                                batch_dim,
                                decoder_batch_size + context_batch_size);

        torch::Tensor prefix_lengths_t =
            stageInt32SliceToDevice(inputs.prefix_lengths, 0, context_batch_size, device, staging, "nan_check_prefix_lengths");
        torch::Tensor input_lengths_t = stageInt32SliceToDevice(inputs.input_lengths,
                                                                decoder_batch_size,
                                                                context_batch_size,
                                                                device,
                                                                staging,
                                                                "nan_check_input_lengths");

        check_and_reset_nan_kv_cache_prefill(layer_base_addrs_t,
                                             kv_cache_block_id_t,
                                             prefix_lengths_t,
                                             input_lengths_t,
                                             nan_flag_t.slice(0, decoder_batch_size, decoder_batch_size + context_batch_size),
                                             static_cast<int64_t>(cache_dtype),
                                             static_cast<int64_t>(context_batch_size),
                                             static_cast<int64_t>(layer_num),
                                             static_cast<int64_t>(num_groups),
                                             layer_to_group_opt,
                                             group_types_opt,
                                             static_cast<int64_t>(batch_dim),
                                             static_cast<int64_t>(decoder_batch_size),
                                             static_cast<int64_t>(max_blocks_per_batch),
                                             static_cast<int64_t>(local_head_num_kv),
                                             static_cast<int64_t>(k_token_size),
                                             static_cast<int64_t>(v_token_size),
                                             static_cast<int64_t>(k_block_size_bytes),
                                             static_cast<int64_t>(v_block_size_bytes),
                                             static_cast<int64_t>(k_token_bytes),
                                             static_cast<int64_t>(v_token_bytes),
                                             static_cast<int64_t>(block_size_bytes),
                                             static_cast<int64_t>(seq_size_per_block));
        did_run = true;
    }

    return did_run;
}

}  // namespace rtp_llm
