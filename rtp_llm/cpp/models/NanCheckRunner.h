#pragma once

#include "rtp_llm/cpp/core/OpData.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

#include <memory>
#include <torch/extension.h>

namespace rtp_llm {

/// Stateful runner that checks for NaN/Inf in KV cache and resets them to zero.
/// Pre-computes invariant KV geometry at construction and reuses device buffers
/// across calls to avoid per-forward GPU memory allocation.
class KvCacheNanCheckRunner {
public:
    static std::unique_ptr<KvCacheNanCheckRunner> create(const AttentionConfigs& attention_config,
                                                         DataType                cache_dtype,
                                                         size_t                  cache_element_size,
                                                         size_t                  layer_num,
                                                         torch::Tensor           layer_base_addr_buffer,
                                                         int64_t                 max_batch_size = 0);

    bool run(const GptModelInputs& inputs, torch::Tensor& nan_flag);

private:
    KvCacheNanCheckRunner(DataType      cache_dtype,
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
                          int64_t       max_batch_size);

    void cacheDeviceCopies(const GptModelInputs& inputs);

    DataType      cache_dtype_;
    size_t        layer_num_;
    torch::Tensor layer_base_addr_buffer_;

    int64_t seq_size_per_block_;
    int64_t local_head_num_kv_;
    int64_t k_token_size_;
    int64_t v_token_size_;
    int64_t k_block_size_bytes_;
    int64_t v_block_size_bytes_;
    int64_t k_token_bytes_;
    int64_t v_token_bytes_;

    torch::Tensor prefix_lengths_d_;
    torch::Tensor input_lengths_d_;

    torch::Tensor layer_to_group_d_;
    torch::Tensor group_types_d_;
    bool          device_copies_cached_ = false;
};

}  // namespace rtp_llm
