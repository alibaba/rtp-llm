#pragma once

#include "rtp_llm/cpp/core/OpData.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

#include <memory>
#include <vector>
#include <torch/extension.h>

namespace rtp_llm {

/// Zeros the last (potentially incomplete) KV cache block for each (batch, group, layer).
/// Derives which blocks to zero purely from GptModelInputs — no coupling to cache manager.
///
/// Only needed when mixed attention groups coexist (LINEAR + FULL), because a block
/// previously used by linear attention (fp32 states) may contain bit patterns that
/// appear as NaN/Inf when reinterpreted by full attention (fp16/bf16).
class BlockZeroRunner {
public:
    static bool needsBlockZero(const std::vector<CacheGroupType>& group_types);

    static std::unique_ptr<BlockZeroRunner> create(const std::vector<torch::Tensor>& layer_kv_buffer_ptrs,
                                                   size_t                            kv_block_stride_bytes,
                                                   size_t                            seq_size_per_block,
                                                   int64_t                           max_batch_size = 0);

    void run(const GptModelInputs& inputs);

private:
    BlockZeroRunner(torch::Tensor layer_base_addr_buffer,
                    size_t        kv_block_stride_bytes,
                    size_t        seq_size_per_block,
                    size_t        layer_num,
                    int64_t       max_batch_size);

    torch::Tensor layer_base_addr_buffer_;
    torch::Tensor token_counts_buffer_;
    torch::Tensor layer_to_group_d_;
    bool          layer_to_group_cached_ = false;
    size_t        kv_block_stride_bytes_;
    size_t        seq_size_per_block_;
    size_t        layer_num_;
};

}  // namespace rtp_llm
