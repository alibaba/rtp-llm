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
    /// Returns true when the cache configuration has both LINEAR and FULL groups,
    /// meaning block zeroing is required to prevent NaN propagation across group types.
    static bool needsBlockZero(const std::vector<CacheGroupType>& group_types);

    /// Factory: caller extracts layer pointers and config values — Runner never sees cache types.
    /// Returns nullptr if inputs are invalid.
    static std::unique_ptr<BlockZeroRunner> create(const std::vector<torch::Tensor>& layer_kv_buffer_ptrs,
                                                   size_t                            kv_block_stride_bytes,
                                                   size_t                            seq_size_per_block);

    void run(const GptModelInputs& inputs);

private:
    BlockZeroRunner(torch::Tensor layer_base_addr_buffer,
                    size_t        kv_block_stride_bytes,
                    size_t        seq_size_per_block,
                    size_t        layer_num);

    torch::Tensor layer_base_addr_buffer_;
    size_t        kv_block_stride_bytes_;
    size_t        seq_size_per_block_;
    size_t        layer_num_;
};

}  // namespace rtp_llm
