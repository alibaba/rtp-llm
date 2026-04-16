#include "rtp_llm/cpp/models/BlockZeroRunner.h"
#include "rtp_llm/cpp/models/KvCacheAddrUtils.h"

#include "rtp_llm/models_py/bindings/common/kernels/block_zero_torch_op.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

bool BlockZeroRunner::needsBlockZero(const std::vector<CacheGroupType>& group_types) {
    bool has_linear = false, has_full = false;
    for (auto gt : group_types) {
        if (gt == CacheGroupType::LINEAR) has_linear = true;
        if (gt == CacheGroupType::FULL)   has_full   = true;
        if (has_linear && has_full) return true;
    }
    return false;
}

BlockZeroRunner::BlockZeroRunner(torch::Tensor layer_base_addr_buffer,
                                 size_t        kv_block_stride_bytes,
                                 size_t        seq_size_per_block,
                                 size_t        layer_num,
                                 int64_t       max_batch_size):
    layer_base_addr_buffer_(std::move(layer_base_addr_buffer)),
    kv_block_stride_bytes_(kv_block_stride_bytes),
    seq_size_per_block_(seq_size_per_block),
    layer_num_(layer_num) {
    if (max_batch_size > 0) {
        token_counts_buffer_ = torch::empty({max_batch_size},
                                            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }
}

std::unique_ptr<BlockZeroRunner> BlockZeroRunner::create(const std::vector<torch::Tensor>& layer_kv_buffer_ptrs,
                                                         size_t                            kv_block_stride_bytes,
                                                         size_t                            seq_size_per_block,
                                                         int64_t                           max_batch_size) {
    if (layer_kv_buffer_ptrs.empty() || kv_block_stride_bytes == 0 || seq_size_per_block == 0) {
        return nullptr;
    }

    auto addr_buffer = buildLayerAddrBuffer(layer_kv_buffer_ptrs);
    if (!addr_buffer.defined()) {
        return nullptr;
    }

    return std::unique_ptr<BlockZeroRunner>(new BlockZeroRunner(
        std::move(addr_buffer), kv_block_stride_bytes, seq_size_per_block,
        layer_kv_buffer_ptrs.size(), max_batch_size));
}

void BlockZeroRunner::run(const GptModelInputs& inputs) {
#if !USING_CUDA && !USING_ROCM
    return;
#else
    if (!inputs.kv_cache_block_id.defined() || !inputs.input_lengths.defined()) {
        return;
    }

    auto block_id_sizes = inputs.kv_cache_block_id.sizes();
    if (block_id_sizes.size() != 3) {
        return;
    }

    const int64_t decode_bs  = inputs.sequence_lengths.defined() ? inputs.sequence_lengths.size(0) : 0;
    const int64_t total_bs   = inputs.input_lengths.size(0);
    const int64_t context_bs = total_bs - decode_bs;

    if (decode_bs + context_bs <= 0) {
        return;
    }

    torch::Tensor token_counts;
    auto build_token_counts = [&]() {
        std::vector<torch::Tensor> parts;
        parts.reserve(2);
        if (decode_bs > 0) {
            parts.push_back(inputs.sequence_lengths.slice(0, 0, decode_bs));
        }
        if (context_bs > 0) {
            parts.push_back(inputs.input_lengths.slice(0, decode_bs, total_bs));
        }
        return torch::cat(parts).to(torch::kInt32).contiguous();
    };

    auto counts_host = build_token_counts();
    if (token_counts_buffer_.defined() && token_counts_buffer_.size(0) >= total_bs) {
        token_counts_buffer_.slice(0, 0, total_bs).copy_(counts_host, /*non_blocking=*/true);
        token_counts = token_counts_buffer_.slice(0, 0, total_bs);
    } else {
        token_counts = counts_host.cuda();
    }

    auto layer_to_group_opt =
        (inputs.kv_cache_layer_to_group.defined()
         && inputs.kv_cache_layer_to_group.size(0) >= static_cast<int64_t>(layer_num_))
            ? std::optional<torch::Tensor>(inputs.kv_cache_layer_to_group.cuda())
            : std::nullopt;

    zero_incomplete_kv_cache_blocks(layer_base_addr_buffer_,
                             inputs.kv_cache_block_id,
                             token_counts,
                             layer_to_group_opt,
                             static_cast<int64_t>(kv_block_stride_bytes_),
                             static_cast<int64_t>(seq_size_per_block_));
#endif
}

}  // namespace rtp_llm
