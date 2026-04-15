#include "rtp_llm/cpp/models/BlockZeroRunner.h"

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
                                 size_t        layer_num):
    layer_base_addr_buffer_(std::move(layer_base_addr_buffer)),
    kv_block_stride_bytes_(kv_block_stride_bytes),
    seq_size_per_block_(seq_size_per_block),
    layer_num_(layer_num) {}

std::unique_ptr<BlockZeroRunner> BlockZeroRunner::create(const std::vector<torch::Tensor>& layer_kv_buffer_ptrs,
                                                         size_t                            kv_block_stride_bytes,
                                                         size_t                            seq_size_per_block) {
    if (layer_kv_buffer_ptrs.empty() || kv_block_stride_bytes == 0 || seq_size_per_block == 0) {
        return nullptr;
    }

    // GPU addresses are stored as int64_t because PyTorch lacks an unsigned 64-bit
    // tensor dtype.  The bit pattern is preserved through matching reinterpret_casts
    // on both store and load sides; no pointer arithmetic is performed on the int64_t
    // representation, so the signedness does not affect correctness.
    static_assert(sizeof(void*) <= sizeof(int64_t), "GPU pointer must fit in int64_t");
    auto  num_layers = layer_kv_buffer_ptrs.size();
    auto  layer_addrs = torch::empty({static_cast<int64_t>(num_layers)}, torch::kInt64);
    auto* addr_data   = layer_addrs.data_ptr<int64_t>();
    for (size_t i = 0; i < num_layers; ++i) {
        if (!layer_kv_buffer_ptrs[i].defined()) {
            RTP_LLM_LOG_WARNING("BlockZeroRunner: layer_kv_buffer_ptrs[%zu] is undefined", i);
            return nullptr;
        }
        addr_data[i] = reinterpret_cast<int64_t>(layer_kv_buffer_ptrs[i].data_ptr());
    }

    return std::unique_ptr<BlockZeroRunner>(new BlockZeroRunner(
        layer_addrs.cuda(), kv_block_stride_bytes, seq_size_per_block, num_layers));
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

    std::vector<torch::Tensor> parts;
    parts.reserve(2);
    if (decode_bs > 0) {
        parts.push_back(inputs.sequence_lengths.slice(0, 0, decode_bs));
    }
    if (context_bs > 0) {
        parts.push_back(inputs.input_lengths.slice(0, decode_bs, total_bs));
    }
    auto token_counts = torch::cat(parts).to(torch::kInt32).contiguous().cuda();

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
