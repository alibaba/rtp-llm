#pragma once

#include <torch/extension.h>
#include <set>
#include <cstdint>

#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BlockInfo.h"

namespace rtp_llm {

class LayerBlockConverterImpl: public LayerBlockConverter {
public:
    explicit LayerBlockConverterImpl(const std::shared_ptr<KVCacheAllocator>& allocator): allocator_(allocator) {}

    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override {
        auto block_infos = allocator_->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
        std::vector<BlockInfo> result;
        result.reserve(block_infos.size());
        for (const auto& info : block_infos) {
            if (info.addr != nullptr && info.size_bytes > 0) {
                result.push_back(info);
            }
        }
        return result;
    }

    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        const auto                                layout = allocator_->allLayerCacheBase();
        std::vector<std::pair<BlockInfo, size_t>> result;
        std::set<std::pair<uintptr_t, size_t>>    seen;
        auto                                      append_tensor = [&](const torch::Tensor& t, size_t block_bytes) {
            if (!t.defined() || t.numel() == 0) {
                return;
            }
            RTP_LLM_CHECK_WITH_INFO(t.is_contiguous() && block_bytes > 0 && t.nbytes() % block_bytes == 0,
                                    "P2P registration requires contiguous whole physical blocks");
            if (!seen.emplace(reinterpret_cast<uintptr_t>(t.data_ptr()), t.nbytes()).second) {
                return;
            }
            BlockInfo info;
            info.is_cuda = t.is_cuda();
            info.device_index = t.is_cuda() ? static_cast<int32_t>(t.get_device()) : 0;
            info.scalar_type = static_cast<int32_t>(t.scalar_type());
            info.addr        = t.data_ptr();
            info.size_bytes  = static_cast<size_t>(t.nbytes());
            // The backend may split a large arena into multiple MRs. Never
            // cut a transferable physical block, or require one MR per arena.
            result.push_back({info, block_bytes});
        };
        auto append_region =
            [&](size_t layer, KVCacheRegionName region, const torch::Tensor& kv, const torch::Tensor& scale) {
                if ((!kv.defined() || kv.numel() == 0) && (!scale.defined() || scale.numel() == 0)) {
                    return;
                }
                const auto blocks = allocator_->convertIndexToBuffer(static_cast<int>(layer), region, 0);
                RTP_LLM_CHECK_WITH_INFO(!blocks.empty(), "missing P2P physical block geometry");
                append_tensor(kv, blocks[0].size_bytes);
                if (scale.defined() && scale.numel()) {
                    RTP_LLM_CHECK_WITH_INFO(blocks.size() > 1, "missing P2P scale block geometry");
                    append_tensor(scale, blocks[1].size_bytes);
                }
            };
        for (size_t layer = 0; layer < layout.layers_to_kv_buffer_ptrs.size(); ++layer) {
            const auto scale = layer < layout.layers_to_scale_buffer_ptrs.size() ?
                                   layout.layers_to_scale_buffer_ptrs[layer] :
                                   torch::Tensor();
            append_region(layer, KVCacheRegionName::DEFAULT, layout.layers_to_kv_buffer_ptrs[layer], scale);
        }
        for (size_t layer = 0; layer < layout.layers_to_kv_buffer_ptrs_by_attn.size(); ++layer) {
            for (size_t region = 0; region < layout.layers_to_kv_buffer_ptrs_by_attn[layer].size(); ++region) {
                const auto scale = layer < layout.layers_to_scale_buffer_ptrs_by_attn.size()
                                           && region < layout.layers_to_scale_buffer_ptrs_by_attn[layer].size() ?
                                       layout.layers_to_scale_buffer_ptrs_by_attn[layer][region] :
                                       torch::Tensor();
                append_region(layer,
                              static_cast<KVCacheRegionName>(region),
                              layout.layers_to_kv_buffer_ptrs_by_attn[layer][region],
                              scale);
            }
        }
        for (size_t layer = 0; layer < layout.mla_host_cache_by_layer.size(); ++layer) {
            const auto& info = layout.mla_host_cache_by_layer[layer];
            if (!info.hbm_cache.defined() || !info.hbm_tokens) {
                continue;
            }
            const auto blocks = allocator_->convertIndexToBuffer(static_cast<int>(layer), 0);
            RTP_LLM_CHECK_WITH_INFO(blocks.size() == 1, "tiered MLA needs one physical KV block");
            // Only complete HBM blocks are transfer destinations. Resident
            // slots and per-layer padding are private to the attention kernel.
            const auto full_hbm = info.hbm_cache.flatten(0, 1).narrow(0, 0, info.hbm_tokens);
            append_tensor(full_hbm, blocks[0].size_bytes);
        }
        return result;
    }

private:
    std::shared_ptr<KVCacheAllocator> allocator_;
};

}  // namespace rtp_llm
