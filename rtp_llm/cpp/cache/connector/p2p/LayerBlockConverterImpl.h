#pragma once

#include <cstdint>
#include <set>
#include <tuple>

#include <torch/extension.h>

#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BlockInfo.h"

namespace rtp_llm {

class LayerBlockConverterImpl: public LayerBlockConverter {
public:
    explicit LayerBlockConverterImpl(const std::shared_ptr<KVCacheAllocator>& allocator): allocator_(allocator) {}

    std::vector<BlockInfo> convertIndexToBufferByTag(
        int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const override {
        return filterValid(
            allocator_->convertIndexToBufferByTag(layer_id, tag, block_id, partition_count, partition_id));
    }

    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        const auto                                layout = allocator_->allLayerCacheBase();
        std::vector<std::pair<BlockInfo, size_t>> result;
        using BufferKey = std::tuple<uintptr_t, size_t, bool, int32_t, int32_t>;
        std::set<BufferKey> seen;
        auto                append_tensor = [&result, &seen](const torch::Tensor& t) {
            if (!t.defined() || t.numel() == 0) {
                return;
            }
            BlockInfo info;
            info.is_cuda         = t.is_cuda();
            info.device_index    = t.is_cuda() ? static_cast<int32_t>(t.get_device()) : 0;
            info.scalar_type     = static_cast<int32_t>(t.scalar_type());
            info.addr            = t.data_ptr();
            info.size_bytes      = static_cast<size_t>(t.nbytes());
            const size_t aligned = static_cast<size_t>(t.nbytes());
            if (!seen.emplace(reinterpret_cast<uintptr_t>(info.addr),
                              info.size_bytes,
                              info.is_cuda,
                              info.device_index,
                              info.scalar_type)
                     .second) {
                return;
            }
            result.push_back({info, aligned});
        };
        for (const auto& [tag, group_layout] : layout.groups()) {
            (void)tag;
            if (group_layout.empty()) {
                continue;
            }
            for (const auto& layer : group_layout.layers()) {
                append_tensor(layer.kv_addr);
                append_tensor(layer.kv_scale_addr);
            }
        }
        return result;
    }

private:
    static std::vector<BlockInfo> filterValid(const std::vector<BlockInfo>& block_infos) {
        std::vector<BlockInfo> result;
        result.reserve(block_infos.size());
        for (const auto& info : block_infos) {
            if (info.addr != nullptr && info.size_bytes > 0) {
                result.push_back(info);
            }
        }
        return result;
    }

    std::shared_ptr<KVCacheAllocator> allocator_;
};

}  // namespace rtp_llm
