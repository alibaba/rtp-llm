#pragma once

#include <torch/extension.h>

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
        auto                                      append_tensor = [&result](const torch::Tensor& t) {
            if (!t.defined() || t.numel() == 0) {
                return;
            }
            BlockInfo info;
            info.is_cuda = t.is_cuda();
            info.device_index = t.is_cuda() ? static_cast<int32_t>(t.get_device()) : 0;
            info.scalar_type     = static_cast<int32_t>(t.scalar_type());
            info.addr            = t.data_ptr();
            info.size_bytes      = static_cast<size_t>(t.nbytes());
            const size_t aligned = static_cast<size_t>(t.nbytes());
            result.push_back({info, aligned});
        };
        for (const auto& t : layout.layers_to_kv_buffer_ptrs) {
            append_tensor(t);
        }
        for (const auto& t : layout.layers_to_scale_buffer_ptrs) {
            append_tensor(t);
        }
        return result;
    }

private:
    std::shared_ptr<KVCacheAllocator> allocator_;
};

}  // namespace rtp_llm
