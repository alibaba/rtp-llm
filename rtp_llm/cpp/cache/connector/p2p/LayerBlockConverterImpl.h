#pragma once

#include <cstdint>
#include <set>
#include <tuple>

#include <torch/extension.h>

#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/BlockInfo.h"

namespace rtp_llm {

class LayerBlockConverterImpl: public LayerBlockConverter {
public:
    explicit LayerBlockConverterImpl(const std::shared_ptr<CoordinatorCacheManager>& coordinator_manager):
        coordinator_manager_(coordinator_manager) {}

    std::vector<BlockInfo> convertIndexToBuffer(
        int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const override {
        return filterValid(
            coordinator_manager_->convertIndexToBuffer(layer_id, tag, block_id, partition_count, partition_id));
    }

    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        const auto                                layout = coordinator_manager_->allLayerCacheBase();
        std::vector<std::pair<BlockInfo, size_t>> result;
        using BufferKey = std::tuple<uintptr_t, size_t, bool, int32_t, int32_t>;
        std::set<BufferKey> seen;
        auto                append_buffer = [&result, &seen](const BlockInfo& info) {
            if (!seen.emplace(reinterpret_cast<uintptr_t>(info.addr),
                              info.size_bytes,
                              info.is_cuda,
                              info.device_index,
                              info.scalar_type)
                     .second) {
                return;
            }
            result.push_back({info, info.size_bytes});
        };
        auto append_tensor = [&append_buffer](const torch::Tensor& t) {
            if (t.defined() && t.numel() > 0) {
                append_buffer({t.is_cuda(),
                               t.is_cuda() ? static_cast<int32_t>(t.get_device()) : 0,
                               static_cast<int32_t>(t.scalar_type()),
                               t.data_ptr(),
                               static_cast<size_t>(t.nbytes())});
            }
        };
        for (const auto& [tag, group_layout] : layout.groups()) {
            if (group_layout.empty()) {
                continue;
            }
            for (const auto& layer : group_layout.layers()) {
                if (layer.offload) {
                    const auto& storage = *layer.offload;
                    for (const bool gpu : {true, false}) {
                        const size_t blocks = gpu ? storage.hbm_blocks : storage.total_blocks - storage.hbm_blocks;
                        if (blocks == 0) {
                            continue;
                        }
                        for (auto buffer : coordinator_manager_->convertIndexToBuffer(
                                 storage.owner_layer, tag, gpu ? 0 : storage.hbm_blocks)) {
                            buffer.size_bytes *= blocks;
                            append_buffer(buffer);
                        }
                    }
                    continue;
                }
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

    std::shared_ptr<CoordinatorCacheManager> coordinator_manager_;
};

}  // namespace rtp_llm
