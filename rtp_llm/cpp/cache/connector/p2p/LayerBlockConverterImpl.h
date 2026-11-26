#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
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

    std::vector<BlockInfo> convertLinearStatePartition(
        int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const override {
        return coordinator_manager_->convertLinearStatePartition(
            layer_id, tag, block_id, partition_count, partition_id);
    }

    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        const auto  layout = coordinator_manager_->allLayerCacheBase();
        const auto& pools  = coordinator_manager_->groupBlockPools();
        if (!pools.empty() && pools.front()->sharedBacking()) {
            return sharedRegistrationRegions(layout, pools.front()->sharedBacking()->buffer());
        }

        std::vector<std::pair<BlockInfo, size_t>> result;
        using BufferKey = std::tuple<uintptr_t, size_t, bool, int32_t, int32_t>;
        std::set<BufferKey> seen;
        forEachLayerTensor(layout, [&](const torch::Tensor& tensor) {
            const auto info = registrationInfo(tensor);
            if (seen.emplace(reinterpret_cast<uintptr_t>(info.addr),
                             info.size_bytes,
                             info.is_cuda,
                             info.device_index,
                             info.scalar_type)
                    .second) {
                // Match the physical block boundary in independent pools.
                result.emplace_back(info, tensor.stride(0) * tensor.element_size());
            }
        });
        return result;
    }

private:
    template<typename Visit>
    static void forEachLayerTensor(const GroupedCacheLayerLayout& layout, Visit visit) {
        for (const auto& [tag, group] : layout.groups()) {
            (void)tag;
            if (group.empty()) {
                continue;
            }
            for (const auto& layer : group.layers()) {
                for (const auto& tensor : {layer.kv_addr, layer.kv_scale_addr}) {
                    if (tensor.defined() && tensor.numel() > 0) {
                        visit(tensor);
                    }
                }
            }
        }
    }

    static BlockInfo registrationInfo(const torch::Tensor& tensor) {
        BlockInfo info;
        info.is_cuda      = tensor.is_cuda();
        info.device_index = info.is_cuda ? static_cast<int32_t>(tensor.get_device()) : 0;
        info.scalar_type  = static_cast<int32_t>(tensor.scalar_type());
        info.addr         = tensor.data_ptr();
        info.size_bytes   = static_cast<size_t>(tensor.nbytes());
        return info;
    }

    static std::vector<std::pair<BlockInfo, size_t>> sharedRegistrationRegions(const GroupedCacheLayerLayout& layout,
                                                                               const torch::Tensor&           backing) {
        const auto arena_bytes = static_cast<size_t>(backing.nbytes());
        size_t     alignment   = 1;
        size_t     used_bytes  = 0;
        forEachLayerTensor(layout, [&](const torch::Tensor& tensor) {
            const auto rows    = static_cast<size_t>(tensor.size(0));
            const auto payload = static_cast<size_t>(tensor.nbytes()) / rows;
            const auto stride  = tensor.stride(0) * tensor.element_size();
            const auto offset  = tensor.storage_offset() * tensor.element_size();
            used_bytes         = std::max(used_bytes, offset + (rows - 1) * stride + payload);

            // A cut must not bisect a layer block. Uniform layouts can cut
            // between layers; mixed layouts need the whole slot as alignment.
            const auto boundary = stride % payload == 0 && offset % payload == 0 ? payload : stride;
            const auto factor   = boundary / std::gcd(alignment, boundary);
            // No interior cut is possible when the common multiple exceeds the arena.
            alignment = factor > arena_bytes / alignment ? arena_bytes : alignment * factor;
        });

        if (used_bytes == 0) {
            return {};
        }

        // Every layer aliases the same storage. Register disjoint physical
        // regions, including the final partial region if it contains a block.
        std::vector<std::pair<BlockInfo, size_t>> result;
        const auto                                prefix_bytes = arena_bytes / alignment * alignment;
        if (prefix_bytes > 0) {
            result.emplace_back(registrationInfo(backing.narrow(0, 0, prefix_bytes)), alignment);
        }
        if (used_bytes > prefix_bytes) {
            const auto tail_bytes = arena_bytes - prefix_bytes;
            result.emplace_back(registrationInfo(backing.narrow(0, prefix_bytes, tail_bytes)), tail_bytes);
        }
        return result;
    }

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
