#pragma once

#include <cstddef>

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
        std::vector<std::pair<BlockInfo, size_t>> result;
        if (!allocator_) {
            return result;
        }
        appendBlockPoolLayoutBuffers(result);
        if (!result.empty()) {
            return result;
        }

        const auto layout        = allocator_->allLayerCacheBase();
        auto       append_tensor = [&result](const torch::Tensor& t) {
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
    void appendBlockPoolLayoutBuffers(std::vector<std::pair<BlockInfo, size_t>>& result) const {
        auto block_pool = allocator_->getBlockPool();
        if (!block_pool || !block_pool->getBaseAddress()) {
            return;
        }

        const auto& layouts = block_pool->memoryLayouts();
        if (layouts.empty()) {
            return;
        }

        const bool is_cuda      = block_pool->where() == MemoryType::MEMORY_GPU;
        auto       append_range = [&result, is_cuda](void* base, size_t offset, size_t bytes, size_t stride) {
            if (bytes == 0 || stride == 0) {
                return;
            }
            BlockInfo info;
            info.is_cuda      = is_cuda;
            info.device_index = 0;
            info.scalar_type  = 0;
            info.addr         = static_cast<char*>(base) + static_cast<ptrdiff_t>(offset);
            info.size_bytes   = bytes;
            result.push_back({info, stride});
        };

        void* base = block_pool->getBaseAddress();
        for (const auto& layout_cfg : layouts) {
            append_range(base,
                         layout_cfg.kv_cache_offset_bytes,
                         layout_cfg.kv_block_pool_size_bytes,
                         layout_cfg.kv_block_stride_bytes);
            if (layout_cfg.hasScale()) {
                append_range(base,
                             layout_cfg.kv_scale_offset_bytes,
                             layout_cfg.kv_scale_pool_size_bytes,
                             layout_cfg.kv_scale_stride_bytes);
            }
        }
    }

private:
    std::shared_ptr<KVCacheAllocator> allocator_;
};

}  // namespace rtp_llm
