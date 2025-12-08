#pragma once

#include "rtp_llm/cpp/disaggregate/transfer/LayerBlockConvertor.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"

namespace rtp_llm {

class LayerBlockConvertorImpl: public LayerBlockConvertor {
public:
    LayerBlockConvertorImpl(const std::shared_ptr<KVCacheAllocator>& allocator): allocator_(allocator) {}

    std::vector<BufferPtr>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override {
        // TODO: hack not support asymmetric partition
        auto                   kv_cache = allocator_->convertIndexToBuffer(layer_id, block_id);
        std::vector<BufferPtr> buffers;
        if (kv_cache.k_addr) {
            buffers.push_back(kv_cache.k_addr);
        }
        if (kv_cache.v_addr) {
            buffers.push_back(kv_cache.v_addr);
        }
        return buffers;
    }

private:
    std::shared_ptr<KVCacheAllocator> allocator_;
};

}  // namespace rtp_llm