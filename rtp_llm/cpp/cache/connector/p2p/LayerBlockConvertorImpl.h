#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerBlockConvertor.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"

namespace rtp_llm {

class LayerBlockConvertorImpl: public LayerBlockConvertor {
public:
    LayerBlockConvertorImpl(const std::shared_ptr<KVCacheAllocator>& allocator): allocator_(allocator) {}

    std::vector<BufferPtr>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override {
        return allocator_->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
    }

    std::vector<std::pair<BufferPtr, size_t>> getAllBuffers() const override {
        return allocator_->getAllBuffers();
    }

private:
    std::shared_ptr<KVCacheAllocator> allocator_;
};

}  // namespace rtp_llm