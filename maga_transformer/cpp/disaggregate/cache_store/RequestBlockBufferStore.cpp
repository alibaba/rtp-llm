#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, RequestBlockBufferStore);

RequestBlockBufferStore::RequestBlockBufferStore(const std::shared_ptr<MemoryUtil>& memory_util, void* stream):
    memory_util_(memory_util), stream_(stream) {}

bool RequestBlockBufferStore::setRequestBlockBuffer(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    // event to sync wait compute
    if (request_block_buffer->getEvent()) {
        if (!memory_util_->gpuEventBarrier(request_block_buffer->getEvent().get())) {
            AUTIL_LOG(WARN,
                      "set request block buffer to request block buffer store failed, request id %s",
                      request_block_buffer->getRequestId().c_str());
            return false;
        }
    }

    auto store_request_block_buffer = getOrInsertRequestBlockBuffer(request_block_buffer->getRequestId());
    if (store_request_block_buffer == nullptr) {
        AUTIL_LOG(WARN,
                  "set request block buffer to request block buffer store failed, request id %s",
                  request_block_buffer->getRequestId().c_str());
        return false;
    }

    auto blocks = request_block_buffer->getBlocks();
    for (auto iter : blocks) {
        auto& block = iter.second;
        if (isValidBlock(block)) {
            store_request_block_buffer->addBlock(block);
            continue;
        }

        auto valid_block = makeValidBlock(block);
        if (!valid_block) {
            AUTIL_LOG(WARN,
                      "set request block buffer to request block buffer store failed, request id %s",
                      request_block_buffer->getRequestId().c_str());
            return false;
        }
        store_request_block_buffer->addBlock(valid_block);
    }

    return true;
}

std::shared_ptr<BlockBuffer> RequestBlockBufferStore::getBlockBuffer(const std::string& requestid,
                                                                     const std::string& blockid) const {
    auto request_block_buffer = getRequestBlockBuffer(requestid);
    if (request_block_buffer == nullptr) {
        return nullptr;
    }
    return request_block_buffer->getBlock(blockid);
}

std::shared_ptr<RequestBlockBuffer> RequestBlockBufferStore::getRequestBlockBuffer(const std::string& requestid) const {
    std::shared_lock<std::shared_mutex> lock(request_cache_map_mutex_);

    auto iter = request_cache_map_.find(requestid);
    if (iter != request_cache_map_.end()) {
        return iter->second;
    }
    return nullptr;
}

std::shared_ptr<RequestBlockBuffer>
RequestBlockBufferStore::getOrInsertRequestBlockBuffer(const std::string& requestid) {
    std::unique_lock<std::shared_mutex> lock(request_cache_map_mutex_);

    auto iter = request_cache_map_.find(requestid);
    if (iter != request_cache_map_.end()) {
        return iter->second;
    }

    auto ret = request_cache_map_.insert(std::make_pair(requestid, std::make_shared<RequestBlockBuffer>(requestid)));
    if (!ret.second) {
        AUTIL_LOG(WARN,
                  "request block buffer store new request block buffer to request map failed, request id %s",
                  requestid.c_str());
        return nullptr;
    }

    return ret.first->second;
}

bool RequestBlockBufferStore::isValidBlock(const std::shared_ptr<BlockBuffer>& block) {
    if (memory_util_->rdmaMode()) {
        return memory_util_->isMemoryMr(block->addr.get(), block->len, block->gpu_mem, block->adopted);
    }
    return block->gpu_mem == false;
}

std::shared_ptr<BlockBuffer> RequestBlockBufferStore::makeValidBlock(const std::shared_ptr<BlockBuffer>& block) {
    // addr allocated by MemoryUtil, will be mr addr in rdma
    auto addr = std::shared_ptr<void>(memory_util_->mallocCPU(block->len), [memory_util = memory_util_](void* p) {
        memory_util->deregUserMr(p, false);
        memory_util->freeCPU(p);
    });

    if (addr == nullptr) {
        AUTIL_LOG(WARN, "make valid block failed, alloc buffer failed, block %s", block->key.c_str());
        return nullptr;
    }

    if (!memory_util_->regUserMr(addr.get(), block->len, false)) {
        AUTIL_LOG(WARN, "malloc valid block mr failed, block %s", block->key.c_str());
        return nullptr;
    }

    auto new_block = std::make_shared<BlockBuffer>(block->key, addr, block->len, false, true);

    if (!copyBlock(new_block, block)) {
        return nullptr;
    }
    return new_block;
}

bool RequestBlockBufferStore::copyBlock(const std::shared_ptr<BlockBuffer>& dst_block,
                                        const std::shared_ptr<BlockBuffer>& src_block) {
    // same block, no need copy
    if (dst_block == src_block) {
        return true;
    }

    // different ptr, same addr, no need copy
    if (dst_block->addr == src_block->addr) {
        return true;
    }

    AUTIL_INTERVAL_LOG(1000, INFO, "copy block cache once, may affect performance");

    if (dst_block->len != src_block->len) {
        AUTIL_LOG(WARN,
                  "copy block cache failed, block %s len %d vs %d not equal",
                  src_block->key.c_str(),
                  src_block->len,
                  dst_block->len);
        return false;
    }

    AUTIL_LOG(INFO, "copy block cache once, may affect performance");

    if (!memory_util_->memcopy(
            dst_block->addr.get(), dst_block->gpu_mem, src_block->addr.get(), src_block->gpu_mem, src_block->len)) {
        AUTIL_LOG(WARN, "block cache store copy from gpu to gpu failed");
        return false;
    }
    return true;
}

void RequestBlockBufferStore::delRequestBlockBuffer(const std::string& requestid) {
    std::unique_lock<std::shared_mutex> lock(request_cache_map_mutex_);
    request_cache_map_.erase(requestid);
}

}  // namespace rtp_llm