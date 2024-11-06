#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "src/fastertransformer/utils/logger.h"

namespace rtp_llm {

RequestBlockBufferStore::RequestBlockBufferStore(const std::shared_ptr<MemoryUtil>& memory_util, void* stream):
    memory_util_(memory_util), stream_(stream) {}

void RequestBlockBufferStore::setStoreBlockBufferCallBack(const std::string& requestid, StoreBlockBufferCallbackFunc&& callback){
    std::unique_lock<std::shared_mutex> lock(request_cache_map_mutex_);
    auto it = request_cache_map_.find(requestid);
    if (it == request_cache_map_.end()) {
        FT_LOG_DEBUG("set call back not find requestid %s", requestid.c_str());
        request_cache_map_.insert(std::make_pair(requestid, std::make_shared<RequestBlockBufferInfo>(std::make_shared<RequestBlockBuffer>(requestid), nullptr)));
    }
    request_cache_map_[requestid]->callback_ = std::move(callback);
}

void RequestBlockBufferStore::runStoreBlockBufferCallBack(const std::string& requestid, const std::shared_ptr<BlockBuffer>& block){
    std::shared_lock<std::shared_mutex> lock(request_cache_map_mutex_);
    if (request_cache_map_.find(requestid) == request_cache_map_.end()) {
        return;
    }
    StoreBlockBufferCallbackFunc callback = request_cache_map_[requestid]->callback_;
    if(callback){
        callback(block);
    }
}

bool RequestBlockBufferStore::setRequestBlockBuffer(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    // event to sync wait compute
    if (request_block_buffer->getEvent()) {
        if (!memory_util_->gpuEventBarrier(request_block_buffer->getEvent().get())) {
            FT_LOG_WARNING(
                      "set request block buffer to request block buffer store failed, request id %s",
                      request_block_buffer->getRequestId().c_str());
            return false;
        }
    }

    auto store_request_block_buffer = getOrInsertRequestBlockBuffer(request_block_buffer->getRequestId());
    if (store_request_block_buffer == nullptr) {
        FT_LOG_WARNING(
                  "set request block buffer to request block buffer store failed, request id %s",
                  request_block_buffer->getRequestId().c_str());
        return false;
    }

    auto blocks = request_block_buffer->getBlocks();
    for (auto iter : blocks) {
        auto& block = iter.second;
        if (isValidBlock(block)) {
            store_request_block_buffer->addBlock(block);
            runStoreBlockBufferCallBack(request_block_buffer->getRequestId(), block);
            continue;
        }

        auto valid_block = makeValidBlock(block);
        if (!valid_block) {
            FT_LOG_WARNING(
                      "set request block buffer to request block buffer store failed, request id %s",
                      request_block_buffer->getRequestId().c_str());
            return false;
        }
        store_request_block_buffer->addBlock(valid_block);

        runStoreBlockBufferCallBack(request_block_buffer->getRequestId(), valid_block);
    }    
    return true;
}

void RequestBlockBufferStore::debugInfo() {
    std::string                         debug = "";
    std::shared_lock<std::shared_mutex> lock(request_cache_map_mutex_);
    for (auto block : request_cache_map_) {
        std::ostringstream single;
        single << "request id: " << block.first << " block ids: ";
        for (auto s: block.second->block_buffer_->getBlocks()) {
            single << s.first << " ";
        }
        single << "\n";
        debug += single.str();
    }
    FT_LOG_INFO("reqeut block buffer debug info: %s", debug.c_str());
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
        return iter->second->block_buffer_;
    }
    return nullptr;
}

std::shared_ptr<RequestBlockBuffer>
RequestBlockBufferStore::getOrInsertRequestBlockBuffer(const std::string& requestid) {
    std::unique_lock<std::shared_mutex> lock(request_cache_map_mutex_);

    auto iter = request_cache_map_.find(requestid);
    if (iter != request_cache_map_.end()) {
        return iter->second->block_buffer_;
    }

    auto ret = request_cache_map_.insert(std::make_pair(requestid, std::make_shared<RequestBlockBufferInfo>(std::make_shared<RequestBlockBuffer>(requestid), nullptr)));
    if (!ret.second) {
        FT_LOG_WARNING(
                  "request block buffer store new request block buffer to request map failed, request id %s",
                  requestid.c_str());
        return nullptr;
    }

    return ret.first->second->block_buffer_;
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
        FT_LOG_WARNING("make valid block failed, alloc buffer failed, block %s", block->key.c_str());
        return nullptr;
    }

    if (!memory_util_->regUserMr(addr.get(), block->len, false)) {
        FT_LOG_WARNING("malloc valid block mr failed, block %s", block->key.c_str());
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

    if (dst_block->len != src_block->len) {
        FT_LOG_WARNING(
                  "copy block cache failed, block %s len %d vs %d not equal",
                  src_block->key.c_str(),
                  src_block->len,
                  dst_block->len);
        return false;
    }

    FT_INTERVAL_LOG(120, INFO, "copy block cache once, may affect performance");

    if (!memory_util_->memcopy(
            dst_block->addr.get(), dst_block->gpu_mem, src_block->addr.get(), src_block->gpu_mem, src_block->len)) {
        FT_LOG_WARNING("block cache store copy failed");
        return false;
    }
    return true;
}

void RequestBlockBufferStore::delRequestBlockBuffer(const std::string& requestid) {
    std::unique_lock<std::shared_mutex> lock(request_cache_map_mutex_);
    request_cache_map_.erase(requestid);
}

}  // namespace rtp_llm