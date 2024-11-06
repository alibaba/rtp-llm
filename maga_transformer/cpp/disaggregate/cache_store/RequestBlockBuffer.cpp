#include <mutex>
#include <unordered_map>
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBuffer.h"

namespace rtp_llm {

RequestBlockBuffer::RequestBlockBuffer(const std::string& requestid): requestid_(requestid) {}

RequestBlockBuffer::RequestBlockBuffer(const std::string& requestid, const std::shared_ptr<void>& event):
    requestid_(requestid), event_(event) {}

const std::string& RequestBlockBuffer::getRequestId() const {
    return requestid_;
}

const std::shared_ptr<void>& RequestBlockBuffer::getEvent() const {
    return event_;
}

std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> RequestBlockBuffer::getBlocks() const {
    std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
    return blocks_;
}

std::shared_ptr<BlockBuffer> RequestBlockBuffer::getBlock(const std::string& id) const {
    std::shared_lock<std::shared_mutex> lock(blocks_mutex_);

    auto iter = blocks_.find(id);
    if (iter != blocks_.end()) {
        return iter->second;
    }
    return nullptr;
}

size_t RequestBlockBuffer::getBlocksCount() const {
    std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
    return blocks_.size();
}

void RequestBlockBuffer::addBlock(const std::shared_ptr<BlockBuffer>& block) {
    if (block == nullptr) {
        return;
    }

    std::unique_lock<std::shared_mutex> lock(blocks_mutex_);
    blocks_[block->key] = block;
}

void RequestBlockBuffer::addBlock(
    const std::string& key, const std::shared_ptr<void>& addr, uint32_t len, bool gpu_mem, bool adopted) {
    auto block = std::make_shared<BlockBuffer>(key, addr, len, gpu_mem, adopted);
    addBlock(block);
}

bool RequestBlockBuffer::isValid() const {
    std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
    for (auto iter : blocks_) {
        if (iter.second->addr == nullptr || iter.second->len == 0) {
            return false;
        }
    }
    return true;
}

}  // namespace rtp_llm