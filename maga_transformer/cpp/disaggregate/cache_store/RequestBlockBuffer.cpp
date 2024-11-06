#include <mutex>
#include <unordered_map>
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "src/fastertransformer/utils/logger.h"

namespace rtp_llm {

RequestBlockBuffer::RequestBlockBuffer(const std::string& requestid): requestid_(requestid) {}

RequestBlockBuffer::RequestBlockBuffer(const std::string& requestid, const std::shared_ptr<void>& event):
    requestid_(requestid), event_(event) {}

RequestBlockBuffer::~RequestBlockBuffer() {
    triggerWatchFunc(false, {});
}

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

    {
        std::unique_lock<std::shared_mutex> lock(blocks_mutex_);
        blocks_[block->key] = block;
    }
    triggerWatchFunc(true, {block});
}

void RequestBlockBuffer::addBlock(
    const std::string& key, const std::shared_ptr<void>& addr, uint32_t len, bool gpu_mem, bool adopted) {
    auto block = std::make_shared<BlockBuffer>(key, addr, len, gpu_mem, adopted);
    addBlock(block);
}

void RequestBlockBuffer::addBlocks(const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    {
        std::unique_lock<std::shared_mutex> lock(blocks_mutex_);
        for (auto& block : blocks) {
            blocks_[block->key] = block;
        }
    }

    triggerWatchFunc(true, blocks);
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

bool RequestBlockBuffer::setWatchFunc(RequestBlockBuffer::WatchFunc&& watch_func) {
    {
        std::unique_lock<std::shared_mutex> lock(watch_func_mutex_);
        if (watch_func_ != nullptr) {
            FT_LOG_WARNING("set request block buffer watch func twice, request id is %s", requestid_.c_str());
            return false;
        }
        watch_func_ = watch_func;
    }

    // current blocks trigger once
    // set callback then trigger will not miss new blocks
    std::vector<std::shared_ptr<BlockBuffer>> blocks;
    {
        std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
        for (auto iter : blocks_) {
            blocks.push_back(iter.second);
        }
    }
    if (!blocks.empty()) {
        triggerWatchFunc(true, blocks);
    }
    return true;
}

void RequestBlockBuffer::triggerWatchFunc(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    std::shared_lock<std::shared_mutex> lock(watch_func_mutex_);
    if (watch_func_) {
        watch_func_(ok, blocks);
    }
}

}  // namespace rtp_llm