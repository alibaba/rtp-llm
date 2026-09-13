#include <mutex>
#include <unordered_map>
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RequestBlockBuffer::RequestBlockBuffer(const std::string& requestid, const std::string& request_key):
    requestid_(requestid), request_key_(request_key) {}

RequestBlockBuffer::RequestBlockBuffer(const std::string& requestid, std::shared_ptr<torch::Event> event):
    requestid_(requestid), event_(std::move(event)) {}

RequestBlockBuffer::~RequestBlockBuffer() {}

void RequestBlockBuffer::notifyRequestDone() {
    // request block buffer 关联的request已经结束，触发所有回调
    triggerWatchFunc(false, {});
}

const std::string& RequestBlockBuffer::getRequestId() const {
    return requestid_;
}

const std::string& RequestBlockBuffer::getRequestKey() const {
    return request_key_.empty() ? requestid_ : request_key_;
}

const torch::Event* RequestBlockBuffer::getEvent() const {
    return event_.get();
}

std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> RequestBlockBuffer::getBlocks() const {
    std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
    return blocks_;
}

void RequestBlockBuffer::forEachBlock(
    const std::function<void(const std::string& key, const std::shared_ptr<BlockBuffer>& block)>& callback) const {
    std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
    for (const auto& entry : blocks_) {
        callback(entry.first, entry.second);
    }
}

void RequestBlockBuffer::mergeBlocksFrom(RequestBlockBuffer& src) {
    std::vector<std::shared_ptr<BlockBuffer>> merged_blocks;
    {
        std::scoped_lock lock(blocks_mutex_, src.blocks_mutex_);
        const size_t     moved = src.blocks_.size();
        blocks_.merge(src.blocks_);
        blocks_size_ += src.blocks_size_;
        src.blocks_size_ = 0;
        if (moved > 0 && has_watch_funcs_.load(std::memory_order_acquire)) {
            // Snapshot under the already-held blocks lock; watch funcs run
            // after the locks are released so they may call back into this
            // buffer (addBlock/getBlocks) without deadlocking.
            merged_blocks.reserve(moved);
            for (const auto& entry : blocks_) {
                merged_blocks.push_back(entry.second);
            }
        }
    }
    if (!merged_blocks.empty()) {
        triggerWatchFunc(true, merged_blocks);
    }
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

size_t RequestBlockBuffer::getBlocksSize() const {
    std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
    return blocks_size_;
}

void RequestBlockBuffer::addBlock(const std::shared_ptr<BlockBuffer>& block) {
    if (block == nullptr) {
        return;
    }

    {
        std::unique_lock<std::shared_mutex> lock(blocks_mutex_);
        blocks_[block->key] = block;
        blocks_size_ += block->len;
    }
    // Skip the per-block watch-func lock and the one-element argument vector
    // when no watch func is registered: the PD load path adds one block per
    // (layer, physical block) pair, which is O(layers * blocks) per request.
    if (has_watch_funcs_.load(std::memory_order_acquire)) {
        triggerWatchFunc(true, {block});
    }
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
            blocks_size_ += block->len;
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
    // set callback
    {
        std::unique_lock<std::shared_mutex> lock(watch_func_mutex_);
        watch_funcs_.push_back(watch_func);
    }
    // Publish the flag only after the func is visible in the vector so that
    // addBlock either sees the flag (and triggers the new block itself) or the
    // snapshot below (taken after the insert) includes the block.
    has_watch_funcs_.store(true, std::memory_order_release);

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
    std::vector<WatchFunc> tmp_watch_funcs;
    {
        std::shared_lock<std::shared_mutex> lock(watch_func_mutex_);
        tmp_watch_funcs = watch_funcs_;
    }

    for (auto watch_func : tmp_watch_funcs) {
        if (watch_func) {
            watch_func(ok, blocks);
        }
    }
}

std::string RequestBlockBuffer::debugInfo() const {
    std::ostringstream stream;
    stream << "request id: " << requestid_;
    if (!request_key_.empty()) {
        stream << ", request key: " << request_key_;
    }
    stream << ", blocks count: " << getBlocksCount();
    if (!watch_funcs_.empty()) {
        stream << ", has watch func";
    } else {
        stream << ", no watch func";
    }
    stream << ", block keys: ";
    auto blocks = getBlocks();
    for (const auto& block : blocks) {
        stream << block.first << " ";
    }
    return stream.str();
}

}  // namespace rtp_llm