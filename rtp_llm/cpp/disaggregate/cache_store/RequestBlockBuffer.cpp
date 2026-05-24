#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <utility>
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
    return setWatchFunc(std::move(watch_func), nullptr);
}

bool RequestBlockBuffer::setWatchFunc(RequestBlockBuffer::WatchFunc&&                        watch_func,
                                      std::shared_ptr<const std::unordered_set<std::string>> filter_keys) {
    auto      active_filter_keys = filter_keys;
    WatchFunc active_watch_func  = std::move(watch_func);
    // set callback
    {
        std::unique_lock<std::shared_mutex> lock(watch_func_mutex_);
        watch_funcs_.push_back({active_watch_func, std::move(filter_keys)});
    }

    // current blocks trigger once
    // set callback then trigger will not miss new blocks. Replay only to the
    // newly registered watcher; broadcasting through triggerWatchFunc() makes
    // every existing layer watcher rescan the same block list.
    std::vector<std::shared_ptr<BlockBuffer>> blocks;
    {
        std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
        for (auto iter : blocks_) {
            if (active_filter_keys == nullptr || active_filter_keys->empty()
                || active_filter_keys->find(iter.first) != active_filter_keys->end()) {
                blocks.push_back(iter.second);
            }
        }
    }
    if (!blocks.empty() && active_watch_func) {
        active_watch_func(true, blocks);
    }
    return true;
}

void RequestBlockBuffer::triggerWatchFunc(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    std::vector<Watcher> tmp_watch_funcs;
    {
        std::shared_lock<std::shared_mutex> lock(watch_func_mutex_);
        tmp_watch_funcs = watch_funcs_;
    }

    for (auto& watcher : tmp_watch_funcs) {
        if (!watcher.func) {
            continue;
        }
        if (!ok || watcher.filter_keys == nullptr || watcher.filter_keys->empty()) {
            watcher.func(ok, blocks);
            continue;
        }

        std::vector<std::shared_ptr<BlockBuffer>> matched_blocks;
        matched_blocks.reserve(std::min(blocks.size(), watcher.filter_keys->size()));
        for (auto& block : blocks) {
            if (block != nullptr && watcher.filter_keys->find(block->key) != watcher.filter_keys->end()) {
                matched_blocks.push_back(block);
            }
        }
        if (!matched_blocks.empty()) {
            watcher.func(true, matched_blocks);
        }
    }
}

std::string RequestBlockBuffer::debugInfo() const {
    std::ostringstream stream;
    stream << "request id: " << requestid_ << ", blocks count: " << getBlocksCount();
    if (!watch_funcs_.empty()) {
        stream << ", has watch func";
    } else {
        stream << ", no watch func";
    }
    return stream.str();
}

}  // namespace rtp_llm
