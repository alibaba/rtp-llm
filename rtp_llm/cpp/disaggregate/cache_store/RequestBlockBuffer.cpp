#include <exception>
#include <mutex>
#include <unordered_map>
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

std::exception_ptr invokeWatchFuncs(
    const std::vector<RequestBlockBuffer::WatchFunc>&             watch_funcs,
    bool                                                          ok,
    const std::vector<std::shared_ptr<BlockBuffer>>&              blocks,
    std::exception_ptr                                            first_error = nullptr) {
    for (const auto& watch_func : watch_funcs) {
        if (!watch_func) {
            continue;
        }
        try {
            watch_func(ok, blocks);
        } catch (...) {
            if (!first_error) {
                first_error = std::current_exception();
            }
        }
    }
    return first_error;
}

}  // namespace

RequestBlockBuffer::RequestBlockBuffer(const std::string& requestid, const std::string& request_key):
    requestid_(requestid), request_key_(request_key) {}

RequestBlockBuffer::RequestBlockBuffer(const std::string& requestid, std::shared_ptr<torch::Event> event):
    requestid_(requestid), event_(std::move(event)) {}

RequestBlockBuffer::~RequestBlockBuffer() {}

void RequestBlockBuffer::notifyRequestDone() {
    std::vector<WatchFunc> watch_funcs;
    {
        std::unique_lock<std::shared_mutex> lock(watch_func_mutex_);
        if (request_done_) {
            return;
        }
        request_done_              = true;
        pending_done_watch_funcs_ = std::move(watch_funcs_);
        if (active_watch_dispatches_ == 0) {
            watch_funcs = std::move(pending_done_watch_funcs_);
        }
    }

    auto callback_error = invokeWatchFuncs(watch_funcs, false, {});
    if (callback_error) {
        std::rethrow_exception(callback_error);
    }
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
    bool request_done = false;
    {
        std::unique_lock<std::shared_mutex> lock(watch_func_mutex_);
        request_done = request_done_;
        if (!request_done) {
            watch_funcs_.push_back(std::move(watch_func));
        }
    }

    if (request_done) {
        if (watch_func) {
            watch_func(false, {});
        }
        return false;
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
    std::vector<WatchFunc> tmp_watch_funcs;
    {
        std::unique_lock<std::shared_mutex> lock(watch_func_mutex_);
        if (request_done_) {
            return;
        }
        tmp_watch_funcs = watch_funcs_;
        ++active_watch_dispatches_;
    }

    auto callback_error = invokeWatchFuncs(tmp_watch_funcs, ok, blocks);

    std::vector<WatchFunc> done_watch_funcs;
    {
        std::unique_lock<std::shared_mutex> lock(watch_func_mutex_);
        --active_watch_dispatches_;
        if (request_done_ && active_watch_dispatches_ == 0) {
            done_watch_funcs = std::move(pending_done_watch_funcs_);
        }
    }
    callback_error = invokeWatchFuncs(done_watch_funcs, false, {}, callback_error);
    if (callback_error) {
        std::rethrow_exception(callback_error);
    }
}

std::string RequestBlockBuffer::debugInfo() const {
    std::ostringstream stream;
    stream << "request id: " << requestid_ << ", blocks count: " << getBlocksCount();
    std::shared_lock<std::shared_mutex> lock(watch_func_mutex_);
    if (!watch_funcs_.empty()) {
        stream << ", has watch func";
    } else {
        stream << ", no watch func";
    }
    return stream.str();
}

}  // namespace rtp_llm
