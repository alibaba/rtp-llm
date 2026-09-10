#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// [KVDIAG] opt-in (RTP_LLM_KV_DIAG=1). The PD KV-load contract can fail with every
// observable field matching - same request id, same block keys, same byte lengths,
// same store, publication inserted, watch registered - and still produce only a
// client-side timeout, because the only thing that connects a publication to a
// waiting peer is this object's watch list. These markers expose the object
// IDENTITY and the watch-list SIZE at registration, at publication and at trigger,
// which is what distinguishes "watch on a different object" from "trigger ran but
// the callback found nothing to write".
static bool kvDiagEnabled() {
    static const bool value = [] {
        const char* e = ::getenv("RTP_LLM_KV_DIAG");
        return e != nullptr && *e != '\0' && std::string(e) != "0";
    }();
    return value;
}

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

    if (kvDiagEnabled()) {
        static std::atomic<int> add_budget{8000};
        if (add_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            size_t nwatch = 0;
            {
                std::shared_lock<std::shared_mutex> l(watch_func_mutex_);
                nwatch = watch_funcs_.size();
            }
            // watch_funcs==0 HERE is the whole ballgame: the blocks landed in a
            // buffer nobody is watching, so no reply is ever sent and the peer
            // times out with no error anywhere.
            RTP_LLM_LOG_WARNING("[KVDIAG-BUF-ADD] buf=%p reqid=%s added=%zu watch_funcs=%zu first=%s",
                                (void*)this,
                                requestid_.c_str(),
                                blocks.size(),
                                nwatch,
                                blocks.empty() ? "<none>" : blocks[0]->key.c_str());
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

    if (kvDiagEnabled()) {
        static std::atomic<int> watch_budget{8000};
        if (watch_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            size_t nblk = 0;
            {
                std::shared_lock<std::shared_mutex> l(blocks_mutex_);
                nblk = blocks_.size();
            }
            size_t nwatch = 0;
            {
                std::shared_lock<std::shared_mutex> l(watch_func_mutex_);
                nwatch = watch_funcs_.size();
            }
            RTP_LLM_LOG_WARNING("[KVDIAG-BUF-WATCH] buf=%p reqid=%s blocks_now=%zu watch_funcs_now=%zu",
                                (void*)this,
                                requestid_.c_str(),
                                nblk,
                                nwatch);
        }
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
        std::shared_lock<std::shared_mutex> lock(watch_func_mutex_);
        tmp_watch_funcs = watch_funcs_;
    }

    if (kvDiagEnabled()) {
        static std::atomic<int> trig_budget{8000};
        if (trig_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            RTP_LLM_LOG_WARNING("[KVDIAG-BUF-TRIG] buf=%p reqid=%s ok=%d blocks=%zu invoking_watch_funcs=%zu",
                                (void*)this,
                                requestid_.c_str(),
                                (int)ok,
                                blocks.size(),
                                tmp_watch_funcs.size());
        }
    }

    for (auto watch_func : tmp_watch_funcs) {
        if (watch_func) {
            watch_func(ok, blocks);
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