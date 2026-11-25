#include "rtp_llm/cpp/disaggregate/cache_store/LoadContext.h"

#include "autil/Scope.h"

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ErrorCodeUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <algorithm>
#include <map>
#include <thread>
#include <unordered_set>

namespace rtp_llm {

SyncContext::SyncContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load):
    cache_store_(cache_store), combine_load_(combine_load) {}

void SyncContext::call(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                       int64_t                                                 timeout_ms,
                       CheckCancelFunc                                         check_cancel_func) {
    if (request_block_buffers.empty()) {
        return;
    }

    auto cache_store = cache_store_.lock();
    if (cache_store == nullptr) {
        error_info_ = ErrorInfo(ErrorCode::UNKNOWN_ERROR, ErrorCodeToString(ErrorCode::UNKNOWN_ERROR));
        RTP_LLM_LOG_WARNING("load failed, cache store is nullptr");
        return;
    }

    start_time_ms_     = autil::TimeUtility::currentTimeInMilliSeconds();
    deadline_ms_       = start_time_ms_ + timeout_ms;
    check_cancel_func_ = check_cancel_func;

    if (combine_load_) {  // for rdma only call rpc once
        auto new_buffer = std::make_shared<RequestBlockBuffer>(request_block_buffers[0]->getRequestId());
        for (auto& request_block_buffer : request_block_buffers) {
            auto blocks = request_block_buffer->getBlocks();
            for (auto& [_, block] : blocks) {
                new_buffer->addBlock(block);
            }
        }
        request_block_buffers_ = {new_buffer};
    } else {
        request_block_buffers_ = request_block_buffers;
    }

    expect_layer_cnt_ = request_block_buffers_.size();

    for (auto& request_block_buffer : request_block_buffers_) {
        if (!doCall(request_block_buffer, timeout_ms)) {
            updateResult(false, CacheStoreErrorCode::InvalidParams, request_block_buffer);
        }
    }
}

void SyncContext::updateResult(bool                                       success,
                               CacheStoreErrorCode                        ec,
                               const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!success) {
        if (!timed_out_ && error_info_.code() != ErrorCode::CANCELLED) {
            const auto error_code = transCacheStoreErrorCode(ec);
            error_info_           = ErrorInfo(error_code, ErrorCodeToString(error_code));
        }
        failed_request_block_buffers_.push_back(request_block_buffer);
        RTP_LLM_LOG_WARNING("request %s call finished, state:[%s], error code[%s], cost time %ldms",
                            request_block_buffer->getRequestKey().c_str(),
                            success ? "success" : "failed",
                            CacheStoreErrorCodeToString(ec).c_str(),
                            autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms_);
    } else {
        RTP_LLM_LOG_DEBUG("request %s call finished, state:[%s], cost time %ldms",
                          request_block_buffer->getRequestKey().c_str(),
                          success ? "success" : "failed",
                          autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms_);
    }

    if (++done_layer_cnt_ == expect_layer_cnt_) {
        cond_.notify_all();
    }
}

void SyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto                         once_time_ms = 30;
    while (true) {
        if (done_layer_cnt_ == expect_layer_cnt_) {
            return;
        }

        if (autil::TimeUtility::currentTimeInMilliSeconds() >= deadline_ms_) {
            auto error_code = ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT;
            error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
            timed_out_      = true;
            RTP_LLM_LOG_INFO("load context wait done on timeout");
            return;
        }

        if (check_cancel_func_ != nullptr && check_cancel_func_()) {
            auto error_code = ErrorCode::CANCELLED;
            error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
            RTP_LLM_LOG_INFO("load context wait done on cancelled");
            return;
        }

        // sync wait, safe to use this
        if (cond_.wait_for(lock, std::chrono::milliseconds(once_time_ms), [this] {
                return done_layer_cnt_ == expect_layer_cnt_;
            })) {
            return;
        }
    }
}

bool SyncContext::success() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_info_.ok();
}

std::string SyncContext::getErrorInfoString() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_info_.ToString();
}

ErrorInfo SyncContext::getErrorInfo() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_info_;
}

std::vector<std::string> SyncContext::failedBlockDebugInfos() const {
    std::unique_lock<std::mutex> lock(mutex_);
    if (error_info_.ok() || error_info_.code() == ErrorCode::CANCELLED) {
        return {};
    }

    const auto& buffers =
        (timed_out_ || failed_request_block_buffers_.empty()) ? request_block_buffers_ : failed_request_block_buffers_;
    std::vector<std::string> debug_infos;
    debug_infos.reserve(buffers.size());
    for (const auto& buffer : buffers) {
        debug_infos.push_back(buffer == nullptr ? "null" : buffer->debugInfo());
    }
    return debug_infos;
}

namespace {
// One monotonic timer queue for abandoned loads. Entries hold only weak destination references.
class IsolationDeadlines {
public:
    using Clock = std::chrono::steady_clock;

    static IsolationDeadlines& instance() {
        static IsolationDeadlines timers;
        return timers;
    }
    void schedule(Clock::time_point deadline, std::function<void()> action) {
        std::lock_guard<std::mutex> lock(mutex_);
        actions_.emplace(deadline, std::move(action));
        changed_.notify_one();
    }

private:
    IsolationDeadlines(): worker_([this] { run(); }) {}
    ~IsolationDeadlines() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closing_ = true;
        }
        changed_.notify_one();
        worker_.join();
    }
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!closing_) {
            if (actions_.empty()) {
                changed_.wait(lock);
                continue;
            }
            const auto deadline = actions_.begin()->first;
            if (Clock::now() < deadline) {
                changed_.wait_until(lock, deadline);
                continue;
            }
            auto action = std::move(actions_.begin()->second);
            actions_.erase(actions_.begin());
            lock.unlock();
            action();
            action = {};
            lock.lock();
        }
    }
    std::mutex                                              mutex_;
    std::condition_variable                                 changed_;
    std::multimap<Clock::time_point, std::function<void()>> actions_;
    bool                                                    closing_ = false;
    std::thread                                             worker_;
};
}  // namespace

// This object can outlive LoadContext, gRPC and the logical completion callback.
// Keep only the destination owner and observers alive while an issued write lacks proof.
class LoadContext::DestinationWrites: public std::enable_shared_from_this<DestinationWrites> {
public:
    void setLifetime(std::shared_ptr<void> lifetime) {
        std::lock_guard<std::mutex> lock(mutex_);
        lifetime_ = std::move(lifetime);
    }

    struct Write {
        std::function<void()> fence;
    };
    using Operation = std::shared_ptr<Write>;

    void finish(const Operation& write) {
        std::function<void()>        unused_fence;
        std::unique_lock<std::mutex> lock(mutex_);
        if (pending_.erase(write)) {
            unused_fence = std::move(write->fence);
            releaseIfStopped(lock);
        }
    }

    void beginWrite(const Operation& write) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.insert(write);
        if (lifetime_ || !observers_.empty()) {
            pending_owner_ = shared_from_this();
        }
    }

    void setFence(const Operation& write, std::function<void()> fence) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!pending_.count(write)) {
            return;
        }
        if (!recovery_due_) {
            write->fence = std::move(fence);
            return;
        }
        lock.unlock();
        invokeFence(fence);
    }

    void scheduleRecovery(std::chrono::milliseconds isolation) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if (recovery_scheduled_ || (sealed_ && pending_.empty())) {
                return;
            }
            IsolationDeadlines::instance().schedule(IsolationDeadlines::Clock::now() + isolation,
                                                    [weak = weak_from_this()] {
                                                        if (auto self = weak.lock()) {
                                                            self->recover();
                                                        }
                                                    });
            recovery_scheduled_ = true;
        } catch (...) {
            RTP_LLM_LOG_ERROR("Failed to schedule KV destination recovery; retaining the write references");
        }
    }

    void seal() {
        std::unique_lock<std::mutex> lock(mutex_);
        sealed_ = true;
        releaseIfStopped(lock);
    }

    bool stopped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sealed_ && pending_.empty();
    }

    bool wait(std::chrono::steady_clock::time_point deadline) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cond_.wait_until(lock, deadline, [this] { return sealed_ && pending_.empty(); });
    }

    void onStopped(std::function<void()> callback) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (sealed_ && pending_.empty()) {
            lock.unlock();
            callback();
            return;
        }
        observers_.push_back(std::move(callback));
        if (!pending_.empty()) {
            pending_owner_ = shared_from_this();
        }
    }

private:
    static void invokeFence(const std::function<void()>& fence) noexcept {
        try {
            if (fence) {
                fence();
            }
        } catch (...) {
            RTP_LLM_LOG_ERROR("KV destination fence failed; retaining the write references");
        }
    }

    void recover() {
        std::unique_lock<std::mutex> lock(mutex_);
        recovery_due_ = true;
        // Transports may acknowledge synchronously and remove the current operation.
        while (true) {
            auto it =
                std::find_if(pending_.begin(), pending_.end(), [](const auto& write) { return bool(write->fence); });
            if (it == pending_.end()) {
                return;
            }
            auto fence = std::move((*it)->fence);
            lock.unlock();
            invokeFence(fence);
            lock.lock();
        }
    }

    void releaseIfStopped(std::unique_lock<std::mutex>& lock) {
        if (!sealed_ || !pending_.empty()) {
            return;
        }
        auto observers = std::move(observers_);
        auto lifetime  = std::move(lifetime_);
        pending_owner_.reset();
        lock.unlock();
        lifetime.reset();
        cond_.notify_all();
        for (auto& observer : observers) {
            observer();
        }
    }

    mutable std::mutex                 mutex_;
    std::condition_variable            cond_;
    std::unordered_set<Operation>      pending_;
    bool                               recovery_scheduled_ = false;
    bool                               recovery_due_       = false;
    bool                               sealed_             = false;
    std::shared_ptr<void>              lifetime_;
    std::shared_ptr<DestinationWrites> pending_owner_;
    std::vector<std::function<void()>> observers_;
};

LoadContext::LoadContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load):
    SyncContext(cache_store, combine_load), destination_writes_(std::make_shared<DestinationWrites>()) {}

bool LoadContext::destinationWritesStopped() const {
    return destination_writes_->stopped();
}

bool LoadContext::waitForDestinationWritesStopped(std::chrono::steady_clock::time_point deadline) const {
    return destination_writes_->wait(deadline);
}

void LoadContext::onDestinationWritesStopped(std::function<void()> callback) {
    destination_writes_->onStopped(std::move(callback));
}

void LoadContext::recoverDestinationWritesAfterTimeout(std::chrono::milliseconds isolation) noexcept {
    destination_writes_->scheduleRecovery(isolation);
}

std::function<void()> LoadContext::destinationRecoveryCallback(std::chrono::milliseconds delay) const {
    return [weak = std::weak_ptr<DestinationWrites>(destination_writes_), delay] {
        if (auto writes = weak.lock()) {
            writes->scheduleRecovery(delay);
        }
    };
}

void LoadContext::load(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffer,
                       const std::string&                                      ip,
                       uint32_t                                                port,
                       uint32_t                                                rdma_port,
                       int64_t                                                 timeout_ms,
                       CheckCancelFunc                                         check_cancel_func,
                       int                                                     partition_count,
                       int                                                     partition_id) {
    autil::ScopeGuard seal([this] { destination_writes_->seal(); });
    if (!request_block_buffer.empty() && request_block_buffer.front()) {
        destination_writes_->setLifetime(request_block_buffer.front()->takeLoadWriterLifetime());
    }
    peer_ip_         = ip;
    port_            = port;
    rdma_port_       = rdma_port;
    partition_count_ = partition_count;
    partition_id_    = partition_id;
    call(request_block_buffer, timeout_ms, check_cancel_func);
}

bool LoadContext::doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) {
    auto cache_store = cache_store_.lock();
    if (!cache_store) {
        return false;
    }

    auto                       write = std::make_shared<DestinationWrites::Write>();
    CacheStoreLoadDoneCallback callback(
        [request_block_buffer, shared_this = shared_from_this()](bool success, CacheStoreErrorCode ec) {
            shared_this->updateResult(success, ec, request_block_buffer);
        },
        [state = destination_writes_, write] { state->finish(write); },
        [state = destination_writes_, write](std::function<void()> fence) {
            state->setFence(write, std::move(fence));
        });
    destination_writes_->beginWrite(write);
    cache_store->load(request_block_buffer,
                      std::move(callback),
                      peer_ip_,
                      port_,
                      rdma_port_,
                      timeout_ms,
                      partition_count_,
                      partition_id_);
    return true;
}

StoreContext::StoreContext(const std::shared_ptr<CacheStore>& cache_store): SyncContext(cache_store, true) {}

void StoreContext::store(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                         int64_t                                                 timeout_ms) {
    call(request_block_buffers, timeout_ms, nullptr);
}

bool StoreContext::doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) {
    auto cache_store = cache_store_.lock();

    auto store_layer_callback = [request_block_buffer, shared_this = shared_from_this()](bool                success,
                                                                                         CacheStoreErrorCode ec) {
        shared_this->updateResult(success, ec, request_block_buffer);
    };
    cache_store->store(request_block_buffer, store_layer_callback);
    return true;
}

}  // namespace rtp_llm
