#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcTransport.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iterator>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/model_rpc/MultimodalPbConverter.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

inline constexpr const char* kReasonConnectionError    = "connection_error";
inline constexpr const char* kReasonGrpcError          = "grpc_error";
inline constexpr const char* kReasonReleaseQueueFull   = "release_queue_full";
inline constexpr const char* kReasonReleaseClientStop  = "release_client_stopping";
inline constexpr size_t      kMaxPendingReleaseHandles = 1024;
inline constexpr unsigned    kMaxReleaseAttempts       = 6;

bool retryableRelease(const grpc::Status& status) {
    switch (status.error_code()) {
        case grpc::StatusCode::DEADLINE_EXCEEDED:
        case grpc::StatusCode::UNAVAILABLE:
        case grpc::StatusCode::RESOURCE_EXHAUSTED:
        case grpc::StatusCode::CANCELLED:
        case grpc::StatusCode::UNKNOWN:
        case grpc::StatusCode::INTERNAL:
            return true;
        default:
            return false;
    }
}

std::vector<std::string> uniqueReleaseHandles(const std::vector<std::string>& handles) {
    std::vector<std::string>        unique;
    std::unordered_set<std::string> seen;
    unique.reserve(handles.size());
    for (const auto& handle : handles) {
        if (!handle.empty() && seen.insert(handle).second) {
            unique.push_back(handle);
        }
    }
    return unique;
}

std::string pendingReleaseKey(const std::string& endpoint, const std::string& handle) {
    return endpoint + '\0' + handle;
}

class GrpcMMControlClient: public MMControlClient {
public:
    GrpcMMControlClient(MMTransportMetricsPtr metrics, int64_t release_timeout_ms):
        metrics_(std::move(metrics)), release_timeout_ms_(std::max<int64_t>(1, release_timeout_ms)) {
        release_thread_ = std::thread(&GrpcMMControlClient::releaseLoop, this);
    }

    ~GrpcMMControlClient() override {
        size_t abandoned_handles = 0;
        {
            std::lock_guard<std::mutex> lock(release_mutex_);
            stopping_         = true;
            abandoned_handles = pending_release_keys_.size();
            release_queue_.clear();
            pending_release_keys_.clear();
        }
        release_cv_.notify_one();
        if (release_thread_.joinable()) {
            release_thread_.join();
        }
        if (abandoned_handles > 0) {
            RTP_LLM_LOG_WARNING("multimodal release client stopped with %zu queued handle(s); "
                                "encoder slot GC will reclaim them",
                                abandoned_handles);
        }
    }

    ErrorResult<MultimodalOutputPB>
    request(const std::string& endpoint, MultimodalInputsPB& request_pb, DeadlineBudget& budget) override {
        if (budget.exhausted()) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "vit rpc budget exhausted before the call");
        }
        auto connection_status = pool_.getConnection(endpoint);
        if (!connection_status.ok()) {
            metrics_->reportRpcClientError(endpoint, kReasonConnectionError);
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, connection_status.status().ToString());
        }
        auto& connection = connection_status.value();
        auto  stub       = connection.stub;

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(budget.remainingMs()));
        MultimodalOutputPB receipt;
        const int64_t      request_bytes = request_pb.ByteSizeLong();
        const auto         start         = std::chrono::steady_clock::now();
        auto               status        = stub->RemoteMultimodalEmbedding(&context, request_pb, &receipt);
        const int64_t      cost_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
        metrics_->reportRpcMetrics(endpoint, cost_us, request_bytes, receipt.ByteSizeLong());

        if (!status.ok()) {
            RTP_LLM_LOG_WARNING("[MM-RPC] request_id=%ld endpoint=%s grpc_code=%d elapsed_us=%ld error=%s",
                                request_pb.request_id(),
                                endpoint.c_str(),
                                static_cast<int>(status.error_code()),
                                cost_us,
                                status.error_message().c_str());
            metrics_->reportRpcClientError(
                endpoint, kReasonGrpcError, std::to_string(static_cast<int>(status.error_code())));
            if (auto error_info = parseMultimodalErrorMessage(status.error_message())) {
                return *error_info;
            }
            if (status.error_code() == grpc::StatusCode::UNAVAILABLE
                || status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
                return ErrorInfo(ErrorCode::MM_REMOTE_RPC_FAILED, status.error_message());
            }
            RTP_LLM_LOG_WARNING("unclassified multimodal RPC error is not retryable, grpc code [%d], message [%s]",
                                static_cast<int>(status.error_code()),
                                status.error_message().c_str());
            return ErrorInfo(ErrorCode::UNKNOWN_ERROR, status.error_message());
        }
        return receipt;
    }

    void
    release(const std::string& endpoint, const std::vector<std::string>& handles, DeadlineBudget& budget) override {
        const int64_t remaining      = budget.remainingMs();
        const auto    unique_handles = uniqueReleaseHandles(handles);
        if (unique_handles.empty()) {
            return;
        }
        // Cleanup owns its budget; an expired inference request must not lose the lease.
        if (remaining <= 0
            || retryableRelease(releaseNow(endpoint, unique_handles, std::min(release_timeout_ms_, remaining)))) {
            releaseAsync(endpoint, unique_handles);
        }
    }

    void releaseAsync(std::string endpoint, std::vector<std::string> handles) override {
        handles = uniqueReleaseHandles(handles);
        if (handles.empty()) {
            return;
        }
        const char* drop_reason = nullptr;
        {
            std::lock_guard<std::mutex> lock(release_mutex_);
            if (stopping_) {
                drop_reason = kReasonReleaseClientStop;
            } else {
                std::vector<std::string> new_handles;
                new_handles.reserve(handles.size());
                for (const auto& handle : handles) {
                    if (pending_release_keys_.count(pendingReleaseKey(endpoint, handle)) == 0) {
                        new_handles.push_back(handle);
                    }
                }
                if (new_handles.empty()) {
                    return;
                }
                if (pending_release_keys_.size() + new_handles.size() > kMaxPendingReleaseHandles) {
                    drop_reason = kReasonReleaseQueueFull;
                } else {
                    for (const auto& handle : new_handles) {
                        pending_release_keys_.insert(pendingReleaseKey(endpoint, handle));
                    }
                    release_queue_.push_back(ReleaseTask{endpoint, std::move(new_handles)});
                }
            }
        }
        if (drop_reason != nullptr) {
            metrics_->reportRpcClientError(endpoint, drop_reason);
            RTP_LLM_LOG_WARNING("[MM-RDMA-RELEASE] async release dropped leases=%zu endpoint=%s first_lease=%s "
                                "reason=%s; encoder slot GC will reclaim them",
                                handles.size(),
                                endpoint.c_str(),
                                handles.front().c_str(),
                                drop_reason);
            return;
        }
        release_cv_.notify_one();
    }

private:
    struct ReleaseTask {
        std::string                           endpoint;
        std::vector<std::string>              handles;
        unsigned                              attempts = 0;
        std::chrono::steady_clock::time_point ready_at = std::chrono::steady_clock::now();
    };

    grpc::Status releaseNow(const std::string& endpoint, const std::vector<std::string>& handles, int64_t timeout_ms) {
        if (handles.empty()) {
            return grpc::Status::OK;
        }
        auto connection_status = pool_.getConnection(endpoint);
        if (!connection_status.ok()) {
            RTP_LLM_LOG_WARNING("[MM-RDMA-RELEASE] connection failed endpoint=%s leases=%zu first_lease=%s error=%s",
                                endpoint.c_str(),
                                handles.size(),
                                handles.front().c_str(),
                                connection_status.status().ToString().c_str());
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, connection_status.status().ToString());
        }
        auto&               stub = connection_status.value().stub;
        grpc::ClientContext rel_ctx;
        rel_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms));
        ReleaseLeasePB rel;
        for (const auto& handle : handles) {
            rel.add_lease_id(handle);
        }
        EmptyPB    empty;
        const auto start      = std::chrono::steady_clock::now();
        auto       rel_status = stub->ReleaseRdmaLease(&rel_ctx, rel, &empty);
        if (!rel_status.ok()) {
            RTP_LLM_LOG_WARNING(
                "[MM-RDMA-RELEASE] RPC failed endpoint=%s leases=%zu first_lease=%s "
                "timeout_ms=%ld elapsed_ms=%.3f grpc_code=%d error=%s",
                endpoint.c_str(),
                handles.size(),
                handles.front().c_str(),
                timeout_ms,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
                static_cast<int>(rel_status.error_code()),
                rel_status.error_message().c_str());
        }
        return rel_status;
    }

    void releaseLoop() {
        while (true) {
            ReleaseTask task;
            {
                std::unique_lock<std::mutex> lock(release_mutex_);
                while (true) {
                    if (stopping_) {
                        return;
                    }
                    if (release_queue_.empty()) {
                        release_cv_.wait(lock);
                        continue;
                    }
                    auto       next = std::min_element(release_queue_.begin(),
                                                 release_queue_.end(),
                                                 [](const auto& a, const auto& b) { return a.ready_at < b.ready_at; });
                    const auto now  = std::chrono::steady_clock::now();
                    if (next->ready_at > now) {
                        const auto ready_at = next->ready_at;
                        release_cv_.wait_until(lock, ready_at);
                        continue;
                    }
                    task = std::move(*next);
                    release_queue_.erase(next);
                    // Batch due leases for the same endpoint and retry round. Fresh
                    // cleanup can run while another endpoint is backing off.
                    for (auto it = release_queue_.begin(); it != release_queue_.end();) {
                        if (it->endpoint == task.endpoint && it->attempts == task.attempts && it->ready_at <= now) {
                            task.handles.insert(task.handles.end(),
                                                std::make_move_iterator(it->handles.begin()),
                                                std::make_move_iterator(it->handles.end()));
                            it = release_queue_.erase(it);
                        } else {
                            ++it;
                        }
                    }
                    break;
                }
            }
            const auto status = releaseNow(task.endpoint, task.handles, release_timeout_ms_);
            ++task.attempts;
            std::lock_guard<std::mutex> lock(release_mutex_);
            if (stopping_) {
                return;
            }
            if (retryableRelease(status) && task.attempts < kMaxReleaseAttempts) {
                const auto delay_ms = 100 * (1u << (task.attempts - 1));
                task.ready_at       = std::chrono::steady_clock::now() + std::chrono::milliseconds(delay_ms);
                RTP_LLM_LOG_WARNING("[MM-RDMA-RELEASE] retry scheduled endpoint=%s leases=%zu attempt=%u delay_ms=%u",
                                    task.endpoint.c_str(),
                                    task.handles.size(),
                                    task.attempts + 1,
                                    delay_ms);
                // Retain dedup keys until success or exhaustion, including during backoff.
                release_queue_.push_back(std::move(task));
                continue;
            }
            if (!status.ok()) {
                metrics_->reportRpcClientError(
                    task.endpoint, "release_retry_exhausted", std::to_string(static_cast<int>(status.error_code())));
                RTP_LLM_LOG_WARNING("[MM-RDMA-RELEASE] giving up endpoint=%s leases=%zu attempts=%u; "
                                    "encoder slot GC will reclaim them",
                                    task.endpoint.c_str(),
                                    task.handles.size(),
                                    task.attempts);
            }
            for (const auto& handle : task.handles) {
                pending_release_keys_.erase(pendingReleaseKey(task.endpoint, handle));
            }
        }
    }

    MultimodalRpcPool               pool_;
    MMTransportMetricsPtr           metrics_;
    int64_t                         release_timeout_ms_;
    std::mutex                      release_mutex_;
    std::condition_variable         release_cv_;
    std::deque<ReleaseTask>         release_queue_;
    std::unordered_set<std::string> pending_release_keys_;
    bool                            stopping_ = false;
    std::thread                     release_thread_;
};

class GrpcInlineReceiptReader: public MMTerminalReceiptReader {
public:
    const char* name() const override {
        return "grpc-inline";
    }

    ErrorResult<MultimodalOutput> consumeTerminal(const MultimodalOutputPB& receipt,
                                                  DeliveryContext& /*context*/) override {
        return MultimodalPbConverter::inlineOutputFromPb(receipt);
    }
};

}  // namespace

std::unique_ptr<MMControlClient> createGrpcMMControlClient(MMTransportMetricsPtr metrics, int64_t release_timeout_ms) {
    return std::make_unique<GrpcMMControlClient>(std::move(metrics), release_timeout_ms);
}

std::unique_ptr<MMTerminalReceiptReader> createGrpcInlineReceiptReader() {
    return std::make_unique<GrpcInlineReceiptReader>();
}

}  // namespace rtp_llm
