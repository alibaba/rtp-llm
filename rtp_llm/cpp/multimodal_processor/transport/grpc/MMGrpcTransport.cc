#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcTransport.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iterator>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/model_rpc/MultimodalPbConverter.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

inline constexpr const char* kReasonConnectionError  = "connection_error";
inline constexpr const char* kReasonGrpcError        = "grpc_error";
inline constexpr const char* kReasonReleaseQueueFull   = "release_queue_full";
inline constexpr size_t      kMaxPendingReleaseHandles = 1024;

class GrpcMMControlClient: public MMControlClient {
public:
    GrpcMMControlClient(MMTransportMetricsPtr metrics, int64_t release_timeout_ms):
        metrics_(std::move(metrics)), release_timeout_ms_(release_timeout_ms) {
        release_thread_ = std::thread(&GrpcMMControlClient::releaseLoop, this);
    }

    ~GrpcMMControlClient() override {
        std::deque<ReleaseTask> pending;
        {
            std::lock_guard<std::mutex> lock(release_mutex_);
            stopping_ = true;
            pending.swap(release_queue_);
            pending_release_handles_ = 0;
        }
        release_cv_.notify_one();
        if (release_thread_.joinable()) {
            release_thread_.join();
        }
        // Do not discard leases owned by a client that is shutting down. The worker
        // thread has stopped, so drain the remaining batches synchronously. A failed
        // RPC is still covered by the encoder-side TTL GC.
        for (const auto& task : pending) {
            releaseNow(task.endpoint, task.handles, release_timeout_ms_);
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

    void release(const std::string&              endpoint,
                 const std::vector<std::string>& handles,
                 DeadlineBudget&                 budget) override {
        const int64_t remaining = budget.remainingMs();
        if (handles.empty() || remaining <= 0) {
            return;
        }
        releaseNow(endpoint, handles, std::min(release_timeout_ms_, remaining));
    }

    void releaseAsync(std::string endpoint, std::vector<std::string> handles) override {
        if (handles.empty()) {
            return;
        }
        bool send_sync = false;
        {
            std::lock_guard<std::mutex> lock(release_mutex_);
            if (stopping_) {
                send_sync = true;
            } else if (pending_release_handles_ + handles.size() <= kMaxPendingReleaseHandles) {
                pending_release_handles_ += handles.size();
                release_queue_.push_back(ReleaseTask{std::move(endpoint), std::move(handles)});
            } else {
                // Never drop a successful-read release. Sending synchronously applies
                // backpressure only when the bounded async queue is saturated.
                metrics_->reportRpcClientError(endpoint, kReasonReleaseQueueFull);
                RTP_LLM_LOG_WARNING("multimodal release queue is full; sending %zu handle(s) synchronously",
                                    handles.size());
                send_sync = true;
            }
        }
        if (send_sync) {
            releaseNow(endpoint, handles, release_timeout_ms_);
            return;
        }
        release_cv_.notify_one();
    }

private:
    struct ReleaseTask {
        std::string              endpoint;
        std::vector<std::string> handles;
    };

    void releaseNow(const std::string& endpoint, const std::vector<std::string>& handles, int64_t timeout_ms) {
        if (handles.empty() || timeout_ms <= 0) {
            return;
        }
        auto connection_status = pool_.getConnection(endpoint);
        if (!connection_status.ok()) {
            return;
        }
        auto&               stub = connection_status.value().stub;
        grpc::ClientContext rel_ctx;
        rel_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms));
        ReleaseLeasePB rel;
        for (const auto& handle : handles) {
            rel.add_lease_id(handle);
        }
        EmptyPB empty;
        auto    rel_status = stub->ReleaseRdmaLease(&rel_ctx, rel, &empty);
        if (!rel_status.ok()) {
            RTP_LLM_LOG_WARNING("ReleaseRdmaLease(%zu leases) failed: %s",
                                handles.size(),
                                rel_status.error_message().c_str());
        }
    }

    void releaseLoop() {
        while (true) {
            std::unordered_map<std::string, std::vector<std::string>> batches;
            {
                std::unique_lock<std::mutex> lock(release_mutex_);
                release_cv_.wait(lock, [this] { return stopping_ || !release_queue_.empty(); });
                if (stopping_) {
                    return;
                }
                while (!release_queue_.empty()) {
                    auto task = std::move(release_queue_.front());
                    release_queue_.pop_front();
                    pending_release_handles_ -= task.handles.size();
                    auto& handles = batches[task.endpoint];
                    handles.insert(handles.end(),
                                   std::make_move_iterator(task.handles.begin()),
                                   std::make_move_iterator(task.handles.end()));
                }
            }
            for (const auto& [endpoint, handles] : batches) {
                releaseNow(endpoint, handles, release_timeout_ms_);
            }
        }
    }

    MultimodalRpcPool       pool_;
    MMTransportMetricsPtr   metrics_;
    int64_t                 release_timeout_ms_;
    std::mutex              release_mutex_;
    std::condition_variable release_cv_;
    std::deque<ReleaseTask>  release_queue_;
    size_t                   pending_release_handles_ = 0;
    bool                     stopping_ = false;
    std::thread              release_thread_;
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

std::unique_ptr<MMControlClient> createGrpcMMControlClient(MMTransportMetricsPtr metrics,
                                                           int64_t release_timeout_ms) {
    return std::make_unique<GrpcMMControlClient>(std::move(metrics), release_timeout_ms);
}

std::unique_ptr<MMTerminalReceiptReader> createGrpcInlineReceiptReader() {
    return std::make_unique<GrpcInlineReceiptReader>();
}

}  // namespace rtp_llm
