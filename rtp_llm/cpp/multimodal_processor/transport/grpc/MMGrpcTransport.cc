#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcTransport.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/model_rpc/MultimodalPbConverter.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcError.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/telemetry/RpcTraceHelper.h"

namespace rtp_llm {

namespace {

inline constexpr const char* kReasonConnectionError    = "connection_error";
inline constexpr const char* kReasonGrpcError          = "grpc_error";
inline constexpr const char* kReasonReleaseQueueFull   = "release_queue_full";
inline constexpr const char* kReasonReleaseClientStop  = "release_client_stopping";
inline constexpr size_t      kMaxPendingReleaseHandles = 1024;

opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> startEmbeddingRpcSpan(const std::string& endpoint,
                                                                                   int64_t            request_id) {
    if (!telemetry::TelemetryRuntime::isActive()) {
        return {};
    }
    try {
        auto parent = opentelemetry::trace::Tracer::GetCurrentSpan();
        if (!parent->GetContext().IsValid()) {
            return {};
        }
        const auto separator = endpoint.rfind(':');
        const auto host      = separator == std::string::npos ? endpoint : endpoint.substr(0, separator);
        const auto port      = separator == std::string::npos ? 0 : std::stoi(endpoint.substr(separator + 1));
        return telemetry::startChildClientSpan("rtp_llm.vit.remote_multimodal_embedding",
                                               parent,
                                               host,
                                               port,
                                               request_id,
                                               0,
                                               "MultimodalRpcService/RemoteMultimodalEmbedding");
    } catch (...) {
        return {};
    }
}

class GrpcMMControlClient: public MMControlClient {
public:
    GrpcMMControlClient(MMTransportMetricsPtr metrics, int64_t release_timeout_ms):
        metrics_(std::move(metrics)), release_timeout_ms_(release_timeout_ms) {
        release_thread_ = std::thread(&GrpcMMControlClient::releaseLoop, this);
    }

    ~GrpcMMControlClient() override {
        {
            std::lock_guard<std::mutex> lock(release_mutex_);
            stopping_ = true;
        }
        release_cv_.notify_one();
        release_thread_.join();
    }

    ErrorResult<MultimodalOutputPB>
    request(const std::string& endpoint, MultimodalInputsPB& request_pb, DeadlineBudget& budget) override {
        if (budget.exhausted()) {
            return ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "vit rpc budget exhausted before the call");
        }
        auto connection_status = pool_.getConnection(endpoint);
        if (!connection_status.ok()) {
            metrics_->reportRpcClientError(endpoint, kReasonConnectionError);
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, connection_status.status().ToString());
        }
        auto& connection = connection_status.value();
        auto  stub       = connection.stub;

        if (request_pb.support_rdma()) {
            std::lock_guard<std::mutex> lock(release_mutex_);
            const auto                  it = pending_.find(endpoint);
            if (it != pending_.end() && it->second.handles.size() >= kMaxPendingReleaseHandles) {
                metrics_->reportRpcClientError(endpoint, kReasonReleaseQueueFull);
                return ErrorInfo(ErrorCode::CONCURRENCY_LIMIT_ERROR, "ViT RDMA release backlog is full");
            }
        }

        auto                     server_context = budget.serverContext();
        grpc::PropagationOptions options;
        options.enable_deadline_propagation().enable_cancellation_propagation();
        auto context = server_context ? grpc::ClientContext::FromServerContext(*server_context, options) :
                                        std::make_unique<grpc::ClientContext>();
        if (server_context) {
            for (const auto* key : {"x-dashscope-uid", "x-dashscope-service", "x-dashscope-request-id"}) {
                auto it = server_context->client_metadata().find(key);
                if (it != server_context->client_metadata().end()) {
                    context->AddMetadata(key, std::string(it->second.data(), it->second.size()));
                }
            }
        }
        auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(budget.remainingMs());
        if (server_context) {
            deadline = std::min(deadline, server_context->deadline());
        }
        context->set_deadline(deadline);
        if (request_pb.support_rdma()) {
            context->AddMetadata("x-rtp-rdma-claim", "1");
        }
        MultimodalOutputPB             receipt;
        const int64_t                  request_bytes = request_pb.ByteSizeLong();
        const auto                     start         = std::chrono::steady_clock::now();
        grpc::Status                   status;
        auto                           span = startEmbeddingRpcSpan(endpoint, request_pb.request_id());
        telemetry::GrpcStatusSpanGuard span_guard(span, &status);
        telemetry::injectSpanToClientContext(context.get(), span);
        status = stub->RemoteMultimodalEmbedding(context.get(), request_pb, &receipt);
        span_guard.finish();
        const int64_t cost_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
        metrics_->reportRpcMetrics(endpoint, cost_us, request_bytes, receipt.ByteSizeLong());

        if (!status.ok()) {
            metrics_->reportRpcClientError(
                endpoint, kReasonGrpcError, std::to_string(static_cast<int>(status.error_code())));
            auto error_info = parseMultimodalGrpcError(status);
            if (error_info.code() == ErrorCode::UNKNOWN_ERROR) {
                RTP_LLM_LOG_WARNING("multimodal RPC returned UNKNOWN_ERROR, grpc code [%d], message [%s]",
                                    static_cast<int>(status.error_code()),
                                    status.error_message().c_str());
            }
            return error_info;
        }
        if (receipt.rdma_claim_required()) {
            ReleaseLeasePB claim;
            for (const auto& slot : receipt.output_rdma_slots()) {
                claim.add_lease_id(slot.rdma_descriptor().lease_id());
            }
            grpc::ClientContext claim_context;
            claim_context.set_deadline(deadline);
            EmptyPB    empty;
            const auto claimed = stub->ClaimRdmaLease(&claim_context, claim, &empty);
            if (!claimed.ok()) {
                // This caller has never posted a READ, including when the claim reply got lost.
                releaseAsync(endpoint, std::vector<std::string>(claim.lease_id().begin(), claim.lease_id().end()));
                return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                                 "ViT RDMA lease claim failed: " + claimed.error_message());
            }
        }
        return receipt;
    }

    void release(const std::string& endpoint, const std::vector<std::string>& handles, DeadlineBudget&) override {
        releaseAsync(endpoint, handles);
    }

    void releaseAsync(std::string endpoint, std::vector<std::string> handles) override {
        if (handles.empty()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(release_mutex_);
            auto&                       pending = pending_[endpoint];
            for (auto& handle : handles) {
                if (!handle.empty()) {
                    pending.handles.insert(std::move(handle));
                }
            }
        }
        release_cv_.notify_one();
    }

private:
    using Clock = std::chrono::steady_clock;
    struct Pending {
        std::unordered_set<std::string> handles;
        Clock::time_point               retry_at{};
        unsigned                        failures = 0;
    };

    bool releaseNow(const std::string& endpoint, const std::vector<std::string>& handles) {
        auto connection = pool_.getConnection(endpoint);
        if (!connection.ok()) {
            RTP_LLM_LOG_WARNING("cannot connect to %s to release %zu RDMA leases", endpoint.c_str(), handles.size());
            return false;
        }
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(release_timeout_ms_));
        ReleaseLeasePB request;
        for (const auto& handle : handles) {
            request.add_lease_id(handle);
        }
        EmptyPB empty;
        auto    status = connection.value().stub->ReleaseRdmaLease(&context, request, &empty);
        if (!status.ok()) {
            RTP_LLM_LOG_WARNING("ReleaseRdmaLease(%zu leases) on %s failed: %s",
                                handles.size(),
                                endpoint.c_str(),
                                status.error_message().c_str());
        }
        return status.ok();
    }

    void releaseLoop() {
        std::unique_lock<std::mutex> lock(release_mutex_);
        for (;;) {
            if (pending_.empty()) {
                if (stopping_) {
                    return;
                }
                release_cv_.wait(lock);
                continue;
            }
            const auto                                                    now  = Clock::now();
            auto                                                          wake = Clock::time_point::max();
            std::vector<std::pair<std::string, std::vector<std::string>>> batches;
            for (const auto& [endpoint, pending] : pending_) {
                if (pending.retry_at <= now) {
                    batches.emplace_back(endpoint,
                                         std::vector<std::string>(pending.handles.begin(), pending.handles.end()));
                } else {
                    wake = std::min(wake, pending.retry_at);
                }
            }
            if (batches.empty()) {
                release_cv_.wait_until(lock, wake);
                continue;
            }
            for (const auto& [endpoint, handles] : batches) {
                lock.unlock();
                const bool ok = handles.empty() || releaseNow(endpoint, handles);
                lock.lock();
                auto& pending = pending_.at(endpoint);
                if (ok) {
                    for (const auto& handle : handles) {
                        pending.handles.erase(handle);
                    }
                    pending.failures = 0;
                } else {
                    ++pending.failures;
                    if (stopping_ && pending.failures >= 3) {
                        RTP_LLM_LOG_WARNING("RDMA release shutdown: %zu unacknowledged leases on %s",
                                            pending.handles.size(),
                                            endpoint.c_str());
                        metrics_->reportRpcClientError(endpoint, kReasonReleaseClientStop);
                        pending_.erase(endpoint);
                        continue;
                    }
                    pending.retry_at = Clock::now() + std::chrono::milliseconds(50u << std::min(pending.failures, 5u));
                }
                if (pending.handles.empty()) {
                    pending_.erase(endpoint);
                }
            }
        }
    }

    MultimodalRpcPool                        pool_;
    MMTransportMetricsPtr                    metrics_;
    int64_t                                  release_timeout_ms_;
    std::mutex                               release_mutex_;
    std::condition_variable                  release_cv_;
    std::unordered_map<std::string, Pending> pending_;
    bool                                     stopping_ = false;
    std::thread                              release_thread_;
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
