#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcTransport.h"

#include <algorithm>
#include <chrono>
#include <utility>

#include "rtp_llm/cpp/model_rpc/MMRpcCodec.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

inline constexpr const char* kReasonConnectionError = "connection_error";
inline constexpr const char* kReasonGrpcError       = "grpc_error";

class GrpcMMControlClient: public MMControlClient {
public:
    GrpcMMControlClient(MMTransportMetricsPtr metrics, int64_t release_timeout_ms):
        metrics_(std::move(metrics)), release_timeout_ms_(release_timeout_ms) {}

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
        auto connection_status = pool_.getConnection(endpoint);
        if (!connection_status.ok()) {
            return;
        }
        auto&               stub = connection_status.value().stub;
        grpc::ClientContext rel_ctx;
        rel_ctx.set_deadline(std::chrono::system_clock::now()
                             + std::chrono::milliseconds(std::min(release_timeout_ms_, remaining)));
        ReleaseEmbeddingPB rel;
        for (const auto& handle : handles) {
            rel.add_handle(handle);
        }
        EmptyPB empty;
        auto    rel_status = stub->ReleaseMultimodalEmbedding(&rel_ctx, rel, &empty);
        if (!rel_status.ok()) {
            RTP_LLM_LOG_WARNING("ReleaseMultimodalEmbedding(%zu handles) failed: %s",
                                handles.size(),
                                rel_status.error_message().c_str());
        }
    }

private:
    MultimodalRpcPool     pool_;
    MMTransportMetricsPtr metrics_;
    int64_t               release_timeout_ms_;
};

class GrpcInlineReceiptReader: public MMTerminalReceiptReader {
public:
    const char* name() const override {
        return "grpc-inline";
    }

    ErrorResult<MultimodalOutput> consumeTerminal(const MultimodalOutputPB& receipt,
                                                  DeliveryContext& /*context*/) override {
        return MMRpcCodec::transMMOutput(&receipt);
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
