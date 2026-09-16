#pragma once

// LLM-side ViT output transport. gRPC carries requests and receipts; readers consume the selected
// data plane.

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

// Keep the direct client and ViT proxy defaults aligned: 120s worker budget plus 5s margin.
constexpr int64_t kDefaultVitRpcTimeoutMs  = 125 * 1000;
constexpr int64_t kVitRpcTimeoutMarginMs   = 5 * 1000;

inline constexpr const char* kMetricSourceInferenceClient = "inference_client";

// Resolve unset inputs to the configured default, then use the longest complete RPC budget.
int64_t resolveRpcTimeoutMs(const MultimodalInputsPB& request,
                            int64_t                   default_rpc_timeout_ms = kDefaultVitRpcTimeoutMs,
                            int64_t                   rpc_timeout_margin_ms  = kVitRpcTimeoutMarginMs);

// Shared deadline for RPC, RDMA reads, release and fallback.
class DeadlineBudget {
public:
    explicit DeadlineBudget(int64_t total_ms):
        deadline_(std::chrono::steady_clock::now() + std::chrono::milliseconds(total_ms)) {}

    int64_t remainingMs() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(deadline_ - std::chrono::steady_clock::now())
            .count();
    }

    bool exhausted() const {
        return remainingMs() <= 0;
    }

private:
    std::chrono::steady_clock::time_point deadline_;
};

// Transport metrics shared by the control client and readers.
class MMTransportMetrics {
public:
    explicit MMTransportMetrics(kmonitor::MetricsReporterPtr reporter): reporter_(std::move(reporter)) {}

    void reportRpcClientError(const std::string& endpoint,
                              const std::string& reason,
                              const std::string& grpc_code = "") const;

    void
    reportRpcMetrics(const std::string& endpoint, int64_t cost_us, int64_t request_bytes, int64_t response_bytes) const;

private:
    kmonitor::MetricsReporterPtr reporter_;
};

using MMTransportMetricsPtr = std::shared_ptr<const MMTransportMetrics>;

// Shared request, receipt and release control channel.
class MMControlClient {
public:
    virtual ~MMControlClient() = default;

    virtual ErrorResult<MultimodalOutputPB>
    request(const std::string& endpoint, MultimodalInputsPB& request_pb, DeadlineBudget& budget) = 0;

    // Best-effort; encoder slot GC is the backstop.
    virtual void
    release(const std::string& endpoint, const std::vector<std::string>& handles, DeadlineBudget& budget) = 0;

    // Success-path release must not delay the consumer. Implementations own all queued data.
    virtual void releaseAsync(std::string endpoint, std::vector<std::string> handles) = 0;
};

struct DeliveryContext {
    const std::string& endpoint;
    DeadlineBudget&    budget;
    MMControlClient&   control;
};

class ConsumeResult {
public:
    static ConsumeResult success(MultimodalOutput output) {
        ConsumeResult result;
        result.succeeded_ = true;
        result.output_    = std::move(output);
        return result;
    }

    static ConsumeResult failure(ErrorInfo error) {
        ConsumeResult result;
        result.error_ = std::move(error);
        return result;
    }

    bool succeeded() const {
        return succeeded_;
    }
    const ErrorInfo& error() const {
        return error_;
    }
    MultimodalOutput& output() {
        return output_;
    }

private:
    bool             succeeded_ = false;
    MultimodalOutput output_;
    ErrorInfo        error_;
};

// Consumer-side data-plane interface.
class MMReceiptReader {
public:
    virtual ~MMReceiptReader() = default;

    virtual const char* name() const = 0;

    virtual bool advertise(const std::string& endpoint, MultimodalInputsPB& request_pb) = 0;

    virtual bool matches(const MultimodalOutputPB& receipt) const = 0;

    virtual ConsumeResult consume(const MultimodalOutputPB& receipt, DeliveryContext& context) = 0;

    // Release resources named by a receipt that cannot be consumed locally.
    virtual void discard(const MultimodalOutputPB& /*receipt*/, DeliveryContext& /*context*/) {}
};

// Always-available fallback. Its narrowed API cannot request another retry.
class MMTerminalReceiptReader: public MMReceiptReader {
public:
    virtual ErrorResult<MultimodalOutput> consumeTerminal(const MultimodalOutputPB& receipt,
                                                          DeliveryContext&          context) = 0;

    bool advertise(const std::string& /*endpoint*/, MultimodalInputsPB& /*request_pb*/) override {
        return true;
    }
    bool matches(const MultimodalOutputPB& /*receipt*/) const override {
        return true;
    }

    ConsumeResult consume(const MultimodalOutputPB& receipt, DeliveryContext& context) final {
        auto result = consumeTerminal(receipt, context);
        if (!result.ok()) {
            return ConsumeResult::failure(result.status());
        }
        return ConsumeResult::success(std::move(result.value()));
    }
};

// Owns the control channel and the explicitly configured receipt reader.
class MMRemoteOutputTransport {
public:
    MMRemoteOutputTransport(std::vector<std::unique_ptr<MMReceiptReader>> readers,
                            std::unique_ptr<MMTerminalReceiptReader>     terminal,
                            std::unique_ptr<MMControlClient>             control,
                            int64_t default_rpc_timeout_ms = kDefaultVitRpcTimeoutMs,
                            int64_t rpc_timeout_margin_ms  = kVitRpcTimeoutMarginMs):
        readers_(std::move(readers)),
        terminal_(std::move(terminal)),
        control_(std::move(control)),
        default_rpc_timeout_ms_(default_rpc_timeout_ms),
        rpc_timeout_margin_ms_(rpc_timeout_margin_ms) {}

    ErrorResult<MultimodalOutput> fetch(const std::string& endpoint, MultimodalInputsPB& request_pb);

private:
    // nullptr means the receipt is inline.
    MMReceiptReader* matchReader(const MultimodalOutputPB& receipt) const;

    std::vector<std::unique_ptr<MMReceiptReader>> readers_;
    std::unique_ptr<MMTerminalReceiptReader>      terminal_;
    std::unique_ptr<MMControlClient>              control_;
    int64_t                                       default_rpc_timeout_ms_;
    int64_t                                       rpc_timeout_margin_ms_;
};

}  // namespace rtp_llm
