#pragma once

// LLM-side ViT output transport. gRPC carries requests and receipts; readers consume the selected
// data plane. A failed candidate gets one retry with all candidate capabilities withdrawn.

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

// Keep the direct client and ViT proxy defaults aligned.
constexpr int64_t kDefaultVitRpcTimeoutMs = 30 * 1000;

inline constexpr const char* kMetricSourceInferenceClient = "inference_client";

// Use the longest per-input timeout for the whole request.
int64_t resolveRpcTimeoutMs(const MultimodalInputsPB& request);

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

    void reportCircuitState(const std::string& backend, const std::string& endpoint, bool open) const;

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
};

struct DeliveryContext {
    const std::string& endpoint;
    DeadlineBudget&    budget;
    MMControlClient&   control;
};

// A retry spends one additional ViT forward; failure is terminal.
class ConsumeResult {
public:
    static ConsumeResult success(MultimodalOutput output) {
        ConsumeResult result;
        result.state_  = State::SUCCESS;
        result.output_ = std::move(output);
        return result;
    }

    static ConsumeResult retry(std::string reason) {
        ConsumeResult result;
        result.state_  = State::RETRY;
        result.reason_ = std::move(reason);
        return result;
    }

    static ConsumeResult failure(ErrorInfo error) {
        ConsumeResult result;
        result.state_ = State::FAILURE;
        result.error_ = std::move(error);
        return result;
    }

    bool succeeded() const {
        return state_ == State::SUCCESS;
    }
    bool needsRetry() const {
        return state_ == State::RETRY;
    }
    const std::string& reason() const {
        return reason_;
    }
    const ErrorInfo& error() const {
        return error_;
    }
    MultimodalOutput& output() {
        return output_;
    }

private:
    enum class State { SUCCESS, RETRY, FAILURE };
    State            state_ = State::FAILURE;
    MultimodalOutput output_;
    ErrorInfo        error_;
    std::string      reason_;
};

// Consumer-side data-plane interface.
class MMReceiptReader {
public:
    virtual ~MMReceiptReader() = default;

    virtual const char* name() const = 0;

    // Return false without mutating the request when the data plane is unavailable.
    virtual bool advertise(const std::string& endpoint, MultimodalInputsPB& request_pb) = 0;

    // Undo advertise() before a fallback request.
    virtual void withdraw(MultimodalInputsPB& request_pb) = 0;

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
    void withdraw(MultimodalInputsPB& /*request_pb*/) override {}
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

// Owns the control channel, candidate readers and inline fallback.
class MMRemoteOutputTransport {
public:
    MMRemoteOutputTransport(std::vector<std::unique_ptr<MMReceiptReader>> readers,
                            std::unique_ptr<MMTerminalReceiptReader>     terminal,
                            std::unique_ptr<MMControlClient>             control,
                            MMTransportMetricsPtr                        metrics):
        readers_(std::move(readers)),
        terminal_(std::move(terminal)),
        control_(std::move(control)),
        metrics_(std::move(metrics)) {}

    ErrorResult<MultimodalOutput> fetch(const std::string& endpoint, MultimodalInputsPB& request_pb);

private:
    // Withdraw every candidate and retry once with the terminal path.
    ErrorResult<MultimodalOutput> degradeToTerminal(const std::string&  endpoint,
                                                    MultimodalInputsPB& request_pb,
                                                    DeliveryContext&    context,
                                                    const std::string&  reason);

    // nullptr means the receipt is inline.
    MMReceiptReader* matchReader(const MultimodalOutputPB& receipt) const;

    std::vector<std::unique_ptr<MMReceiptReader>> readers_;
    std::unique_ptr<MMTerminalReceiptReader>      terminal_;
    std::unique_ptr<MMControlClient>              control_;
    MMTransportMetricsPtr                         metrics_;
};

}  // namespace rtp_llm
