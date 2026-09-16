#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransport.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace rtp_llm {

int64_t resolveRpcTimeoutMs(const MultimodalInputsPB& request,
                            int64_t                   default_rpc_timeout_ms,
                            int64_t                   rpc_timeout_margin_ms) {
    int64_t max_timeout_ms = 0;
    for (const auto& mm_input : request.multimodal_inputs()) {
        const int64_t configured_timeout_ms = mm_input.mm_preprocess_config().mm_timeout_ms();
        const int64_t resolved_timeout_ms = configured_timeout_ms > 0
                                                ? configured_timeout_ms + rpc_timeout_margin_ms
                                                : default_rpc_timeout_ms;
        max_timeout_ms = std::max(max_timeout_ms, resolved_timeout_ms);
    }
    return max_timeout_ms > 0 ? max_timeout_ms : default_rpc_timeout_ms;
}

// ---- MMTransportMetrics ----

void MMTransportMetrics::reportRpcClientError(const std::string& endpoint,
                                              const std::string& reason,
                                              const std::string& grpc_code) const {
    if (!reporter_) {
        return;
    }
    kmonitor::MetricsTags error_tags;
    error_tags.AddTag("source", kMetricSourceInferenceClient);
    error_tags.AddTag("target", endpoint);
    error_tags.AddTag("reason", reason);
    if (!grpc_code.empty()) {
        error_tags.AddTag("grpc_code", grpc_code);
    }
    reporter_->report(1, "rtp_llm_vit_rpc_client_error_qps", kmonitor::MetricType::QPS, &error_tags, true);
}

void MMTransportMetrics::reportRpcMetrics(const std::string& endpoint,
                                          int64_t            cost_us,
                                          int64_t            request_bytes,
                                          int64_t            response_bytes) const {
    if (!reporter_) {
        return;
    }
    kmonitor::MetricsTags tags;
    tags.AddTag("source", kMetricSourceInferenceClient);
    tags.AddTag("target", endpoint);
    reporter_->report(cost_us, "rtp_llm_vit_rpc_client_rt_us", kmonitor::MetricType::GAUGE, &tags, true);
    reporter_->report(request_bytes, "rtp_llm_vit_rpc_request_bytes", kmonitor::MetricType::GAUGE, &tags, true);
    reporter_->report(response_bytes, "rtp_llm_vit_rpc_response_bytes", kmonitor::MetricType::GAUGE, &tags, true);
}

// ---- MMRemoteOutputTransport ----

ErrorResult<MultimodalOutput> MMRemoteOutputTransport::fetch(const std::string&  endpoint,
                                                             MultimodalInputsPB& request_pb) {
    DeadlineBudget budget(resolveRpcTimeoutMs(request_pb, default_rpc_timeout_ms_, rpc_timeout_margin_ms_));
    DeliveryContext context{endpoint, budget, *control_};

    std::vector<MMReceiptReader*> advertised;
    for (auto& reader : readers_) {
        if (reader->advertise(endpoint, request_pb)) {
            advertised.push_back(reader.get());
        }
    }

    auto receipt = control_->request(endpoint, request_pb, budget);
    if (!receipt.ok()) {
        return receipt.status();
    }

    auto* matched = matchReader(receipt.value());
    if (matched != nullptr
        && std::find(advertised.begin(), advertised.end(), matched) == advertised.end()) {
        // Reject an unadvertised data plane and release its remote resources.
        matched->discard(receipt.value(), context);
        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                         std::string("vit answered with a '") + matched->name()
                             + "' receipt that was never advertised");
    }

    if (matched != nullptr) {
        auto result = matched->consume(receipt.value(), context);
        if (result.succeeded()) {
            return std::move(result.output());
        }
        return result.error();
    }
    if (!advertised.empty()) {
        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                         "vit returned an inline receipt while RDMA transport was required");
    }
    return terminal_->consumeTerminal(receipt.value(), context);
}

MMReceiptReader* MMRemoteOutputTransport::matchReader(const MultimodalOutputPB& receipt) const {
    for (const auto& reader : readers_) {
        if (reader->matches(receipt)) {
            return reader.get();
        }
    }
    return nullptr;
}

}  // namespace rtp_llm
