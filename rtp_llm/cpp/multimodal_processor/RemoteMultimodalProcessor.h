#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <torch/python.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/MMTransportMode.h"
#include "rtp_llm/cpp/model_rpc/MultimodalPbConverter.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransportFactory.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace py = pybind11;

namespace rtp_llm {

class RemoteMultimodalProcessor: public MultimodalProcessor {
public:
    // Keep the legacy EmbeddingCppEngine path on its existing inline gRPC data plane.
    RemoteMultimodalProcessor(const MMModelConfig&         mm_model_config,
                              int64_t                      max_seq_len,
                              kmonitor::MetricsReporterPtr metrics_reporter = nullptr):
        RemoteMultimodalProcessor(mm_model_config, max_seq_len, grpcOnlyTransportConfig(), metrics_reporter) {}

    RemoteMultimodalProcessor(const MMModelConfig&         mm_model_config,
                              int64_t                      max_seq_len,
                              const MMTransportConfig&     transport_config,
                              kmonitor::MetricsReporterPtr metrics_reporter = nullptr,
                              int                          device_id        = -1):
        RemoteMultimodalProcessor(
            py::none(), mm_model_config, max_seq_len, transport_config, metrics_reporter, device_id) {}

    RemoteMultimodalProcessor(py::object                   prompt_expander,
                              const MMModelConfig&         mm_model_config,
                              int64_t                      max_seq_len,
                              const MMTransportConfig&     transport_config,
                              kmonitor::MetricsReporterPtr metrics_reporter = nullptr,
                              int                          device_id        = -1):
        MultimodalProcessor(py::none(), mm_model_config, max_seq_len, metrics_reporter, prompt_expander),
        output_transport_(createMMRemoteOutputTransport(transport_config, metrics_reporter, device_id)) {}

    RemoteMultimodalProcessor(py::object                               prompt_expander,
                              const MMModelConfig&                     mm_model_config,
                              int64_t                                  max_seq_len,
                              std::unique_ptr<MMRemoteOutputTransport> output_transport):
        MultimodalProcessor(py::none(), mm_model_config, max_seq_len, nullptr, prompt_expander),
        output_transport_(std::move(output_transport)) {}

    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port = "",
                                                      const std::string& rendered_prompt = "") override {
        if (ip_port == "") {
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, "ip:port is empty in remote multimodal processing");
        }
        if (!rendered_prompt.empty() && prompt_expander_.is_none()) {
            return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR,
                             "remote multimodal prompt expansion is not available on the LLM backend");
        }
        auto request_pb    = MultimodalPbConverter::inputsToPb(mm_inputs);
        auto output_result = output_transport_->fetch(ip_port, request_pb);
        if (!output_result.ok()) {
            return output_result.status();
        }
        auto output = std::move(output_result.value());
        if (output.expanded_token_ids.has_value()) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR,
                             "remote multimodal service must not provide expanded prompt token IDs");
        }
        return std::move(output);
    }

private:
    static MMTransportConfig grpcOnlyTransportConfig() {
        MMTransportConfig config;
        config.mode = kMMTransportModeGrpc;
        return config;
    }

    std::unique_ptr<MMRemoteOutputTransport> output_transport_;
};

}  // namespace rtp_llm
