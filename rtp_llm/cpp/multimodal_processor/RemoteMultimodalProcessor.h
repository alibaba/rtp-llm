#pragma once

#include <memory>
#include <string>
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
        MultimodalProcessor(py::none(), mm_model_config, max_seq_len, metrics_reporter),
        output_transport_(createMMRemoteOutputTransport(transport_config, metrics_reporter, device_id)) {}

    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port    = "",
                                                      int64_t                                     request_id = 0,
                                                      grpc::ServerContext*      server_context               = nullptr,
                                                      const MultimodalInputsPB* wire_inputs = nullptr) override {
        if (ip_port == "") {
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, "ip:port is empty in remote multimodal processing");
        }
        auto request_pb = wire_inputs ? *wire_inputs : MultimodalPbConverter::inputsToPb(mm_inputs, request_id);
        request_pb.set_request_id(request_id);
        // RDMA capability belongs to this consumer and is advertised by its transport.
        request_pb.clear_support_rdma();
        if (wire_inputs) {
            // Generate RPCs carry media references. As in QueryConverter, ignore
            // tensor payloads supplied alongside them.
            for (auto& item : *request_pb.mutable_multimodal_inputs()) {
                item.clear_multimodal_tensor();
            }
        }
        return output_transport_->fetch(ip_port, request_pb, server_context);
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
