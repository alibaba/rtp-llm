#pragma once

#include <memory>
#include <string>
#include <vector>
#include <torch/python.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/MMRpcCodec.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransportFactory.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace py = pybind11;

namespace rtp_llm {

class RemoteMultimodalProcessor: public MultimodalProcessor {
public:
    RemoteMultimodalProcessor(const MMModelConfig&         mm_model_config,
                              int64_t                      max_seq_len,
                              const MMTransportConfig&     transport_config,
                              kmonitor::MetricsReporterPtr metrics_reporter = nullptr):
        MultimodalProcessor(py::none(), mm_model_config, max_seq_len, metrics_reporter),
        output_transport_(createMMRemoteOutputTransport(transport_config, metrics_reporter)) {}

    ErrorResult<MultimodalOutput>
    MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs, std::string ip_port = "") override {
        if (ip_port == "") {
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, "ip:port is empty in remote multimodal processing");
        }
        auto request_pb = MMRpcCodec::transMMInputsPB(mm_inputs);
        return output_transport_->fetch(ip_port, request_pb);
    }

private:
    std::unique_ptr<MMRemoteOutputTransport> output_transport_;
};

}  // namespace rtp_llm
