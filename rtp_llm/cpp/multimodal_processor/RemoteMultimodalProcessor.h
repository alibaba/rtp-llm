#pragma once

#include <memory>
#include <string>
#include <vector>
#include <torch/python.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/MMTransportMode.h"
#include "rtp_llm/cpp/model_rpc/MultimodalPbConverter.h"
#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"
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
                                                      std::string ip_port = "") override {
        if (ip_port == "") {
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, "ip:port is empty in remote multimodal processing");
        }
        auto request_pb = MultimodalPbConverter::inputsToPb(mm_inputs);
        return output_transport_->fetch(ip_port, request_pb);
    }

    ErrorResult<MultimodalOutput>
    V41MultimodalEmbedding(const V41RequestInputs& inputs, const std::string& ip_port, int64_t timeout_ms) override {
        MultimodalInputsPB request;
        request.set_timeout_ms(timeout_ms);
        auto* typed = request.mutable_v41_inputs();
        typed->set_schema_version(1);
        auto token_types = inputs.token_types.to(torch::kCPU).to(torch::kInt32).contiguous();
        auto image_mask  = inputs.image_mask.to(torch::kCPU).to(torch::kBool).contiguous();
        for (int64_t index = 0; index < token_types.numel(); ++index) {
            typed->add_token_types(token_types.data_ptr<int32_t>()[index]);
        }
        for (int64_t index = 0; index < image_mask.numel(); ++index) {
            typed->add_image_mask(image_mask.data_ptr<bool>()[index]);
        }
        for (const auto& image : inputs.images) {
            auto* output = typed->add_images();
            output->set_start(image.start);
            output->set_n_vit_h(image.n_vit_h);
            output->set_n_vit_w(image.n_vit_w);
            TensorPbConvert::torchToPb(output->mutable_patches(), image.patches);
            auto types = image.types.to(torch::kCPU).to(torch::kInt32).contiguous();
            for (int64_t index = 0; index < types.numel(); ++index) {
                output->add_types(types.data_ptr<int32_t>()[index]);
            }
            output->set_content_sha256(image.content_sha256);
            output->set_processor_identity(image.processor_identity);
        }
        return output_transport_->fetch(ip_port, request);
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
