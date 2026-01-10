#pragma once

#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace py = pybind11;

namespace rtp_llm {

class RemoteMultimodalProcessor: public MultimodalProcessor {
public:
    RemoteMultimodalProcessor(const MMModelConfig& mm_model_config, int64_t max_seq_len):
        MultimodalProcessor(py::none(), mm_model_config, max_seq_len) {}

private:
    MultimodalRpcPool pool_;
    std::string       vit_cluster_name_;

    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port = "") {
        if (ip_port == "") {
            return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR, "ip:port is empty in remote multimodal processing");
        }
        auto connection_status = pool_.getConnection(ip_port);
        if (!connection_status.ok()) {
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, connection_status.status().ToString());
        }
        auto&               connection = connection_status.value();
        auto                stub       = connection.stub;
        MultimodalOutputsPB output_pb;
        grpc::ClientContext context;
        auto status = stub->RemoteMultimodalEmbedding(&context, QueryConverter::transMMInputsPB(mm_inputs), &output_pb);

        if (!status.ok()) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, status.error_message());
        }
        return QueryConverter::transMMOutput(&output_pb);
    }
};

}  // namespace rtp_llm
