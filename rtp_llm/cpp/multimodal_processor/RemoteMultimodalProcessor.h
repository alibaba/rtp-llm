#pragma once

#include <functional>
#include <algorithm>
#include <memory>
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

    static std::unique_ptr<grpc::ClientContext> makeClientContext(grpc::ServerContext* server_context) {
        if (server_context == nullptr) {
            return std::make_unique<grpc::ClientContext>();
        }
        grpc::PropagationOptions options;
        options.enable_deadline_propagation().enable_cancellation_propagation();
        return grpc::ClientContext::FromServerContext(*server_context, options);
    }

    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port    = "",
                                                      int64_t                                     request_id = 0,
                                                      grpc::ServerContext* server_context = nullptr) {
        if (ip_port == "") {
            return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR, "ip:port is empty in remote multimodal processing");
        }
        auto connection_status = pool_.getConnection(ip_port);
        if (!connection_status.ok()) {
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, connection_status.status().ToString());
        }
        auto&              connection = connection_status.value();
        auto               stub       = connection.stub;
        MultimodalOutputPB output_pb;
        auto               context = makeClientContext(server_context);
        auto status = stub->RemoteMultimodalEmbedding(
            context.get(), QueryConverter::transMMInputsPB(mm_inputs, request_id), &output_pb);

        if (!status.ok()) {
            if (status.error_code() == grpc::StatusCode::CANCELLED) {
                return ErrorInfo(ErrorCode::CANCELLED, status.error_message());
            }
            if (status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED) {
                return ErrorInfo(ErrorCode::CONCURRENCY_LIMIT_ERROR, status.error_message());
            }
            if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
                return ErrorInfo(ErrorCode::GENERATE_TIMEOUT, status.error_message());
            }
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, status.error_message());
        }
        return QueryConverter::transMMOutput(&output_pb);
    }
};

}  // namespace rtp_llm
