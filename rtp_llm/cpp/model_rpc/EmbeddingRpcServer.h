#pragma once
#include "grpc++/grpc++.h"
#include <pybind11/pybind11.h>
#include "rtp_llm/cpp/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/cpp/multimodal_processor/LocalMultimodalProcessor.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include <iostream>
#include <memory>
#include <string>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
using namespace pybind11;
namespace rtp_llm {

class EmbeddingRpcServiceImpl: public EmbeddingRpcService::Service {
public:
    explicit EmbeddingRpcServiceImpl(std::shared_ptr<EmbeddingEngine>     engine,
                                     py::object                           pyRenderer,
                                     py::object                           pyHandler,
                                     std::shared_ptr<MultimodalProcessor> mm_processor,
                                     bool                                 need_post_process):
        embedding_engine_(engine),
        pyRenderer_(pyRenderer),
        pyHandler_(pyHandler),
        mm_processor_(mm_processor),
        need_post_process_(need_post_process) {}
    explicit EmbeddingRpcServiceImpl() {};
    grpc::Status embedding(grpc::ServerContext* context, const EmbeddingInputPB* request, EmbeddingOutputPB* response);
    grpc::Status health(grpc::ServerContext* context, const EmbeddingHealthRequestPB* request, EmptyPB* writer);

private:
    std::shared_ptr<EmbeddingEngine>     embedding_engine_ = nullptr;
    pybind11::object                     pyRenderer_;
    pybind11::object                     pyHandler_;
    std::shared_ptr<MultimodalProcessor> mm_processor_      = nullptr;
    bool                                 need_post_process_ = false;
};

}  // namespace rtp_llm
