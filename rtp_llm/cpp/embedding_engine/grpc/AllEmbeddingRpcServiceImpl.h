#pragma once
#include "grpc++/grpc++.h"
#include <pybind11/pybind11.h>
#include "rtp_llm/proto/all_embedding_rpc_service.grpc.pb.h"
#include "rtp_llm/proto/all_embedding_rpc_service.pb.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "kmonitor/client/MetricsReporter.h"
#include <iostream>
#include <memory>
#include <string>

namespace rtp_llm {
class AllEmbeddingRpcServiceImpl: public AllEmbeddingRpcService::Service {
public:
    explicit AllEmbeddingRpcServiceImpl(std::shared_ptr<EmbeddingEngine> engine, py::object pyRenderer):
        embedding_engine_(engine), pyRenderer_(pyRenderer) {}
    grpc::Status decode(grpc::ServerContext* context, const AllEmbeddingInput* request, AllEmbeddingOutput* writer);

protected:
    void fill(AllEmbeddingOutput* writer, torch::Tensor result);

private:
    std::shared_ptr<EmbeddingEngine> embedding_engine_ = nullptr;
    pybind11::object                 pyRenderer_;
};

}  // namespace rtp_llm
