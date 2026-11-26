#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include <grpcpp/grpcpp.h>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm::block_tree_cache_test {

class CompletionRpcTestServer final: public RpcService::Service {
public:
    using Handler = std::function<grpc::Status(size_t, const FunctionRequestPB&, FunctionResponsePB&)>;

    CompletionRpcTestServer(size_t rank, Handler handler): rank_(rank), handler_(std::move(handler)) {}

    ~CompletionRpcTestServer() {
        if (server_) {
            server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(2));
            server_->Wait();
        }
    }

    bool start() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(this);
        server_ = builder.BuildAndStart();
        return server_ && port_ > 0;
    }

    std::string address() const { return "127.0.0.1:" + std::to_string(port_); }

    grpc::Status ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request,
                                 FunctionResponsePB* response) override {
        return handler_(rank_, *request, *response);
    }

private:
    size_t rank_;
    Handler handler_;
    int port_{0};
    std::unique_ptr<grpc::Server> server_;
};

}
