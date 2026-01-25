#include "rtp_llm/cpp/cache/connector/p2p/test/TestRpcServer.h"
#include <thread>
#include <atomic>

namespace rtp_llm {

::grpc::Status TestRpcService::ExecuteFunction(::grpc::ServerContext*     context,
                                               const ::FunctionRequestPB* request,
                                               ::FunctionResponsePB*      response) {
    // 处理 p2p_request
    if (request->has_p2p_request()) {
        // 区分 CANCEL_READ 请求
        if (request->p2p_request().type() == P2PConnectorBroadcastType::CANCEL_READ
            || request->p2p_request().type() == P2PConnectorBroadcastType::CANCEL_HANDLE_READ) {
            broadcast_tp_cancel_call_count_++;
            response->mutable_p2p_response()->set_success(true);
            return ::grpc::Status::OK;
        }
    }

    broadcast_tp_call_count_++;

    if (sleep_millis_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_millis_));
    }

    if (context->IsCancelled()) {
        return ::grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
    }

    // 处理 p2p_request
    if (request->has_p2p_request()) {
        response->mutable_p2p_response()->set_success(p2p_response_success_);
    }

    return rpc_response_status_;
}

::grpc::Status TestRpcService::StartLoad(::grpc::ServerContext*                  context,
                                         const ::P2PConnectorStartLoadRequestPB* request,
                                         ::P2PConnectorStartLoadResponsePB*      response) {
    start_load_call_count_++;

    if (sleep_millis_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_millis_));
    }

    if (context->IsCancelled()) {
        return ::grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
    }

    // 设置响应
    response->set_success(start_load_response_success_);
    response->set_first_generate_token_id(first_generate_token_id_);

    return rpc_response_status_;
}

::grpc::Status TestRpcService::GenerateStreamCall(::grpc::ServerContext*                     context,
                                                  const ::GenerateInputPB*                   request,
                                                  ::grpc::ServerWriter<::GenerateOutputsPB>* writer) {
    generate_stream_call_count_++;

    if (sleep_millis_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_millis_));
    }

    if (context->IsCancelled()) {
        return ::grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
    }

    if (!generate_stream_call_success_) {
        return ::grpc::Status(grpc::StatusCode::INTERNAL, "generate stream call failed");
    }

    // 发送一个响应并完成
    ::GenerateOutputsPB response;
    response.mutable_flatten_output()->add_finished(true);
    if (!writer->Write(response)) {
        return ::grpc::Status(grpc::StatusCode::INTERNAL, "failed to write response");
    }

    return rpc_response_status_;
}

void TestRpcService::setSleepMillis(int ms) {
    sleep_millis_ = ms;
}

void TestRpcService::setP2PResponseSuccess(bool success) {
    p2p_response_success_ = success;
}

void TestRpcService::setStartLoadResponseSuccess(bool success) {
    start_load_response_success_ = success;
}

void TestRpcService::setRpcResponseStatus(const ::grpc::Status& status) {
    rpc_response_status_ = status;
}

void TestRpcService::setFirstGenerateTokenId(int64_t token_id) {
    first_generate_token_id_ = token_id;
}

void TestRpcService::setGenerateStreamCallSuccess(bool success) {
    generate_stream_call_success_ = success;
}

int TestRpcService::getBroadcastTpCallCount() const {
    return broadcast_tp_call_count_.load();
}

int TestRpcService::getBroadcastTpCancelCallCount() const {
    return broadcast_tp_cancel_call_count_.load();
}

int TestRpcService::getStartLoadCallCount() const {
    return start_load_call_count_.load();
}

int TestRpcService::getGenerateStreamCallCount() const {
    return generate_stream_call_count_.load();
}

void TestRpcService::resetCallCounts() {
    broadcast_tp_call_count_        = 0;
    broadcast_tp_cancel_call_count_ = 0;
    start_load_call_count_          = 0;
    generate_stream_call_count_     = 0;
}

bool TestRpcServer::start() {
    if (!service_) {
        return false;
    }

    std::string         bind_addr = "0.0.0.0:0";
    grpc::ServerBuilder builder;
    builder.AddListeningPort(bind_addr, grpc::InsecureServerCredentials(), &listen_port_);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    if (!server_ || listen_port_ == 0) {
        return false;
    }
    return true;
}

int TestRpcServer::listenPort() const {
    return listen_port_;
}

TestRpcService* TestRpcServer::service() const {
    return service_.get();
}

void TestRpcServer::shutdown() {
    if (server_) {
        server_->Shutdown();
        server_->Wait();
        server_.reset();
    }
}

}  // namespace rtp_llm
