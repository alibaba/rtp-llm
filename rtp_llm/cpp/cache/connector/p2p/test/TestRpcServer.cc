#include "rtp_llm/cpp/cache/connector/p2p/test/TestRpcServer.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/unknown_field_set.h>

#include <cstring>
#include <thread>
#include <atomic>

namespace rtp_llm {

namespace {

std::string encodePackedVarints(std::initializer_list<uint32_t> values) {
    std::string bytes;
    bytes.reserve(values.size() * 2);
    for (uint32_t value : values) {
        uint8_t  buffer[10];
        uint8_t* ptr = google::protobuf::io::CodedOutputStream::WriteVarint32ToArray(value, buffer);
        bytes.append(reinterpret_cast<const char*>(buffer), ptr - buffer);
    }
    return bytes;
}

std::string encodeFp32TensorBytes(std::initializer_list<float> values) {
    std::string bytes;
    bytes.resize(values.size() * sizeof(float));
    std::memcpy(&bytes[0], values.begin(), bytes.size());
    return bytes;
}

}  // namespace

::grpc::Status TestRpcService::ExecuteFunction(::grpc::ServerContext*     context,
                                               const ::FunctionRequestPB* request,
                                               ::FunctionResponsePB*      response) {
    // 处理 p2p_request
    if (request->has_p2p_request()) {
        // 区分 CANCEL_READ 请求
        if (request->p2p_request().type() == P2PConnectorBroadcastType::CANCEL_READ
            || request->p2p_request().type() == P2PConnectorBroadcastType::CANCEL_HANDLE_READ) {
            broadcast_tp_cancel_call_count_++;
            auto* p2p_response = response->mutable_p2p_response();
            p2p_response->set_error_code(ErrorCodePB::NONE_ERROR);
            p2p_response->set_error_message("");
            return ::grpc::Status::OK;
        }
    }

    broadcast_tp_call_count_++;

    if (request->has_p2p_request()) {
        std::lock_guard<std::mutex> lock(last_broadcast_tp_request_mutex_);
        last_broadcast_tp_request_.CopyFrom(request->p2p_request());
    }

    if (sleep_millis_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_millis_));
    }

    if (context->IsCancelled()) {
        return ::grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
    }

    // 处理 p2p_request
    if (request->has_p2p_request()) {
        auto* p2p_response = response->mutable_p2p_response();
        if (p2p_response_success_) {
            p2p_response->set_error_code(ErrorCodePB::NONE_ERROR);
            p2p_response->set_error_message("");
        } else {
            p2p_response->set_error_code(ErrorCodePB::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED);
            p2p_response->set_error_message("test p2p response failed");
        }
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

    if (start_load_app_error_pb_ != ErrorCodePB::NONE_ERROR) {
        response->set_error_code(start_load_app_error_pb_);
        response->set_error_message(start_load_app_error_message_);
        return rpc_response_status_;
    }

    // 设置响应
    if (start_load_response_success_) {
        response->set_error_code(ErrorCodePB::NONE_ERROR);
        if (use_legacy_start_load_response_) {
            auto* unknown_fields = response->GetReflection()->MutableUnknownFields(response);
            unknown_fields->AddVarint(1, first_generate_token_id_);
            unknown_fields->AddVarint(2, 10);
            unknown_fields->AddVarint(3, 4);
            unknown_fields->AddVarint(4, 6);
            unknown_fields->AddVarint(11, 2);
            unknown_fields->AddLengthDelimited(5, encodePackedVarints({7, 8}));

            TensorPB tensor_pb;
            tensor_pb.set_data_type(TensorPB::FP32);
            tensor_pb.add_shape(1);
            tensor_pb.add_shape(2);
            std::string fp32_bytes = encodeFp32TensorBytes({0.1f, 0.2f});
            tensor_pb.set_fp32_data(fp32_bytes.data(), fp32_bytes.size());
            std::string tensor_bytes;
            tensor_pb.SerializeToString(&tensor_bytes);
            unknown_fields->AddLengthDelimited(6, tensor_bytes);
            unknown_fields->AddLengthDelimited(7, tensor_bytes);
            unknown_fields->AddLengthDelimited(8, encodePackedVarints({11, 12}));
        } else {
            auto* payload = response->mutable_payload();
            payload->set_has_first_generate_token(true);
            payload->set_first_generate_token_id(first_generate_token_id_);
        }
    } else {
        response->set_error_code(ErrorCodePB::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED);
        response->set_error_message("test start load response failed");
    }

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

void TestRpcService::setStartLoadApplicationError(ErrorCodePB code, const std::string& message) {
    start_load_app_error_pb_      = code;
    start_load_app_error_message_ = message;
}

void TestRpcService::setRpcResponseStatus(const ::grpc::Status& status) {
    rpc_response_status_ = status;
}

void TestRpcService::setFirstGenerateTokenId(int64_t token_id) {
    first_generate_token_id_ = token_id;
}

void TestRpcService::setUseLegacyStartLoadResponse(bool use_legacy) {
    use_legacy_start_load_response_ = use_legacy;
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

P2PConnectorBroadcastTpRequestPB TestRpcService::getLastBroadcastTpRequest() const {
    std::lock_guard<std::mutex> lock(last_broadcast_tp_request_mutex_);
    return last_broadcast_tp_request_;
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
    {
        std::lock_guard<std::mutex> lock(last_broadcast_tp_request_mutex_);
        last_broadcast_tp_request_.Clear();
    }
    start_load_call_count_          = 0;
    generate_stream_call_count_     = 0;
    use_legacy_start_load_response_ = false;
    start_load_app_error_pb_        = ErrorCodePB::NONE_ERROR;
    start_load_app_error_message_.clear();
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
