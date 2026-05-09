#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferService.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

namespace {

::mooncake_transfer::MooncakeTransferErrorCodePB toProtoErrorCode(TransferErrorCode error_code) {
    switch (error_code) {
        case TransferErrorCode::OK:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_NONE_ERROR;
        case TransferErrorCode::TIMEOUT:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_CONTEXT_TIMEOUT;
        case TransferErrorCode::CANCELLED:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_TASK_CANCELLED;
        case TransferErrorCode::BUFFER_MISMATCH:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_BUFFER_MISMATCH;
        default:
            return ::mooncake_transfer::MOONCAKE_TRANSFER_UNKNOWN_ERROR;
    }
}

TransferErrorCode fromProtoErrorCode(::mooncake_transfer::MooncakeTransferErrorCodePB error_code) {
    switch (error_code) {
        case ::mooncake_transfer::MOONCAKE_TRANSFER_NONE_ERROR:
            return TransferErrorCode::OK;
        case ::mooncake_transfer::MOONCAKE_TRANSFER_CONTEXT_TIMEOUT:
            return TransferErrorCode::TIMEOUT;
        case ::mooncake_transfer::MOONCAKE_TRANSFER_TASK_CANCELLED:
            return TransferErrorCode::CANCELLED;
        case ::mooncake_transfer::MOONCAKE_TRANSFER_BUFFER_MISMATCH:
            return TransferErrorCode::BUFFER_MISMATCH;
        default:
            return TransferErrorCode::UNKNOWN;
    }
}

}  // namespace

MooncakeTransferService::MooncakeTransferService(IMooncakeControlPlaneHandler* handler,
                                                 const kmonitor::MetricsReporterPtr& metrics_reporter):
    handler_(handler), metrics_reporter_(metrics_reporter) {}

void MooncakeTransferService::prepare(::google::protobuf::RpcController* controller,
                                      const ::mooncake_transfer::MooncakePrepareRequest* request,
                                      ::mooncake_transfer::MooncakePrepareResponse* response,
                                      ::google::protobuf::Closure* done) {
    (void)controller;

    if (!handler_) {
        response->set_error_code(::mooncake_transfer::MOONCAKE_TRANSFER_UNKNOWN_ERROR);
        response->set_error_message("MooncakeTransferService handler is null");
        done->Run();
        return;
    }

    MooncakeRemoteDescriptor descriptor;
    TransferErrorCode        error_code = TransferErrorCode::OK;
    std::string              error_message;
    const bool prepared = handler_->prepareDescriptor(
        request->has_unique_key() ? request->unique_key() : std::string(),
        request->has_deadline_ms() ? request->deadline_ms() : 0,
        &descriptor,
        &error_code,
        &error_message);

    response->set_error_code(toProtoErrorCode(prepared ? TransferErrorCode::OK : error_code));
    if (!error_message.empty()) {
        response->set_error_message(error_message);
    }
    if (prepared) {
        response->set_segment_name(descriptor.segment_name);
        for (const auto& block : descriptor.blocks) {
            auto* proto_block = response->add_descriptors();
            proto_block->set_cache_key(block.cache_key);
            proto_block->set_block_index(block.block_index);
            proto_block->set_target_addr(block.target_addr);
            proto_block->set_len(block.len);
        }
    }
    done->Run();
}

void MooncakeTransferService::finish(::google::protobuf::RpcController* controller,
                                     const ::mooncake_transfer::MooncakeFinishRequest* request,
                                     ::mooncake_transfer::MooncakeFinishResponse* response,
                                     ::google::protobuf::Closure* done) {
    (void)controller;

    if (!handler_) {
        response->set_error_code(::mooncake_transfer::MOONCAKE_TRANSFER_UNKNOWN_ERROR);
        response->set_error_message("MooncakeTransferService handler is null");
        done->Run();
        return;
    }

    TransferErrorCode response_error_code = TransferErrorCode::OK;
    std::string       response_error_message;
    const bool finished = handler_->finishTransfer(
        request->has_unique_key() ? request->unique_key() : std::string(),
        request->has_success() ? request->success() : false,
        request->has_error_code() ? fromProtoErrorCode(request->error_code()) : TransferErrorCode::UNKNOWN,
        request->has_error_message() ? request->error_message() : std::string(),
        &response_error_code,
        &response_error_message);

    response->set_error_code(toProtoErrorCode(finished ? TransferErrorCode::OK : response_error_code));
    if (!response_error_message.empty()) {
        response->set_error_message(response_error_message);
    }
    done->Run();
}

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm
