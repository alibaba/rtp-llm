#pragma once
#include <memory>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/PrefillBatchRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"

namespace rtp_llm {

class RemoteRpcServiceImpl: public LocalRpcServiceImpl {
public:
    RemoteRpcServiceImpl() {}
    ~RemoteRpcServiceImpl() {}
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                      py::object                                             mm_process_engine) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        if (!prefill_server_) {
            auto error_msg = "server not implement GenerateStreamCall";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        return prefill_server_->GenerateStreamCall(context, request, writer);
    }

    grpc::Status
    RemoteFinish(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response) override {
        if (!prefill_server_) {
            auto error_msg = "server not implement RemoteFinish";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        return prefill_server_->RemoteFinish(context, request, response);
    }

    grpc::Status EnqueueBatch(grpc::ServerContext*         context,
                              const EnqueueBatchRequestPB* request,
                              EnqueueBatchResponsePB*      response) override {
        if (!prefill_server_) {
            auto error_msg = "server not implement EnqueueBatch";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, error_msg);
        }
        return prefill_server_->EnqueueBatch(context, request, response);
    }

    grpc::Status EnqueueGroup(grpc::ServerContext*         context,
                              const EnqueueGroupRequestPB* request,
                              EnqueueBatchResponsePB*      response) override {
        if (!prefill_server_) {
            auto error_msg = "server not implement EnqueueGroup";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, error_msg);
        }
        return prefill_server_->EnqueueGroup(context, request, response);
    }

    grpc::Status FetchResponse(grpc::ServerContext*                   context,
                               const FetchRequestPB*                  request,
                               grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        if (!prefill_server_) {
            auto error_msg = "server not implement FetchResponse";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, error_msg);
        }
        return prefill_server_->FetchResponse(context, request, writer);
    }

    // AutoTPM Cancel: Prefill role only; Decode role
    // returns gRPC UNIMPLEMENTED.
    grpc::Status
    Cancel(grpc::ServerContext* context, const CancelRequestPB* request, CancelResponsePB* response) override {
        if (!prefill_server_) {
            return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "Cancel is only supported on Prefill role");
        }
        return prefill_server_->Cancel(context, request, response);
    }

    grpc::Status RemoteLoad(grpc::ServerContext*          context,
                            const BroadcastLoadRequestPB* request,
                            BroadcastLoadResponsePB*      response) override {
        if (!decode_server_) {
            auto error_msg = "server not implement RemoteLoad";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        return decode_server_->RemoteLoad(context, request, response);
    }

    grpc::Status RemoteGenerate(grpc::ServerContext* context, ServerStream* stream) override {
        if (!decode_server_) {
            auto error_msg = "server not implement RemoteGenerate";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        return decode_server_->RemoteGenerate(context, stream);
    }

    grpc::Status BatchGenerateCall(grpc::ServerContext*        context,
                                   const BatchGenerateInputPB* request,
                                   BatchGenerateOutputsPB*     response) override {
        return decode_server_ ? decode_server_->BatchGenerateCall(context, request, response) :
                                LocalRpcServiceImpl::BatchGenerateCall(context, request, response);
    }

    grpc::Status RemoteGenerateNew(grpc::ServerContext*              context,
                                   const RemoteGenerateRequestPBNew* request,
                                   RemoteGenerateResponsePBNew*      response) override {
        auto error_msg = "server not implement RemoteGenerateNew";
        RTP_LLM_LOG_ERROR(error_msg);
        return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, error_msg);
    }

    grpc::Status RemoteStore(grpc::ServerContext*        context,
                             const RemoteStoreRequestPB* request,
                             RemoteStoreResponsePB*      response) override {
        auto error_msg = "server not implement RemoteStore";
        RTP_LLM_LOG_ERROR(error_msg);
        return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, error_msg);
    }

    grpc::Status
    RemoteFinishNew(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response) override {
        auto error_msg = "server not implement RemoteFinishNew";
        RTP_LLM_LOG_ERROR(error_msg);
        return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, error_msg);
    }

    void setExternalDrainCallbacks(std::function<void()> begin, std::function<size_t()> count) override {
        if (decode_server_) {
            decode_server_->setExternalDrainCallbacks(std::move(begin), std::move(count));
        } else {
            LocalRpcServiceImpl::setExternalDrainCallbacks(std::move(begin), std::move(count));
        }
    }

    void beginShutdown() override {
        LocalRpcServiceImpl::beginShutdown();
        if (prefill_server_) {
            prefill_server_->beginShutdown();
        } else if (decode_server_) {
            decode_server_->beginShutdown();
        }
    }

    bool waitForDrain(std::chrono::system_clock::time_point deadline) override {
        return decode_server_ ? decode_server_->waitForDrain(deadline) : LocalRpcServiceImpl::waitForDrain(deadline);
    }

    grpc::Status ExecuteFunction(grpc::ServerContext*     context,
                                 const FunctionRequestPB* request,
                                 FunctionResponsePB*      response) override {
        if (request->has_shutdown_request() && decode_server_) {
            return decode_server_->shutdownStatus(request->shutdown_request(), response->mutable_shutdown_response());
        }
        return LocalRpcServiceImpl::ExecuteFunction(context, request, response);
    }

    void stop() override {
        if (prefill_server_) {
            prefill_server_->beginShutdown();
            prefill_server_->stop();
        } else {
            decode_server_->stop();
        }
    }

private:
    std::shared_ptr<PrefillBatchRpcServer> prefill_server_;
    std::shared_ptr<DecodeRpcServer>       decode_server_;
};

}  // namespace rtp_llm
