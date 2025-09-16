#pragma once
#include <memory>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew.h"

namespace rtp_llm {

class RemoteRpcServiceImpl: public LocalRpcServiceImpl {
public:
    RemoteRpcServiceImpl() {}
    ~RemoteRpcServiceImpl() {}
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        if (decode_entrance_) {
            if (!decode_server_new_) {
                auto error_msg = "server not implement GenerateStreamCall";
                RTP_LLM_LOG_ERROR(error_msg);
                return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
            }
            return decode_server_new_->GenerateStreamCall(context, request, writer);
        }

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

    grpc::Status RemoteGenerateNew(grpc::ServerContext*              context,
                                   const RemoteGenerateRequestPBNew* request,
                                   RemoteGenerateResponsePBNew*      response) override {
        if (!prefill_server_new_) {
            auto error_msg = "server not implement RemoteGenerateNew";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        return prefill_server_new_->RemoteGenerateNew(context, request, response);
    }

    grpc::Status RemoteStore(grpc::ServerContext*        context,
                             const RemoteStoreRequestPB* request,
                             RemoteStoreResponsePB*      response) override {
        if (!prefill_server_new_) {
            auto error_msg = "server not implement RemoteStore";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        return prefill_server_new_->RemoteStore(context, request, response);
    }

    grpc::Status
    RemoteFinishNew(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response) override {
        if (!prefill_server_new_) {
            auto error_msg = "server not implement RemoteFinishNew";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        return prefill_server_new_->RemoteFinish(context, request, response);
    }

    void stop() override {
        if (prefill_server_) {
            prefill_server_->stop();
        } else {
            decode_server_->stop();
        }
    }

private:
    std::shared_ptr<PrefillRpcServer>    prefill_server_;
    std::shared_ptr<DecodeRpcServer>     decode_server_;
    bool                                 decode_entrance_ = false;
    std::shared_ptr<PrefillRpcServerNew> prefill_server_new_;
    std::shared_ptr<DecodeRpcServerNew>  decode_server_new_;
};

}  // namespace rtp_llm