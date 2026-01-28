#pragma once
#include <memory>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew2.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"

namespace rtp_llm {

class RemoteRpcServiceImpl: public LocalRpcServiceImpl {
public:
    RemoteRpcServiceImpl() {}
    ~RemoteRpcServiceImpl() {}
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      py::object                                             mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        if (decode_entrance_) {
            if (decode_server_new2_) {
                return decode_server_new2_->GenerateStreamCall(context, request, writer);
            } else if (prefill_server_new2_) {
                return prefill_server_new2_->GenerateStreamCall(context, request, writer);
            } else {
                auto error_msg = "server not implement GenerateStreamCall";
                RTP_LLM_LOG_ERROR(error_msg);
                return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
            }
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

    grpc::Status StartLoad(grpc::ServerContext*                  context,
                           const P2PConnectorStartLoadRequestPB* request,
                           P2PConnectorStartLoadResponsePB*      response) override {
        if (prefill_server_new2_) {
            return prefill_server_new2_->StartLoad(context, request, response);
        }
        auto error_msg = "server not implement StartLoad";
        RTP_LLM_LOG_ERROR(error_msg);
        return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
    }

    void stop() override {
        if (prefill_server_) {
            prefill_server_->stop();
        } else {
            decode_server_->stop();
        }
        if (decode_server_new2_) {
            decode_server_new2_->stop();
        }
        if (prefill_server_new2_) {
            prefill_server_new2_->stop();
        }
    }

private:
    std::shared_ptr<PrefillRpcServer>     prefill_server_;
    std::shared_ptr<DecodeRpcServer>      decode_server_;
    bool                                  decode_entrance_ = false;
    std::shared_ptr<PrefillRpcServerNew2> prefill_server_new2_;
    std::shared_ptr<DecodeRpcServerNew2>  decode_server_new2_;
};

}  // namespace rtp_llm