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
                      py::object                                             mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        // 记录query access log (只记录请求到达时间)
        if (decode_entrance_ && decode_server_new_) {
            LOG_RPC_QUERY_INFO(decode_server_new_->getRpcAccessLogConfig(), GenerateStreamCall, request);
        } else if (prefill_server_) {
            LOG_RPC_QUERY_INFO(prefill_server_->getRpcAccessLogConfig(), GenerateStreamCall, request);
        }

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
        // 记录query access log (只记录请求到达时间)
        LOG_RPC_QUERY_INFO(prefill_server_->getRpcAccessLogConfig(), RemoteFinish, request);

        if (!prefill_server_) {
            auto error_msg = "server not implement RemoteFinish";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        auto status = prefill_server_->RemoteFinish(context, request, response);

        // 记录access log (记录请求和响应，无论成功与否)
        LOG_RPC_ACCESS_INFO(prefill_server_->getRpcAccessLogConfig(), RemoteFinish, request, response, status);

        return status;
    }

    grpc::Status RemoteLoad(grpc::ServerContext*          context,
                            const BroadcastLoadRequestPB* request,
                            BroadcastLoadResponsePB*      response) override {
        // 记录query access log (只记录请求到达时间)
        LOG_RPC_QUERY_INFO(decode_server_->getRpcAccessLogConfig(), RemoteLoad, request);

        if (!decode_server_) {
            auto error_msg = "server not implement RemoteLoad";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        auto status = decode_server_->RemoteLoad(context, request, response);

        // 记录access log (记录请求和响应，无论成功与否)
        LOG_RPC_ACCESS_INFO(decode_server_->getRpcAccessLogConfig(), RemoteLoad, request, response, status);

        return status;
    }

    grpc::Status RemoteGenerate(grpc::ServerContext* context, ServerStream* stream) override {
        // For streaming calls, we only log the query access log (只记录请求到达时间)
        // The server implementation handles the detailed logging
        // Since we don't have access to the request object directly in this method signature,
        // we can't log the full request details here

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
        // 记录query access log (只记录请求到达时间)
        LOG_RPC_QUERY_INFO(prefill_server_new_->getRpcAccessLogConfig(), RemoteGenerateNew, request);

        if (!prefill_server_new_) {
            auto error_msg = "server not implement RemoteGenerateNew";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        auto status = prefill_server_new_->RemoteGenerateNew(context, request, response);

        // 记录access log (记录请求和响应，无论成功与否)
        LOG_RPC_ACCESS_INFO(prefill_server_new_->getRpcAccessLogConfig(), RemoteGenerateNew, request, response, status);

        return status;
    }

    grpc::Status RemoteStore(grpc::ServerContext*        context,
                             const RemoteStoreRequestPB* request,
                             RemoteStoreResponsePB*      response) override {
        // 记录query access log (只记录请求到达时间)
        LOG_RPC_QUERY_INFO(prefill_server_new_->getRpcAccessLogConfig(), RemoteStore, request);

        if (!prefill_server_new_) {
            auto error_msg = "server not implement RemoteStore";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        auto status = prefill_server_new_->RemoteStore(context, request, response);

        // 记录access log (记录请求和响应，无论成功与否)
        LOG_RPC_ACCESS_INFO(prefill_server_new_->getRpcAccessLogConfig(), RemoteStore, request, response, status);

        return status;
    }

    grpc::Status
    RemoteFinishNew(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response) override {
        // 记录query access log (只记录请求到达时间)
        LOG_RPC_QUERY_INFO(prefill_server_new_->getRpcAccessLogConfig(), RemoteFinishNew, request);

        if (!prefill_server_new_) {
            auto error_msg = "server not implement RemoteFinishNew";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        auto status = prefill_server_new_->RemoteFinish(context, request, response);

        // 记录access log (记录请求和响应，无论成功与否)
        LOG_RPC_ACCESS_INFO(prefill_server_new_->getRpcAccessLogConfig(), RemoteFinishNew, request, response, status);

        return status;
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