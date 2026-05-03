#pragma once
#include <memory>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"

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

    void stop() override {
        if (prefill_server_) {
            prefill_server_->stop();
        } else {
            decode_server_->stop();
        }
    }

    // V1 FlexLB-controlled async path. FetchResponse/Cancel live on Prefill
    // (Decode pushes tokens back through RemoteGenerate, Frontend pulls from
    // Prefill's ResponseBuffer). Enqueue is a DP0 fan-out stub for later.
    grpc::Status FetchResponse(grpc::ServerContext*                   context,
                               const FetchRequestPB*                  request,
                               grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        if (!prefill_server_) {
            return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "FetchResponse requires Prefill role");
        }
        return prefill_server_->FetchResponse(context, request, writer);
    }

    grpc::Status Cancel(grpc::ServerContext* context, const CancelRequestPB* request, EmptyPB* response) override {
        if (!prefill_server_) {
            return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "Cancel requires Prefill role");
        }
        return prefill_server_->Cancel(context, request, response);
    }

    // V1 FlexLB → DP0 batch submit + DP0 → peer single-slot admission. Only Prefill role
    // owns the queue/ResponseBuffer; Decode role sees UNIMPLEMENTED.
    grpc::Status BatchEnqueue(grpc::ServerContext*         context,
                              const BatchEnqueueRequestPB* request,
                              BatchEnqueueResponsePB*      response) override {
        if (!prefill_server_) {
            return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "BatchEnqueue requires Prefill role");
        }
        return prefill_server_->BatchEnqueue(context, request, response);
    }

    grpc::Status
    Enqueue(grpc::ServerContext* context, const EnqueueRequestPB* request, EnqueueResponsePB* response) override {
        if (!prefill_server_) {
            return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "Enqueue requires Prefill role");
        }
        return prefill_server_->Enqueue(context, request, response);
    }

    std::shared_ptr<PrefillRpcServer> prefillServer() const {
        return prefill_server_;
    }

private:
    std::shared_ptr<PrefillRpcServer> prefill_server_;
    std::shared_ptr<DecodeRpcServer>  decode_server_;
};

}  // namespace rtp_llm
