#pragma once
#include <memory>
#include <mutex>
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/ShutdownCoordinator.h"

namespace rtp_llm {

class RemoteRpcServiceImpl: public LocalRpcServiceImpl {
public:
    grpc::Status init(const EngineInitParams&                                params,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                      py::object                                             mm_process_engine) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override;

    grpc::Status BatchGenerateCall(grpc::ServerContext*        context,
                                   const BatchGenerateInputPB* request,
                                   BatchGenerateOutputsPB*     response) override;

    grpc::Status ExecuteFunction(grpc::ServerContext*     context,
                                 const FunctionRequestPB* request,
                                 FunctionResponsePB*      response) override;

    size_t onflightRequestNum() override;
    void   beginShutdown() override;
    bool   waitForDrain(ShutdownCoordinator::Deadline deadline) override;

    grpc::Status EnqueueBatch(grpc::ServerContext*         context,
                              const EnqueueBatchRequestPB* request,
                              EnqueueBatchResponsePB*      response) override {
        if (prefill_server_) {
            return prefill_server_->EnqueueBatch(context, request, response);
        }
        return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "EnqueueBatch requires Prefill role");
    }

    grpc::Status
    Cancel(grpc::ServerContext* context, const CancelRequestPB* request, CancelResponsePB* response) override {
        if (!request || request->request_id() <= 0 || !response) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "cancel request missing request_id");
        }
        if (prefill_server_)
            return prefill_server_->Cancel(context, request, response);
        if (decode_server_)
            return decode_server_->Cancel(context, request, response);
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "PD server is not initialized");
    }

    grpc::Status StartLoad(grpc::ServerContext*                  context,
                           const P2PConnectorStartLoadRequestPB* request,
                           P2PConnectorStartLoadResponsePB*      response) override {
        if (prefill_server_) {
            return prefill_server_->StartLoad(context, request, response);
        }
        return grpc::Status(grpc::StatusCode::INTERNAL, "server not implement StartLoad");
    }

    grpc::Status GetPeerInfo(grpc::ServerContext*        context,
                             const GetPeerInfoRequestPB* request,
                             GetPeerInfoResponsePB*      response) override {
        if (prefill_server_) {
            return prefill_server_->GetPeerInfo(context, request, response);
        }
        return grpc::Status(grpc::StatusCode::INTERNAL, "server not implement GetPeerInfo");
    }

    void stop() override {
        beginShutdown();
        if (prefill_server_) {
            prefill_server_->stop();
        }
        if (decode_server_) {
            decode_server_->stop();
        }
    }

private:
    bool                       tryAdmitRequest();
    void                       finishRequest();
    ShutdownCoordinator::State drainState();

    // Admission and closure share this lock: a request cannot pass the gate
    // while the coordinator observes an idle rank before its server counter rises.
    std::mutex                           admission_mutex_;
    bool                                 draining_{false};
    size_t                               admitted_requests_{0};
    bool                                 requires_group_drain_{false};
    std::unique_ptr<ShutdownCoordinator>  shutdown_coordinator_;
    std::shared_ptr<PrefillRpcServer>      prefill_server_;
    std::shared_ptr<DecodeRpcServer>       decode_server_;
};

}  // namespace rtp_llm
