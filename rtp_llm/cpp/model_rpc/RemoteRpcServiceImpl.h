#pragma once
#include <algorithm>
#include <chrono>
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
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                      py::object                                             mm_process_engine) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        return withRequestAdmission([&]() {
            if (!prefill_server_) {
                auto error_msg = "server not implement GenerateStreamCall";
                RTP_LLM_LOG_ERROR(error_msg);
                return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
            }
            return prefill_server_->GenerateStreamCall(context, request, writer);
        });
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
        auto permit = remote_load_admission_gate_.tryAcquire();
        if (!permit) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "server is quiescing remote loads");
        }
        if (!decode_server_) {
            auto error_msg = "server not implement RemoteLoad";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        return decode_server_->RemoteLoad(context, request, response);
    }

    grpc::Status QuiesceRemoteLoad(grpc::ServerContext*                context,
                                   const RemoteLoadQuiesceRequestPB* request,
                                   RemoteLoadQuiesceResponsePB*      response) override {
        if (!decode_server_) {
            auto error_msg = "server not implement QuiesceRemoteLoad";
            RTP_LLM_LOG_ERROR(error_msg);
            return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        }
        return decode_server_->QuiesceRemoteLoad(context, request, response);
    }

    grpc::Status RemoteGenerate(grpc::ServerContext* context, ServerStream* stream) override {
        return withRequestAdmission([&]() {
            if (!decode_server_) {
                auto error_msg = "server not implement RemoteGenerate";
                RTP_LLM_LOG_ERROR(error_msg);
                return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
            }
            return decode_server_->RemoteGenerate(context, stream);
        });
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
        RTP_LLM_CHECK_WITH_INFO(prepareStop(std::chrono::milliseconds::max()),
                                "remote load leases failed to quiesce before service stop");
        if (prefill_server_) {
            prefill_server_->stop();
        } else if (decode_server_) {
            decode_server_->stop();
        }
    }

    void beginDrain() override {
        LocalRpcServiceImpl::beginDrain();
        remote_load_admission_gate_.close();
    }

    bool prepareStop(std::chrono::milliseconds grace) override {
        const auto deadline = grace == std::chrono::milliseconds::max() ?
                                  std::chrono::steady_clock::time_point::max() :
                                  std::chrono::steady_clock::now() + std::max(grace, std::chrono::milliseconds::zero());
        remote_load_admission_gate_.close();
        if (!remote_load_admission_gate_.waitUntil(deadline)) {
            return false;
        }

        std::chrono::milliseconds remaining = std::chrono::milliseconds::max();
        if (deadline != std::chrono::steady_clock::time_point::max()) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                remaining = std::chrono::milliseconds::zero();
            } else {
                remaining = std::max(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now),
                                     std::chrono::milliseconds(1));
            }
        }
        if (prefill_server_) {
            return prefill_server_->drainRemoteLoads(remaining);
        }
        if (decode_server_) {
            return decode_server_->drainRemoteLoads(remaining);
        }
        return true;
    }

private:
    std::shared_ptr<PrefillRpcServer> prefill_server_;
    std::shared_ptr<DecodeRpcServer>  decode_server_;
    RequestAdmissionGate             remote_load_admission_gate_;
};

}  // namespace rtp_llm
