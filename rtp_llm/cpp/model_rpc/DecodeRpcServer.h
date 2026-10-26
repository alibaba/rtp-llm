#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PDCancelRegistry.h"
#include "rtp_llm/cpp/model_rpc/PrefillServerCaller.h"
#include "autil/LoopThread.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/ShutdownCoordinator.h"

namespace rtp_llm {

std::string makeDecodeEntranceUniqueKey(const std::string& bind_ip, int64_t unique_key_id, int64_t current_time_us);

GenerateInputPB makeDecodeEntranceHandoffRequest(const GenerateInputPB& request, const std::string& handoff_unique_key);

class DecodeRpcServer: public LocalRpcServer {
public:
    DecodeRpcServer() = default;
    ~DecodeRpcServer();

public:
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                      py::object                                             mm_process_engine) override;

    grpc::Status Cancel(grpc::ServerContext* context, const CancelRequestPB* request, CancelResponsePB* response);

    void setExternalDrainCallbacks(std::function<void()> begin, std::function<size_t()> count) {
        external_begin_shutdown_ = std::move(begin);
        external_onflight_       = std::move(count);
    }
    void         beginShutdown();
    bool         waitForDrain(ShutdownCoordinator::Deadline deadline);
    grpc::Status shutdownStatus(const ShutdownRequestPB& request, ShutdownResponsePB* response);

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   server_context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* response_writer);

private:
    std::unique_ptr<PDCancelRegistry> cancel_registry_;
    autil::LoopThreadPtr              cancel_cleanup_thread_;
    Pool<RpcService>                  cancel_pool_;
    void                              cancelCleanupTick();
    grpc::Status                      preparePDRequest(const GenerateInputPB&          request,
                                                       int64_t                         deadline_ms,
                                                       std::shared_ptr<GenerateInput>& input,
                                                       GenerateInputPB&                prefill_request,
                                                       PrefillPeerInfo&                peer_info,
                                                       grpc::ServerContext*            server_context);
    static grpc::Status               parsePrefillDpAddr(const std::string& addr, std::string* ip, uint32_t* port);
    void updateAuxInfo(GenerateOutputsPB& outputs_pb, std::shared_ptr<GenerateStream>& stream) override;

private:
    std::function<void()>                external_begin_shutdown_ = [] {};
    std::function<size_t()>              external_onflight_       = [] { return size_t(0); };
    std::mutex                           admission_mutex_;
    bool                                 draining_ = false;
    std::unique_ptr<ShutdownCoordinator> shutdown_coordinator_;
    bool                                 requires_group_drain_ = false;
    std::atomic<int64_t>                 unique_key_id_{0};
    std::shared_ptr<PrefillServerCaller> prefill_server_caller_;
};

}  // namespace rtp_llm
