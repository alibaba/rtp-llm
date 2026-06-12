#pragma once

#include "grpc++/grpc++.h"
#include "autil/LoopThread.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace rtp_llm {

class PrefillRpcServerNew2: public RemoteRpcServer {
public:
    PrefillRpcServerNew2() {}
    ~PrefillRpcServerNew2();

    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      py::object                                             mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) override;

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer);

    ::grpc::Status StartLoad(::grpc::ServerContext*                context,
                             const P2PConnectorStartLoadRequestPB* request,
                             P2PConnectorStartLoadResponsePB*      response);

    ::grpc::Status
    GetPeerInfo(::grpc::ServerContext* context, const GetPeerInfoRequestPB* request, GetPeerInfoResponsePB* response);

private:
    // Per-onflight tracker for [HANG-DIAG] watchdog. Each GenerateStreamCall
    // registers an entry on entry and removes it on return; the background
    // hang_diag_thread_ periodically scans for entries that have been alive
    // beyond a threshold and reports them with which step they last reached.
    // This is how we will catch the 5/22 P1-B-style stuck requests (where
    // GenerateStreamCall thread enters but never returns and prints nothing).
    enum class GenerateStreamStep : int {
        kEntry = 0,           // RemoteRpcServiceImpl entry, just past pd_separation/unique_key checks
        kAfterTransQuery,     // QueryConverter::transQuery + mm_processor done
        kAfterEngineEnqueue,  // engine_->enqueue returned (stream created and pushed to scheduler)
        kAfterPollStream,     // pollStreamOutput returned (success or error)
    };

    struct OnflightTracker {
        int64_t          request_id{0};
        int64_t          start_us{0};
        std::atomic<int> step{static_cast<int>(GenerateStreamStep::kEntry)};
    };

    class OnflightScope {
    public:
        OnflightScope(PrefillRpcServerNew2* owner, int64_t request_id);
        ~OnflightScope();
        void markStep(GenerateStreamStep s);

    private:
        PrefillRpcServerNew2*            owner_;
        int64_t                          request_id_;
        std::shared_ptr<OnflightTracker> tracker_;
    };

    void               hangDiagTick();
    static const char* stepName(int step);

    mutable std::mutex                                            onflight_trackers_mutex_;
    std::unordered_map<int64_t, std::shared_ptr<OnflightTracker>> onflight_trackers_;
    autil::LoopThreadPtr                                          hang_diag_thread_;
    int64_t                                                       hang_diag_warn_threshold_ms_{60 * 1000};
    int64_t                                                       local_rpc_port_{0};

    // Pre-computed gRPC addresses for all DP groups (tp_rank=0 entry points).
    // Built once during init() from p2p_worker_addrs, eliminating fragile port
    // arithmetic in GetPeerInfo().  Format: "ip:grpc_port".
    std::vector<std::string> dp_grpc_addrs_;
};

}  // namespace rtp_llm
