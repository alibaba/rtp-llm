#pragma once

#include "grpc++/grpc++.h"
#include "autil/LoopThread.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PDCancelRegistry.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

class PrefillRpcServer: public LocalRpcServer {
public:
    PrefillRpcServer() {}
    ~PrefillRpcServer();

    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                      py::object                                             mm_process_engine) override;

    grpc::Status Cancel(grpc::ServerContext* context, const CancelRequestPB* request, CancelResponsePB* response);

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer);

    grpc::Status
    EnqueueBatch(grpc::ServerContext* context, const EnqueueBatchRequestPB* request, EnqueueBatchResponsePB* response);

    ::grpc::Status StartLoad(::grpc::ServerContext*                context,
                             const P2PConnectorStartLoadRequestPB* request,
                             P2PConnectorStartLoadResponsePB*      response);

    ::grpc::Status
    GetPeerInfo(::grpc::ServerContext* context, const GetPeerInfoRequestPB* request, GetPeerInfoResponsePB* response);

private:
    std::unique_ptr<PDCancelRegistry> cancel_registry_;
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
        OnflightScope(PrefillRpcServer* owner, int64_t request_id);
        ~OnflightScope();
        void markStep(GenerateStreamStep s);

    private:
        PrefillRpcServer*                owner_;
        int64_t                          request_id_;
        std::shared_ptr<OnflightTracker> tracker_;
    };

    struct BatchEntry {
        // Reserved before preprocessing; cleared when the local context finishes.
        bool                             reserved{false};
        std::unique_ptr<GenerateContext> context;
        std::weak_ptr<GenerateStream>    stream;
        int64_t                          deadline_ms{0};
        int64_t                          request_deadline_ms{0};
        bool                             attach_pending{false};
        bool                             expired{false};
    };
    void                                        registerBatchAttach(const std::string&       key,
                                                                    const GenerateStreamPtr& stream,
                                                                    int64_t                  timeout_ms,
                                                                    int64_t                  request_deadline_ms,
                                                                    int64_t                  now_ms);
    bool                                        attachBatch(const std::string& key, int64_t now_ms);
    void                                        expireBatchAttachments(int64_t now_ms);
    std::mutex                                  batch_mutex_;
    std::unordered_map<std::string, BatchEntry> batch_entries_;

    void               hangDiagTick();
    void               batchContextCleanupTick();
    void               cancelCleanupTick();
    int64_t            last_cancel_tick_ms_{0};
    static const char* stepName(int step);

    mutable std::mutex                                            onflight_trackers_mutex_;
    std::unordered_map<int64_t, std::shared_ptr<OnflightTracker>> onflight_trackers_;
    autil::LoopThreadPtr                                          hang_diag_thread_;
    autil::LoopThreadPtr                                          batch_context_cleanup_thread_;
    int64_t                                                       hang_diag_warn_threshold_ms_{60 * 1000};
};

}  // namespace rtp_llm
