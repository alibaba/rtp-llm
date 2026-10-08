#pragma once

#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

// Decode retains the control record until local and Prefill cleanup are complete.
// Admission and the cancel latch share a lock, including cancel-before-enqueue.
class PDCancelRegistry {
public:
    struct Entry {
        explicit Entry(TaskIdentity task): identity(task) {}
        TaskIdentity      identity;
        std::string       unique_key;
        std::string       downstream_address;
        int64_t           deadline_ms = 0;
        ErrorInfo         cancel_reason;
        std::atomic<bool> canceled{false};
        GenerateStreamPtr stream;
        std::atomic<bool> local_done{false};
        bool              downstream_done = false;
        std::atomic<bool> terminal{false};
        int64_t           retain_until_ms = 0;
    };
    using Handle = std::shared_ptr<Entry>;

    explicit PDCancelRegistry(std::shared_ptr<RpcServerRuntimeMeta> meta): meta_(std::move(meta)) {}

    ErrorInfo           admit(const GenerateInputPB& request,
                              const std::string&     key,
                              const std::string&     downstream_address,
                              int64_t                deadline_ms,
                              Handle&                handle);
    Handle              find(int64_t request_id);
    void                attach(const Handle& handle, const GenerateStreamPtr& stream);
    void                finishLocal(const Handle& handle);
    CancelStatusPB      cancel(int64_t request_id, const ErrorInfo& reason);
    bool                isCanceled(const std::string& unique_key);
    std::vector<Handle> pending();
    void                finishDownstream(const Handle& handle);
    bool                complete(const Handle& handle);

    void beginRead(const std::string& key);
    void endRead(const std::string& key);
    class ReadGuard {
    public:
        ReadGuard(PDCancelRegistry& registry, std::string key): registry_(registry), key_(std::move(key)) {
            registry_.beginRead(key_);
        }
        ~ReadGuard() {
            registry_.endRead(key_);
        }

    private:
        PDCancelRegistry& registry_;
        std::string       key_;
    };

    // Declare before GenerateContext so local completion is observed only after
    // context cleanup, including Prefill RPC joins and P2P load cancellation.
    class CallGuard {
    public:
        CallGuard(PDCancelRegistry& registry, Handle handle): registry_(registry), handle_(std::move(handle)) {}
        ~CallGuard() {
            registry_.finishLocal(handle_);
        }

    private:
        PDCancelRegistry& registry_;
        Handle            handle_;
    };

private:
    void                                    trimLocked(int64_t now_ms);
    static constexpr int64_t                kRetentionMs = 10 * 60 * 1000;
    std::mutex                              mutex_;
    std::unordered_map<int64_t, Handle>     entries_;
    struct CancelFence {
        int64_t   expires_at_ms;
        ErrorInfo reason;
    };
    std::unordered_map<int64_t, CancelFence> absent_fences_;
    std::unordered_map<std::string, size_t> active_reads_;
    std::shared_ptr<RpcServerRuntimeMeta>   meta_;
};

}  // namespace rtp_llm
