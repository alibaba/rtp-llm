#pragma once

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskRecovery.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskReclaimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionForget.h"
namespace rtp_llm {
class BlockTreeCache;
class BlockTreeReclaimer {
public:
    explicit BlockTreeReclaimer(BlockTreeCache& tree);
    ~BlockTreeReclaimer();
    void configure(int rank, int64_t interval_ms, int64_t timeout_ms, int64_t force_release_timeout_ms, std::shared_ptr<BroadcastManager> broadcast);
    template<typename Participant>
    void bindReclaimers(Participant& participant) const {
        participant.setReclaimers(coordinator_, worker_registry_);
    }
    bool start();
    void stop() noexcept;
    bool startFrozenTaskRecovery();
    void stopFrozenTaskRecovery() noexcept;
    std::vector<std::shared_ptr<FrozenTask>> frozenTasks() const;
    TransferTaskRegistry::State localTransferCompletion(const TransferCompletionKey&) const;
    TransferTaskRegistry::ForgetResult localForgetTransferCompletion(const TransferCompletionKey&) const;
    TransferCompletionQueryResult queryTransferCompletion(const FrozenTask::ExecutionInfo&, const std::vector<size_t>&, int) const;
    TransferCompletionForgetResult forgetTransferCompletion(const TransferCompletionTarget&, const std::vector<TransferCompletionKey>&, int) const;
    static bool executeFunction(BlockTreeReclaimer*, int rank, const FunctionRequestPB&, FunctionResponsePB&);
    static bool validConfig(int64_t interval, int64_t timeout, int64_t force_release_timeout);
private:
    void cleanupLoop() noexcept;
    BlockTreeCache& tree_;
    int rank_{0};
    int64_t interval_ms_{0};
    int64_t timeout_ms_{1000};
    int64_t force_release_timeout_ms_{60000};
    std::shared_ptr<BroadcastManager> transfer_broadcast_manager_;
    std::shared_ptr<TransferCompletionCleanup> completion_cleanup_;
    std::shared_ptr<FrozenTaskReclaimer> coordinator_;
    std::shared_ptr<TransferTaskRegistry> worker_registry_{std::make_shared<TransferTaskRegistry>()};
    std::unique_ptr<FrozenTaskRecovery> frozen_task_recovery_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stopped_{false};
    std::thread cleanup_thread_;
};
}
