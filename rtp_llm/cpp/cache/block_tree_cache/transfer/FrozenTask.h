#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionCleanup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

struct TransferCompletionQueryResult;

class FrozenTask {
public:
    struct TargetBlock {
        std::shared_ptr<IBlockPool> pool;
        BlockIdxType block{NULL_BLOCK_IDX};
    };

    struct ExecutionInfo {
        RegistryType registry_type{RegistryType::BLOCK_TRANSFER};
        std::string trace_id;
        bool uses_rpc{false};
        size_t rank_count{0};
        std::vector<TargetBlock> target_blocks;
    };

    enum class State { PREPARED, FROZEN, RELEASING, RELEASED };

    struct Snapshot {
        State   state{State::PREPARED};
        int64_t created_at_us{0};
        int64_t frozen_at_us{0};
        int64_t released_at_us{0};
        size_t  target_block_count{0};
    };

    struct RecoverySnapshot {
        uint64_t                                             query_count{0};
        int64_t                                              last_query_at_us{0};
        std::shared_ptr<const TransferCompletionQueryResult> last_query;
    };

private:
    struct TargetPool {
        std::shared_ptr<IBlockPool> pool;
        BlockIdList                 blocks;
        std::function<void()>      release;
    };

public:
    class PreparedExecution {
    public:
        PreparedExecution(PreparedExecution&&) noexcept            = default;
        PreparedExecution& operator=(PreparedExecution&&) noexcept = default;
        PreparedExecution(const PreparedExecution&)                = delete;
        PreparedExecution& operator=(const PreparedExecution&)     = delete;

    private:
        friend class FrozenTask;
        PreparedExecution(ExecutionInfo execution, std::vector<TargetPool> targets, size_t target_block_count):
            execution_(std::move(execution)),
            targets_(std::move(targets)),
            target_block_count_(target_block_count) {}

        ExecutionInfo           execution_;
        std::vector<TargetPool> targets_;
        size_t                  target_block_count_{0};
    };

    // Complete all target deduplication and release preparation before freeze.
    // The caller retains ordinary ownership until freeze and retention finish.
    static PreparedExecution prepare(ExecutionInfo execution);

    FrozenTask(PreparedExecution&& prepared, TransferResult failure,
               TransferCompletionCleanup::Prepared cleanup = {});
    ~FrozenTask() = default;
    FrozenTask(const FrozenTask&)            = delete;
    FrozenTask& operator=(const FrozenTask&) = delete;

    void freeze() noexcept;
    bool completeRecovery(TransferCompletionCleanup& cleanup_queue) noexcept;

    // Construct immutable query results before calling this method. Recording
    // does not change the original failure or release any target references.
    // A null result is ignored.
    void             recordRecoveryQuery(std::shared_ptr<const TransferCompletionQueryResult> query) noexcept;
    RecoverySnapshot recoverySnapshot() const noexcept;

    const ExecutionInfo& execution() const noexcept {
        return execution_;
    }
    const TransferResult& failure() const noexcept {
        return failure_;
    }
    Snapshot    snapshot() const noexcept;

private:
    const ExecutionInfo           execution_;
    const std::vector<TargetPool> targets_;
    const TransferResult          failure_;
    const size_t                  target_block_count_;
    const int64_t                 created_at_us_;
    TransferCompletionCleanup::Prepared cleanup_;
    mutable std::mutex            mutex_;
    State                         state_{State::PREPARED};
    int64_t                       frozen_at_us_{0};
    int64_t                       released_at_us_{0};
    RecoverySnapshot              recovery_;
};

}  // namespace rtp_llm
