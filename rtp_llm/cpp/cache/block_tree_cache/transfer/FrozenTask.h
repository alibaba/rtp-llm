#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

// One coordinator execution trace owns one hold for each distinct target slot.
// Sources are immutable diagnostics only. No tree node or business settlement
// callback survives here, and destruction never releases frozen references.
class FrozenTask {
public:
    struct ExpectedPeer {
        size_t      rank{0};
        std::string address;
    };

    struct BlockInfo {
        std::shared_ptr<IBlockPool> pool;
        BlockIdxType                block{NULL_BLOCK_IDX};
        size_t                      member_index{0};
        std::string                 pool_name;
    };

    struct DescriptorInfo {
        size_t                 group_set_id{0};
        size_t                 path_index{0};
        Tier                   source_tier{Tier::NONE};
        Tier                   target_tier{Tier::NONE};
        std::vector<BlockInfo> sources;
        std::vector<BlockInfo> targets;
        std::optional<size_t>  rank{};
        std::string            group_tag{};
        std::string            source_uri{};
    };

    struct ExecutionInfo {
        TransferExecutionTrace      trace_id{uint64_t{0}};
        std::string                 operation;
        std::optional<int64_t>      request_id;
        uint64_t                    workflow_id{0};
        uint64_t                    parent_trace_id{0};
        bool                        remote{false};
        size_t                      expected_rank_count{0};
        std::vector<ExpectedPeer>   expected_peers;
        std::vector<DescriptorInfo> descriptors;
    };

    enum class State { PREPARED, FROZEN, RELEASING, RELEASED };

    struct Snapshot {
        State   state{State::PREPARED};
        int64_t created_at_us{0};
        int64_t frozen_at_us{0};
        int64_t released_at_us{0};
        size_t  target_block_count{0};
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

    FrozenTask(PreparedExecution&& prepared, TransferResult failure);
    ~FrozenTask() = default;
    FrozenTask(const FrozenTask&)            = delete;
    FrozenTask& operator=(const FrozenTask&) = delete;

    // Idempotent, single-use lifecycle. freeze after RELEASED is a no-op.
    // Concurrent unfreeze callers do not repeat decrements; RELEASING remains
    // visible until the owner finishes. Notifications run without this mutex.
    void freeze() noexcept;
    void unfreeze() noexcept;

    const ExecutionInfo& execution() const noexcept {
        return execution_;
    }
    const TransferResult& failure() const noexcept {
        return failure_;
    }
    Snapshot    snapshot() const noexcept;
    std::string debugString() const;

private:
    const ExecutionInfo           execution_;
    const std::vector<TargetPool> targets_;
    const TransferResult          failure_;
    const size_t                  target_block_count_;
    const int64_t                 created_at_us_;
    mutable std::mutex            mutex_;
    State                         state_{State::PREPARED};
    int64_t                       frozen_at_us_{0};
    int64_t                       released_at_us_{0};
};

}  // namespace rtp_llm
