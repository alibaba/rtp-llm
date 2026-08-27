#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

struct KdaShadowKey {
    int64_t  request_id{0};
    uint64_t generation_epoch{0};

    bool operator==(const KdaShadowKey& other) const {
        return request_id == other.request_id && generation_epoch == other.generation_epoch;
    }
};

struct KdaShadowKeyHash {
    size_t operator()(const KdaShadowKey& key) const;
};

enum class KdaShadowState : uint8_t {
    ABSENT = 0,
    RESERVED,
    LOADING,
    READY,
    RELEASING,
    RELEASED,
    ERROR,
};

enum class KdaShadowCommandType : uint8_t {
    RESERVE = 0,
    ADOPT,
    LOAD,
    COMMIT,
    FAIL,
    ROLLBACK,
    RELEASE,
};

struct KdaShadowCommand {
    KdaShadowCommandType type{KdaShadowCommandType::RESERVE};
    KdaShadowKey         key;
    int                  seq_len{0};
    std::string          error;
    BlockIndicesType     adopted_blocks;
    BlockIndicesType     adopted_kernel_blocks;
};

struct KdaShadowRecord {
    KdaShadowKey      key;
    KdaShadowState    state{KdaShadowState::ABSENT};
    int               seq_len{0};
    BlockIndicesType  blocks;
    BlockIndicesType  kernel_blocks;
    bool              owns_blocks{true};
    std::string       error;
};

struct KdaShadowResult {
    bool              success{false};
    bool              idempotent{false};
    KdaShadowState    state{KdaShadowState::ABSENT};
    BlockIndicesType  blocks;
    BlockIndicesType  kernel_blocks;
    std::string       error;
};

class KdaShadowBlockAllocator {
public:
    virtual ~KdaShadowBlockAllocator() = default;
    virtual bool reserve(int seq_len, BlockIndicesType& blocks, BlockIndicesType& kernel_blocks) = 0;
    virtual bool release(const BlockIndicesType& blocks)         = 0;
};

// Production adapter for one rank-local KDA LINEAR cache group.
class HybridKdaShadowBlockAllocator final: public KdaShadowBlockAllocator {
public:
    HybridKdaShadowBlockAllocator(HybridKVCacheAllocatorPtr allocator, int group_id);

    bool reserve(int seq_len, BlockIndicesType& blocks, BlockIndicesType& kernel_blocks) override;
    bool release(const BlockIndicesType& blocks) override;

private:
    HybridKVCacheAllocatorPtr allocator_;
    int                       group_id_{-1};
};

// Rank-local lifecycle registry.  Stage 4 broadcasts the same command to all
// KTP ranks and combines their ACKs; this class deliberately owns no network
// synchronization.
class KdaShadowRegistry {
public:
    explicit KdaShadowRegistry(std::shared_ptr<KdaShadowBlockAllocator> allocator);
    ~KdaShadowRegistry();

    KdaShadowResult apply(const KdaShadowCommand& command);
    std::optional<KdaShadowRecord> record(const KdaShadowKey& key) const;
    std::optional<KdaShadowRecord> readyRecord(const KdaShadowKey& key) const;

    // Padding/fake rows are represented by std::nullopt and produce an empty
    // block-table row.  Every real row must reference a READY record.
    std::vector<BlockIndicesType>
    buildReadyBlockRows(const std::vector<std::optional<KdaShadowKey>>& rank_major_keys) const;
    std::vector<BlockIndicesType>
    buildReadyKernelBlockRows(const std::vector<std::optional<KdaShadowKey>>& rank_major_keys) const;
    std::vector<KdaShadowRecord> readyRecords() const;

    size_t liveRecordCount() const;
    size_t liveBlockCount() const;

private:
    KdaShadowResult result(const KdaShadowRecord& record, bool success, bool idempotent = false) const;
    bool            releaseBlocks(KdaShadowRecord& record, std::string& error);
    static bool     isLive(KdaShadowState state);

    std::shared_ptr<KdaShadowBlockAllocator> allocator_;
    mutable std::mutex                       mutex_;
    std::unordered_map<KdaShadowKey, KdaShadowRecord, KdaShadowKeyHash> records_;
};

}  // namespace rtp_llm
