#pragma once

#include <cstddef>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class HostStagingBlockPool;

// The hardware adapter creates leases. The common WorkLoop retains and
// releases the existing allocation without its allocator or backing types.
class HostStagingBlockLease {
public:
    HostStagingBlockLease(HostStagingBlockPool* pool, size_t block_id);
    HostStagingBlockLease(const HostStagingBlockLease&) = delete;
    HostStagingBlockLease& operator=(const HostStagingBlockLease&) = delete;

    HostStagingBlockLease(HostStagingBlockLease&& other) noexcept:
        pool_(std::exchange(other.pool_, nullptr)), block_id_(other.block_id_), release_(other.release_) {}

    HostStagingBlockLease& operator=(HostStagingBlockLease&& other) noexcept {
        if (this != &other) {
            release();
            pool_ = std::exchange(other.pool_, nullptr);
            block_id_ = other.block_id_;
            release_ = other.release_;
        }
        return *this;
    }

    ~HostStagingBlockLease() {
        release();
    }

    HostBufferView blockBuffer(size_t payload_bytes) const;

    bool belongsTo(const HostStagingBlockPool* pool) const {
        return pool_ == pool;
    }

private:
    void release() {
        if (pool_ != nullptr)
            release_(pool_, block_id_);
    }

    HostStagingBlockPool* pool_;
    size_t block_id_;
    void (*release_)(HostStagingBlockPool*, size_t);
};

}  // namespace rtp_llm
