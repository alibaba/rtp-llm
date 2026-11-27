#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"

namespace rtp_llm {

struct HostBlockPoolConfig: public BlockPoolConfigBase {
    size_t payload_bytes{0};
    size_t stride_bytes{0};
    size_t alignment{4096};
};

struct HostBlockBuffer {
    BlockIdxType block;
    void*        addr;
    size_t       payload_bytes;
    size_t       stride_bytes;
};

// Native Host allocations use IBlockPool's slot and reference lifetime. A
// local backing maps the usable slots to fixed-stride rows. A delegated pool
// tracks the same slots while an executor in another process owns their bytes.
class HostBlockPool: public IBlockPool {
public:
    explicit HostBlockPool(std::shared_ptr<const HostBlockPoolConfig> config);
    ~HostBlockPool() override;

    // RTP supplies the pinned backing through its native memory provider.
    bool init();
    bool initBacking(void* base, size_t mapped_bytes, size_t first_usable_offset, std::shared_ptr<void> owner = {});
    void initDelegated();

    // Address lookup does not acquire a reference. The caller holds a lease
    // while accessing the bytes and keeps borrowed backing alive until close.
    HostBlockBuffer blockBuffer(BlockIdxType block) const;

    size_t payloadBytes() const;
    size_t strideBytes() const;
    size_t blockSizeBytes() const override;

    std::string debugString() const override;

private:
    const HostBlockPoolConfig& config() const;
    size_t backingBytes(size_t first_usable_offset) const;

    std::shared_ptr<void> backing_owner_;
    uint8_t* first_usable_row_{nullptr};
};

}  // namespace rtp_llm
