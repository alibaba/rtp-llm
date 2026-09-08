#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"
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

// HostBlockPool backs every block with a fixed-stride slice of a single contiguous
// pinned CPU memory tensor. All lifecycle behavior (malloc/free/incRef/decRef/metrics) is inherited unchanged from
// IBlockPool; this class only owns the host backing tensor and exposes blockBuffer()
// to map a valid physical block index to its backing address.
class HostBlockPool: public IBlockPool {
public:
    explicit HostBlockPool(std::shared_ptr<const HostBlockPoolConfig> config);
    ~HostBlockPool() override;

    // Validates payload/stride/alignment, allocates pinned CPU memory, and marks the
    // pool initialized. Allocation failures and invariant violations raise errors.
    bool init();

    // Returns the backing buffer for a valid physical block. RTP_LLM_CHECK-fails if
    // the pool is not initialized or the block index is out of range.
    HostBlockBuffer blockBuffer(BlockIdxType block) const;

    size_t payloadBytes() const;
    size_t strideBytes() const;
    size_t blockSizeBytes() const override;

    std::string debugString() const override;

private:
    const HostBlockPoolConfig& config() const;

private:
    std::optional<AlignedHostMemory> backing_;
};

}  // namespace rtp_llm
