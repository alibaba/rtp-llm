#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/IBlockPool.h"

namespace rtp_llm {

struct HostBlockPoolConfig: public BlockPoolConfigBase {
    size_t payload_bytes{0};
    size_t stride_bytes{0};
    bool   enable_pinned{true};
    size_t alignment{4096};
};

struct HostBlockBuffer {
    BlockIdxType block;
    void*        addr;
    size_t       payload_bytes;
    size_t       stride_bytes;
};

// HostBlockPool backs every block with a fixed-stride slice of a single contiguous
// host memory tensor (pinned when possible, falling back to pageable memory). All
// lifecycle behavior (malloc/free/incRef/decRef/metrics) is inherited unchanged from
// IBlockPool; this class only owns the host backing tensor and exposes blockBuffer()
// to map a valid physical block index to its backing address.
class HostBlockPool: public IBlockPool {
public:
    explicit HostBlockPool(std::shared_ptr<const HostBlockPoolConfig> config);
    ~HostBlockPool() override;

    // Validates the config's payload/stride/alignment invariants, allocates the host
    // backing tensor (pinned if config().enable_pinned, falling back to pageable on
    // failure), and marks the pool initialized. Always returns true; invariant
    // violations are enforced via RTP_LLM_CHECK.
    bool init();

    bool isPinned() const;

    // Returns the backing buffer for a valid physical block. RTP_LLM_CHECK-fails if
    // the pool is not initialized or the block index is out of range.
    HostBlockBuffer blockBuffer(BlockIdxType block) const;

    size_t payloadBytes() const;
    size_t strideBytes() const;

    std::string debugString() const override;

private:
    const HostBlockPoolConfig& config() const;

private:
    torch::Tensor backing_;
    void*         base_ptr_{nullptr};
    bool          pinned_{false};
};

}  // namespace rtp_llm
