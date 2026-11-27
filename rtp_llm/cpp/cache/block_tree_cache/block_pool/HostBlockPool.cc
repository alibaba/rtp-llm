#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace rtp_llm {
namespace {

std::shared_ptr<const HostBlockPoolConfig>
validateHostBlockPoolConfig(std::shared_ptr<const HostBlockPoolConfig> config) {
    if (!config || config->pool_type != BlockPoolType::HOST || config->physical_block_count <= 1
        || config->growable) {
        throw std::invalid_argument("Host block pool requires a fixed positive capacity");
    }
    return config;
}

}  // namespace

HostBlockPool::HostBlockPool(std::shared_ptr<const HostBlockPoolConfig> config):
    IBlockPool(validateHostBlockPoolConfig(std::move(config))) {}

HostBlockPool::~HostBlockPool() = default;

const HostBlockPoolConfig& HostBlockPool::config() const {
    return configAs<HostBlockPoolConfig>(BlockPoolType::HOST);
}

size_t HostBlockPool::backingBytes(size_t first_usable_offset) const {
    const auto& cfg = config();
    if (!cfg.payload_bytes || cfg.stride_bytes < cfg.payload_bytes || !cfg.alignment
        || cfg.stride_bytes % cfg.alignment || first_usable_offset % cfg.alignment) {
        throw std::invalid_argument("Invalid Host block layout for pool " + poolName());
    }
    const size_t usable = cfg.physical_block_count - 1;
    if (usable > (std::numeric_limits<size_t>::max() - first_usable_offset) / cfg.stride_bytes) {
        throw std::invalid_argument("Host block backing size overflow for pool " + poolName());
    }
    return first_usable_offset + usable * cfg.stride_bytes;
}

bool HostBlockPool::initBacking(void* base,
                                size_t mapped_bytes,
                                size_t first_usable_offset,
                                std::shared_ptr<void> owner) {
    if (initialized()) {
        throw std::logic_error("Host block pool is already initialized: " + poolName());
    }
    const size_t needed = backingBytes(first_usable_offset);
    const auto address = reinterpret_cast<uintptr_t>(base);
    if (!base || needed > mapped_bytes || address % config().alignment
        || mapped_bytes > std::numeric_limits<uintptr_t>::max() - address) {
        throw std::invalid_argument("Invalid Host backing for pool " + poolName());
    }
    backing_owner_ = std::move(owner);
    first_usable_row_ = static_cast<uint8_t*>(base) + first_usable_offset;
    markInitialized();
    return true;
}

void HostBlockPool::initDelegated() {
    if (initialized()) {
        throw std::logic_error("Host block pool is already initialized: " + poolName());
    }
    markInitialized();
}

HostBlockBuffer HostBlockPool::blockBuffer(BlockIdxType block) const {
    if (!first_usable_row_ || !validBlock(block)) {
        throw std::invalid_argument("Host backing or block is unavailable in pool " + poolName());
    }
    const auto& cfg = config();
    auto* row = first_usable_row_ + static_cast<size_t>(block - 1) * cfg.stride_bytes;
    return {block, row, cfg.payload_bytes, cfg.stride_bytes};
}

size_t HostBlockPool::payloadBytes() const {
    return config().payload_bytes;
}

size_t HostBlockPool::strideBytes() const {
    return config().stride_bytes;
}

size_t HostBlockPool::blockSizeBytes() const {
    return payloadBytes();
}

std::string HostBlockPool::debugString() const {
    std::ostringstream oss;
    oss << "HostBlockPool{" << IBlockPool::debugString() << ", payload_bytes=" << payloadBytes()
        << ", stride_bytes=" << strideBytes() << "}";
    return oss.str();
}

}  // namespace rtp_llm
