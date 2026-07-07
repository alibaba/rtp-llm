#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"

#include <cstdint>
#include <exception>
#include <sstream>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

HostBlockPool::HostBlockPool(std::shared_ptr<const HostBlockPoolConfig> config): IBlockPool(config) {
    RTP_LLM_CHECK(config != nullptr);
    RTP_LLM_CHECK(config->pool_type == BlockPoolType::HOST);
    RTP_LLM_CHECK(config->free_block_order_policy == FreeBlockOrderPolicy::ANY_ORDER);
}

HostBlockPool::~HostBlockPool() = default;

const HostBlockPoolConfig& HostBlockPool::config() const {
    return configAs<HostBlockPoolConfig>(BlockPoolType::HOST);
}

bool HostBlockPool::init() {
    const auto& cfg = config();
    RTP_LLM_CHECK_WITH_INFO(
        cfg.payload_bytes > 0, "host block pool [%s] payload_bytes must be > 0", cfg.pool_name.c_str());
    RTP_LLM_CHECK_WITH_INFO(cfg.stride_bytes >= cfg.payload_bytes,
                             "host block pool [%s] stride_bytes [%zu] must be >= payload_bytes [%zu]",
                             cfg.pool_name.c_str(),
                             cfg.stride_bytes,
                             cfg.payload_bytes);
    RTP_LLM_CHECK_WITH_INFO(
        cfg.alignment > 0, "host block pool [%s] alignment must be > 0", cfg.pool_name.c_str());
    RTP_LLM_CHECK_WITH_INFO(cfg.stride_bytes % cfg.alignment == 0,
                             "host block pool [%s] stride_bytes [%zu] must be a multiple of alignment [%zu]",
                             cfg.pool_name.c_str(),
                             cfg.stride_bytes,
                             cfg.alignment);

    // block 0's slot is allocated as backing but is never handed out by malloc().
    const size_t total_bytes = cfg.physical_block_count * cfg.stride_bytes;
    auto         cpu         = torch::empty({static_cast<int64_t>(total_bytes)},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));

    bool pinned = false;
    if (cfg.enable_pinned) {
        try {
            backing_ = cpu.pin_memory();
            pinned   = backing_.is_pinned();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING(
                "pin host block pool memory failed, fallback to pageable CPU memory, pool_name=%s error=%s",
                cfg.pool_name.c_str(),
                e.what());
        }
        if (!pinned) {
            RTP_LLM_LOG_WARNING(
                "host block pool pin_memory unavailable, fallback to pageable CPU memory, pool_name=%s",
                cfg.pool_name.c_str());
            backing_ = cpu;
        }
    } else {
        backing_ = cpu;
    }

    pinned_   = pinned;
    base_ptr_ = backing_.data_ptr();

    markInitialized();
    return true;
}

bool HostBlockPool::isPinned() const {
    return pinned_;
}

HostBlockBuffer HostBlockPool::blockBuffer(BlockIdxType block) const {
    RTP_LLM_CHECK(initialized());
    RTP_LLM_CHECK(isAllocated(block));
    const auto& cfg = config();
    void*       addr =
        static_cast<uint8_t*>(base_ptr_) + static_cast<size_t>(block) * cfg.stride_bytes;
    return HostBlockBuffer{block, addr, cfg.payload_bytes, cfg.stride_bytes};
}

size_t HostBlockPool::payloadBytes() const {
    return config().payload_bytes;
}

size_t HostBlockPool::strideBytes() const {
    return config().stride_bytes;
}

std::string HostBlockPool::debugString() const {
    std::ostringstream oss;
    oss << "HostBlockPool{" << IBlockPool::debugString() << ", payload_bytes=" << payloadBytes()
        << ", stride_bytes=" << strideBytes() << ", pinned=" << pinned_ << "}";
    return oss.str();
}

}  // namespace rtp_llm
