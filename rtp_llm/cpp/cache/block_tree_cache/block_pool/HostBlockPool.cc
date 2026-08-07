#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sstream>

#include <sys/mman.h>
#include <unistd.h>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

// Exclude the host block pool backing from core dumps (mirrors the removed legacy pool
// behavior). Best-effort: failures and the absence of MADV_DONTDUMP only warn and never
// fail initialization.
void markHostBlockPoolDontDump(const char* pool_name, void* ptr, size_t size) {
#ifdef MADV_DONTDUMP
    if (ptr == nullptr || size == 0) {
        return;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        page_size = 4096;
    }

    const auto begin         = reinterpret_cast<uintptr_t>(ptr);
    const auto page_mask     = static_cast<uintptr_t>(page_size - 1);
    const auto aligned_begin = begin & ~page_mask;
    const auto aligned_end   = (begin + size + page_mask) & ~page_mask;
    const auto aligned_size  = static_cast<size_t>(aligned_end - aligned_begin);

    if (madvise(reinterpret_cast<void*>(aligned_begin), aligned_size, MADV_DONTDUMP) != 0) {
        RTP_LLM_LOG_WARNING("madvise MADV_DONTDUMP failed for host block pool, pool_name=%s ptr=%p, size=%zu, "
                            "error=%s",
                            pool_name,
                            ptr,
                            size,
                            std::strerror(errno));
    } else {
        RTP_LLM_LOG_INFO("madvise MADV_DONTDUMP success for host block pool, pool_name=%s ptr=%p, size=%zu, "
                         "aligned_ptr=%p, aligned_size=%zu",
                         pool_name,
                         ptr,
                         size,
                         reinterpret_cast<void*>(aligned_begin),
                         aligned_size);
    }
#else
    RTP_LLM_LOG_WARNING(
        "MADV_DONTDUMP is not defined, host block pool may be included in coredump, pool_name=%s ptr=%p, size=%zu",
        pool_name,
        ptr,
        size);
#endif
}

}  // namespace

HostBlockPool::HostBlockPool(std::shared_ptr<const HostBlockPoolConfig> config): IBlockPool(config) {
    RTP_LLM_CHECK(config != nullptr);
    RTP_LLM_CHECK(config->pool_type == BlockPoolType::HOST);
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
    RTP_LLM_CHECK_WITH_INFO(cfg.alignment > 0, "host block pool [%s] alignment must be > 0", cfg.pool_name.c_str());
    RTP_LLM_CHECK_WITH_INFO(cfg.stride_bytes % cfg.alignment == 0,
                            "host block pool [%s] stride_bytes [%zu] must be a multiple of alignment [%zu]",
                            cfg.pool_name.c_str(),
                            cfg.stride_bytes,
                            cfg.alignment);

    // block 0's slot is allocated as backing but is never handed out by malloc().
    const size_t total_bytes = cfg.physical_block_count * cfg.stride_bytes;
    backing_.emplace(total_bytes, cfg.alignment, cfg.enable_pinned, cfg.pool_name);
    markHostBlockPoolDontDump(cfg.pool_name.c_str(), backing_->data(), total_bytes);
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    RTP_LLM_LOG_INFO("backing selected: pool_name=%s payload_bytes=%zu stride_bytes=%zu "
                     "physical_block_count=%zu total_size=%zu bytes total_size_mb=%.2f is_pinned=%d ptr=%p",
                     cfg.pool_name.c_str(),
                     cfg.payload_bytes,
                     cfg.stride_bytes,
                     cfg.physical_block_count,
                     total_bytes,
                     static_cast<double>(total_bytes) / kBytesPerMB,
                     backing_->isPinned(),
                     backing_->data());

    markInitialized();
    return true;
}

bool HostBlockPool::isPinned() const {
    return backing_.has_value() && backing_->isPinned();
}

HostBlockBuffer HostBlockPool::blockBuffer(BlockIdxType block) const {
    RTP_LLM_CHECK(initialized());
    RTP_LLM_CHECK(validBlock(block));
    const auto& cfg  = config();
    void*       addr = backing_->data() + static_cast<size_t>(block) * cfg.stride_bytes;
    return HostBlockBuffer{block, addr, cfg.payload_bytes, cfg.stride_bytes};
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
        << ", stride_bytes=" << strideBytes() << ", pinned=" << isPinned() << "}";
    return oss.str();
}

}  // namespace rtp_llm
