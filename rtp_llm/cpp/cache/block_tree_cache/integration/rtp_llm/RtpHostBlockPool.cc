#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include <sys/mman.h>
#include <unistd.h>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

// Excluding pinned cache pages from core dumps is best-effort.
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

bool HostBlockPool::init() {
    const auto& cfg = config();
    // RTP reserves the first physical row in its pinned allocation.
    const size_t total_bytes = backingBytes(cfg.stride_bytes);
    auto         backing     = std::make_shared<AlignedHostMemory>(total_bytes, cfg.alignment, cfg.pool_name);
    void* const  data        = backing->data();
    markHostBlockPoolDontDump(cfg.pool_name.c_str(), data, total_bytes);
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    RTP_LLM_LOG_INFO("backing selected: pool_name=%s payload_bytes=%zu stride_bytes=%zu "
                     "physical_block_count=%zu total_size=%zu bytes total_size_mb=%.2f ptr=%p",
                     cfg.pool_name.c_str(),
                     cfg.payload_bytes,
                     cfg.stride_bytes,
                     cfg.physical_block_count,
                     total_bytes,
                     static_cast<double>(total_bytes) / kBytesPerMB,
                     data);
    return initBacking(data, total_bytes, cfg.stride_bytes, std::move(backing));
}

}  // namespace rtp_llm
