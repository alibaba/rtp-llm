#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/connector/memory/DiskBlockIO.h"

namespace rtp_llm {

struct DiskBlockPoolConfig {
    std::string mount_path;
    int64_t     local_rank{0};
    int64_t     world_rank{0};
    size_t      disk_size_bytes{0};
    size_t      block_size_bytes{0};
    bool        buffered_io{true};
};

class DiskBlockPool {
public:
    explicit DiskBlockPool(DiskBlockPoolConfig config, std::unique_ptr<IDiskBlockIO> io = nullptr);
    ~DiskBlockPool();

    bool init();

    // Slot allocation is driven by the copy-plan owner, matching the existing
    // memory connector metadata model. Follower ranks receive the slot id in
    // the broadcast copy plan and use it as an externally assigned file offset;
    // they do not independently allocate or evict disk slots.
    std::optional<int32_t> malloc();
    void                   requestReference(int32_t slot);
    void                   requestFree(int32_t slot);
    void                   blockCacheReference(int32_t slot);
    void                   blockCacheFree(int32_t slot);

    bool read(int32_t slot, void* dst, size_t bytes);
    bool write(int32_t slot, const void* src, size_t bytes);

    size_t             totalSlots() const;
    size_t             freeSlots() const;
    size_t             availableSlots() const;
    size_t             blockSizeBytes() const;
    size_t             slotStrideBytes() const;
    size_t             readBytes() const;
    size_t             writeBytes() const;
    const std::string& filePath() const;
    std::string        debugString() const;

    static size_t alignUp(size_t value, size_t alignment);

private:
    struct SlotState {
        uint32_t request_ref{0};
        uint32_t cache_ref{0};
    };

    bool initDirectoryAndLock();
    bool cleanupStaleFiles();
    bool initFile();
    bool validSlot(int32_t slot) const;
    void tryFreeSlotLocked(int32_t slot);
    void unlockAndClose();

private:
    DiskBlockPoolConfig           config_;
    std::unique_ptr<IDiskBlockIO> io_;
    std::string                   work_dir_;
    std::string                   lock_path_;
    std::string                   file_path_;
    int                           lock_fd_{-1};
    size_t                        slot_stride_bytes_{0};
    size_t                        slot_count_{0};
    mutable std::mutex            mutex_;
    std::set<int32_t>             free_slots_;
    std::vector<SlotState>        slots_;
    std::atomic<size_t>           read_bytes_{0};
    std::atomic<size_t>           write_bytes_{0};
};

using DiskBlockPoolPtr = std::shared_ptr<DiskBlockPool>;

}  // namespace rtp_llm
