#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/connector/memory/DiskSpillTypes.h"

namespace rtp_llm {

// DiskSpillFileManager owns ONE logical disk: its base path, its segment files,
// its O_DIRECT/buffered mode, and a pool of aligned staging buffers used for
// pread/pwrite. It does NOT own metadata (committed/inflight indices) — that
// stays in DiskSpillBlockCache. It does NOT own threads — IO is dispatched
// through DiskSpillIoWorker which owns the per-disk worker pool.
//
// Directory layout: README §"临时文件和目录"
//   ${base_path}/rtp_llm_mem_spill/
//     schema_${schema_hash}/
//       rank_${world_rank}/
//         host_${hostname}_pid_${pid}_uuid_${startup_uuid}/   <- flock here
//           disk_${disk_id}_seg_${seg_id}.bin
//
// O_DIRECT is preferred. Probe on init; fall back to buffered + posix_fadvise(DONTNEED)
// when supported by config. If direct_io_required and probe fails, init returns false.
class DiskSpillFileManager {
public:
    struct Config {
        std::string base_path;                  // root, e.g. "/mnt/nvme0/rtp_kvcache"
        size_t      disk_id{0};
        size_t      capacity_bytes{0};
        size_t      segment_bytes{256UL * 1024 * 1024};  // default 256MB per segment file
        size_t      slot_stride_bytes{0};       // logical slot stride (already aligned)
        size_t      align_bytes{0};             // 0 = auto-detect; fallback 4096
        bool        direct_io{true};
        bool        direct_io_required{false};
        bool        cleanup_on_destroy{true};
        bool        cleanup_old_startup_dirs{true};
        std::string schema_hash;
        int         world_rank{0};
        std::string hostname;
        std::string startup_uuid;
        int         max_staging_buffers{32};    // per-disk staging buffer pool size
    };

    enum class IoMode : uint8_t {
        DIRECT   = 0,
        BUFFERED = 1,
        NONE     = 2,
    };

    struct Stats {
        IoMode io_mode{IoMode::NONE};
        size_t slot_count{0};
        size_t segment_count{0};
        size_t used_bytes{0};       // sum of slot_stride * (in-use slots)
        size_t free_bytes{0};
        size_t staging_used{0};
        size_t staging_total{0};
        bool   unhealthy{false};
        size_t consecutive_failures{0};
    };

    // Holds an aligned heap buffer carved from posix_memalign. Returned by the
    // pool as shared_ptr; on destruction returns the slot to the pool.
    class StagingBuffer {
    public:
        StagingBuffer() = default;
        StagingBuffer(void* addr, size_t bytes): addr_(addr), bytes_(bytes) {}

        void*  data() const {
            return addr_;
        }
        size_t size() const {
            return bytes_;
        }
        bool   valid() const {
            return addr_ != nullptr && bytes_ > 0;
        }

    private:
        void*  addr_{nullptr};
        size_t bytes_{0};
    };

public:
    explicit DiskSpillFileManager(Config config);
    ~DiskSpillFileManager();

    DiskSpillFileManager(const DiskSpillFileManager&)            = delete;
    DiskSpillFileManager& operator=(const DiskSpillFileManager&) = delete;

    // Init runs:
    //   1. cleanupOldStartupDirs() (if enabled)
    //   2. create startup dir + flock
    //   3. open + ftruncate/posix_fallocate the first segment file
    //   4. O_DIRECT probe + alignment detection (or buffered fallback)
    //   5. allocate staging buffer pool
    // Returns false on any unrecoverable error; metric error_type set in last_error_.
    bool init();

    // Stop accepting new IO. Drains nothing; caller (IoWorker) ensures workers idle first.
    void shutdown();

    // Slot IO. Returns true on full bytes written/read. EINTR-safe.
    // Returns false on short IO, EIO, ENOSPC, EINVAL — increments consecutive_failures
    // and may mark disk unhealthy.
    bool pwriteSlot(int slot_id, const void* data, size_t bytes);
    bool preadSlot(int slot_id, void* data, size_t bytes);

    // Staging buffer pool — caller-owned wrapper around posix_memalign. Returns
    // nullptr if pool is exhausted. Pool size is configured at init.
    std::shared_ptr<StagingBuffer> acquireStagingBuffer();
    void                           releaseStagingBuffer(std::shared_ptr<StagingBuffer> buffer);

    // Unhealthy-disk probe. Called by IoWorker on a periodic schedule. Issues a
    // small aligned read+write+verify. On success consecutive_failures reset to 0.
    bool probeHealth();

    // Read-only accessors
    size_t                slotCount() const;
    size_t                alignBytes() const;
    IoMode                ioMode() const;
    bool                  isUnhealthy() const;
    const std::string&    lastError() const;
    Stats                 getStats() const;
    const std::string&    pathHash() const;     // log-safe identifier for this disk
    int                   diskId() const {
        return static_cast<int>(config_.disk_id);
    }
    size_t segmentBytes() const {
        return config_.segment_bytes;
    }

    // Test-only: force unhealthy state to test recovery via probeHealth().
    void forceUnhealthy_TestOnly() {
        unhealthy_.store(true);
    }

private:
    bool detectAlignment();
    bool openSegments();
    bool probeDirectIO();
    bool ensureStartupDirAndLock();
    void releaseStartupLock();
    void cleanupRunDir();
    void cleanupOldStartupDirs();
    bool allocateStagingPool();
    void releaseStagingPool();
    int  fdForSlot(int slot_id, off_t& out_offset) const;
    bool ioRead(int fd, void* buf, size_t bytes, off_t offset) const;
    bool ioWrite(int fd, const void* buf, size_t bytes, off_t offset) const;
    void recordFailure(const std::string& error_type);
    void recordSuccess();
    std::string makePathHash() const;

    struct Segment {
        std::string path;
        int         fd{-1};
        size_t      slots_in_segment{0};
    };

private:
    Config                                            config_;
    size_t                                            align_bytes_{4096};
    size_t                                            slots_per_segment_{0};
    size_t                                            slot_count_{0};
    IoMode                                            io_mode_{IoMode::NONE};
    std::vector<Segment>                              segments_;
    std::string                                       run_dir_;
    std::string                                       schema_dir_;
    std::string                                       rank_dir_;
    int                                               lock_fd_{-1};
    std::atomic<bool>                                 unhealthy_{false};
    std::atomic<size_t>                               consecutive_failures_{0};
    mutable std::mutex                                error_mutex_;
    std::string                                       last_error_;
    std::string                                       path_hash_;
    mutable std::mutex                                staging_mutex_;
    std::vector<std::shared_ptr<StagingBuffer>>       staging_free_list_;
    std::vector<std::shared_ptr<StagingBuffer>>       staging_all_;
    std::atomic<size_t>                               staging_used_{0};
};

using DiskSpillFileManagerPtr = std::shared_ptr<DiskSpillFileManager>;

}  // namespace rtp_llm
