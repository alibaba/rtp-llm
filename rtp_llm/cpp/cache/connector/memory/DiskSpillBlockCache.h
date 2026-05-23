#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"

namespace rtp_llm {

class DiskSpillBlockCache {
public:
    struct DiskConfig {
        std::string path;
        size_t      capacity_mb{0};
    };

    struct InitConfig {
        std::vector<DiskConfig> disks;
        size_t                  block_size{0};
        size_t                  align_bytes{4096};
        bool                    cleanup_on_destroy{true};
    };

    struct DiskSlot {
        CacheKeyType cache_key{0};
        int          disk_id{-1};
        int          slot_id{-1};
        uint64_t     generation{0};
        uint64_t     key_generation{0};
        size_t       block_size{0};
        bool         is_complete{false};
    };

    struct MatchResult {
        bool     matched{false};
        int      disk_id{-1};
        int      slot_id{-1};
        uint64_t generation{0};
        size_t   block_size{0};
        bool     is_complete{false};
    };

    struct Status {
        size_t total_slot_num{0};
        size_t free_slot_num{0};
        size_t item_num{0};
        size_t leaked_slot_num{0};
    };

public:
    explicit DiskSpillBlockCache(InitConfig config);
    ~DiskSpillBlockCache();

    static std::vector<DiskConfig> parseDiskConfigs(const std::string& spec);

    bool init();

    MatchResult match(CacheKeyType cache_key);
    bool        contains(CacheKeyType cache_key) const;

    std::optional<DiskSlot> reserve(CacheKeyType cache_key, size_t block_size, bool is_complete);
    bool                    writeReserved(const DiskSlot& slot, const void* data, size_t bytes);
    bool                    commit(const DiskSlot& slot);
    bool                    abort(const DiskSlot& slot);
    bool                    store(const MemoryBlockCache::CacheItem& item, const void* data, size_t bytes);

    std::optional<DiskSlot> takeForRead(CacheKeyType cache_key);
    bool                    readTaken(const DiskSlot& slot, void* data, size_t bytes);
    bool                    releaseReadSlot(const DiskSlot& slot);

    bool                    putExternalSlot(const DiskSlot& slot, const void* data, size_t bytes);
    bool                    deleteSlot(const DiskSlot& slot);
    std::optional<DiskSlot> invalidate(CacheKeyType cache_key);

    Status status() const;

private:
    enum class SlotState {
        FREE      = 0,
        RESERVED  = 1,
        COMMITTED = 2,
        READING   = 3,
        LEAKED    = 4,
    };

    struct SlotRecord {
        SlotState    state{SlotState::FREE};
        CacheKeyType cache_key{0};
        uint64_t     generation{0};
        uint64_t     key_generation{0};
        size_t       block_size{0};
        bool         is_complete{false};
    };

    struct DiskFile {
        std::string             base_path;
        std::string             run_dir;
        std::string             file_path;
        int                     fd{-1};
        size_t                  slot_count{0};
        std::vector<int>        free_slots;
        std::vector<SlotRecord> slots;
    };

private:
    bool initDisk(size_t disk_id, const DiskConfig& disk_config);
    bool validateSlotLocked(const DiskSlot& slot) const;
    bool preadFull(int fd, void* data, size_t bytes, off_t offset) const;
    bool pwriteFull(int fd, const void* data, size_t bytes, off_t offset) const;
    void freeSlotLocked(const DiskSlot& slot);
    void closeFiles();
    void cleanupRunDirs();

private:
    InitConfig                                 config_;
    size_t                                     slot_stride_bytes_{0};
    std::vector<DiskFile>                      disks_;
    std::unordered_map<CacheKeyType, DiskSlot> index_;
    std::unordered_map<CacheKeyType, uint64_t> key_generations_;
    mutable std::mutex                         mutex_;
    size_t                                     next_disk_cursor_{0};
};

using DiskSpillBlockCachePtr = std::shared_ptr<DiskSpillBlockCache>;

}  // namespace rtp_llm
