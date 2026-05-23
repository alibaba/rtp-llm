#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/connector/memory/DiskSpillFileManager.h"
#include "rtp_llm/cpp/cache/connector/memory/DiskSpillIoWorker.h"
#include "rtp_llm/cpp/cache/connector/memory/DiskSpillTypes.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"

namespace rtp_llm {

// DiskSpillBlockCache: owns N logical disks, each delegated to a DiskSpillFileManager
// + DiskSpillIoWorker. Maintains three metadata indices per README §"Disk metadata 原子语义":
//
//   committed_index : cache_key -> DiskItem
//   inflight_write  : cache_key -> DiskItem   (between reserve() and commit()/abort())
//   inflight_read   : slot      -> DiskItem   (between takeForRead() and releaseTaken())
//
// Generation tracking:
//   key_gen   bumped on every put/invalidate/spill of the same cache_key
//   slot_gen  bumped on every reservation of the same physical (disk_id, slot_id)
// Both must match at commit() time and at readTaken() time to prevent ABA between
// concurrent spill / invalidate / committed-slot LRU in-place reuse.
//
// In-place slot reuse: when reserve() finds no free slot on any disk, the oldest
// committed slot is evicted. Old key is removed from committed_index with key_gen
// bumped; slot enters RESERVED with slot_gen bumped. The OLD spill's commit/read
// then fails generation check, the NEW spill proceeds. Slots with active
// inflight_read are NEVER candidates for in-place reuse (use-after-free guard).
class DiskSpillBlockCache: public std::enable_shared_from_this<DiskSpillBlockCache> {
public:
    struct DiskConfig {
        std::string path;
        size_t      capacity_mb{0};
    };

    struct InitConfig {
        std::vector<DiskConfig> disks;
        size_t                  block_size{0};
        size_t                  align_bytes{4096};
        size_t                  segment_bytes{256UL * 1024 * 1024};
        bool                    direct_io{true};
        bool                    direct_io_required{false};
        bool                    cleanup_on_destroy{true};
        bool                    cleanup_old_startup_dirs{true};
        std::string             schema_hash;
        int                     world_rank{0};
        std::string             hostname;
        std::string             startup_uuid;  // auto-generated if empty
        int                     io_threads_per_disk{2};
        int                     io_queue_size{1024};
        int                     max_staging_buffers_per_disk{32};
    };

    // Lightweight "what's on disk" handle handed across the connector / coordinator.
    // is_complete / block_size travel with the handle so workers don't have to
    // round-trip back to master to learn payload size.
    struct DiskItem {
        CacheKeyType cache_key{0};
        int          disk_id{-1};
        int          slot_id{-1};
        DiskGen      gen{};
        size_t       block_size{0};
        bool         is_complete{true};
    };

    // Backward-compat alias.
    using DiskSlot = DiskItem;

    struct MatchResult {
        bool         matched{false};
        int          disk_id{-1};
        int          slot_id{-1};
        DiskGen      gen{};
        size_t       block_size{0};
        bool         is_complete{false};
    };

    struct Status {
        size_t total_slot_num{0};
        size_t free_slot_num{0};
        size_t committed_slot_num{0};
        size_t inflight_write_slot_num{0};
        size_t inflight_read_slot_num{0};
        size_t leaked_slot_num{0};
        size_t used_bytes{0};
        size_t free_bytes{0};
        size_t staging_used{0};
        size_t unhealthy_disk_num{0};
    };

    // RAII handle returned by takeForRead. Destructor releases the slot back to
    // FREE if not explicitly released. Designed for the read path where copy
    // failure or RPC exception must not leak a slot. Move-only.
    class TakenDiskItem {
    public:
        TakenDiskItem() = default;
        TakenDiskItem(const DiskItem& item, std::weak_ptr<DiskSpillBlockCache> owner)
            : item_(item), owner_(std::move(owner)), released_(false), valid_(true) {}

        TakenDiskItem(const TakenDiskItem&)            = delete;
        TakenDiskItem& operator=(const TakenDiskItem&) = delete;

        TakenDiskItem(TakenDiskItem&& other) noexcept {
            *this = std::move(other);
        }
        TakenDiskItem& operator=(TakenDiskItem&& other) noexcept {
            if (this == &other) {
                return *this;
            }
            tryRelease();
            item_           = other.item_;
            owner_          = std::move(other.owner_);
            released_       = other.released_;
            valid_          = other.valid_;
            other.released_ = true;
            other.valid_    = false;
            return *this;
        }

        ~TakenDiskItem() {
            tryRelease();
        }

        const DiskItem& item() const {
            return item_;
        }
        bool valid() const {
            return valid_ && !released_;
        }

        // Caller explicitly releases the slot back to FREE — happens after the read
        // (or copy plan setup failure) completes. Idempotent.
        void release();

    private:
        void tryRelease() {
            if (!released_ && valid_) {
                release();
            }
        }

        DiskItem                            item_{};
        std::weak_ptr<DiskSpillBlockCache>  owner_;
        bool                                released_{true};
        bool                                valid_{false};
    };

public:
    static std::shared_ptr<DiskSpillBlockCache> create(InitConfig config);
    static std::vector<DiskConfig>              parseDiskConfigs(const std::string& spec);

    ~DiskSpillBlockCache();

    DiskSpillBlockCache(const DiskSpillBlockCache&)            = delete;
    DiskSpillBlockCache& operator=(const DiskSpillBlockCache&) = delete;

    bool init();
    void shutdown();

    // Match: returns success if key is currently in committed_index AND backing
    // slot record is still valid. Does NOT lock the slot — caller may need to
    // call takeForRead later.
    MatchResult match(CacheKeyType cache_key);
    bool        contains(CacheKeyType cache_key) const;

    // Master-side reserve: allocate a logical slot for the given key. Uses
    // round-robin across disks; if all disks full, attempts in-place LRU reuse of
    // an oldest committed slot whose state allows it (no inflight_read, no
    // release_pending). Returns nullopt if even LRU reuse fails (all candidates
    // pinned). The returned DiskItem carries the new key_gen/slot_gen.
    std::optional<DiskItem> reserve(CacheKeyType cache_key, size_t block_size, bool is_complete);

    // Synchronous local pwrite to the reserved slot (master-side fast path).
    // For async, use writeReservedAsync via IoWorker submit (Phase D).
    bool writeReserved(const DiskItem& slot, const void* data, size_t bytes);

    // Promote a RESERVED slot to COMMITTED. Validates generations; if either
    // key_gen or slot_gen mismatches, commit fails and slot must be aborted by
    // caller.
    bool commit(const DiskItem& slot);

    // Abort an in-flight reservation; returns slot to free list (or LEAKED on
    // delete failure path).
    bool abort(const DiskItem& slot);

    // Convenience: reserve + writeReserved + commit in one shot. Used by
    // single-rank tests.
    bool store(const MemoryBlockCache::CacheItem& item, const void* data, size_t bytes);

    // Master-side take for read. On success, key is removed from committed_index
    // and slot enters READING state; the TakenDiskItem RAII owns the slot's
    // release lifecycle.
    std::optional<TakenDiskItem> takeForRead(CacheKeyType cache_key);

    // Raw variant for callers that prefer manual release lifecycle (e.g. the
    // connector that stashes a DiskItem inside a CopyPlan and explicitly calls
    // releaseTakenSlot in the plan's done callback). Same semantics as
    // takeForRead() but without RAII — caller MUST call releaseTakenSlot(item)
    // exactly once.
    std::optional<DiskItem> takeForReadRaw(CacheKeyType cache_key);

    // Synchronous local pread on a taken slot.
    bool readTaken(const DiskItem& slot, void* data, size_t bytes);

    // Worker-side: master broadcasts to put a specific slot's bytes locally.
    // Worker maintains a parallel slot record but does NOT serve match. On
    // mismatch with prior commit, old entry is invalidated first.
    bool putExternalSlot(const DiskItem& slot, const void* data, size_t bytes);

    // Master OR worker: delete a slot (e.g. abort cleanup). Idempotent. Returns
    // false only on generation mismatch (older delete arrived after a newer reuse).
    bool deleteSlot(const DiskItem& slot);

    // Master-side: remove a key from committed_index AND bump key_gen so any
    // in-flight commit of the OLD key incarnation fails. Used by putToCache when
    // GPU writes a fresh version of the key back to memory cache.
    std::optional<DiskItem> invalidate(CacheKeyType cache_key);

    Status status() const;

    // For metrics/log surfaces
    size_t totalSlotNum() const;
    size_t alignBytes() const;
    size_t blockSize() const {
        return config_.block_size;
    }
    bool   isMaster() const {
        return master_mode_;
    }
    const InitConfig& config() const {
        return config_;
    }

    // Test/observability
    std::vector<std::shared_ptr<DiskSpillFileManager>> fileManagers() const {
        std::vector<std::shared_ptr<DiskSpillFileManager>> out;
        out.reserve(disks_.size());
        for (const auto& d : disks_) {
            out.push_back(d.file_manager);
        }
        return out;
    }
    std::vector<std::shared_ptr<DiskSpillIoWorker>> ioWorkers() const {
        std::vector<std::shared_ptr<DiskSpillIoWorker>> out;
        out.reserve(disks_.size());
        for (const auto& d : disks_) {
            out.push_back(d.io_worker);
        }
        return out;
    }

    // Internal: called by TakenDiskItem destructor / release().
    void releaseTakenSlot(const DiskItem& item);

    // Test-only: switch to worker-mode where reserve/commit refuse but
    // putExternalSlot/deleteSlot work. Connector sets this based on tp_rank.
    void setMasterMode(bool on) {
        master_mode_ = on;
    }

private:
    explicit DiskSpillBlockCache(InitConfig config);

    enum class SlotState : uint8_t {
        FREE      = 0,
        RESERVED  = 1,
        COMMITTED = 2,
        READING   = 3,
        ABORTING  = 4,
        LEAKED    = 5,
    };

    struct SlotRecord {
        SlotState    state{SlotState::FREE};
        CacheKeyType cache_key{0};
        DiskGen      gen{};
        size_t       block_size{0};
        bool         is_complete{false};
        std::list<int>::iterator lru_it;       // iterator into per-disk committed_lru_
        bool                     lru_linked{false};
    };

    struct DiskRecord {
        std::shared_ptr<DiskSpillFileManager> file_manager;
        std::shared_ptr<DiskSpillIoWorker>    io_worker;
        std::vector<SlotRecord>               slots;
        std::vector<int>                      free_slots;
        std::list<int>                        committed_lru;  // front=oldest
    };

    bool initDisks();
    bool reserveSlotLocked(CacheKeyType        cache_key,
                           size_t              block_size,
                           bool                is_complete,
                           int&                out_disk_id,
                           int&                out_slot_id,
                           DiskGen&            out_gen,
                           bool                allow_lru_reuse);

    void freeSlotLocked(int disk_id, int slot_id);
    void lruEraseLocked(int disk_id, int slot_id);
    void lruPushBackLocked(int disk_id, int slot_id);
    int  pickLruReuseSlotLocked(int target_disk_id);  // returns slot_id or -1

    bool validateSlot(int disk_id, int slot_id) const;
    bool checkGenLocked(int disk_id, int slot_id, const DiskGen& expected) const;

    InitConfig                                   config_;
    bool                                         master_mode_{true};
    size_t                                       slot_stride_bytes_{0};
    std::vector<DiskRecord>                      disks_;
    std::unordered_map<CacheKeyType, DiskItem>   committed_index_;
    std::unordered_map<CacheKeyType, DiskItem>   inflight_write_;
    std::unordered_map<int, DiskItem>            inflight_read_slot_keys_;  // packed (disk_id<<32|slot_id) -> item
    std::unordered_map<CacheKeyType, uint64_t>   key_generations_;
    mutable std::mutex                           mutex_;
    size_t                                       next_disk_cursor_{0};
};

using DiskSpillBlockCachePtr = std::shared_ptr<DiskSpillBlockCache>;

}  // namespace rtp_llm
