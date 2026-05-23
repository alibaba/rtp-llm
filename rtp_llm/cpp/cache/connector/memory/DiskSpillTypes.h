#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

// Twin-generation tracking on disk slots.
//
// key_gen: bumped on every memory put / invalidate / spill of the same cache key.
//   Used to detect ABA where a key is evicted, spilled, invalidated, then re-spilled
//   while an in-flight pwrite/commit of an older incarnation is still pending.
//
// slot_gen: bumped on every reservation of the same physical (disk_id, slot_id).
//   Used to detect ABA where a slot is freed and reused for a different key while
//   an old read/write op still holds a reference to it.
//
// A commit is only valid when both DiskSpillBlockCache::key_generations_[key] == key_gen
// AND DiskFile::slots[slot_id].generation == slot_gen.
struct DiskGen {
    uint64_t key_gen{0};
    uint64_t slot_gen{0};
};

// Phase of a physical disk slot in DiskSpillBlockCache.
enum class DiskSlotState : uint8_t {
    FREE      = 0,  // owned by free_slots vector; safe to reserve
    RESERVED  = 1,  // allocated by reserve(); awaiting writeReserved+commit (or abort)
    COMMITTED = 2,  // in committed_index; can serve match/takeForRead
    READING   = 3,  // takeForRead() handed out a TakenDiskItem; not yet released
    ABORTING  = 4,  // pending abort/delete; not reusable until cleanup confirmed
    LEAKED    = 5,  // delete failed / rank disagreed; permanently unusable until probe
};

// Logical lifecycle of a spill job tracked by DiskSpillCommitCoordinator.
//
// RESERVED       — master reserved slot; staging not started
// STAGING        — workers acked stage; pwrite queued
// PWRITE_INFLIGHT — at least one rank's pwrite still pending
// COMMITTED      — all ranks pwrite ok; master commit done; slot in committed_index
// ABORTING       — one rank failed/timeout; broadcasting DELETE_DISK_SLOT
// LEAKED         — abort failed on at least one rank; slot perma-unusable
// FREE           — terminal; slot back in free_slots (after successful abort)
enum class SpillStageState : uint8_t {
    RESERVED        = 0,
    STAGING         = 1,
    PWRITE_INFLIGHT = 2,
    COMMITTED       = 3,
    ABORTING        = 4,
    LEAKED          = 5,
    FREE            = 6,
};

// Per-rank view of a disk slot file. Used by the commit coordinator to track
// abort cleanup so a logical slot stays LEAKED until all ranks confirm DELETED.
enum class RankFileSlotState : uint8_t {
    UNKNOWN       = 0,
    STAGED        = 1,
    WRITTEN       = 2,
    DELETED       = 3,
    DELETE_FAILED = 4,
};

// Slot record handed across the connector / coordinator / file manager boundaries.
struct DiskSlotHandle {
    CacheKeyType cache_key{0};
    int32_t      disk_id{-1};
    int32_t      slot_id{-1};
    DiskGen      gen{};
    size_t       logical_bytes{0};
    bool         is_complete{true};
};

// Pull-status response shape returned by SPILL_WRITE_STATUS RPC.
enum class SpillWriteStatus : uint8_t {
    PENDING      = 0,
    SUCCESS      = 1,
    FAILED       = 2,
    UNKNOWN_JOB  = 3,
};

using SpillJobId = uint64_t;

// Error categories surfaced through rtp_llm_kv_cache_memory_disk_cache_error_qps
// tag `error_type`. Strings are stable and align with README test plan §9 case 4.
namespace disk_error {
constexpr const char* kInitPath                = "init_path";
constexpr const char* kInitCapacity            = "init_capacity";
constexpr const char* kInitSchema              = "init_schema";
constexpr const char* kCapabilityMismatch      = "capability_mismatch";
constexpr const char* kRankTopologyMismatch    = "rank_topology_mismatch";
constexpr const char* kDirectIoRequired        = "direct_io_required";
constexpr const char* kDirectIoFallback        = "direct_io_fallback";
constexpr const char* kStagingNoBuffer         = "staging_no_buffer";
constexpr const char* kStagingBudgetTimeout    = "staging_budget_timeout";
constexpr const char* kStageAckTimeout         = "stage_ack_timeout";
constexpr const char* kQueueFull               = "queue_full";
constexpr const char* kNoSlot                  = "no_slot";
constexpr const char* kInvalidMemoryBlock      = "invalid_memory_block";
constexpr const char* kPwrite                  = "pwrite";
constexpr const char* kPread                   = "pread";
constexpr const char* kShortIo                 = "short_io";
constexpr const char* kRankWrite               = "rank_write";
constexpr const char* kRankRead                = "rank_read";
constexpr const char* kTpBroadcast             = "tp_broadcast";
constexpr const char* kTpBroadcastTimeout      = "tp_broadcast_timeout";
constexpr const char* kTpBroadcastAbort        = "tp_broadcast_abort";
constexpr const char* kProtocolViolation       = "protocol_violation";
constexpr const char* kGenerationCancel        = "generation_cancel";
constexpr const char* kDuplicateKey            = "duplicate_key";
constexpr const char* kReadRescanMiss          = "read_rescan_miss";
constexpr const char* kH2d                     = "h2d";
constexpr const char* kDeleteSlot              = "delete_slot";
constexpr const char* kSlotLeak                = "slot_leak";
constexpr const char* kCleanupStartupDir       = "cleanup_startup_dir";
constexpr const char* kDiskUnhealthy           = "disk_unhealthy";
constexpr const char* kSlotUseAfterFreeDetect  = "slot_use_after_free_detected";
}  // namespace disk_error

// Capability handshake constants
constexpr uint32_t kDiskSpillProtocolVersion = 1;

}  // namespace rtp_llm
