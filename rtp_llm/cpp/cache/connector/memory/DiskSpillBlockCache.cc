#include "rtp_llm/cpp/cache/connector/memory/DiskSpillBlockCache.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <unistd.h>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

size_t alignUpTo(size_t value, size_t alignment) {
    RTP_LLM_CHECK_WITH_INFO(alignment != 0, "disk spill alignment must not be zero");
    return ((value + alignment - 1) / alignment) * alignment;
}

std::string trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\n\r");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\n\r");
    return value.substr(begin, end - begin + 1);
}

std::string randomUuid() {
    std::random_device                      rd;
    std::mt19937_64                         gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream                      oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << dist(gen);
    return oss.str();
}

std::string detectHostname() {
    char buf[256] = {0};
    if (::gethostname(buf, sizeof(buf) - 1) == 0) {
        return buf;
    }
    return "unknown";
}

}  // namespace

void DiskSpillBlockCache::TakenDiskItem::release() {
    if (released_ || !valid_) {
        return;
    }
    released_ = true;
    if (auto owner = owner_.lock()) {
        owner->releaseTakenSlot(item_);
    }
}

std::shared_ptr<DiskSpillBlockCache> DiskSpillBlockCache::create(InitConfig config) {
    return std::shared_ptr<DiskSpillBlockCache>(new DiskSpillBlockCache(std::move(config)));
}

DiskSpillBlockCache::DiskSpillBlockCache(InitConfig config): config_(std::move(config)) {}

DiskSpillBlockCache::~DiskSpillBlockCache() {
    shutdown();
}

std::vector<DiskSpillBlockCache::DiskConfig> DiskSpillBlockCache::parseDiskConfigs(const std::string& spec) {
    std::vector<DiskConfig> disks;
    std::stringstream       ss(spec);
    std::string             token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (token.empty()) {
            continue;
        }
        const auto eq_pos = token.rfind('=');
        RTP_LLM_CHECK_WITH_INFO(eq_pos != std::string::npos && eq_pos > 0 && eq_pos + 1 < token.size(),
                                "invalid MEMORY_CACHE_DISK_PATHS item [%s], expected path=capacity_mb",
                                token.c_str());
        DiskConfig disk;
        disk.path        = trim(token.substr(0, eq_pos));
        const auto cap_s = trim(token.substr(eq_pos + 1));
        RTP_LLM_CHECK_WITH_INFO(!disk.path.empty(), "memory cache disk path must not be empty");
        disk.capacity_mb = static_cast<size_t>(std::stoull(cap_s));
        RTP_LLM_CHECK_WITH_INFO(
            disk.capacity_mb > 0, "memory cache disk capacity must be positive, item [%s]", token.c_str());
        disks.push_back(std::move(disk));
    }
    return disks;
}

bool DiskSpillBlockCache::init() {
    RTP_LLM_CHECK_WITH_INFO(config_.block_size > 0, "disk spill block_size must be > 0");
    RTP_LLM_CHECK_WITH_INFO(!config_.disks.empty(), "disk spill disks must not be empty");
    if (config_.startup_uuid.empty()) {
        config_.startup_uuid = randomUuid();
    }
    if (config_.hostname.empty()) {
        config_.hostname = detectHostname();
    }
    if (config_.schema_hash.empty()) {
        // schema hash is mandatory in production. Tests can pass any string.
        config_.schema_hash = "default";
    }
    if (config_.align_bytes == 0) {
        config_.align_bytes = 4096;
    }
    slot_stride_bytes_ = alignUpTo(config_.block_size, config_.align_bytes);

    if (!initDisks()) {
        shutdown();
        return false;
    }
    RTP_LLM_LOG_INFO("disk spill block cache init success, disks=%zu block_size=%zu slot_stride=%zu align=%zu "
                     "schema_hash=%s startup_uuid=%s",
                     disks_.size(),
                     config_.block_size,
                     slot_stride_bytes_,
                     config_.align_bytes,
                     config_.schema_hash.c_str(),
                     config_.startup_uuid.c_str());
    return true;
}

void DiskSpillBlockCache::shutdown() {
    for (auto& d : disks_) {
        if (d.io_worker) {
            d.io_worker->stop();
        }
    }
    for (auto& d : disks_) {
        if (d.file_manager) {
            d.file_manager->shutdown();
        }
    }
    disks_.clear();
    committed_index_.clear();
    inflight_write_.clear();
    inflight_read_slot_keys_.clear();
    key_generations_.clear();
}

bool DiskSpillBlockCache::initDisks() {
    disks_.reserve(config_.disks.size());
    for (size_t i = 0; i < config_.disks.size(); ++i) {
        DiskRecord record;
        DiskSpillFileManager::Config fcfg;
        fcfg.base_path                = config_.disks[i].path;
        fcfg.disk_id                  = i;
        fcfg.capacity_bytes           = config_.disks[i].capacity_mb * 1024UL * 1024UL;
        fcfg.segment_bytes            = config_.segment_bytes;
        fcfg.slot_stride_bytes        = slot_stride_bytes_;
        fcfg.align_bytes              = config_.align_bytes;
        fcfg.direct_io                = config_.direct_io;
        fcfg.direct_io_required       = config_.direct_io_required;
        fcfg.cleanup_on_destroy       = config_.cleanup_on_destroy;
        fcfg.cleanup_old_startup_dirs = config_.cleanup_old_startup_dirs;
        fcfg.schema_hash              = config_.schema_hash;
        fcfg.world_rank               = config_.world_rank;
        fcfg.hostname                 = config_.hostname;
        fcfg.startup_uuid             = config_.startup_uuid;
        fcfg.max_staging_buffers      = config_.max_staging_buffers_per_disk;

        record.file_manager = std::make_shared<DiskSpillFileManager>(fcfg);
        if (!record.file_manager->init()) {
            RTP_LLM_LOG_ERROR("disk spill init disk failed, path=%s error=%s",
                              config_.disks[i].path.c_str(),
                              record.file_manager->lastError().c_str());
            return false;
        }

        DiskSpillIoWorker::Config wcfg;
        wcfg.write_threads = config_.io_threads_per_disk;
        wcfg.read_threads  = std::max(1, config_.io_threads_per_disk);
        wcfg.queue_size    = config_.io_queue_size;
        record.io_worker   = std::make_shared<DiskSpillIoWorker>(record.file_manager, wcfg);
        if (!record.io_worker->start()) {
            RTP_LLM_LOG_ERROR("disk spill io worker start failed, path=%s", config_.disks[i].path.c_str());
            return false;
        }

        const size_t sc = record.file_manager->slotCount();
        record.slots.resize(sc);
        record.free_slots.reserve(sc);
        for (int s = static_cast<int>(sc) - 1; s >= 0; --s) {
            record.free_slots.push_back(s);
        }
        disks_.push_back(std::move(record));
    }
    return true;
}

bool DiskSpillBlockCache::validateSlot(int disk_id, int slot_id) const {
    return disk_id >= 0 && static_cast<size_t>(disk_id) < disks_.size() && slot_id >= 0
           && static_cast<size_t>(slot_id) < disks_[disk_id].slots.size();
}

bool DiskSpillBlockCache::checkGenLocked(int disk_id, int slot_id, const DiskGen& expected) const {
    const auto& rec = disks_[disk_id].slots[slot_id];
    if (rec.gen.slot_gen != expected.slot_gen) {
        return false;
    }
    if (expected.key_gen == 0) {
        return true;
    }
    const auto it = key_generations_.find(rec.cache_key);
    if (it == key_generations_.end()) {
        return false;
    }
    return it->second == expected.key_gen && rec.gen.key_gen == expected.key_gen;
}

void DiskSpillBlockCache::lruEraseLocked(int disk_id, int slot_id) {
    auto& rec = disks_[disk_id].slots[slot_id];
    if (rec.lru_linked) {
        disks_[disk_id].committed_lru.erase(rec.lru_it);
        rec.lru_linked = false;
    }
}

void DiskSpillBlockCache::lruPushBackLocked(int disk_id, int slot_id) {
    auto& rec = disks_[disk_id].slots[slot_id];
    if (rec.lru_linked) {
        disks_[disk_id].committed_lru.erase(rec.lru_it);
    }
    disks_[disk_id].committed_lru.push_back(slot_id);
    rec.lru_it = std::prev(disks_[disk_id].committed_lru.end());
    rec.lru_linked = true;
}

void DiskSpillBlockCache::freeSlotLocked(int disk_id, int slot_id) {
    if (!validateSlot(disk_id, slot_id)) {
        return;
    }
    auto& disk = disks_[disk_id];
    auto& rec  = disk.slots[slot_id];
    lruEraseLocked(disk_id, slot_id);
    rec.state       = SlotState::FREE;
    rec.cache_key   = 0;
    rec.block_size  = 0;
    rec.is_complete = false;
    // slot_gen is NOT reset; it monotonically increases to defeat ABA.
    if (std::find(disk.free_slots.begin(), disk.free_slots.end(), slot_id) == disk.free_slots.end()) {
        disk.free_slots.push_back(slot_id);
    }
}

int DiskSpillBlockCache::pickLruReuseSlotLocked(int target_disk_id) {
    auto& disk = disks_[target_disk_id];
    // Scan committed_lru front (oldest) and pick the first whose state is
    // strictly COMMITTED with no inflight_read entry. inflight_read is tracked
    // by inflight_read_slot_keys_ keyed on (disk_id<<32 | slot_id).
    for (auto it = disk.committed_lru.begin(); it != disk.committed_lru.end(); ++it) {
        const int slot_id = *it;
        const auto& rec   = disk.slots[slot_id];
        if (rec.state != SlotState::COMMITTED) {
            continue;
        }
        const int key = (target_disk_id << 24) | slot_id;
        (void)key;
        const auto packed = (static_cast<int64_t>(target_disk_id) << 32) | static_cast<uint32_t>(slot_id);
        if (inflight_read_slot_keys_.find(static_cast<int>(packed)) != inflight_read_slot_keys_.end()) {
            continue;
        }
        return slot_id;
    }
    return -1;
}

bool DiskSpillBlockCache::reserveSlotLocked(CacheKeyType cache_key,
                                            size_t       block_size,
                                            bool         is_complete,
                                            int&         out_disk_id,
                                            int&         out_slot_id,
                                            DiskGen&     out_gen,
                                            bool         allow_lru_reuse) {
    // First pass: round-robin a free slot
    for (size_t probe = 0; probe < disks_.size(); ++probe) {
        const size_t did = (next_disk_cursor_ + probe) % disks_.size();
        auto&        d   = disks_[did];
        if (d.file_manager && d.file_manager->isUnhealthy()) {
            continue;
        }
        if (d.free_slots.empty()) {
            continue;
        }
        const int slot_id = d.free_slots.back();
        d.free_slots.pop_back();
        auto& rec = d.slots[slot_id];
        rec.gen.slot_gen += 1;
        const uint64_t new_key_gen = ++key_generations_[cache_key];
        rec.gen.key_gen            = new_key_gen;
        rec.state                  = SlotState::RESERVED;
        rec.cache_key              = cache_key;
        rec.block_size             = block_size;
        rec.is_complete            = is_complete;
        next_disk_cursor_          = (did + 1) % disks_.size();
        out_disk_id                = static_cast<int>(did);
        out_slot_id                = slot_id;
        out_gen                    = rec.gen;
        return true;
    }

    if (!allow_lru_reuse) {
        return false;
    }

    // Second pass: LRU in-place reuse a committed slot
    for (size_t probe = 0; probe < disks_.size(); ++probe) {
        const size_t did = (next_disk_cursor_ + probe) % disks_.size();
        auto&        d   = disks_[did];
        if (d.file_manager && d.file_manager->isUnhealthy()) {
            continue;
        }
        const int reuse_slot = pickLruReuseSlotLocked(static_cast<int>(did));
        if (reuse_slot < 0) {
            continue;
        }
        auto& rec = d.slots[reuse_slot];
        // bump OLD key gen (defeat any in-flight commit/read of the old incarnation)
        if (rec.cache_key != 0) {
            committed_index_.erase(rec.cache_key);
            ++key_generations_[rec.cache_key];
        }
        lruEraseLocked(static_cast<int>(did), reuse_slot);
        rec.gen.slot_gen += 1;
        const uint64_t new_key_gen = ++key_generations_[cache_key];
        rec.gen.key_gen            = new_key_gen;
        rec.state                  = SlotState::RESERVED;
        rec.cache_key              = cache_key;
        rec.block_size             = block_size;
        rec.is_complete            = is_complete;
        next_disk_cursor_          = (did + 1) % disks_.size();
        out_disk_id                = static_cast<int>(did);
        out_slot_id                = reuse_slot;
        out_gen                    = rec.gen;
        return true;
    }
    return false;
}

std::optional<DiskSpillBlockCache::DiskItem>
DiskSpillBlockCache::reserve(CacheKeyType cache_key, size_t block_size, bool is_complete) {
    if (!master_mode_) {
        RTP_LLM_LOG_WARNING("disk spill reserve refused on worker mode, key=%ld", cache_key);
        return std::nullopt;
    }
    if (block_size == 0 || block_size > config_.block_size || disks_.empty()) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    // If key already in committed_index or inflight_write, invalidate old first.
    if (auto cit = committed_index_.find(cache_key); cit != committed_index_.end()) {
        freeSlotLocked(cit->second.disk_id, cit->second.slot_id);
        committed_index_.erase(cit);
        ++key_generations_[cache_key];
    }
    if (auto wit = inflight_write_.find(cache_key); wit != inflight_write_.end()) {
        // a previous reserve hasn't committed/aborted yet; mark stale by bumping
        // gen and removing from inflight_write. Old DiskItem holders will fail
        // commit().
        freeSlotLocked(wit->second.disk_id, wit->second.slot_id);
        inflight_write_.erase(wit);
        ++key_generations_[cache_key];
    }
    int     did    = -1;
    int     sid    = -1;
    DiskGen gen{};
    if (!reserveSlotLocked(cache_key, block_size, is_complete, did, sid, gen, /*allow_lru_reuse=*/true)) {
        return std::nullopt;
    }
    DiskItem item;
    item.cache_key   = cache_key;
    item.disk_id     = did;
    item.slot_id     = sid;
    item.gen         = gen;
    item.block_size  = block_size;
    item.is_complete = is_complete;
    inflight_write_[cache_key] = item;
    return item;
}

bool DiskSpillBlockCache::writeReserved(const DiskItem& slot, const void* data, size_t bytes) {
    if (!data || bytes == 0 || bytes > config_.block_size) {
        return false;
    }
    std::shared_ptr<DiskSpillIoWorker> io;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!validateSlot(slot.disk_id, slot.slot_id) || !checkGenLocked(slot.disk_id, slot.slot_id, slot.gen)) {
            return false;
        }
        const auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
        if (rec.state != SlotState::RESERVED || rec.cache_key != slot.cache_key) {
            return false;
        }
        io = disks_[slot.disk_id].io_worker;
    }
    // Aligned write through file manager. bytes is expected to be a logical
    // block_size (<= slot_stride_bytes). For O_DIRECT mode the caller-supplied
    // buffer MUST be align_bytes-aligned and bytes MUST be a multiple of align;
    // for non-direct mode any bytes works. Callers that don't satisfy alignment
    // should use a staging buffer from the file manager.
    return io && io->pwriteSync(slot.slot_id, data, bytes);
}

bool DiskSpillBlockCache::commit(const DiskItem& slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validateSlot(slot.disk_id, slot.slot_id) || !checkGenLocked(slot.disk_id, slot.slot_id, slot.gen)) {
        return false;
    }
    auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
    if (rec.cache_key != slot.cache_key) {
        return false;
    }
    // If state is RESERVED: promote to COMMITTED. If already COMMITTED (idempotent
    // commit, e.g. retry), it's still OK as long as gens match.
    if (rec.state == SlotState::RESERVED) {
        rec.state = SlotState::COMMITTED;
        lruPushBackLocked(slot.disk_id, slot.slot_id);
        inflight_write_.erase(slot.cache_key);
    } else if (rec.state != SlotState::COMMITTED) {
        return false;
    }
    committed_index_[slot.cache_key] = slot;
    return true;
}

bool DiskSpillBlockCache::abort(const DiskItem& slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validateSlot(slot.disk_id, slot.slot_id)) {
        return false;
    }
    auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
    if (rec.gen.slot_gen != slot.gen.slot_gen) {
        // slot has already been reused; this abort is for a stale incarnation
        return false;
    }
    if (rec.state == SlotState::COMMITTED) {
        committed_index_.erase(slot.cache_key);
    }
    inflight_write_.erase(slot.cache_key);
    freeSlotLocked(slot.disk_id, slot.slot_id);
    return true;
}

bool DiskSpillBlockCache::store(const MemoryBlockCache::CacheItem& item, const void* data, size_t bytes) {
    auto slot = reserve(item.cache_key, bytes, item.is_complete);
    if (!slot.has_value()) {
        return false;
    }
    if (!writeReserved(*slot, data, bytes)) {
        abort(*slot);
        return false;
    }
    if (!commit(*slot)) {
        abort(*slot);
        return false;
    }
    return true;
}

std::optional<DiskSpillBlockCache::DiskItem> DiskSpillBlockCache::takeForReadRaw(CacheKeyType cache_key) {
    if (!master_mode_) {
        return std::nullopt;
    }
    DiskItem captured;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = committed_index_.find(cache_key);
        if (it == committed_index_.end()) {
            return std::nullopt;
        }
        captured = it->second;
        if (!validateSlot(captured.disk_id, captured.slot_id)
            || !checkGenLocked(captured.disk_id, captured.slot_id, captured.gen)) {
            committed_index_.erase(it);
            return std::nullopt;
        }
        auto& rec = disks_[captured.disk_id].slots[captured.slot_id];
        if (rec.state != SlotState::COMMITTED || rec.cache_key != cache_key) {
            committed_index_.erase(it);
            return std::nullopt;
        }
        rec.state = SlotState::READING;
        lruEraseLocked(captured.disk_id, captured.slot_id);
        committed_index_.erase(it);
        const auto packed = (static_cast<int64_t>(captured.disk_id) << 32) | static_cast<uint32_t>(captured.slot_id);
        inflight_read_slot_keys_[static_cast<int>(packed)] = captured;
    }
    return captured;
}

std::optional<DiskSpillBlockCache::TakenDiskItem> DiskSpillBlockCache::takeForRead(CacheKeyType cache_key) {
    if (!master_mode_) {
        return std::nullopt;
    }
    DiskItem captured;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = committed_index_.find(cache_key);
        if (it == committed_index_.end()) {
            return std::nullopt;
        }
        captured = it->second;
        if (!validateSlot(captured.disk_id, captured.slot_id)
            || !checkGenLocked(captured.disk_id, captured.slot_id, captured.gen)) {
            committed_index_.erase(it);
            return std::nullopt;
        }
        auto& rec = disks_[captured.disk_id].slots[captured.slot_id];
        if (rec.state != SlotState::COMMITTED || rec.cache_key != cache_key) {
            committed_index_.erase(it);
            return std::nullopt;
        }
        rec.state = SlotState::READING;
        lruEraseLocked(captured.disk_id, captured.slot_id);
        committed_index_.erase(it);
        const auto packed = (static_cast<int64_t>(captured.disk_id) << 32) | static_cast<uint32_t>(captured.slot_id);
        inflight_read_slot_keys_[static_cast<int>(packed)] = captured;
    }
    return TakenDiskItem(captured, weak_from_this());
}

bool DiskSpillBlockCache::readTaken(const DiskItem& slot, void* data, size_t bytes) {
    if (!data || bytes == 0 || bytes > config_.block_size) {
        return false;
    }
    std::shared_ptr<DiskSpillIoWorker> io;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!validateSlot(slot.disk_id, slot.slot_id) || !checkGenLocked(slot.disk_id, slot.slot_id, slot.gen)) {
            return false;
        }
        const auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
        if ((rec.state != SlotState::READING && rec.state != SlotState::COMMITTED) || rec.cache_key != slot.cache_key) {
            return false;
        }
        io = disks_[slot.disk_id].io_worker;
    }
    return io && io->preadSync(slot.slot_id, data, bytes);
}

void DiskSpillBlockCache::releaseTakenSlot(const DiskItem& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validateSlot(item.disk_id, item.slot_id)) {
        return;
    }
    const auto packed = (static_cast<int64_t>(item.disk_id) << 32) | static_cast<uint32_t>(item.slot_id);
    inflight_read_slot_keys_.erase(static_cast<int>(packed));
    auto& rec = disks_[item.disk_id].slots[item.slot_id];
    if (rec.gen.slot_gen != item.gen.slot_gen) {
        // slot got reused for a new key while read was in flight — no-op
        return;
    }
    if (rec.state == SlotState::READING) {
        freeSlotLocked(item.disk_id, item.slot_id);
    }
}

bool DiskSpillBlockCache::putExternalSlot(const DiskItem& slot, const void* data, size_t bytes) {
    if (!data || bytes == 0 || bytes > config_.block_size) {
        return false;
    }
    std::shared_ptr<DiskSpillIoWorker> io;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!validateSlot(slot.disk_id, slot.slot_id)) {
            return false;
        }
        auto& rec  = disks_[slot.disk_id].slots[slot.slot_id];
        auto& disk = disks_[slot.disk_id];
        // worker-side: trust the master. Update key_gen tracking.
        auto& cur_kgen = key_generations_[slot.cache_key];
        if (slot.gen.key_gen != 0) {
            if (cur_kgen > slot.gen.key_gen) {
                return false;
            }
            cur_kgen = slot.gen.key_gen;
        } else if (cur_kgen == 0) {
            cur_kgen = ++key_generations_[slot.cache_key];
        }
        if (rec.state == SlotState::COMMITTED) {
            committed_index_.erase(rec.cache_key);
        }
        lruEraseLocked(slot.disk_id, slot.slot_id);
        rec.state       = SlotState::COMMITTED;  // workers don't go through RESERVED
        rec.cache_key   = slot.cache_key;
        rec.gen         = slot.gen;
        rec.block_size  = slot.block_size;
        rec.is_complete = slot.is_complete;
        disk.free_slots.erase(std::remove(disk.free_slots.begin(), disk.free_slots.end(), slot.slot_id),
                               disk.free_slots.end());
        lruPushBackLocked(slot.disk_id, slot.slot_id);
        committed_index_[slot.cache_key] = slot;
        io                               = disk.io_worker;
    }
    if (!io || !io->pwriteSync(slot.slot_id, data, bytes)) {
        // rollback
        std::lock_guard<std::mutex> lock(mutex_);
        committed_index_.erase(slot.cache_key);
        freeSlotLocked(slot.disk_id, slot.slot_id);
        return false;
    }
    return true;
}

bool DiskSpillBlockCache::deleteSlot(const DiskItem& slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validateSlot(slot.disk_id, slot.slot_id)) {
        return false;
    }
    auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
    if (rec.gen.slot_gen != slot.gen.slot_gen) {
        return false;
    }
    if (slot.gen.key_gen != 0 && rec.gen.key_gen != slot.gen.key_gen) {
        return false;
    }
    if (rec.state == SlotState::COMMITTED) {
        committed_index_.erase(rec.cache_key);
    }
    inflight_write_.erase(rec.cache_key);
    const auto packed = (static_cast<int64_t>(slot.disk_id) << 32) | static_cast<uint32_t>(slot.slot_id);
    inflight_read_slot_keys_.erase(static_cast<int>(packed));
    freeSlotLocked(slot.disk_id, slot.slot_id);
    return true;
}

std::optional<DiskSpillBlockCache::DiskItem> DiskSpillBlockCache::invalidate(CacheKeyType cache_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++key_generations_[cache_key];
    DiskItem item;
    bool     found = false;
    if (auto cit = committed_index_.find(cache_key); cit != committed_index_.end()) {
        item  = cit->second;
        found = true;
        if (validateSlot(item.disk_id, item.slot_id)) {
            freeSlotLocked(item.disk_id, item.slot_id);
        }
        committed_index_.erase(cit);
    }
    if (auto wit = inflight_write_.find(cache_key); wit != inflight_write_.end()) {
        if (!found) {
            item  = wit->second;
            found = true;
        }
        if (validateSlot(wit->second.disk_id, wit->second.slot_id)) {
            freeSlotLocked(wit->second.disk_id, wit->second.slot_id);
        }
        inflight_write_.erase(wit);
    }
    if (!found) {
        return std::nullopt;
    }
    return item;
}

DiskSpillBlockCache::MatchResult DiskSpillBlockCache::match(CacheKeyType cache_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = committed_index_.find(cache_key);
    if (it == committed_index_.end()) {
        return {};
    }
    const auto& slot = it->second;
    if (!validateSlot(slot.disk_id, slot.slot_id) || !checkGenLocked(slot.disk_id, slot.slot_id, slot.gen)) {
        return {};
    }
    const auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
    if (rec.state != SlotState::COMMITTED || rec.cache_key != cache_key) {
        return {};
    }
    return {true, slot.disk_id, slot.slot_id, slot.gen, rec.block_size, rec.is_complete};
}

bool DiskSpillBlockCache::contains(CacheKeyType cache_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return committed_index_.find(cache_key) != committed_index_.end();
}

DiskSpillBlockCache::Status DiskSpillBlockCache::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Status                      s;
    s.committed_slot_num      = committed_index_.size();
    s.inflight_write_slot_num = inflight_write_.size();
    s.inflight_read_slot_num  = inflight_read_slot_keys_.size();
    for (const auto& disk : disks_) {
        s.total_slot_num += disk.slots.size();
        s.free_slot_num += disk.free_slots.size();
        for (const auto& rec : disk.slots) {
            if (rec.state == SlotState::LEAKED) {
                ++s.leaked_slot_num;
            }
        }
        if (disk.file_manager) {
            const auto fs = disk.file_manager->getStats();
            s.staging_used += fs.staging_used;
            if (fs.unhealthy) {
                ++s.unhealthy_disk_num;
            }
        }
    }
    const size_t used_slots = s.committed_slot_num + s.inflight_write_slot_num + s.inflight_read_slot_num;
    s.used_bytes            = used_slots * slot_stride_bytes_;
    s.free_bytes            = s.free_slot_num * slot_stride_bytes_;
    return s;
}

size_t DiskSpillBlockCache::totalSlotNum() const {
    size_t n = 0;
    for (const auto& d : disks_) {
        n += d.slots.size();
    }
    return n;
}

size_t DiskSpillBlockCache::alignBytes() const {
    return config_.align_bytes;
}

}  // namespace rtp_llm
