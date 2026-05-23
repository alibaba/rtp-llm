#include "rtp_llm/cpp/cache/connector/memory/DiskSpillBlockCache.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <sstream>
#include <unistd.h>
#include <fcntl.h>

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

std::string hostName() {
    char buf[256] = {0};
    if (::gethostname(buf, sizeof(buf) - 1) != 0) {
        return "unknown";
    }
    return std::string(buf);
}

std::string randomHex() {
    std::random_device                      rd;
    std::mt19937_64                         gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream                      oss;
    oss << std::hex << dist(gen);
    return oss.str();
}

}  // namespace

DiskSpillBlockCache::DiskSpillBlockCache(InitConfig config): config_(std::move(config)) {}

DiskSpillBlockCache::~DiskSpillBlockCache() {
    closeFiles();
    if (config_.cleanup_on_destroy) {
        cleanupRunDirs();
    }
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
    slot_stride_bytes_ = alignUpTo(config_.block_size, config_.align_bytes);
    RTP_LLM_CHECK_WITH_INFO(slot_stride_bytes_ >= config_.block_size,
                            "invalid disk spill slot stride, block_size=%zu stride=%zu",
                            config_.block_size,
                            slot_stride_bytes_);

    disks_.reserve(config_.disks.size());
    for (size_t i = 0; i < config_.disks.size(); ++i) {
        if (!initDisk(i, config_.disks[i])) {
            closeFiles();
            cleanupRunDirs();
            return false;
        }
    }
    RTP_LLM_LOG_INFO("disk spill block cache init success, disks=%zu block_size=%zu slot_stride=%zu total_slots=%zu",
                     disks_.size(),
                     config_.block_size,
                     slot_stride_bytes_,
                     status().total_slot_num);
    return true;
}

bool DiskSpillBlockCache::initDisk(size_t disk_id, const DiskConfig& disk_config) {
    const size_t capacity_bytes = disk_config.capacity_mb * 1024UL * 1024UL;
    const size_t slot_count     = capacity_bytes / slot_stride_bytes_;
    if (slot_count == 0 || slot_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        RTP_LLM_LOG_ERROR("disk spill init disk failed, path=%s capacity_mb=%zu slot_stride=%zu slot_count=%zu",
                          disk_config.path.c_str(),
                          disk_config.capacity_mb,
                          slot_stride_bytes_,
                          slot_count);
        return false;
    }

    DiskFile disk;
    disk.base_path = disk_config.path;
    const auto run_dir_name =
        "rtp_llm_memory_disk_spill_host_" + hostName() + "_pid_" + std::to_string(::getpid()) + "_" + randomHex();
    std::filesystem::path run_dir = std::filesystem::path(disk.base_path) / run_dir_name;
    std::error_code       ec;
    std::filesystem::create_directories(run_dir, ec);
    if (ec) {
        RTP_LLM_LOG_ERROR("disk spill create dir failed, path=%s err=%s", run_dir.c_str(), ec.message().c_str());
        return false;
    }

    disk.run_dir   = run_dir.string();
    disk.file_path = (run_dir / ("disk_" + std::to_string(disk_id) + ".bin")).string();
    disk.fd        = ::open(disk.file_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (disk.fd < 0) {
        RTP_LLM_LOG_ERROR(
            "disk spill open file failed, path=%s errno=%d(%s)", disk.file_path.c_str(), errno, std::strerror(errno));
        return false;
    }

    const off_t file_size = static_cast<off_t>(slot_count * slot_stride_bytes_);
    if (::ftruncate(disk.fd, file_size) != 0) {
        RTP_LLM_LOG_ERROR("disk spill ftruncate failed, path=%s size=%ld errno=%d(%s)",
                          disk.file_path.c_str(),
                          static_cast<long>(file_size),
                          errno,
                          std::strerror(errno));
        ::close(disk.fd);
        disk.fd = -1;
        return false;
    }

    std::vector<char> probe(std::min<size_t>(slot_stride_bytes_, 4096), 0);
    if (!pwriteFull(disk.fd, probe.data(), probe.size(), 0) || !preadFull(disk.fd, probe.data(), probe.size(), 0)) {
        RTP_LLM_LOG_ERROR("disk spill probe IO failed, path=%s", disk.file_path.c_str());
        ::close(disk.fd);
        disk.fd = -1;
        return false;
    }

    disk.slot_count = slot_count;
    disk.slots.resize(slot_count);
    disk.free_slots.reserve(slot_count);
    for (int slot_id = static_cast<int>(slot_count) - 1; slot_id >= 0; --slot_id) {
        disk.free_slots.push_back(slot_id);
    }
    disks_.push_back(std::move(disk));
    return true;
}

DiskSpillBlockCache::MatchResult DiskSpillBlockCache::match(CacheKeyType cache_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = index_.find(cache_key);
    if (it == index_.end()) {
        return {};
    }
    const auto& slot = it->second;
    if (!validateSlotLocked(slot)) {
        return {};
    }
    const auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
    if (rec.state != SlotState::COMMITTED || rec.cache_key != cache_key || rec.generation != slot.generation) {
        return {};
    }
    return {true, slot.disk_id, slot.slot_id, slot.generation, rec.block_size, rec.is_complete};
}

bool DiskSpillBlockCache::contains(CacheKeyType cache_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return index_.find(cache_key) != index_.end();
}

std::optional<DiskSpillBlockCache::DiskSlot>
DiskSpillBlockCache::reserve(CacheKeyType cache_key, size_t block_size, bool is_complete) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (block_size == 0 || block_size > config_.block_size || disks_.empty()) {
        return std::nullopt;
    }
    if (auto old = index_.find(cache_key); old != index_.end()) {
        freeSlotLocked(old->second);
        index_.erase(old);
    }
    const uint64_t key_generation = ++key_generations_[cache_key];

    for (size_t probe = 0; probe < disks_.size(); ++probe) {
        const size_t disk_id = (next_disk_cursor_ + probe) % disks_.size();
        auto&        disk    = disks_[disk_id];
        if (disk.free_slots.empty()) {
            continue;
        }
        const int slot_id = disk.free_slots.back();
        disk.free_slots.pop_back();
        auto& rec     = disk.slots[slot_id];
        rec.state     = SlotState::RESERVED;
        rec.cache_key = cache_key;
        rec.generation += 1;
        rec.key_generation = key_generation;
        rec.block_size     = block_size;
        rec.is_complete    = is_complete;
        next_disk_cursor_  = (disk_id + 1) % disks_.size();
        return DiskSlot{
            cache_key, static_cast<int>(disk_id), slot_id, rec.generation, key_generation, block_size, is_complete};
    }
    return std::nullopt;
}

bool DiskSpillBlockCache::writeReserved(const DiskSlot& slot, const void* data, size_t bytes) {
    if (data == nullptr || bytes == 0 || bytes > config_.block_size) {
        return false;
    }
    int fd = -1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!validateSlotLocked(slot)) {
            return false;
        }
        auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
        if (rec.state != SlotState::RESERVED || rec.generation != slot.generation || rec.cache_key != slot.cache_key) {
            return false;
        }
        fd = disks_[slot.disk_id].fd;
    }
    const off_t offset = static_cast<off_t>(slot.slot_id * slot_stride_bytes_);
    return pwriteFull(fd, data, bytes, offset);
}

bool DiskSpillBlockCache::commit(const DiskSlot& slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validateSlotLocked(slot)) {
        return false;
    }
    auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
    if (rec.state == SlotState::COMMITTED && rec.generation == slot.generation && rec.cache_key == slot.cache_key) {
        if (slot.key_generation != 0 && key_generations_[slot.cache_key] != slot.key_generation) {
            return false;
        }
        index_[slot.cache_key] = slot;
        return true;
    }
    if (rec.state != SlotState::RESERVED || rec.generation != slot.generation || rec.cache_key != slot.cache_key) {
        return false;
    }
    if (slot.key_generation != 0
        && (rec.key_generation != slot.key_generation || key_generations_[slot.cache_key] != slot.key_generation)) {
        return false;
    }
    if (auto old = index_.find(slot.cache_key); old != index_.end()) {
        freeSlotLocked(old->second);
        index_.erase(old);
    }
    rec.state              = SlotState::COMMITTED;
    index_[slot.cache_key] = slot;
    return true;
}

bool DiskSpillBlockCache::abort(const DiskSlot& slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validateSlotLocked(slot)) {
        return false;
    }
    auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
    if (rec.generation != slot.generation) {
        return false;
    }
    if (rec.state == SlotState::COMMITTED) {
        index_.erase(slot.cache_key);
    }
    freeSlotLocked(slot);
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

std::optional<DiskSpillBlockCache::DiskSlot> DiskSpillBlockCache::takeForRead(CacheKeyType cache_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = index_.find(cache_key);
    if (it == index_.end()) {
        return std::nullopt;
    }
    auto slot = it->second;
    if (!validateSlotLocked(slot)) {
        index_.erase(it);
        return std::nullopt;
    }
    auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
    if (rec.state != SlotState::COMMITTED || rec.generation != slot.generation || rec.cache_key != cache_key) {
        index_.erase(it);
        return std::nullopt;
    }
    rec.state = SlotState::READING;
    index_.erase(it);
    return slot;
}

bool DiskSpillBlockCache::readTaken(const DiskSlot& slot, void* data, size_t bytes) {
    if (data == nullptr || bytes == 0 || bytes > config_.block_size) {
        return false;
    }
    int fd = -1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!validateSlotLocked(slot)) {
            return false;
        }
        const auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
        if ((rec.state != SlotState::READING && rec.state != SlotState::COMMITTED) || rec.generation != slot.generation
            || rec.cache_key != slot.cache_key) {
            return false;
        }
        if (slot.key_generation != 0 && rec.key_generation != slot.key_generation) {
            return false;
        }
        fd = disks_[slot.disk_id].fd;
    }
    const off_t offset = static_cast<off_t>(slot.slot_id * slot_stride_bytes_);
    return preadFull(fd, data, bytes, offset);
}

bool DiskSpillBlockCache::releaseReadSlot(const DiskSlot& slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validateSlotLocked(slot)) {
        return false;
    }
    auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
    if (rec.generation != slot.generation) {
        return false;
    }
    if (slot.key_generation != 0 && rec.key_generation != slot.key_generation) {
        return false;
    }
    if (rec.state == SlotState::READING || rec.state == SlotState::COMMITTED || rec.state == SlotState::RESERVED) {
        index_.erase(slot.cache_key);
        freeSlotLocked(slot);
        return true;
    }
    return rec.state == SlotState::FREE;
}

bool DiskSpillBlockCache::putExternalSlot(const DiskSlot& slot, const void* data, size_t bytes) {
    if (data == nullptr || bytes == 0 || bytes > config_.block_size) {
        return false;
    }
    auto local_slot = slot;
    int  fd         = -1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (local_slot.disk_id < 0 || static_cast<size_t>(local_slot.disk_id) >= disks_.size() || local_slot.slot_id < 0
            || static_cast<size_t>(local_slot.slot_id) >= disks_[local_slot.disk_id].slots.size()) {
            return false;
        }
        auto& rec = disks_[local_slot.disk_id].slots[local_slot.slot_id];
        if (local_slot.key_generation == 0) {
            if (rec.state == SlotState::RESERVED && rec.cache_key == local_slot.cache_key
                && rec.generation == local_slot.generation && rec.key_generation != 0) {
                local_slot.key_generation = rec.key_generation;
            } else {
                local_slot.key_generation = ++key_generations_[local_slot.cache_key];
            }
        } else {
            auto& current_key_generation = key_generations_[local_slot.cache_key];
            if (current_key_generation > local_slot.key_generation) {
                return false;
            }
            current_key_generation = local_slot.key_generation;
        }
        if (rec.state == SlotState::COMMITTED) {
            index_.erase(rec.cache_key);
        }
        rec.state          = SlotState::RESERVED;
        rec.cache_key      = local_slot.cache_key;
        rec.generation     = local_slot.generation;
        rec.key_generation = local_slot.key_generation;
        rec.block_size     = local_slot.block_size;
        rec.is_complete    = local_slot.is_complete;
        auto& free_slots   = disks_[local_slot.disk_id].free_slots;
        free_slots.erase(std::remove(free_slots.begin(), free_slots.end(), local_slot.slot_id), free_slots.end());
        fd = disks_[local_slot.disk_id].fd;
    }
    const off_t offset = static_cast<off_t>(local_slot.slot_id * slot_stride_bytes_);
    if (!pwriteFull(fd, data, bytes, offset)) {
        abort(local_slot);
        return false;
    }
    return commit(local_slot);
}

bool DiskSpillBlockCache::deleteSlot(const DiskSlot& slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validateSlotLocked(slot)) {
        return false;
    }
    auto& rec = disks_[slot.disk_id].slots[slot.slot_id];
    if (rec.generation != slot.generation) {
        return false;
    }
    if (slot.key_generation != 0 && rec.key_generation != slot.key_generation) {
        return false;
    }
    if (rec.state == SlotState::COMMITTED) {
        index_.erase(rec.cache_key);
    }
    freeSlotLocked(slot);
    return true;
}

std::optional<DiskSpillBlockCache::DiskSlot> DiskSpillBlockCache::invalidate(CacheKeyType cache_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++key_generations_[cache_key];
    const auto it = index_.find(cache_key);
    if (it == index_.end()) {
        return std::nullopt;
    }
    auto slot = it->second;
    if (validateSlotLocked(slot)) {
        freeSlotLocked(slot);
    }
    index_.erase(it);
    return slot;
}

DiskSpillBlockCache::Status DiskSpillBlockCache::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Status                      status;
    status.item_num = index_.size();
    for (const auto& disk : disks_) {
        status.total_slot_num += disk.slot_count;
        status.free_slot_num += disk.free_slots.size();
        for (const auto& rec : disk.slots) {
            if (rec.state == SlotState::LEAKED) {
                ++status.leaked_slot_num;
            }
        }
    }
    return status;
}

bool DiskSpillBlockCache::validateSlotLocked(const DiskSlot& slot) const {
    return slot.disk_id >= 0 && static_cast<size_t>(slot.disk_id) < disks_.size() && slot.slot_id >= 0
           && static_cast<size_t>(slot.slot_id) < disks_[slot.disk_id].slots.size();
}

bool DiskSpillBlockCache::preadFull(int fd, void* data, size_t bytes, off_t offset) const {
    auto*  cursor = static_cast<char*>(data);
    size_t done   = 0;
    while (done < bytes) {
        const auto ret = ::pread(fd, cursor + done, bytes - done, offset + static_cast<off_t>(done));
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            RTP_LLM_LOG_WARNING("disk spill pread failed, errno=%d(%s)", errno, std::strerror(errno));
            return false;
        }
        if (ret == 0) {
            return false;
        }
        done += static_cast<size_t>(ret);
    }
    return true;
}

bool DiskSpillBlockCache::pwriteFull(int fd, const void* data, size_t bytes, off_t offset) const {
    const auto* cursor = static_cast<const char*>(data);
    size_t      done   = 0;
    while (done < bytes) {
        const auto ret = ::pwrite(fd, cursor + done, bytes - done, offset + static_cast<off_t>(done));
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            RTP_LLM_LOG_WARNING("disk spill pwrite failed, errno=%d(%s)", errno, std::strerror(errno));
            return false;
        }
        if (ret == 0) {
            return false;
        }
        done += static_cast<size_t>(ret);
    }
    return true;
}

void DiskSpillBlockCache::freeSlotLocked(const DiskSlot& slot) {
    if (!validateSlotLocked(slot)) {
        return;
    }
    auto& disk         = disks_[slot.disk_id];
    auto& rec          = disk.slots[slot.slot_id];
    rec.state          = SlotState::FREE;
    rec.cache_key      = 0;
    rec.key_generation = 0;
    rec.block_size     = 0;
    rec.is_complete    = false;
    if (std::find(disk.free_slots.begin(), disk.free_slots.end(), slot.slot_id) == disk.free_slots.end()) {
        disk.free_slots.push_back(slot.slot_id);
    }
}

void DiskSpillBlockCache::closeFiles() {
    for (auto& disk : disks_) {
        if (disk.fd >= 0) {
            ::close(disk.fd);
            disk.fd = -1;
        }
    }
}

void DiskSpillBlockCache::cleanupRunDirs() {
    for (const auto& disk : disks_) {
        if (!disk.run_dir.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(disk.run_dir, ec);
            if (ec) {
                RTP_LLM_LOG_WARNING(
                    "disk spill cleanup dir failed, path=%s err=%s", disk.run_dir.c_str(), ec.message().c_str());
            }
        }
    }
}

}  // namespace rtp_llm
