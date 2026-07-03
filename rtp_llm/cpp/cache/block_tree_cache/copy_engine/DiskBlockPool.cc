#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/DiskBlockPool.h"

#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sstream>
#include <sys/file.h>
#include <sys/stat.h>
#include <utility>
#include <unistd.h>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {
namespace {

constexpr size_t kDiskIOAlignment = 4096;

std::string joinPath(const std::string& parent, const std::string& child) {
    if (parent.empty() || parent.back() == '/') {
        return parent + child;
    }
    return parent + "/" + child;
}

bool mkdirIfMissing(const std::string& path) {
    if (::mkdir(path.c_str(), 0755) == 0) {
        return true;
    }
    if (errno == EEXIST) {
        struct stat st;
        return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
    }
    return false;
}

bool directoryExists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

}  // namespace

DiskMountGuard::~DiskMountGuard() {
    unlockAndClose();
}

bool DiskMountGuard::init(const std::string& mount_path) {
    mount_path_ = mount_path;
    work_dir_   = joinPath(mount_path_, "rtp_llm_disk_kv");
    lock_path_  = joinPath(work_dir_, ".lock");
    if (!initDirectoryAndLock() || !cleanupStaleFiles()) {
        unlockAndClose();
        return false;
    }
    RTP_LLM_LOG_INFO("disk kv mount guard init success: %s", debugString().c_str());
    return true;
}

const std::string& DiskMountGuard::workDir() const {
    return work_dir_;
}

const std::string& DiskMountGuard::mountPath() const {
    return mount_path_;
}

std::string DiskMountGuard::debugString() const {
    std::ostringstream oss;
    oss << "DiskMountGuard{mount=" << mount_path_ << ", work_dir=" << work_dir_ << ", lock=" << lock_path_ << "}";
    return oss.str();
}

bool DiskMountGuard::initDirectoryAndLock() {
    if (!directoryExists(mount_path_)) {
        RTP_LLM_LOG_ERROR("disk kv mount path does not exist or is not a directory, mount=%s, error=%s",
                          mount_path_.c_str(),
                          std::strerror(errno));
        return false;
    }

    if (!mkdirIfMissing(work_dir_)) {
        RTP_LLM_LOG_ERROR("create disk kv directory failed, mount=%s, work_dir=%s, error=%s",
                          mount_path_.c_str(),
                          work_dir_.c_str(),
                          std::strerror(errno));
        return false;
    }

    lock_fd_ = ::open(lock_path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (lock_fd_ < 0) {
        RTP_LLM_LOG_ERROR("open disk kv lock failed, lock=%s, error=%s", lock_path_.c_str(), std::strerror(errno));
        return false;
    }
    if (::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
        RTP_LLM_LOG_ERROR("lock disk kv mount failed, lock=%s, error=%s", lock_path_.c_str(), std::strerror(errno));
        unlockAndClose();
        return false;
    }
    return true;
}

bool DiskMountGuard::cleanupStaleFiles() {
    DIR* dir = ::opendir(work_dir_.c_str());
    if (dir == nullptr) {
        RTP_LLM_LOG_ERROR("open disk kv work dir failed, dir=%s, error=%s", work_dir_.c_str(), std::strerror(errno));
        return false;
    }
    while (auto* entry = ::readdir(dir)) {
        const std::string name(entry->d_name);
        if (name == "." || name == ".." || name == ".lock") {
            continue;
        }
        const bool framework_file =
            (startsWith(name, "rank_") && (name.size() >= 3 && name.substr(name.size() - 3) == ".kv"))
            || (name.size() >= 4 && name.substr(name.size() - 4) == ".tmp");
        if (!framework_file) {
            continue;
        }
        const auto path = joinPath(work_dir_, name);
        if (::unlink(path.c_str()) != 0 && errno != ENOENT) {
            RTP_LLM_LOG_ERROR(
                "remove stale disk kv file failed, file=%s, error=%s", path.c_str(), std::strerror(errno));
            ::closedir(dir);
            return false;
        }
    }
    ::closedir(dir);
    return true;
}

void DiskMountGuard::unlockAndClose() {
    if (lock_fd_ >= 0) {
        ::flock(lock_fd_, LOCK_UN);
        ::close(lock_fd_);
        lock_fd_ = -1;
    }
}

DiskBlockPool::DiskBlockPool(DiskBlockPoolConfig config, std::unique_ptr<IDiskBlockIO> io):
    config_(std::move(config)), io_(std::move(io)) {
    if (!io_) {
        io_ = std::make_unique<PosixDiskBlockIO>();
    }
}

DiskBlockPool::~DiskBlockPool() {
    if (io_) {
        io_->close();
    }
}

size_t DiskBlockPool::alignUp(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

bool DiskBlockPool::init() {
    if (config_.work_dir.empty() || config_.disk_size_bytes == 0 || config_.block_size_bytes == 0) {
        RTP_LLM_LOG_ERROR("init disk block pool failed, invalid config: %s", debugString().c_str());
        return false;
    }
    slot_stride_bytes_ = alignUp(config_.block_size_bytes, kDiskIOAlignment);
    slot_count_        = config_.disk_size_bytes / slot_stride_bytes_;
    if (slot_count_ == 0) {
        RTP_LLM_LOG_ERROR("init disk block pool failed, disk size too small, disk=%zu, block=%zu, stride=%zu",
                          config_.disk_size_bytes,
                          config_.block_size_bytes,
                          slot_stride_bytes_);
        return false;
    }

    if (!initFile()) {
        if (!file_path_.empty()) {
            ::unlink(file_path_.c_str());
        }
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.assign(slot_count_, SlotState{});
        free_slots_.clear();
        for (size_t i = 0; i < slot_count_; ++i) {
            free_slots_.insert(static_cast<int32_t>(i));
        }
    }

    RTP_LLM_LOG_INFO("disk kv block pool init success: %s", debugString().c_str());
    return true;
}

bool DiskBlockPool::initFile() {
    file_path_ =
        joinPath(config_.work_dir,
                 fmtstr("rank_%ld_world_%ld_%s.kv", config_.local_rank, config_.world_rank, config_.tag.c_str()));
    return io_->openAndPreallocate(file_path_, slot_count_ * slot_stride_bytes_, config_.buffered_io);
}

std::optional<int32_t> DiskBlockPool::malloc() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_slots_.empty()) {
        return std::nullopt;
    }
    const auto slot = *free_slots_.begin();
    free_slots_.erase(free_slots_.begin());
    slots_[static_cast<size_t>(slot)].request_ref++;
    return slot;
}

bool DiskBlockPool::validSlot(int32_t slot) const {
    return slot >= 0 && static_cast<size_t>(slot) < slot_count_;
}

void DiskBlockPool::requestReference(int32_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validSlot(slot)) {
        return;
    }
    auto& state = slots_[static_cast<size_t>(slot)];
    state.request_ref++;
    free_slots_.erase(slot);
}

void DiskBlockPool::requestFree(int32_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validSlot(slot)) {
        return;
    }
    auto& state = slots_[static_cast<size_t>(slot)];
    if (state.request_ref > 0) {
        state.request_ref--;
    }
    tryFreeSlotLocked(slot);
}

void DiskBlockPool::blockCacheReference(int32_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validSlot(slot)) {
        return;
    }
    auto& state = slots_[static_cast<size_t>(slot)];
    state.cache_ref++;
    free_slots_.erase(slot);
}

void DiskBlockPool::blockCacheFree(int32_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validSlot(slot)) {
        return;
    }
    auto& state = slots_[static_cast<size_t>(slot)];
    if (state.cache_ref > 0) {
        state.cache_ref--;
    }
    tryFreeSlotLocked(slot);
}

void DiskBlockPool::tryFreeSlotLocked(int32_t slot) {
    auto& state = slots_[static_cast<size_t>(slot)];
    if (state.request_ref == 0 && state.cache_ref == 0) {
        free_slots_.insert(slot);
    }
}

bool DiskBlockPool::read(int32_t slot, void* dst, size_t bytes) {
    if (!validSlot(slot) || bytes > slot_stride_bytes_) {
        return false;
    }
    const uint64_t offset = static_cast<uint64_t>(slot) * slot_stride_bytes_;
    if (!io_->read(offset, dst, bytes)) {
        return false;
    }
    read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

bool DiskBlockPool::write(int32_t slot, const void* src, size_t bytes) {
    if (!validSlot(slot) || bytes > slot_stride_bytes_) {
        return false;
    }
    const uint64_t offset = static_cast<uint64_t>(slot) * slot_stride_bytes_;
    if (!io_->write(offset, src, bytes)) {
        return false;
    }
    write_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

size_t DiskBlockPool::totalSlots() const {
    return slot_count_;
}

size_t DiskBlockPool::freeSlots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_slots_.size();
}

size_t DiskBlockPool::availableSlots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      available = 0;
    for (const auto& state : slots_) {
        if (state.request_ref == 0) {
            ++available;
        }
    }
    return available;
}

size_t DiskBlockPool::blockSizeBytes() const {
    return config_.block_size_bytes;
}

size_t DiskBlockPool::slotStrideBytes() const {
    return slot_stride_bytes_;
}

size_t DiskBlockPool::readBytes() const {
    return read_bytes_.load(std::memory_order_relaxed);
}

size_t DiskBlockPool::writeBytes() const {
    return write_bytes_.load(std::memory_order_relaxed);
}

const std::string& DiskBlockPool::filePath() const {
    return file_path_;
}

std::string DiskBlockPool::debugString() const {
    std::ostringstream oss;
    oss << "DiskBlockPool{work_dir=" << config_.work_dir << ", file=" << file_path_
        << ", local_rank=" << config_.local_rank << ", world_rank=" << config_.world_rank << ", tag=" << config_.tag
        << ", disk_size=" << config_.disk_size_bytes << ", block_size=" << config_.block_size_bytes
        << ", stride=" << slot_stride_bytes_ << ", slots=" << slot_count_
        << ", io=" << (config_.buffered_io ? "buffered" : "direct") << "}";
    return oss.str();
}

}  // namespace rtp_llm
