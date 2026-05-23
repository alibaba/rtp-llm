#include "rtp_llm/cpp/cache/connector/memory/DiskSpillFileManager.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <linux/fs.h>
#include <random>
#include <sstream>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

constexpr size_t kMinAlign = 512;
constexpr size_t kMaxAlign = 65536;
constexpr size_t kUnhealthyThreshold = 5;

uint64_t hashString(const std::string& s) {
    std::hash<std::string> h;
    return static_cast<uint64_t>(h(s));
}

std::string toHexShort(uint64_t v) {
    std::ostringstream oss;
    oss << std::hex << v;
    return oss.str();
}

bool removePathRecursive(const std::string& path, std::string& error) {
    struct stat st;
    if (::lstat(path.c_str(), &st) != 0) {
        if (errno == ENOENT) {
            return true;
        }
        error = "lstat failed errno=" + std::to_string(errno) + "(" + std::strerror(errno) + ")";
        return false;
    }
    if (!S_ISDIR(st.st_mode)) {
        if (::unlink(path.c_str()) == 0 || errno == ENOENT) {
            return true;
        }
        error = "unlink failed errno=" + std::to_string(errno) + "(" + std::strerror(errno) + ")";
        return false;
    }

    DIR* dir = ::opendir(path.c_str());
    if (dir == nullptr) {
        error = "opendir failed errno=" + std::to_string(errno) + "(" + std::strerror(errno) + ")";
        return false;
    }
    bool ok = true;
    while (auto* entry = ::readdir(dir)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        std::string child_error;
        if (!removePathRecursive(path + "/" + name, child_error)) {
            ok    = false;
            error = child_error;
            break;
        }
    }
    ::closedir(dir);
    if (!ok) {
        return false;
    }
    if (::rmdir(path.c_str()) == 0 || errno == ENOENT) {
        return true;
    }
    error = "rmdir failed errno=" + std::to_string(errno) + "(" + std::strerror(errno) + ")";
    return false;
}

}  // namespace

DiskSpillFileManager::DiskSpillFileManager(Config config): config_(std::move(config)) {}

DiskSpillFileManager::~DiskSpillFileManager() {
    shutdown();
}

bool DiskSpillFileManager::init() {
    if (config_.base_path.empty()) {
        last_error_ = "base_path empty";
        return false;
    }
    if (config_.slot_stride_bytes == 0) {
        last_error_ = "slot_stride_bytes must be > 0";
        return false;
    }
    if (config_.segment_bytes == 0 || config_.segment_bytes < config_.slot_stride_bytes) {
        last_error_ = "segment_bytes < slot_stride_bytes";
        return false;
    }
    if (config_.capacity_bytes < config_.slot_stride_bytes) {
        last_error_ = "capacity_bytes < slot_stride_bytes";
        return false;
    }
    if (config_.schema_hash.empty()) {
        last_error_ = "schema_hash empty";
        return false;
    }
    if (config_.startup_uuid.empty()) {
        last_error_ = "startup_uuid empty";
        return false;
    }
    if (config_.hostname.empty()) {
        char buf[256] = {0};
        if (::gethostname(buf, sizeof(buf) - 1) == 0) {
            config_.hostname = buf;
        } else {
            config_.hostname = "unknown";
        }
    }

    path_hash_ = makePathHash();

    if (!detectAlignment()) {
        return false;
    }
    if (!ensureStartupDirAndLock()) {
        return false;
    }
    if (!probeDirectIO()) {
        // probeDirectIO sets last_error_ + io_mode_
        if (config_.direct_io_required) {
            return false;
        }
    }
    if (!openSegments()) {
        return false;
    }
    if (!allocateStagingPool()) {
        return false;
    }

    RTP_LLM_LOG_INFO("disk spill file manager init success, disk_id=%zu path_hash=%s io_mode=%s align=%zu "
                     "slot_stride=%zu segment_bytes=%zu segments=%zu slots=%zu",
                     config_.disk_id,
                     path_hash_.c_str(),
                     io_mode_ == IoMode::DIRECT ? "direct" : "buffered",
                     align_bytes_,
                     config_.slot_stride_bytes,
                     config_.segment_bytes,
                     segments_.size(),
                     slot_count_);
    return true;
}

void DiskSpillFileManager::shutdown() {
    for (auto& seg : segments_) {
        if (seg.fd >= 0) {
            ::close(seg.fd);
            seg.fd = -1;
        }
    }
    segments_.clear();
    releaseStagingPool();
    releaseStartupLock();
    if (config_.cleanup_on_destroy) {
        cleanupRunDir();
    }
    io_mode_ = IoMode::NONE;
}

bool DiskSpillFileManager::detectAlignment() {
    if (config_.align_bytes > 0) {
        align_bytes_ = std::max(kMinAlign, std::min(config_.align_bytes, kMaxAlign));
        return true;
    }
    // Probe logical block size of the underlying device. We open the base_path
    // (or its parent if it doesn't exist yet) and ioctl(BLKSSZGET). Path-on-tmpfs
    // and other non-block fs returns ENOTTY -> fallback 4096.
    std::error_code ec;
    if (!std::filesystem::exists(config_.base_path, ec)) {
        std::filesystem::create_directories(config_.base_path, ec);
        if (ec) {
            last_error_ = "create base_path failed: " + ec.message();
            return false;
        }
    }
    int fd = ::open(config_.base_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        align_bytes_ = 4096;
        return true;
    }
    int bsz = 0;
    if (::ioctl(fd, BLKSSZGET, &bsz) == 0 && bsz > 0) {
        align_bytes_ = std::max(kMinAlign, std::min<size_t>(bsz, kMaxAlign));
    } else {
        align_bytes_ = 4096;
    }
    ::close(fd);
    // Ensure slot_stride is aligned to align_bytes_; caller already aligned upstream
    if (config_.slot_stride_bytes % align_bytes_ != 0) {
        last_error_ = "slot_stride_bytes (" + std::to_string(config_.slot_stride_bytes)
                      + ") not aligned to align_bytes (" + std::to_string(align_bytes_) + ")";
        return false;
    }
    return true;
}

bool DiskSpillFileManager::ensureStartupDirAndLock() {
    // path layout: ${base}/rtp_llm_mem_spill/schema_${hash}/rank_${rank}/host_${host}_pid_${pid}_uuid_${uuid}/
    std::filesystem::path root = std::filesystem::path(config_.base_path) / "rtp_llm_mem_spill";
    schema_dir_                = (root / ("schema_" + config_.schema_hash)).string();
    rank_dir_                  = (std::filesystem::path(schema_dir_) / ("rank_" + std::to_string(config_.world_rank)))
                    .string();
    std::ostringstream run_name;
    run_name << "host_" << config_.hostname << "_pid_" << ::getpid() << "_uuid_" << config_.startup_uuid;
    run_dir_ = (std::filesystem::path(rank_dir_) / run_name.str()).string();

    std::error_code ec;
    std::filesystem::create_directories(run_dir_, ec);
    if (ec) {
        last_error_ = "create run_dir failed: " + ec.message() + " path=" + run_dir_;
        return false;
    }

    // flock the run_dir to mark it as in-use
    const std::string lock_path = run_dir_ + "/.run.lock";
    lock_fd_                    = ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (lock_fd_ < 0) {
        last_error_ = "open lock file failed errno=" + std::to_string(errno);
        return false;
    }
    if (::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
        last_error_ = "flock lock file failed errno=" + std::to_string(errno);
        ::close(lock_fd_);
        lock_fd_ = -1;
        return false;
    }

    if (config_.cleanup_old_startup_dirs) {
        cleanupOldStartupDirs();
    }
    return true;
}

void DiskSpillFileManager::releaseStartupLock() {
    if (lock_fd_ >= 0) {
        ::flock(lock_fd_, LOCK_UN);
        ::close(lock_fd_);
        lock_fd_ = -1;
    }
}

void DiskSpillFileManager::cleanupRunDir() {
    if (run_dir_.empty()) {
        return;
    }
    std::string error;
    if (!removePathRecursive(run_dir_, error)) {
        RTP_LLM_LOG_WARNING("disk spill cleanup run_dir failed, path_hash=%s err=%s",
                            path_hash_.c_str(),
                            error.c_str());
    }
}

void DiskSpillFileManager::cleanupOldStartupDirs() {
    // Walk rank_dir_ siblings; remove any host_*_pid_*_uuid_* dir whose lock
    // is not currently held (flock LOCK_EX|LOCK_NB succeeds) AND mtime is older
    // than 10 minutes. Conservative: never remove our own dir, and skip on any
    // ambiguity. Failures are logged but non-fatal.
    if (rank_dir_.empty()) {
        return;
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(rank_dir_, ec)) {
        return;
    }
    const auto now      = std::chrono::system_clock::now();
    const auto stale_us = std::chrono::minutes(10);
    for (auto it = std::filesystem::directory_iterator(rank_dir_, ec); !ec && it != std::filesystem::directory_iterator();
         it.increment(ec)) {
        const auto& entry = *it;
        if (!entry.is_directory(ec) || entry.path().string() == run_dir_) {
            continue;
        }
        const auto name = entry.path().filename().string();
        if (name.find("host_") != 0) {
            continue;
        }
        const std::string sib_lock = entry.path().string() + "/.run.lock";
        struct stat       st;
        if (::stat(sib_lock.c_str(), &st) != 0) {
            continue;
        }
        const auto mtime_us =
            std::chrono::seconds(st.st_mtim.tv_sec) + std::chrono::microseconds(st.st_mtim.tv_nsec / 1000);
        const auto age = std::chrono::system_clock::now().time_since_epoch() - mtime_us;
        (void)now;
        if (age < stale_us) {
            continue;
        }
        const int fd = ::open(sib_lock.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            continue;
        }
        const bool acquired = (::flock(fd, LOCK_EX | LOCK_NB) == 0);
        if (acquired) {
            ::flock(fd, LOCK_UN);
            ::close(fd);
            const auto  entry_path = entry.path().string();
            std::string error;
            if (!removePathRecursive(entry_path, error)) {
                RTP_LLM_LOG_WARNING("disk spill cleanup stale dir failed, path=%s err=%s",
                                    entry_path.c_str(),
                                    error.c_str());
            } else {
                RTP_LLM_LOG_INFO("disk spill cleaned stale dir, path=%s", entry_path.c_str());
            }
        } else {
            ::close(fd);
        }
    }
}

bool DiskSpillFileManager::probeDirectIO() {
    if (!config_.direct_io) {
        io_mode_ = IoMode::BUFFERED;
        return true;
    }
    // Try opening a probe file with O_DIRECT
    const std::string probe_path = run_dir_ + "/.probe_direct";
    int fd = ::open(probe_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_DIRECT | O_TRUNC, 0600);
    if (fd < 0) {
        last_error_ = "O_DIRECT open failed errno=" + std::to_string(errno);
        io_mode_    = IoMode::BUFFERED;
        return false;
    }
    // posix_memalign aligned buffer
    void*       buf = nullptr;
    const auto  ret = ::posix_memalign(&buf, align_bytes_, align_bytes_);
    if (ret != 0 || buf == nullptr) {
        ::close(fd);
        ::unlink(probe_path.c_str());
        last_error_ = "posix_memalign failed: " + std::to_string(ret);
        io_mode_    = IoMode::BUFFERED;
        return false;
    }
    std::memset(buf, 0, align_bytes_);
    bool ok = (::pwrite(fd, buf, align_bytes_, 0) == static_cast<ssize_t>(align_bytes_));
    if (ok) {
        ok = (::pread(fd, buf, align_bytes_, 0) == static_cast<ssize_t>(align_bytes_));
    }
    if (!ok && align_bytes_ < 4096) {
        // try larger alignment
        free(buf);
        align_bytes_ = 4096;
        if (::posix_memalign(&buf, align_bytes_, align_bytes_) == 0 && buf != nullptr) {
            std::memset(buf, 0, align_bytes_);
            ok = (::pwrite(fd, buf, align_bytes_, 0) == static_cast<ssize_t>(align_bytes_));
            if (ok) {
                ok = (::pread(fd, buf, align_bytes_, 0) == static_cast<ssize_t>(align_bytes_));
            }
        }
    }
    free(buf);
    ::close(fd);
    ::unlink(probe_path.c_str());

    if (ok) {
        io_mode_ = IoMode::DIRECT;
        return true;
    }
    io_mode_ = IoMode::BUFFERED;
    if (config_.direct_io_required) {
        last_error_ = "O_DIRECT probe failed and direct_io_required=true";
        return false;
    }
    RTP_LLM_LOG_WARNING("disk spill O_DIRECT probe failed, falling back to buffered, path_hash=%s",
                        path_hash_.c_str());
    return true;
}

bool DiskSpillFileManager::openSegments() {
    slots_per_segment_ = config_.segment_bytes / config_.slot_stride_bytes;
    if (slots_per_segment_ == 0) {
        last_error_ = "slots_per_segment == 0";
        return false;
    }
    slot_count_                  = config_.capacity_bytes / config_.slot_stride_bytes;
    const size_t segments_needed = (slot_count_ + slots_per_segment_ - 1) / slots_per_segment_;
    segments_.reserve(segments_needed);

    int open_flags = O_CREAT | O_RDWR | O_CLOEXEC;
    if (io_mode_ == IoMode::DIRECT) {
        open_flags |= O_DIRECT;
    }
    size_t remaining_slots = slot_count_;
    for (size_t seg_id = 0; seg_id < segments_needed; ++seg_id) {
        Segment seg;
        seg.path = run_dir_ + "/disk_" + std::to_string(config_.disk_id) + "_seg_" + std::to_string(seg_id) + ".bin";
        seg.fd   = ::open(seg.path.c_str(), open_flags, 0600);
        if (seg.fd < 0) {
            last_error_ = "open segment failed path=" + seg.path + " errno=" + std::to_string(errno);
            return false;
        }
        const size_t cur_slots   = std::min(remaining_slots, slots_per_segment_);
        const size_t cur_bytes   = cur_slots * config_.slot_stride_bytes;
        if (::ftruncate(seg.fd, static_cast<off_t>(cur_bytes)) != 0) {
            ::close(seg.fd);
            last_error_ = "ftruncate failed path=" + seg.path + " errno=" + std::to_string(errno);
            return false;
        }
        seg.slots_in_segment = cur_slots;
        segments_.push_back(std::move(seg));
        remaining_slots -= cur_slots;
        if (remaining_slots == 0) {
            break;
        }
    }
    return true;
}

bool DiskSpillFileManager::allocateStagingPool() {
    const int pool_size = std::max(1, config_.max_staging_buffers);
    staging_all_.reserve(pool_size);
    staging_free_list_.reserve(pool_size);
    for (int i = 0; i < pool_size; ++i) {
        void*       addr = nullptr;
        const auto  ret  = ::posix_memalign(&addr, align_bytes_, config_.slot_stride_bytes);
        if (ret != 0 || addr == nullptr) {
            last_error_ = "posix_memalign staging buffer failed: " + std::to_string(ret);
            return false;
        }
        auto buf = std::shared_ptr<StagingBuffer>(new StagingBuffer(addr, config_.slot_stride_bytes),
                                                   [](StagingBuffer* p) {
                                                       if (p && p->data()) {
                                                           free(p->data());
                                                       }
                                                       delete p;
                                                   });
        staging_all_.push_back(buf);
        staging_free_list_.push_back(buf);
    }
    return true;
}

void DiskSpillFileManager::releaseStagingPool() {
    std::lock_guard<std::mutex> lock(staging_mutex_);
    staging_free_list_.clear();
    staging_all_.clear();
    staging_used_.store(0);
}

std::shared_ptr<DiskSpillFileManager::StagingBuffer> DiskSpillFileManager::acquireStagingBuffer() {
    std::lock_guard<std::mutex> lock(staging_mutex_);
    if (staging_free_list_.empty()) {
        return nullptr;
    }
    auto buf = staging_free_list_.back();
    staging_free_list_.pop_back();
    staging_used_.fetch_add(1);
    return buf;
}

void DiskSpillFileManager::releaseStagingBuffer(std::shared_ptr<StagingBuffer> buffer) {
    if (!buffer) {
        return;
    }
    std::lock_guard<std::mutex> lock(staging_mutex_);
    staging_free_list_.push_back(buffer);
    if (staging_used_.load() > 0) {
        staging_used_.fetch_sub(1);
    }
}

int DiskSpillFileManager::fdForSlot(int slot_id, off_t& out_offset) const {
    if (slot_id < 0 || static_cast<size_t>(slot_id) >= slot_count_) {
        return -1;
    }
    const size_t seg_id        = static_cast<size_t>(slot_id) / slots_per_segment_;
    const size_t slot_in_seg   = static_cast<size_t>(slot_id) % slots_per_segment_;
    if (seg_id >= segments_.size()) {
        return -1;
    }
    out_offset = static_cast<off_t>(slot_in_seg * config_.slot_stride_bytes);
    return segments_[seg_id].fd;
}

bool DiskSpillFileManager::ioWrite(int fd, const void* buf, size_t bytes, off_t offset) const {
    const auto* cursor = static_cast<const char*>(buf);
    size_t      done   = 0;
    while (done < bytes) {
        const auto ret = ::pwrite(fd, cursor + done, bytes - done, offset + static_cast<off_t>(done));
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (ret == 0) {
            return false;
        }
        done += static_cast<size_t>(ret);
    }
    return true;
}

bool DiskSpillFileManager::ioRead(int fd, void* buf, size_t bytes, off_t offset) const {
    auto*  cursor = static_cast<char*>(buf);
    size_t done   = 0;
    while (done < bytes) {
        const auto ret = ::pread(fd, cursor + done, bytes - done, offset + static_cast<off_t>(done));
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (ret == 0) {
            return false;
        }
        done += static_cast<size_t>(ret);
    }
    return true;
}

bool DiskSpillFileManager::pwriteSlot(int slot_id, const void* data, size_t bytes) {
    if (unhealthy_.load()) {
        return false;
    }
    if (data == nullptr || bytes == 0 || bytes > config_.slot_stride_bytes) {
        recordFailure(disk_error::kShortIo);
        return false;
    }
    if (io_mode_ == IoMode::DIRECT) {
        if (bytes % align_bytes_ != 0 || (reinterpret_cast<uintptr_t>(data) % align_bytes_) != 0) {
            recordFailure(disk_error::kPwrite);
            RTP_LLM_LOG_WARNING("disk spill pwrite direct alignment violation, slot=%d bytes=%zu align=%zu",
                                slot_id,
                                bytes,
                                align_bytes_);
            return false;
        }
    }
    off_t      offset = 0;
    const int  fd     = fdForSlot(slot_id, offset);
    if (fd < 0) {
        recordFailure(disk_error::kPwrite);
        return false;
    }
    const bool ok = ioWrite(fd, data, bytes, offset);
    if (!ok) {
        const int saved = errno;
        recordFailure(saved == EIO ? disk_error::kPwrite : disk_error::kShortIo);
        RTP_LLM_LOG_WARNING("disk spill pwrite failed, path_hash=%s slot=%d bytes=%zu errno=%d(%s)",
                            path_hash_.c_str(),
                            slot_id,
                            bytes,
                            saved,
                            std::strerror(saved));
        return false;
    }
    if (io_mode_ == IoMode::BUFFERED) {
        // Drop pages from page cache; we don't want kvcache occupying RAM twice.
        ::posix_fadvise(fd, offset, static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
    }
    recordSuccess();
    return true;
}

bool DiskSpillFileManager::preadSlot(int slot_id, void* data, size_t bytes) {
    if (unhealthy_.load()) {
        return false;
    }
    if (data == nullptr || bytes == 0 || bytes > config_.slot_stride_bytes) {
        recordFailure(disk_error::kShortIo);
        return false;
    }
    if (io_mode_ == IoMode::DIRECT) {
        if (bytes % align_bytes_ != 0 || (reinterpret_cast<uintptr_t>(data) % align_bytes_) != 0) {
            recordFailure(disk_error::kPread);
            return false;
        }
    }
    off_t      offset = 0;
    const int  fd     = fdForSlot(slot_id, offset);
    if (fd < 0) {
        recordFailure(disk_error::kPread);
        return false;
    }
    const bool ok = ioRead(fd, data, bytes, offset);
    if (!ok) {
        const int saved = errno;
        recordFailure(saved == EIO ? disk_error::kPread : disk_error::kShortIo);
        RTP_LLM_LOG_WARNING("disk spill pread failed, path_hash=%s slot=%d bytes=%zu errno=%d(%s)",
                            path_hash_.c_str(),
                            slot_id,
                            bytes,
                            saved,
                            std::strerror(saved));
        return false;
    }
    if (io_mode_ == IoMode::BUFFERED) {
        ::posix_fadvise(fd, offset, static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
    }
    recordSuccess();
    return true;
}

bool DiskSpillFileManager::probeHealth() {
    if (segments_.empty()) {
        return false;
    }
    // small aligned round-trip using a staging buffer
    auto buf = acquireStagingBuffer();
    if (!buf) {
        return false;
    }
    const size_t probe_bytes = std::min<size_t>(config_.slot_stride_bytes, std::max<size_t>(align_bytes_, 4096));
    std::memset(buf->data(), 0xCC, probe_bytes);
    bool ok = ioWrite(segments_[0].fd, buf->data(), probe_bytes, 0);
    if (ok) {
        std::memset(buf->data(), 0, probe_bytes);
        ok = ioRead(segments_[0].fd, buf->data(), probe_bytes, 0);
    }
    releaseStagingBuffer(buf);
    if (ok) {
        unhealthy_.store(false);
        consecutive_failures_.store(0);
        return true;
    }
    return false;
}

size_t DiskSpillFileManager::slotCount() const {
    return slot_count_;
}

size_t DiskSpillFileManager::alignBytes() const {
    return align_bytes_;
}

DiskSpillFileManager::IoMode DiskSpillFileManager::ioMode() const {
    return io_mode_;
}

bool DiskSpillFileManager::isUnhealthy() const {
    return unhealthy_.load();
}

const std::string& DiskSpillFileManager::lastError() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}

const std::string& DiskSpillFileManager::pathHash() const {
    return path_hash_;
}

DiskSpillFileManager::Stats DiskSpillFileManager::getStats() const {
    Stats stats;
    stats.io_mode       = io_mode_;
    stats.slot_count    = slot_count_;
    stats.segment_count = segments_.size();
    stats.free_bytes    = 0;
    stats.used_bytes    = 0;
    {
        std::lock_guard<std::mutex> lock(staging_mutex_);
        stats.staging_used  = staging_used_.load();
        stats.staging_total = staging_all_.size();
    }
    stats.unhealthy            = unhealthy_.load();
    stats.consecutive_failures = consecutive_failures_.load();
    return stats;
}

void DiskSpillFileManager::recordFailure(const std::string& error_type) {
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = error_type;
    }
    const auto fails = consecutive_failures_.fetch_add(1) + 1;
    if (fails >= kUnhealthyThreshold) {
        const bool was_healthy = !unhealthy_.exchange(true);
        if (was_healthy) {
            RTP_LLM_LOG_WARNING("disk spill marked UNHEALTHY, path_hash=%s consecutive_failures=%zu last_error=%s",
                                path_hash_.c_str(),
                                fails,
                                error_type.c_str());
        }
    }
}

void DiskSpillFileManager::recordSuccess() {
    consecutive_failures_.store(0);
}

std::string DiskSpillFileManager::makePathHash() const {
    std::ostringstream oss;
    oss << config_.base_path << "|" << config_.disk_id << "|" << config_.schema_hash << "|" << config_.world_rank;
    return toHexShort(hashString(oss.str()));
}

}  // namespace rtp_llm
