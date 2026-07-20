#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskMountGuard.h"

#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sstream>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {
namespace {

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
        struct stat st{};
        return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
    }
    return false;
}

bool directoryExists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool hasSuffix(const std::string& name, const std::string& suffix) {
    return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

BlockTreeDiskMountGuard::~BlockTreeDiskMountGuard() {
    unlockAndClose();
}

bool BlockTreeDiskMountGuard::init(const std::string& mount_path) {
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

const std::string& BlockTreeDiskMountGuard::workDir() const {
    return work_dir_;
}

const std::string& BlockTreeDiskMountGuard::mountPath() const {
    return mount_path_;
}

std::string BlockTreeDiskMountGuard::debugString() const {
    std::ostringstream oss;
    oss << "BlockTreeDiskMountGuard{mount=" << mount_path_ << ", work_dir=" << work_dir_ << ", lock=" << lock_path_
        << "}";
    return oss.str();
}

bool BlockTreeDiskMountGuard::initDirectoryAndLock() {
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

bool BlockTreeDiskMountGuard::cleanupStaleFiles() {
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
            (startsWith(name, "disk_block_pool_") && hasSuffix(name, ".bin")) || hasSuffix(name, ".tmp");
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

void BlockTreeDiskMountGuard::unlockAndClose() {
    if (lock_fd_ >= 0) {
        ::flock(lock_fd_, LOCK_UN);
        ::close(lock_fd_);
        lock_fd_ = -1;
    }
}

}  // namespace rtp_llm
