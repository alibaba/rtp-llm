#pragma once

#include <string>

namespace rtp_llm {

// Owns one disk mount for the L3 disk block pool: creates the "rtp_llm_disk_kv" work dir,
// holds an exclusive flock on it for the guard's lifetime, and clears stale backing/tmp
// files left by a previous process.
class BlockTreeDiskMountGuard {
public:
    BlockTreeDiskMountGuard() = default;
    ~BlockTreeDiskMountGuard();

    BlockTreeDiskMountGuard(const BlockTreeDiskMountGuard&)            = delete;
    BlockTreeDiskMountGuard& operator=(const BlockTreeDiskMountGuard&) = delete;

    // Returns true on success. On any failure the lock/fd are released before returning.
    bool init(const std::string& mount_path);

    const std::string& workDir() const;
    const std::string& mountPath() const;
    std::string        debugString() const;

private:
    bool initDirectoryAndLock();
    bool cleanupStaleFiles();
    void unlockAndClose();

    std::string mount_path_;
    std::string work_dir_;
    std::string lock_path_;
    int         lock_fd_{-1};
};

}  // namespace rtp_llm
