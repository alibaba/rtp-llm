#include "rtp_llm/cpp/cache/DistStorage3FSFile.h"

#include <filesystem>
#include <sys/stat.h>
#include <sys/statvfs.h>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace threefs {

inline struct timespec createTimeoutTimeSpec(int timeout_ms) {
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_ms / 1000;
    timeout.tv_nsec += (timeout_ms % 1000) * 1000000;
    return timeout;
}

DistStorage3FSFile::DistStorage3FSFile(const ThreeFSFileConfig& config,
                                       const ThreeFSIovHandle&  read_iov_handle,
                                       const ThreeFSIovHandle&  write_iov_handle):
    config_(config),
    read_iov_handle_(read_iov_handle),
    write_iov_handle_(write_iov_handle),
    full_path_file_name_(config.mountpoint + config.folder_name + config.filename) {}

DistStorage3FSFile::~DistStorage3FSFile() {
    close();
}

bool DistStorage3FSFile::isExist() {
    struct stat file_stat;
    return ::stat(full_path_file_name_.c_str(), &file_stat) == 0;
}

bool DistStorage3FSFile::open(bool write) {
    if (write) {  // assume write only trigger once
        close();
    }

    if (fd_ != -1) {
        RTP_LLM_LOG_DEBUG("file already opened, filename: %s", config_.filename.c_str());
        return true;
    }

    int flags = O_RDWR;
    if (write) {
        flags |= O_CREAT;
    }

    int fd = -1;
    fd     = ::open(full_path_file_name_.c_str(), flags, 0666);
    if (fd == -1) {
        RTP_LLM_LOG_WARNING(
            "open file failed, filepath: %s, fd: %d, errno: %s", full_path_file_name_.c_str(), fd, strerror(errno));
        return false;
    }

    // note: must register fd after create ior/iov
    auto ret = hf3fs_reg_fd(fd, 0);
    if (ret > 0) {
        // 直接失败, reopen放在外面做. TODO:
        close();
        RTP_LLM_LOG_WARNING("open file failed, hf3fs_reg_fd failed, errno: %s, file: %s, fd: %d",
                            strerror(ret),
                            full_path_file_name_.c_str(),
                            fd);
        return false;
    }
    fd_ = fd;
    return true;
}

bool DistStorage3FSFile::write(const std::vector<DistStorage::Iov>& iovs) {
    size_t file_len = 0;
    for (const auto& iov : iovs) {
        file_len += iov.len;
    }

    auto handle = initIovIor(write_iov_handle_, file_len, kDefaultWriteSizePerIo, false);
    if (handle == nullptr) {
        RTP_LLM_LOG_WARNING("write failed, init iov/ior failed, filename: %s", config_.filename.c_str());
        return false;
    }

    auto& ior      = handle->ior_handle.ior;
    auto& iov      = handle->iov_handle.iov;
    auto  iov_base = handle->iov_handle.iov_base;

    // copy blocks
    int64_t iov_offset = 0;
    auto    cuda_util  = handle->iov_handle.cuda_util;
    for (const auto& iov : iovs) {
        iov_offset += iov.len;
        if (iov.gpu_mem) {
            cuda_util->copyAsyncDeviceToHost(iov_base + iov_offset, iov.data.get(), iov.len);
        } else {
            memcpy(iov_base + iov_offset, iov.data.get(), iov.len);
        }
    }
    cuda_util->sync();

    // 读时预先打开文件, 写时推迟到实际写时再打开文件, 因为写时需要创建文件
    if (!open(true)) {
        RTP_LLM_LOG_WARNING("write failed, open file failed, filename: %s", config_.filename.c_str());
        return false;
    }

    auto result = doWrite(handle, iovs, file_len);
    if (!result) {
        RTP_LLM_LOG_WARNING("write failed, do write failed, filename: %s", config_.filename.c_str());
    }
    releaseIovIor(handle);
    return result;
}

bool DistStorage3FSFile::doWrite(const std::shared_ptr<ThreeFSHandle>& handle,
                                 const std::vector<DistStorage::Iov>&  iovs,
                                 size_t                                write_total_size) {
    auto& ior      = handle->ior_handle.ior;
    auto& iov      = handle->iov_handle.iov;
    auto  iov_base = handle->iov_handle.iov_base;

    const auto iov_block_size  = handle->iov_handle.iov_block_size;
    const auto ior_entries     = handle->ior_handle.ior_entries;
    int64_t    remaining_size  = write_total_size;
    int64_t    iov_offset      = 0;
    int        submit_io_count = 0;
    while (remaining_size > 0) {
        uint64_t cur_write_len = 0;
        if (iov_block_size > 0) {
            cur_write_len = std::min(remaining_size, calcLeftSizeInBlock(iov_block_size, iov_offset));
        } else {
            cur_write_len = remaining_size > kDefaultWriteSizePerIo ? kDefaultWriteSizePerIo : remaining_size;
        }

        auto ret = hf3fs_prep_io(ior, iov, false, iov_base + iov_offset, fd_, iov_offset, cur_write_len, nullptr);
        if (ret < 0) {
            // TODO(LXQ): 需要处理掉之前已提交的io请求(submit然后wait)
            RTP_LLM_LOG_WARNING(
                "write to 3fs failed, hf3fs_prep_io failed, errno: %s, filename: %s, submit_io_count: %d, ior_entries: %d, cur_write_len: %lu, iov_block_size: %lu, remaining_size: %ld, total size: %ld",
                strerror(-ret),
                config_.filename.c_str(),
                submit_io_count,
                ior_entries,
                cur_write_len,
                iov_block_size,
                remaining_size,
                write_total_size);
            return false;
        }
        ++submit_io_count;
        iov_offset += cur_write_len;
        remaining_size -= cur_write_len;

        if (submit_io_count < ior_entries && remaining_size > 0) {
            continue;
        }

        ret = hf3fs_submit_ios(ior);
        if (ret != 0) {
            RTP_LLM_LOG_WARNING("write to 3fs failed, hf3fs_submit_ios failed, errno: %s", strerror(-ret));
            return false;
        }

        bool async_wait_io = false;
        bool last_io       = remaining_size <= 0;
        if (config_.write_thread_pool) {
            // async wait io
            auto work_item = [this, handle, submit_io_count, write_total_size, last_io]() {
                waitForWriteIos(handle, submit_io_count, write_total_size, last_io);
                releaseIovIor(handle);
            };
            if (auto error_code = config_.write_thread_pool->pushTask(work_item, false);
                error_code != autil::ThreadPool::ERROR_NONE) {
                RTP_LLM_LOG_WARNING("write to 3fs failed, push work item failed, error code: %d, file: %s",
                                    error_code,
                                    config_.filename.c_str());
                async_wait_io = false;
                // return false;
            } else {
                async_wait_io = true;
            }
        }
        if (!async_wait_io) {
            // sync wait io
            if (!waitForWriteIos(handle, submit_io_count, write_total_size, last_io)) {
                RTP_LLM_LOG_WARNING(
                    "write to 3fs failed, wait for write ios failed, file: %s, cur_write_len: %lu, iov_block_size: %lu",
                    config_.filename.c_str(),
                    cur_write_len,
                    iov_block_size);
                return false;
            }
        }
        submit_io_count = 0;  // reset submit io count
    }
    // file sync?
    return true;
}
bool DistStorage3FSFile::waitForWriteIos(const std::shared_ptr<ThreeFSHandle>& handle,
                                         int32_t                               submit_io_count,
                                         int64_t                               write_total_len,
                                         bool                                  last_io) const {
    if (submit_io_count <= 0) {
        return true;
    }
    if (handle == nullptr || handle->ior_handle.ior == nullptr) {
        return false;
    }

    hf3fs_cqe cqes[submit_io_count];
    int       timeout_ms         = 2000;  // TODO(LXQ) timeout threhold
    auto      time_spec          = createTimeoutTimeSpec(timeout_ms);
    auto      ior                = handle->ior_handle.ior;
    int       completed_io_count = hf3fs_wait_for_ios(ior, cqes, submit_io_count, submit_io_count, &time_spec);
    if (completed_io_count < 0) {
        RTP_LLM_LOG_WARNING(
            "wait write io failed, hf3fs_wait_for_ios failed, errno: %s, submit io count: %d, file: %s, write total len: %ld",
            strerror(-completed_io_count),
            submit_io_count,
            config_.filename.c_str(),
            write_total_len);
        return false;
    } else if (completed_io_count == 0) {
        RTP_LLM_LOG_WARNING(
            "wait write io but hf3fs_wait_for_ios return 0, maybe timeout(%d ms), submit io count: %d, file: %s, write total len: %ld",
            timeout_ms,
            submit_io_count,
            config_.filename.c_str(),
            write_total_len);
    }

    for (int i = 0; i < completed_io_count; ++i) {
        if (cqes[i].result < 0) {
            RTP_LLM_LOG_WARNING(
                "wait write io failed, cqe result errno: %s, submit_io_count: %d, completed_io_count: %d, file: %s, write total len: %ld",
                strerror(-cqes[i].result),
                submit_io_count,
                completed_io_count,
                config_.filename.c_str(),
                write_total_len);
            return false;
        }
    }

    if (last_io) {
        if (fd_ != -1) {
            ::fsync(fd_);
        }
    }
    return true;
}

int64_t DistStorage3FSFile::calcLeftSizeInBlock(int64_t iov_block_size, int64_t iov_offset) const {
    // 计算当前 iov block 块剩余可用的大小, 避免跨 block 读写
    const int64_t block_start        = (iov_offset / iov_block_size) * iov_block_size;
    const int64_t block_end          = block_start + iov_block_size;
    const int64_t left_size_in_block = block_end - iov_offset;
    return left_size_in_block;
}

bool DistStorage3FSFile::read(const std::vector<DistStorage::Iov>& iovs) {
    if (!open()) {
        return false;
    }

    size_t file_len = 0;
    for (const auto& iov : iovs) {
        file_len += iov.len;
    }

    auto handle = initIovIor(write_iov_handle_, file_len, kDefaultWriteSizePerIo, false);
    if (handle == nullptr) {
        RTP_LLM_LOG_WARNING("write failed, init iov/ior failed, filename: %s", config_.filename.c_str());
        return false;
    }

    if (file_len > handle->iov_handle.iov_size) {
        RTP_LLM_LOG_WARNING("read failed, read size exceed iov size, read size: %lu, iov size: %lu",
                            file_len,
                            handle->iov_handle.iov_size,
                            file_len);
        return false;
    }

    bool result = doRead(handle, iovs, file_len);
    releaseIovIor(handle);
    return result;
}

bool DistStorage3FSFile::doRead(const std::shared_ptr<ThreeFSHandle>& handle,
                                const std::vector<DistStorage::Iov>&  iovs,
                                size_t                                file_len) {
    if (handle == nullptr) {
        RTP_LLM_LOG_WARNING("read failed, handle is nullptr");
        return false;
    }
    auto& ior      = handle->ior_handle.ior;
    auto& iov      = handle->iov_handle.iov;
    auto  iov_base = handle->iov_handle.iov_base;

    const auto ior_entries     = handle->ior_handle.ior_entries;
    int64_t    iov_offset      = 0;
    int64_t    file_offset     = 0;
    int32_t    submit_io_count = 0;

    // 目前的实现是一次读完, 可以改成边读边Copy到显存
    for (auto& recv_iov : iovs) {
        // read one iov
        const auto iov_block_size = recv_iov.len;
        uint64_t   remaining_size = iov_block_size;
        while (remaining_size > 0) {
            uint64_t cur_read_len = std::min(remaining_size, (uint64_t)calcLeftSizeInBlock(iov_block_size, iov_offset));
            auto ret = hf3fs_prep_io(ior, iov, true, iov_base + iov_offset, fd_, file_offset, cur_read_len, nullptr);
            if (ret < 0) {
                RTP_LLM_LOG_WARNING(
                    "read block failed, hf3fs_prep_io failed, errno: %s, filename: %s, fd: %d, submit_io_count: %d, ior_entries: %d, cur_read_len: %lu, iov_block_size: %lu, remaining_size: %ld, total size: %ld",
                    strerror(-ret),
                    config_.filename.c_str(),
                    fd_,
                    submit_io_count,
                    ior_entries,
                    cur_read_len,
                    iov_block_size,
                    remaining_size,
                    file_len);
                return false;
            }
            ++submit_io_count;
            iov_offset += cur_read_len;  // 这里iov_offset 一直和file_offset一致.
            file_offset += cur_read_len;
            remaining_size -= cur_read_len;

            // submit io when enough
            if (submit_io_count >= ior_entries) {
                // 没得读或者submit_io_count 达到最大
                if (!submitAndWaitForReadIos(handle, submit_io_count)) {
                    RTP_LLM_LOG_WARNING(
                        "read block failed, read submit/wait io failed. file: %s, cur_read_len: %lu, iov_block_size: %lu",
                        config_.filename.c_str(),
                        cur_read_len,
                        iov_block_size);
                    return false;
                }
                submit_io_count = 0;
            }
        }
    }
    if (submit_io_count > 0) {  // last io
        if (!submitAndWaitForReadIos(handle, submit_io_count)) {
            RTP_LLM_LOG_WARNING("read block failed, read submit/wait io failed. file: %s", config_.filename.c_str());
            return false;
        }
    }

    iov_offset     = 0;
    auto cuda_util = handle->iov_handle.cuda_util;
    for (auto& iov : iovs) {
        if (iov.data != nullptr) {
            if (iov.gpu_mem) {
                cuda_util->copyAsyncHostToDevice(iov.data.get(), iov_base + iov_offset, iov.len);
            } else {
                memcpy(iov.data.get(), iov_base + iov_offset, iov.len);
            }
        }
        iov_offset += iov.len;
    }
    cuda_util->sync();
    return true;
}

bool DistStorage3FSFile::submitAndWaitForReadIos(const std::shared_ptr<ThreeFSHandle>& handle,
                                                 int32_t                               submit_io_count) const {
    if (submit_io_count <= 0) {
        return true;
    }

    auto ior = handle->ior_handle.ior;
    auto ret = hf3fs_submit_ios(ior);
    if (ret != 0) {
        RTP_LLM_LOG_WARNING("submit read io failed, hf3fs_submit_ios failed, errno: %s", strerror(-ret));
        return false;
    }

    hf3fs_cqe cqes[submit_io_count];
    int       timeout_ms         = 1000;  // 1s
    auto      time_spec          = createTimeoutTimeSpec(timeout_ms);
    auto      completed_io_count = hf3fs_wait_for_ios(ior, cqes, submit_io_count, submit_io_count, &time_spec);
    if (completed_io_count < 0) {
        RTP_LLM_LOG_WARNING("wait read io failed, hf3fs_wait_for_ios failed, errno: %s, submit io count: %d",
                            strerror(-completed_io_count),
                            submit_io_count);
        return false;
    } else if (completed_io_count == 0) {
        RTP_LLM_LOG_WARNING("wait read io but hf3fs_wait_for_ios return 0, maybe timeout(%d ms), submit io count: %d",
                            timeout_ms,
                            submit_io_count);
        return false;
    }
    for (int i = 0; i < completed_io_count; ++i) {
        if (cqes[i].result < 0) {
            RTP_LLM_LOG_WARNING(
                "wait read io failed, cqe read result error: %s, submit_io_count: %d, completed_io_count: %d",
                strerror(-cqes[i].result),
                submit_io_count,
                completed_io_count);
            return false;
        }
    }
    if (completed_io_count != submit_io_count) {
        RTP_LLM_LOG_WARNING(
            "wait read io failed, submit io count not equal completed io count, submit io count: %d, completed io count: %d",
            submit_io_count,
            completed_io_count);
        return false;
    }
    return true;
}

bool DistStorage3FSFile::del() {
    if (fd_ != -1) {
        hf3fs_dereg_fd(fd_);
        ::close(fd_);
        fd_ = -1;
    }
    // TODO: should check result?
    ::remove(full_path_file_name_.c_str());
    RTP_LLM_LOG_DEBUG("remove 3fs file: %s", full_path_file_name_.c_str());
}

bool DistStorage3FSFile::close() {
    if (fd_ != -1) {
        hf3fs_dereg_fd(fd_);
        ::close(fd_);
        fd_ = -1;
    }
}

std::shared_ptr<ThreeFSHandle>
DistStorage3FSFile::initIovIor(ThreeFSIovHandle& iov_handle, int64_t file_len, int32_t size_per_io, bool for_read) {
    auto iov_buffer = iov_handle.iov_mempool->alloc(file_len);
    if (iov_buffer == nullptr) {
        RTP_LLM_LOG_WARNING(
            "init iov/ior failed, mempool alloc failed, alloc len: %zu, mempool free size: %zu, file: %s",
            file_len,
            iov_handle.iov_mempool->freeSize(),
            config_.filename.c_str());
        return nullptr;
    }

    // 写时由于是异步wait io, 所以文件太大可能会出现ior entry不够用的情况, 此处根据文件长度计算所需的ior entry;
    // 读是同步读所以不会出现此问题
    ThreeFSIorHandle ior_handle;
    const int32_t    iov_block_size = iov_handle.iov_block_size != 0 ? iov_handle.iov_block_size : size_per_io;
    ior_handle.ior_entries          = file_len / iov_block_size + 1;

    struct hf3fs_ior* ior{nullptr};
    if (!createIor(ior, for_read, ior_handle.ior_entries, ior_handle.ior_io_depth, ior_handle.ior_timeout_ms)) {
        RTP_LLM_LOG_WARNING(
            "init iov/ior failed, create ior failed, filename: %s, ior entries: %d, ior io depth: %d, ior timeout ms: %d",
            config_.filename.c_str(),
            ior_handle.ior_entries,
            ior_handle.ior_io_depth,
            ior_handle.ior_timeout_ms);
        return nullptr;
    }

    auto handle                 = std::make_shared<ThreeFSHandle>();
    handle->iov_handle          = iov_handle;
    handle->iov_handle.iov_size = file_len;
    handle->iov_handle.iov_base = static_cast<uint8_t*>(iov_buffer);
    handle->ior_handle          = ior_handle;
    handle->ior_handle.ior      = ior;

    return handle;
    return nullptr;
}

bool DistStorage3FSFile::createIor(
    struct hf3fs_ior*& ior, bool for_read, int ior_entries, int ior_io_depth, int ior_timeout_ms) const {
    if (config_.mountpoint.empty()) {
        RTP_LLM_LOG_WARNING("create ior failed, mountpoint is empty");
        return false;
    }

    ior = new struct hf3fs_ior();
    auto ret =
        hf3fs_iorcreate4(ior, config_.mountpoint.c_str(), ior_entries, for_read, ior_io_depth, ior_timeout_ms, -1, 0);
    if (ret != 0) {
        RTP_LLM_LOG_WARNING(
            "hf3fs_iorcreate4 failed, read: %d, errno: %s, mountpoint: %s, entries: %d, io depth: %d, timeout: %d",
            for_read,
            strerror(-ret),
            config_.mountpoint.c_str(),
            ior_entries,
            ior_io_depth,
            ior_timeout_ms);
        hf3fs_iordestroy(ior);
        delete ior;
        ior = nullptr;
        return false;
    }
    return true;
}

void DistStorage3FSFile::releaseIovIor(const std::shared_ptr<ThreeFSHandle>& handle) {
    if (handle == nullptr) {
        return;
    }
    auto& iov_handle = handle->iov_handle;
    if (iov_handle.iov_base && iov_handle.iov_mempool) {
        iov_handle.iov_mempool->free(static_cast<void*>(iov_handle.iov_base));
    }

    // iov_handle.iov      = nullptr;
    iov_handle.iov_base = nullptr;
    auto& ior_handle    = handle->ior_handle;
    if (ior_handle.ior != nullptr) {
        hf3fs_iordestroy(ior_handle.ior);
        delete ior_handle.ior;
        ior_handle.ior = nullptr;
    }
}

}  // namespace threefs

}  // namespace rtp_llm