#include "rtp_llm/cpp/cache/DistStorage3FSFile.h"

#include <filesystem>
#include <sys/stat.h>
#include <sys/statvfs.h>

#include "rtp_llm/cpp/cache/DistKvCacheMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "autil/Scope.h"

namespace rtp_llm {

namespace threefs {

inline struct timespec createTimeoutTimeSpec(int timeout_ms) {
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_ms / 1000;
    timeout.tv_nsec += (timeout_ms % 1000) * 1000000;
    return timeout;
}

inline float calcMiBs(int64_t len_byte, int64_t cost_us) {
    if (cost_us == 0) {
        return 0;
    }
    return len_byte * 1.0 / 1024 / 1024 * 1000 * 1000 / cost_us;
}

DistStorage3FSFile::DistStorage3FSFile(const ThreeFSFileConfig& config,
                                       const ThreeFSIovHandle&  read_iov_handle,
                                       const ThreeFSIovHandle&  write_iov_handle,
                                       size_t                   read_timeout_ms,
                                       size_t                   write_timeout_ms):
    config_(config),
    read_iov_handle_(read_iov_handle),
    write_iov_handle_(write_iov_handle),
    filepath_(config.filepath),
    read_timeout_ms_(read_timeout_ms),
    write_timeout_ms_(write_timeout_ms) {}

DistStorage3FSFile::~DistStorage3FSFile() {
    close();
}

bool DistStorage3FSFile::exists() const {
    struct stat file_stat;
    if (auto ret = ::stat(filepath_.c_str(), &file_stat); ret != 0) {
        return false;
    }
    return static_cast<int64_t>(file_stat.st_size) > 0;
}

bool DistStorage3FSFile::open(bool write) {
    if (write) {  // assume write only trigger once
        close();
    }

    if (fd_ != -1) {
        RTP_LLM_LOG_DEBUG("file already opened, file: %s", filepath_.c_str());
        return true;
    }

    if (write) {
        try {
            const auto parent_dir = std::filesystem::path(filepath_).parent_path();
            std::filesystem::create_directories(parent_dir);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("create directories failed, file: %s, exception: %s", filepath_.c_str(), e.what());
            return false;
        }
    }

    int flags = O_RDWR;
    if (write) {
        flags |= O_CREAT;
    }

    int fd = -1;
    fd     = ::open(filepath_.c_str(), flags, 0666);
    if (fd == -1) {
        RTP_LLM_LOG_WARNING(
            "open file failed, file: %s, write: %d, fd: %d, errno: %s", filepath_.c_str(), write, fd, strerror(errno));
        return false;
    }

    auto ret = hf3fs_reg_fd(fd, 0);
    if (ret > 0) {
        // 直接失败, reopen放在外面做. TODO:
        close();
        RTP_LLM_LOG_WARNING(
            "hf3fs_reg_fd failed, file: %s, write: %d, fd: %d, errno: %s", filepath_.c_str(), write, fd, strerror(ret));
        return false;
    }
    fd_ = fd;
    return true;
}

bool DistStorage3FSFile::write(const std::vector<DistStorage::Iov>& iovs) {
    size_t write_len = 0;
    for (const auto& iov : iovs) {
        if (iov.ignore) {
            continue;
        }
        write_len += iov.len;
        if (iov.data == nullptr) {
            RTP_LLM_LOG_WARNING("write failed, iov data is null, file: %s", filepath_.c_str());
            return false;
        }
    }
    if (iovs.empty() || write_len == 0) {
        RTP_LLM_LOG_WARNING(
            "write but iovs are invalid, size: %zu, len: %zu, file: %s", iovs.size(), write_len, filepath_.c_str());
        return true;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(config_.metrics_reporter);
    DistKvCacheMetrics::markTotalWriteBeginUs(metrics);

    auto handle = initIovIor(write_iov_handle_, write_len, kDefaultWriteSizePerIo, false);
    if (handle == nullptr) {
        DistKvCacheMetrics::markTotalWriteDoneUs(metrics);
        RTP_LLM_LOG_WARNING("write failed, init iov/ior failed, file: %s", filepath_.c_str());
        return false;
    }

    // copy blocks
    int64_t iov_offset = 0;
    auto    iov_base   = handle->iov_handle.iov_base;
    auto    cuda_util  = handle->iov_handle.cuda_util;
    DistKvCacheMetrics::markWriteCudaCopyBeginUs(metrics);
    for (const auto& iov : iovs) {
        if (iov.ignore) {
            continue;
        }
        if (iov.gpu_mem) {
            cuda_util->copyAsyncDeviceToHost(iov_base + iov_offset, iov.data.get(), iov.len);
        } else {
            memcpy(iov_base + iov_offset, iov.data.get(), iov.len);
        }
        iov_offset += iov.len;
    }
    cuda_util->sync();
    DistKvCacheMetrics::markWriteCudaCopyDoneUs(metrics);

    if (!open(true)) {
        DistKvCacheMetrics::markTotalWriteDoneUs(metrics);
        RTP_LLM_LOG_WARNING("write failed, open file failed, file: %s", filepath_.c_str());
        releaseIovIor(handle);
        return false;
    }

    if (!doWrite(handle, iovs, write_len, metrics)) {
        RTP_LLM_LOG_WARNING("write failed, do write failed, file: %s", filepath_.c_str());
        releaseIovIor(handle);
        del();
        DistKvCacheMetrics::markTotalWriteDoneUs(metrics);
        return false;
    }

    DistKvCacheMetrics::markTotalWriteDoneUs(metrics);
    DistKvCacheMetrics::setTotalWriteLen(metrics, write_len);
    if (metrics) {
        DistKvCacheMetrics::setTotalWriteThroughput(metrics, calcMiBs(write_len, metrics->TotalWriteCostUs()));
        RTP_LLM_LOG_DEBUG("write to 3fs, file: %s, iov num: %zu, len: %zu, cost: %ld us",
                          filepath_.c_str(),
                          iovs.size(),
                          write_len,
                          metrics->TotalWriteCostUs());
    }

    // 由于是异步写, 所以不能在此处 release iov/ior
    return true;
}

bool DistStorage3FSFile::doWrite(const std::shared_ptr<ThreeFSHandle>& handle,
                                 const std::vector<DistStorage::Iov>&  iovs,
                                 size_t                                write_total_size,
                                 std::shared_ptr<DistKvCacheMetrics>   metrics) {
    auto& ior      = handle->ior_handle.ior;
    auto& iov      = handle->iov_handle.iov;
    auto  iov_base = handle->iov_handle.iov_base;

    const auto iov_block_size  = handle->iov_handle.iov_block_size;
    const auto ior_entries     = handle->ior_handle.ior_entries;
    int64_t    remaining_size  = write_total_size;
    int64_t    iov_offset      = 0;
    int        submit_io_count = 0;

    DistKvCacheMetrics::markWriteBlockBeginUs(metrics);

    while (remaining_size > 0) {
        uint64_t cur_write_len = 0;
        if (iov_block_size > 0) {
            cur_write_len = std::min(remaining_size, calcLeftSizeInBlock(iov_block_size, iov_offset));
        } else {
            cur_write_len = remaining_size > kDefaultWriteSizePerIo ? kDefaultWriteSizePerIo : remaining_size;
        }

        auto ret = hf3fs_prep_io(ior, iov, false, iov_base + iov_offset, fd_, iov_offset, cur_write_len, nullptr);
        if (ret < 0) {
            DistKvCacheMetrics::markWriteBlockDoneUs(metrics);
            // TODO(LXQ): 需要处理掉之前已提交的io请求(submit然后wait)
            RTP_LLM_LOG_WARNING(
                "write to 3fs failed, hf3fs_prep_io failed, errno: %s, file: %s, submit_io_count: %d, ior_entries: %d, cur_write_len: %lu, iov_block_size: %lu, remaining_size: %ld, total size: %ld",
                strerror(-ret),
                filepath_.c_str(),
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
            DistKvCacheMetrics::markWriteBlockDoneUs(metrics);
            RTP_LLM_LOG_WARNING("write to 3fs failed, hf3fs_submit_ios failed, errno: %s, ior_entries: %d",
                                strerror(-ret),
                                ior_entries);
            return false;
        }

        bool async_wait_io = false;
        bool last_io       = remaining_size <= 0;
        if (config_.write_thread_pool) {
            // async wait io
            auto task =
                [shared_this = shared_from_this(), handle, submit_io_count, write_total_size, last_io, metrics]() {
                    shared_this->waitForWriteIos(handle, submit_io_count, write_total_size, last_io, metrics);
                };
            if (auto error_code = config_.write_thread_pool->pushTask(task, false);
                error_code != autil::ThreadPool::ERROR_NONE) {
                RTP_LLM_LOG_WARNING("write to 3fs failed, push work item failed, error code: %d, file: %s",
                                    error_code,
                                    filepath_.c_str());
                async_wait_io = false;
                // return false;
            } else {
                async_wait_io = true;
            }
            DistKvCacheMetrics::setWriteThreadPoolWorkItemCount(metrics, config_.write_thread_pool->getItemCount());
        }
        if (!async_wait_io) {
            // sync wait io
            if (!waitForWriteIos(handle, submit_io_count, write_total_size, last_io, metrics)) {
                DistKvCacheMetrics::markWriteBlockDoneUs(metrics);
                RTP_LLM_LOG_WARNING(
                    "write to 3fs failed, wait for write ios failed, file: %s, iov_size: %zu, iov_offset: %ld, cur_write_len: %lu, iov_block_size: %lu, ior_entries: %d",
                    filepath_.c_str(),
                    handle->iov_handle.iov_size,
                    iov_offset,
                    cur_write_len,
                    iov_block_size,
                    ior_entries);
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
                                         bool                                  last_io,
                                         std::shared_ptr<DistKvCacheMetrics>   metrics) {
    auto release_func = [shared_this = shared_from_this(), last_io, handle, write_total_len, &metrics]() {
        if (last_io) {
            if (shared_this->fd_ != -1) {
                if (::fsync(shared_this->fd_) != 0) {
                    RTP_LLM_LOG_WARNING(
                        "fsync failed, errno: %s, file: %s", strerror(errno), shared_this->filepath_.c_str());
                    shared_this->del();
                }
            }
            DistKvCacheMetrics::markWriteBlockDoneUs(metrics);
            if (metrics) {
                DistKvCacheMetrics::setWriteBlockThroughput(metrics,
                                                            calcMiBs(write_total_len, metrics->WriteBlockCostUs()));
            }
            shared_this->releaseIovIor(handle);
        }
    };
    autil::ScopeGuard guard(release_func);

    if (handle == nullptr || handle->ior_handle.ior == nullptr) {
        return false;
    }

    hf3fs_cqe cqes[submit_io_count];
    auto      time_spec          = createTimeoutTimeSpec(static_cast<int>(write_timeout_ms_));
    auto      ior                = handle->ior_handle.ior;
    int       completed_io_count = hf3fs_wait_for_ios(ior, cqes, submit_io_count, submit_io_count, &time_spec);
    if (completed_io_count < 0) {
        RTP_LLM_LOG_WARNING(
            "wait write io failed, hf3fs_wait_for_ios failed, errno: %s, file: %s, submit io count: %d, ior_entries: %d, write_total_len: %ld, iov_size: %zu",
            strerror(-completed_io_count),
            filepath_.c_str(),
            submit_io_count,
            handle->ior_handle.ior_entries,
            write_total_len,
            handle->iov_handle.iov_size);
        return false;
    } else if (completed_io_count == 0) {
        RTP_LLM_LOG_WARNING(
            "wait write io but hf3fs_wait_for_ios return 0, maybe timeout(%zu ms), file: %s, submit io count: %d, ior_entries: %d, write_total_len: %ld, iov_size: %zu",
            write_timeout_ms_,
            filepath_.c_str(),
            submit_io_count,
            handle->ior_handle.ior_entries,
            write_total_len,
            handle->iov_handle.iov_size);
    }

    for (int i = 0; i < completed_io_count; ++i) {
        if (cqes[i].result < 0) {
            RTP_LLM_LOG_WARNING(
                "wait write io failed, cqe result errno: %s, file: %s, submit_io_count: %d, completed_io_count: %d, ior_entries: %d, write_total_len: %ld, iov_size: %zu",
                strerror(-cqes[i].result),
                filepath_.c_str(),
                submit_io_count,
                completed_io_count,
                handle->ior_handle.ior_entries,
                write_total_len,
                handle->iov_handle.iov_size);
            return false;
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
    size_t  read_len       = 0;
    int64_t offset_to_read = -1;
    for (const auto& iov : iovs) {
        if (iov.ignore) {
            continue;
        }
        if (offset_to_read == -1) {
            offset_to_read = read_len;
        }
        read_len += iov.len;
        if (iov.data == nullptr) {
            RTP_LLM_LOG_WARNING("read failed, iov data is nullptr");
            return false;
        }
    }
    if (iovs.empty() || read_len == 0 || offset_to_read == -1) {
        RTP_LLM_LOG_DEBUG("read but iovs are invalid, file: %s, size: %zu, len: %zu, offset: %ld",
                          filepath_.c_str(),
                          iovs.size(),
                          read_len,
                          offset_to_read);
        return true;
    }
    if (const auto file_len = getFileLength(); offset_to_read + static_cast<int64_t>(read_len) > file_len) {
        RTP_LLM_LOG_WARNING(
            "read failed, read len exceed file len, file: %s, offset: %ld, read len: %zu, file len: %ld",
            filepath_.c_str(),
            offset_to_read,
            read_len,
            file_len);
        return false;
    }

    if (!open()) {
        return false;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(config_.metrics_reporter);
    DistKvCacheMetrics::markTotalReadBeginUs(metrics);

    auto handle = initIovIor(read_iov_handle_, read_len, kDefaultReadSizePerIo, true);
    if (handle == nullptr) {
        DistKvCacheMetrics::markTotalReadDoneUs(metrics);
        RTP_LLM_LOG_WARNING("read failed, init iov/ior failed, file: %s", filepath_.c_str());
        return false;
    }

    if (!doRead(handle, offset_to_read, read_len, metrics)) {
        RTP_LLM_LOG_WARNING("read failed, do read failed, file: %s", filepath_.c_str());
        releaseIovIor(handle);
        DistKvCacheMetrics::markTotalReadDoneUs(metrics);
        return false;
    }

    DistKvCacheMetrics::markReadCudaCopyBeginUs(metrics);
    auto    cuda_util  = handle->iov_handle.cuda_util;
    auto    iov_base   = handle->iov_handle.iov_base;
    int64_t iov_offset = 0;
    for (auto& iov : iovs) {
        if (iov.ignore) {
            continue;
        }
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
    DistKvCacheMetrics::markReadCudaCopyDoneUs(metrics);

    releaseIovIor(handle);

    DistKvCacheMetrics::markTotalReadDoneUs(metrics);
    DistKvCacheMetrics::setTotalReadLen(metrics, read_len);
    if (metrics) {
        DistKvCacheMetrics::setTotalReadThroughput(metrics, calcMiBs(read_len, metrics->TotalReadCostUs()));
        RTP_LLM_LOG_DEBUG("read from 3fs, file: %s, iov num: %zu, len: %zu, cost: %ld us",
                          filepath_.c_str(),
                          iovs.size(),
                          read_len,
                          metrics->TotalReadCostUs());
    }

    return true;
}

bool DistStorage3FSFile::doRead(const std::shared_ptr<ThreeFSHandle>& handle,
                                int64_t                               file_offset,
                                size_t                                read_total_len,
                                std::shared_ptr<DistKvCacheMetrics>   metrics) const {
    auto& ior      = handle->ior_handle.ior;
    auto& iov      = handle->iov_handle.iov;
    auto  iov_base = handle->iov_handle.iov_base;

    const auto ior_entries     = handle->ior_handle.ior_entries;
    const auto iov_block_size  = handle->iov_handle.iov_block_size;
    int64_t    iov_offset      = 0;
    int32_t    submit_io_count = 0;
    int64_t    remaining_size  = read_total_len;

    DistKvCacheMetrics::markReadBlockBeginUs(metrics);

    // 目前的实现是一次读完, 可以改成边读边Copy到显存
    while (remaining_size > 0) {
        uint64_t cur_read_len = 0;
        if (iov_block_size > 0) {
            cur_read_len = std::min(remaining_size, calcLeftSizeInBlock(iov_block_size, iov_offset));
        } else {
            cur_read_len = remaining_size > kDefaultReadSizePerIo ? kDefaultReadSizePerIo : remaining_size;
        }

        auto ret = hf3fs_prep_io(ior, iov, true, iov_base + iov_offset, fd_, file_offset, cur_read_len, nullptr);
        if (ret < 0) {
            DistKvCacheMetrics::markReadBlockDoneUs(metrics);
            RTP_LLM_LOG_WARNING(
                "read failed, hf3fs_prep_io failed, errno: %s, file: %s, fd: %d, submit_io_count: %d, ior_entries: %d, cur_read_len: %lu, iov_block_size: %lu, remaining_size: %ld, total size: %ld",
                strerror(-ret),
                filepath_.c_str(),
                fd_,
                submit_io_count,
                ior_entries,
                cur_read_len,
                iov_block_size,
                remaining_size,
                read_total_len);
            return false;
        }
        ++submit_io_count;
        iov_offset += cur_read_len;  // 这里iov_offset 一直和file_offset一致.
        file_offset += cur_read_len;
        remaining_size -= cur_read_len;

        if (submit_io_count < ior_entries && remaining_size > 0) {
            continue;
        }

        // submit_io_count 达到最大或者没得读
        if (!submitAndWaitForReadIos(handle, submit_io_count)) {
            DistKvCacheMetrics::markReadBlockDoneUs(metrics);
            RTP_LLM_LOG_WARNING(
                "read failed, read submit/wait io failed. file: %s, iov_size: %zu, iov_offset: %ld, file_offset: %ld, cur_read_len: %lu, iov_block_size: %lu, submit_io_count: %d, ior_entries: %d",
                filepath_.c_str(),
                handle->iov_handle.iov_size,
                iov_offset,
                file_offset,
                cur_read_len,
                iov_block_size,
                submit_io_count,
                ior_entries);
            return false;
        }
        submit_io_count = 0;  // reset submit io count
    }

    DistKvCacheMetrics::markReadBlockDoneUs(metrics);
    DistKvCacheMetrics::setReadBlockLen(metrics, read_total_len);
    if (metrics) {
        DistKvCacheMetrics::setReadBlockThroughput(metrics, calcMiBs(read_total_len, metrics->ReadBlockCostUs()));
    }

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
    auto      time_spec          = createTimeoutTimeSpec(static_cast<int>(read_timeout_ms_));
    auto      completed_io_count = hf3fs_wait_for_ios(ior, cqes, submit_io_count, submit_io_count, &time_spec);
    if (completed_io_count < 0) {
        RTP_LLM_LOG_WARNING("wait read io failed, hf3fs_wait_for_ios failed, errno: %s, submit io count: %d",
                            strerror(-completed_io_count),
                            submit_io_count);
        return false;
    } else if (completed_io_count == 0) {
        RTP_LLM_LOG_WARNING("wait read io but hf3fs_wait_for_ios return 0, maybe timeout(%zu ms), submit io count: %d",
                            read_timeout_ms_,
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
    close();
    // TODO: should check result?
    ::remove(filepath_.c_str());
    RTP_LLM_LOG_DEBUG("remove 3fs file: %s", filepath_.c_str());
    return true;
}

bool DistStorage3FSFile::close() {
    if (fd_ != -1) {
        hf3fs_dereg_fd(fd_);
        ::close(fd_);
        fd_ = -1;
    }
    return true;
}

std::shared_ptr<ThreeFSHandle>
DistStorage3FSFile::initIovIor(ThreeFSIovHandle& iov_handle, int64_t file_len, int32_t size_per_io, bool for_read) {
    auto iov_buffer = iov_handle.iov_mempool->alloc(file_len);
    if (iov_buffer == nullptr) {
        RTP_LLM_LOG_WARNING(
            "init iov/ior failed, mempool alloc failed, read: %d, alloc len: %zu, mempool free size: %zu, file: %s",
            for_read,
            file_len,
            iov_handle.iov_mempool->freeSize(),
            filepath_.c_str());
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
            "init iov/ior failed, create ior failed, file: %s, ior entries: %d, ior io depth: %d, ior timeout ms: %d",
            filepath_.c_str(),
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

void DistStorage3FSFile::releaseIovIor(const std::shared_ptr<ThreeFSHandle>& handle) const {
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

std::optional<int64_t> DistStorage3FSFile::getFileLength() const {
    struct stat file_stat;
    if (auto ret = ::stat(filepath_.c_str(), &file_stat); ret != 0) {
        RTP_LLM_LOG_WARNING("get file length failed, stat failed, file: %s, ret: %d, errno: %s",
                            filepath_.c_str(),
                            ret,
                            strerror(errno));
        return std::nullopt;
    }
    return static_cast<int64_t>(file_stat.st_size);  // byte
}

void DistStorage3FSFile::verify(const std::vector<DistStorage::Iov>&    write_iovs,
                                const std::shared_ptr<ThreeFSCudaUtil>& cuda_util) {
    // wait for 3fs write finished
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::vector<DistStorage::Iov> read_iovs = write_iovs;
    for (auto& iov : read_iovs) {
        auto buf    = malloc(iov.len);
        iov.data    = std::shared_ptr<void>(buf, [](void* p) { free(p); });
        iov.gpu_mem = false;
    }

    if (!read(read_iovs)) {
        RTP_LLM_LOG_WARNING("verify failed, read failed after write");
        return;
    }

    if (write_iovs.size() != read_iovs.size()) {
        RTP_LLM_LOG_WARNING(
            "verify failed, write iovs size not equal to read iovs size, write iovs size: %zu, read iovs size: %zu",
            write_iovs.size(),
            read_iovs.size());
        return;
    }

    bool success = true;
    for (int i = 0; i < write_iovs.size(); ++i) {
        const auto write_iov = write_iovs[i];
        const auto read_iov  = read_iovs[i];
        if (write_iov.len != read_iov.len) {
            RTP_LLM_LOG_WARNING(
                "[%d]verify failed, write iov len not equal to read iov len, write iov len: %lu, read iov len: %lu",
                i,
                write_iov.len,
                read_iov.len);
            success = false;
            break;
        }
        auto write_iov_buf = write_iov.data;
        if (write_iov.gpu_mem) {
            write_iov_buf = std::shared_ptr<void>(malloc(write_iov.len), [](void* p) { free(p); });
            cuda_util->copyAsyncDeviceToHost(write_iov_buf.get(), write_iov.data.get(), write_iov.len);
        }
        if (std::memcmp(write_iov_buf.get(), read_iov.data.get(), write_iov.len) != 0) {
            RTP_LLM_LOG_WARNING("[%d]verify failed, write iov not equal to read iov", i);
            success = false;
            break;
        }
    }

    if (success) {
        RTP_LLM_LOG_INFO("verify success, iovs size: %zu", write_iovs.size());
    } else {
        RTP_LLM_LOG_WARNING("verify failed, iovs size: %zu", write_iovs.size());
    }
}

}  // namespace threefs

}  // namespace rtp_llm