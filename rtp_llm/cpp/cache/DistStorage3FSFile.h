#pragma once

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "hf3fs_usrbio.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/ThreeFSMempool.h"
#include "rtp_llm/cpp/cache/ThreeFSCudaUtil.h"
#include "rtp_llm/cpp/cache/DistStorage.h"
#include "autil/LockFreeThreadPool.h"

namespace rtp_llm {

namespace threefs {

class DistKvCacheMetrics;

struct ThreeFSIovHandle {
    struct hf3fs_iov*                iov{nullptr};
    uint8_t*                         iov_base{nullptr};  // iov addr
    size_t                           iov_size{0};        // iov 的共享内存大小
    size_t                           iov_block_size{0};  // 每个共享内存块的大小, 0 表示单个大型共享内存块
    std::shared_ptr<ThreeFSMempool>  iov_mempool;
    std::shared_ptr<ThreeFSCudaUtil> cuda_util;
};

struct ThreeFSIorHandle {
    struct hf3fs_ior* ior{nullptr};
    int               ior_entries{1024};  // 可以提交的最大读/写请求数, 表示 io 请求个数的上限
    int               ior_io_depth{0};    // ior 中的 io 深度, 表示每次提交的 io 请求数
    int               ior_timeout_ms{0};  // io 批处理的最大等待时间, 仅在 io_depth 为负数时生效
};

struct ThreeFSHandle {
    ThreeFSIorHandle ior_handle;
    ThreeFSIovHandle iov_handle;
};

struct ThreeFSFileConfig {
    std::string                                mountpoint;
    std::string                                filepath;
    std::shared_ptr<autil::LockFreeThreadPool> write_thread_pool;
    kmonitor::MetricsReporterPtr               metrics_reporter;
};

// single file in 3fs
class DistStorage3FSFile: public std::enable_shared_from_this<DistStorage3FSFile> {
public:
    DistStorage3FSFile(const ThreeFSFileConfig& config,
                       const ThreeFSIovHandle&  read_iov_handle,
                       const ThreeFSIovHandle&  write_iov_handle,
                       size_t                   read_timeout_ms  = 1000,
                       size_t                   write_timeout_ms = 2000);
    virtual ~DistStorage3FSFile();

public:
    // virtual for mock test
    virtual bool exists() const;
    virtual bool open(bool write = false);
    virtual bool write(const std::vector<DistStorage::Iov>& iovs);
    virtual bool read(const std::vector<DistStorage::Iov>& iovs);
    virtual bool del();
    virtual bool close();

private:
    bool doWrite(const std::shared_ptr<ThreeFSHandle>& handle,
                 const std::vector<DistStorage::Iov>&  iovs,
                 size_t                                write_total_size,
                 std::shared_ptr<DistKvCacheMetrics>   metrics);
    bool waitForWriteIos(const std::shared_ptr<ThreeFSHandle>& handle,
                         int32_t                               submit_io_count,
                         int64_t                               write_total_len,
                         bool                                  last_io,
                         std::shared_ptr<DistKvCacheMetrics>   metrics);

    bool doRead(const std::shared_ptr<ThreeFSHandle>& handle,
                int64_t                               file_offset,
                size_t                                read_total_len,
                std::shared_ptr<DistKvCacheMetrics>   metrics) const;
    bool submitAndWaitForReadIos(const std::shared_ptr<ThreeFSHandle>& handle, int32_t submit_io_count) const;

    std::shared_ptr<ThreeFSHandle>
         initIovIor(ThreeFSIovHandle& iov_handle, int64_t file_len, int32_t size_per_io, bool for_read);
    void releaseIovIor(const std::shared_ptr<ThreeFSHandle>& handle) const;
    bool createIor(struct hf3fs_ior*& ior, bool for_read, int ior_entries, int ior_io_depth, int ior_timeout_ms) const;

    int64_t calcLeftSizeInBlock(int64_t iov_block_size, int64_t iov_offset) const;

    std::optional<int64_t> getFileLength() const;
    void verify(const std::vector<DistStorage::Iov>& write_iovs, const std::shared_ptr<ThreeFSCudaUtil>& cuda_util);

private:
    const ThreeFSFileConfig config_;
    ThreeFSIovHandle        read_iov_handle_;
    ThreeFSIovHandle        write_iov_handle_;
    int32_t                 fd_{-1};  // file descriptor
    const std::string       filepath_;
    const size_t            read_timeout_ms_;
    const size_t            write_timeout_ms_;

    const int32_t kDefaultFileLengthForRead{1ULL << 30};  // 1GB
    const int32_t kDefaultReadSizePerIo{1ULL << 20};      // 1MB
    const int32_t kDefaultWriteSizePerIo{1ULL << 20};     // 1MB
};
}  // namespace threefs

}  // namespace rtp_llm