#pragma once

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "autil/WorkItem.h"
#include "hf3fs_usrbio.h"
#include "kmonitor/client/MetricsReporter.h"

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace autil {
class LockFreeThreadPool;
}

namespace rtp_llm {
class CacheManager;

namespace threefs {
class ThreeFSCudaUtil;
class ThreeFSMempool;
class ThreeFSMetrics;

struct ThreeFSIovHandle {
    struct hf3fs_iov*                iov{nullptr};
    uint8_t*                         iov_base{nullptr};  // iov addr
    size_t                           iov_size{0};        // iov 的共享内存大小
    size_t                           iov_block_size{0};  // 每个共享内存块的大小, 0 表示单个大型共享内存块
    std::shared_ptr<ThreeFSMempool>  iov_mempool;
    std::shared_ptr<ThreeFSCudaUtil> cuda_util;
};

// TODO(LXQ): io depth 和 timeout 暂不设置
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
    std::string                                folder_name;
    std::string                                filename;
    rtp_llm::BufferPtr                         k_cache;
    rtp_llm::BufferPtr                         v_cache;
    CacheConfig                                cache_config;
    std::shared_ptr<autil::LockFreeThreadPool> write_thread_pool;
    kmonitor::MetricsReporterPtr               metrics_reporter;
};

// ThreeFSMeta 与 block 文件存储格式有关, 谨慎修改!
#pragma pack(push, 1)
struct ThreeFSMeta {
    int64_t cache_key{0};
    int64_t offset{0};
};
#pragma pack(pop)

class ThreeFSItem final {
public:
    struct SingleBlock {
        std::shared_ptr<void> data;
        size_t                len{0};
    };

public:
    int64_t                  cache_key{0};
    std::vector<SingleBlock> blocks;
};

class ThreeFSFile final: public std::enable_shared_from_this<ThreeFSFile> {
public:
    ThreeFSFile(const ThreeFSFileConfig& config,
                const ThreeFSIovHandle&  read_iov_handle,
                const ThreeFSIovHandle&  write_iov_handle);
    ~ThreeFSFile();

public:
    bool match(const std::vector<int64_t>& cache_keys);
    bool read(const std::vector<int64_t>& cache_keys, const std::vector<int32_t>& block_indices);
    bool write(const std::vector<int64_t>& cache_keys, const std::vector<int32_t>& block_indices);

private:
    bool doMatch(const std::shared_ptr<ThreeFSHandle>& handle, const std::vector<int64_t>& cache_keys);
    bool doRead(const std::shared_ptr<ThreeFSHandle>& handle,
                const std::vector<int64_t>&           cache_keys,
                const std::vector<int32_t>&           block_indices);
    bool doWrite(const std::shared_ptr<ThreeFSHandle>& handle,
                 const std::vector<int64_t>&           cache_keys,
                 const std::vector<int32_t>&           block_indices);
    std::shared_ptr<ThreeFSHandle> initIovIor(int cache_key_count, bool for_read, bool read_for_match = false);
    bool                           checkIovHandle(const ThreeFSIovHandle& iov_handle) const;
    bool createIor(struct hf3fs_ior*& ior, bool for_read, int ior_entries, int ior_io_depth, int ior_timeout_ms) const;
    void releaseIovIor(const std::shared_ptr<ThreeFSHandle>& handle);
    bool writeTo3FS(const std::shared_ptr<ThreeFSHandle>& handle,
                    const std::vector<ThreeFSItem>&       items,
                    std::shared_ptr<ThreeFSMetrics>       metrics = nullptr);
    bool waitForWriteIos(const std::shared_ptr<ThreeFSHandle>& handle,
                         int32_t                               submit_io_count,
                         int64_t                               write_total_len,
                         bool                                  last_io,
                         std::shared_ptr<ThreeFSMetrics>       metrics = nullptr) const;
    bool submitAndWaitForReadIos(const std::shared_ptr<ThreeFSHandle>& handle, int32_t submit_io_count) const;
    bool readMetas(const std::shared_ptr<ThreeFSHandle>& handle, std::shared_ptr<ThreeFSMetrics> metrics = nullptr);
    bool checkAllCacheKeyMatched(const std::vector<int64_t>& cache_keys) const;
    int64_t calcLeftSizeInBlock(int64_t iov_block_size, int64_t iov_offset) const;

    std::tuple<void*, void*> convertIndexToAddr(int block_index, int layer_id) const;
    int32_t                  getMetaTotalLength(int32_t block_count) const;

    bool                   fileExists() const;
    std::string            getFilepath() const;
    size_t                 calcFileLength(int cache_key_count) const;
    bool                   openFile(bool read = true);
    void                   removeFile();
    std::optional<int64_t> getFileLength() const;
    std::optional<int64_t> getFileCreateTimeInSecs() const;
    int                    getFdFromMap(const std::string& filename) const;
    void                   addFdToMap(const std::string& filename, int fd) const;
    void                   removeFdInMap(const std::string& filename) const;
    int32_t                getFdMaxNum() const;

    // for test
    bool verify(const std::shared_ptr<ThreeFSHandle>& handle,
                const std::vector<int64_t>&           cache_keys,
                const std::vector<int32_t>&           block_indices);

private:
    const ThreeFSFileConfig    config_;
    const ThreeFSIovHandle     read_iov_handle_;
    const ThreeFSIovHandle     write_iov_handle_;
    int32_t                    fd_{-1};         // file descriptor
    std::map<int64_t, int64_t> cache_key_map_;  // {cache_key: file offset}
    mutable std::shared_mutex  cache_key_map_mutex_;

    const int32_t kDefaultFileLengthForRead{1ULL << 30};  // 1GB
    const int32_t kDefaultReadSizePerIo{1ULL << 20};      // 1MB
    const int32_t kDefaultWriteSizePerIo{1ULL << 20};     // 1MB
    const int32_t kReservedLength{1024};                  // 1KB

    static std::map<std::string, int> filename_to_fd_map_;  // {filename: fd}
    static std::shared_mutex          filename_to_fd_map_mutex_;

    // for async write, access function waitForWriteIos
    friend class WaitIoWorkItem;

    // for test
    bool cache_meta_{true};
    bool enable_kvcache_verify_{false};
};

class WaitIoWorkItem final: public autil::WorkItem {
public:
    WaitIoWorkItem(const std::shared_ptr<ThreeFSFile>&   threefs_file,
                   const std::shared_ptr<ThreeFSHandle>& handle,
                   int                                   submit_io_count,
                   int                                   write_len,
                   bool                                  last_io,
                   std::shared_ptr<ThreeFSMetrics>       metrics = nullptr):
        threefs_file_(threefs_file),
        handle_(handle),
        submit_io_count_(submit_io_count),
        write_len_(write_len),
        last_io_(last_io),
        metrics_(metrics) {}
    ~WaitIoWorkItem() override {
        threefs_file_.reset();
    }

public:
    void process() override;

private:
    std::shared_ptr<ThreeFSFile>    threefs_file_;
    std::shared_ptr<ThreeFSHandle>  handle_;
    int                             submit_io_count_;
    int                             write_len_;
    bool                            last_io_;
    std::shared_ptr<ThreeFSMetrics> metrics_;
};

}  // namespace threefs
}  // namespace rtp_llm