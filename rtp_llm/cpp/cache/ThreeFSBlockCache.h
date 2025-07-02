#pragma once

#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hf3fs_usrbio.h"
#include "kmonitor/client/MetricsReporter.h"

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/ThreeFSFile.h"
#include "rtp_llm/cpp/cache/ThreeFSMempool.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace autil {
class LockFreeThreadPool;
}

namespace rtp_llm {
class CacheManager;

namespace threefs {
class ThreeFSCudaUtil;
class ThreeFSMempool;

class ThreeFSBlockCache final {
public:
    ThreeFSBlockCache(const rtp_llm::BufferPtr&           k_cache,
                      const rtp_llm::BufferPtr&           v_cache,
                      const CacheConfig&                  config,
                      const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);
    ~ThreeFSBlockCache();

public:
    bool init();
    bool matchCache(const std::string& kvcache_key, const std::vector<int64_t>& cache_keys);
    bool getCache(const std::string&          kvcache_key,
                  const std::vector<int64_t>& cache_keys,
                  const std::vector<int32_t>& block_indices);
    bool putCache(const std::string&          kvcache_key,
                  const std::vector<int64_t>& cache_keys,
                  const std::vector<int32_t>& block_indices);
    void removeCache(const std::string& kvcache_key);

    std::pair<int64_t, int64_t> getFileSizeAndAge(const std::string& filename) const;

private:
    bool                     initIovHandle(bool for_read, const std::shared_ptr<ThreeFSCudaUtil>& cuda_util);
    struct hf3fs_iov*        createIov(const std::string& mountpoint, size_t iov_size, size_t iov_block_size) const;
    void                     removeOldIov() const;
    void                     releaseIovs();
    void                     releaseIov(ThreeFSIovHandle& iov_handle) const;
    std::pair<void*, size_t> pageAlign(void* ptr, size_t size, int64_t page_size = 4096) const;

    bool                         fileExists(const std::string& filename) const;
    std::shared_ptr<ThreeFSFile> get3FSFile(const std::string& filename, int cache_key_count, bool for_match);
    std::shared_ptr<ThreeFSFile> create3FSFile(const std::string& filename) const;
    std::string                  getFilePath(const std::string& filename) const;
    bool                         needWrite(const std::string& filename) const;
    bool                         checkFile(const std::string& filename) const;

    std::tuple<size_t, size_t, size_t> getFileSystemInfo() const;
    void                               reportMetrics() const;

private:
    rtp_llm::BufferPtr           k_cache_;
    rtp_llm::BufferPtr           v_cache_;
    CacheConfig                  cache_config_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    std::atomic<bool>            stop_report_metrics_{false};
    std::thread                  report_metrics_thread_;

    std::string      mountpoint_{"/3fs/stage/3fs/"};
    std::string      folder_name_{"rtp_llm/"};
    ThreeFSIovHandle read_iov_handle_;
    ThreeFSIovHandle write_iov_handle_;

    const size_t kDefaultReadIovSize{1ULL << 32};        // 4GB
    const size_t kDefaultReadIovBlockSize{0};            // 0
    const size_t kDefaultWriteIovSize{1ULL << 32};       // 4GB
    const size_t kDefaultWriteIovBlockSize{1ULL << 20};  // 1MB

    // {filename: 3fs file}
    std::unordered_map<std::string, std::shared_ptr<ThreeFSFile>> file_map_;
    std::shared_mutex                                             file_map_mutex_;

    // for 3fs async write
    const size_t                               thread_num_{4};
    const size_t                               queue_size_{1000};
    const std::string                          thread_name_{"3FSWriteThread"};
    std::shared_ptr<autil::LockFreeThreadPool> write_thread_pool_;
};

}  // namespace threefs
}  // namespace rtp_llm