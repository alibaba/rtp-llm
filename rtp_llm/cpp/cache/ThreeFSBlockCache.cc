#include "rtp_llm/cpp/cache/ThreeFSBlockCache.h"

#include <filesystem>
#include <sys/stat.h>
#include <sys/statvfs.h>

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/ThreeFSCudaUtil.h"
#include "rtp_llm/cpp/cache/ThreeFSMempool.h"
#include "rtp_llm/cpp/cache/ThreeFSMetrics.h"

namespace rtp_llm::threefs {

ThreeFSBlockCache::ThreeFSBlockCache(const rtp_llm::BufferPtr&           k_cache,
                                     const rtp_llm::BufferPtr&           v_cache,
                                     const CacheConfig&                  config,
                                     const kmonitor::MetricsReporterPtr& metrics_reporter):
    k_cache_(k_cache), v_cache_(v_cache), cache_config_(config), metrics_reporter_(metrics_reporter) {
    if (auto enable_async_write = autil::EnvUtil::getEnv("THREEFS_ENABLE_ASYNC_WRITE", true); enable_async_write) {
        RTP_LLM_LOG_INFO("3fs enable async write, thread num: %lu, queue size: %lu", thread_num_, queue_size_);
        write_thread_pool_ =
            std::make_shared<autil::LockFreeThreadPool>(thread_num_, queue_size_, nullptr, thread_name_);
        write_thread_pool_->start();
    }
    // mountpoint 和 folder 一般不会被修改, 这里环境变量控制只是为了方便测试
    if (auto mountpoint = autil::EnvUtil::getEnv("THREEFS_MOUNTPOINT", std::string("")); !mountpoint.empty()) {
        if (mountpoint.back() != '/') {
            mountpoint.push_back('/');
        }
        mountpoint_ = mountpoint;
    };
    if (auto folder_name = autil::EnvUtil::getEnv("THREEFS_FOLDER_NAME", std::string("")); !folder_name.empty()) {
        if (folder_name.back() != '/') {
            folder_name.push_back('/');
        }
        folder_name_ = folder_name;
    };
    RTP_LLM_LOG_INFO("3fs mountpoint: %s, folder name: %s", mountpoint_.c_str(), folder_name_.c_str());
}

ThreeFSBlockCache::~ThreeFSBlockCache() {
    RTP_LLM_LOG_INFO("3fs block cache destructor");
    {
        std::unique_lock<std::shared_mutex> lock(file_map_mutex_);
        file_map_.clear();
    }
    if (write_thread_pool_) {
        write_thread_pool_->stop();
        write_thread_pool_->waitFinish();
        write_thread_pool_.reset();
    }

    stop_report_metrics_.store(true);
    if (report_metrics_thread_.joinable()) {
        report_metrics_thread_.join();
    }

    releaseIovs();
}

bool ThreeFSBlockCache::matchCache(const std::string& kvcache_key, const std::vector<int64_t>& cache_keys) {
    if (cache_keys.empty()) {
        RTP_LLM_LOG_WARNING("match failed, cache key list is empty");
        return false;
    }

    if (!checkFile(kvcache_key)) {
        RTP_LLM_LOG_DEBUG("match failed, check file failed, kvcache key: %s", kvcache_key.c_str());
        {
            std::unique_lock<std::shared_mutex> lock(file_map_mutex_);
            file_map_.erase(kvcache_key);
        }
        return false;
    }

    const auto cache_key_count = static_cast<int>(cache_keys.size());
    auto       threefs_file    = get3FSFile(kvcache_key, cache_key_count, true);
    if (!threefs_file) {
        RTP_LLM_LOG_WARNING("match failed, 3fs file is nullptr, kvcache key: %s", kvcache_key.c_str());
        return false;
    }

    if (!threefs_file->match(cache_keys)) {
        // 文件存在, 但是 match 失败, 有一种可能是文件存在但被删除了, 之后又重建了一个同名文件, 此时要重新打开文件
        std::unique_lock<std::shared_mutex> lock(file_map_mutex_);
        file_map_.erase(kvcache_key);
        return false;
    }

    return true;
}

bool ThreeFSBlockCache::getCache(const std::string&          kvcache_key,
                                 const std::vector<int64_t>& cache_keys,
                                 const std::vector<int32_t>& block_indices) {
    if (cache_keys.empty() && block_indices.empty()) {
        return true;
    }
    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_WARNING(
            "get kvcache failed, cache key size not equal to block index size, cache key size: %lu, block index size: %lu",
            cache_keys.size(),
            block_indices.size());
        return false;
    }

    const auto cache_key_count = static_cast<int>(cache_keys.size());
    auto       threefs_file    = get3FSFile(kvcache_key, cache_key_count, false);
    if (!threefs_file) {
        RTP_LLM_LOG_WARNING("get kvcache failed, 3fs file is nullptr, kvcache key: %s", kvcache_key.c_str());
        return false;
    }

    return threefs_file->read(cache_keys, block_indices);
}

bool ThreeFSBlockCache::putCache(const std::string&          kvcache_key,
                                 const std::vector<int64_t>& cache_keys,
                                 const std::vector<int32_t>& block_indices) {
    if (cache_keys.empty() && block_indices.empty()) {
        return true;
    }
    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_WARNING(
            "put kvcache failed, cache key size not equal to block index size, cache key size: %lu, block index size: %lu",
            cache_keys.size(),
            block_indices.size());
        return false;
    }

    auto threefs_file = create3FSFile(kvcache_key);
    if (!threefs_file) {
        RTP_LLM_LOG_WARNING("put kvcache failed, create 3fs file failed, kvcache key: %s", kvcache_key.c_str());
        return false;
    }

    if (!threefs_file->write(cache_keys, block_indices)) {
        RTP_LLM_LOG_WARNING("put kvcache failed, 3fs file write failed, kvcache key: %s", kvcache_key.c_str());
        return false;
    }
    return true;
}

void ThreeFSBlockCache::removeCache(const std::string& kvcache_key) {
    if (kvcache_key.empty()) {
        return;
    }

    {
        std::unique_lock<std::shared_mutex> lock(file_map_mutex_);
        file_map_.erase(kvcache_key);
    }

    const std::string filepath = getFilePath(kvcache_key);
    RTP_LLM_LOG_DEBUG("remove 3fs file: %s", filepath.c_str());
    std::remove(filepath.c_str());
}

std::shared_ptr<ThreeFSFile>
ThreeFSBlockCache::get3FSFile(const std::string& filename, int cache_key_count, bool for_match) {
    std::shared_ptr<ThreeFSFile> threefs_file;
    {
        std::unique_lock<std::shared_mutex> lock(file_map_mutex_);
        if (file_map_.count(filename) != 0) {
            threefs_file = file_map_.at(filename);
        }
        if (threefs_file == nullptr) {
            threefs_file        = create3FSFile(filename);
            file_map_[filename] = threefs_file;
        }
    }
    return threefs_file;
}

bool ThreeFSBlockCache::init() {
    if (!std::filesystem::exists(mountpoint_)) {
        RTP_LLM_LOG_WARNING("init failed, 3fs mountpoint not exists: %s", mountpoint_.c_str());
        return false;
    }

    auto cuda_util = std::make_shared<ThreeFSCudaUtil>();
    if (!cuda_util->init()) {
        RTP_LLM_LOG_WARNING("init failed, cuda util init failed");
        return false;
    }

    removeOldIov();

    bool for_read = true;
    if (!initIovHandle(for_read, cuda_util)) {
        RTP_LLM_LOG_WARNING("init read iov mempool failed");
        return false;
    }

    for_read = false;
    if (!initIovHandle(for_read, cuda_util)) {
        RTP_LLM_LOG_WARNING("init write iov mempool failed");
        releaseIovs();
        return false;
    }
    RTP_LLM_LOG_INFO("read iov size: %zu, read iov block size: %zu; write iov size: %zu, write iov block size: %zu",
                     read_iov_handle_.iov_size,
                     read_iov_handle_.iov_block_size,
                     write_iov_handle_.iov_size,
                     write_iov_handle_.iov_block_size);

    if (metrics_reporter_) {
        stop_report_metrics_.store(false);
        report_metrics_thread_ = std::thread(&ThreeFSBlockCache::reportMetrics, this);
    } else {
        stop_report_metrics_.store(true);
    }
    return true;
}

bool ThreeFSBlockCache::initIovHandle(bool for_read, const std::shared_ptr<ThreeFSCudaUtil>& cuda_util) {
    if (mountpoint_.empty()) {
        RTP_LLM_LOG_WARNING("init iov mempool failed, mountpoint is empty");
        return false;
    }

    // iov block size
    size_t iov_block_size = 0;
    if (for_read) {
        iov_block_size = kDefaultReadIovBlockSize;
        if (auto read_iov_block_size_env = autil::EnvUtil::getEnv("THREEFS_READ_IOV_BLOCK_SIZE", -1LL);
            read_iov_block_size_env != -1) {
            iov_block_size = read_iov_block_size_env;
        }
    } else {
        iov_block_size = kDefaultWriteIovBlockSize;
        if (auto write_iov_block_size_env = autil::EnvUtil::getEnv("THREEFS_WRITE_IOV_BLOCK_SIZE", -1LL);
            write_iov_block_size_env != -1) {
            iov_block_size = write_iov_block_size_env;
        }
    }

    // iov size
    size_t iov_size = 0;
    if (for_read) {
        iov_size = kDefaultReadIovSize;
        if (auto read_iov_size_env = autil::EnvUtil::getEnv("THREEFS_READ_IOV_SIZE", -1LL); read_iov_size_env != -1) {
            iov_size = read_iov_size_env;
        }
    } else {
        iov_size = kDefaultWriteIovSize;
        if (auto write_iov_size_env = autil::EnvUtil::getEnv("THREEFS_WRITE_IOV_SIZE", -1LL);
            write_iov_size_env != -1) {
            iov_size = write_iov_size_env;
        }
    }
    if (iov_block_size != 0 && iov_size % iov_block_size != 0) {
        iov_size = (iov_size / iov_block_size + 1) * iov_block_size;
    }

    auto iov = createIov(mountpoint_, iov_size, iov_block_size);
    if (iov == nullptr) {
        RTP_LLM_LOG_WARNING(
            "create iov failed, read: %d, iov size: %zu, iov block size: %zu", for_read, iov_size, iov_block_size);
        return false;
    }

    auto mempool = std::make_shared<ThreeFSMempool>(iov->base, iov_size, iov_block_size);
    if (!mempool->init()) {
        RTP_LLM_LOG_WARNING("mempool init failed, read: %d, iov base: %p, iov size: %zu, iov block size: %zu",
                            for_read,
                            iov->base,
                            iov_size,
                            iov_block_size);
        releaseIovs();
        return false;
    }

    if (cuda_util && !cuda_util->registerHost(iov->base, iov_size)) {
        RTP_LLM_LOG_WARNING("cuda register iov failed, iov base: %p, iov size: %zu", iov->base, iov_size);
    }

    if (for_read) {
        read_iov_handle_ = {iov, nullptr, iov_size, iov_block_size, mempool, cuda_util};
    } else {
        write_iov_handle_ = {iov, nullptr, iov_size, iov_block_size, mempool, cuda_util};
    }
    return true;
}

struct hf3fs_iov*
ThreeFSBlockCache::createIov(const std::string& mountpoint, size_t iov_size, size_t iov_block_size) const {
    if (mountpoint.empty()) {
        RTP_LLM_LOG_WARNING("create iov failed, mountpoint is empty");
        return nullptr;
    }
    if (iov_size <= 0) {
        RTP_LLM_LOG_WARNING("create iov failed, iov size is invalid: %zu", iov_size);
        return nullptr;
    }

    auto iov = new struct hf3fs_iov();
    auto ret = hf3fs_iovcreate(iov, mountpoint.c_str(), iov_size, iov_block_size, -1);
    if (ret != 0) {
        RTP_LLM_LOG_WARNING("hf3fs_iovcreate failed, errno: %s, mountpoint: %s, iov size: %lu, iov block size: %lu",
                            strerror(-ret),
                            mountpoint.c_str(),
                            iov_size,
                            iov_block_size);
        hf3fs_iovdestroy(iov);
        delete iov;
        iov = nullptr;
        return nullptr;
    }
    return iov;
}

void ThreeFSBlockCache::removeOldIov() const {
    // 删除旧的 shm iov , 避免 shm 空间不够用
    namespace fs = std::filesystem;

    const std::string shm_path = "/dev/shm/";
    const std::string prefix   = "hf3fs-iov-";

    try {
        fs::path dir(shm_path);
        if (!fs::exists(dir)) {
            return;
        }
        if (!fs::is_directory(dir)) {
            return;
        }

        const auto threshold = std::chrono::seconds(300);  // 5min
        const auto now       = fs::file_time_type::clock::now();

        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!fs::is_regular_file(entry.status())) {
                continue;
            }

            std::string filename = entry.path().filename().string();
            if (filename.find(prefix) != 0) {
                continue;
            }

            const auto age = now - fs::last_write_time(entry.path());
            if (age > threshold) {
                try {
                    RTP_LLM_LOG_INFO("remove old shm iov file: %s", filename.c_str());
                    fs::remove(entry.path());
                } catch (const fs::filesystem_error& e) {
                    RTP_LLM_LOG_WARNING(
                        "found exception when remove old iov file: %s, exception: %s", filename.c_str(), e.what());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        RTP_LLM_LOG_WARNING("found exception when remove old iov file, exception: %s", e.what());
    }
}

void ThreeFSBlockCache::releaseIovs() {
    releaseIov(read_iov_handle_);
    releaseIov(write_iov_handle_);
}

void ThreeFSBlockCache::releaseIov(ThreeFSIovHandle& iov_handle) const {
    if (iov_handle.iov != nullptr) {
        if (iov_handle.cuda_util) {
            iov_handle.cuda_util->unregisterHost(iov_handle.iov->base);
        }
        hf3fs_iovdestroy(iov_handle.iov);
        delete iov_handle.iov;
        iov_handle.iov = nullptr;
    }
    iov_handle.iov_mempool.reset();
    iov_handle.cuda_util.reset();
}

std::pair<void*, size_t> ThreeFSBlockCache::pageAlign(void* ptr, size_t size, int64_t page_size) const {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr % page_size == 0) {
        return std::make_pair(ptr, size);
    }
    uintptr_t aligned_addr = (addr + page_size - 1) & ~(page_size - 1);
    auto      new_size     = size - (aligned_addr - addr);
    return std::make_pair(reinterpret_cast<void*>(aligned_addr), new_size);
}

std::shared_ptr<ThreeFSFile> ThreeFSBlockCache::create3FSFile(const std::string& filename) const {
    ThreeFSFileConfig config{
        mountpoint_, folder_name_, filename, k_cache_, v_cache_, cache_config_, write_thread_pool_, metrics_reporter_};
    auto threefs_file = std::make_shared<ThreeFSFile>(config, read_iov_handle_, write_iov_handle_);
    return threefs_file;
}

bool ThreeFSBlockCache::fileExists(const std::string& filename) const {
    const auto  filepath = getFilePath(filename);
    struct stat file_stat;
    return ::stat(filepath.c_str(), &file_stat) == 0;
}

std::string ThreeFSBlockCache::getFilePath(const std::string& filename) const {
    return mountpoint_ + folder_name_ + filename;
}

std::pair<int64_t, int64_t> ThreeFSBlockCache::getFileSizeAndAge(const std::string& filename) const {
    const auto  filepath = getFilePath(filename);
    struct stat file_stat;
    if (auto ret = ::stat(filepath.c_str(), &file_stat); ret != 0) {
        return std::make_pair(-1, -1);
    }

    const auto file_size     = static_cast<int64_t>(file_stat.st_size);
    time_t     last_mod_time = file_stat.st_mtime;
    time_t     current_time;
    time(&current_time);
    double     diff_secs = difftime(current_time, last_mod_time);
    const auto file_age  = static_cast<int64_t>(diff_secs);
    return std::make_pair(file_size, file_age);
}

bool ThreeFSBlockCache::needWrite(const std::string& filename) const {
    const auto  filepath = getFilePath(filename);
    struct stat file_stat;
    if (auto ret = ::stat(filepath.c_str(), &file_stat); ret != 0) {
        if (errno == ENOENT) {
            // file not exist
            return true;
        }
        RTP_LLM_LOG_WARNING("file exist but stat failed, overwrite file: %s, ret: %d, errno: %s",
                            filepath.c_str(),
                            ret,
                            strerror(errno));
        return true;
    }

    const auto file_size     = static_cast<int64_t>(file_stat.st_size);
    time_t     last_mod_time = file_stat.st_mtime;
    time_t     current_time;
    time(&current_time);
    double     diff_secs = difftime(current_time, last_mod_time);
    const auto file_age  = static_cast<int64_t>(diff_secs);

    // 最后修改时间距今超过 60s 且 文件长度为 0 的文件可能是之前写入失败了, 需要重新写入
    // 之所以判断文件修改时间, 是因为此时可能有其他人正在写相同 kvcache 的文件
    const int32_t threshold_sec = 60;
    if (file_age > threshold_sec && file_size <= 0) {
        RTP_LLM_LOG_DEBUG(
            "file len is 0 and create time is too long, overwrite file: %s, age: %ld", filename.c_str(), file_age);
        return true;
    }
    return false;
}

bool ThreeFSBlockCache::checkFile(const std::string& filename) const {
    const auto  filepath = getFilePath(filename);
    struct stat file_stat;
    if (auto ret = ::stat(filepath.c_str(), &file_stat); ret != 0) {
        if (errno == ENOENT) {
            // file not exist
            RTP_LLM_LOG_DEBUG("file not exist, file: %s", filename.c_str());
            return false;
        }
        RTP_LLM_LOG_WARNING(
            "file exist but stat failed, file: %s, ret: %d, errno: %s", filepath.c_str(), ret, strerror(errno));
        return false;
    }

    const auto file_size = static_cast<int64_t>(file_stat.st_size);
    if (file_size == 0) {
        RTP_LLM_LOG_DEBUG("file len is 0, file: %s", filename.c_str());
        return false;
    }
    return true;
}

std::tuple<size_t, size_t, size_t> ThreeFSBlockCache::getFileSystemInfo() const {
    struct statvfs stat;
    if (::statvfs(mountpoint_.c_str(), &stat) != 0) {
        RTP_LLM_LOG_WARNING("get 3fs file system info failed, statvfs failed, mountpoint: %s", mountpoint_.c_str());
        return {};
    }

    // stat.f_bsize: 文件系统块大小 (基本块大小，可能不是实际物理块大小)
    // stat.f_frsize: 片段大小 (文件系统分配的最小单位，通常等于 f_bsize)
    // stat.f_blocks: 文件系统中的总块数
    // stat.f_bfree: 文件系统中的空闲块数 (包括保留给root的)
    // stat.f_bavail: 文件系统中的非特权用户可用块数

    const size_t total_bytes     = stat.f_blocks * stat.f_frsize;
    const size_t free_bytes      = stat.f_bfree * stat.f_frsize;
    const size_t available_bytes = stat.f_bavail * stat.f_frsize;
    const size_t used_bytes      = total_bytes - free_bytes;
    return std::make_tuple(total_bytes, used_bytes, available_bytes);
}

void ThreeFSBlockCache::reportMetrics() const {
    while (!stop_report_metrics_.load()) {
        if (metrics_reporter_) {
            auto metrics          = ThreeFSMetricsFactory::createMetrics(metrics_reporter_);
            auto read_iov_mempool = read_iov_handle_.iov_mempool;
            if (read_iov_mempool) {
                ThreeFSMetrics::setReadIovAllocatedSize(metrics, read_iov_mempool->allocatedSize());
                ThreeFSMetrics::setReadIovAllocatedCount(metrics, read_iov_mempool->allocatedBlockCount());
                ThreeFSMetrics::setReadIovFreeSize(metrics, read_iov_mempool->freeSize());
            }

            auto write_iov_mempool = write_iov_handle_.iov_mempool;
            if (write_iov_mempool) {
                ThreeFSMetrics::setWriteIovAllocatedSize(metrics, write_iov_mempool->allocatedSize());
                ThreeFSMetrics::setWriteIovAllocatedCount(metrics, write_iov_mempool->allocatedBlockCount());
                ThreeFSMetrics::setWriteIovFreeSize(metrics, write_iov_mempool->freeSize());
            }

            const auto [_, used_bytes, available_bytes] = getFileSystemInfo();
            ThreeFSMetrics::setFSUsedSize(metrics, used_bytes);
            ThreeFSMetrics::setFSFreeSize(metrics, available_bytes);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace rtp_llm::threefs