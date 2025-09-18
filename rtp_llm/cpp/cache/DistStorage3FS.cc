#include "rtp_llm/cpp/cache/DistStorage3FS.h"

#include <filesystem>
#include <sys/stat.h>
#include <sys/statvfs.h>

#include "rtp_llm/cpp/cache/ThreeFSCudaUtil.h"
#include "rtp_llm/cpp/cache/ThreeFSMempool.h"
#include "rtp_llm/cpp/cache/DistKvCacheMetrics.h"

namespace rtp_llm::threefs {

DistStorage3FS::DistStorage3FS(const kmonitor::MetricsReporterPtr& metrics_reporter):
    metrics_reporter_(metrics_reporter) {}

DistStorage3FS::~DistStorage3FS() {
    RTP_LLM_LOG_INFO("DistStorage3FS destructor");

    clearFileCache();

    if (write_thread_pool_) {
        write_thread_pool_->stop();
        write_thread_pool_->waitFinish();
        write_thread_pool_.reset();
    }

    stop_report_metrics_.store(true);
    if (report_metrics_thread_.joinable()) {
        report_metrics_thread_.join();
    }

    releaseIovHandle(read_iov_handle_);
    releaseIovHandle(write_iov_handle_);
}

bool DistStorage3FS::init(const DistStorage3FSInitParams& init_params) {
    RTP_LLM_LOG_INFO("3fs init params: [%s]", init_params.toString().c_str());
    if (!checkInitParams(init_params)) {
        RTP_LLM_LOG_WARNING("3fs init failed, check init params failed, params: [%s]", init_params.toString().c_str());
        return false;
    }
    init_params_ = init_params;

    file_cache_ =
        std::make_shared<LRUCache<std::string, std::shared_ptr<DistStorage3FSFile>>>(init_params_.file_cache_capacity);

    if (init_params_.enable_async_write) {
        write_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
            init_params_.write_thread_num, init_params_.write_queue_size, nullptr, "3FSWriteThread");
        if (!write_thread_pool_->start()) {
            RTP_LLM_LOG_WARNING("init failed, start async thread pool failed, thread num: %lu, queue size: %lu",
                                init_params_.write_thread_num,
                                init_params_.write_queue_size);
            return false;
        }
    }

    auto cuda_util = std::make_shared<ThreeFSCudaUtil>();
    if (!cuda_util->init()) {
        RTP_LLM_LOG_WARNING("dist storage 3fs init failed, cuda util init failed");
        return false;
    }

    deleteIovShm();

    // read iov
    if (!initIovHandle(read_iov_handle_, init_params_.read_iov_block_size, init_params_.read_iov_size, cuda_util)) {
        RTP_LLM_LOG_WARNING("init read iov handle failed");
        return false;
    }

    // write iov
    if (!initIovHandle(write_iov_handle_, init_params_.write_iov_block_size, init_params_.write_iov_size, cuda_util)) {
        RTP_LLM_LOG_WARNING("init write iov handle failed");
        releaseIovHandle(read_iov_handle_);
        return false;
    }

    if (metrics_reporter_) {
        stop_report_metrics_.store(false);
        report_metrics_thread_ = std::thread(&DistStorage3FS::reportMetrics, this);
    } else {
        stop_report_metrics_.store(true);
    }

    return true;
}

bool DistStorage3FS::checkInitParams(const DistStorage3FSInitParams& init_params) const {
    const auto& mountpoint = init_params.mountpoint;
    if (mountpoint.empty()) {
        RTP_LLM_LOG_WARNING("init failed, 3fs mountpoint is empty");
        return false;
    }
    if (!std::filesystem::exists(mountpoint)) {
        RTP_LLM_LOG_WARNING("init failed, 3fs mountpoint not exists: %s", mountpoint.c_str());
        return false;
    }

    const auto& root_dir = init_params.root_dir;
    if (root_dir.empty()) {
        RTP_LLM_LOG_WARNING("init failed, 3fs root dir is empty");
        return false;
    }
    const auto root_dir_path = std::filesystem::path(mountpoint) / root_dir;
    if (!std::filesystem::exists(root_dir_path)) {
        RTP_LLM_LOG_WARNING("init failed, 3fs root dir not exists: %s", root_dir_path.c_str());
        return false;
    }
    if (init_params.file_cache_capacity <= 0) {
        RTP_LLM_LOG_WARNING("init failed, 3fs file cache capacity is invalid: %zu", init_params.file_cache_capacity);
        return false;
    }
    return true;
}

bool DistStorage3FS::lookup(const DistStorage::Item& item) {
    auto file = getFile(item);
    if (file) {
        return file->exists();
    }
    return false;
}

bool DistStorage3FS::get(const DistStorage::Item& item) {
    auto file = getFile(item, true);
    if (file) {
        return file->read(item.iovs);
    }
    return false;
}

bool DistStorage3FS::put(const DistStorage::Item& item) {
    auto file = getFile(item);
    if (file) {
        return file->write(item.iovs);
    }
    return false;
}

bool DistStorage3FS::del(const DistStorage::Item& item) {
    auto file = getFile(item);
    if (file) {
        return file->del();
    }
    return false;
}

std::shared_ptr<DistStorage3FSFile> DistStorage3FS::getFile(const DistStorage::Item& item, bool cache_file) {
    auto key = item.key;
    if (auto file = getFileFromCache(key); file != nullptr) {
        return file;
    }

    const auto filepath = makeFilepath(item.metas);
    if (filepath.empty()) {
        return nullptr;
    }

    ThreeFSFileConfig config   = {init_params_.mountpoint, filepath, write_thread_pool_, metrics_reporter_};
    auto              new_file = std::make_shared<DistStorage3FSFile>(
        config, read_iov_handle_, write_iov_handle_, init_params_.read_timeout_ms, init_params_.write_timeout_ms);
    if (cache_file) {
        putFileToCache(key, new_file);
    }
    return new_file;
}

std::shared_ptr<DistStorage3FSFile> DistStorage3FS::getFileFromCache(const std::string& key) {
    std::unique_lock<std::mutex> lock(file_cache_mutex_);
    auto [found, file] = file_cache_->get(key);
    if (found) {
        return file;
    }
    return nullptr;
}

void DistStorage3FS::putFileToCache(const std::string& key, const std::shared_ptr<DistStorage3FSFile>& file) {
    std::unique_lock<std::mutex> lock(file_cache_mutex_);
    file_cache_->put(key, file);
}

void DistStorage3FS::clearFileCache() {
    std::unique_lock<std::mutex> lock(file_cache_mutex_);
    if (file_cache_) {
        file_cache_->clear();
    }
}

std::string DistStorage3FS::makeFilepath(const std::map<std::string, std::string>& metas) const {
    // kvcache filename format:
    // /mountpoint/root_dir/biz_name/ckpt_path_hash(/lora_path_hash)/seq_size_per_block/dtype/use_mla/tp_size/tp_rank/layout_version/last_cache_key
    const std::vector<std::string> required_keys = {"BIZ_NAME",
                                                    "LAYOUT_VERSION",
                                                    "CKPT_PATH",
                                                    "LORA_CKPT_PATH",
                                                    "SEQ_SIZE_PER_BLOCK",
                                                    "DTYPE",
                                                    "USE_MLA",
                                                    "TP_SIZE",
                                                    "TP_RANK",
                                                    "ITEM_KEY"};
    for (const auto& key : required_keys) {
        if (metas.find(key) == metas.end()) {
            RTP_LLM_LOG_WARNING("make filepath failed, metas missing key: %s", key.c_str());
            return "";
        }
    }

    const auto& biz_name           = metas.at("BIZ_NAME");
    const auto& layout_version     = metas.at("LAYOUT_VERSION");
    const auto& ckpt_path          = metas.at("CKPT_PATH");
    const auto& lora_ckpt_path     = metas.at("LORA_CKPT_PATH");
    const auto& seq_size_per_block = metas.at("SEQ_SIZE_PER_BLOCK");
    const auto& dtype              = metas.at("DTYPE");
    const auto& use_mla            = metas.at("USE_MLA");
    const auto& tp_size            = metas.at("TP_SIZE");
    const auto& tp_rank            = metas.at("TP_RANK");
    const auto& item_key           = metas.at("ITEM_KEY");

    const auto filepath = std::filesystem::path(init_params_.mountpoint) / init_params_.root_dir / biz_name
                          / layout_version / ckpt_path / lora_ckpt_path / seq_size_per_block / dtype / use_mla / tp_size
                          / tp_rank / item_key;
    return filepath.lexically_normal().string();
}

void DistStorage3FS::deleteIovShm() const {
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

bool DistStorage3FS::initIovHandle(ThreeFSIovHandle&                       handle,
                                   size_t                                  iov_block_size,
                                   size_t                                  iov_size,
                                   const std::shared_ptr<ThreeFSCudaUtil>& cuda_util) {
    if (iov_block_size != 0 && iov_size % iov_block_size != 0) {
        iov_size = (iov_size / iov_block_size + 1) * iov_block_size;
    }

    auto iov = createIov(init_params_.mountpoint, iov_size, iov_block_size);
    if (iov == nullptr) {
        RTP_LLM_LOG_WARNING("create iov failed, iov size: %zu, iov block size: %zu", iov_size, iov_block_size);
        return false;
    }

    auto mempool = std::make_shared<ThreeFSMempool>(iov->base, iov_size, iov_block_size);
    if (!mempool->init()) {
        RTP_LLM_LOG_WARNING("mempool init failed, iov base: %p, iov size: %zu, iov block size: %zu",
                            iov->base,
                            iov_size,
                            iov_block_size);
        releaseIov(iov);
        return false;
    }

    if (!cuda_util->registerHost(iov->base, iov_size)) {
        RTP_LLM_LOG_WARNING("cuda register iov failed, iov base: %p, expect iov size: %zu, actual iov size: %zu",
                            iov->base,
                            iov_size,
                            iov->size);
        releaseIov(iov);
        return false;
    }
    handle = {iov, nullptr, iov_size, iov_block_size, mempool, cuda_util};
    return true;
}

struct hf3fs_iov*
DistStorage3FS::createIov(const std::string& mountpoint, size_t iov_size, size_t iov_block_size) const {
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

void DistStorage3FS::releaseIov(struct hf3fs_iov* iov) const {
    if (iov == nullptr) {
        return;
    }
    hf3fs_iovdestroy(iov);
    delete iov;
}

void DistStorage3FS::releaseIovHandle(ThreeFSIovHandle& iov_handle) {
    if (iov_handle.iov != nullptr) {
        if (iov_handle.cuda_util) {
            iov_handle.cuda_util->unregisterHost(iov_handle.iov->base);
        }
        releaseIov(iov_handle.iov);
        iov_handle.iov = nullptr;
    }
    iov_handle.iov_mempool.reset();
    iov_handle.cuda_util.reset();
}

void DistStorage3FS::reportMetrics() const {
    while (!stop_report_metrics_.load()) {
        if (metrics_reporter_) {
            auto metrics          = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
            auto read_iov_mempool = read_iov_handle_.iov_mempool;
            if (read_iov_mempool) {
                DistKvCacheMetrics::setReadIovAllocatedSize(metrics, read_iov_mempool->allocatedSize());
                DistKvCacheMetrics::setReadIovAllocatedCount(metrics, read_iov_mempool->allocatedBlockCount());
                DistKvCacheMetrics::setReadIovFreeSize(metrics, read_iov_mempool->freeSize());
            }

            auto write_iov_mempool = write_iov_handle_.iov_mempool;
            if (write_iov_mempool) {
                DistKvCacheMetrics::setWriteIovAllocatedSize(metrics, write_iov_mempool->allocatedSize());
                DistKvCacheMetrics::setWriteIovAllocatedCount(metrics, write_iov_mempool->allocatedBlockCount());
                DistKvCacheMetrics::setWriteIovFreeSize(metrics, write_iov_mempool->freeSize());
            }

            const auto [_, used_bytes, available_bytes] = getFileSystemInfo();
            DistKvCacheMetrics::setFSUsedSize(metrics, used_bytes);
            DistKvCacheMetrics::setFSFreeSize(metrics, available_bytes);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

std::tuple<size_t, size_t, size_t> DistStorage3FS::getFileSystemInfo() const {
    struct statvfs stat;
    if (::statvfs(init_params_.mountpoint.c_str(), &stat) != 0) {
        RTP_LLM_LOG_WARNING("get 3fs file system info failed, statvfs failed, mountpoint: %s",
                            init_params_.mountpoint.c_str());
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

}  // namespace rtp_llm::threefs