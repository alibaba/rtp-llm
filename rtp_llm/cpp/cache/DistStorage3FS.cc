#include "rtp_llm/cpp/cache/DistStorage3FS.h"
#include "rtp_llm/cpp/cache/ThreeFSMempool.h"

#include <filesystem>

using rtp_llm::threefs::DistStorage3FSFile;
using rtp_llm::threefs::ThreeFSCudaUtil;
using rtp_llm::threefs::ThreeFSIovHandle;
using rtp_llm::threefs::ThreeFSFileConfig;
using rtp_llm::threefs::ThreeFSMempool;

namespace rtp_llm {

DistStorage3FS::DistStorage3FS(const kmonitor::MetricsReporterPtr& metrics_reporter):
    metrics_reporter_(metrics_reporter) {}

DistStorage3FS::~DistStorage3FS() {
    // TODO: 为什么是这个顺序
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

    releaseIovHandle(read_iov_handle_);
    releaseIovHandle(write_iov_handle_);
}

bool DistStorage3FS::init(const DistStorage3FSInitParams& init_params) {
    init_params_ = init_params;

    if (!init_params_.mountpoint.empty() && init_params_.mountpoint.back() != '/') {
        init_params_.mountpoint += '/';
    }

    if (!init_params_.folder_name.empty() && init_params_.folder_name.back() != '/') {
        init_params_.folder_name += '/';
    }

    if (init_params_.enable_async_write) {
        write_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
            init_params_.async_write_thread_num, init_params_.async_write_queue_size, nullptr, "3FSWriteThread");
        if (!write_thread_pool_->start()) {
            RTP_LLM_LOG_INFO(
                "dist storage 3fs init failed, start async thread pool failed, thread num: %lu, queue size: %lu",
                init_params_.async_write_thread_num,
                init_params_.async_write_queue_size);
            return false;
        }
        RTP_LLM_LOG_INFO("3fs enable async write, thread num: %lu, queue size: %lu",
                         init_params_.async_write_thread_num,
                         init_params_.async_write_queue_size);
    }

    if (!std::filesystem::exists(init_params_.mountpoint)) {
        RTP_LLM_LOG_WARNING("init failed, 3fs mountpoint not exists: %s", init_params_.mountpoint.c_str());
        return false;
    }

    auto cuda_util_ = std::make_shared<ThreeFSCudaUtil>();
    if (!cuda_util_->init()) {
        RTP_LLM_LOG_WARNING("dist storage 3fs init failed, cuda util init failed");
        return false;
    }

    // read iov
    if (!initIovHandle(read_iov_handle_, init_params_.read_iov_block_size, init_params_.read_iov_size)) {
        RTP_LLM_LOG_WARNING("init read iov handle failed");
        return false;
    }
    RTP_LLM_LOG_INFO("3fs read iov size: %zu, read iov block size: %zu",
                     init_params_.read_iov_size,
                     init_params_.read_iov_block_size);

    if (!initIovHandle(write_iov_handle_, init_params.write_iov_block_size, init_params_.write_iov_size)) {
        RTP_LLM_LOG_WARNING("init write iov handle failed");
        return false;
    }
    RTP_LLM_LOG_INFO("3fs write iov size: %zu, write iov block size: %zu",
                     init_params_.write_iov_size,
                     init_params_.write_iov_block_size);

    if (metrics_reporter_) {
        stop_report_metrics_.store(false);
        // TODO: report_metrics_thread_ = std::thread(&ThreeFSBlockCache::reportMetrics, this);
    } else {
        stop_report_metrics_.store(true);
    }

    RTP_LLM_LOG_INFO("3fs init done, mountpoint: %s, folder name: %s",
                     init_params_.mountpoint.c_str(),
                     init_params_.folder_name.c_str());
    return true;
}
bool DistStorage3FS::lookup(const DistStorage::Item& item) {
    auto file = getFile(item);
    if (file == nullptr || !file->isExist()) {
        return false;
    }
    return true;
}

bool DistStorage3FS::get(const DistStorage::Item& item) {
    auto file = getFile(item);
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
        if (!file->del()) {
            return false;
        }
        removeFile(item);
    }
    return true;
}

std::shared_ptr<DistStorage3FSFile> DistStorage3FS::getFile(const DistStorage::Item& item) {
    // TODO: 轮转
    auto key = item.key;

    std::unique_lock<std::shared_mutex> lock(file_map_mutex_);
    auto                                iter = file_map_.find(key);
    if (iter != file_map_.end()) {
        return iter->second;
    }

    // TODO: config
    ThreeFSFileConfig config;
    auto              new_file = std::make_shared<DistStorage3FSFile>(config, read_iov_handle_, write_iov_handle_);
    file_map_[key]             = new_file;
    return new_file;
}

void DistStorage3FS::removeFile(const DistStorage::Item& item) {
    auto                                key = item.key;
    std::unique_lock<std::shared_mutex> lock(file_map_mutex_);
    file_map_.erase(key);
}

void DistStorage3FS::removeOldIov() const {
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

bool DistStorage3FS::initIovHandle(ThreeFSIovHandle& handle, size_t iov_block_size, size_t iov_size) {
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

    if (!cuda_util_->registerHost(iov->base, iov_size)) {
        RTP_LLM_LOG_WARNING("cuda register iov failed, iov base: %p, iov size: %zu", iov->base, iov_size);
        releaseIov(iov);
        return false;
    }
    handle = {iov, nullptr, iov_size, iov_block_size, mempool, cuda_util_};
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
    return nullptr;
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

}  // namespace rtp_llm