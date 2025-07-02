#include "rtp_llm/cpp/cache/ThreeFSFile.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <time.h>
#include <sstream>
#include <sys/resource.h>

#include "autil/EnvUtil.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/cache/ThreeFSCudaUtil.h"
#include "rtp_llm/cpp/cache/ThreeFSMempool.h"
#include "rtp_llm/cpp/cache/ThreeFSMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::threefs {

std::map<std::string, int> ThreeFSFile::filename_to_fd_map_;
std::shared_mutex          ThreeFSFile::filename_to_fd_map_mutex_;

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

ThreeFSFile::ThreeFSFile(const ThreeFSFileConfig& config,
                         const ThreeFSIovHandle&  read_iov_handle,
                         const ThreeFSIovHandle&  write_iov_handle):
    config_(config), read_iov_handle_(read_iov_handle), write_iov_handle_(write_iov_handle) {
    cache_meta_ = autil::EnvUtil::getEnv("THREEFS_CACHE_META", cache_meta_);
    // for debug
    enable_kvcache_verify_ = autil::EnvUtil::getEnv("THREEFS_ENABLE_KVCACHE_VERIFY", enable_kvcache_verify_);
    if (enable_kvcache_verify_) {
        RTP_LLM_LOG_INFO("3fs enable kvcache verify");
    }
}

ThreeFSFile::~ThreeFSFile() {
    RTP_LLM_LOG_DEBUG("3fs file destructor, file: %s", config_.filename.c_str());
    if (fd_ != -1) {
        hf3fs_dereg_fd(fd_);
        ::close(fd_);
        fd_ = -1;
    }
}

bool ThreeFSFile::match(const std::vector<int64_t>& cache_keys) {
    if (cache_keys.empty()) {
        RTP_LLM_LOG_WARNING("match failed, cache key list is empty");
        return false;
    }

    auto handle = initIovIor(static_cast<int>(cache_keys.size()), true, true);
    if (handle == nullptr) {
        RTP_LLM_LOG_WARNING("match failed, init iov/ior failed, filename: %s", config_.filename.c_str());
        return false;
    }

    bool result = doMatch(handle, cache_keys);
    releaseIovIor(handle);
    return result;
}

bool ThreeFSFile::doMatch(const std::shared_ptr<ThreeFSHandle>& handle, const std::vector<int64_t>& cache_keys) {
    if (handle == nullptr) {
        RTP_LLM_LOG_WARNING("match failed, handle is nullptr");
        return false;
    }

    auto metrics = ThreeFSMetricsFactory::createMetrics(config_.metrics_reporter);
    if (!readMetas(handle, metrics)) {
        RTP_LLM_LOG_WARNING("match failed, read metas failed, filename: %s", config_.filename.c_str());
        return false;
    }

    return checkAllCacheKeyMatched(cache_keys);
}

bool ThreeFSFile::checkAllCacheKeyMatched(const std::vector<int64_t>& cache_keys) const {
    std::shared_lock<std::shared_mutex> lock(cache_key_map_mutex_);
    for (const auto cache_key : cache_keys) {
        if (cache_key_map_.count(cache_key) == 0) {
            std::string cache_keys_str;
            for (const auto& [cache_key, offset] : cache_key_map_) {
                cache_keys_str += std::to_string(cache_key) + ',';
            }
            RTP_LLM_LOG_WARNING(
                "all cache key not matched, file: %s, not matched cache key: %ld, remote cache key: [%lu|%s]",
                config_.filename.c_str(),
                cache_key,
                cache_key_map_.size(),
                cache_keys_str.c_str());
            return false;
        }
    }
    return true;
}

bool ThreeFSFile::read(const std::vector<int64_t>& cache_keys, const std::vector<int32_t>& block_indices) {
    if (cache_keys.empty() && block_indices.empty()) {
        return true;
    }
    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_WARNING(
            "read failed, cache key size not equal to block index size, cache key size: %lu, block index size: %lu",
            cache_keys.size(),
            block_indices.size());
        return false;
    }

    auto handle = initIovIor(static_cast<int>(cache_keys.size()), true);
    if (handle == nullptr) {
        RTP_LLM_LOG_WARNING("read failed, init iov/ior failed, filename: %s", config_.filename.c_str());
        return false;
    }

    bool result = doRead(handle, cache_keys, block_indices);
    releaseIovIor(handle);
    return result;
}

bool ThreeFSFile::doRead(const std::shared_ptr<ThreeFSHandle>& handle,
                         const std::vector<int64_t>&           cache_keys,
                         const std::vector<int32_t>&           block_indices) {
    if (handle == nullptr) {
        RTP_LLM_LOG_WARNING("read failed, handle is nullptr");
        return false;
    }
    if (!config_.k_cache || !config_.v_cache) {
        RTP_LLM_LOG_WARNING("read failed, k cache or v cache is nullptr. k cache: %p, v cache: %p",
                            config_.k_cache.get(),
                            config_.v_cache.get());
        return false;
    }

    const int     cache_key_count = static_cast<int>(cache_keys.size());
    const int64_t block_len_per_cache_key =
        config_.cache_config.layer_num * (config_.cache_config.k_block_stride + config_.cache_config.v_block_stride);
    const int64_t block_total_len = block_len_per_cache_key * cache_key_count;
    if (block_total_len > handle->iov_handle.iov_size) {
        RTP_LLM_LOG_WARNING(
            "read failed, read size exceed iov size, read size: %lu, iov size: %lu, block_len_per_cache_key: %ld, block_total_len: %d",
            block_total_len,
            handle->iov_handle.iov_size,
            block_len_per_cache_key,
            block_total_len);
        return false;
    }

    auto metrics = ThreeFSMetricsFactory::createMetrics(config_.metrics_reporter);
    ThreeFSMetrics::markTotalReadBeginUs(metrics);

    // non-zero ranks need to read meta first
    if (!readMetas(handle, metrics)) {
        ThreeFSMetrics::markTotalReadDoneUs(metrics);
        RTP_LLM_LOG_WARNING("read failed, read metas failed, filename: %s", config_.filename.c_str());
        return false;
    }
    if (!checkAllCacheKeyMatched(cache_keys)) {
        ThreeFSMetrics::markTotalReadDoneUs(metrics);
        RTP_LLM_LOG_WARNING("read failed, all cache key matched failed, filename: %s", config_.filename.c_str());
        return false;
    }
    std::map<int64_t, int64_t> cache_key_map;
    {
        std::shared_lock<std::shared_mutex> lock(cache_key_map_mutex_);
        cache_key_map = cache_key_map_;
    }

    ThreeFSMetrics::markReadBlockBeginUs(metrics);
    auto& ior      = handle->ior_handle.ior;
    auto& iov      = handle->iov_handle.iov;
    auto  iov_base = handle->iov_handle.iov_base;

    const auto                 ior_entries     = handle->ior_handle.ior_entries;
    int64_t                    iov_offset      = 0;
    int32_t                    submit_io_count = 0;
    std::map<int64_t, int64_t> cache_key_2_iov_offset;
    for (int32_t cache_key_pos = 0; cache_key_pos < cache_key_count; ++cache_key_pos) {
        const auto cache_key              = cache_keys[cache_key_pos];
        auto       file_offset            = cache_key_map[cache_key];
        cache_key_2_iov_offset[cache_key] = iov_offset;

        const auto iov_block_size = handle->iov_handle.iov_block_size;
        auto       remaining_size = block_len_per_cache_key;
        while (remaining_size > 0) {
            uint64_t cur_read_len = 0;
            if (iov_block_size > 0) {
                cur_read_len = std::min(remaining_size, calcLeftSizeInBlock(iov_block_size, iov_offset));
            } else {
                cur_read_len = remaining_size > kDefaultReadSizePerIo ? kDefaultReadSizePerIo : remaining_size;
            }

            auto ret = hf3fs_prep_io(ior, iov, true, iov_base + iov_offset, fd_, file_offset, cur_read_len, nullptr);
            if (ret < 0) {
                ThreeFSMetrics::markReadBlockDoneUs(metrics);
                ThreeFSMetrics::markTotalReadDoneUs(metrics);
                // TODO(LXQ): 需要处理掉之前已提交的io请求(submit然后wait)
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
                    block_len_per_cache_key);
                return false;
            }
            ++submit_io_count;
            iov_offset += cur_read_len;
            file_offset += cur_read_len;
            remaining_size -= cur_read_len;

            if (submit_io_count < ior_entries) {
                if (remaining_size <= 0) {
                    break;
                }
                continue;
            }

            if (!submitAndWaitForReadIos(handle, submit_io_count)) {
                ThreeFSMetrics::markReadBlockDoneUs(metrics);
                ThreeFSMetrics::markTotalReadDoneUs(metrics);
                RTP_LLM_LOG_WARNING(
                    "read block failed, read submit/wait io failed. file: %s, cur_read_len: %lu, iov_block_size: %lu",
                    config_.filename.c_str(),
                    cur_read_len,
                    iov_block_size);
                return false;
            }
            submit_io_count = 0;
        }

        if (submit_io_count == 0) {
            continue;
        }
        if (submit_io_count < ior_entries && cache_key_pos + 1 != cache_key_count) {
            continue;
        }

        if (!submitAndWaitForReadIos(handle, submit_io_count)) {
            ThreeFSMetrics::markReadBlockDoneUs(metrics);
            ThreeFSMetrics::markTotalReadDoneUs(metrics);
            RTP_LLM_LOG_WARNING(
                "read block failed, read submit/wait io failed. file: %s, iov_block_size: %lu, block_len_per_cache_key: %ld",
                config_.filename.c_str(),
                iov_block_size,
                block_len_per_cache_key);
            return false;
        }
        submit_io_count = 0;
    }

    ThreeFSMetrics::markReadBlockDoneUs(metrics);
    ThreeFSMetrics::markReadCudaCopyBeginUs(metrics);
    auto cuda_util = handle->iov_handle.cuda_util;

    for (int i = 0; i < cache_key_count; ++i) {
        const auto cache_key   = cache_keys[i];
        const auto block_index = block_indices[i];
        auto       iov_offset  = cache_key_2_iov_offset[cache_key];
        const auto k_block_len = config_.cache_config.k_block_stride;
        const auto v_block_len = config_.cache_config.v_block_stride;
        for (int layer_index = 0; layer_index < config_.cache_config.layer_num; ++layer_index) {
            auto [k_addr, v_addr] = convertIndexToAddr(block_index, layer_index);
            cuda_util->copyAsyncHostToDevice(k_addr, iov_base + iov_offset, k_block_len);
            iov_offset += k_block_len;
            cuda_util->copyAsyncHostToDevice(v_addr, iov_base + iov_offset, v_block_len);
            iov_offset += v_block_len;
        }
    }
    cuda_util->sync();
    ThreeFSMetrics::markReadCudaCopyDoneUs(metrics);
    ThreeFSMetrics::markTotalReadDoneUs(metrics);

    const auto read_total_len = getMetaTotalLength(cache_key_count) + block_total_len;
    ThreeFSMetrics::setTotalReadLen(metrics, read_total_len);
    ThreeFSMetrics::setReadBlockLen(metrics, block_total_len);
    if (metrics) {
        ThreeFSMetrics::setTotalReadThroughput(metrics, calcMiBs(read_total_len, metrics->TotalReadCostUs()));
        ThreeFSMetrics::setReadBlockThroughput(metrics, calcMiBs(block_total_len, metrics->ReadBlockCostUs()));
        RTP_LLM_LOG_DEBUG(
            "read from 3fs, filename: %s, total cost(%ld us) = read meta(%ld) + read block(%ld) + cuda memcpy(%ld). total len: %lu, block len: %lu, cache key count: %d",
            config_.filename.c_str(),
            metrics->TotalReadCostUs(),
            metrics->ReadMetaCostUs(),
            metrics->ReadBlockCostUs(),
            metrics->ReadCudaCopyCostUs(),
            read_total_len,
            block_total_len,
            cache_key_count);
    }
    return true;
}

bool ThreeFSFile::readMetas(const std::shared_ptr<ThreeFSHandle>& handle, std::shared_ptr<ThreeFSMetrics> metrics) {
    ThreeFSMetrics::markReadMetaBeginUs(metrics);
    if (!cache_meta_) {
        std::unique_lock<std::shared_mutex> lock(cache_key_map_mutex_);
        cache_key_map_.clear();
    }
    {
        std::shared_lock<std::shared_mutex> lock(cache_key_map_mutex_);
        if (!cache_key_map_.empty()) {
            ThreeFSMetrics::markReadMetaDoneUs(metrics);
            return true;
        }
    }

    std::vector<ThreeFSMeta> metas;
    while (true) {
        auto& ior      = handle->ior_handle.ior;
        auto& iov      = handle->iov_handle.iov;
        auto  iov_base = handle->iov_handle.iov_base;

        // 读 meta count 时预读一些 meta , 减少读的次数
        const int  preread_meta_count = 50;
        const int  single_meta_size   = static_cast<int>(sizeof(ThreeFSMeta));
        const auto iov_block_size     = handle->iov_handle.iov_block_size;
        const auto ior_entries        = handle->ior_handle.ior_entries;
        int64_t    remaining_size     = sizeof(int32_t) + preread_meta_count * single_meta_size;
        int64_t    iov_offset         = 0;
        int64_t    file_offset        = kReservedLength;
        int32_t    submit_io_count    = 0;

        while (remaining_size > 0) {
            uint64_t cur_read_len = 0;
            if (iov_block_size > 0) {
                cur_read_len = std::min(remaining_size, calcLeftSizeInBlock(iov_block_size, iov_offset));
            } else {
                cur_read_len = remaining_size > kDefaultReadSizePerIo ? kDefaultReadSizePerIo : remaining_size;
            }

            auto ret = hf3fs_prep_io(ior, iov, true, iov_base + iov_offset, fd_, file_offset, cur_read_len, nullptr);
            if (ret < 0) {
                ThreeFSMetrics::markReadMetaDoneUs(metrics);
                // TODO(LXQ): 需要处理掉之前已提交的io请求(submit然后wait)
                RTP_LLM_LOG_WARNING(
                    "read meta count failed, hf3fs_prep_io failed, errno: %s, fd: %d, cur_read_len: %lu, iov_block_size: %lu",
                    strerror(-ret),
                    fd_,
                    cur_read_len,
                    iov_block_size);
                return false;
            }
            ++submit_io_count;
            iov_offset += cur_read_len;
            file_offset += cur_read_len;
            remaining_size -= cur_read_len;

            if (submit_io_count < ior_entries && remaining_size > 0) {
                continue;
            }

            if (!submitAndWaitForReadIos(handle, submit_io_count)) {
                ThreeFSMetrics::markReadMetaDoneUs(metrics);
                RTP_LLM_LOG_WARNING(
                    "read meta count failed, read submit/wait io failed. submit io count: %d, file: %s, cur_read_len: %lu, iov_block_size: %lu",
                    submit_io_count,
                    config_.filename.c_str(),
                    cur_read_len,
                    iov_block_size);
                return false;
            }
            submit_io_count = 0;  // reset submit io count
        }

        int32_t meta_count = 0;
        std::memcpy(&meta_count, iov_base, sizeof(int32_t));
        if (meta_count <= 0) {
            ThreeFSMetrics::markReadMetaDoneUs(metrics);
            RTP_LLM_LOG_WARNING("read meta count failed, value is invalid: %d", meta_count);
            return false;
        }

        if (meta_count <= preread_meta_count) {
            ThreeFSMetrics::markReadMetaDoneUs(metrics);
            metas.resize(meta_count);
            std::memcpy(metas.data(), iov_base + sizeof(int32_t), meta_count * single_meta_size);
            break;
        }

        // meta 没读完, 继续读
        submit_io_count = 0;
        iov_offset      = 0;
        file_offset     = kReservedLength + sizeof(int32_t);
        remaining_size  = meta_count * single_meta_size;

        while (remaining_size > 0) {
            uint64_t cur_read_len = 0;
            if (iov_block_size > 0) {
                cur_read_len = std::min(remaining_size, calcLeftSizeInBlock(iov_block_size, iov_offset));
            } else {
                if (remaining_size > preread_meta_count * single_meta_size) {
                    cur_read_len = preread_meta_count * single_meta_size;
                } else {
                    cur_read_len = remaining_size > kDefaultReadSizePerIo ? kDefaultReadSizePerIo : remaining_size;
                }
            }

            auto ret = hf3fs_prep_io(ior, iov, true, iov_base + iov_offset, fd_, file_offset, cur_read_len, nullptr);
            if (ret < 0) {
                ThreeFSMetrics::markReadMetaDoneUs(metrics);
                // TODO(LXQ): 需要处理掉之前已提交的io请求(submit然后wait)
                RTP_LLM_LOG_WARNING(
                    "read meta failed, hf3fs_prep_io failed, errno: %s, fd: %d, cur_read_len: %lu, iov_block_size: lu",
                    strerror(-ret),
                    fd_,
                    cur_read_len,
                    iov_block_size);
                return false;
            }
            ++submit_io_count;
            iov_offset += cur_read_len;
            file_offset += cur_read_len;
            remaining_size -= cur_read_len;

            if (submit_io_count < ior_entries && remaining_size > 0) {
                continue;
            }

            if (!submitAndWaitForReadIos(handle, submit_io_count)) {
                ThreeFSMetrics::markReadMetaDoneUs(metrics);
                RTP_LLM_LOG_WARNING(
                    "read meta failed, read submit/wait io failed. submit io count: %d, file: %s, cur_read_len: %lu, iov_block_size: lu",
                    submit_io_count,
                    config_.filename.c_str(),
                    cur_read_len,
                    iov_block_size);
                return false;
            }
            submit_io_count = 0;
        }
        ThreeFSMetrics::markReadMetaDoneUs(metrics);

        metas.resize(meta_count);
        std::memcpy(metas.data(), iov_base, meta_count * single_meta_size);
        break;
    }

    ThreeFSMetrics::setReadMetaLen(metrics, getMetaTotalLength(static_cast<int>(metas.size())));
    if (metrics) {
        RTP_LLM_LOG_DEBUG("read metas, file: %s, total cost: %ld us, meta count: %lu",
                          config_.filename.c_str(),
                          metrics->ReadMetaCostUs(),
                          metas.size());
    }

    {
        std::unique_lock<std::shared_mutex> lock(cache_key_map_mutex_);
        for (const auto& meta : metas) {
            cache_key_map_[meta.cache_key] = meta.offset;
        }
    }
    return true;
}

bool ThreeFSFile::write(const std::vector<int64_t>& cache_keys, const std::vector<int32_t>& block_indices) {
    if (cache_keys.empty() && block_indices.empty()) {
        return true;
    }
    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_WARNING(
            "write failed, cache key size not equal to block index size, cache key size: %lu, block index size: %lu",
            cache_keys.size(),
            block_indices.size());
        return false;
    }

    auto handle = initIovIor(static_cast<int>(cache_keys.size()), false);
    if (handle == nullptr) {
        RTP_LLM_LOG_WARNING("write failed, init iov/ior failed, filename: %s", config_.filename.c_str());
        return false;
    }

    return doWrite(handle, cache_keys, block_indices);
}

bool ThreeFSFile::doWrite(const std::shared_ptr<ThreeFSHandle>& handle,
                          const std::vector<int64_t>&           cache_keys,
                          const std::vector<int32_t>&           block_indices) {
    if (handle == nullptr) {
        RTP_LLM_LOG_WARNING("write failed, handle is nullptr");
        return false;
    }
    if (!config_.k_cache || !config_.v_cache) {
        RTP_LLM_LOG_WARNING("write failed, k cache or v cache is nullptr. k cache: %p, v cache: %p",
                            config_.k_cache.get(),
                            config_.v_cache.get());
        return false;
    }
    const int cache_key_count = static_cast<int>(cache_keys.size());

    auto metrics = ThreeFSMetricsFactory::createMetrics(config_.metrics_reporter);
    ThreeFSMetrics::markTotalWriteBeginUs(metrics);

    const auto               k_block_len = config_.cache_config.k_block_stride;
    const auto               v_block_len = config_.cache_config.v_block_stride;
    std::vector<ThreeFSItem> items(cache_key_count);

    for (int i = 0; i < cache_key_count; ++i) {
        auto& item     = items[i];
        item.cache_key = cache_keys[i];

        const auto block_index = block_indices[i];
        for (int layer_index = 0; layer_index < config_.cache_config.layer_num; ++layer_index) {
            auto [k_addr, v_addr] = convertIndexToAddr(block_index, layer_index);

            auto k_block = std::shared_ptr<void>(k_addr, [](void* p) {});
            item.blocks.push_back({k_block, k_block_len});

            auto v_block = std::shared_ptr<void>(v_addr, [](void* p) {});
            item.blocks.push_back({v_block, v_block_len});
        }
    }

    const size_t all_block_len   = cache_key_count * config_.cache_config.layer_num * (k_block_len + v_block_len);
    const size_t write_total_len = getMetaTotalLength(cache_key_count) + all_block_len;
    if (write_total_len > handle->iov_handle.iov_size) {
        ThreeFSMetrics::markTotalWriteDoneUs(metrics);
        RTP_LLM_LOG_WARNING("write failed, write size exceed iov size, write size: %lu, iov size: %lu",
                            write_total_len,
                            handle->iov_handle.iov_size);
        return false;
    }

    // 以上校验通过后再创建文件准备写入, 避免创建一个空文件
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (!openFile(false)) {
        ThreeFSMetrics::markTotalWriteDoneUs(metrics);
        RTP_LLM_LOG_WARNING("write block failed, open file failed, filename: %s", config_.filename.c_str());
        return false;
    }

    const auto t2 = std::chrono::high_resolution_clock::now();
    if (!writeTo3FS(handle, items, metrics)) {
        ThreeFSMetrics::markTotalWriteDoneUs(metrics);
        RTP_LLM_LOG_WARNING("write block failed, 3fs write failed, filename: %s", config_.filename.c_str());
        // remove file when write failed
        removeFile();
        return false;
    }

    const auto t3 = std::chrono::high_resolution_clock::now();
    ThreeFSMetrics::markTotalWriteDoneUs(metrics);
    ThreeFSMetrics::setTotalWriteLen(metrics, write_total_len);
    if (metrics) {
        ThreeFSMetrics::setTotalWriteThroughput(metrics, calcMiBs(write_total_len, metrics->TotalWriteCostUs()));

        const auto open_file_cost_us      = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        const auto write_and_copy_cost_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        const auto write_cost_us          = write_and_copy_cost_us - metrics->WriteCudaCopyCostUs();
        RTP_LLM_LOG_DEBUG(
            "write to 3fs, filename: %s, total cost(%ld us) = open file(%ld) + cuda memcpy(%ld) + write cost(%ld). total len: %lu, block len: %d, cache key count: %d",
            config_.filename.c_str(),
            metrics->TotalWriteCostUs(),
            open_file_cost_us,
            metrics->WriteCudaCopyCostUs(),
            write_cost_us,
            write_total_len,
            all_block_len,
            cache_key_count);
    }

    // only for test
    if (enable_kvcache_verify_) {
        verify(handle, cache_keys, block_indices);
    }

    return true;
}

bool ThreeFSFile::writeTo3FS(const std::shared_ptr<ThreeFSHandle>& handle,
                             const std::vector<ThreeFSItem>&       items,
                             std::shared_ptr<ThreeFSMetrics>       metrics) {
    /**
     * File layout:
     * +----------+------------+--------+--------+-----+--------+----------+----------+-----+----------+
     * | reserved | meta count | meta 1 | meta 2 | ... | meta N | blocks 1 | blocks 2 | ... | blocks N |
     * +----------+------------+--------+--------+-----+--------+----------+----------+-----+----------+
     * | 1024B    | 4B         | 16B    | 16B    |     | 16B    | LEN      | LEN      | ... | LEN      |
     * +----------+------------+--------+--------+-----+--------+----------+----------+-----+----------+
     *                         |   ||   |
     *                         |   \/   |
     *                         +-----------+--------+
     *                         | cache key | offset |
     *                         +-----------+--------+
     *                         | 8B        | 8B     |
     *                         +-----------+--------+
     * Total size (Byte): 1024 + 4 + (16 * N) + (Len * N)
     */

    if (items.empty()) {
        RTP_LLM_LOG_WARNING("write to 3fs failed, item list is empty");
        return false;
    }

    const int32_t            meta_count  = static_cast<int32_t>(items.size());
    int                      file_offset = getMetaTotalLength(meta_count);
    std::vector<ThreeFSMeta> metas(meta_count);
    for (int i = 0; i < meta_count; ++i) {
        metas[i].cache_key = items[i].cache_key;
        metas[i].offset    = file_offset;
        file_offset += std::accumulate(
            items[i].blocks.begin(), items[i].blocks.end(), 0, [](size_t num, const ThreeFSItem::SingleBlock& block) {
                return num + block.len;
            });
    }
    const auto write_total_len = file_offset;

    {
        auto& ior      = handle->ior_handle.ior;
        auto& iov      = handle->iov_handle.iov;
        auto  iov_base = handle->iov_handle.iov_base;

        // copy metas
        int64_t iov_offset = 0;
        std::memset(iov_base + iov_offset, 0, kReservedLength);
        iov_offset += kReservedLength;
        std::memcpy(iov_base + iov_offset, &meta_count, sizeof(int32_t));
        iov_offset += sizeof(int32_t);
        std::memcpy(iov_base + iov_offset, metas.data(), meta_count * sizeof(ThreeFSMeta));
        iov_offset += meta_count * sizeof(ThreeFSMeta);

        // copy blocks
        auto cuda_util = handle->iov_handle.cuda_util;
        ThreeFSMetrics::markWriteCudaCopyBeginUs(metrics);
        for (const auto& item : items) {
            for (const auto& block : item.blocks) {
                cuda_util->copyAsyncDeviceToHost(iov_base + iov_offset, block.data.get(), block.len);
                iov_offset += block.len;
            }
        }
        cuda_util->sync();
        ThreeFSMetrics::markWriteCudaCopyDoneUs(metrics);

        // write to 3fs
        ThreeFSMetrics::markWriteBlockBeginUs(metrics);
        const auto write_total_size = iov_offset;
        const auto iov_block_size   = handle->iov_handle.iov_block_size;
        const auto ior_entries      = handle->ior_handle.ior_entries;
        int64_t    remaining_size   = iov_offset;
        iov_offset                  = 0;
        int submit_io_count         = 0;
        while (remaining_size > 0) {
            uint64_t cur_write_len = 0;
            if (iov_block_size > 0) {
                cur_write_len = std::min(remaining_size, calcLeftSizeInBlock(iov_block_size, iov_offset));
            } else {
                cur_write_len = remaining_size > kDefaultWriteSizePerIo ? kDefaultWriteSizePerIo : remaining_size;
            }

            auto ret = hf3fs_prep_io(ior, iov, false, iov_base + iov_offset, fd_, iov_offset, cur_write_len, nullptr);
            if (ret < 0) {
                ThreeFSMetrics::markWriteBlockDoneUs(metrics);
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
                ThreeFSMetrics::markWriteBlockDoneUs(metrics);
                RTP_LLM_LOG_WARNING("write to 3fs failed, hf3fs_submit_ios failed, errno: %s", strerror(-ret));
                return false;
            }

            bool async_wait_io = false;
            bool last_io       = remaining_size <= 0;
            if (config_.write_thread_pool) {
                // async wait io
                auto shared_this = shared_from_this();
                auto work_item =
                    new WaitIoWorkItem(shared_this, handle, submit_io_count, write_total_len, last_io, metrics);
                if (auto error_code = config_.write_thread_pool->pushWorkItem(work_item, false);
                    error_code != autil::ThreadPool::ERROR_NONE) {
                    RTP_LLM_LOG_WARNING("write to 3fs failed, push work item failed, error code: %d, file: %s",
                                        error_code,
                                        config_.filename.c_str());
                    work_item->destroy();
                    work_item     = nullptr;
                    async_wait_io = false;
                    // return false;
                } else {
                    async_wait_io = true;
                }
                ThreeFSMetrics::setWriteThreadPoolWorkItemCount(metrics, config_.write_thread_pool->getItemCount());
            }
            if (!async_wait_io) {
                // sync wait io
                if (!waitForWriteIos(handle, submit_io_count, write_total_len, last_io, metrics)) {
                    ThreeFSMetrics::markWriteBlockDoneUs(metrics);
                    RTP_LLM_LOG_WARNING(
                        "write to 3fs failed, wait for write ios failed, file: %s, cur_write_len: %lu, iov_block_size: %lu",
                        config_.filename.c_str(),
                        cur_write_len,
                        iov_block_size);
                    return false;
                }
                if (last_io) {
                    ThreeFSMetrics::markWriteBlockDoneUs(metrics);
                    if (metrics) {
                        ThreeFSMetrics::setWriteBlockThroughput(metrics,
                                                                calcMiBs(write_total_len, metrics->WriteBlockCostUs()));
                    }
                }
            }

            submit_io_count = 0;  // reset submit io count
        }
    }

    if (cache_meta_) {  // default true
        std::unique_lock<std::shared_mutex> lock(cache_key_map_mutex_);
        for (const auto& meta : metas) {
            cache_key_map_[meta.cache_key] = meta.offset;
        }
    }

    return true;
}

bool ThreeFSFile::waitForWriteIos(const std::shared_ptr<ThreeFSHandle>& handle,
                                  int32_t                               submit_io_count,
                                  int64_t                               write_total_len,
                                  bool                                  last_io,
                                  std::shared_ptr<ThreeFSMetrics>       metrics) const {
    if (submit_io_count <= 0) {
        if (last_io) {
            ThreeFSMetrics::markWriteBlockDoneUs(metrics);
        }
        return true;
    }
    if (handle == nullptr || handle->ior_handle.ior == nullptr) {
        if (last_io) {
            ThreeFSMetrics::markWriteBlockDoneUs(metrics);
        }
        return false;
    }

    hf3fs_cqe cqes[submit_io_count];
    int       timeout_ms         = 2000;  // TODO(LXQ) timeout threhold
    auto      time_spec          = createTimeoutTimeSpec(timeout_ms);
    auto      ior                = handle->ior_handle.ior;
    int       completed_io_count = hf3fs_wait_for_ios(ior, cqes, submit_io_count, submit_io_count, &time_spec);
    if (completed_io_count < 0) {
        ThreeFSMetrics::markWriteBlockDoneUs(metrics);
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
            ThreeFSMetrics::markWriteBlockDoneUs(metrics);
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
        ThreeFSMetrics::markWriteBlockDoneUs(metrics);
        if (metrics) {
            ThreeFSMetrics::setWriteBlockThroughput(metrics, calcMiBs(write_total_len, metrics->WriteBlockCostUs()));
        }
    }

    return true;
}

bool ThreeFSFile::submitAndWaitForReadIos(const std::shared_ptr<ThreeFSHandle>& handle, int32_t submit_io_count) const {
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

std::shared_ptr<ThreeFSHandle> ThreeFSFile::initIovIor(int cache_key_count, bool for_read, bool read_for_match) {
    const auto& iov_handle = for_read ? read_iov_handle_ : write_iov_handle_;
    if (!checkIovHandle(iov_handle)) {
        RTP_LLM_LOG_WARNING("init iov/ior failed, check iov handle failed, read: %d", for_read);
        return nullptr;
    }

    size_t file_len{0};
    if (for_read && read_for_match) {
        file_len = getMetaTotalLength(cache_key_count);
    } else {
        file_len = calcFileLength(cache_key_count);
    }

    auto iov_buffer = iov_handle.iov_mempool->alloc(file_len);
    if (iov_buffer == nullptr) {
        RTP_LLM_LOG_WARNING(
            "init iov/ior failed, mempool alloc failed, read: %d, alloc len: %zu, mempool free size: %zu, file: %s",
            for_read,
            file_len,
            iov_handle.iov_mempool->freeSize(),
            config_.filename.c_str());
        return nullptr;
    }

    // 写时由于是异步wait io, 所以文件太大可能会出现ior entry不够用的情况, 此处根据文件长度计算所需的ior entry;
    // 读是同步读所以不会出现此问题
    ThreeFSIorHandle ior_handle;
    const auto       default_size_per_io = for_read ? kDefaultReadSizePerIo : kDefaultWriteSizePerIo;
    const int32_t    iov_block_size = iov_handle.iov_block_size != 0 ? iov_handle.iov_block_size : default_size_per_io;
    ior_handle.ior_entries          = file_len / iov_block_size + 1;

    struct hf3fs_ior* ior{nullptr};
    if (!createIor(ior, for_read, ior_handle.ior_entries, ior_handle.ior_io_depth, ior_handle.ior_timeout_ms)) {
        RTP_LLM_LOG_WARNING(
            "init iov/ior failed, create ior failed, filename: %s, read: %d, ior entries: %d, ior io depth: %d, ior timeout ms: %d",
            config_.filename.c_str(),
            for_read,
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

    // 读时预先打开文件, 写时推迟到实际写时再打开文件, 因为写时需要创建文件
    if (for_read && !openFile(true)) {
        RTP_LLM_LOG_WARNING("init ior/ior failed, open file failed on read: %s, file: %s",
                            getFilepath().c_str(),
                            config_.filename.c_str());
        releaseIovIor(handle);
        handle = nullptr;
        return nullptr;
    }
    return handle;
}

bool ThreeFSFile::checkIovHandle(const ThreeFSIovHandle& iov_handle) const {
    if (iov_handle.iov == nullptr) {
        RTP_LLM_LOG_WARNING("check iov handle, iov is nullptr");
        return false;
    }
    if (iov_handle.iov_mempool == nullptr) {
        RTP_LLM_LOG_WARNING("check iov handle, iov mempool is nullptr");
        return false;
    }
    if (iov_handle.cuda_util == nullptr) {
        RTP_LLM_LOG_WARNING("check iov handle, cuda util is nullptr");
        return false;
    }
    return true;
}

bool ThreeFSFile::createIor(
    struct hf3fs_ior*& ior, bool for_read, int ior_entries, int ior_io_depth, int ior_timeout_ms) const {
    if (config_.mountpoint.empty()) {
        RTP_LLM_LOG_WARNING("create ior failed, mountpoint is empty");
        return false;
    }

    RTP_LLM_LOG_DEBUG("create ior, 3fs file: %s, read: %d, ior entries: %d, ior io depth: %d, ior timeout ms: %d",
                      config_.filename.c_str(),
                      for_read,
                      ior_entries,
                      ior_io_depth,
                      ior_timeout_ms);

    const auto t1 = std::chrono::high_resolution_clock::now();
    ior           = new struct hf3fs_ior();
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

    const auto t2                 = std::chrono::high_resolution_clock::now();
    const auto create_ior_cost_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    RTP_LLM_LOG_DEBUG(
        "create ior cost: %lu us, file: %s, read: %d", create_ior_cost_us, config_.filename.c_str(), for_read);
    return true;
}

void ThreeFSFile::releaseIovIor(const std::shared_ptr<ThreeFSHandle>& handle) {
    if (handle == nullptr) {
        return;
    }

    const auto t1         = std::chrono::high_resolution_clock::now();
    auto&      iov_handle = handle->iov_handle;
    if (iov_handle.iov_base && iov_handle.iov_mempool) {
        iov_handle.iov_mempool->free(static_cast<void*>(iov_handle.iov_base));
    }
    // iov_handle.iov      = nullptr;
    iov_handle.iov_base = nullptr;

    const auto t2         = std::chrono::high_resolution_clock::now();
    auto&      ior_handle = handle->ior_handle;
    if (ior_handle.ior != nullptr) {
        hf3fs_iordestroy(ior_handle.ior);
        delete ior_handle.ior;
        ior_handle.ior = nullptr;
    }

    const auto t3               = std::chrono::high_resolution_clock::now();
    const auto free_iov_cost    = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto destroy_ior_cost = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    RTP_LLM_LOG_DEBUG("free iov cost: %lu us, destroy ior cost: %lu us", free_iov_cost, destroy_ior_cost);
}

std::tuple<void*, void*> ThreeFSFile::convertIndexToAddr(int block_index, int layer_id) const {
    auto k_offset = config_.cache_config.getKeyOffset(block_index, layer_id);
    auto v_offset = config_.cache_config.getValueOffset(block_index, layer_id);
    auto k_addr   = (void*)((int8_t*)config_.k_cache->data() + k_offset);
    auto v_addr   = (void*)((int8_t*)config_.v_cache->data() + v_offset);
    return std::make_tuple(k_addr, v_addr);
}

int32_t ThreeFSFile::getMetaTotalLength(int32_t meta_count) const {
    return kReservedLength + sizeof(int32_t) + meta_count * sizeof(ThreeFSMeta);
}

int64_t ThreeFSFile::calcLeftSizeInBlock(int64_t iov_block_size, int64_t iov_offset) const {
    // 计算当前 iov block 块剩余可用的大小, 避免跨 block 读写
    const int64_t block_start        = (iov_offset / iov_block_size) * iov_block_size;
    const int64_t block_end          = block_start + iov_block_size;
    const int64_t left_size_in_block = block_end - iov_offset;
    return left_size_in_block;
}

std::string ThreeFSFile::getFilepath() const {
    return config_.mountpoint + config_.folder_name + config_.filename;
}

std::optional<int64_t> ThreeFSFile::getFileLength() const {
    const std::string filepath = getFilepath();
    struct stat       file_stat;
    if (auto ret = ::stat(filepath.c_str(), &file_stat); ret != 0) {
        RTP_LLM_LOG_WARNING("get file length failed, stat failed, file: %s, ret: %d, errno: %s",
                            filepath.c_str(),
                            ret,
                            strerror(errno));
        return std::nullopt;
    }
    return static_cast<int64_t>(file_stat.st_size);  // byte
}

std::optional<int64_t> ThreeFSFile::getFileCreateTimeInSecs() const {
    const std::string filepath = getFilepath();
    struct stat       file_stat;
    if (auto ret = ::stat(filepath.c_str(), &file_stat); ret != 0) {
        RTP_LLM_LOG_WARNING("get file create time failed, stat failed, file: %s, ret: %d, errno: %s",
                            filepath.c_str(),
                            ret,
                            strerror(errno));
        return std::nullopt;
    }

    // file_stat.st_mtime contains the time of last modification (as time_t)
    time_t last_mod_time = file_stat.st_mtime;

    // current time
    time_t current_time;
    time(&current_time);

    // calculate the difference in seconds
    double diff = difftime(current_time, last_mod_time);
    return static_cast<int64_t>(diff);
}

size_t ThreeFSFile::calcFileLength(int cache_key_count) const {
    auto all_block_len = cache_key_count * config_.cache_config.layer_num
                         * (config_.cache_config.k_block_stride + config_.cache_config.v_block_stride);
    auto total_len = getMetaTotalLength(cache_key_count) + all_block_len;
    return total_len;
}

bool ThreeFSFile::openFile(bool read) {
    const auto filename = config_.filename;
    if (fd_ != -1) {
        // file already opened
        RTP_LLM_LOG_DEBUG("file already opened, filename: %s", filename.c_str());
        return true;
    }

    // open file
    const std::string filepath = getFilepath();
    int               flags    = O_RDWR;
    if (!read) {
        flags |= O_CREAT;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    int        fd = -1;
    if (read) {
        // fd = getFdFromMap(filename);
    }
    if (fd == -1) {
        fd = ::open(filepath.c_str(), flags, 0666);
        if (fd == -1) {
            RTP_LLM_LOG_WARNING("open file failed, filepath: %s, fd: %d, errno: %s, read: %d",
                                filepath.c_str(),
                                fd,
                                strerror(errno),
                                read);
            return false;
        }
        if (read) {
            // addFdToMap(filename, fd);
        }
    }
    const auto t2 = std::chrono::high_resolution_clock::now();

    // note: must register fd after create ior/iov
    auto ret = hf3fs_reg_fd(fd, 0);
    if (ret > 0) {
        if (ret == EBADF) {
            // fd is bad, need reopen file
            RTP_LLM_LOG_DEBUG("hf3fs_reg_fd return EBADF, fd: %d, need reopen file: %s", fd, filepath.c_str());
            ::close(fd);

            fd = ::open(filepath.c_str(), flags, 0666);
            if (fd == -1) {
                RTP_LLM_LOG_WARNING("reopen file failed, filepath: %s, fd: %d, errno: %s, read: %d",
                                    filepath.c_str(),
                                    fd,
                                    strerror(errno),
                                    read);
                return false;
            }
            ret = hf3fs_reg_fd(fd, 0);
            if (ret > 0) {
                RTP_LLM_LOG_WARNING(
                    "open file failed, hf3fs_reg_fd failed after reopen file, file: %s, errno: %s, fd: %d",
                    filepath.c_str(),
                    strerror(ret),
                    fd);
                ::close(fd);
                return false;
            }
            removeFdInMap(filename);
            if (read) {
                // addFdToMap(filename, fd);
            }
        } else {
            RTP_LLM_LOG_WARNING("open file failed, hf3fs_reg_fd failed, errno: %s, file: %s, fd: %d, read: %d",
                                strerror(ret),
                                filepath.c_str(),
                                fd,
                                read);
            if (!read) {
                removeFile();
            }
            return false;
        }
    }
    if (read) {
        removeFdInMap(filename);
    }

    const auto t3           = std::chrono::high_resolution_clock::now();
    const auto open_cost_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const auto reg_fd_cost  = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    RTP_LLM_LOG_DEBUG("open file: %s, read: %d, open cost: %lu us, reg fd cost: %lu us",
                      filename.c_str(),
                      read,
                      open_cost_us,
                      reg_fd_cost);

    fd_ = fd;
    return true;
}

void ThreeFSFile::removeFile() {
    if (fd_ != -1) {
        hf3fs_dereg_fd(fd_);
        ::close(fd_);
        fd_ = -1;
    }
    removeFdInMap(config_.filename);
    const std::string filepath = getFilepath();
    RTP_LLM_LOG_DEBUG("remove 3fs file: %s", filepath.c_str());
    ::remove(filepath.c_str());
}

bool ThreeFSFile::fileExists() const {
    const auto  filepath = getFilepath();
    struct stat file_stat;
    return ::stat(filepath.c_str(), &file_stat) == 0;
}

int ThreeFSFile::getFdFromMap(const std::string& filename) const {
    std::unique_lock<std::shared_mutex> lock(filename_to_fd_map_mutex_);
    if (filename_to_fd_map_.count(filename) == 0) {
        return -1;
    }
    auto fd = filename_to_fd_map_.at(filename);
    if (fd == -1) {
        filename_to_fd_map_.erase(filename);
    }
    return fd;
}

void ThreeFSFile::addFdToMap(const std::string& filename, int fd) const {
    std::unique_lock<std::shared_mutex> lock(filename_to_fd_map_mutex_);
    if (filename_to_fd_map_.count(filename) != 0) {
        auto fd = filename_to_fd_map_.at(filename);
        if (fd != -1) {
            ::close(fd);
        }
        filename_to_fd_map_[filename] = fd;
        return;
    }
    static auto fd_max_num = getFdMaxNum();
    if (fd_max_num != -1 && filename_to_fd_map_.size() >= fd_max_num) {
        RTP_LLM_LOG_WARNING("current fd num exceed max num: %d", fd_max_num);
        auto it = filename_to_fd_map_.erase(filename_to_fd_map_.begin());
        auto fd = it->second;
        if (fd != -1) {
            // hf3fs_dereg_fd(fd);
            ::close(fd);
        }
    }
    filename_to_fd_map_[filename] = fd;
}

void ThreeFSFile::removeFdInMap(const std::string& filename) const {
    std::unique_lock<std::shared_mutex> lock(filename_to_fd_map_mutex_);
    if (filename_to_fd_map_.count(filename) != 0) {
        auto fd = filename_to_fd_map_.at(filename);
        if (fd != -1) {
            ::close(fd);
        }
        filename_to_fd_map_.erase(filename);
    }
}

int32_t ThreeFSFile::getFdMaxNum() const {
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == -1) {
        RTP_LLM_LOG_WARNING("get fd max num failed, getrlimit failed: %s", strerror(errno));
        return -1;
    }
    const auto threshold = 0.8;  // 限制最多使用 80% 的 fd
    return static_cast<int32_t>(static_cast<int32_t>(rl.rlim_cur) * threshold);
}

bool ThreeFSFile::verify(const std::shared_ptr<ThreeFSHandle>& handle,
                         const std::vector<int64_t>&           cache_keys,
                         const std::vector<int32_t>&           block_indices) {
    RTP_LLM_LOG_INFO("---------- 3fs read/write verify start ----------");
    // sleep for a while to wait for 3fs async write finish
    std::this_thread::sleep_for(std::chrono::seconds(3));

    if (!readMetas(handle)) {
        RTP_LLM_LOG_WARNING("verify failed, read metas failed, filename: %s", config_.filename.c_str());
        return false;
    }
    std::map<int64_t, int64_t> cache_key_map;
    {
        std::shared_lock<std::shared_mutex> lock(cache_key_map_mutex_);
        cache_key_map = cache_key_map_;
    }
    if (cache_key_map.size() != cache_keys.size()) {
        RTP_LLM_LOG_WARNING("verify failed, equal cache key count not equal, map count: %lu, cache key count: %lu",
                            cache_key_map.size(),
                            cache_keys.size());
        return false;
    }

    for (const auto cache_key : cache_keys) {
        if (cache_key_map.count(cache_key) == 0) {
            std::string read_cache_keys_str;
            for (const auto [key, offset] : cache_key_map) {
                read_cache_keys_str += std::to_string(key) + ",";
            }
            RTP_LLM_LOG_WARNING(
                "verify failed, cache key not found: %lu. write cache key: [%lu|%s], read cache key: [%lu|%s]",
                cache_key,
                cache_keys.size(),
                vectorToString(cache_keys).c_str(),
                cache_key_map.size(),
                read_cache_keys_str.c_str());
            return false;
        }
    }

    const int     cache_key_count = static_cast<int>(cache_keys.size());
    const int64_t block_len_per_cache_key =
        config_.cache_config.layer_num * (config_.cache_config.k_block_stride + config_.cache_config.v_block_stride);
    int64_t read_total_len = block_len_per_cache_key * cache_key_count;
    if (read_total_len > handle->iov_handle.iov_size) {
        RTP_LLM_LOG_WARNING("verify failed, read size exceed iov size, read size: %lu, iov size: %lu",
                            read_total_len,
                            handle->iov_handle.iov_size);
        return false;
    }

    {
        auto& ior      = handle->ior_handle.ior;
        auto& iov      = handle->iov_handle.iov;
        auto& iov_base = handle->iov_handle.iov_base;

        int64_t                    iov_offset      = 0;
        int32_t                    submit_io_count = 0;
        std::map<int64_t, int64_t> cache_key_2_iov_offset;
        for (int32_t cache_key_pos = 0; cache_key_pos < cache_key_count; ++cache_key_pos) {
            const auto cache_key              = cache_keys[cache_key_pos];
            auto       file_offset            = cache_key_map[cache_key];
            cache_key_2_iov_offset[cache_key] = iov_offset;

            const auto iov_block_size = handle->iov_handle.iov_block_size;
            auto       remaining_size = block_len_per_cache_key;
            while (remaining_size > 0) {
                uint64_t cur_read_len = 0;
                if (iov_block_size > 0) {
                    cur_read_len = std::min(remaining_size, calcLeftSizeInBlock(iov_block_size, iov_offset));
                } else {
                    cur_read_len = remaining_size > kDefaultReadSizePerIo ? kDefaultReadSizePerIo : remaining_size;
                }

                auto ret =
                    hf3fs_prep_io(ior, iov, true, iov_base + iov_offset, fd_, file_offset, cur_read_len, nullptr);
                if (ret < 0) {
                    // TODO(LXQ): 需要处理掉之前已提交的io请求(submit然后wait)
                    RTP_LLM_LOG_WARNING(
                        "verify failed, hf3fs_prep_io failed, errno: %s, cur_read_len: %lu, iov_block_size: %lu",
                        strerror(-ret),
                        cur_read_len,
                        iov_block_size);
                    return false;
                }
                ++submit_io_count;
                iov_offset += cur_read_len;
                file_offset += cur_read_len;
                remaining_size -= cur_read_len;

                if (submit_io_count < handle->ior_handle.ior_entries && cache_key_pos + 1 != cache_key_count) {
                    continue;
                }

                if (!submitAndWaitForReadIos(handle, submit_io_count)) {
                    RTP_LLM_LOG_WARNING(
                        "verify failed, read submit/wait io failed, submit io count: %d, cur_read_len: %lu, iov_block_size: %lu",
                        submit_io_count,
                        cur_read_len,
                        iov_block_size);
                    return false;
                }
                submit_io_count = 0;
            }
        }

        const auto k_block_len = config_.cache_config.k_block_stride;
        const auto v_block_len = config_.cache_config.v_block_stride;
        auto       k_buffer    = new char[k_block_len]();
        auto       v_buffer    = new char[v_block_len]();
        auto       k_shared_buffer =
            std::shared_ptr<void>(static_cast<void*>(k_buffer), [](void* p) { delete[] static_cast<char*>(p); });
        auto v_shared_buffer =
            std::shared_ptr<void>(static_cast<void*>(v_buffer), [](void* p) { delete[] static_cast<char*>(p); });
        auto cuda_util = handle->iov_handle.cuda_util;

        for (int i = 0; i < cache_key_count; ++i) {
            const auto cache_key   = cache_keys[i];
            const auto block_index = block_indices[i];
            auto       iov_offset  = cache_key_2_iov_offset[cache_key];
            for (int layer_index = 0; layer_index < config_.cache_config.layer_num; ++layer_index) {
                auto [k_addr, v_addr] = convertIndexToAddr(block_index, layer_index);

                // k cache
                cuda_util->copyAsyncDeviceToHost(k_buffer, k_addr, k_block_len);
                cuda_util->sync();
                if (std::memcmp(k_buffer, iov_base + iov_offset, k_block_len) != 0) {
                    RTP_LLM_LOG_WARNING("verify failed, k cache verify failed, layer index: %d, block index: %d",
                                        layer_index,
                                        block_index);
                    return false;
                }
                iov_offset += k_block_len;

                // v cache
                cuda_util->copyAsyncDeviceToHost(v_buffer, v_addr, v_block_len);
                cuda_util->sync();
                if (std::memcmp(v_buffer, iov_base + iov_offset, v_block_len) != 0) {
                    RTP_LLM_LOG_WARNING("verify failed, v cache verify failed, layer index: %d, block index: %d",
                                        layer_index,
                                        block_index);
                    return false;
                }
                iov_offset += v_block_len;
            }
        }
    }

    RTP_LLM_LOG_INFO("3fs read/write verify success");
    RTP_LLM_LOG_INFO("---------- 3fs read/write verify end ----------");
    return true;
}

void WaitIoWorkItem::process() {
    if (!threefs_file_) {
        if (last_io_) {
            ThreeFSMetrics::markWriteBlockDoneUs(metrics_);
        }
        RTP_LLM_LOG_WARNING("async wait write ios failed, threefs file is null");
        return;
    }
    threefs_file_->waitForWriteIos(handle_, submit_io_count_, write_len_, last_io_, metrics_);
    threefs_file_->releaseIovIor(handle_);
}

}  // namespace rtp_llm::threefs