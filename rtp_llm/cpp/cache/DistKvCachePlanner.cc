#include "rtp_llm/cpp/cache/DistKvCachePlanner.h"

#include "rtp_llm/cpp/cache/CacheManager.h"

namespace rtp_llm {

const int32_t kReservedLength = 1024;  // 1KB

// DefaultDistKvCacheMeta 与 block 文件存储格式有关, 谨慎修改!
#pragma pack(push, 1)
struct DefaultDistKvCacheMetaInfo {
    int64_t cache_key{0};
    int64_t offset{0};
};
#pragma pack(pop)

struct DefaultDistKvCacheMeta {
    char     reserved[kReservedLength];
    uint32_t meta_count{0};
};

DefaultDistKvCachePlanner::DefaultDistKvCachePlanner(CacheManager*                       cache_manager,
                                                     const GptInitParameter&             params,
                                                     const kmonitor::MetricsReporterPtr& metrics_reporter):
    cache_manager_(cache_manager), params_(params), metrics_reporter_(metrics_reporter) {}

std::vector<DistStorage::Item> DefaultDistKvCachePlanner::layout(const std::vector<int64_t>& cache_keys,
                                                                 const std::vector<int32_t>& block_indices,
                                                                 const std::map<std::string, std::string>& metas,
                                                                 int32_t                                   tp_rank,
                                                                 bool                                      skip_iov) {
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
    const auto& cache_config = cache_manager_->cacheConfig();

    DistStorage::Item item;

    // use last cache key construct item key
    item.key = constructKvCacheKey(cache_keys[cache_keys.size() - 1], tp_rank);

    if (skip_iov) {
        return {item};
    }

    // meta_iov + layer_num * block_num * 2[k & v]
    item.iovs.reserve(1 + cache_keys.size() * cache_config.layer_num * 2);

    DistStorage::Iov meta_iov;
    if (!makeMetaIov(meta_iov, cache_keys, block_indices, tp_rank)) {
        return {};
    }
    item.iovs.push_back(meta_iov);

    const auto k_block_len = cache_config.k_block_stride;
    const auto v_block_len = cache_config.v_block_stride;

    for (int i = 0; i < cache_keys.size(); i++) {
        auto block_id = block_indices[i];
        for (int layer_id = 0; layer_id < cache_config.layer_num; layer_id++) {
            auto block_addrs = cache_manager_->convertIndexToAddr(block_id, layer_id);
            if (!block_addrs.k_addr || !block_addrs.v_addr) {
                return {};
            }
            item.iovs.push_back(
                DistStorage::Iov{std::shared_ptr<void>(block_addrs.k_addr, [](void* p) {}), k_block_len, true});
            item.iovs.push_back(
                DistStorage::Iov{std::shared_ptr<void>(block_addrs.v_addr, [](void* p) {}), v_block_len, true});
        }
    }
    return {item};
}

std::string DefaultDistKvCachePlanner::constructKvCacheKey(int64_t last_cache_key, int32_t rank) const {
    // 3fs use the following string as filename:
    // kv_<model_name>_<layer_num>_<local_head_num_kv>_<size_per_head>_<seq_size_per_block>_<dtype>_<last_cache_key>_<rank>
    if (rank == -1) {
        rank = static_cast<int32_t>(params_.tp_rank_);
    }

    const auto& cache_config = cache_manager_->cacheConfig();

    std::ostringstream oss;
    oss << "default_kv_" << params_.model_name_ << "_" << cache_config.layer_num << "_"
        << cache_config.local_head_num_kv << "_" << cache_config.size_per_head << "_" << cache_config.seq_size_per_block
        << "_" << static_cast<int>(cache_config.dtype) << "_" << last_cache_key << "_" << rank;
    return oss.str();
}

bool DefaultDistKvCachePlanner::makeMetaIov(DistStorage::Iov&           meta_iov,
                                            const std::vector<int64_t>& cache_keys,
                                            const std::vector<int32_t>& block_indices,
                                            int32_t                     tp_rank) {
    const auto& cache_config = cache_manager_->cacheConfig();

    const auto k_block_len = cache_config.k_block_stride;
    const auto v_block_len = cache_config.v_block_stride;

    const size_t meta_iov_len =
        kReservedLength + sizeof(int32_t) + cache_keys.size() * sizeof(DefaultDistKvCacheMetaInfo);
    void* meta_iov_buffer = malloc(meta_iov_len);
    if (!meta_iov_buffer) {
        RTP_LLM_LOG_WARNING("default dist cache planner malloc meta iov len %d failed", meta_iov_len);
        return false;
    }
    struct DefaultDistKvCacheMeta meta;
    meta.meta_count = cache_keys.size();
    memcpy(meta_iov_buffer, &meta, sizeof(meta));

    int32_t offset = sizeof(meta);
    for (int i = 0; i < cache_keys.size(); i++) {
        DefaultDistKvCacheMetaInfo meta_info;
        meta_info.cache_key = cache_keys[i];
        meta_info.offset    = offset;
        memcpy(static_cast<char*>(meta_iov_buffer) + offset, &meta_info, sizeof(meta_info));
        offset += sizeof(meta_info);
    }

    meta_iov.data    = std::shared_ptr<void>(meta_iov_buffer, [](void* p) { free(p); });
    meta_iov.len     = meta_iov_len;
    meta_iov.gpu_mem = false;

    return true;
}

bool DefaultDistKvCachePlanner::verify(const std::vector<DistStorage::Item>&     items,
                                       const std::vector<int64_t>&               cache_keys,
                                       const std::vector<int32_t>&               block_indices,
                                       const std::map<std::string, std::string>& metas,
                                       int32_t                                   tp_rank) {
    // TODO: need verify?
    return false;
}

}  // namespace rtp_llm