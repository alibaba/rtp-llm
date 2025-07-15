#include "rtp_llm/cpp/cache/DistKvCachePlanner.h"

#include <filesystem>

#include "rtp_llm/cpp/cache/CacheManager.h"

namespace rtp_llm {

const int32_t kReservedLength = 1024;  // 1KB

// 以下与文件存储格式有关, 谨慎修改!
#pragma pack(push, 1)
struct DefaultDistKvCacheMetaInfo {
    int64_t cache_key{0};
    int64_t offset{0};
};

struct DefaultDistKvCacheMeta {
    char     reserved[kReservedLength]{};
    uint32_t meta_count{0};
};
#pragma pack(pop)

DefaultDistKvCachePlanner::DefaultDistKvCachePlanner(CacheManager*                       cache_manager,
                                                     const GptInitParameter&             gpt_init_params,
                                                     const DistStorage3FSInitParams&     storage_3fs_init_params,
                                                     const kmonitor::MetricsReporterPtr& metrics_reporter):
    cache_manager_(cache_manager),
    gpt_init_params_(gpt_init_params),
    storage_3fs_init_params_(storage_3fs_init_params),
    metrics_reporter_(metrics_reporter) {}

std::vector<DistStorage::Item> DefaultDistKvCachePlanner::layout(const std::vector<int64_t>& cache_keys,
                                                                 const std::vector<int32_t>& block_indices,
                                                                 const std::map<std::string, std::string>& metas) {
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
    DistStorage::Item item;
    item.type = DistStorage::ST_3FS;

    if (metas.count("TP_RANK") == 0) {
        RTP_LLM_LOG_WARNING("cache planner metas missing TP_RANK");
        return {};
    }
    const int32_t tp_rank = std::stoi(metas.at("TP_RANK"));

    // use last cache key construct item key
    item.key = constructKvCacheKey(cache_keys[cache_keys.size() - 1], tp_rank);

    if (metas.count("SKIP_IOV") != 0 && metas.at("SKIP_IOV") == "1") {
        return {item};
    }

    const auto& cache_config = cache_manager_->cacheConfig();
    const auto  k_block_len  = cache_config.k_block_stride;
    const auto  v_block_len  = cache_config.v_block_stride;

    int64_t file_offset = 0;
    if (metas.count("GET") != 0 && metas.at("GET") == "1") {
        if (metas.count("SEQ_CACHE_KEY_NUM") == 0) {
            RTP_LLM_LOG_WARNING("cache planner metas missing SEQ_CACHE_KEY_NUM when get");
            return {};
        }
        const auto seq_cache_key_num = std::stoi(metas.at("SEQ_CACHE_KEY_NUM"));
        file_offset = (seq_cache_key_num - cache_keys.size()) * cache_config.layer_num * (k_block_len + v_block_len);
    }

    // layer_num * block_num * 2[k & v]
    item.iovs.reserve(cache_keys.size() * cache_config.layer_num * 2);

    for (int i = 0; i < cache_keys.size(); i++) {
        auto block_id = block_indices[i];
        for (int layer_id = 0; layer_id < cache_config.layer_num; layer_id++) {
            auto block_addrs = cache_manager_->convertIndexToAddr(block_id, layer_id);
            if (!block_addrs.k_addr || !block_addrs.v_addr) {
                return {};
            }
            item.iovs.push_back(DistStorage::Iov{
                std::shared_ptr<void>(block_addrs.k_addr, [](void* p) {}), k_block_len, file_offset, true});
            file_offset += k_block_len;
            item.iovs.push_back(DistStorage::Iov{
                std::shared_ptr<void>(block_addrs.v_addr, [](void* p) {}), v_block_len, file_offset, true});
            file_offset += v_block_len;
        }
    }

    return {item};
}

std::string DefaultDistKvCachePlanner::constructKvCacheKey(int64_t last_cache_key, int32_t rank) const {
    // 3fs use the following string as filename:
    // <path>/kv_<model_name>_<layer_num>_<local_head_num_kv>_<size_per_head>_<seq_size_per_block>_<dtype>_<last_cache_key>_<rank>

    if (rank == -1) {
        rank = static_cast<int32_t>(gpt_init_params_.tp_rank_);
    }

    auto path = std::filesystem::path(storage_3fs_init_params_.mountpoint) / storage_3fs_init_params_.folder_name;
    const auto& cache_config = cache_manager_->cacheConfig();

    std::ostringstream oss;
    oss << "kv_" << gpt_init_params_.model_name_ << "_" << cache_config.layer_num << "_"
        << cache_config.local_head_num_kv << "_" << cache_config.size_per_head << "_" << cache_config.seq_size_per_block
        << "_" << static_cast<int>(cache_config.dtype) << "_" << last_cache_key << "_" << rank;

    path /= oss.str();
    return path.lexically_normal().string();
}

bool DefaultDistKvCachePlanner::makeMetaIov(DistStorage::Iov&           meta_iov,
                                            const std::vector<int64_t>& cache_keys,
                                            const std::vector<int32_t>& block_indices,
                                            int32_t                     tp_rank) {
    const auto   cache_key_count = static_cast<int32_t>(cache_keys.size());
    const size_t meta_iov_len =
        kReservedLength + sizeof(int32_t) + cache_key_count * sizeof(DefaultDistKvCacheMetaInfo);
    void* meta_iov_buffer = malloc(meta_iov_len);
    if (!meta_iov_buffer) {
        RTP_LLM_LOG_WARNING("default dist cache planner malloc meta iov len %d failed", meta_iov_len);
        return false;
    }

    // copy reserved len and cache key count
    struct DefaultDistKvCacheMeta meta;
    meta.meta_count = cache_key_count;
    memcpy(meta_iov_buffer, &meta, sizeof(meta));

    const auto& cache_config = cache_manager_->cacheConfig();
    const auto  k_block_len  = cache_config.k_block_stride;
    const auto  v_block_len  = cache_config.v_block_stride;

    // file offset of kvcache
    std::vector<DefaultDistKvCacheMetaInfo> meta_infos(cache_key_count);
    int64_t                                 cache_offset = meta_iov_len;
    for (int i = 0; i < cache_key_count; ++i) {
        meta_infos[i].cache_key = cache_keys[i];
        meta_infos[i].offset    = cache_offset;
        cache_offset += cache_config.layer_num * (k_block_len + v_block_len);
    }

    // copy cache keys
    std::memcpy(static_cast<char*>(meta_iov_buffer) + sizeof(meta),
                meta_infos.data(),
                cache_key_count * sizeof(DefaultDistKvCacheMetaInfo));

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
    return true;
}

}  // namespace rtp_llm