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

std::optional<std::string>
DistKvCachePlanner::generateKvCacheKey(const std::map<std::string, std::string>& metas) const {
    try {
        std::ostringstream oss;
        oss << metas.at("BIZ_NAME") << "_" << metas.at("CKPT_PATH") << "_" << metas.at("LORA_CKPT_PATH") << "_"
            << metas.at("SEQ_SIZE_PER_BLOCK") << "_" << metas.at("DTYPE") << "_" << metas.at("USE_MLA") << "_"
            << metas.at("TP_SIZE") << "_" << metas.at("TP_RANK") << "_" << metas.at("LAST_CACHE_KEY");
        return oss.str();
    } catch (const std::exception& e) {
        std::ostringstream oss;
        for (const auto& [key, value] : metas) {
            oss << key << ":" << value << ", ";
        }
        RTP_LLM_LOG_WARNING(
            "found exception when generate kvcache key. metas: [%s], exception: [%s]", oss.str().c_str(), e.what());
    }
    return std::nullopt;
}

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
                                                                 const std::map<std::string, std::string>& metas,
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
    if (cache_keys.empty()) {
        return {};
    }

    DistStorage::Item item;
    item.type                    = DistStorage::ST_3FS;
    item.metas                   = metas;
    item.metas["LAST_CACHE_KEY"] = std::to_string(cache_keys.back());

    const auto kvcache_key = generateKvCacheKey(item.metas);
    if (!kvcache_key.has_value()) {
        return {};
    }
    item.key = kvcache_key.value();

    if (skip_iov) {
        return {item};
    }

    const auto& cache_config = cache_manager_->cacheConfig();
    const auto  k_block_len  = cache_config.k_block_stride;
    const auto  v_block_len  = cache_config.v_block_stride;

    int32_t ignore_cache_key_num = 0;
    if (item.metas.count("GET") != 0 && item.metas.at("GET") == "1") {
        if (const auto it = item.metas.find("IGNORE_CACHE_KEY_NUM"); it != item.metas.end()) {
            ignore_cache_key_num = std::stoi(it->second);
        }
    }

    // layer_num * block_num * 2[k & v]
    item.iovs.reserve(cache_keys.size() * cache_config.layer_num * 2);

    for (int i = 0; i < cache_keys.size(); i++) {
        auto block_id = block_indices[i];
        bool ignore   = i < ignore_cache_key_num;
        for (int layer_id = 0; layer_id < cache_config.layer_num; layer_id++) {
            auto block_addrs = cache_manager_->convertIndexToAddr(block_id, layer_id);
            if (!block_addrs.k_addr || !block_addrs.v_addr) {
                return {};
            }
            item.iovs.push_back(
                DistStorage::Iov{std::shared_ptr<void>(block_addrs.k_addr, [](void* p) {}), k_block_len, true, ignore});
            item.iovs.push_back(
                DistStorage::Iov{std::shared_ptr<void>(block_addrs.v_addr, [](void* p) {}), v_block_len, true, ignore});
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

    auto        path = std::filesystem::path(storage_3fs_init_params_.mountpoint) / storage_3fs_init_params_.root_dir;
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