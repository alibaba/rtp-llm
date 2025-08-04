#include "rtp_llm/cpp/cache/DistKvCachePlanner.h"

#include <filesystem>

#include "rtp_llm/cpp/cache/CacheManager.h"

namespace rtp_llm {

DefaultDistKvCachePlanner::DefaultDistKvCachePlanner(CacheManager*                       cache_manager,
                                                     const GptInitParameter&             gpt_init_params,
                                                     const DistStorage3FSInitParams&     init_params_3fs,
                                                     const kmonitor::MetricsReporterPtr& metrics_reporter):
    cache_manager_(cache_manager),
    gpt_init_params_(gpt_init_params),
    init_params_3fs_(init_params_3fs),
    metrics_reporter_(metrics_reporter) {}

std::vector<DistStorage::Item> DefaultDistKvCachePlanner::layout(const std::vector<int64_t>& cache_keys,
                                                                 const std::vector<int32_t>& block_indices,
                                                                 size_t                      ignore_block_num,
                                                                 const std::map<std::string, std::string>& metas,
                                                                 bool                                      skip_iov) {
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

    // layer_num * block_num * 2[k & v]
    item.iovs.reserve(cache_keys.size() * cache_config.layer_num * 2);

    for (int i = 0; i < cache_keys.size(); i++) {
        auto block_id = block_indices[i];
        bool ignore   = i < ignore_block_num;
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

std::optional<std::string>
DefaultDistKvCachePlanner::generateKvCacheKey(const std::map<std::string, std::string>& metas) const {
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

bool DefaultDistKvCachePlanner::verify(const std::vector<DistStorage::Item>&     items,
                                       const std::vector<int64_t>&               cache_keys,
                                       const std::vector<int32_t>&               block_indices,
                                       const std::map<std::string, std::string>& metas) {
    // TODO: need verify?
    return true;
}

}  // namespace rtp_llm