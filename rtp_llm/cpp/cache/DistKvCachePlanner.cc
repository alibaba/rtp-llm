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

std::vector<DistStorage::Item> DefaultDistKvCachePlanner::layout(const std::vector<size_t>&  cache_keys,
                                                                 const std::vector<int32_t>& block_indices,
                                                                 size_t                      ignore_block_num,
                                                                 const std::map<std::string, std::string>& metas) {
    uint32_t total_len = cache_keys.size();
    if (total_len == 0) {
        return {};
    }

    if (!block_indices.empty() && cache_keys.size() > block_indices.size() + ignore_block_num) {
        RTP_LLM_LOG_WARNING(
            "layout failed, cache key size or block size is invalid, cache key size: %zu, block size: %zu, ignore block num: %d",
            cache_keys.size(),
            block_indices.size(),
            ignore_block_num);
        return {};
    }

    const auto& cache_config = cache_manager_->cacheConfig();
    const auto  k_block_len  = cache_config.k_block_stride;
    const auto  v_block_len  = cache_config.v_block_stride;

    std::shared_ptr<DistStorage::Item> item;
    uint32_t                           item_block_count = 0;
    std::vector<size_t>                item_keys;
    std::vector<DistStorage::Item>     items;

    for (int i = 0; i < total_len; i++) {
        bool ignore = i < ignore_block_num;
        if (item == nullptr) {
            item             = std::make_shared<DistStorage::Item>();
            item->type       = DistStorage::ST_3FS;
            item->metas      = metas;
            item_block_count = 0;
            item_keys.clear();
        }

        if (!block_indices.empty()) {
            for (int layer_id = 0; layer_id < cache_config.layer_num; layer_id++) {
                if (ignore) {
                    item->iovs.push_back(DistStorage::Iov{nullptr, k_block_len, false, ignore});
                    item->iovs.push_back(DistStorage::Iov{nullptr, v_block_len, false, ignore});
                } else {
                    auto block_id    = block_indices.at(i - ignore_block_num);
                    auto block_addrs = cache_manager_->convertIndexToAddr(block_id, layer_id);
                    if (!block_addrs.k_addr || !block_addrs.v_addr) {
                        return {};
                    }
                    item->iovs.push_back(DistStorage::Iov{
                        std::shared_ptr<void>(block_addrs.k_addr, [](void* p) {}), k_block_len, true, ignore});
                    item->iovs.push_back(DistStorage::Iov{
                        std::shared_ptr<void>(block_addrs.v_addr, [](void* p) {}), v_block_len, true, ignore});
                }
            }
        }

        item_block_count++;
        item_keys.push_back(cache_keys[i]);

        if (item_block_count >= gpt_init_params_.kv_cache_config.max_block_size_per_item && item_keys.size() > 0) {
            item->key               = std::to_string(item_keys.front()) + "_" + std::to_string(item_keys.back());
            item->metas["ITEM_KEY"] = item->key;
            RTP_LLM_LOG_DEBUG("push item: %s", item->key.c_str());
            items.push_back(*item);
            item = nullptr;
        }
    }

    if (item != nullptr && item_keys.size() > 0) {
        item->key               = std::to_string(item_keys.front()) + "_" + std::to_string(item_keys.back());
        item->metas["ITEM_KEY"] = item->key;
        RTP_LLM_LOG_DEBUG("push item: %s", item->key.c_str());
        items.push_back(*item);
    }
    return items;
}

bool DefaultDistKvCachePlanner::verify(const std::vector<DistStorage::Item>&     items,
                                       const std::vector<size_t>&                cache_keys,
                                       const std::vector<int32_t>&               block_indices,
                                       const std::map<std::string, std::string>& metas) {
    // TODO: need verify?
    return true;
}

}  // namespace rtp_llm