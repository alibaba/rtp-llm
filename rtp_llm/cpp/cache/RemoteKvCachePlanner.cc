#include "rtp_llm/cpp/cache/RemoteKvCachePlanner.h"

#include <filesystem>

#include "rtp_llm/cpp/cache/CacheManager.h"

namespace rtp_llm {

RemoteKvCachePlanner::RemoteKvCachePlanner(CacheManager*                       cache_manager,
                                           const GptInitParameter&             gpt_init_params,
                                           const kmonitor::MetricsReporterPtr& metrics_reporter):
    cache_manager_(cache_manager), gpt_init_params_(gpt_init_params), metrics_reporter_(metrics_reporter) {}

std::vector<DistStorage::Item> RemoteKvCachePlanner::layout(const std::vector<int64_t>&               cache_keys,
                                                            const std::vector<int32_t>&               block_indices,
                                                            const kv_cache_manager::BlockMask&        block_mask,
                                                            const std::map<std::string, std::string>& metas) {
    // TODO : refactor this
    uint32_t total_len = cache_keys.size();
    if (total_len == 0 || block_indices.empty()) {
        return {};
    }

    const auto& cache_config = cache_manager_->cacheConfig();
    const auto  k_block_len  = cache_config.k_block_stride;
    const auto  v_block_len  = cache_config.v_block_stride;

    std::shared_ptr<DistStorage::Item> item;
    std::vector<DistStorage::Item>     items;
    size_t                             block_i = 0;

    for (int i = 0; i < total_len; i++) {
        bool ignore = std::visit(
            [i](const auto& block_mask) -> bool {
                using T = std::decay_t<decltype(block_mask)>;
                if constexpr (std::is_same_v<kv_cache_manager::BlockMaskOffset, T>) {
                    return i < block_mask;
                } else if constexpr (std::is_same_v<kv_cache_manager::BlockMaskVector, T>) {
                    if (i >= block_mask.size()) {
                        return true;
                    }
                    return block_mask[i];
                }
                return true;
            },
            block_mask);

        if (ignore) {
            continue;
        }

        if (item == nullptr) {
            item       = std::make_shared<DistStorage::Item>();
            item->type = DistStorage::ST_REMOTE;
        }

        if (!block_indices.empty()) {
            for (int layer_id = 0; layer_id < cache_config.layer_num; layer_id++) {
                if (ignore) {
                    item->iovs.push_back(DistStorage::Iov{nullptr, k_block_len, false, ignore});
                    item->iovs.push_back(DistStorage::Iov{nullptr, v_block_len, false, ignore});
                } else {
                    if (block_i >= block_indices.size()) {
                        RTP_LLM_LOG_ERROR("block_i too large [%zu]", block_i);
                        return {};
                    }
                    auto block_id    = block_indices[block_i];
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

        if (!ignore) {
            ++block_i;
        }

        if (!item->iovs.empty()) {
            item->key = cache_keys[i];
            RTP_LLM_LOG_DEBUG("push item: %s", item->key.c_str());
            items.push_back(*item);
            item = nullptr;
        }
    }
    return items;
}

bool RemoteKvCachePlanner::verify(const std::vector<DistStorage::Item>&     items,
                                  const std::vector<int64_t>&               cache_keys,
                                  const std::vector<int32_t>&               block_indices,
                                  const std::map<std::string, std::string>& metas) {
    return true;
}

}  // namespace rtp_llm
