#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/CacheDiagnostics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h"

#include <utility>
#include <type_traits>


namespace rtp_llm {

BlockTreeStorer::BlockTreeStorer(BlockTree*                      tree,
                                 BlockTreeEvictor&               evictor,
                                 CacheObserver&  metrics_reporter,
                                 std::shared_ptr<StorageBackend> storage_backend,
                                 SettledFn                       settled,
                                 bool                            write_remote_on_publish):
    tree_(tree),
    evictor_(evictor),
    metrics_reporter_(metrics_reporter),
    storage_backend_(std::move(storage_backend)),
    settled_(std::move(settled)),
    write_remote_on_publish_(write_remote_on_publish) {}

StorageWriteTask BlockTreeStorer::storeLocked(const CacheKeysType&                              cache_keys,
                                              const std::vector<std::vector<GroupSetResource>>& resources,
                                              Tier                                              target_tier,
                                              bool                                              is_resident,
                                              size_t&                                           resident_prefix_length,
                                              std::optional<int64_t>                            request_id,
                                              size_t                                            remote_reuse_begin,
                                              size_t                                            remote_reuse_end) {
    return storeLockedImpl(cache_keys, resources, target_tier, is_resident, resident_prefix_length,
                           request_id, remote_reuse_begin, remote_reuse_end);
}

StorageWriteTask BlockTreeStorer::storeLocked(const BlockTreeKeys& cache_keys,
                                              const std::vector<std::vector<GroupSetResource>>& resources,
                                              Tier target_tier,
                                              bool is_resident,
                                              size_t& resident_prefix_length,
                                              std::optional<int64_t> request_id,
                                              size_t remote_reuse_begin,
                                              size_t remote_reuse_end) {
    return storeLockedImpl(cache_keys, resources, target_tier, is_resident, resident_prefix_length,
                           request_id, remote_reuse_begin, remote_reuse_end);
}

template<class Keys>
StorageWriteTask BlockTreeStorer::storeLockedImpl(const Keys& cache_keys,
                                                  const std::vector<std::vector<GroupSetResource>>& resources,
                                                  Tier target_tier,
                                                  bool is_resident,
                                                  size_t& resident_prefix_length,
                                                  std::optional<int64_t> request_id,
                                                  size_t remote_reuse_begin,
                                                  size_t remote_reuse_end) {
    resident_prefix_length = 0;
    BTC_CHECK_WITH_INFO(target_tier == Tier::DEVICE || (std::is_same_v<Keys, BlockTreeKeys> && target_tier == Tier::HOST),
                        "unsupported store target tier: %s", tierName(target_tier));
    publishLocalLocked(cache_keys, resources, is_resident, resident_prefix_length);
    return write_remote_on_publish_ && storage_backend_ && target_tier == Tier::DEVICE
               && storage_backend_->supportsWriteTier(Tier::DEVICE) ? storage_backend_->prepareWrite(makeStorageRequest(
                                  cache_keys, resources, request_id, remote_reuse_begin, remote_reuse_end)) :
                              StorageWriteTask{};
}

template<class Keys>
void BlockTreeStorer::publishLocalLocked(const Keys&                                        cache_keys,
                                          const std::vector<std::vector<GroupSetResource>>& resources,
                                          bool                                              is_resident,
                                          size_t&                                           resident_prefix_length) {
    const BlockTreeInsertResult insert_result = tree_->insertNode(cache_keys, resources, false, is_resident);
    if (insert_result.hard_stop) {
        metrics_reporter_.reportInsertHardStop();
    }
    const bool resources_inserted = !insert_result.inserted_nodes.empty() || !insert_result.adopted_nodes.empty();
    if (resources_inserted || !insert_result.newly_resident_nodes.empty()) {
        evictor_.onInserted(insert_result);
    }
    if (resources_inserted) {
        settled_(true);
    }
    resident_prefix_length = insert_result.resident_prefix_length;
}

template<class Keys>
StorageRequest BlockTreeStorer::makeStorageRequest(const Keys&                                       cache_keys,
                                                   const std::vector<std::vector<GroupSetResource>>& resources,
                                                   const std::optional<int64_t>&                     request_id,
                                                   size_t                                            remote_reuse_begin,
                                                   size_t remote_reuse_end) const {
    BTC_CHECK(remote_reuse_begin <= remote_reuse_end && remote_reuse_end <= cache_keys.size());
    const size_t key_count = cache_keys.size() - (remote_reuse_end - remote_reuse_begin);
    auto         keys      = std::make_shared<Keys>();
    keys->reserve(key_count);
    keys->insert(keys->end(), cache_keys.begin(), cache_keys.begin() + remote_reuse_begin);
    keys->insert(keys->end(), cache_keys.begin() + remote_reuse_end, cache_keys.end());
    StorageRequest request;
    if constexpr (std::is_same_v<Keys, CacheKeysType>) {
        request.keys = std::move(keys);
    } else {
        request.opaque_keys = std::move(keys);
    }
    request.handles.resize(key_count);
    request.request_id = request_id;
    size_t output      = 0;
    for (size_t key_index = 0; key_index < resources.size(); ++key_index) {
        if (key_index >= remote_reuse_begin && key_index < remote_reuse_end) {
            continue;
        }
        auto& key_handles = request.handles[output++];
        for (size_t group_set = 0; group_set < tree_->groupSets().size(); ++group_set) {
            const auto& resource = resources[key_index][group_set];
            if (!resource.hasCompleteDeviceValue()) {
                continue;
            }
            const auto& group = *tree_->groupSets()[group_set];
            for (size_t member = 0; member < group.groupTags().size(); ++member) {
                key_handles.push_back({group.groupTags()[member], resource.device_blocks[member]});
            }
        }
    }
    return request;
}

}  // namespace rtp_llm
