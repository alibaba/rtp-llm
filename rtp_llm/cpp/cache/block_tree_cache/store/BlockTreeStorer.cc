#include "rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h"

#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

BlockTreeStorer::BlockTreeStorer(BlockTree*                      tree,
                                 BlockTreeEvictor&               evictor,
                                 BlockTreeCacheMetricsReporter&  metrics_reporter,
                                 std::shared_ptr<StorageBackend> storage_backend,
                                 SettledFn                       settled):
    tree_(tree),
    evictor_(evictor),
    metrics_reporter_(metrics_reporter),
    storage_backend_(std::move(storage_backend)),
    settled_(std::move(settled)) {}

StorageWriteTask BlockTreeStorer::storeLocked(const CacheKeysType&                              cache_keys,
                                              const std::vector<std::vector<GroupSetResource>>& resources,
                                              Tier                                              target_tier,
                                              bool                                              is_resident,
                                              size_t&                                           resident_prefix_length,
                                              std::optional<int64_t>                            request_id,
                                              size_t                                            remote_reuse_begin,
                                              size_t                                            remote_reuse_end) {
    resident_prefix_length = 0;
    RTP_LLM_CHECK_WITH_INFO(target_tier == Tier::DEVICE, "unsupported store target tier: %s", tierName(target_tier));
    resident_prefix_length = publishDeviceLocked(cache_keys, resources, is_resident);
    return storage_backend_ ? storage_backend_->prepareWrite(makeStorageRequest(
                                  cache_keys, resources, request_id, remote_reuse_begin, remote_reuse_end)) :
                              StorageWriteTask{};
}

size_t BlockTreeStorer::publishDeviceLocked(const CacheKeysType&                              cache_keys,
                                            const std::vector<std::vector<GroupSetResource>>& resources,
                                            bool                                              is_resident,
                                            bool                                              is_writeback) {
    const BlockTreeInsertResult insert_result =
        tree_->insertNode(cache_keys, resources, false, is_resident, is_writeback);
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
    return insert_result.resident_prefix_length;
}

StorageRequest BlockTreeStorer::makeStorageRequest(const CacheKeysType&                              cache_keys,
                                                   const std::vector<std::vector<GroupSetResource>>& resources,
                                                   const std::optional<int64_t>&                     request_id,
                                                   size_t                                            remote_reuse_begin,
                                                   size_t remote_reuse_end) const {
    RTP_LLM_CHECK(remote_reuse_begin <= remote_reuse_end && remote_reuse_end <= cache_keys.size());
    const size_t key_count = cache_keys.size() - (remote_reuse_end - remote_reuse_begin);
    auto         keys      = std::make_shared<CacheKeysType>();
    keys->reserve(key_count);
    keys->insert(keys->end(), cache_keys.begin(), cache_keys.begin() + remote_reuse_begin);
    keys->insert(keys->end(), cache_keys.begin() + remote_reuse_end, cache_keys.end());
    StorageRequest request{std::move(keys), std::vector<std::vector<StorageBlockHandle>>(key_count)};
    request.request_id = request_id;
    size_t output      = 0;
    for (size_t key_index = 0; key_index < resources.size(); ++key_index) {
        if (key_index >= remote_reuse_begin && key_index < remote_reuse_end) {
            continue;
        }
        request.handles[output++] = deviceHandles(resources[key_index]);
    }
    return request;
}

StorageWriteTask BlockTreeStorer::preparePublishedWriteLocked(const CacheKeysType&          keys,
                                                              const std::vector<TreeNode*>& path) const {
    if (!storage_backend_) {
        return {};
    }
    StorageRequest request;
    request.keys = std::make_shared<CacheKeysType>(keys);
    request.handles.reserve(path.size());
    for (const auto* node : path) {
        request.handles.push_back(deviceHandles(node->group_set_resources));
    }
    return storage_backend_->prepareWrite(std::move(request));
}

std::vector<StorageBlockHandle> BlockTreeStorer::deviceHandles(const std::vector<GroupSetResource>& groups) const {
    std::vector<StorageBlockHandle> handles;
    for (size_t gid = 0; gid < tree_->groupSets().size(); ++gid) {
        const auto& resource = groups[gid];
        if (!resource.hasCompleteDeviceValue()) {
            continue;
        }
        const auto& tags = tree_->groupSets()[gid]->groupTags();
        for (size_t i = 0; i < tags.size(); ++i) {
            handles.push_back({tags[i], resource.device_blocks[i]});
        }
    }
    return handles;
}

}  // namespace rtp_llm
