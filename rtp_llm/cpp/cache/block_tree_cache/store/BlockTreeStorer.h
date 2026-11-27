#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheObserver.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

namespace rtp_llm {

class BlockTreeStorer {
public:
    // Invoked with the shared cache mutex held.
    using SettledFn = std::function<void(bool check_watermark)>;

    BlockTreeStorer(BlockTree*                      tree,
                    BlockTreeEvictor&               evictor,
                    CacheObserver&  metrics_reporter,
                    std::shared_ptr<StorageBackend> storage_backend,
                    SettledFn                       settled,
                    bool                            write_remote_on_publish = true);

    StorageWriteTask storeLocked(const CacheKeysType&                              cache_keys,
                                 const std::vector<std::vector<GroupSetResource>>& resources,
                                 Tier                                              target_tier,
                                 bool                                              is_resident,
                                 size_t&                                           resident_prefix_length,
                                 std::optional<int64_t>                            request_id         = {},
                                 size_t                                            remote_reuse_begin = 0,
                                 size_t                                            remote_reuse_end   = 0);
    StorageWriteTask storeLocked(const BlockTreeKeys& cache_keys,
                                 const std::vector<std::vector<GroupSetResource>>& resources,
                                 Tier target_tier,
                                 bool is_resident,
                                 size_t& resident_prefix_length,
                                 std::optional<int64_t> request_id,
                                 size_t remote_reuse_begin,
                                 size_t remote_reuse_end);

private:
    template<class Keys>
    StorageWriteTask storeLockedImpl(const Keys& cache_keys,
                                     const std::vector<std::vector<GroupSetResource>>& resources,
                                     Tier target_tier,
                                     bool is_resident,
                                     size_t& resident_prefix_length,
                                     std::optional<int64_t> request_id,
                                     size_t remote_reuse_begin,
                                     size_t remote_reuse_end);
    template<class Keys>
    void           publishLocalLocked(const Keys&                                        cache_keys,
                                       const std::vector<std::vector<GroupSetResource>>& resources,
                                       bool                                              is_resident,
                            size_t&                                           resident_prefix_length);
    template<class Keys>
    StorageRequest makeStorageRequest(const Keys&                                       cache_keys,
                                      const std::vector<std::vector<GroupSetResource>>& resources,
                                      const std::optional<int64_t>&                     request_id,
                                      size_t                                            remote_reuse_begin,
                                      size_t                                            remote_reuse_end) const;

    BlockTree*                      tree_;
    BlockTreeEvictor&               evictor_;
    CacheObserver&  metrics_reporter_;
    std::shared_ptr<StorageBackend> storage_backend_;
    SettledFn                       settled_;
    bool                            write_remote_on_publish_;
};

}  // namespace rtp_llm
