#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <dirent.h>
#include <unistd.h>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/KVCacheGroup.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::test {
namespace tier_cache_test_detail {

using block_tree_cache_test::BlockTreeCacheTestPeer;
using block_tree_cache_test::ScriptedPerRankBlockTransferEngine;

inline constexpr int                  kDsv4GroupCount = 7;
inline constexpr std::chrono::seconds kTransferWaitTimeout{30};
inline const std::vector<std::string> kDsv4Tags = {
    "swa_kv", "csa_kv", "indexer_kv", "indexer_state", "csa_state", "hca_kv", "hca_state"};
inline const std::vector<CacheGroupType> kDsv4Types = {
    CacheGroupType::SWA,
    CacheGroupType::FULL,
    CacheGroupType::FULL,
    CacheGroupType::SWA,
    CacheGroupType::SWA,
    CacheGroupType::FULL,
    CacheGroupType::SWA,
};
inline const std::vector<size_t> kDsv4SwaGroupIds  = {0, 3, 4};
inline const std::vector<size_t> kDsv4FullGroupIds = {1, 2, 5};

enum class TierLayout {
    HOST_ONLY,
    HOST_DISK,
};

enum class LoadFailureSource {
    HOST,
    DISK,
};

inline const char* layoutName(TierLayout layout) {
    return layout == TierLayout::HOST_DISK ? "HostDisk" : "HostOnly";
}

class ScopedTierDiskDirectory {
public:
    ScopedTierDiskDirectory() {
        std::string       pattern = "/tmp/kv_cache_manager_with_tier_cache_XXXXXX";
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        if (char* result = ::mkdtemp(writable.data()); result != nullptr) {
            path_ = result;
        }
    }

    ~ScopedTierDiskDirectory() {
        std::string ignored;
        cleanup(&ignored);
    }

    const std::string& path() const {
        return path_;
    }

    bool cleanup(std::string* error) {
        if (cleaned_) {
            return true;
        }
        if (path_.empty()) {
            if (error != nullptr) {
                *error = "mkdtemp failed";
            }
            return false;
        }

        const std::string        work_dir = path_ + "/rtp_llm_disk_kv";
        std::vector<std::string> files;
        errno = 0;
        if (DIR* dir = ::opendir(work_dir.c_str()); dir != nullptr) {
            while (auto* entry = ::readdir(dir)) {
                const std::string name = entry->d_name;
                if (name == "." || name == "..") {
                    continue;
                }
                const bool known_file = name == ".lock"
                                        || (name.rfind("disk_block_pool_", 0) == 0 && name.size() > 4
                                            && name.substr(name.size() - 4) == ".bin");
                if (!known_file) {
                    ::closedir(dir);
                    if (error != nullptr) {
                        *error = "unexpected entry in test disk directory: " + name;
                    }
                    return false;
                }
                files.push_back(work_dir + "/" + name);
            }
            if (::closedir(dir) != 0) {
                if (error != nullptr) {
                    *error = "closedir failed: " + std::string(std::strerror(errno));
                }
                return false;
            }
        } else if (errno != ENOENT) {
            if (error != nullptr) {
                *error = "opendir failed: " + std::string(std::strerror(errno));
            }
            return false;
        }

        for (const auto& file : files) {
            if (::unlink(file.c_str()) != 0) {
                if (error != nullptr) {
                    *error = "unlink failed for " + file + ": " + std::string(std::strerror(errno));
                }
                return false;
            }
        }
        if (::rmdir(work_dir.c_str()) != 0 && errno != ENOENT) {
            if (error != nullptr) {
                *error = "rmdir work directory failed: " + std::string(std::strerror(errno));
            }
            return false;
        }
        if (::rmdir(path_.c_str()) != 0) {
            if (error != nullptr) {
                *error = "rmdir root failed: " + std::string(std::strerror(errno));
            }
            return false;
        }
        cleaned_ = true;
        return true;
    }

private:
    std::string path_;
    bool        cleaned_{false};
};

class PausableRecordingTransferEngine: public PerRankBlockTransferEngine {
public:
    explicit PausableRecordingTransferEngine(const std::vector<GroupSetPtr>& groups,
                                             size_t                          device_disk_staging_block_count = 4):
        PerRankBlockTransferEngine(groups, {}, device_disk_staging_block_count) {}

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        bool scripted_success = true;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++submitted_batch_count_;
            descriptors_.insert(descriptors_.end(), descriptors.begin(), descriptors.end());
            if (!scripted_results_.empty()) {
                scripted_success = scripted_results_.front();
                scripted_results_.pop_front();
            }
            if (pause_armed_) {
                ++phase_entered_;
                auto context = std::make_shared<TransferBatchAsyncContext>();
                pending_.push_back({descriptors, scripted_success, context});
                cv_.notify_all();
                return context;
            }
        }
        if (!scripted_success) {
            return std::make_shared<CompletedAsyncContext>(
                ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "scripted transfer failure"));
        }
        return PerRankBlockTransferEngine::submit(descriptors);
    }

    void enqueueResult(bool success) {
        std::lock_guard<std::mutex> lock(mutex_);
        scripted_results_.push_back(success);
    }

    void clearScriptedResults() {
        std::lock_guard<std::mutex> lock(mutex_);
        scripted_results_.clear();
    }

    bool armPause() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pause_armed_ && !phase_released_) {
            return false;
        }
        pause_armed_    = true;
        phase_released_ = false;
        phase_entered_  = 0;
        return true;
    }

    bool waitUntilEnteredFor(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] { return phase_entered_ > 0; });
    }

    bool waitUntilEnteredCountFor(size_t count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this, count] { return phase_entered_ >= count; });
    }

    bool waitUntilSubmittedDescriptorCountFor(size_t count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this, count] { return descriptors_.size() >= count; });
    }

    void release() {
        std::vector<PendingSubmit> pending;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            phase_released_ = true;
            pause_armed_    = false;
            pending.swap(pending_);
            cv_.notify_all();
        }
        for (auto& submit : pending) {
            if (!submit.success) {
                submit.context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "scripted transfer failure"));
                continue;
            }
            auto context = PerRankBlockTransferEngine::submit(submit.descriptors);
            context->waitDone();
            submit.context->complete(context->errorInfo());
        }
    }

    std::vector<TransferDescriptor> descriptors() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return descriptors_;
    }

    size_t submittedBatchCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return submitted_batch_count_;
    }

    size_t submittedDescriptorCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return descriptors_.size();
    }

private:
    struct PendingSubmit {
        std::vector<TransferDescriptor>            descriptors;
        bool                                       success;
        std::shared_ptr<TransferBatchAsyncContext> context;
    };

    mutable std::mutex              mutex_;
    std::condition_variable         cv_;
    bool                            pause_armed_{false};
    bool                            phase_released_{true};
    size_t                          phase_entered_{0};
    std::vector<TransferDescriptor> descriptors_;
    size_t                          submitted_batch_count_{0};
    std::deque<bool>                scripted_results_;
    std::vector<PendingSubmit>      pending_;
};

class ScopedTransferRelease {
public:
    explicit ScopedTransferRelease(std::shared_ptr<PausableRecordingTransferEngine> engine):
        engine_(std::move(engine)) {}

    ~ScopedTransferRelease() {
        if (engine_ != nullptr) {
            engine_->release();
        }
    }

private:
    std::shared_ptr<PausableRecordingTransferEngine> engine_;
};

inline bool waitForAsyncContextDoneFor(const std::shared_ptr<AsyncContext>& context,
                                       std::chrono::milliseconds            timeout) {
    if (context == nullptr) {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!context->done() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return context->done();
}

inline bool waitForPendingTasksDoneFor(const BlockTreeCache& cache, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (BlockTreeCacheTestPeer::pendingTasksForTest(cache) != 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return BlockTreeCacheTestPeer::pendingTasksForTest(cache) == 0;
}

inline ModelConfig makeCompactDsv4ModelConfig() {
    ModelConfig config;
    config.num_layers                                                = 5;
    config.hidden_size                                               = 64;
    config.attn_config.head_num                                      = 4;
    config.attn_config.kv_head_num                                   = 1;
    config.attn_config.size_per_head                                 = 16;
    config.attn_config.rope_head_dim                                 = 4;
    config.attn_config.sliding_window                                = 128;
    config.attn_config.indexer_head_dim                              = 8;
    config.attn_config.indexer_head_num                              = 4;
    config.attn_config.indexer_topk                                  = 16;
    config.attn_config.o_groups                                      = 2;
    config.attn_config.o_lora_rank                                   = 8;
    config.attn_config.tokens_per_block                              = 128;
    config.attn_config.layer_compress_ratios                         = {0, 4, 128, 4, 0};
    config.hybrid_attention_config.enable_hybrid_attention           = true;
    config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    setDsv4KvCacheSpecs(config, config.attn_config.layer_compress_ratios);
    setDsv4ExplicitPoolBlocks(config, "hca_state", 0);
    return config;
}

inline void setGroupBlockNums(CacheConfig& config, uint32_t block_num) {
    std::vector<uint32_t> block_nums(static_cast<size_t>(config.groupNums()), block_num);
    std::vector<size_t>   kv_strides;
    std::vector<size_t>   scale_strides;
    kv_strides.reserve(block_nums.size());
    scale_strides.reserve(block_nums.size());
    for (size_t group_id = 0; group_id < block_nums.size(); ++group_id) {
        kv_strides.push_back(config.kvBlockStrideBytesForGroup(group_id));
        scale_strides.push_back(config.kvScaleStrideBytesForGroup(group_id));
    }
    config.setGroupBlockLayout(std::move(block_nums), std::move(kv_strides), std::move(scale_strides));
}

inline CacheConfig makeCompactDsv4CacheConfig(uint32_t block_num) {
    ParallelismConfig parallelism;
    auto              config = CacheConfigCreator::createBasicConfig(makeCompactDsv4ModelConfig(),
                                                        parallelism,
                                                        /*is_mtp=*/false,
                                                        /*mtp_module_num=*/0);
    config.block_num         = block_num;
    config.linear_step       = 1;
    setGroupBlockNums(config, block_num);
    return config;
}

inline KVCacheConfig makeTierConfig(TierLayout layout, const std::string& disk_path, int64_t lower_cache_size_mb = 8) {
    KVCacheConfig config;
    config.reuse_cache                           = true;
    config.reserve_block_ratio                   = 0;
    config.enable_device_cache                   = true;
    config.enable_host_cache                     = true;
    config.host_cache_size_mb                    = lower_cache_size_mb;
    config.host_cache_sync_timeout_ms            = 5000;
    config.enable_disk_cache                     = layout == TierLayout::HOST_DISK;
    config.disk_cache_paths                      = layout == TierLayout::HOST_DISK ? disk_path : "";
    config.disk_cache_size_mb                    = layout == TierLayout::HOST_DISK ? lower_cache_size_mb : 0;
    config.disk_cache_buffered_io                = true;
    config.disk_cache_sync_timeout_ms            = 5000;
    config.disk_cache_staging_block_count        = 2;
    config.linear_step                           = 1;
    return config;
}

inline BatchKVCacheResourcePtr makeResource(const CacheConfig& config, size_t batch_size = 1) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(batch_size);
    resource->initGroups(config.topologyPtr());
    return resource;
}

inline CompleteTokenIdsPtr makeTokenIds(int offset, int seq_len, int max_seq_len, int seq_size_per_block) {
    auto input_ids                  = torch::arange(offset, offset + max_seq_len, torch::kInt32);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = input_ids;
    generate_input->generate_config = std::make_shared<GenerateConfig>();

    auto token_ids = std::make_shared<CompleteTokenIds>(
        /*batch_size=*/1, /*max_batch_size=*/1, max_seq_len + 16, seq_size_per_block);
    token_ids->init(generate_input);
    token_ids->setSeqLength(seq_len);
    return token_ids;
}

inline CompleteTokenIdsPtr makeBatchedTokenIdsWithCommonPrefix(
    int offset, int batch_size, int common_seq_len, int seq_len, int seq_size_per_block) {
    auto input_ids                  = torch::arange(offset, offset + common_seq_len, torch::kInt32);
    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = input_ids;
    generate_input->generate_config = std::make_shared<GenerateConfig>();

    auto token_ids = std::make_shared<CompleteTokenIds>(batch_size, batch_size, seq_len + 16, seq_size_per_block);
    token_ids->init(generate_input);
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        for (int position = common_seq_len; position < seq_len; ++position) {
            token_ids->data(batch_id)[position] = offset + position;
        }
    }
    token_ids->setSeqLength(seq_len);
    return token_ids;
}

inline bool isReusableGroup(const CacheConfig& config, int group_id) {
    return config.policyForGroup(static_cast<size_t>(group_id)).enable_prefix_reuse;
}

inline bool isFullGroup(const CacheConfig& config, int group_id) {
    return config.typeForGroup(static_cast<size_t>(group_id)) == CacheGroupType::FULL;
}

struct PoolSnapshot {
    std::shared_ptr<IBlockPool> pool;
    size_t                      free_blocks{0};
    size_t                      used_blocks{0};
    size_t                      request_refs{0};
    size_t                      cache_refs{0};
    size_t                      eviction_refs{0};
};

inline PoolSnapshot snapshotPool(const std::shared_ptr<IBlockPool>& pool) {
    EXPECT_NE(pool, nullptr);
    if (pool == nullptr) {
        return {};
    }
    return {
        pool,
        pool->freeBlocksNum(),
        pool->usedBlocksNum(),
        0,
        pool->referencedBlocksNum(BlockTreeRefType::CACHE),
        pool->referencedBlocksNum(BlockTreeRefType::EVICTION),
    };
}

inline PoolSnapshot snapshotPool(const DeviceBlockPoolPtr& pool) {
    PoolSnapshot snapshot = snapshotPool(std::static_pointer_cast<IBlockPool>(pool));
    if (pool != nullptr) {
        snapshot.request_refs = pool->referencedBlocksNum();
    }
    return snapshot;
}

inline std::vector<PoolSnapshot> snapshotDevicePools(const std::shared_ptr<KVCacheManager>& manager) {
    std::vector<PoolSnapshot> snapshots;
    const auto                groups = manager->allocator_->cacheGroups();
    snapshots.reserve(groups.size());
    for (const auto& group : groups) {
        snapshots.push_back(snapshotPool(group->blockPool()));
    }
    return snapshots;
}

inline std::vector<PoolSnapshot> snapshotLowerPools(const BlockTreeCache& cache, TierLayout layout) {
    std::vector<PoolSnapshot> snapshots;
    for (const auto& group_set : cache.groupSets()) {
        snapshots.push_back(snapshotPool(group_set->hostPool()));
        if (layout == TierLayout::HOST_DISK) {
            snapshots.push_back(snapshotPool(group_set->diskPool()));
        }
    }
    return snapshots;
}

inline void expectPoolSnapshotsEq(const std::vector<PoolSnapshot>& expected, const std::vector<PoolSnapshot>& actual) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual[i].pool.get(), expected[i].pool.get()) << "pool index " << i;
        EXPECT_EQ(actual[i].free_blocks, expected[i].free_blocks) << expected[i].pool->poolName();
        EXPECT_EQ(actual[i].used_blocks, expected[i].used_blocks) << expected[i].pool->poolName();
        EXPECT_EQ(actual[i].request_refs, expected[i].request_refs) << expected[i].pool->poolName();
        EXPECT_EQ(actual[i].cache_refs, expected[i].cache_refs) << expected[i].pool->poolName();
        EXPECT_EQ(actual[i].eviction_refs, expected[i].eviction_refs) << expected[i].pool->poolName();
    }
}

inline std::vector<const TreeNode*> collectTreeNodes(const BlockTreeCache& cache) {
    std::vector<const TreeNode*> nodes;
    nodes.reserve(cache.tree()->size());
    for (const auto& [_, child] : cache.tree()->root()->children) {
        nodes.push_back(child);
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (const auto& [_, child] : nodes[i]->children) {
            nodes.push_back(child);
        }
    }
    return nodes;
}

inline size_t countTreeResourcesAtTier(const BlockTreeCache& cache, Tier tier) {
    size_t count = 0;
    for (const TreeNode* node : collectTreeNodes(cache)) {
        for (size_t group_set_id = 0; group_set_id < node->group_set_resources.size(); ++group_set_id) {
            const auto& resource = node->group_set_resources[group_set_id];
            if (!resource.is_empty() && resource.getTopTier() == tier) {
                ++count;
            }
        }
    }
    return count;
}

inline std::shared_ptr<IBlockPool> lowerPoolForTier(const GroupSetPtr& group_set, Tier tier) {
    if (group_set == nullptr) {
        return nullptr;
    }
    if (tier == Tier::HOST) {
        return group_set->hostPool();
    }
    if (tier == Tier::DISK) {
        return group_set->diskPool();
    }
    return nullptr;
}

inline std::vector<std::shared_ptr<IBlockPool>> poolsForTier(const GroupSetPtr& group_set, Tier tier) {
    std::vector<std::shared_ptr<IBlockPool>> pools;
    if (group_set == nullptr) {
        return pools;
    }
    if (tier == Tier::DEVICE) {
        pools.reserve(group_set->devicePools().size());
        for (const DeviceBlockPoolPtr& pool : group_set->devicePools()) {
            pools.push_back(std::static_pointer_cast<IBlockPool>(pool));
        }
        return pools;
    }
    const std::shared_ptr<IBlockPool> lower_pool = lowerPoolForTier(group_set, tier);
    if (lower_pool != nullptr) {
        pools.push_back(lower_pool);
    }
    return pools;
}

inline BlockIdxType lowerBlockForTier(const GroupSetResource& resource, Tier tier) {
    if (tier == Tier::HOST) {
        return resource.host_block;
    }
    if (tier == Tier::DISK) {
        return resource.disk_block;
    }
    return NULL_BLOCK_IDX;
}

inline void expectAllTreeResourcesIdleAtDevice(const BlockTreeCache& cache) {
    for (const TreeNode* node : collectTreeNodes(cache)) {
        ASSERT_EQ(node->group_set_resources.size(), cache.groupSets().size());
        for (size_t group_set_id = 0; group_set_id < node->group_set_resources.size(); ++group_set_id) {
            const auto& resource = node->group_set_resources[group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE)
                << "key=" << node->cache_key << " group_set=" << group_set_id;
            if (resource.is_empty()) {
                continue;
            }
            EXPECT_TRUE(resource.isValidSteadyState()) << "key=" << node->cache_key << " group_set=" << group_set_id;
            EXPECT_EQ(resource.getTopTier(), Tier::DEVICE)
                << "key=" << node->cache_key << " group_set=" << group_set_id;
        }
    }
}

inline void expectPathIdleAtDevice(const BlockTreeCache& cache, const CacheKeysType& keys) {
    const auto found = cache.tree()->findNode(keys);
    ASSERT_EQ(found.size(), keys.size());
    for (size_t path_index = 0; path_index < found.size(); ++path_index) {
        const auto* node = found[path_index];
        ASSERT_NE(node, nullptr);
        ASSERT_EQ(node->group_set_resources.size(), cache.groupSets().size());
        bool any_resource = false;
        for (size_t group_set_id = 0; group_set_id < node->group_set_resources.size(); ++group_set_id) {
            const auto& resource = node->group_set_resources[group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE)
                << "path=" << path_index << " group_set=" << group_set_id;
            if (resource.is_empty()) {
                continue;
            }
            any_resource = true;
            EXPECT_TRUE(resource.isValidSteadyState()) << "path=" << path_index << " group_set=" << group_set_id;
            EXPECT_EQ(resource.getTopTier(), Tier::DEVICE) << "path=" << path_index << " group_set=" << group_set_id;
        }
        EXPECT_TRUE(any_resource) << "path=" << path_index;
    }
}

inline void
expectDsv4TierTopology(const std::shared_ptr<KVCacheManager>& manager, const CacheConfig& config, TierLayout layout) {
    ASSERT_NE(manager, nullptr);
    const auto cache = manager->blockTreeCache();
    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(config.groupNums(), kDsv4GroupCount);
    ASSERT_EQ(config.groupTagsSnapshot(), kDsv4Tags);
    ASSERT_EQ(config.groupTypesSnapshot(), kDsv4Types);
    ASSERT_EQ(config.linear_step, 1);

    const auto allocator_groups = manager->allocator_->cacheGroups();
    ASSERT_EQ(allocator_groups.size(), static_cast<size_t>(kDsv4GroupCount));
    std::unordered_set<const IBlockPool*> unique_device_pools;
    for (int group_id = 0; group_id < kDsv4GroupCount; ++group_id) {
        const auto& group = allocator_groups[static_cast<size_t>(group_id)];
        ASSERT_NE(group, nullptr);
        ASSERT_NE(group->blockPool(), nullptr);
        EXPECT_TRUE(unique_device_pools.emplace(group->blockPool().get()).second)
            << "device pool must be independent, group=" << group_id;
        EXPECT_GT(group->blockPool()->freeBlocksNum(), 0u) << "group=" << group_id;
        EXPECT_EQ(isReusableGroup(config, group_id), config.tagForGroup(static_cast<size_t>(group_id)) != "hca_state");
    }
    EXPECT_EQ(unique_device_pools.size(), static_cast<size_t>(kDsv4GroupCount));

    ASSERT_EQ(cache->groupSets().size(), 2u);
    std::vector<size_t>                membership_count(static_cast<size_t>(kDsv4GroupCount), 0);
    std::unordered_set<CacheGroupType> group_set_types;
    for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
        const auto& group_set = cache->groupSets()[group_set_id];
        ASSERT_NE(group_set, nullptr);
        EXPECT_EQ(group_set->groupSetId(), group_set_id);
        EXPECT_TRUE(group_set_types.emplace(group_set->groupType()).second);
        const std::vector<size_t>& expected_group_ids =
            group_set->groupType() == CacheGroupType::SWA ? kDsv4SwaGroupIds : kDsv4FullGroupIds;
        const std::string expected_type_name = group_set->groupType() == CacheGroupType::SWA ? "swa" : "full";
        EXPECT_EQ(group_set->groupIds(), expected_group_ids);
        ASSERT_FALSE(group_set->groupIds().empty());
        ASSERT_EQ(group_set->groupIds().size(), group_set->devicePools().size());
        ASSERT_NE(group_set->hostPool(), nullptr);
        EXPECT_EQ(group_set->hostPool()->poolName(), "block_tree_host_" + expected_type_name);
        EXPECT_GT(group_set->hostPool()->freeBlocksNum(), 0u);
        if (layout == TierLayout::HOST_DISK) {
            ASSERT_NE(group_set->diskPool(), nullptr);
            EXPECT_EQ(group_set->diskPool()->poolName(), "block_tree_disk_" + expected_type_name);
            EXPECT_GT(group_set->diskPool()->freeBlocksNum(), 0u);
        } else {
            EXPECT_EQ(group_set->diskPool(), nullptr);
        }
        for (size_t member_index = 0; member_index < group_set->groupIds().size(); ++member_index) {
            const size_t group_id = group_set->groupIds()[member_index];
            ASSERT_LT(group_id, static_cast<size_t>(kDsv4GroupCount));
            EXPECT_TRUE(config.policyForGroup(group_id).enable_prefix_reuse);
            EXPECT_EQ(group_set->devicePools()[member_index].get(), allocator_groups[group_id]->blockPool().get());
            ++membership_count[group_id];
        }
    }
    for (int group_id = 0; group_id < kDsv4GroupCount; ++group_id) {
        const bool hca_state = config.tagForGroup(static_cast<size_t>(group_id)) == "hca_state";
        EXPECT_EQ(membership_count[static_cast<size_t>(group_id)], hca_state ? 0u : 1u) << "group=" << group_id;
    }
    EXPECT_EQ(group_set_types.size(), 2u);
}

struct SeededPrefix {
    BatchKVCacheResourcePtr       resource;
    CompleteTokenIdsPtr           token_ids;
    CacheKeysType                 full_cache_keys;
    CacheKeysType                 cache_keys;
    std::vector<BlockIndicesType> blocks_by_group;
};

inline void appendDevicePools(const GroupSetPtr& group_set, std::vector<std::shared_ptr<IBlockPool>>& pools) {
    for (const DeviceBlockPoolPtr& pool : group_set->devicePools()) {
        pools.push_back(std::static_pointer_cast<IBlockPool>(pool));
    }
}

inline BlockIndicesType
groupSetSeedBlocksAt(const GroupSetPtr& group_set, const SeededPrefix& seed, size_t path_index) {
    BlockIndicesType blocks;
    blocks.reserve(group_set->groupIds().size());
    for (const size_t group_id : group_set->groupIds()) {
        blocks.push_back(seed.blocks_by_group[group_id][path_index]);
    }
    return blocks;
}

inline BlockIndicesType groupSetRequestBlocksAt(const GroupSetPtr&             group_set,
                                                const BatchKVCacheResourcePtr& resource,
                                                size_t                         batch_id,
                                                size_t                         path_index) {
    BlockIndicesType blocks;
    blocks.reserve(group_set->groupIds().size());
    for (const size_t group_id : group_set->groupIds()) {
        blocks.push_back(resource->blocks(batch_id, static_cast<int>(group_id))[path_index]);
    }
    return blocks;
}

using PathResourcesSnapshot = std::vector<std::vector<GroupSetResource>>;

inline BlockIndicesType resourceBlocksForTier(const GroupSetResource& resource, Tier tier) {
    if (tier == Tier::DEVICE) {
        return resource.device_blocks;
    }
    const BlockIdxType block = lowerBlockForTier(resource, tier);
    return isNullBlockIdx(block) ? BlockIndicesType{} : BlockIndicesType{block};
}

inline size_t countPathResourcesAtTier(const PathResourcesSnapshot& snapshot, Tier tier) {
    size_t count = 0;
    for (const auto& path_resources : snapshot) {
        count += static_cast<size_t>(
            std::count_if(path_resources.begin(), path_resources.end(), [tier](const auto& resource) {
                return !resource.is_empty() && resource.getTopTier() == tier;
            }));
    }
    return count;
}

inline bool pathSnapshotContainsBlocks(const PathResourcesSnapshot& snapshot,
                                       size_t                       group_set_id,
                                       Tier                         tier,
                                       const BlockIndicesType&      blocks) {
    if (blocks.empty()) {
        return false;
    }
    for (const auto& path_resources : snapshot) {
        if (group_set_id < path_resources.size()
            && resourceBlocksForTier(path_resources[group_set_id], tier) == blocks) {
            return true;
        }
    }
    return false;
}

inline std::optional<PathResourcesSnapshot> snapshotPathResources(const BlockTreeCache& cache,
                                                                  const CacheKeysType&  keys) {
    const auto found = cache.tree()->findNode(keys);
    if (found.size() != keys.size()) {
        return std::nullopt;
    }
    PathResourcesSnapshot snapshot;
    snapshot.reserve(found.size());
    for (const auto* node : found) {
        if (node == nullptr || node->group_set_resources.size() != cache.groupSets().size()) {
            return std::nullopt;
        }
        snapshot.push_back(node->group_set_resources);
    }
    return snapshot;
}

inline void expectFullTierPathUnchanged(const BlockTreeCache&                           cache,
                                        const CacheKeysType&                            keys,
                                        Tier                                            tier,
                                        const std::vector<std::vector<BlockIdxType>>&   expected_blocks,
                                        const std::vector<std::shared_ptr<IBlockPool>>& pools,
                                        size_t                                          capacity) {
    ASSERT_TRUE(tier == Tier::HOST || tier == Tier::DISK);
    const auto maybe_current = snapshotPathResources(cache, keys);
    ASSERT_TRUE(maybe_current.has_value());
    ASSERT_EQ(maybe_current->size(), expected_blocks.size());
    ASSERT_EQ(pools.size(), cache.groupSets().size());
    for (size_t path_index = 0; path_index < maybe_current->size(); ++path_index) {
        ASSERT_EQ((*maybe_current)[path_index].size(), cache.groupSets().size());
        ASSERT_EQ(expected_blocks[path_index].size(), cache.groupSets().size());
        for (size_t group_set_id = 0; group_set_id < cache.groupSets().size(); ++group_set_id) {
            const auto& resource = (*maybe_current)[path_index][group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE)
                << "path=" << path_index << " group_set=" << group_set_id;
            EXPECT_EQ(resource.getTopTier(), tier) << "path=" << path_index << " group_set=" << group_set_id;
            EXPECT_EQ(lowerBlockForTier(resource, tier), expected_blocks[path_index][group_set_id])
                << "path=" << path_index << " group_set=" << group_set_id;
            EXPECT_TRUE(pools[group_set_id]->isAllocated(expected_blocks[path_index][group_set_id]))
                << "path=" << path_index << " group_set=" << group_set_id;
            EXPECT_EQ(pools[group_set_id]->treeRefCount(expected_blocks[path_index][group_set_id]), 1u)
                << "path=" << path_index << " group_set=" << group_set_id;
        }
    }
    for (const auto& pool : pools) {
        EXPECT_EQ(pool->freeBlocksNum(), 0u) << pool->poolName();
        EXPECT_EQ(pool->usedBlocksNum(), capacity) << pool->poolName();
        EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), capacity) << pool->poolName();
        EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u) << pool->poolName();
    }
}

inline uint8_t payloadPattern(int group_id, size_t path_index, size_t local_layer) {
    return static_cast<uint8_t>(0x10 + group_id * 0x18 + static_cast<int>(path_index) * 0x06
                                + static_cast<int>(local_layer));
}

inline bool
fillDeviceBlockLayer(const DeviceBlockPoolPtr& pool, BlockIdxType block, size_t local_layer, uint8_t pattern) {
    if (pool == nullptr || isNullBlockIdx(block)) {
        return false;
    }
    const auto buffers = pool->convertIndexToBuffer(static_cast<int>(local_layer), block);
    if (buffers.empty()) {
        return false;
    }
    for (const auto& buffer : buffers) {
        if (buffer.addr == nullptr || buffer.size_bytes == 0) {
            return false;
        }
        auto view = torch::from_blob(buffer.addr,
                                     {static_cast<int64_t>(buffer.size_bytes)},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        view.fill_(pattern);
        (void)view.cpu();
    }
    return true;
}

inline bool
deviceBlockLayerHasPattern(const DeviceBlockPoolPtr& pool, BlockIdxType block, size_t local_layer, uint8_t pattern) {
    if (pool == nullptr || isNullBlockIdx(block)) {
        return false;
    }
    const auto buffers = pool->convertIndexToBuffer(static_cast<int>(local_layer), block);
    if (buffers.empty()) {
        return false;
    }
    for (const auto& buffer : buffers) {
        if (buffer.addr == nullptr || buffer.size_bytes == 0) {
            return false;
        }
        auto view = torch::from_blob(buffer.addr,
                                     {static_cast<int64_t>(buffer.size_bytes)},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        if (!view.eq(pattern).all().cpu().item<bool>()) {
            return false;
        }
    }
    return true;
}

inline bool fillGroupBlockPayload(const std::shared_ptr<KVCacheManager>& manager,
                                  const CacheConfig&                     config,
                                  int                                    group_id,
                                  BlockIdxType                           block,
                                  size_t                                 path_index,
                                  bool                                   poison) {
    if (manager == nullptr || group_id < 0 || group_id >= config.groupNums()) {
        return false;
    }
    const auto groups = manager->allocator_->cacheGroups();
    if (static_cast<size_t>(group_id) >= groups.size() || groups[static_cast<size_t>(group_id)] == nullptr) {
        return false;
    }
    const auto& layer_ids = config.layerIdsForGroup(static_cast<size_t>(group_id));
    if (layer_ids.empty()) {
        return false;
    }
    for (size_t local_layer = 0; local_layer < layer_ids.size(); ++local_layer) {
        uint8_t pattern = payloadPattern(group_id, path_index, local_layer);
        if (poison) {
            pattern ^= 0xFF;
        }
        if (!fillDeviceBlockLayer(groups[static_cast<size_t>(group_id)]->blockPool(), block, local_layer, pattern)) {
            return false;
        }
    }
    return true;
}

inline bool groupBlockPayloadMatches(const std::shared_ptr<KVCacheManager>& manager,
                                     const CacheConfig&                     config,
                                     int                                    group_id,
                                     BlockIdxType                           block,
                                     size_t                                 path_index) {
    if (manager == nullptr || group_id < 0 || group_id >= config.groupNums()) {
        return false;
    }
    const auto groups = manager->allocator_->cacheGroups();
    if (static_cast<size_t>(group_id) >= groups.size() || groups[static_cast<size_t>(group_id)] == nullptr) {
        return false;
    }
    const auto& layer_ids = config.layerIdsForGroup(static_cast<size_t>(group_id));
    if (layer_ids.empty()) {
        return false;
    }
    for (size_t local_layer = 0; local_layer < layer_ids.size(); ++local_layer) {
        if (!deviceBlockLayerHasPattern(groups[static_cast<size_t>(group_id)]->blockPool(),
                                        block,
                                        local_layer,
                                        payloadPattern(group_id, path_index, local_layer))) {
            return false;
        }
    }
    return true;
}

inline bool
fillSeedPayload(const std::shared_ptr<KVCacheManager>& manager, const CacheConfig& config, const SeededPrefix& seed) {
    if (manager == nullptr || seed.blocks_by_group.size() != static_cast<size_t>(config.groupNums())) {
        return false;
    }
    const auto groups = manager->allocator_->cacheGroups();
    if (groups.size() != seed.blocks_by_group.size()) {
        return false;
    }
    for (int group_id = 0; group_id < config.groupNums(); ++group_id) {
        if (!isReusableGroup(config, group_id)) {
            continue;
        }
        const auto& blocks = seed.blocks_by_group[static_cast<size_t>(group_id)];
        if (blocks.size() != seed.cache_keys.size()) {
            return false;
        }
        for (size_t path_index = 0; path_index < blocks.size(); ++path_index) {
            if (!fillGroupBlockPayload(manager, config, group_id, blocks[path_index], path_index, /*poison=*/false)) {
                return false;
            }
        }
    }
    return true;
}

inline std::optional<size_t>
cpCanonicalBlockPosition(const CPSlotMapper& mapper, const CacheConfig& config, int group_id, size_t path_index) {
    if (group_id < 0 || group_id >= config.groupNums()) {
        return std::nullopt;
    }
    const size_t gid = static_cast<size_t>(group_id);
    if (mapper.blockRoundRobinGroup(config, gid) || mapper.compactLastRankGroup(config, gid)) {
        return path_index;
    }
    return (path_index + 1) * static_cast<size_t>(mapper.cpSize()) - 1;
}

inline bool fillCpCanonicalSeedPayload(const std::shared_ptr<KVCacheManager>& manager,
                                       const CacheConfig&                     config,
                                       const CPSlotMapper&                    mapper,
                                       const SeededPrefix&                    seed) {
    if (manager == nullptr || seed.blocks_by_group.size() != static_cast<size_t>(config.groupNums())) {
        return false;
    }
    for (int group_id = 0; group_id < config.groupNums(); ++group_id) {
        if (!isReusableGroup(config, group_id)) {
            continue;
        }
        const auto& blocks = seed.blocks_by_group[static_cast<size_t>(group_id)];
        for (size_t path_index = 0; path_index < seed.cache_keys.size(); ++path_index) {
            const auto position = cpCanonicalBlockPosition(mapper, config, group_id, path_index);
            if (!position.has_value() || *position >= blocks.size() || isNullBlockIdx(blocks[*position])
                || !fillGroupBlockPayload(manager, config, group_id, blocks[*position], path_index, /*poison=*/false)) {
                return false;
            }
        }
    }
    return true;
}

inline bool pathDevicePayloadMatches(const std::shared_ptr<KVCacheManager>& manager,
                                     const BlockTreeCache&                  cache,
                                     const CacheKeysType&                   keys) {
    const auto maybe_resources = snapshotPathResources(cache, keys);
    if (manager == nullptr || !maybe_resources.has_value()) {
        return false;
    }
    const auto groups = manager->allocator_->cacheGroups();
    for (size_t path_index = 0; path_index < maybe_resources->size(); ++path_index) {
        for (size_t group_set_id = 0; group_set_id < cache.groupSets().size(); ++group_set_id) {
            const auto& group_set = cache.groupSets()[group_set_id];
            const auto& resource  = (*maybe_resources)[path_index][group_set_id];
            if (!resource.hasTier(Tier::DEVICE) || group_set->groupIds().size() != resource.device_blocks.size()) {
                return false;
            }
            for (size_t member_index = 0; member_index < group_set->groupIds().size(); ++member_index) {
                const size_t group_id = group_set->groupIds()[member_index];
                if (group_id >= groups.size()
                    || !groupBlockPayloadMatches(manager,
                                                 manager->cacheConfig(),
                                                 static_cast<int>(group_id),
                                                 resource.device_blocks[member_index],
                                                 path_index)) {
                    return false;
                }
            }
        }
    }
    return true;
}

inline bool requestReusesExpectedPath(const BlockTreeCache&          cache,
                                      const CacheConfig&             config,
                                      const CacheKeysType&           keys,
                                      const BatchKVCacheResourcePtr& request_resource,
                                      size_t                         logical_reuse_blocks) {
    if (request_resource == nullptr || logical_reuse_blocks == 0 || logical_reuse_blocks > keys.size()) {
        return false;
    }
    const auto found = cache.tree()->findNode(keys);
    if (found.size() != keys.size()) {
        return false;
    }
    for (size_t group_set_id = 0; group_set_id < cache.groupSets().size(); ++group_set_id) {
        const auto& group_set = cache.groupSets()[group_set_id];
        if (group_set == nullptr || group_set->groupIds().size() != group_set->devicePools().size()) {
            return false;
        }
        const size_t reuse_count = group_set->computeReuseBlockCount(logical_reuse_blocks);
        if (reuse_count == 0 || reuse_count > logical_reuse_blocks) {
            return false;
        }
        const size_t reuse_begin = logical_reuse_blocks - reuse_count;
        for (size_t member_index = 0; member_index < group_set->groupIds().size(); ++member_index) {
            const size_t group_id = group_set->groupIds()[member_index];
            if (group_id >= static_cast<size_t>(config.groupNums())) {
                return false;
            }
            const auto& request_blocks = request_resource->blocks(0, static_cast<int>(group_id));
            if (request_blocks.size() < logical_reuse_blocks) {
                return false;
            }
            for (size_t path_index = reuse_begin; path_index < logical_reuse_blocks; ++path_index) {
                const auto* node = found[path_index];
                if (node == nullptr || group_set_id >= node->group_set_resources.size()) {
                    return false;
                }
                const auto& tree_resource = node->group_set_resources[group_set_id];
                if (tree_resource.transfer_state != GroupSetTransferState::IDLE || !tree_resource.hasTier(Tier::DEVICE)
                    || tree_resource.device_blocks.size() != group_set->groupIds().size()
                    || request_blocks[path_index] != tree_resource.device_blocks[member_index]) {
                    return false;
                }
            }
        }
    }
    return true;
}

inline bool requestReusesExpectedCpCanonicalPath(const BlockTreeCache&          cache,
                                                 const CacheConfig&             config,
                                                 const CPSlotMapper&            mapper,
                                                 const CacheKeysType&           canonical_keys,
                                                 const BatchKVCacheResourcePtr& request_resource,
                                                 size_t                         logical_reuse_blocks) {
    if (request_resource == nullptr || logical_reuse_blocks == 0 || logical_reuse_blocks > canonical_keys.size()) {
        return false;
    }
    const auto found = cache.tree()->findNode(canonical_keys);
    if (found.size() != canonical_keys.size()) {
        return false;
    }
    for (size_t group_set_id = 0; group_set_id < cache.groupSets().size(); ++group_set_id) {
        const auto& group_set = cache.groupSets()[group_set_id];
        if (group_set == nullptr || group_set->groupIds().size() != group_set->devicePools().size()) {
            return false;
        }
        const size_t reuse_count = group_set->computeReuseBlockCount(logical_reuse_blocks);
        if (reuse_count == 0 || reuse_count > logical_reuse_blocks) {
            return false;
        }
        const size_t reuse_begin = logical_reuse_blocks - reuse_count;
        for (size_t member_index = 0; member_index < group_set->groupIds().size(); ++member_index) {
            const int group_id = static_cast<int>(group_set->groupIds()[member_index]);
            if (group_id < 0 || group_id >= config.groupNums()) {
                return false;
            }
            const auto group_type = config.typeForGroup(static_cast<size_t>(group_id));
            if ((group_type == CacheGroupType::FULL
                 && !mapper.blockRoundRobinGroup(config, static_cast<size_t>(group_id)))
                || (group_type == CacheGroupType::SWA
                    && !mapper.compactLastRankGroup(config, static_cast<size_t>(group_id)))) {
                return false;
            }
            const auto& request_blocks = request_resource->blocks(0, group_id);
            for (size_t path_index = reuse_begin; path_index < logical_reuse_blocks; ++path_index) {
                const auto  position = cpCanonicalBlockPosition(mapper, config, group_id, path_index);
                const auto* node     = found[path_index];
                if (!position.has_value() || *position >= request_blocks.size() || node == nullptr
                    || group_set_id >= node->group_set_resources.size()) {
                    return false;
                }
                const auto& tree_resource = node->group_set_resources[group_set_id];
                if (tree_resource.transfer_state != GroupSetTransferState::IDLE || !tree_resource.hasTier(Tier::DEVICE)
                    || tree_resource.device_blocks.size() != group_set->groupIds().size()
                    || request_blocks[*position] != tree_resource.device_blocks[member_index]) {
                    return false;
                }
            }
        }
    }
    return true;
}

inline bool requestReusedPayloadMatchesExpectedPath(const std::shared_ptr<KVCacheManager>& manager,
                                                    const BlockTreeCache&                  cache,
                                                    const CacheConfig&                     config,
                                                    const CacheKeysType&                   keys,
                                                    const BatchKVCacheResourcePtr&         request_resource,
                                                    size_t                                 logical_reuse_blocks) {
    if (manager == nullptr || request_resource == nullptr || logical_reuse_blocks == 0
        || logical_reuse_blocks > keys.size()) {
        return false;
    }
    const auto found = cache.tree()->findNode(keys);
    if (found.size() != keys.size()) {
        return false;
    }
    for (const auto& group_set : cache.groupSets()) {
        if (group_set == nullptr) {
            return false;
        }
        const size_t reuse_count = group_set->computeReuseBlockCount(logical_reuse_blocks);
        if (reuse_count == 0 || reuse_count > logical_reuse_blocks) {
            return false;
        }
        const size_t reuse_begin = logical_reuse_blocks - reuse_count;
        for (const size_t raw_group_id : group_set->groupIds()) {
            if (raw_group_id >= static_cast<size_t>(config.groupNums())) {
                return false;
            }
            const auto& blocks = request_resource->blocks(0, static_cast<int>(raw_group_id));
            if (blocks.size() < logical_reuse_blocks) {
                return false;
            }
            for (size_t path_index = reuse_begin; path_index < logical_reuse_blocks; ++path_index) {
                if (isNullBlockIdx(blocks[path_index])
                    || !groupBlockPayloadMatches(
                        manager, config, static_cast<int>(raw_group_id), blocks[path_index], path_index)) {
                    return false;
                }
            }
        }
    }
    return true;
}

inline std::optional<double> oneUsedBlockWatermarkRatio(const std::vector<std::shared_ptr<IBlockPool>>& pools) {
    if (pools.empty()) {
        return std::nullopt;
    }
    size_t max_capacity = 0;
    for (const auto& pool : pools) {
        if (pool == nullptr || pool->usedBlocksNum() != 1) {
            return std::nullopt;
        }
        max_capacity = std::max(max_capacity, pool->totalBlocksNum());
    }
    if (max_capacity == 0) {
        return std::nullopt;
    }
    const double ratio = 0.5 / static_cast<double>(max_capacity);
    for (const auto& pool : pools) {
        if (static_cast<size_t>(static_cast<double>(pool->totalBlocksNum()) * ratio) != 0) {
            return std::nullopt;
        }
    }
    return ratio;
}

inline std::optional<double> blockExcessWatermarkRatio(const std::vector<std::shared_ptr<IBlockPool>>& pools,
                                                       size_t                                          excess_blocks) {
    if (pools.empty() || pools.front() == nullptr || excess_blocks == 0) {
        return std::nullopt;
    }
    const size_t capacity = pools.front()->totalBlocksNum();
    const size_t used     = pools.front()->usedBlocksNum();
    if (capacity == 0 || used < excess_blocks || used > capacity) {
        return std::nullopt;
    }
    const size_t target = used - excess_blocks;
    const double ratio  = (static_cast<double>(target) + 0.5) / static_cast<double>(capacity);
    for (const auto& pool : pools) {
        if (pool == nullptr || pool->totalBlocksNum() != capacity || pool->usedBlocksNum() != used
            || static_cast<size_t>(static_cast<double>(capacity) * ratio) != target) {
            return std::nullopt;
        }
    }
    return ratio;
}

inline std::optional<double> zeroTargetWatermarkRatio(const std::vector<std::shared_ptr<IBlockPool>>& pools) {
    if (pools.empty()) {
        return std::nullopt;
    }
    size_t max_capacity = 0;
    for (const auto& pool : pools) {
        if (pool == nullptr || pool->usedBlocksNum() == 0) {
            return std::nullopt;
        }
        max_capacity = std::max(max_capacity, pool->totalBlocksNum());
    }
    if (max_capacity == 0) {
        return std::nullopt;
    }
    const double ratio = 0.5 / static_cast<double>(max_capacity);
    for (const auto& pool : pools) {
        if (static_cast<size_t>(static_cast<double>(pool->totalBlocksNum()) * ratio) != 0) {
            return std::nullopt;
        }
    }
    return ratio;
}

inline std::optional<SeededPrefix> seedDevicePrefix(const std::shared_ptr<KVCacheManager>& manager,
                                                    const CacheConfig&                     config,
                                                    int                                    token_offset,
                                                    int                                    cached_blocks) {
    const int    seq_size_per_block = static_cast<int>(config.seq_size_per_block);
    const int    seq_len            = cached_blocks * seq_size_per_block;
    SeededPrefix seed;
    seed.resource  = makeResource(config);
    seed.token_ids = makeTokenIds(token_offset, seq_len, seq_len + 1, seq_size_per_block);

    MallocInfo malloc_info{seed.resource, seed.token_ids};
    malloc_info.reuse_cache         = true;
    malloc_info.enable_cache_lookup = false;
    const auto result               = manager->malloc(malloc_info);
    if (!result.success) {
        return std::nullopt;
    }

    seed.full_cache_keys = seed.resource->cacheKeys();
    seed.cache_keys      = seed.full_cache_keys;
    seed.blocks_by_group.reserve(static_cast<size_t>(config.groupNums()));
    for (int group_id = 0; group_id < config.groupNums(); ++group_id) {
        seed.blocks_by_group.push_back(seed.resource->blocks(0, group_id));
    }

    // The next token makes the last allocated block a cacheable full prefix block,
    // matching the production stream insertion lifecycle.
    seed.token_ids->setSeqLength(seq_len + 1);
    manager->insertIntoCache(InsertInfo{seed.resource, seed.token_ids, /*is_resident=*/false});
    manager->free(FreeInfo{seed.resource, seed.token_ids});
    return seed;
}

inline std::optional<SeededPrefix> seedCpCanonicalDevicePrefix(const std::shared_ptr<KVCacheManager>& manager,
                                                               const CacheConfig&                     config,
                                                               const std::shared_ptr<CPSlotMapper>&   mapper,
                                                               int                                    token_offset,
                                                               int                                    logical_blocks) {
    if (manager == nullptr || mapper == nullptr || !mapper->isSharded() || logical_blocks <= 0
        || logical_blocks % mapper->cpSize() != 0) {
        return std::nullopt;
    }
    const int    seq_size_per_block = static_cast<int>(config.seq_size_per_block);
    const int    seq_len            = logical_blocks * seq_size_per_block;
    SeededPrefix seed;
    seed.resource  = makeResource(config);
    seed.token_ids = makeTokenIds(token_offset, seq_len, seq_len, seq_size_per_block);

    MallocInfo malloc_info{seed.resource, seed.token_ids};
    malloc_info.reuse_cache         = true;
    malloc_info.enable_cache_lookup = false;
    const auto result               = manager->malloc(malloc_info);
    if (!result.success) {
        return std::nullopt;
    }

    seed.full_cache_keys = seed.resource->cacheKeys();
    seed.cache_keys      = mapper->canonicalCacheKeys(seed.full_cache_keys);
    seed.blocks_by_group.reserve(static_cast<size_t>(config.groupNums()));
    for (int group_id = 0; group_id < config.groupNums(); ++group_id) {
        seed.blocks_by_group.push_back(seed.resource->blocks(0, group_id));
    }

    // The sequence is exactly CP-virtual-block aligned, so manager insertion
    // must retain the final canonical key rather than exercising a partial tail.
    if (!seed.resource->lastBlockAligned()) {
        manager->free(FreeInfo{seed.resource, seed.token_ids});
        return std::nullopt;
    }
    manager->insertIntoCache(InsertInfo{seed.resource, seed.token_ids, /*is_resident=*/false});
    manager->free(FreeInfo{seed.resource, seed.token_ids});
    return seed;
}

struct DeviceSlotSnapshot {
    CacheKeysType             path_keys;
    size_t                    group_set_id{0};
    std::vector<BlockIdxType> device_blocks;
};

inline std::optional<std::vector<DeviceSlotSnapshot>> snapshotDeviceSlots(const BlockTreeCache&            cache,
                                                                          const std::vector<SeededPrefix>& seeds) {
    std::vector<DeviceSlotSnapshot> snapshots;
    for (const auto& seed : seeds) {
        const auto found = cache.tree()->findNode(seed.cache_keys);
        if (found.size() != seed.cache_keys.size()) {
            return std::nullopt;
        }
        for (size_t path_index = 0; path_index < found.size(); ++path_index) {
            const auto* node = found[path_index];
            if (node == nullptr || node->group_set_resources.size() != cache.groupSets().size()) {
                return std::nullopt;
            }
            for (size_t group_set_id = 0; group_set_id < cache.groupSets().size(); ++group_set_id) {
                const auto& resource = node->group_set_resources[group_set_id];
                if (!resource.hasTier(Tier::DEVICE)) {
                    continue;
                }
                snapshots.push_back({CacheKeysType(seed.cache_keys.begin(), seed.cache_keys.begin() + path_index + 1),
                                     group_set_id,
                                     resource.device_blocks});
            }
        }
    }
    return snapshots;
}

inline void reclaimAndExpectInitialPools(const std::shared_ptr<KVCacheManager>& manager,
                                         const std::vector<PoolSnapshot>&       initial_device,
                                         const std::vector<PoolSnapshot>&       initial_lower,
                                         TierLayout                             layout) {
    auto cache = manager->blockTreeCache();
    ASSERT_NE(cache, nullptr);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    for (const Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, /*num_blocks=*/4096, tier);
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    }
    EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
    const auto stats = cache->getStats();
    EXPECT_EQ(stats.tree_node_count, 0u);
    EXPECT_EQ(stats.device_heap_total_size, 0u);
    EXPECT_EQ(stats.host_heap_total_size, 0u);
    EXPECT_EQ(stats.disk_heap_total_size, 0u);
    expectPoolSnapshotsEq(initial_device, snapshotDevicePools(manager));
    expectPoolSnapshotsEq(initial_lower, snapshotLowerPools(*cache, layout));
}

class KVCacheManagerWithTierCacheTest: public ::testing::TestWithParam<TierLayout> {
protected:
    void SetUp() override {
        old_core_dump_on_exception_                  = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
        rtp_llm::initLogger();
        createDevice();
    }

    void TearDown() override {
        if (manager_ != nullptr && manager_->blockTreeCache() != nullptr) {
            block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*manager_->blockTreeCache());
        }
        manager_.reset();
        transfer_engine_.reset();
        const std::string disk_root = disk_dir_.path();
        std::string       cleanup_error;
        EXPECT_TRUE(disk_dir_.cleanup(&cleanup_error)) << cleanup_error;
        errno = 0;
        EXPECT_EQ(::access(disk_root.c_str(), F_OK), -1);
        EXPECT_EQ(errno, ENOENT);
        StaticConfig::user_ft_core_dump_on_exception = old_core_dump_on_exception_;
    }

    void initManager(uint32_t device_blocks, int64_t lower_cache_size_mb = 8) {
        ASSERT_FALSE(disk_dir_.path().empty());
        cache_config_ = makeCompactDsv4CacheConfig(device_blocks);

        manager_ = std::make_shared<KVCacheManager>(cache_config_,
                                                    /*warmup=*/false,
                                                    nullptr,
                                                    makeTierConfig(GetParam(), disk_dir_.path(), lower_cache_size_mb));
        ASSERT_TRUE(manager_->init());
        ASSERT_NE(manager_->blockTreeCache(), nullptr);
        auto cache = manager_->blockTreeCache();
        ASSERT_TRUE(cache->isHostCacheEnabled());
        ASSERT_EQ(cache->isDiskCacheEnabled(), GetParam() == TierLayout::HOST_DISK);
        ASSERT_NO_FATAL_FAILURE(expectDsv4TierTopology(manager_, cache_config_, GetParam()));

        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DISK, 0.0);
        transfer_engine_ = std::make_shared<ScriptedPerRankBlockTransferEngine>(cache->groupSets());
        BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, transfer_engine_);
    }

    template<typename TransferEngine>
    void moveAllPathResourcesToTier(const std::shared_ptr<BlockTreeCache>&          cache,
                                    const SeededPrefix&                             seed,
                                    Tier                                            source_tier,
                                    Tier                                            target_tier,
                                    const std::vector<std::shared_ptr<IBlockPool>>& source_pools,
                                    const std::shared_ptr<TransferEngine>&          engine) {
        ASSERT_NE(cache, nullptr);
        ASSERT_NE(engine, nullptr);
        const auto ratio = zeroTargetWatermarkRatio(source_pools);
        ASSERT_TRUE(ratio.has_value());
        auto maybe_before = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(maybe_before.has_value());
        size_t remaining = countPathResourcesAtTier(*maybe_before, source_tier);
        ASSERT_GT(remaining, 0u);
        const size_t max_rounds = remaining;

        for (size_t round = 0; remaining > 0 && round < max_rounds; ++round) {
            maybe_before = snapshotPathResources(*cache, seed.cache_keys);
            ASSERT_TRUE(maybe_before.has_value());
            std::vector<size_t> source_used_before;
            std::vector<size_t> target_used_before;
            source_used_before.reserve(cache->groupSets().size());
            target_used_before.reserve(cache->groupSets().size());
            for (const auto& group_set : cache->groupSets()) {
                const std::vector<std::shared_ptr<IBlockPool>> current_source_pools =
                    poolsForTier(group_set, source_tier);
                const std::vector<std::shared_ptr<IBlockPool>> current_target_pools =
                    poolsForTier(group_set, target_tier);
                ASSERT_FALSE(current_source_pools.empty());
                ASSERT_FALSE(current_target_pools.empty());
                size_t source_used = 0;
                size_t target_used = 0;
                for (const std::shared_ptr<IBlockPool>& pool : current_source_pools) {
                    source_used += pool->usedBlocksNum();
                }
                for (const std::shared_ptr<IBlockPool>& pool : current_target_pools) {
                    target_used += pool->usedBlocksNum();
                }
                source_used_before.push_back(source_used);
                target_used_before.push_back(target_used);
            }
            const size_t submits_before = engine->submittedDescriptorCount();
            BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, source_tier, *ratio);
            BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
            BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, source_tier, 0.0);
            ASSERT_TRUE(waitForPendingTasksDoneFor(
                *cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
                << "source=" << tierName(source_tier) << " target=" << tierName(target_tier) << " round=" << round
                << " pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
                << " submits=" << engine->submittedDescriptorCount();

            const auto descriptors = engine->descriptors();
            ASSERT_GT(descriptors.size(), submits_before) << "round=" << round;
            for (size_t index = submits_before; index < descriptors.size(); ++index) {
                const auto& descriptor = descriptors[index];
                ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
                EXPECT_EQ(descriptor.source_tier, source_tier) << "round=" << round;
                EXPECT_EQ(descriptor.target_tier, target_tier) << "round=" << round;
                EXPECT_TRUE(pathSnapshotContainsBlocks(
                    *maybe_before, descriptor.group_set_id, source_tier, descriptor.blocksAt(source_tier)))
                    << "round=" << round << " group_set=" << descriptor.group_set_id;
            }

            const auto maybe_after = snapshotPathResources(*cache, seed.cache_keys);
            ASSERT_TRUE(maybe_after.has_value());
            const size_t next = countPathResourcesAtTier(*maybe_after, source_tier);
            ASSERT_LT(next, remaining) << "round=" << round;
            bool any_capacity_progress = false;
            for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
                const GroupSetPtr&                             group_set = cache->groupSets()[group_set_id];
                const std::vector<std::shared_ptr<IBlockPool>> current_source_pools =
                    poolsForTier(group_set, source_tier);
                const std::vector<std::shared_ptr<IBlockPool>> current_target_pools =
                    poolsForTier(group_set, target_tier);
                ASSERT_FALSE(current_source_pools.empty());
                ASSERT_FALSE(current_target_pools.empty());
                size_t source_used = 0;
                size_t target_used = 0;
                for (const std::shared_ptr<IBlockPool>& pool : current_source_pools) {
                    source_used += pool->usedBlocksNum();
                    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u) << "round=" << round;
                }
                for (const std::shared_ptr<IBlockPool>& pool : current_target_pools) {
                    target_used += pool->usedBlocksNum();
                    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::EVICTION), 0u) << "round=" << round;
                }
                EXPECT_LE(source_used, source_used_before[group_set_id]) << "round=" << round;
                EXPECT_GE(target_used, target_used_before[group_set_id]) << "round=" << round;
                any_capacity_progress = any_capacity_progress || source_used < source_used_before[group_set_id]
                                        || target_used > target_used_before[group_set_id];
            }
            EXPECT_TRUE(any_capacity_progress) << "round=" << round;
            for (size_t path_index = 0; path_index < maybe_after->size(); ++path_index) {
                for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
                    const auto& resource = (*maybe_after)[path_index][group_set_id];
                    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE)
                        << "round=" << round << " path=" << path_index << " group_set=" << group_set_id;
                    const Tier top_tier = resource.getTopTier();
                    ASSERT_TRUE(top_tier == source_tier || top_tier == target_tier)
                        << "round=" << round << " path=" << path_index << " group_set=" << group_set_id;
                    if (top_tier == Tier::DEVICE) {
                        const auto& device_pools = cache->groupSets()[group_set_id]->devicePools();
                        ASSERT_EQ(device_pools.size(), resource.device_blocks.size());
                        for (size_t member_group_id = 0; member_group_id < device_pools.size(); ++member_group_id) {
                            EXPECT_EQ(device_pools[member_group_id]->refCount(resource.device_blocks[member_group_id]),
                                      1u)
                                << "round=" << round << " path=" << path_index << " group_set=" << group_set_id
                                << " member=" << member_group_id;
                        }
                    } else {
                        const auto lower_pool = lowerPoolForTier(cache->groupSets()[group_set_id], top_tier);
                        ASSERT_NE(lower_pool, nullptr);
                        EXPECT_EQ(lower_pool->treeRefCount(lowerBlockForTier(resource, top_tier)), 1u)
                            << "round=" << round << " path=" << path_index << " group_set=" << group_set_id;
                    }
                }
            }
            remaining = next;
        }
        ASSERT_EQ(remaining, 0u);
    }

    void expectPausedHostLoadState(const std::shared_ptr<BlockTreeCache>& cache,
                                   const SeededPrefix&                    seed,
                                   const PathResourcesSnapshot&           host_snapshot,
                                   const BatchKVCacheResourcePtr&         request,
                                   PathResourcesSnapshot*                 observed_snapshot,
                                   const PathResourcesSnapshot*           expected_snapshot = nullptr) {
        ASSERT_NE(cache, nullptr);
        ASSERT_NE(request, nullptr);
        const auto maybe_current = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(maybe_current.has_value());
        const auto& current = *maybe_current;
        ASSERT_EQ(current.size(), seed.cache_keys.size());
        ASSERT_EQ(host_snapshot.size(), current.size());
        if (expected_snapshot != nullptr) {
            ASSERT_EQ(expected_snapshot->size(), current.size());
        }

        for (size_t path_index = 0; path_index < current.size(); ++path_index) {
            ASSERT_EQ(current[path_index].size(), cache->groupSets().size());
            ASSERT_EQ(host_snapshot[path_index].size(), cache->groupSets().size());
            if (expected_snapshot != nullptr) {
                ASSERT_EQ((*expected_snapshot)[path_index].size(), current[path_index].size());
            }
            for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
                const auto& group_set = cache->groupSets()[group_set_id];
                const auto& resource  = current[path_index][group_set_id];
                if (expected_snapshot != nullptr) {
                    const auto& expected = (*expected_snapshot)[path_index][group_set_id];
                    EXPECT_EQ(resource.device_blocks, expected.device_blocks);
                    EXPECT_EQ(resource.host_block, expected.host_block);
                    EXPECT_EQ(resource.disk_block, expected.disk_block);
                    EXPECT_EQ(resource.transfer_state, expected.transfer_state);
                    EXPECT_EQ(resource.candidate_meta.last_access_seq, expected.candidate_meta.last_access_seq);
                    EXPECT_EQ(resource.candidate_meta.admission_seq, expected.candidate_meta.admission_seq);
                    EXPECT_EQ(resource.candidate_meta.hit_count, expected.candidate_meta.hit_count);
                    EXPECT_EQ(resource.candidate_meta.tier_enter_time_us, expected.candidate_meta.tier_enter_time_us);
                }

                ASSERT_TRUE(resource.hasTier(Tier::HOST));
                EXPECT_EQ(resource.host_block, host_snapshot[path_index][group_set_id].host_block);
                const size_t reuse_count = group_set->computeReuseBlockCount(seed.cache_keys.size());
                ASSERT_GT(reuse_count, 0u);
                ASSERT_LE(reuse_count, seed.cache_keys.size());
                const size_t reuse_begin = seed.cache_keys.size() - reuse_count;
                if (path_index < reuse_begin) {
                    for (size_t member_group_id = 0; member_group_id < group_set->groupIds().size();
                         ++member_group_id) {
                        const int               group_id = static_cast<int>(group_set->groupIds()[member_group_id]);
                        const BlockIndicesType& blocks   = request->blocks(0, group_id);
                        ASSERT_EQ(blocks.size(), seed.cache_keys.size() + 1);
                        EXPECT_TRUE(isNullBlockIdx(blocks[path_index]))
                            << "non-active SWA prefix must remain sparse, group=" << group_id << " path=" << path_index;
                    }
                    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
                    EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 1u);
                    continue;
                }
                EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
                EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 2u);
                ASSERT_EQ(group_set->groupIds().size(), group_set->devicePools().size());
                for (size_t member_group_id = 0; member_group_id < group_set->groupIds().size(); ++member_group_id) {
                    const int               group_id = static_cast<int>(group_set->groupIds()[member_group_id]);
                    const BlockIndicesType& blocks   = request->blocks(0, group_id);
                    ASSERT_EQ(blocks.size(), seed.cache_keys.size() + 1);
                    ASSERT_FALSE(isNullBlockIdx(blocks[path_index]));
                    EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(blocks[path_index]), 2u);
                }
            }
        }
        if (observed_snapshot != nullptr) {
            *observed_snapshot = current;
        }
    }

    void expectHostLoadSettledAtDevice(const std::shared_ptr<BlockTreeCache>& cache,
                                       const SeededPrefix&                    seed,
                                       const PathResourcesSnapshot&           host_snapshot,
                                       const BatchKVCacheResourcePtr&         request) {
        ASSERT_NE(cache, nullptr);
        ASSERT_NE(request, nullptr);
        const auto maybe_current = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(maybe_current.has_value());
        const auto& current = *maybe_current;
        ASSERT_EQ(current.size(), seed.cache_keys.size());
        ASSERT_EQ(host_snapshot.size(), current.size());

        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            ASSERT_EQ(group_set->groupIds().size(), group_set->devicePools().size());
            const size_t reuse_count = group_set->computeReuseBlockCount(seed.cache_keys.size());
            ASSERT_GT(reuse_count, 0u);
            ASSERT_LE(reuse_count, seed.cache_keys.size());
            const size_t reuse_begin = seed.cache_keys.size() - reuse_count;
            for (size_t path_index = 0; path_index < current.size(); ++path_index) {
                const auto& resource = current[path_index][group_set_id];
                EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
                if (path_index < reuse_begin) {
                    for (size_t member_group_id = 0; member_group_id < group_set->groupIds().size();
                         ++member_group_id) {
                        const int               group_id = static_cast<int>(group_set->groupIds()[member_group_id]);
                        const BlockIndicesType& blocks   = request->blocks(0, group_id);
                        ASSERT_EQ(blocks.size(), seed.cache_keys.size() + 1);
                        EXPECT_TRUE(isNullBlockIdx(blocks[path_index]));
                    }
                    ASSERT_TRUE(resource.hasTier(Tier::HOST));
                    EXPECT_EQ(resource.getTopTier(), Tier::HOST);
                    EXPECT_EQ(resource.host_block, host_snapshot[path_index][group_set_id].host_block);
                    EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 1u);
                    continue;
                }
                ASSERT_TRUE(resource.hasTier(Tier::DEVICE));
                EXPECT_EQ(resource.getTopTier(), Tier::DEVICE);
                BlockIndicesType expected_device_blocks;
                expected_device_blocks.reserve(group_set->groupIds().size());
                for (size_t member_group_id = 0; member_group_id < group_set->groupIds().size(); ++member_group_id) {
                    const int               group_id = static_cast<int>(group_set->groupIds()[member_group_id]);
                    const BlockIndicesType& blocks   = request->blocks(0, group_id);
                    ASSERT_EQ(blocks.size(), seed.cache_keys.size() + 1);
                    ASSERT_FALSE(isNullBlockIdx(blocks[path_index]));
                    expected_device_blocks.push_back(blocks[path_index]);
                    EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(blocks[path_index]), 2u);
                }
                EXPECT_EQ(resource.device_blocks, expected_device_blocks);
                EXPECT_FALSE(group_set->hostPool()->isAllocated(host_snapshot[path_index][group_set_id].host_block));
            }
        }
    }

    void runLowerTierLoadFailureScenario(LoadFailureSource failure_source) {
        ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/8));
        ASSERT_NE(manager_, nullptr);
        auto cache = manager_->blockTreeCache();

        auto pausable_engine = std::make_shared<PausableRecordingTransferEngine>(cache->groupSets());
        BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, pausable_engine);
        transfer_engine_.reset();

        const auto initial_device = snapshotDevicePools(manager_);
        const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
        auto       maybe_seed     = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/1);
        ASSERT_TRUE(maybe_seed.has_value());
        auto seed = std::move(*maybe_seed);
        ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));
        ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
        ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));

        std::vector<std::shared_ptr<IBlockPool>> device_pools;
        for (const auto& group_set : cache->groupSets()) {
            device_pools.insert(device_pools.end(), group_set->devicePools().begin(), group_set->devicePools().end());
        }
        const auto device_ratio = oneUsedBlockWatermarkRatio(device_pools);
        ASSERT_TRUE(device_ratio.has_value());
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
        BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
        EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

        auto maybe_host = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(maybe_host.has_value());
        ASSERT_EQ(maybe_host->size(), 1u);
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            const auto& resource  = (*maybe_host)[0][group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            ASSERT_TRUE(resource.hasTier(Tier::HOST));
            EXPECT_EQ(resource.getTopTier(), Tier::HOST);
            EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 1u);
        }

        if (failure_source == LoadFailureSource::DISK) {
            std::vector<std::shared_ptr<IBlockPool>> host_pools;
            for (const auto& group_set : cache->groupSets()) {
                host_pools.push_back(group_set->hostPool());
            }
            const auto host_ratio = oneUsedBlockWatermarkRatio(host_pools);
            ASSERT_TRUE(host_ratio.has_value());
            BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, *host_ratio);
            BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
            BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
            block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
        }
        EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

        auto maybe_lower = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(maybe_lower.has_value());
        ASSERT_EQ(maybe_lower->size(), 1u);
        std::vector<Tier>         source_tiers(cache->groupSets().size(), Tier::NONE);
        std::vector<BlockIdxType> source_blocks(cache->groupSets().size(), NULL_BLOCK_IDX);
        size_t                    host_source_count = 0;
        size_t                    disk_source_count = 0;
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            const auto& resource  = (*maybe_lower)[0][group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            source_tiers[group_set_id] = resource.getTopTier();
            if (source_tiers[group_set_id] == Tier::HOST) {
                source_blocks[group_set_id] = resource.host_block;
                ++host_source_count;
                EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 1u);
            } else {
                ASSERT_EQ(source_tiers[group_set_id], Tier::DISK);
                source_blocks[group_set_id] = resource.disk_block;
                ++disk_source_count;
                EXPECT_EQ(group_set->diskPool()->treeRefCount(resource.disk_block), 1u);
            }
        }
        if (failure_source == LoadFailureSource::HOST) {
            EXPECT_EQ(host_source_count, cache->groupSets().size());
            EXPECT_EQ(disk_source_count, 0u);
        } else if (failure_source == LoadFailureSource::DISK) {
            EXPECT_EQ(host_source_count, 0u);
            EXPECT_EQ(disk_source_count, cache->groupSets().size());
        }

        const auto   device_before_failure = snapshotDevicePools(manager_);
        const auto   lower_before_failure  = snapshotLowerPools(*cache, GetParam());
        const auto   stats_before_failure  = cache->getStats();
        const size_t descriptors_before_failure = pausable_engine->submittedDescriptorCount();
        const size_t batches_before_failure     = pausable_engine->submittedBatchCount();

        pausable_engine->enqueueResult(/*success=*/false);
        ASSERT_TRUE(pausable_engine->armPause());
        ScopedTransferRelease failure_release(pausable_engine);

        const int seq_size_per_block = static_cast<int>(cache_config_.seq_size_per_block);
        auto      failed_resource    = makeResource(cache_config_);
        auto      failed_token_ids =
            makeTokenIds(/*offset=*/0, 2 * seq_size_per_block, 2 * seq_size_per_block, seq_size_per_block);
        MallocInfo failed_info{failed_resource, failed_token_ids};
        failed_info.reuse_cache         = true;
        failed_info.enable_cache_lookup = true;
        const auto failed_result        = manager_->malloc(failed_info);
        ASSERT_TRUE(failed_result.success);
        EXPECT_EQ(failed_result.reuse_len, 0);
        EXPECT_EQ(failed_result.host_reuse_len, 0);
        EXPECT_EQ(failed_result.disk_reuse_len, 0);
        ASSERT_NE(failed_result.async_context, nullptr);
        const bool failure_entered = pausable_engine->waitUntilEnteredFor(
            std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout));
        if (!failure_entered) {
            pausable_engine->release();
        }
        ASSERT_TRUE(failure_entered);
        EXPECT_FALSE(failed_result.async_context->done());
        EXPECT_GT(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

        auto maybe_loading = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(maybe_loading.has_value());
        std::vector<BlockIndicesType> failed_targets(cache->groupSets().size());
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            const auto& resource  = (*maybe_loading)[0][group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
            EXPECT_EQ(resource.getTopTier(), source_tiers[group_set_id]);
            if (source_tiers[group_set_id] == Tier::HOST) {
                EXPECT_EQ(resource.host_block, source_blocks[group_set_id]);
                EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 2u);
            } else {
                EXPECT_EQ(resource.disk_block, source_blocks[group_set_id]);
                EXPECT_EQ(group_set->diskPool()->treeRefCount(resource.disk_block), 2u);
            }
            ASSERT_EQ(group_set->groupIds().size(), group_set->devicePools().size());
            for (size_t member_group_id = 0; member_group_id < group_set->groupIds().size(); ++member_group_id) {
                const int               group_id = static_cast<int>(group_set->groupIds()[member_group_id]);
                const BlockIndicesType& blocks   = failed_resource->blocks(0, group_id);
                ASSERT_FALSE(blocks.empty());
                ASSERT_FALSE(isNullBlockIdx(blocks.front()));
                failed_targets[group_set_id].push_back(blocks.front());
                EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(blocks.front()), 2u);
                ASSERT_TRUE(fillGroupBlockPayload(
                    manager_, cache_config_, group_id, blocks.front(), /*path_index=*/0, /*poison=*/true));
            }
        }

        pausable_engine->release();
        failed_result.async_context->waitDone();
        ASSERT_TRUE(failed_result.async_context->done());
        EXPECT_FALSE(failed_result.async_context->success());
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
        EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);

        const auto descriptors_after_failure = pausable_engine->descriptors();
        EXPECT_EQ(pausable_engine->submittedBatchCount() - batches_before_failure, cache->groupSets().size());
        ASSERT_EQ(descriptors_after_failure.size() - descriptors_before_failure, host_source_count + disk_source_count);
        for (size_t index = descriptors_before_failure; index < descriptors_after_failure.size(); ++index) {
            EXPECT_EQ(descriptors_after_failure[index].source_tier,
                      failure_source == LoadFailureSource::HOST ? Tier::HOST : Tier::DISK);
            EXPECT_EQ(descriptors_after_failure[index].target_tier, Tier::DEVICE);
        }

        auto maybe_failed = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(maybe_failed.has_value());
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            const auto& resource  = (*maybe_failed)[0][group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
            EXPECT_EQ(resource.getTopTier(), source_tiers[group_set_id]);
            if (source_tiers[group_set_id] == Tier::HOST) {
                EXPECT_EQ(resource.host_block, source_blocks[group_set_id]);
                EXPECT_EQ(group_set->hostPool()->treeRefCount(resource.host_block), 1u);
            } else {
                EXPECT_EQ(resource.disk_block, source_blocks[group_set_id]);
                EXPECT_EQ(group_set->diskPool()->treeRefCount(resource.disk_block), 1u);
            }
            ASSERT_EQ(group_set->devicePools().size(), failed_targets[group_set_id].size());
            for (size_t member_group_id = 0; member_group_id < group_set->devicePools().size(); ++member_group_id) {
                EXPECT_EQ(
                    group_set->devicePools()[member_group_id]->refCount(failed_targets[group_set_id][member_group_id]),
                    1u);
            }
        }
        expectPoolSnapshotsEq(lower_before_failure, snapshotLowerPools(*cache, GetParam()));
        const auto stats_after_failure = cache->getStats();
        EXPECT_EQ(stats_after_failure.tree_node_count, stats_before_failure.tree_node_count);
        EXPECT_EQ(stats_after_failure.device_heap_total_size, stats_before_failure.device_heap_total_size);
        EXPECT_EQ(stats_after_failure.host_heap_total_size, stats_before_failure.host_heap_total_size);
        EXPECT_EQ(stats_after_failure.disk_heap_total_size, stats_before_failure.disk_heap_total_size);

        manager_->free(FreeInfo{failed_resource, failed_token_ids});
        expectPoolSnapshotsEq(device_before_failure, snapshotDevicePools(manager_));
        expectPoolSnapshotsEq(lower_before_failure, snapshotLowerPools(*cache, GetParam()));
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const GroupSetPtr& group_set = cache->groupSets()[group_set_id];
            ASSERT_EQ(group_set->devicePools().size(), failed_targets[group_set_id].size());
            for (size_t member_group_id = 0; member_group_id < group_set->devicePools().size(); ++member_group_id) {
                EXPECT_FALSE(group_set->devicePools()[member_group_id]->isAllocated(
                    failed_targets[group_set_id][member_group_id]));
            }
        }

        ASSERT_TRUE(pausable_engine->armPause());
        ScopedTransferRelease retry_release(pausable_engine);
        auto                  retry_resource = makeResource(cache_config_);
        auto                  retry_token_ids =
            makeTokenIds(/*offset=*/0, 2 * seq_size_per_block, 2 * seq_size_per_block, seq_size_per_block);
        MallocInfo retry_info{retry_resource, retry_token_ids};
        retry_info.reuse_cache         = true;
        retry_info.enable_cache_lookup = true;
        const auto retry_result        = manager_->malloc(retry_info);
        ASSERT_TRUE(retry_result.success);
        EXPECT_EQ(retry_result.reuse_len, 0);
        EXPECT_EQ(retry_result.host_reuse_len, 0);
        EXPECT_EQ(retry_result.disk_reuse_len, 0);
        ASSERT_NE(retry_result.async_context, nullptr);
        const bool retry_entered = pausable_engine->waitUntilEnteredFor(
            std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout));
        if (!retry_entered) {
            pausable_engine->release();
        }
        ASSERT_TRUE(retry_entered);
        EXPECT_FALSE(retry_result.async_context->done());

        auto maybe_retry_loading = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(maybe_retry_loading.has_value());
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set = cache->groupSets()[group_set_id];
            const auto& resource  = (*maybe_retry_loading)[0][group_set_id];
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
            EXPECT_EQ(resource.getTopTier(), source_tiers[group_set_id]);
            for (size_t member_group_id = 0; member_group_id < group_set->groupIds().size(); ++member_group_id) {
                const int               group_id = static_cast<int>(group_set->groupIds()[member_group_id]);
                const BlockIndicesType& blocks   = retry_resource->blocks(0, group_id);
                ASSERT_FALSE(blocks.empty());
                ASSERT_FALSE(isNullBlockIdx(blocks.front()));
                ASSERT_TRUE(fillGroupBlockPayload(
                    manager_, cache_config_, group_id, blocks.front(), /*path_index=*/0, /*poison=*/true));
            }
        }

        pausable_engine->release();
        retry_result.async_context->waitDone();
        ASSERT_TRUE(retry_result.async_context->done());
        ASSERT_TRUE(retry_result.async_context->success()) << retry_result.async_context->errorInfo().ToString();
        block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
        EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), 0);
        ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
        ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));
        ASSERT_TRUE(requestReusesExpectedPath(
            *cache, cache_config_, seed.cache_keys, retry_resource, /*logical_reuse_blocks=*/1));
        ASSERT_TRUE(requestReusedPayloadMatchesExpectedPath(
            manager_, *cache, cache_config_, seed.cache_keys, retry_resource, /*logical_reuse_blocks=*/1));

        manager_->free(FreeInfo{retry_resource, retry_token_ids});
        ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
    }

    void runConcurrentLowerHitJoinScenario(Tier source_tier, bool transfer_success) {
        ASSERT_TRUE(source_tier == Tier::HOST || source_tier == Tier::DISK);
        ASSERT_EQ(GetParam(), source_tier == Tier::DISK ? TierLayout::HOST_DISK : TierLayout::HOST_ONLY);
        ASSERT_NO_FATAL_FAILURE(initManager(/*device_blocks=*/16));
        ASSERT_NE(manager_, nullptr);
        auto cache = manager_->blockTreeCache();

        auto engine = std::make_shared<PausableRecordingTransferEngine>(cache->groupSets());
        BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, engine);
        transfer_engine_.reset();

        const auto initial_device = snapshotDevicePools(manager_);
        const auto initial_lower  = snapshotLowerPools(*cache, GetParam());
        auto       maybe_seed     = seedDevicePrefix(manager_, cache_config_, /*token_offset=*/0, /*cached_blocks=*/1);
        ASSERT_TRUE(maybe_seed.has_value());
        auto seed = std::move(*maybe_seed);
        ASSERT_TRUE(fillSeedPayload(manager_, cache_config_, seed));

        std::vector<std::shared_ptr<IBlockPool>> device_pools;
        for (const auto& group_set : cache->groupSets()) {
            for (const DeviceBlockPoolPtr& device_pool : group_set->devicePools()) {
                device_pools.push_back(std::static_pointer_cast<IBlockPool>(device_pool));
            }
        }
        const auto device_ratio = oneUsedBlockWatermarkRatio(device_pools);
        ASSERT_TRUE(device_ratio.has_value());
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, *device_ratio);
        BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.0);
        ASSERT_TRUE(waitForPendingTasksDoneFor(
            *cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
            << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
            << " submits=" << engine->submittedDescriptorCount();

        if (source_tier == Tier::DISK) {
            std::vector<std::shared_ptr<IBlockPool>> host_pools;
            for (const auto& group_set : cache->groupSets()) {
                host_pools.push_back(group_set->hostPool());
            }
            const auto host_ratio = oneUsedBlockWatermarkRatio(host_pools);
            ASSERT_TRUE(host_ratio.has_value());
            BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, *host_ratio);
            BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
            BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::HOST, 0.0);
            ASSERT_TRUE(waitForPendingTasksDoneFor(
                *cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
                << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
                << " submits=" << engine->submittedDescriptorCount();
        }

        auto maybe_source = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(maybe_source.has_value());
        ASSERT_EQ(maybe_source->size(), 1u);
        std::vector<BlockIdxType> lower_sources(cache->groupSets().size(), NULL_BLOCK_IDX);
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set   = cache->groupSets()[group_set_id];
            const auto& resource    = (*maybe_source)[0][group_set_id];
            const auto  source_pool = lowerPoolForTier(group_set, source_tier);
            ASSERT_NE(source_pool, nullptr);
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
            ASSERT_TRUE(resource.hasTier(source_tier));
            EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
            EXPECT_EQ(resource.getTopTier(), source_tier);
            lower_sources[group_set_id] = lowerBlockForTier(resource, source_tier);
            EXPECT_EQ(source_pool->treeRefCount(lower_sources[group_set_id]), 1u);
        }

        const auto device_before_load = snapshotDevicePools(manager_);
        const auto lower_before_load  = snapshotLowerPools(*cache, GetParam());
        if (!transfer_success) {
            engine->enqueueResult(/*success=*/false);
        }
        ASSERT_TRUE(engine->armPause());
        ScopedTransferRelease release(engine);

        const int seq_size_per_block = static_cast<int>(cache_config_.seq_size_per_block);
        auto      first_resource     = makeResource(cache_config_);
        auto      first_tokens       = makeTokenIds(
            /*offset=*/0, 2 * seq_size_per_block, 2 * seq_size_per_block, seq_size_per_block);
        MallocInfo first_info{first_resource, first_tokens};
        first_info.reuse_cache           = true;
        first_info.enable_cache_lookup   = true;
        const size_t submits_before_load = engine->submittedDescriptorCount();
        const auto   first_result        = manager_->malloc(first_info);
        ASSERT_TRUE(first_result.success);
        EXPECT_EQ(first_result.reuse_len, 0);
        EXPECT_EQ(first_result.host_reuse_len, 0);
        EXPECT_EQ(first_result.disk_reuse_len, 0);
        ASSERT_NE(first_result.async_context, nullptr);
        const bool entered = engine->waitUntilEnteredCountFor(
            cache->groupSets().size(), std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout));
        if (!entered) {
            engine->release();
        }
        ASSERT_TRUE(entered);
        EXPECT_FALSE(first_result.async_context->done());
        const size_t expected_submits_before_join = submits_before_load + cache->groupSets().size();
        ASSERT_TRUE(engine->waitUntilSubmittedDescriptorCountFor(
            expected_submits_before_join,
            std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)));
        const auto   descriptors_before_join = engine->descriptors();
        const size_t submits_before_join     = descriptors_before_join.size();
        const int    pending_before_join = BlockTreeCacheTestPeer::pendingTasksForTest(*cache);
        ASSERT_GT(pending_before_join, 0);
        ASSERT_EQ(submits_before_join, expected_submits_before_join);
        for (size_t index = submits_before_load; index < descriptors_before_join.size(); ++index) {
            const auto& descriptor = descriptors_before_join[index];
            ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
            EXPECT_EQ(descriptor.source_tier, source_tier);
            EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
            EXPECT_EQ(descriptor.singleBlockAt(source_tier), lower_sources[descriptor.group_set_id]);
        }

        auto maybe_first_loading = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(maybe_first_loading.has_value());
        std::vector<BlockIndicesType> load_targets(cache->groupSets().size());
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set   = cache->groupSets()[group_set_id];
            const auto& resource    = (*maybe_first_loading)[0][group_set_id];
            const auto  source_pool = lowerPoolForTier(group_set, source_tier);
            ASSERT_NE(source_pool, nullptr);
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
            EXPECT_EQ(lowerBlockForTier(resource, source_tier), lower_sources[group_set_id]);
            EXPECT_EQ(source_pool->treeRefCount(lower_sources[group_set_id]), 2u);
            EXPECT_EQ(source_pool->referencedBlocksNum(BlockTreeRefType::LOAD), 1u);
            ASSERT_EQ(group_set->groupIds().size(), group_set->devicePools().size());
            for (size_t member_group_id = 0; member_group_id < group_set->groupIds().size(); ++member_group_id) {
                const int               group_id = static_cast<int>(group_set->groupIds()[member_group_id]);
                const BlockIndicesType& blocks   = first_resource->blocks(0, group_id);
                ASSERT_FALSE(blocks.empty());
                ASSERT_FALSE(isNullBlockIdx(blocks.front()));
                load_targets[group_set_id].push_back(blocks.front());
                EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(blocks.front()), 2u);
                EXPECT_EQ(group_set->devicePools()[member_group_id]->referencedBlocksNum(), 2u);
            }
        }

        auto second_resource = makeResource(cache_config_);
        auto second_tokens =
            makeTokenIds(/*offset=*/0, 2 * seq_size_per_block, 2 * seq_size_per_block, seq_size_per_block);
        MallocInfo second_info{second_resource, second_tokens};
        second_info.reuse_cache         = true;
        second_info.enable_cache_lookup = true;
        const auto second_result        = manager_->malloc(second_info);
        ASSERT_TRUE(second_result.success);
        EXPECT_EQ(second_result.reuse_len, 0);
        EXPECT_EQ(second_result.host_reuse_len, 0);
        EXPECT_EQ(second_result.disk_reuse_len, 0);
        ASSERT_NE(second_result.async_context, nullptr);
        EXPECT_FALSE(second_result.async_context->done());
        EXPECT_EQ(engine->submittedDescriptorCount(), submits_before_join);
        EXPECT_EQ(BlockTreeCacheTestPeer::pendingTasksForTest(*cache), pending_before_join)
            << "a joiner must not submit another load task";

        auto maybe_joined = snapshotPathResources(*cache, seed.cache_keys);
        ASSERT_TRUE(maybe_joined.has_value());
        for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
            const auto& group_set   = cache->groupSets()[group_set_id];
            const auto& resource    = (*maybe_joined)[0][group_set_id];
            const auto  source_pool = lowerPoolForTier(group_set, source_tier);
            ASSERT_NE(source_pool, nullptr);
            EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
            EXPECT_EQ(lowerBlockForTier(resource, source_tier), lower_sources[group_set_id]);
            EXPECT_EQ(source_pool->treeRefCount(lower_sources[group_set_id]), 2u)
                << "a joiner references the in-flight target, not the source";
            ASSERT_EQ(group_set->groupIds().size(), load_targets[group_set_id].size());
            for (size_t member_group_id = 0; member_group_id < group_set->groupIds().size(); ++member_group_id) {
                const int               group_id      = static_cast<int>(group_set->groupIds()[member_group_id]);
                const BlockIndicesType& first_blocks  = first_resource->blocks(0, group_id);
                const BlockIndicesType& second_blocks = second_resource->blocks(0, group_id);
                ASSERT_FALSE(first_blocks.empty());
                ASSERT_FALSE(second_blocks.empty());
                EXPECT_EQ(first_blocks.front(), load_targets[group_set_id][member_group_id]);
                EXPECT_EQ(second_blocks.front(), load_targets[group_set_id][member_group_id]);
                EXPECT_EQ(
                    group_set->devicePools()[member_group_id]->refCount(load_targets[group_set_id][member_group_id]),
                    3u);
                EXPECT_EQ(group_set->devicePools()[member_group_id]->referencedBlocksNum(), 3u);
                EXPECT_EQ(group_set->devicePools()[member_group_id]->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
            }
        }

        engine->release();
        ASSERT_TRUE(waitForAsyncContextDoneFor(
            first_result.async_context, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
            << "first context timed out; pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
            << " submits=" << engine->submittedDescriptorCount();
        ASSERT_TRUE(waitForAsyncContextDoneFor(
            second_result.async_context, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
            << "second context timed out; pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
            << " submits=" << engine->submittedDescriptorCount();
        first_result.async_context->waitDone();
        second_result.async_context->waitDone();
        ASSERT_TRUE(first_result.async_context->done());
        ASSERT_TRUE(second_result.async_context->done());
        ASSERT_TRUE(waitForPendingTasksDoneFor(
            *cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
            << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
            << " submits=" << engine->submittedDescriptorCount();
        const auto descriptors_after_load = engine->descriptors();
        ASSERT_GE(descriptors_after_load.size(), submits_before_join);
        for (size_t index = submits_before_load; index < descriptors_after_load.size(); ++index) {
            const auto& descriptor = descriptors_after_load[index];
            ASSERT_LT(descriptor.group_set_id, cache->groupSets().size());
            EXPECT_EQ(descriptor.source_tier, source_tier);
            EXPECT_EQ(descriptor.target_tier, Tier::DEVICE);
            EXPECT_EQ(descriptor.singleBlockAt(source_tier), lower_sources[descriptor.group_set_id]);
        }

        if (transfer_success) {
            ASSERT_TRUE(first_result.async_context->success()) << first_result.async_context->errorInfo().ToString();
            ASSERT_TRUE(second_result.async_context->success()) << second_result.async_context->errorInfo().ToString();
            ASSERT_NO_FATAL_FAILURE(expectPathIdleAtDevice(*cache, seed.cache_keys));
            ASSERT_TRUE(pathDevicePayloadMatches(manager_, *cache, seed.cache_keys));
            ASSERT_TRUE(requestReusesExpectedPath(
                *cache, cache_config_, seed.cache_keys, first_resource, /*logical_reuse_blocks=*/1));
            ASSERT_TRUE(requestReusesExpectedPath(
                *cache, cache_config_, seed.cache_keys, second_resource, /*logical_reuse_blocks=*/1));
            for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
                const auto& group_set   = cache->groupSets()[group_set_id];
                const auto  source_pool = lowerPoolForTier(group_set, source_tier);
                ASSERT_NE(source_pool, nullptr);
                EXPECT_FALSE(source_pool->isAllocated(lower_sources[group_set_id]));
                ASSERT_EQ(group_set->devicePools().size(), load_targets[group_set_id].size());
                for (size_t member_group_id = 0; member_group_id < group_set->devicePools().size(); ++member_group_id) {
                    EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(
                                  load_targets[group_set_id][member_group_id]),
                              3u);
                    EXPECT_EQ(group_set->devicePools()[member_group_id]->referencedBlocksNum(), 3u);
                    EXPECT_EQ(group_set->devicePools()[member_group_id]->referencedBlocksNum(BlockTreeRefType::CACHE),
                              1u);
                }
            }
            manager_->free(FreeInfo{first_resource, first_tokens});
            for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
                const GroupSetPtr& group_set = cache->groupSets()[group_set_id];
                for (size_t member_group_id = 0; member_group_id < group_set->devicePools().size(); ++member_group_id) {
                    EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(
                                  load_targets[group_set_id][member_group_id]),
                              2u);
                }
            }
            manager_->free(FreeInfo{second_resource, second_tokens});
            for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
                const GroupSetPtr& group_set = cache->groupSets()[group_set_id];
                for (size_t member_group_id = 0; member_group_id < group_set->devicePools().size(); ++member_group_id) {
                    EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(
                                  load_targets[group_set_id][member_group_id]),
                              1u);
                }
            }
        } else {
            EXPECT_FALSE(first_result.async_context->success());
            EXPECT_FALSE(second_result.async_context->success());
            auto maybe_failed = snapshotPathResources(*cache, seed.cache_keys);
            ASSERT_TRUE(maybe_failed.has_value());
            for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
                const auto& group_set   = cache->groupSets()[group_set_id];
                const auto& resource    = (*maybe_failed)[0][group_set_id];
                const auto  source_pool = lowerPoolForTier(group_set, source_tier);
                ASSERT_NE(source_pool, nullptr);
                EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
                ASSERT_TRUE(resource.hasTier(source_tier));
                EXPECT_EQ(lowerBlockForTier(resource, source_tier), lower_sources[group_set_id]);
                EXPECT_EQ(source_pool->treeRefCount(lower_sources[group_set_id]), 1u);
                ASSERT_EQ(group_set->devicePools().size(), load_targets[group_set_id].size());
                for (size_t member_group_id = 0; member_group_id < group_set->devicePools().size(); ++member_group_id) {
                    EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(
                                  load_targets[group_set_id][member_group_id]),
                              2u);
                    EXPECT_EQ(group_set->devicePools()[member_group_id]->referencedBlocksNum(), 3u);
                    EXPECT_EQ(group_set->devicePools()[member_group_id]->referencedBlocksNum(BlockTreeRefType::CACHE),
                              0u);
                }
            }
            expectPoolSnapshotsEq(lower_before_load, snapshotLowerPools(*cache, GetParam()));
            manager_->free(FreeInfo{first_resource, first_tokens});
            for (size_t group_set_id = 0; group_set_id < cache->groupSets().size(); ++group_set_id) {
                const GroupSetPtr& group_set = cache->groupSets()[group_set_id];
                for (size_t member_group_id = 0; member_group_id < group_set->devicePools().size(); ++member_group_id) {
                    EXPECT_EQ(group_set->devicePools()[member_group_id]->refCount(
                                  load_targets[group_set_id][member_group_id]),
                              1u);
                }
            }
            manager_->free(FreeInfo{second_resource, second_tokens});
            expectPoolSnapshotsEq(device_before_load, snapshotDevicePools(manager_));

            auto retry_resource = makeResource(cache_config_);
            auto retry_tokens =
                makeTokenIds(/*offset=*/0, 2 * seq_size_per_block, 2 * seq_size_per_block, seq_size_per_block);
            MallocInfo retry_info{retry_resource, retry_tokens};
            retry_info.reuse_cache         = true;
            retry_info.enable_cache_lookup = true;
            const auto retry_result        = manager_->malloc(retry_info);
            ASSERT_TRUE(retry_result.success);
            EXPECT_EQ(retry_result.reuse_len, 0);
            EXPECT_EQ(retry_result.host_reuse_len, 0);
            EXPECT_EQ(retry_result.disk_reuse_len, 0);
            ASSERT_NE(retry_result.async_context, nullptr);
            ASSERT_TRUE(
                waitForAsyncContextDoneFor(retry_result.async_context,
                                           std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
                << "retry context timed out; pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
                << " submits=" << engine->submittedDescriptorCount();
            retry_result.async_context->waitDone();
            ASSERT_TRUE(retry_result.async_context->success()) << retry_result.async_context->errorInfo().ToString();
            ASSERT_TRUE(waitForPendingTasksDoneFor(
                *cache, std::chrono::duration_cast<std::chrono::milliseconds>(kTransferWaitTimeout)))
                << "pending=" << BlockTreeCacheTestPeer::pendingTasksForTest(*cache)
                << " submits=" << engine->submittedDescriptorCount();
            ASSERT_TRUE(requestReusesExpectedPath(
                *cache, cache_config_, seed.cache_keys, retry_resource, /*logical_reuse_blocks=*/1));
            ASSERT_TRUE(requestReusedPayloadMatchesExpectedPath(
                manager_, *cache, cache_config_, seed.cache_keys, retry_resource, /*logical_reuse_blocks=*/1));
            manager_->free(FreeInfo{retry_resource, retry_tokens});
        }

        ASSERT_NO_FATAL_FAILURE(reclaimAndExpectInitialPools(manager_, initial_device, initial_lower, GetParam()));
    }

    ScopedTierDiskDirectory                             disk_dir_;
    CacheConfig                                         cache_config_;
    std::shared_ptr<KVCacheManager>                     manager_;
    std::shared_ptr<ScriptedPerRankBlockTransferEngine> transfer_engine_;

private:
    bool old_core_dump_on_exception_{false};
};

}  // namespace tier_cache_test_detail
}  // namespace rtp_llm::test
