#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

class CoreDumpGuard {
public:
    CoreDumpGuard(): old_(StaticConfig::user_ft_core_dump_on_exception) {
        StaticConfig::user_ft_core_dump_on_exception = false;
    }
    ~CoreDumpGuard() {
        StaticConfig::user_ft_core_dump_on_exception = old_;
    }

private:
    bool old_;
};

CacheConfig makeSingleConfig() {
    return test::makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                          /*block_num=*/8,
                                          /*tokens_per_block=*/4,
                                          DataType::TYPE_FP16,
                                          /*local_head_num_kv=*/1,
                                          /*size_per_head=*/8);
}

CacheConfig makeSwaConfig() {
    CacheConfig config;
    config.dtype                       = DataType::TYPE_FP16;
    config.layer_num                   = 2;
    config.layer_all_num               = 2;
    config.block_num                   = 8;
    config.seq_size_per_block          = 4;
    config.kernel_seq_size_per_block   = 4;
    config.group_layer_num             = 2;
    config.use_independent_block_pools = true;

    auto spec = test::makeResolvedMhaSpec(
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4);
    auto policy                  = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.enable_prefix_reuse   = true;
    policy.sliding_window_size   = 128;
    const size_t stride          = spec->block_size_bytes();
    config.kv_block_stride_bytes = stride;
    config.kv_block_size_bytes   = stride;
    config.block_size_bytes      = stride;
    config.layer_to_block_stride_bytes.assign(2, static_cast<int>(stride));
    test::configureIndexedTestGroups(config, {spec}, {{0, 1}}, {CacheGroupType::SWA}, {policy});
    config.setGroupBlockLayout({8}, {stride}, {0});
    return config;
}

CacheConfig makeHybridConfig(bool independent_pools, bool disable_linear_reuse = false) {
    CacheConfig config;
    config.dtype                       = DataType::TYPE_FP16;
    config.layer_num                   = 4;
    config.layer_all_num               = 4;
    config.block_num                   = 8;
    config.seq_size_per_block          = 4;
    config.kernel_seq_size_per_block   = 4;
    config.linear_step                 = 2;
    config.group_layer_num             = 2;
    config.use_independent_block_pools = independent_pools;

    auto linear_spec = test::makeResolvedLinearSpec(DataType::TYPE_FP16,
                                                    /*local_num_k_heads=*/1,
                                                    /*local_num_v_heads=*/1,
                                                    /*head_k_dim=*/1,
                                                    /*head_v_dim=*/1,
                                                    /*conv_kernel_dim=*/2,
                                                    /*seq_size_per_block=*/4,
                                                    DataType::TYPE_FP16,
                                                    DataType::TYPE_FP16);
    auto full_spec   = test::makeResolvedMhaSpec(
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4);

    auto linear_policy                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    linear_policy.enable_prefix_reuse = !disable_linear_reuse;
    auto full_policy                  = defaultCacheGroupPolicy(CacheGroupType::FULL);
    test::configureIndexedTestGroups(config,
                                     {linear_spec, full_spec},
                                     {{1, 3}, {0, 2}},
                                     {CacheGroupType::LINEAR, CacheGroupType::FULL},
                                     {linear_policy, full_policy});

    const size_t linear_stride   = linear_spec->block_size_bytes();
    const size_t full_stride     = full_spec->block_size_bytes();
    config.kv_block_stride_bytes = std::max(linear_stride, full_stride);
    config.kv_block_size_bytes   = 2 * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;
    config.block_size_bytes      = config.kv_block_size_bytes;
    config.layer_to_block_stride_bytes.assign(4, static_cast<int>(config.kv_block_stride_bytes));
    config.setGroupBlockLayout({8, 8}, {linear_stride, full_stride}, {0, 0});
    return config;
}

CacheConfig makeSharedBackingCascadeConfig() {
    CacheConfig config = makeHybridConfig(/*independent_pools=*/false);

    const std::shared_ptr<const KVCacheSpec>& linear_spec     = config.specForGroup(0);
    const std::shared_ptr<const KVCacheSpec>& full_spec       = config.specForGroup(1);
    const CacheGroupPolicy                    linear_policy   = config.policyForGroup(0);
    const CacheGroupPolicy                    full_policy     = config.policyForGroup(1);
    const size_t                              linear_stride   = config.kvBlockStrideBytesForGroup(0);
    const size_t                              full_stride     = config.kvBlockStrideBytesForGroup(1);
    std::vector<KVCacheSpecPtr>               reordered_specs = {
        full_spec->clone(),
        linear_spec->clone(),
    };

    // Put FULL first so a watermark primary plan cascades to LINEAR.
    // The HybridType allocator still uses one physical DeviceBlockPool for both logical groups.
    test::configureIndexedTestGroups(config,
                                     reordered_specs,
                                     {{0, 2}, {1, 3}},
                                     {CacheGroupType::FULL, CacheGroupType::LINEAR},
                                     {full_policy, linear_policy});
    config.setGroupBlockLayout({8, 8}, {full_stride, linear_stride}, {0, 0});
    return config;
}

CacheConfig makeIncompatibleFullGroupsConfig() {
    CacheConfig config;
    config.dtype                       = DataType::TYPE_FP16;
    config.layer_num                   = 2;
    config.layer_all_num               = 2;
    config.block_num                   = 9;
    config.seq_size_per_block          = 4;
    config.kernel_seq_size_per_block   = 2;
    config.group_layer_num             = 1;
    config.use_independent_block_pools = true;

    auto first = test::makeResolvedMhaSpec(
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/4, /*seq_size_per_block=*/4);
    auto second = test::makeResolvedMhaSpec(
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/2);
    auto first_policy  = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto second_policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    test::configureIndexedTestGroups(config,
                                     {first, second},
                                     {{0}, {1}},
                                     {CacheGroupType::FULL, CacheGroupType::FULL},
                                     {first_policy, second_policy});

    const size_t first_stride    = first->block_size_bytes();
    const size_t second_stride   = second->block_size_bytes();
    config.kv_block_stride_bytes = std::max(first_stride, second_stride);
    config.kv_block_size_bytes   = config.kv_block_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes;
    config.layer_to_block_stride_bytes.assign(2, static_cast<int>(config.kv_block_stride_bytes));
    config.setGroupBlockLayout({9, 7}, {first_stride, second_stride}, {0, 0});
    return config;
}

CacheConfig makeCompatibleFullGroupsConfig() {
    CacheConfig config;
    config.dtype                       = DataType::TYPE_FP16;
    config.layer_num                   = 2;
    config.layer_all_num               = 2;
    config.block_num                   = 8;
    config.seq_size_per_block          = 4;
    config.kernel_seq_size_per_block   = 4;
    config.group_layer_num             = 1;
    config.use_independent_block_pools = true;

    auto first = test::makeResolvedMhaSpec(
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4);
    auto second = test::makeResolvedMhaSpec(
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4);
    auto first_policy  = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto second_policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    test::configureIndexedTestGroups(config,
                                     {first, second},
                                     {{0}, {1}},
                                     {CacheGroupType::FULL, CacheGroupType::FULL},
                                     {first_policy, second_policy});

    const size_t stride          = first->block_size_bytes();
    config.kv_block_stride_bytes = stride;
    config.kv_block_size_bytes   = stride;
    config.block_size_bytes      = stride;
    config.layer_to_block_stride_bytes.assign(2, static_cast<int>(stride));
    config.setGroupBlockLayout({8, 8}, {stride, stride}, {0, 0});
    return config;
}

CacheConfig makeCompatibleSwaGroupsConfig(int second_window) {
    CacheConfig config;
    config.dtype                       = DataType::TYPE_FP16;
    config.layer_num                   = 2;
    config.layer_all_num               = 2;
    config.block_num                   = 8;
    config.seq_size_per_block          = 4;
    config.kernel_seq_size_per_block   = 4;
    config.group_layer_num             = 1;
    config.use_independent_block_pools = true;

    auto first = test::makeResolvedMhaSpec(
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4);
    auto second = test::makeResolvedMhaSpec(
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4);
    auto first_policy                 = defaultCacheGroupPolicy(CacheGroupType::SWA);
    first_policy.enable_prefix_reuse  = true;
    first_policy.sliding_window_size  = 128;
    auto second_policy                = defaultCacheGroupPolicy(CacheGroupType::SWA);
    second_policy.enable_prefix_reuse = true;
    second_policy.sliding_window_size = second_window;
    test::configureIndexedTestGroups(
        config, {first, second}, {{0}, {1}}, {CacheGroupType::SWA, CacheGroupType::SWA}, {first_policy, second_policy});

    const size_t stride          = first->block_size_bytes();
    config.kv_block_stride_bytes = stride;
    config.kv_block_size_bytes   = stride;
    config.block_size_bytes      = stride;
    config.layer_to_block_stride_bytes.assign(2, static_cast<int>(stride));
    config.setGroupBlockLayout({8, 8}, {stride, stride}, {0, 0});
    return config;
}

CacheConfig makeReusableGroupsAroundDisabledConfig() {
    CacheConfig config;
    config.dtype                       = DataType::TYPE_FP16;
    config.layer_num                   = 3;
    config.layer_all_num               = 3;
    config.block_num                   = 8;
    config.seq_size_per_block          = 4;
    config.kernel_seq_size_per_block   = 4;
    config.group_layer_num             = 1;
    config.use_independent_block_pools = true;

    std::vector<std::shared_ptr<KVCacheSpec>> specs;
    std::vector<CacheGroupPolicy>             policies;
    for (size_t group_id = 0; group_id < 3; ++group_id) {
        specs.push_back(test::makeResolvedMhaSpec(
            DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4));
        auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
        policy.enable_prefix_reuse = group_id != 1;
        policies.push_back(policy);
    }
    test::configureIndexedTestGroups(
        config, specs, {{0}, {1}, {2}}, {CacheGroupType::FULL, CacheGroupType::FULL, CacheGroupType::FULL}, policies);

    const size_t stride          = config.specForGroup(0)->block_size_bytes();
    config.kv_block_stride_bytes = stride;
    config.kv_block_size_bytes   = stride;
    config.block_size_bytes      = stride;
    config.layer_to_block_stride_bytes.assign(3, static_cast<int>(stride));
    config.setGroupBlockLayout({8, 8, 8}, {stride, stride, stride}, {0, 0, 0});
    return config;
}

template<typename Allocator>
std::shared_ptr<Allocator> initAllocator(const CacheConfig& config) {
    auto allocator = std::make_shared<Allocator>(config);
    EXPECT_TRUE(allocator->init());
    return allocator;
}

class GroupViewHybridPoolAllocator: public HybridPoolKVCacheAllocator {
public:
    using HybridPoolKVCacheAllocator::HybridPoolKVCacheAllocator;

    void overrideGroups(std::vector<KVCacheGroupPtr> groups) {
        exposed_groups_ = std::move(groups);
    }

    std::vector<KVCacheGroupPtr> cacheGroups() const override {
        return exposed_groups_.empty() ? HybridPoolKVCacheAllocator::cacheGroups() : exposed_groups_;
    }

private:
    std::vector<KVCacheGroupPtr> exposed_groups_;
};

std::shared_ptr<GroupViewHybridPoolAllocator> initViewAllocator(const CacheConfig& config) {
    auto allocator = std::make_shared<GroupViewHybridPoolAllocator>(config);
    EXPECT_TRUE(allocator->init());
    return allocator;
}

KVCacheGroupPtr cloneGroupWithId(const KVCacheGroupPtr& source, int group_id) {
    GroupBase group  = source->config();
    auto      result = std::make_shared<FullKVCacheGroup>(std::move(group), source->blockPool(), group_id);
    EXPECT_TRUE(result->init());
    return result;
}

KVCacheGroupPtr cloneGroupWithBlockNum(const KVCacheGroupPtr& source, uint32_t block_num) {
    GroupBase group = source->config();
    group.block_num = block_num;
    auto result     = std::make_shared<FullKVCacheGroup>(std::move(group), source->blockPool(), source->group_id());
    EXPECT_TRUE(result->init());
    return result;
}

KVCacheGroupPtr cloneGroupWithPool(const KVCacheGroupPtr& source, const DeviceBlockPoolPtr& pool) {
    GroupBase group  = source->config();
    auto      result = std::make_shared<FullKVCacheGroup>(std::move(group), pool, source->group_id());
    EXPECT_TRUE(result->init());
    return result;
}

void expectFactoryRejects(const CacheConfig&                       config,
                          const std::shared_ptr<KVCacheAllocator>& allocator,
                          const KVCacheConfig&                     kv_cache_config = {}) {
    CoreDumpGuard guard;
    try {
        EXPECT_EQ(createBlockTreeCache(config, kv_cache_config, allocator), nullptr);
    } catch (const std::exception&) {
        SUCCEED();
    }
    EXPECT_EQ(allocator->blockTreeCache(), nullptr);
}

void expectTargetGroupsBoundById(const BlockTreeCachePtr& cache, const KVCacheAllocatorPtr& allocator) {
    ASSERT_NE(cache, nullptr);
    const auto groups = allocator->cacheGroups();
    ASSERT_FALSE(cache->groupSets().empty());
    for (const auto& target_group : groups) {
        ASSERT_NE(target_group, nullptr);
        ASSERT_GE(target_group->group_id(), 0);
        const size_t group_id = static_cast<size_t>(target_group->group_id());
        if (!target_group->prefixReuseEnabled()) {
            continue;
        }

        const auto group_set_it =
            std::find_if(cache->groupSets().begin(), cache->groupSets().end(), [&](const GroupSetPtr& group_set) {
                return group_set != nullptr
                       && std::find(group_set->groupIds().begin(), group_set->groupIds().end(), group_id)
                              != group_set->groupIds().end();
            });
        ASSERT_NE(group_set_it, cache->groupSets().end()) << group_id;
        const auto local_it =
            std::find((*group_set_it)->groupIds().begin(), (*group_set_it)->groupIds().end(), group_id);
        ASSERT_NE(local_it, (*group_set_it)->groupIds().end());
        const size_t member_group_id =
            static_cast<size_t>(std::distance((*group_set_it)->groupIds().begin(), local_it));
        const auto& device_pool = (*group_set_it)->devicePools()[member_group_id];
        ASSERT_NE(device_pool, nullptr);
        EXPECT_EQ(device_pool.get(), target_group->blockPool().get());
    }
}

std::vector<BlockIdxType>
insertOneKeyThroughAllocator(const CacheConfig& config, const KVCacheAllocatorPtr& allocator, CacheKeyType key) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(config.topologyPtr());
    resource->setBatchCacheKeys(0, CacheKeysType{key});

    std::vector<BlockIdxType> blocks(static_cast<size_t>(config.groupNums()), NULL_BLOCK_IDX);
    const auto                groups = allocator->cacheGroups();
    EXPECT_EQ(groups.size(), blocks.size());
    for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
        const auto& pool      = groups[group_id]->blockPool();
        const auto  allocated = pool->malloc(1);
        EXPECT_TRUE(allocated.has_value());
        if (!allocated.has_value()) {
            continue;
        }
        pool->incRef(*allocated, BlockRefType::REQUEST);
        blocks[group_id] = allocated->front();
        resource->setBatchBlocks(0, static_cast<int>(group_id), BlockIndicesType{allocated->front()});
    }
    allocator->insertIntoCache(InsertInfo{resource, nullptr, /*is_resident=*/false});
    return blocks;
}

void releaseInsertedRequestBlocks(const KVCacheAllocatorPtr& allocator, const std::vector<BlockIdxType>& blocks) {
    const auto groups = allocator->cacheGroups();
    ASSERT_EQ(groups.size(), blocks.size());
    for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
        if (!isNullBlockIdx(blocks[group_id])) {
            groups[group_id]->release({blocks[group_id]}, BlockRefType::REQUEST);
        }
    }
}

void writeDevicePattern(void* address, size_t bytes, uint8_t pattern) {
    ASSERT_NE(address, nullptr);
    auto device = torch::from_blob(
        address, {static_cast<int64_t>(bytes)}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto host = torch::full({static_cast<int64_t>(bytes)}, pattern, torch::TensorOptions(torch::kUInt8));
    device.copy_(host);
    runtimeSyncAndCheck();
}

void expectDevicePattern(const void* address, size_t bytes, uint8_t pattern) {
    ASSERT_NE(address, nullptr);
    auto        device = torch::from_blob(const_cast<void*>(address),
                                          {static_cast<int64_t>(bytes)},
                                   torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
    auto        host   = device.cpu();
    const auto* data   = host.data_ptr<uint8_t>();
    for (size_t i = 0; i < bytes; ++i) {
        ASSERT_EQ(data[i], pattern) << "byte=" << i;
    }
}

class BlockTreeCacheFactoryTest: public DeviceTestBase {};

class InlineFactoryExecutor: public StorageBackendExecutor {
public:
    bool start() override {
        return true;
    }
    bool submit(Task task) override {
        task();
        return true;
    }
    void shutdown() noexcept override {}
};

class ShutdownCountingStorageBackend: public StorageBackend {
public:
    ShutdownCountingStorageBackend(std::shared_ptr<size_t> shutdown_count, std::shared_ptr<size_t> resolved_count):
        StorageBackend(std::make_shared<InlineFactoryExecutor>()),
        shutdown_count_(std::move(shutdown_count)),
        resolved_count_(std::move(resolved_count)) {}

    ~ShutdownCountingStorageBackend() override {
        const auto buffers = convertIndexToBuffer(/*layer_id=*/0, /*group_id=*/0, /*block_id=*/0);
        if (!buffers.empty() && buffers.front().addr != nullptr) {
            ++*resolved_count_;
        }
        ++*shutdown_count_;
        shutdown();
    }

protected:
    bool initImpl() override {
        return true;
    }
    StorageMatchResult matchImpl(const StorageRequest& request) override {
        return {request.handles.size(), nullptr};
    }
    void readImpl(const StorageRequest&, const std::shared_ptr<StorageBackendMatchMeta>&) override {}
    void writeImpl(const StorageRequest&) override {}

private:
    std::shared_ptr<size_t> shutdown_count_;
    std::shared_ptr<size_t> resolved_count_;
};

class CountingStorageBackend: public StorageBackend {
public:
    CountingStorageBackend(): StorageBackend(std::make_shared<InlineFactoryExecutor>()) {}
    ~CountingStorageBackend() override {
        shutdown();
    }
    size_t matchCalls() const {
        return match_calls_;
    }
    const CacheKeysType& matchKeys() const {
        return match_keys_;
    }
    size_t localMatchedBlocks() const {
        return local_matched_blocks_;
    }
    const std::vector<size_t>& matchHandleCounts() const {
        return match_handle_counts_;
    }

protected:
    bool initImpl() override {
        return true;
    }
    StorageMatchResult matchImpl(const StorageRequest& request) override {
        ++match_calls_;
        match_keys_           = *request.keys;
        local_matched_blocks_ = request.local_matched_blocks_num;
        match_handle_counts_.clear();
        for (const auto& handles : request.handles) {
            match_handle_counts_.push_back(handles.size());
        }
        return {local_matched_blocks_, nullptr};
    }
    void readImpl(const StorageRequest&, const std::shared_ptr<StorageBackendMatchMeta>&) override {}
    void writeImpl(const StorageRequest&) override {}

private:
    size_t              match_calls_{0};
    CacheKeysType       match_keys_;
    size_t              local_matched_blocks_{0};
    std::vector<size_t> match_handle_counts_;
};

class FailingInitStorageBackend: public StorageBackend {
public:
    size_t initCalls() const {
        return init_calls_;
    }

protected:
    bool initImpl() override {
        ++init_calls_;
        return false;
    }
    StorageMatchResult matchImpl(const StorageRequest&) override {
        return {};
    }
    void readImpl(const StorageRequest&, const std::shared_ptr<StorageBackendMatchMeta>&) override {}
    void writeImpl(const StorageRequest&) override {}

private:
    size_t init_calls_{0};
};

}  // namespace

TEST(BlockTreeCacheFactoryUtilityTest, UsableBlockCountReservesBlockZeroWithinBudget) {
    EXPECT_EQ(computeHostUsableBlockCount(4 * 4096, 4096), 3u);
    EXPECT_EQ(computeHostUsableBlockCount(4096, 4096), 0u);
    EXPECT_EQ(computeHostUsableBlockCount(100, 4096), 0u);
    EXPECT_EQ(computeHostUsableBlockCount(4096, 0), 0u);
}

TEST(BlockTreeCacheFactoryUtilityTest, ResolveDiskMountPathUsesLocalRankAndRejectsInvalidShape) {
    EXPECT_EQ(resolveDiskMountPath("/mnt/d0,/mnt/d1,/mnt/d2", 3, 0), "/mnt/d0");
    EXPECT_EQ(resolveDiskMountPath(" /mnt/d0 , /mnt/d1 ", 2, 1), "/mnt/d1");

    CoreDumpGuard guard;
    EXPECT_ANY_THROW(resolveDiskMountPath("/mnt/d0,/mnt/d1", 3, 0));
    EXPECT_ANY_THROW(resolveDiskMountPath("", 1, 0));
    EXPECT_ANY_THROW(resolveDiskMountPath("/mnt/d0,/mnt/d1", 2, 2));
    EXPECT_ANY_THROW(resolveDiskMountPath("/mnt/d0,/mnt/d1", 2, -1));
}

TEST_F(BlockTreeCacheFactoryTest, SingleTypeBindsExistingTargetGroupAndPool) {
    const auto config    = makeSingleConfig();
    auto       allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_EQ(allocator->cacheGroups().size(), 1u);
    expectTargetGroupsBoundById(cache, allocator);
}

TEST_F(BlockTreeCacheFactoryTest, RemoteBackendResolverDoesNotKeepAttachedCacheAndAllocatorAlive) {
    const auto    config         = makeSingleConfig();
    auto          allocator      = initAllocator<SingleTypeKVCacheAllocator>(config);
    auto          shutdown_count = std::make_shared<size_t>(0);
    auto          resolved_count = std::make_shared<size_t>(0);
    auto          backend        = std::make_shared<ShutdownCountingStorageBackend>(shutdown_count, resolved_count);
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_remote_cache = true;
    auto cache = createBlockTreeCache(config, kv_cache_config, allocator, ParallelismConfig{}, backend);
    ASSERT_NE(cache, nullptr);
    allocator->attachBlockTreeCache(cache);

    std::weak_ptr<KVCacheAllocator> weak_allocator = allocator;
    std::weak_ptr<BlockTreeCache>   weak_cache     = cache;
    std::weak_ptr<StorageBackend>   weak_backend   = backend;
    cache.reset();
    backend.reset();
    allocator.reset();

    EXPECT_TRUE(weak_allocator.expired());
    EXPECT_TRUE(weak_cache.expired());
    EXPECT_TRUE(weak_backend.expired());
    EXPECT_EQ(*resolved_count, 1u);
    EXPECT_EQ(*shutdown_count, 1u);
}

TEST_F(BlockTreeCacheFactoryTest, RemoteMatchSkipsBackendWhenNoGroupSupportsPrefixReuse) {
    auto config   = makeSingleConfig();
    auto policies = config.groupPoliciesSnapshot();
    ASSERT_EQ(policies.size(), 1u);
    policies[0].enable_prefix_reuse = false;
    config.setGroupPolicies(std::move(policies));

    auto          allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
    auto          backend   = std::make_shared<CountingStorageBackend>();
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_remote_cache = true;
    auto cache = createBlockTreeCache(config, kv_cache_config, allocator, ParallelismConfig{}, backend);
    ASSERT_NE(cache, nullptr);
    ASSERT_TRUE(cache->groupSets().empty());

    const BlockTreeMatchResult result = cache->match({1, 2});
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_EQ(backend->matchCalls(), 0u);
}

TEST_F(BlockTreeCacheFactoryTest, RemoteMatchReceivesCompleteKeysAndExplicitLocalBoundary) {
    const auto    config    = makeSingleConfig();
    auto          allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
    auto          backend   = std::make_shared<CountingStorageBackend>();
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_remote_cache = true;
    auto cache = createBlockTreeCache(config, kv_cache_config, allocator, ParallelismConfig{}, backend);
    ASSERT_NE(cache, nullptr);
    allocator->attachBlockTreeCache(cache);

    constexpr CacheKeyType local_key      = 700;
    const auto             request_blocks = insertOneKeyThroughAllocator(config, allocator, local_key);
    BlockTreeMatchResult   result         = cache->match({local_key, local_key + 1});
    EXPECT_EQ(result.matched_device_blocks, 1u);
    auto context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(context, nullptr);
    size_t matched_blocks = 0;
    context->setMatchCallback([&](LoadAsyncContext& current, size_t matched) {
        matched_blocks = matched;
        return current.commit();
    });
    context->startBackendMatch();

    EXPECT_EQ(backend->matchCalls(), 1u);
    EXPECT_EQ(backend->matchKeys(), (CacheKeysType{local_key, local_key + 1}));
    EXPECT_EQ(backend->localMatchedBlocks(), 1u);
    EXPECT_EQ(backend->matchHandleCounts(), (std::vector<size_t>{1, 1}));
    EXPECT_EQ(matched_blocks, 1u);
    EXPECT_TRUE(context->success());

    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);
    releaseInsertedRequestBlocks(allocator, request_blocks);
}

TEST_F(BlockTreeCacheFactoryTest, RemoteBackendInitFailureIsFatal) {
    const auto    config    = makeSingleConfig();
    auto          allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
    auto          backend   = std::make_shared<FailingInitStorageBackend>();
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_remote_cache = true;

    CoreDumpGuard guard;
    EXPECT_ANY_THROW(createBlockTreeCache(config, kv_cache_config, allocator, ParallelismConfig{}, backend));
    EXPECT_EQ(backend->initCalls(), 1u);
}

TEST_F(BlockTreeCacheFactoryTest, RemoteBackendIsDroppedWhenRemoteCacheDisabled) {
    const auto    config    = makeSingleConfig();
    auto          allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
    auto          backend   = std::make_shared<CountingStorageBackend>();
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_remote_cache = false;

    auto cache = createBlockTreeCache(config, kv_cache_config, allocator, ParallelismConfig{}, backend);
    ASSERT_NE(cache, nullptr);
    EXPECT_FALSE(cache->isRemoteCacheEnabled());
    EXPECT_EQ(cache->storageBackend(), nullptr);
    EXPECT_EQ(backend->matchCalls(), 0u);

    const BlockTreeMatchResult result = cache->match({1, 2});
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_EQ(backend->matchCalls(), 0u);
}

TEST_F(BlockTreeCacheFactoryTest, SwaGroupSetUsesDeclaredPolicyWindow) {
    const auto config = makeSwaConfig();
    ASSERT_EQ(config.policyForGroup(0).sliding_window_size, 128);
    auto allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->groupSets().size(), 1u);
    auto swa_group = std::dynamic_pointer_cast<SWAGroupSet>(cache->groupSets().front());
    ASSERT_NE(swa_group, nullptr);
    EXPECT_EQ(swa_group->groupIds(), (std::vector<size_t>{0}));
    EXPECT_EQ(swa_group->slidingWindowSize(), 128u);
}

TEST_F(BlockTreeCacheFactoryTest, HybridTypeBindsExistingTargetGroupsById) {
    const auto config    = makeHybridConfig(/*independent_pools=*/false);
    auto       allocator = initAllocator<HybridTypeKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_EQ(allocator->cacheGroups().size(), 2u);
    expectTargetGroupsBoundById(cache, allocator);
}

TEST_F(BlockTreeCacheFactoryTest, HybridPoolBindsIndependentPoolsAndNonContiguousLayerViews) {
    const auto config    = makeHybridConfig(/*independent_pools=*/true);
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_EQ(allocator->cacheGroups().size(), 2u);
    ASSERT_NE(allocator->cacheGroups()[0]->blockPool(), allocator->cacheGroups()[1]->blockPool());
    expectTargetGroupsBoundById(cache, allocator);

    for (const auto& group : allocator->cacheGroups()) {
        const auto& pool  = group->blockPool();
        const auto  block = pool->malloc(1);
        ASSERT_TRUE(block.has_value());
        ASSERT_EQ(block->size(), 1u);
        pool->incRef(*block, BlockRefType::REQUEST);
        const int  global_layer = group->config().layer_ids.back();
        const int  local_layer  = static_cast<int>(group->config().layer_ids.size() - 1);
        const auto via_group    = group->convertIndexToAddr(global_layer, block->front());
        const auto via_backing  = pool->convertIndexToAddr(local_layer, block->front());
        EXPECT_EQ(via_group.kv_addr, via_backing.kv_addr);
        pool->decRef(*block, BlockRefType::REQUEST);
    }
}

TEST_F(BlockTreeCacheFactoryTest, PerRankBlockTransferEnginePreservesNonContiguousGlobalLayerProjectionRoundTrip) {
    const auto    config    = makeHybridConfig(/*independent_pools=*/true);
    auto          allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_host_cache        = true;
    kv_cache_config.host_cache_size_mb       = 1;
    kv_cache_config.enable_host_cache_pinned = false;
    auto cache                               = createBlockTreeCache(config, kv_cache_config, allocator);
    ASSERT_NE(cache, nullptr);

    const auto target_groups = allocator->cacheGroups();
    ASSERT_EQ(target_groups.size(), 2u);
    const auto& full_group = target_groups[1];
    ASSERT_NE(full_group, nullptr);
    ASSERT_EQ(full_group->config().layer_ids, (std::vector<int>{0, 2}));

    const auto group_set_it =
        std::find_if(cache->groupSets().begin(), cache->groupSets().end(), [](const GroupSetPtr& group_set) {
            return group_set != nullptr && group_set->groupIds() == std::vector<size_t>{1};
        });
    ASSERT_NE(group_set_it, cache->groupSets().end());
    const auto& group_set = *group_set_it;
    ASSERT_NE(group_set->hostPool(), nullptr);
    EXPECT_FALSE(group_set->hostPool()->isPinned());
    ASSERT_EQ(group_set->devicePools().size(), 1u);

    const auto device_blocks = full_group->blockPool()->malloc(1);
    ASSERT_TRUE(device_blocks.has_value());
    ASSERT_EQ(device_blocks->size(), 1u);
    full_group->blockPool()->incRef(*device_blocks, BlockRefType::REQUEST);
    const BlockIdxType device_block = device_blocks->front();
    const BlockIdxType host_block   = group_set->allocateSingleBlock(Tier::HOST, BlockRefType::REQUEST);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    const size_t layer_bytes = full_group->config().kv_block_stride_bytes + full_group->config().kv_scale_stride_bytes;
    ASSERT_GT(layer_bytes, 0u);
    writeDevicePattern(full_group->convertIndexToAddr(/*global_layer=*/0, device_block).kv_addr, layer_bytes, 0x31);
    writeDevicePattern(full_group->convertIndexToAddr(/*global_layer=*/2, device_block).kv_addr, layer_bytes, 0x72);

    EXPECT_TRUE(
        cache->executeTransfer({TransferDescriptor::deviceToHost(group_set->groupSetId(), {device_block}, host_block)}));
    writeDevicePattern(full_group->convertIndexToAddr(/*global_layer=*/0, device_block).kv_addr, layer_bytes, 0x00);
    writeDevicePattern(full_group->convertIndexToAddr(/*global_layer=*/2, device_block).kv_addr, layer_bytes, 0x00);
    EXPECT_TRUE(
        cache->executeTransfer({TransferDescriptor::hostToDevice(group_set->groupSetId(), host_block, {device_block})}));

    expectDevicePattern(full_group->convertIndexToAddr(/*global_layer=*/0, device_block).kv_addr, layer_bytes, 0x31);
    expectDevicePattern(full_group->convertIndexToAddr(/*global_layer=*/2, device_block).kv_addr, layer_bytes, 0x72);

    group_set->releaseSingleBlock(Tier::HOST, host_block, BlockRefType::REQUEST);
    full_group->blockPool()->decRef(*device_blocks, BlockRefType::REQUEST);
}

TEST_F(BlockTreeCacheFactoryTest, ReorderedAllocatorGroupsStillMapByEmbeddedGroupId) {
    const auto config    = makeHybridConfig(/*independent_pools=*/true);
    auto       allocator = initViewAllocator(config);
    auto       groups    = allocator->HybridPoolKVCacheAllocator::cacheGroups();
    ASSERT_EQ(groups.size(), 2u);
    std::reverse(groups.begin(), groups.end());
    allocator->overrideGroups(groups);

    auto cache = createBlockTreeCache(config, KVCacheConfig{}, allocator);
    expectTargetGroupsBoundById(cache, allocator);
    ASSERT_EQ(cache->groupSets().size(), 2u);
    EXPECT_EQ(cache->groupSets()[0]->groupIds(), (std::vector<size_t>{0}));
    EXPECT_EQ(cache->groupSets()[1]->groupIds(), (std::vector<size_t>{1}));
}

TEST_F(BlockTreeCacheFactoryTest, DuplicateMissingAndOutOfRangeGroupIdsFailClosed) {
    const auto config = makeHybridConfig(/*independent_pools=*/true);

    {
        auto allocator = initViewAllocator(config);
        auto groups    = allocator->HybridPoolKVCacheAllocator::cacheGroups();
        allocator->overrideGroups({groups[0], groups[0]});
        expectFactoryRejects(config, allocator);
    }
    {
        auto allocator = initViewAllocator(config);
        auto groups    = allocator->HybridPoolKVCacheAllocator::cacheGroups();
        allocator->overrideGroups({groups[0]});
        expectFactoryRejects(config, allocator);
    }
    {
        auto allocator = initViewAllocator(config);
        auto groups    = allocator->HybridPoolKVCacheAllocator::cacheGroups();
        allocator->overrideGroups({groups[0], cloneGroupWithId(groups[1], 2)});
        expectFactoryRejects(config, allocator);
    }
    {
        auto allocator = initViewAllocator(config);
        auto groups    = allocator->HybridPoolKVCacheAllocator::cacheGroups();
        allocator->overrideGroups({groups[0], cloneGroupWithId(groups[1], -1)});
        expectFactoryRejects(config, allocator);
    }
    {
        auto allocator = initViewAllocator(config);
        auto groups    = allocator->HybridPoolKVCacheAllocator::cacheGroups();
        allocator->overrideGroups({cloneGroupWithId(groups[0], 1), groups[1]});
        expectFactoryRejects(config, allocator);
    }
}

TEST_F(BlockTreeCacheFactoryTest, AllocatorConfigAndDirectPoolMismatchesFailClosed) {
    const auto config = makeHybridConfig(/*independent_pools=*/true);

    {
        auto allocator = initViewAllocator(config);
        auto groups    = allocator->HybridPoolKVCacheAllocator::cacheGroups();
        ASSERT_EQ(groups.size(), 2u);
        allocator->overrideGroups({groups[0], cloneGroupWithBlockNum(groups[1], groups[1]->config().block_num + 1)});
        expectFactoryRejects(config, allocator);
    }
    {
        auto allocator = initViewAllocator(config);
        auto groups    = allocator->HybridPoolKVCacheAllocator::cacheGroups();
        ASSERT_EQ(groups.size(), 2u);
        allocator->overrideGroups({groups[0], cloneGroupWithPool(groups[1], groups[0]->blockPool())});
        expectFactoryRejects(config, allocator);
    }
}

TEST_F(BlockTreeCacheFactoryTest, PrefixReuseDisabledGroupStaysAllocatorOwnedButIsExcludedFromTree) {
    const auto config    = makeHybridConfig(/*independent_pools=*/true, /*disable_linear_reuse=*/true);
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(allocator->cacheGroups().size(), 2u);
    ASSERT_EQ(cache->groupSets().size(), 1u);
    EXPECT_EQ(cache->groupSets()[0]->groupIds(), (std::vector<size_t>{1}));
}

TEST_F(BlockTreeCacheFactoryTest, SameTypeGroupsWithDifferentPolicyShapeSeqAndStrideAreNeverAggregated) {
    const auto config    = makeIncompatibleFullGroupsConfig();
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->groupSets().size(), 2u);
    EXPECT_EQ(cache->groupSets()[0]->groupIds(), (std::vector<size_t>{0}));
    EXPECT_EQ(cache->groupSets()[1]->groupIds(), (std::vector<size_t>{1}));
    EXPECT_NE(cache->groupSets()[0]->devicePools()[0], cache->groupSets()[1]->devicePools()[0]);
}

TEST_F(BlockTreeCacheFactoryTest, CompatibleGroupsAggregate) {
    const auto    config    = makeCompatibleFullGroupsConfig();
    auto          allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    KVCacheConfig kv_cache_config;
    auto          cache = createBlockTreeCache(config, kv_cache_config, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->groupSets().size(), 1u);
    EXPECT_EQ(cache->groupSets()[0]->groupIds(), (std::vector<size_t>{0, 1}));
    ASSERT_EQ(cache->groupSets()[0]->devicePools().size(), 2u);
}

TEST_F(BlockTreeCacheFactoryTest, ProductionEvictionConfigurationPropagatesToBlockTreeCache) {
    const auto config    = makeSingleConfig();
    auto       allocator = initAllocator<SingleTypeKVCacheAllocator>(config);

    KVCacheConfig kv_cache_config;
    kv_cache_config.device_eviction_policy = "FIFO";
    kv_cache_config.host_eviction_policy   = "lfu";
    kv_cache_config.disk_eviction_policy   = "LrU";

    auto cache = createBlockTreeCache(config, kv_cache_config, allocator);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->config().device_eviction_policy, EvictionPolicy::FIFO);
    EXPECT_EQ(cache->config().host_eviction_policy, EvictionPolicy::LFU);
    EXPECT_EQ(cache->config().disk_eviction_policy, EvictionPolicy::LRU);
}

TEST_F(BlockTreeCacheFactoryTest, UnsupportedProductionEvictionPolicyFailsClosed) {
    using PolicyField = std::string                        KVCacheConfig::*;
    const std::vector<std::pair<const char*, PolicyField>> policy_fields = {
        {"device", &KVCacheConfig::device_eviction_policy},
        {"host", &KVCacheConfig::host_eviction_policy},
        {"disk", &KVCacheConfig::disk_eviction_policy},
    };

    const auto config = makeSingleConfig();
    for (const auto& [tier, policy_field] : policy_fields) {
        SCOPED_TRACE(tier);
        auto              allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
        KVCacheConfig     kv_cache_config;
        BlockTreeCachePtr cache;
        kv_cache_config.*policy_field = "clock";
        EXPECT_NO_THROW(cache = createBlockTreeCache(config, kv_cache_config, allocator));
        EXPECT_EQ(cache, nullptr);
        EXPECT_EQ(allocator->blockTreeCache(), nullptr);
    }
}

TEST_F(BlockTreeCacheFactoryTest, CompatibleSwaGroupsAggregateOnlyWhenPolicyWindowsMatch) {
    for (const int second_window : {128, 64}) {
        SCOPED_TRACE(second_window);
        const auto config    = makeCompatibleSwaGroupsConfig(second_window);
        auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
        auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

        ASSERT_NE(cache, nullptr);
        const size_t expected_group_count = second_window == 128 ? 1u : 2u;
        ASSERT_EQ(cache->groupSets().size(), expected_group_count);
        if (second_window == 128) {
            EXPECT_EQ(cache->groupSets().front()->groupIds(), (std::vector<size_t>{0, 1}));
        } else {
            EXPECT_EQ(cache->groupSets()[0]->groupIds(), (std::vector<size_t>{0}));
            EXPECT_EQ(cache->groupSets()[1]->groupIds(), (std::vector<size_t>{1}));
        }
    }
}

TEST_F(BlockTreeCacheFactoryTest, CompatibleInsertPacksOneGroupSetResourceInGroupIdOrder) {
    const auto config    = makeCompatibleFullGroupsConfig();
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->attachBlockTreeCache(cache);

    ASSERT_EQ(cache->groupSets().size(), 1u);
    EXPECT_EQ(cache->groupSets()[0]->groupIds(), (std::vector<size_t>{0, 1}));
    const auto blocks = insertOneKeyThroughAllocator(config, allocator, /*key=*/700);

    auto match = cache->match(CacheKeysType{700});
    ASSERT_EQ(match.matched_device_blocks, 1u);
    ASSERT_EQ(cache->matchedBlocksForGroup(0, match.matched_device_resources), (BlockIndicesType{blocks[0]}));
    ASSERT_EQ(cache->matchedBlocksForGroup(1, match.matched_device_resources), (BlockIndicesType{blocks[1]}));
    block_tree_cache_test::releaseRequestRefsForTest(*cache, match.matched_device_resources);

    releaseInsertedRequestBlocks(allocator, blocks);
}

TEST_F(BlockTreeCacheFactoryTest, BlockTreeCacheCanOnlyBeAttachedOnce) {
    const auto config    = makeSingleConfig();
    auto       allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);
    ASSERT_NE(cache, nullptr);

    EXPECT_THROW(allocator->attachBlockTreeCache(nullptr), std::runtime_error);
    allocator->attachBlockTreeCache(cache);
    EXPECT_EQ(allocator->blockTreeCache(), cache);
    EXPECT_THROW(allocator->attachBlockTreeCache(cache), std::runtime_error);
}

TEST_F(BlockTreeCacheFactoryTest, SurvivingGroupCallbackKeepsBlockTreeCacheAliveWithoutAllocator) {
    const auto config    = makeSingleConfig();
    auto       allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->attachBlockTreeCache(cache);

    KVCacheGroupPtr group = allocator->cacheGroups().front();
    ASSERT_NE(group, nullptr);
    const DeviceBlockPoolPtr pool        = group->blockPool();
    const size_t             free_blocks = pool->freeBlocksNum();
    auto                     allocated   = pool->malloc(free_blocks);
    ASSERT_TRUE(allocated.has_value());
    pool->incRef(*allocated, BlockRefType::REQUEST);

    std::weak_ptr<BlockTreeCache> cache_lifetime = cache;
    allocator.reset();
    cache.reset();

    EXPECT_FALSE(cache_lifetime.expired());
    EXPECT_FALSE(group->ensureFreeBlocks(1));

    pool->decRef(*allocated, BlockRefType::REQUEST);
    group.reset();
    EXPECT_TRUE(cache_lifetime.expired());
}

TEST_F(BlockTreeCacheFactoryTest, MiddleDisabledGroupIsExcludedWithoutShiftingReusableGroupIds) {
    const auto config    = makeReusableGroupsAroundDisabledConfig();
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->attachBlockTreeCache(cache);

    ASSERT_EQ(cache->groupSets().size(), 1u);
    EXPECT_EQ(cache->groupSets()[0]->groupIds(), (std::vector<size_t>{0, 2}));

    const auto blocks = insertOneKeyThroughAllocator(config, allocator, /*key=*/701);
    auto       match  = cache->match(CacheKeysType{701});
    ASSERT_EQ(match.matched_device_blocks, 1u);
    EXPECT_TRUE(cache->matchedBlocksForGroup(1, match.matched_device_resources).empty());
    for (const size_t group_id : {0u, 2u}) {
        ASSERT_EQ(cache->matchedBlocksForGroup(group_id, match.matched_device_resources),
                  (BlockIndicesType{blocks[group_id]}));
    }
    block_tree_cache_test::releaseRequestRefsForTest(*cache, match.matched_device_resources);

    releaseInsertedRequestBlocks(allocator, blocks);
}

TEST_F(BlockTreeCacheFactoryTest, SharedPhysicalBackingWatermarkSharesPendingReleasesAcrossGroupSets) {
    const auto config    = makeSharedBackingCascadeConfig();
    auto       allocator = initAllocator<HybridTypeKVCacheAllocator>(config);

    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_host_cache  = true;
    kv_cache_config.host_cache_size_mb = 1;
    auto cache                         = createBlockTreeCache(config, kv_cache_config, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->attachBlockTreeCache(cache);

    const auto allocator_groups = allocator->cacheGroups();
    ASSERT_EQ(allocator_groups.size(), 2u);
    ASSERT_EQ(allocator_groups[0]->blockPool().get(), allocator_groups[1]->blockPool().get());
    auto backing = allocator_groups[0]->blockPool();
    ASSERT_EQ(backing->totalBlocksNum(), 7u);

    ASSERT_EQ(cache->groupSets().size(), 2u);
    ASSERT_EQ(cache->groupSets()[0]->devicePools()[0].get(), backing.get());
    ASSERT_EQ(cache->groupSets()[1]->devicePools()[0].get(), backing.get());

    auto scripted_copy =
        std::make_shared<block_tree_cache_test::ScriptedPerRankBlockTransferEngine>(cache->groupSets());
    block_tree_cache_test::BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, scripted_copy);
    block_tree_cache_test::BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.6);

    std::vector<std::vector<BlockIdxType>> request_blocks;
    for (CacheKeyType key : {800, 801, 802}) {
        request_blocks.push_back(insertOneKeyThroughAllocator(config, allocator, key));
    }
    ASSERT_EQ(backing->freeBlocksNum(), 1u);
    for (const auto& blocks : request_blocks) {
        releaseInsertedRequestBlocks(allocator, blocks);
    }

    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    // One FULL+LINEAR plan contributes two physical releases. Both GroupSets
    // share the pending count for their common backing pool.
    EXPECT_EQ(scripted_copy->submittedDescriptorCount(), 2u);
    EXPECT_EQ(backing->freeBlocksNum(), 3u);
    EXPECT_LT(backing->freeBlocksNum(), backing->totalBlocksNum());
    const auto descriptors = scripted_copy->descriptors();
    ASSERT_EQ(descriptors.size(), 2u);
    std::vector<int> submitted_groups;
    for (const auto& descriptor : descriptors) {
        ASSERT_EQ(descriptor.source_tier, Tier::DEVICE);
        ASSERT_EQ(descriptor.target_tier, Tier::HOST);
        ASSERT_EQ(descriptor.source_blocks.size(), 1u);
        submitted_groups.push_back(descriptor.group_set_id);
    }
    EXPECT_EQ(std::count(submitted_groups.begin(), submitted_groups.end(), 0), 1);
    EXPECT_EQ(std::count(submitted_groups.begin(), submitted_groups.end(), 1), 1);

    block_tree_cache_test::BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, /*num_blocks=*/100, Tier::DEVICE);
    block_tree_cache_test::BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, /*num_blocks=*/100, Tier::HOST);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
}

TEST_F(BlockTreeCacheFactoryTest, FailedWatermarkPlanStopsThisPassAndRecomputesOnNextTrigger) {
    const auto config    = makeSingleConfig();
    auto       allocator = initAllocator<SingleTypeKVCacheAllocator>(config);

    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_host_cache  = true;
    kv_cache_config.host_cache_size_mb = 1;
    auto cache                         = createBlockTreeCache(config, kv_cache_config, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->attachBlockTreeCache(cache);

    auto scripted_copy =
        std::make_shared<block_tree_cache_test::ScriptedPerRankBlockTransferEngine>(cache->groupSets());
    block_tree_cache_test::BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, scripted_copy);
    scripted_copy->enqueue(/*success=*/false);

    auto       backing = allocator->cacheGroups().front()->blockPool();
    const auto blocks  = insertOneKeyThroughAllocator(config, allocator, /*key=*/810);
    ASSERT_EQ(backing->freeBlocksNum(), 6u);
    block_tree_cache_test::BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, Tier::DEVICE, 0.01);
    releaseInsertedRequestBlocks(allocator, blocks);
    block_tree_cache_test::BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    // The failed accepted async plan is not recursively retried in the same
    // maintenance pass; rollback leaves the physical deficit intact.
    EXPECT_EQ(scripted_copy->submittedDescriptorCount(), 1u);
    EXPECT_EQ(backing->freeBlocksNum(), 6u);

    scripted_copy->clear();
    block_tree_cache_test::BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    EXPECT_EQ(scripted_copy->submittedDescriptorCount(), 1u);
    EXPECT_EQ(backing->freeBlocksNum(), 7u);

    block_tree_cache_test::BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, /*num_blocks=*/100, Tier::HOST);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
}

TEST_F(BlockTreeCacheFactoryTest, DeviceMinFreeDoesNotTriggerBlockTreeWatermarkEviction) {
    const auto config    = makeSingleConfig();
    auto       allocator = initAllocator<SingleTypeKVCacheAllocator>(config);

    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_host_cache            = true;
    kv_cache_config.host_cache_size_mb           = 1;
    kv_cache_config.device_cache_min_free_blocks = 7;
    auto cache                                   = createBlockTreeCache(config, kv_cache_config, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->attachBlockTreeCache(cache);

    auto scripted_copy =
        std::make_shared<block_tree_cache_test::ScriptedPerRankBlockTransferEngine>(cache->groupSets());
    block_tree_cache_test::BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*cache, scripted_copy);

    const auto blocks  = insertOneKeyThroughAllocator(config, allocator, /*key=*/811);
    auto       backing = allocator->cacheGroups().front()->blockPool();
    ASSERT_EQ(backing->freeBlocksNum(), 6u);
    releaseInsertedRequestBlocks(allocator, blocks);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);

    EXPECT_EQ(scripted_copy->submittedDescriptorCount(), 0u);
    EXPECT_EQ(backing->freeBlocksNum(), 6u);
}

TEST_F(BlockTreeCacheFactoryTest, IncompatibleGroupsKeepSeparateGroupSetResources) {
    const auto config    = makeIncompatibleFullGroupsConfig();
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->attachBlockTreeCache(cache);

    ASSERT_EQ(cache->groupSets().size(), 2u);
    const auto blocks = insertOneKeyThroughAllocator(config, allocator, /*key=*/702);
    auto       match  = cache->match(CacheKeysType{702});
    ASSERT_EQ(match.matched_device_blocks, 1u);
    EXPECT_EQ(cache->matchedBlocksForGroup(0, match.matched_device_resources), (BlockIndicesType{blocks[0]}));
    EXPECT_EQ(cache->matchedBlocksForGroup(1, match.matched_device_resources), (BlockIndicesType{blocks[1]}));
    // Each block has one request holder, one tree holder, and one match holder.
    EXPECT_EQ(cache->groupSets()[0]->devicePools()[0]->refCount(blocks[0]), 3u);
    EXPECT_EQ(cache->groupSets()[1]->devicePools()[0]->refCount(blocks[1]), 3u);
    block_tree_cache_test::releaseRequestRefsForTest(*cache, match.matched_device_resources);
    // Releasing the match leaves the request and tree holders alive.
    EXPECT_EQ(cache->groupSets()[0]->devicePools()[0]->refCount(blocks[0]), 2u);
    EXPECT_EQ(cache->groupSets()[1]->devicePools()[0]->refCount(blocks[1]), 2u);

    releaseInsertedRequestBlocks(allocator, blocks);
}

TEST_F(BlockTreeCacheFactoryTest, ReinsertRefillsOnlyEmptyIdleGroupSetResource) {
    const auto    config    = makeIncompatibleFullGroupsConfig();
    auto          allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_host_cache  = true;
    kv_cache_config.host_cache_size_mb = 1;
    auto cache                         = createBlockTreeCache(config, kv_cache_config, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->attachBlockTreeCache(cache);

    ASSERT_EQ(cache->groupSets().size(), 2u);
    const auto& group_set_a = cache->groupSets()[0];
    const auto& group_set_b = cache->groupSets()[1];
    ASSERT_EQ(group_set_a->groupIds(), (std::vector<size_t>{0}));
    ASSERT_EQ(group_set_b->groupIds(), (std::vector<size_t>{1}));
    ASSERT_EQ(group_set_a->devicePools().size(), 1u);
    ASSERT_EQ(group_set_b->devicePools().size(), 1u);
    ASSERT_NE(group_set_a->hostPool(), nullptr);

    const auto original_blocks = insertOneKeyThroughAllocator(config, allocator, /*key=*/703);
    ASSERT_EQ(original_blocks.size(), 2u);
    releaseInsertedRequestBlocks(allocator, original_blocks);

    auto find = cache->tree()->findNode(CacheKeysType{703});
    ASSERT_EQ(find.size(), 1u);
    TreeNode* node = find.back();
    ASSERT_EQ(node->group_set_resources.size(), 2u);
    auto clear_group_set_a = [&](BlockIdxType block) {
        const MultiNodeResource device_resource{0, Tier::DEVICE, {{node, {block}}}};
        node->group_set_resources[0].evictFromTier(Tier::DEVICE);
        group_set_a->unreferenceBlocks(device_resource, BlockRefType::BLOCK_CACHE);
    };

    const int    b_layer = config.layerIdsForGroup(1).front();
    const size_t b_bytes = config.kvBlockStrideBytesForGroup(1) + config.kvScaleStrideBytesForGroup(1);
    ASSERT_GT(b_bytes, 0u);
    writeDevicePattern(
        allocator->cacheGroups()[1]->convertIndexToAddr(b_layer, original_blocks[1]).kv_addr, b_bytes, 0x5a);

    // Simulate a partial resource left by an interrupted workflow. Leaf eviction
    // now cascades to every group set and cannot create this state by itself.
    clear_group_set_a(original_blocks[0]);
    ASSERT_TRUE(node->group_set_resources[0].is_empty());
    ASSERT_EQ(node->group_set_resources[1].device_blocks, (BlockIndicesType{original_blocks[1]}));
    const size_t b_ref_before  = group_set_b->devicePools()[0]->refCount(original_blocks[1]);
    const auto   b_meta_before = node->group_set_resources[1].candidate_meta;
    const auto   before_refill = cache->getKeySnapshot(/*limit=*/16);

    const auto refill_a = allocator->cacheGroups()[0]->blockPool()->malloc(1);
    ASSERT_TRUE(refill_a.has_value());
    ASSERT_EQ(refill_a->size(), 1u);
    allocator->cacheGroups()[0]->blockPool()->incRef(*refill_a, BlockRefType::REQUEST);
    GroupSetResource incoming_a;
    incoming_a.device_blocks = {refill_a->front()};
    GroupSetResource incoming_b;
    incoming_b.device_blocks = {original_blocks[1]};
    const std::vector<std::vector<GroupSetResource>> refill_resources{{incoming_a, incoming_b}};
    cache->insert(CacheKeysType{703}, refill_resources, Tier::DEVICE);

    const auto after_refill = cache->getKeySnapshot(/*limit=*/16);
    EXPECT_EQ(after_refill.version, before_refill.version + 1);
    EXPECT_EQ(after_refill.keys, before_refill.keys);
    ASSERT_EQ(node->group_set_resources[0].device_blocks, (BlockIndicesType{refill_a->front()}));
    EXPECT_EQ(group_set_a->devicePools()[0]->refCount(refill_a->front()), 2u);
    EXPECT_EQ(node->group_set_resources[1].device_blocks, (BlockIndicesType{original_blocks[1]}));
    EXPECT_EQ(group_set_b->devicePools()[0]->refCount(original_blocks[1]), b_ref_before);
    EXPECT_EQ(node->group_set_resources[1].candidate_meta.last_access_seq, b_meta_before.last_access_seq);
    EXPECT_EQ(node->group_set_resources[1].candidate_meta.admission_seq, b_meta_before.admission_seq);
    EXPECT_EQ(node->group_set_resources[1].candidate_meta.hit_count, b_meta_before.hit_count);
    expectDevicePattern(
        allocator->cacheGroups()[1]->convertIndexToAddr(b_layer, original_blocks[1]).kv_addr, b_bytes, 0x5a);

    EXPECT_EQ(cache->getStats().device_heap_total_size, 2u);
    block_tree_cache_test::releaseDeviceBlocks(
        *cache, allocator->cacheGroups()[0]->blockPool(), *refill_a, BlockRefType::REQUEST);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 2u);

    const size_t a_ref_before_duplicate = group_set_a->devicePools()[0]->refCount(refill_a->front());
    const auto   before_duplicate       = cache->getKeySnapshot(/*limit=*/16);
    cache->insert(CacheKeysType{703}, refill_resources, Tier::DEVICE);
    const auto after_duplicate = cache->getKeySnapshot(/*limit=*/16);
    EXPECT_EQ(after_duplicate.version, before_duplicate.version);
    EXPECT_EQ(group_set_a->devicePools()[0]->refCount(refill_a->front()), a_ref_before_duplicate);
    EXPECT_EQ(group_set_b->devicePools()[0]->refCount(original_blocks[1]), b_ref_before);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 2u);

    const auto nonempty_replacement = allocator->cacheGroups()[0]->blockPool()->malloc(1);
    ASSERT_TRUE(nonempty_replacement.has_value());
    ASSERT_EQ(nonempty_replacement->size(), 1u);
    allocator->cacheGroups()[0]->blockPool()->incRef(*nonempty_replacement, BlockRefType::REQUEST);
    GroupSetResource nonempty_incoming_a;
    nonempty_incoming_a.device_blocks = {nonempty_replacement->front()};
    const auto before_nonempty        = cache->getKeySnapshot(/*limit=*/16);
    cache->insert(CacheKeysType{703}, {{nonempty_incoming_a, incoming_b}}, Tier::DEVICE);
    EXPECT_EQ(cache->getKeySnapshot(/*limit=*/16).version, before_nonempty.version);
    EXPECT_EQ(node->group_set_resources[0].device_blocks, (BlockIndicesType{refill_a->front()}));
    EXPECT_EQ(group_set_a->devicePools()[0]->refCount(refill_a->front()), a_ref_before_duplicate);
    allocator->cacheGroups()[0]->blockPool()->decRef(*nonempty_replacement, BlockRefType::REQUEST);

    clear_group_set_a(refill_a->front());
    ASSERT_TRUE(node->group_set_resources[0].is_empty());

    const BlockIdxType host_a = group_set_a->allocateSingleBlock(Tier::HOST, BlockRefType::BLOCK_CACHE);
    ASSERT_NE(host_a, NULL_BLOCK_IDX);
    node->group_set_resources[0].host_block = host_a;
    const auto host_replacement             = allocator->cacheGroups()[0]->blockPool()->malloc(1);
    ASSERT_TRUE(host_replacement.has_value());
    ASSERT_EQ(host_replacement->size(), 1u);
    allocator->cacheGroups()[0]->blockPool()->incRef(*host_replacement, BlockRefType::REQUEST);
    GroupSetResource blocked_incoming_a;
    blocked_incoming_a.device_blocks = {host_replacement->front()};
    const auto before_host           = cache->getKeySnapshot(/*limit=*/16);
    cache->insert(CacheKeysType{703}, {{blocked_incoming_a, incoming_b}}, Tier::DEVICE);
    EXPECT_EQ(cache->getKeySnapshot(/*limit=*/16).version, before_host.version);
    EXPECT_EQ(node->group_set_resources[0].host_block, host_a);
    EXPECT_FALSE(node->group_set_resources[0].hasTier(Tier::DEVICE));
    EXPECT_EQ(group_set_a->hostPool()->refCount(host_a), 1u);
    node->group_set_resources[0].host_block = NULL_BLOCK_IDX;
    group_set_a->releaseSingleBlock(Tier::HOST, host_a, BlockRefType::BLOCK_CACHE);

    for (const auto state : {GroupSetTransferState::DEMOTING, GroupSetTransferState::LOADING}) {
        SCOPED_TRACE(state == GroupSetTransferState::DEMOTING ? "demoting" : "loading");
        node->group_set_resources[0].transfer_state = state;
        const auto before_in_flight                 = cache->getKeySnapshot(/*limit=*/16);
        cache->insert(CacheKeysType{703}, {{blocked_incoming_a, incoming_b}}, Tier::DEVICE);
        EXPECT_EQ(cache->getKeySnapshot(/*limit=*/16).version, before_in_flight.version);
        EXPECT_TRUE(node->group_set_resources[0].is_empty());
        EXPECT_EQ(node->group_set_resources[0].transfer_state, state);
    }
    node->group_set_resources[0].transfer_state = GroupSetTransferState::IDLE;
    allocator->cacheGroups()[0]->blockPool()->decRef(*host_replacement, BlockRefType::REQUEST);

    EXPECT_EQ(group_set_b->devicePools()[0]->refCount(original_blocks[1]), b_ref_before);
    expectDevicePattern(
        allocator->cacheGroups()[1]->convertIndexToAddr(b_layer, original_blocks[1]).kv_addr, b_bytes, 0x5a);
}

TEST_F(BlockTreeCacheFactoryTest, InsertRejectsWrongShapeAndFailsFastOnInvalidGroupPayloads) {
    const auto config    = makeCompatibleFullGroupsConfig();
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);
    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->groupSets().size(), 1u);
    ASSERT_EQ(cache->groupSets()[0]->devicePools().size(), 2u);

    const auto groups = allocator->cacheGroups();
    ASSERT_EQ(groups.size(), 2u);
    const auto first  = groups[0]->blockPool()->malloc(1);
    const auto second = groups[1]->blockPool()->malloc(1);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    ASSERT_EQ(first->size(), 1u);
    ASSERT_EQ(second->size(), 1u);
    groups[0]->blockPool()->incRef(*first, BlockRefType::REQUEST);
    groups[1]->blockPool()->incRef(*second, BlockRefType::REQUEST);

    auto expect_rejected_without_mutation = [&](CacheKeyType                               key,
                                                std::vector<std::vector<GroupSetResource>> resources) {
        const auto before = cache->getKeySnapshot(/*limit=*/32);
        cache->insert(CacheKeysType{key}, resources, Tier::DEVICE);
        const auto after = cache->getKeySnapshot(/*limit=*/32);
        EXPECT_EQ(after.version, before.version);
        EXPECT_EQ(after.keys, before.keys);
        EXPECT_EQ(cache->getStats().tree_node_count, 0u);
    };

    expect_rejected_without_mutation(/*key=*/710, {});
    GroupSetResource valid;
    valid.device_blocks = {first->front(), second->front()};
    expect_rejected_without_mutation(/*key=*/711, {{valid, valid}});

    auto expect_failfast_without_cache_hold = [&](CacheKeyType                               key,
                                                  std::vector<std::vector<GroupSetResource>> resources) {
        const auto before          = cache->getKeySnapshot(/*limit=*/32);
        const auto first_refcount  = groups[0]->blockPool()->refCount(first->front());
        const auto second_refcount = groups[1]->blockPool()->refCount(second->front());
        EXPECT_ANY_THROW(cache->insert(CacheKeysType{key}, resources, Tier::DEVICE));
        const auto after = cache->getKeySnapshot(/*limit=*/32);
        EXPECT_EQ(after.version, before.version);
        EXPECT_EQ(after.keys, before.keys);
        EXPECT_EQ(cache->getStats().tree_node_count, 0u);
        EXPECT_EQ(groups[0]->blockPool()->refCount(first->front()), first_refcount);
        EXPECT_EQ(groups[1]->blockPool()->refCount(second->front()), second_refcount);
    };

    GroupSetResource partially_null;
    partially_null.device_blocks = {first->front(), NULL_BLOCK_IDX};
    expect_failfast_without_cache_hold(/*key=*/713, {{partially_null}});

    groups[0]->blockPool()->decRef(*first, BlockRefType::REQUEST);
    groups[1]->blockPool()->decRef(*second, BlockRefType::REQUEST);
}

TEST_F(BlockTreeCacheFactoryTest, SharedPoolGroupSetPayloadUsesTopologyLogicalStrides) {
    auto config = makeHybridConfig(/*independent_pools=*/false);
    ASSERT_EQ(config.layer_to_block_stride_bytes.size(), 4u);
    auto allocator = initAllocator<HybridTypeKVCacheAllocator>(config);
    auto cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->groupSets().size(), 2u);
    for (const auto& group_set : cache->groupSets()) {
        size_t expected_payload = 0;
        for (const size_t group_id : group_set->groupIds()) {
            const auto& group = config.topology().groupById(group_id);
            expected_payload += group.layer_ids.size() * (group.kv_block_stride_bytes + group.kv_scale_stride_bytes);
        }
        EXPECT_EQ(group_set->payloadBytes(), expected_payload);
    }

    const auto& linear = config.topology().groupById(0);
    ASSERT_FALSE(linear.layer_ids.empty());
    EXPECT_NE(static_cast<size_t>(config.layer_to_block_stride_bytes[static_cast<size_t>(linear.layer_ids[0])]),
              linear.kv_block_stride_bytes + linear.kv_scale_stride_bytes);
}

TEST_F(BlockTreeCacheFactoryTest, SharedPoolGroupSetDoesNotDependOnPhysicalStrideTable) {
    for (int malformed_case = 0; malformed_case < 3; ++malformed_case) {
        SCOPED_TRACE(malformed_case);
        auto config = makeHybridConfig(/*independent_pools=*/false);
        if (malformed_case == 0) {
            config.layer_to_block_stride_bytes.resize(3);
        } else if (malformed_case == 1) {
            config.layer_to_block_stride_bytes[3] = 0;
        } else {
            config.layer_to_block_stride_bytes[3] = -1;
        }
        auto allocator = initAllocator<HybridTypeKVCacheAllocator>(config);
        auto cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);
        ASSERT_NE(cache, nullptr);
        ASSERT_EQ(cache->groupSets().size(), 2u);
        for (const auto& group_set : cache->groupSets()) {
            EXPECT_GT(group_set->payloadBytes(), 0u);
        }
    }
}

TEST_F(BlockTreeCacheFactoryTest, LegacySingleGroupAllowsMissingPhysicalStrideTableFallback) {
    auto config = makeSingleConfig();
    config.layer_to_block_stride_bytes.clear();
    auto allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
    auto cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->groupSets().size(), 1u);
    EXPECT_EQ(cache->groupSets()[0]->groupIds(), (std::vector<size_t>{0}));
    const size_t fallback_stride =
        config.topology().groupById(0).kv_block_stride_bytes + config.topology().groupById(0).kv_scale_stride_bytes;
    EXPECT_EQ(cache->groupSets()[0]->payloadBytes(), config.topology().groupById(0).layer_ids.size() * fallback_stride);
}

TEST_F(BlockTreeCacheFactoryTest, CreatesDiskCacheWithoutHostCache) {
    const auto                               config    = makeSingleConfig();
    auto                                     allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
    block_transfer_engine_test::TempDirGuard disk_dir("block_tree_cache_factory_l3_only");
    KVCacheConfig                            kv_cache_config;
    kv_cache_config.enable_host_cache      = false;
    kv_cache_config.enable_disk_cache      = true;
    kv_cache_config.disk_cache_size_mb     = 1;
    kv_cache_config.disk_cache_paths       = disk_dir.path;
    kv_cache_config.disk_cache_buffered_io = true;

    auto cache = createBlockTreeCache(config, kv_cache_config, allocator, ParallelismConfig{});
    ASSERT_NE(cache, nullptr);
    EXPECT_FALSE(cache->isHostCacheEnabled());
    EXPECT_TRUE(cache->isDiskCacheEnabled());
    EXPECT_EQ(cache->config().device_disk_staging_block_count, 4u);
    EXPECT_EQ(cache->config().max_descriptors_per_transfer_batch, 64u);
    ASSERT_FALSE(cache->groupSets().empty());
    for (const auto& group_set : cache->groupSets()) {
        ASSERT_NE(group_set, nullptr);
        EXPECT_EQ(group_set->hostPool(), nullptr);
        EXPECT_NE(group_set->diskPool(), nullptr);
    }
}

TEST_F(BlockTreeCacheFactoryTest, RejectsDiskCacheForReusableLinearGroupSet) {
    const auto config    = makeHybridConfig(/*independent_pools=*/false);
    auto       allocator = initAllocator<HybridTypeKVCacheAllocator>(config);
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_disk_cache  = true;
    kv_cache_config.disk_cache_size_mb = 1;
    kv_cache_config.disk_cache_paths   = "/unused/rejected-before-disk-init";

    expectFactoryRejects(config, allocator, kv_cache_config);
}

TEST_F(BlockTreeCacheFactoryTest, DiskCacheAllowsDisabledLinearReuse) {
    const auto config = makeHybridConfig(/*independent_pools=*/false, /*disable_linear_reuse=*/true);
    auto allocator    = initAllocator<HybridTypeKVCacheAllocator>(config);
    block_transfer_engine_test::TempDirGuard disk_dir("block_tree_cache_factory_disabled_linear_l3");
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_disk_cache      = true;
    kv_cache_config.disk_cache_size_mb     = 1;
    kv_cache_config.disk_cache_paths       = disk_dir.path;
    kv_cache_config.disk_cache_buffered_io = true;

    auto cache = createBlockTreeCache(config, kv_cache_config, allocator, ParallelismConfig{});
    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->groupSets().size(), 1u);
    EXPECT_EQ(cache->groupSets().front()->groupType(), CacheGroupType::FULL);
}

TEST_F(BlockTreeCacheFactoryTest, DiskStagingBlockCountPropagatesAndValidates) {
    const auto config = makeSingleConfig();

    const auto makeDiskKvCacheConfig = [](const std::string& disk_path) {
        KVCacheConfig kv_cache_config;
        kv_cache_config.enable_host_cache  = false;
        kv_cache_config.enable_disk_cache  = true;
        kv_cache_config.disk_cache_size_mb = 1;
        kv_cache_config.disk_cache_paths   = disk_path;
        return kv_cache_config;
    };

    {
        auto                                     allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
        block_transfer_engine_test::TempDirGuard disk_dir("block_tree_cache_factory_staging_overrides");
        auto                                     kv_cache_config = makeDiskKvCacheConfig(disk_dir.path);
        kv_cache_config.disk_cache_staging_block_count           = 2;

        auto cache = createBlockTreeCache(config, kv_cache_config, allocator, ParallelismConfig{});
        ASSERT_NE(cache, nullptr);
        EXPECT_EQ(cache->config().device_disk_staging_block_count, 2u);
    }

    for (const int64_t bad_block_count : {int64_t{0}, int64_t{-1}, int64_t{3}}) {
        auto                                     allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
        block_transfer_engine_test::TempDirGuard disk_dir("block_tree_cache_factory_staging_bad_blocks");
        auto                                     kv_cache_config = makeDiskKvCacheConfig(disk_dir.path);
        kv_cache_config.disk_cache_staging_block_count           = bad_block_count;
        expectFactoryRejects(config, allocator, kv_cache_config);
    }

    // Disk disabled: staging block count is not validated and creation succeeds.
    {
        auto          allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
        KVCacheConfig kv_cache_config;
        kv_cache_config.disk_cache_staging_block_count = 0;
        auto cache                                     = createBlockTreeCache(config, kv_cache_config, allocator);
        ASSERT_NE(cache, nullptr);
    }
}

TEST_F(BlockTreeCacheFactoryTest, TransferBatchLimitPropagatesAndValidates) {
    const auto config = makeSingleConfig();
    {
        auto          allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
        KVCacheConfig kv_cache_config;
        kv_cache_config.memory_cache_max_descriptors_per_transfer_batch = 17;
        auto cache = createBlockTreeCache(config, kv_cache_config, allocator);
        ASSERT_NE(cache, nullptr);
        EXPECT_EQ(cache->config().max_descriptors_per_transfer_batch, 17u);
    }
    for (const int64_t invalid_limit : {int64_t{0}, int64_t{-1}}) {
        auto          allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
        KVCacheConfig kv_cache_config;
        kv_cache_config.memory_cache_max_descriptors_per_transfer_batch = invalid_limit;
        expectFactoryRejects(config, allocator, kv_cache_config);
    }
}

TEST_F(BlockTreeCacheFactoryTest, DerivesEachLocalTierFromItsOwnSwitch) {
    const auto config = makeSingleConfig();

    for (const bool device_on : {false, true}) {
        for (const bool host_on : {false, true}) {
            for (const bool disk_on : {false, true}) {
                SCOPED_TRACE("L1=" + std::to_string(device_on) + " L2=" + std::to_string(host_on)
                             + " L3=" + std::to_string(disk_on));
                auto                                     allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
                block_transfer_engine_test::TempDirGuard disk_dir("block_tree_cache_factory_tier_matrix");
                KVCacheConfig                            kv_cache_config;
                kv_cache_config.enable_device_cache = device_on;
                kv_cache_config.enable_host_cache   = host_on;
                kv_cache_config.enable_disk_cache   = disk_on;
                if (host_on) {
                    kv_cache_config.host_cache_size_mb = 1;
                }
                if (disk_on) {
                    kv_cache_config.disk_cache_size_mb     = 1;
                    kv_cache_config.disk_cache_paths       = disk_dir.path;
                    kv_cache_config.disk_cache_buffered_io = true;
                }

                auto cache = createBlockTreeCache(config, kv_cache_config, allocator, ParallelismConfig{});
                ASSERT_NE(cache, nullptr);
                EXPECT_EQ(cache->isDeviceCacheEnabled(), device_on);
                EXPECT_EQ(cache->isHostCacheEnabled(), host_on);
                EXPECT_EQ(cache->isDiskCacheEnabled(), disk_on);
                ASSERT_FALSE(cache->groupSets().empty());
                for (const auto& group_set : cache->groupSets()) {
                    ASSERT_NE(group_set, nullptr);
                    EXPECT_EQ(group_set->hostPool() != nullptr, host_on);
                    EXPECT_EQ(group_set->diskPool() != nullptr, disk_on);
                }
            }
        }
    }
}

TEST_F(BlockTreeCacheFactoryTest, RejectsTierEnabledWithoutItsOwnCapacity) {
    const auto config = makeSingleConfig();

    {
        auto          allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
        KVCacheConfig kv_cache_config;
        kv_cache_config.enable_host_cache  = true;
        kv_cache_config.host_cache_size_mb = 0;
        expectFactoryRejects(config, allocator, kv_cache_config);
    }

    {
        auto                                     allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
        block_transfer_engine_test::TempDirGuard disk_dir("block_tree_cache_factory_l3_no_capacity");
        KVCacheConfig                            kv_cache_config;
        kv_cache_config.enable_disk_cache  = true;
        kv_cache_config.disk_cache_size_mb = 0;
        kv_cache_config.disk_cache_paths   = disk_dir.path;
        expectFactoryRejects(config, allocator, kv_cache_config);
    }

    {
        auto          allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
        KVCacheConfig kv_cache_config;
        kv_cache_config.enable_disk_cache  = true;
        kv_cache_config.disk_cache_size_mb = 1;
        kv_cache_config.disk_cache_paths   = "";
        expectFactoryRejects(config, allocator, kv_cache_config);
    }
}

TEST_F(BlockTreeCacheFactoryTest, Factory_CreatesExecutableFullSWAConfig) {
    if (!block_tree_cache_test::cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    CacheConfig cache_config;
    cache_config.dtype                       = TYPE_FP16;
    cache_config.layer_num                   = 3;
    cache_config.layer_all_num               = 3;
    cache_config.block_num                   = 8;
    cache_config.seq_size_per_block          = 1;
    cache_config.kernel_seq_size_per_block   = 1;
    cache_config.use_independent_block_pools = true;

    std::vector<KVCacheSpecPtr> specs;
    for (size_t group_id = 0; group_id < 3; ++group_id) {
        specs.push_back(test::makeResolvedMhaSpec(
            DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/1));
    }
    test::configureIndexedTestGroups(
        cache_config, specs, {{0}, {1}, {2}}, {CacheGroupType::FULL, CacheGroupType::FULL, CacheGroupType::SWA});
    auto policies                   = cache_config.groupPoliciesSnapshot();
    policies[2].enable_prefix_reuse = true;
    policies[2].sliding_window_size = 2;
    cache_config.setGroupPolicies(policies);

    const size_t stride = specs.front()->block_size_bytes();
    cache_config.setGroupBlockLayout({8, 8, 8}, {stride, stride, stride}, {0, 0, 0});
    cache_config.kv_block_stride_bytes       = stride;
    cache_config.kv_block_size_bytes         = stride;
    cache_config.block_size_bytes            = stride;
    cache_config.layer_to_block_stride_bytes = {
        static_cast<int>(stride), static_cast<int>(stride), static_cast<int>(stride)};

    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(cache_config);
    ASSERT_TRUE(allocator->init());
    ASSERT_EQ(allocator->groupBlockPools().size(), 3u);

    block_transfer_engine_test::TempDirGuard disk_dir("block_tree_cache_factory_full_swa");
    KVCacheConfig                            kv_cache_config;
    kv_cache_config.enable_device_cache    = true;
    kv_cache_config.enable_host_cache      = true;
    kv_cache_config.host_cache_size_mb     = 1;
    kv_cache_config.enable_disk_cache      = true;
    kv_cache_config.disk_cache_size_mb     = 1;
    kv_cache_config.disk_cache_paths       = disk_dir.path;
    kv_cache_config.disk_cache_buffered_io = true;

    BlockTreeCachePtr factory_cache =
        createBlockTreeCache(cache_config, kv_cache_config, allocator, ParallelismConfig{});
    ASSERT_NE(factory_cache, nullptr);
    ASSERT_TRUE(factory_cache->isInitialized());
    ASSERT_EQ(factory_cache->groupSets().size(), 2u);
    EXPECT_EQ(factory_cache->groupSets()[0]->groupIds(), (std::vector<size_t>{0, 1}));
    EXPECT_EQ(factory_cache->groupSets()[1]->groupIds(), (std::vector<size_t>{2}));

    auto swa_group = std::dynamic_pointer_cast<SWAGroupSet>(factory_cache->groupSets()[1]);
    ASSERT_NE(swa_group, nullptr);
    EXPECT_EQ(swa_group->slidingWindowSize(), 2u);
    EXPECT_EQ(swa_group->seqSizePerBlock(), 1u);

    for (const GroupSetPtr& group : factory_cache->groupSets()) {
        ASSERT_NE(group, nullptr);
        ASSERT_NE(group->hostPool(), nullptr);
        ASSERT_NE(group->diskPool(), nullptr);
        ASSERT_EQ(group->groupIds().size(), group->devicePools().size());
        EXPECT_EQ(group->groupIds().size(), group->devicePools().size());
        EXPECT_EQ(group->hostPool()->payloadBytes(), group->payloadBytes());
        EXPECT_EQ(group->diskPool()->payloadBytes(), group->payloadBytes());

        block_tree_cache_test::MultiNodeBlocks device_blocks =
            block_tree_cache_test::allocateDeviceBlocksForTest(*group, 1, BlockRefType::REQUEST);
        ASSERT_EQ(device_blocks.size(), 1u);
        ASSERT_EQ(device_blocks[0].size(), group->devicePools().size());
        const BlockIdxType host_block = group->allocateSingleBlock(Tier::HOST, BlockRefType::REQUEST);
        const BlockIdxType disk_block = group->allocateSingleBlock(Tier::DISK, BlockRefType::REQUEST);
        ASSERT_NE(host_block, NULL_BLOCK_IDX);
        ASSERT_NE(disk_block, NULL_BLOCK_IDX);

        EXPECT_TRUE(factory_cache->executeTransfer(
            {TransferDescriptor::deviceToHost(group->groupSetId(), device_blocks[0], host_block)}));
        EXPECT_TRUE(factory_cache->executeTransfer(
            {TransferDescriptor::hostToDisk(group->groupSetId(), host_block, disk_block)}));

        block_tree_cache_test::unreferenceDeviceBlocksForTest(*group, device_blocks, BlockRefType::REQUEST);
        group->releaseSingleBlock(Tier::HOST, host_block, BlockRefType::REQUEST);
        group->releaseSingleBlock(Tier::DISK, disk_block, BlockRefType::REQUEST);
    }
}

}  // namespace rtp_llm
