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
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
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
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4, "swa");
    auto policy                  = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.evict_policy          = CacheEvictPolicy::INDEPENDENT;
    policy.enable_prefix_reuse   = true;
    policy.sliding_window_size   = 128;
    const size_t stride          = spec->block_size_bytes();
    config.kv_block_stride_bytes = stride;
    config.kv_block_size_bytes   = stride;
    config.block_size_bytes      = stride;
    config.layer_to_block_stride_bytes.assign(2, static_cast<int>(stride));
    config.fromGroupedSpecs({spec}, {{0, 1}}, {CacheGroupType::SWA}, {"swa"}, {policy});
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
                                                    DataType::TYPE_FP16,
                                                    "linear");
    auto full_spec   = test::makeResolvedMhaSpec(
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4, "full");

    auto linear_policy                = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    linear_policy.enable_prefix_reuse = !disable_linear_reuse;
    auto full_policy                  = defaultCacheGroupPolicy(CacheGroupType::FULL);
    config.fromGroupedSpecs({linear_spec, full_spec},
                            {{1, 3}, {0, 2}},
                            {CacheGroupType::LINEAR, CacheGroupType::FULL},
                            {"linear", "full"},
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
    auto config = makeHybridConfig(/*independent_pools=*/false);

    const auto                  linear_spec     = config.specForGroup(0);
    const auto                  full_spec       = config.specForGroup(1);
    const auto                  linear_policy   = config.policyForGroup(0);
    const auto                  full_policy     = config.policyForGroup(1);
    const size_t                linear_stride   = config.kvBlockStrideBytesForGroup(0);
    const size_t                full_stride     = config.kvBlockStrideBytesForGroup(1);
    std::vector<KVCacheSpecPtr> reordered_specs = {
        full_spec->clone(),
        linear_spec->clone(),
    };

    // Put FULL first so a watermark primary plan cascades to LINEAR. Both
    // logical groups still use HybridType's one physical BlockPool.
    config.fromGroupedSpecs(reordered_specs,
                            {{0, 2}, {1, 3}},
                            {CacheGroupType::FULL, CacheGroupType::LINEAR},
                            {"full", "linear"},
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
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/4, /*seq_size_per_block=*/4, "full_a");
    auto second = test::makeResolvedMhaSpec(
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/2, "full_b");
    auto first_policy          = defaultCacheGroupPolicy(CacheGroupType::FULL);
    auto second_policy         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    second_policy.evict_policy = CacheEvictPolicy::INDEPENDENT;
    config.fromGroupedSpecs({first, second},
                            {{0}, {1}},
                            {CacheGroupType::FULL, CacheGroupType::FULL},
                            {"full_a", "full_b"},
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

CacheConfig makeCompatibleFullGroupsConfig(CacheEvictPolicy evict_policy) {
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
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4, "full_a");
    auto second = test::makeResolvedMhaSpec(
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4, "full_b");
    auto policy         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.evict_policy = evict_policy;
    config.fromGroupedSpecs({first, second},
                            {{0}, {1}},
                            {CacheGroupType::FULL, CacheGroupType::FULL},
                            {"full_a", "full_b"},
                            {policy, policy});

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
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4, "swa_a");
    auto second = test::makeResolvedMhaSpec(
        DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4, "swa_b");
    auto first_policy                 = defaultCacheGroupPolicy(CacheGroupType::SWA);
    first_policy.evict_policy         = CacheEvictPolicy::CHAIN;
    first_policy.enable_prefix_reuse  = true;
    first_policy.sliding_window_size  = 128;
    auto second_policy                = defaultCacheGroupPolicy(CacheGroupType::SWA);
    second_policy.evict_policy        = CacheEvictPolicy::CHAIN;
    second_policy.enable_prefix_reuse = true;
    second_policy.sliding_window_size = second_window;
    config.fromGroupedSpecs({first, second},
                            {{0}, {1}},
                            {CacheGroupType::SWA, CacheGroupType::SWA},
                            {"swa_a", "swa_b"},
                            {first_policy, second_policy});

    const size_t stride          = first->block_size_bytes();
    config.kv_block_stride_bytes = stride;
    config.kv_block_size_bytes   = stride;
    config.block_size_bytes      = stride;
    config.layer_to_block_stride_bytes.assign(2, static_cast<int>(stride));
    config.setGroupBlockLayout({8, 8}, {stride, stride}, {0, 0});
    return config;
}

CacheConfig makeReusableGroupsAroundDisabledConfig(bool reverse_reusable_tags) {
    CacheConfig config;
    config.dtype                       = DataType::TYPE_FP16;
    config.layer_num                   = 3;
    config.layer_all_num               = 3;
    config.block_num                   = 8;
    config.seq_size_per_block          = 4;
    config.kernel_seq_size_per_block   = 4;
    config.group_layer_num             = 1;
    config.use_independent_block_pools = true;

    const std::vector<std::string>            tags = reverse_reusable_tags ?
                                                         std::vector<std::string>{"full_b", "disabled", "full_a"} :
                                                         std::vector<std::string>{"full_a", "disabled", "full_b"};
    std::vector<std::shared_ptr<KVCacheSpec>> specs;
    std::vector<CacheGroupPolicy>             policies;
    for (const auto& tag : tags) {
        specs.push_back(test::makeResolvedMhaSpec(
            DataType::TYPE_FP16, /*local_head_num_kv=*/1, /*size_per_head=*/8, /*seq_size_per_block=*/4, tag));
        auto policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
        policy.evict_policy        = CacheEvictPolicy::CHAIN;
        policy.enable_prefix_reuse = tag != "disabled";
        policies.push_back(policy);
    }
    config.fromGroupedSpecs(std::move(specs),
                            {{0}, {1}, {2}},
                            {CacheGroupType::FULL, CacheGroupType::FULL, CacheGroupType::FULL},
                            tags,
                            std::move(policies));

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
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
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
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    EXPECT_TRUE(allocator->init());
    return allocator;
}

KVCacheGroupPtr cloneGroupWithTag(const KVCacheGroupPtr& source, std::string tag) {
    GroupBase group = source->config();
    auto      spec  = group.spec->clone();
    spec->tag       = tag;
    group.tag       = std::move(tag);
    group.spec      = std::move(spec);
    auto result     = std::make_shared<FullKVCacheGroup>(std::move(group), source->blockPool(), /*group_id=*/99);
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

void expectTargetGroupsBoundByTag(const BlockTreeCachePtr& cache, const KVCacheAllocatorPtr& allocator) {
    ASSERT_NE(cache, nullptr);
    const auto groups = allocator->cacheGroups();
    for (const auto& target_group : groups) {
        ASSERT_NE(target_group, nullptr);
        EXPECT_EQ(cache->deviceKVGroup(target_group->tag()).get(), target_group.get());
        if (!target_group->prefixReuseEnabled()) {
            continue;
        }

        const auto component_it = std::find_if(
            cache->componentGroups().begin(), cache->componentGroups().end(), [&](const ComponentGroupPtr& component) {
                return component != nullptr && component->tags() == std::vector<std::string>{target_group->tag()};
            });
        ASSERT_NE(component_it, cache->componentGroups().end()) << target_group->tag();
        ASSERT_EQ((*component_it)->devicePools().size(), 1u);
        const auto& adapter = (*component_it)->devicePools().front();
        ASSERT_NE(adapter, nullptr);
        EXPECT_EQ(adapter->backingPool().get(), target_group->blockPool().get());
        EXPECT_EQ(adapter->addressView().get(), target_group.get());
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
    for (size_t gid = 0; gid < groups.size(); ++gid) {
        const auto allocated = groups[gid]->blockPool()->malloc(1);
        EXPECT_EQ(allocated.size(), 1u);
        if (allocated.empty()) {
            continue;
        }
        blocks[gid] = allocated.front();
        resource->setBatchBlocks(0, static_cast<int>(gid), BlockIndicesType{allocated.front()});
    }
    allocator->insertIntoCache(InsertInfo{resource, nullptr, /*is_resident=*/false});
    return blocks;
}

void releaseInsertedRequestBlocks(const KVCacheAllocatorPtr& allocator, const std::vector<BlockIdxType>& blocks) {
    const auto groups = allocator->cacheGroups();
    ASSERT_EQ(groups.size(), blocks.size());
    for (size_t gid = 0; gid < groups.size(); ++gid) {
        if (!isNullBlockIdx(blocks[gid])) {
            groups[gid]->blockPool()->requestFree(blocks[gid]);
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

}  // namespace

TEST(BlockTreeCacheFactoryUtilityTest, UsableBlockCountReservesBlockZeroWithinBudget) {
    EXPECT_EQ(computeHostUsableBlockCount(4 * 4096, 4096), 3u);
    EXPECT_EQ(computeHostUsableBlockCount(4096, 4096), 0u);
    EXPECT_EQ(computeHostUsableBlockCount(100, 4096), 0u);
    EXPECT_EQ(computeHostUsableBlockCount(4096, 0), 0u);
}

TEST(BlockTreeCacheFactoryUtilityTest, ShouldPinHostBlockPoolHonorsEnv) {
    ::unsetenv("RTP_LLM_PIN_HOST_BLOCK_POOL");
    EXPECT_TRUE(shouldPinHostBlockPool());
    ::setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "0", 1);
    EXPECT_FALSE(shouldPinHostBlockPool());
    ::setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "off", 1);
    EXPECT_FALSE(shouldPinHostBlockPool());
    ::setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "FALSE", 1);
    EXPECT_FALSE(shouldPinHostBlockPool());
    ::setenv("RTP_LLM_PIN_HOST_BLOCK_POOL", "1", 1);
    EXPECT_TRUE(shouldPinHostBlockPool());
    ::unsetenv("RTP_LLM_PIN_HOST_BLOCK_POOL");
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
    expectTargetGroupsBoundByTag(cache, allocator);
}

TEST_F(BlockTreeCacheFactoryTest, SwaComponentGroupUsesDeclaredPolicyWindow) {
    const auto config = makeSwaConfig();
    ASSERT_EQ(config.policyForGroup(0).sliding_window_size, 128);
    auto allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->componentGroups().size(), 1u);
    auto swa_group = std::dynamic_pointer_cast<SWAComponentGroup>(cache->componentGroups().front());
    ASSERT_NE(swa_group, nullptr);
    EXPECT_EQ(swa_group->tags(), (std::vector<std::string>{"swa"}));
    EXPECT_EQ(swa_group->slidingWindowSize(), 128u);
}

TEST_F(BlockTreeCacheFactoryTest, HybridTypeBindsExistingTargetGroupsByTag) {
    const auto config    = makeHybridConfig(/*independent_pools=*/false);
    auto       allocator = initAllocator<HybridTypeKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_EQ(allocator->cacheGroups().size(), 2u);
    expectTargetGroupsBoundByTag(cache, allocator);
}

TEST_F(BlockTreeCacheFactoryTest, HybridPoolBindsIndependentPoolsAndNonContiguousLayerViews) {
    const auto config    = makeHybridConfig(/*independent_pools=*/true);
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_EQ(allocator->cacheGroups().size(), 2u);
    ASSERT_NE(allocator->cacheGroups()[0]->blockPool(), allocator->cacheGroups()[1]->blockPool());
    expectTargetGroupsBoundByTag(cache, allocator);

    for (const auto& group : allocator->cacheGroups()) {
        const auto block = group->blockPool()->malloc(1);
        ASSERT_EQ(block.size(), 1u);
        const int  global_layer = group->config().layer_ids.back();
        const int  local_layer  = static_cast<int>(group->config().layer_ids.size() - 1);
        const auto via_cache    = cache->componentGroups()[config.topology().groupIdForTag(group->tag())]
                                   ->devicePools()[0]
                                   ->convertIndexToAddr(global_layer, block[0]);
        const auto via_backing = group->blockPool()->convertIndexToAddr(local_layer, block[0]);
        EXPECT_EQ(via_cache.kv_addr, via_backing.kv_addr);
        group->blockPool()->requestFree(block);
    }
}

TEST_F(BlockTreeCacheFactoryTest, CopyEnginePreservesNonContiguousGlobalLayerProjectionRoundTrip) {
    const auto    config    = makeHybridConfig(/*independent_pools=*/true);
    auto          allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_memory_cache        = true;
    kv_cache_config.enable_tiered_memory_cache = true;
    kv_cache_config.memory_cache_size_mb       = 1;
    auto cache                                 = createBlockTreeCache(config, kv_cache_config, allocator);
    ASSERT_NE(cache, nullptr);

    const auto target_groups = allocator->cacheGroups();
    const auto full_group_it =
        std::find_if(target_groups.begin(), target_groups.end(), [](const KVCacheGroupPtr& group) {
            return group != nullptr && group->tag() == "full";
        });
    ASSERT_NE(full_group_it, target_groups.end());
    const auto& full_group = *full_group_it;
    ASSERT_EQ(full_group->config().layer_ids, (std::vector<int>{0, 2}));

    const auto component_it = std::find_if(
        cache->componentGroups().begin(), cache->componentGroups().end(), [](const ComponentGroupPtr& component) {
            return component != nullptr && component->tags() == std::vector<std::string>{"full"};
        });
    ASSERT_NE(component_it, cache->componentGroups().end());
    const auto& component = *component_it;
    ASSERT_NE(component->hostPool(), nullptr);
    ASSERT_EQ(component->devicePools().size(), 1u);

    const auto device_blocks = full_group->blockPool()->malloc(1);
    ASSERT_EQ(device_blocks.size(), 1u);
    const BlockIdxType device_block = device_blocks.front();
    const BlockIdxType host_block   = component->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);

    const size_t layer_bytes = full_group->config().kv_block_stride_bytes + full_group->config().kv_scale_stride_bytes;
    ASSERT_GT(layer_bytes, 0u);
    writeDevicePattern(full_group->convertIndexToAddr(/*global_layer=*/0, device_block).kv_addr, layer_bytes, 0x31);
    writeDevicePattern(full_group->convertIndexToAddr(/*global_layer=*/2, device_block).kv_addr, layer_bytes, 0x72);

    EXPECT_EQ(cache->executeTransfer(
                  TransferDescriptor::deviceToHost(component->component_group_id, {device_block}, host_block)),
              CopyStatus::OK);
    writeDevicePattern(full_group->convertIndexToAddr(/*global_layer=*/0, device_block).kv_addr, layer_bytes, 0x00);
    writeDevicePattern(full_group->convertIndexToAddr(/*global_layer=*/2, device_block).kv_addr, layer_bytes, 0x00);
    EXPECT_EQ(cache->executeTransfer(
                  TransferDescriptor::hostToDevice(component->component_group_id, host_block, {device_block})),
              CopyStatus::OK);

    expectDevicePattern(full_group->convertIndexToAddr(/*global_layer=*/0, device_block).kv_addr, layer_bytes, 0x31);
    expectDevicePattern(full_group->convertIndexToAddr(/*global_layer=*/2, device_block).kv_addr, layer_bytes, 0x72);

    component->releaseSingleBlock(Tier::HOST, host_block);
    full_group->blockPool()->requestFree(device_blocks);
}

TEST_F(BlockTreeCacheFactoryTest, ReorderedAllocatorGroupsStillMapByStableTag) {
    const auto config    = makeHybridConfig(/*independent_pools=*/true);
    auto       allocator = initViewAllocator(config);
    auto       groups    = allocator->HybridPoolKVCacheAllocator::cacheGroups();
    ASSERT_EQ(groups.size(), 2u);
    std::reverse(groups.begin(), groups.end());
    allocator->overrideGroups(groups);

    auto cache = createBlockTreeCache(config, KVCacheConfig{}, allocator);
    expectTargetGroupsBoundByTag(cache, allocator);
    ASSERT_EQ(cache->componentGroups().size(), 2u);
    EXPECT_EQ(cache->componentGroups()[0]->tags(), (std::vector<std::string>{"linear"}));
    EXPECT_EQ(cache->componentGroups()[1]->tags(), (std::vector<std::string>{"full"}));
}

TEST_F(BlockTreeCacheFactoryTest, DuplicateMissingUnknownEmptyAndNonExactViewsFailClosed) {
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
        allocator->overrideGroups({groups[0], cloneGroupWithTag(groups[1], "unknown")});
        expectFactoryRejects(config, allocator);
    }
    {
        auto allocator = initViewAllocator(config);
        auto groups    = allocator->HybridPoolKVCacheAllocator::cacheGroups();
        allocator->overrideGroups({groups[0], cloneGroupWithTag(groups[1], "")});
        expectFactoryRejects(config, allocator);
    }
    {
        auto allocator = initViewAllocator(config);
        auto groups    = allocator->HybridPoolKVCacheAllocator::cacheGroups();
        allocator->overrideGroups({cloneGroupWithTag(groups[0], "unknown_linear"), groups[1]});
        expectFactoryRejects(config, allocator);
    }
}

TEST_F(BlockTreeCacheFactoryTest, PrefixReuseDisabledGroupStaysAllocatorOwnedButIsExcludedFromTree) {
    const auto config    = makeHybridConfig(/*independent_pools=*/true, /*disable_linear_reuse=*/true);
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(allocator->cacheGroups().size(), 2u);
    EXPECT_EQ(cache->deviceKVGroup("linear").get(), allocator->cacheGroups()[0].get());
    EXPECT_EQ(cache->deviceKVGroup("full").get(), allocator->cacheGroups()[1].get());
    ASSERT_EQ(cache->componentGroups().size(), 1u);
    EXPECT_EQ(cache->componentGroups()[0]->tags(), (std::vector<std::string>{"full"}));
}

TEST_F(BlockTreeCacheFactoryTest, SameTypeGroupsWithDifferentPolicyShapeSeqAndStrideAreNeverAggregated) {
    const auto config    = makeIncompatibleFullGroupsConfig();
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->componentGroups().size(), 2u);
    EXPECT_EQ(cache->componentGroups()[0]->tags(), (std::vector<std::string>{"full_a"}));
    EXPECT_EQ(cache->componentGroups()[1]->tags(), (std::vector<std::string>{"full_b"}));
    EXPECT_NE(cache->componentGroups()[0]->devicePools()[0]->backingPool(),
              cache->componentGroups()[1]->devicePools()[0]->backingPool());
}

TEST_F(BlockTreeCacheFactoryTest, OnlyCompatibleChainGroupsAggregate) {
    const auto config    = makeCompatibleFullGroupsConfig(CacheEvictPolicy::CHAIN);
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->componentGroups().size(), 1u);
    EXPECT_EQ(cache->componentGroups()[0]->tags(), (std::vector<std::string>{"full_a", "full_b"}));
    ASSERT_EQ(cache->componentGroups()[0]->devicePools().size(), 2u);
}

TEST_F(BlockTreeCacheFactoryTest, CompatibleSwaGroupsAggregateOnlyWhenPolicyWindowsMatch) {
    for (const int second_window : {128, 64}) {
        SCOPED_TRACE(second_window);
        const auto config    = makeCompatibleSwaGroupsConfig(second_window);
        auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
        auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

        ASSERT_NE(cache, nullptr);
        const size_t expected_group_count = second_window == 128 ? 1u : 2u;
        ASSERT_EQ(cache->componentGroups().size(), expected_group_count);
        if (second_window == 128) {
            EXPECT_EQ(cache->componentGroups().front()->tags(), (std::vector<std::string>{"swa_a", "swa_b"}));
        } else {
            EXPECT_EQ(cache->componentGroups()[0]->tags(), (std::vector<std::string>{"swa_a"}));
            EXPECT_EQ(cache->componentGroups()[1]->tags(), (std::vector<std::string>{"swa_b"}));
        }
    }
}

TEST_F(BlockTreeCacheFactoryTest, CompatibleChainInsertPacksOneComponentSlotInStableTagOrder) {
    const auto config    = makeCompatibleFullGroupsConfig(CacheEvictPolicy::CHAIN);
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->setBlockTreeCache(cache.get());

    ASSERT_EQ(cache->componentGroups().size(), 1u);
    EXPECT_EQ(cache->componentGroups()[0]->tags(), (std::vector<std::string>{"full_a", "full_b"}));
    const auto blocks = insertOneKeyThroughAllocator(config, allocator, /*key=*/700);

    auto match = cache->match(CacheKeysType{700});
    ASSERT_EQ(match.matched_blocks, 1u);
    ASSERT_EQ(match.group_block_indices.at("full_a"), (BlockIndicesType{blocks[0]}));
    ASSERT_EQ(match.group_block_indices.at("full_b"), (BlockIndicesType{blocks[1]}));
    cache->releaseMatchedBlocks(match.matched_block_sets);

    releaseInsertedRequestBlocks(allocator, blocks);
    allocator->setBlockTreeCache(nullptr);
}

TEST_F(BlockTreeCacheFactoryTest, MiddleDisabledTagIsExcludedWithoutShiftingReorderedReusableTags) {
    for (const bool reverse_reusable_tags : {false, true}) {
        SCOPED_TRACE(reverse_reusable_tags ? "reversed" : "declared");
        const auto config    = makeReusableGroupsAroundDisabledConfig(reverse_reusable_tags);
        auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
        auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);
        ASSERT_NE(cache, nullptr);
        allocator->setBlockTreeCache(cache.get());

        const std::vector<std::string> expected_tags = reverse_reusable_tags ?
                                                           std::vector<std::string>{"full_b", "full_a"} :
                                                           std::vector<std::string>{"full_a", "full_b"};
        ASSERT_EQ(cache->componentGroups().size(), 1u);
        EXPECT_EQ(cache->componentGroups()[0]->tags(), expected_tags);

        const auto blocks = insertOneKeyThroughAllocator(config, allocator, /*key=*/701);
        auto       match  = cache->match(CacheKeysType{701});
        ASSERT_EQ(match.matched_blocks, 1u);
        EXPECT_EQ(match.group_block_indices.count("disabled"), 0u);
        for (const auto& tag : expected_tags) {
            const size_t gid = config.topology().groupIdForTag(tag);
            ASSERT_EQ(match.group_block_indices.at(tag), (BlockIndicesType{blocks[gid]}));
        }
        cache->releaseMatchedBlocks(match.matched_block_sets);

        releaseInsertedRequestBlocks(allocator, blocks);
        allocator->setBlockTreeCache(nullptr);
    }
}

TEST_F(BlockTreeCacheFactoryTest, SharedPhysicalBackingWatermarkCountsPrimaryAndCascadeCreditsOnce) {
    const auto config    = makeSharedBackingCascadeConfig();
    auto       allocator = initAllocator<HybridTypeKVCacheAllocator>(config);

    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_memory_cache          = true;
    kv_cache_config.enable_tiered_memory_cache   = true;
    kv_cache_config.memory_cache_size_mb         = 1;
    kv_cache_config.device_cache_min_free_blocks = 4;
    auto cache                                   = createBlockTreeCache(config, kv_cache_config, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->setBlockTreeCache(cache.get());

    const auto allocator_groups = allocator->cacheGroups();
    ASSERT_EQ(allocator_groups.size(), 2u);
    ASSERT_EQ(allocator_groups[0]->blockPool().get(), allocator_groups[1]->blockPool().get());
    auto backing = allocator_groups[0]->blockPool();
    ASSERT_EQ(backing->totalBlocksNum(), 7u);

    ASSERT_EQ(cache->componentGroups().size(), 2u);
    ASSERT_EQ(cache->componentGroups()[0]->devicePools()[0]->backingPool().get(), backing.get());
    ASSERT_EQ(cache->componentGroups()[1]->devicePools()[0]->backingPool().get(), backing.get());

    auto scripted_copy =
        std::make_shared<block_tree_cache_test::ScriptedCopyEngine>(cache->componentGroups(), cache->components());
    block_tree_cache_test::BlockTreeCacheTestPeer::setCopyEngineForTest(*cache, scripted_copy);

    std::vector<std::vector<BlockIdxType>> request_blocks;
    for (CacheKeyType key : {800, 801, 802}) {
        request_blocks.push_back(insertOneKeyThroughAllocator(config, allocator, key));
    }
    ASSERT_EQ(backing->freeBlocksNum(), 1u);
    for (const auto& blocks : request_blocks) {
        releaseInsertedRequestBlocks(allocator, blocks);
    }

    cache->onBlocksReleased();
    cache->waitForPendingTasks();

    // One physical deficit is shared by both logical adapters. A FULL+LINEAR
    // cascade contributes two releases, then one forward-only LINEAR plan
    // supplies the remaining credit without over-evicting.
    EXPECT_EQ(scripted_copy->submitCount(), 3u);
    EXPECT_EQ(backing->freeBlocksNum(), 4u);
    EXPECT_LT(backing->freeBlocksNum(), backing->totalBlocksNum());
    const auto descriptors = scripted_copy->descriptors();
    ASSERT_EQ(descriptors.size(), 3u);
    std::vector<int> submitted_groups;
    for (const auto& descriptor : descriptors) {
        ASSERT_EQ(descriptor.source_tier, Tier::DEVICE);
        ASSERT_EQ(descriptor.target_tier, Tier::HOST);
        ASSERT_EQ(descriptor.device_blocks.size(), 1u);
        submitted_groups.push_back(descriptor.component_group_id);
    }
    EXPECT_EQ(std::count(submitted_groups.begin(), submitted_groups.end(), 0), 1);
    EXPECT_EQ(std::count(submitted_groups.begin(), submitted_groups.end(), 1), 2);

    cache->reclaimBlocks(/*num_blocks=*/100, Tier::DEVICE);
    cache->reclaimBlocks(/*num_blocks=*/100, Tier::HOST);
    cache->waitForPendingTasks();
    allocator->setBlockTreeCache(nullptr);
}

TEST_F(BlockTreeCacheFactoryTest, FailedWatermarkPlanStopsThisPassAndRecomputesOnNextTrigger) {
    const auto config    = makeSingleConfig();
    auto       allocator = initAllocator<SingleTypeKVCacheAllocator>(config);

    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_memory_cache          = true;
    kv_cache_config.enable_tiered_memory_cache   = true;
    kv_cache_config.memory_cache_size_mb         = 1;
    kv_cache_config.device_cache_min_free_blocks = 7;
    auto cache                                   = createBlockTreeCache(config, kv_cache_config, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->setBlockTreeCache(cache.get());

    auto scripted_copy =
        std::make_shared<block_tree_cache_test::ScriptedCopyEngine>(cache->componentGroups(), cache->components());
    block_tree_cache_test::BlockTreeCacheTestPeer::setCopyEngineForTest(*cache, scripted_copy);
    scripted_copy->enqueue(CopyStatus::DEVICE_IO_ERROR);

    const auto blocks  = insertOneKeyThroughAllocator(config, allocator, /*key=*/810);
    auto       backing = allocator->cacheGroups().front()->blockPool();
    ASSERT_EQ(backing->freeBlocksNum(), 6u);
    releaseInsertedRequestBlocks(allocator, blocks);
    cache->onBlocksReleased();
    cache->waitForPendingTasks();

    // The failed accepted async plan is not recursively retried in the same
    // maintenance pass; rollback leaves the physical deficit intact.
    EXPECT_EQ(scripted_copy->submitCount(), 1u);
    EXPECT_EQ(backing->freeBlocksNum(), 6u);

    scripted_copy->clear();
    cache->onBlocksReleased();
    cache->waitForPendingTasks();
    EXPECT_EQ(scripted_copy->submitCount(), 1u);
    EXPECT_EQ(backing->freeBlocksNum(), 7u);

    cache->reclaimBlocks(/*num_blocks=*/100, Tier::HOST);
    cache->waitForPendingTasks();
    allocator->setBlockTreeCache(nullptr);
}

TEST_F(BlockTreeCacheFactoryTest, IndependentGroupsNeverAggregateOrEnterAnotherGroupsCascade) {
    const auto config    = makeCompatibleFullGroupsConfig(CacheEvictPolicy::INDEPENDENT);
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->componentGroups().size(), 2u);
    EXPECT_EQ(cache->componentGroups()[0]->tags(), (std::vector<std::string>{"full_a"}));
    EXPECT_EQ(cache->componentGroups()[1]->tags(), (std::vector<std::string>{"full_b"}));
}

TEST_F(BlockTreeCacheFactoryTest, IndependentInsertKeepsSeparateComponentSlots) {
    const auto config    = makeCompatibleFullGroupsConfig(CacheEvictPolicy::INDEPENDENT);
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->setBlockTreeCache(cache.get());

    ASSERT_EQ(cache->componentGroups().size(), 2u);
    const auto blocks = insertOneKeyThroughAllocator(config, allocator, /*key=*/702);
    auto       match  = cache->match(CacheKeysType{702});
    ASSERT_EQ(match.matched_blocks, 1u);
    EXPECT_EQ(match.group_block_indices.at("full_a"), (BlockIndicesType{blocks[0]}));
    EXPECT_EQ(match.group_block_indices.at("full_b"), (BlockIndicesType{blocks[1]}));
    EXPECT_EQ(cache->componentGroups()[0]->devicePools()[0]->refCount(blocks[0]), 2u);
    EXPECT_EQ(cache->componentGroups()[1]->devicePools()[0]->refCount(blocks[1]), 2u);
    cache->releaseMatchedBlocks(match.matched_block_sets);
    EXPECT_EQ(cache->componentGroups()[0]->devicePools()[0]->refCount(blocks[0]), 1u);
    EXPECT_EQ(cache->componentGroups()[1]->devicePools()[0]->refCount(blocks[1]), 1u);

    releaseInsertedRequestBlocks(allocator, blocks);
    allocator->setBlockTreeCache(nullptr);
}

TEST_F(BlockTreeCacheFactoryTest, IndependentReinsertRefillsOnlyEmptyIdleComponentSlot) {
    const auto    config    = makeCompatibleFullGroupsConfig(CacheEvictPolicy::INDEPENDENT);
    auto          allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_memory_cache        = true;
    kv_cache_config.enable_tiered_memory_cache = true;
    kv_cache_config.memory_cache_size_mb       = 1;
    auto cache                                 = createBlockTreeCache(config, kv_cache_config, allocator);
    ASSERT_NE(cache, nullptr);
    allocator->setBlockTreeCache(cache.get());

    ASSERT_EQ(cache->componentGroups().size(), 2u);
    const auto& component_a = cache->componentGroups()[0];
    const auto& component_b = cache->componentGroups()[1];
    ASSERT_EQ(component_a->tags(), (std::vector<std::string>{"full_a"}));
    ASSERT_EQ(component_b->tags(), (std::vector<std::string>{"full_b"}));
    ASSERT_EQ(component_a->devicePools().size(), 1u);
    ASSERT_EQ(component_b->devicePools().size(), 1u);
    ASSERT_NE(component_a->hostPool(), nullptr);

    const auto original_blocks = insertOneKeyThroughAllocator(config, allocator, /*key=*/703);
    ASSERT_EQ(original_blocks.size(), 2u);
    releaseInsertedRequestBlocks(allocator, original_blocks);
    cache->onBlocksReleased();

    auto find = cache->tree()->findNode(CacheKeysType{703});
    ASSERT_EQ(find.matched_blocks, 1u);
    ASSERT_NE(find.matched_node, nullptr);
    TreeNode* node = find.matched_node;
    ASSERT_EQ(node->group_slots.size(), 2u);

    const int    b_layer = config.layerIdsForGroup(1).front();
    const size_t b_bytes = config.kvBlockStrideBytesForGroup(1) + config.kvScaleStrideBytesForGroup(1);
    ASSERT_GT(b_bytes, 0u);
    writeDevicePattern(
        allocator->cacheGroups()[1]->convertIndexToAddr(b_layer, original_blocks[1]).kv_addr, b_bytes, 0x5a);

    ASSERT_EQ(cache->evictForTag("full_a", 1), 1);
    cache->waitForPendingTasks();
    ASSERT_TRUE(node->group_slots[0].is_empty());
    ASSERT_EQ(node->group_slots[1].device_blocks, (BlockIndicesType{original_blocks[1]}));
    const size_t b_ref_before  = component_b->devicePools()[0]->refCount(original_blocks[1]);
    const auto   b_meta_before = node->group_slots[1].candidate_meta;
    const auto   before_refill = cache->getKeySnapshot(/*limit=*/16);

    const auto refill_a = allocator->cacheGroups()[0]->blockPool()->malloc(1);
    ASSERT_EQ(refill_a.size(), 1u);
    GroupSlot incoming_a;
    incoming_a.device_blocks = {refill_a[0]};
    GroupSlot incoming_b;
    incoming_b.device_blocks = {original_blocks[1]};
    const std::vector<std::vector<GroupSlot>> refill_slots{{incoming_a, incoming_b}};
    cache->insert(nullptr, CacheKeysType{703}, refill_slots);

    const auto after_refill = cache->getKeySnapshot(/*limit=*/16);
    EXPECT_EQ(after_refill.version, before_refill.version + 1);
    EXPECT_EQ(after_refill.keys, before_refill.keys);
    ASSERT_EQ(node->group_slots[0].device_blocks, (BlockIndicesType{refill_a[0]}));
    EXPECT_EQ(component_a->devicePools()[0]->refCount(refill_a[0]), 1u);
    EXPECT_EQ(node->group_slots[1].device_blocks, (BlockIndicesType{original_blocks[1]}));
    EXPECT_EQ(component_b->devicePools()[0]->refCount(original_blocks[1]), b_ref_before);
    EXPECT_EQ(node->group_slots[1].candidate_meta.last_access_seq, b_meta_before.last_access_seq);
    EXPECT_EQ(node->group_slots[1].candidate_meta.admission_seq, b_meta_before.admission_seq);
    EXPECT_EQ(node->group_slots[1].candidate_meta.hit_count, b_meta_before.hit_count);
    expectDevicePattern(
        allocator->cacheGroups()[1]->convertIndexToAddr(b_layer, original_blocks[1]).kv_addr, b_bytes, 0x5a);

    EXPECT_EQ(cache->getStats().device_heap_total_size, 1u);
    allocator->cacheGroups()[0]->blockPool()->requestFree(refill_a);
    cache->onBlocksReleased();
    EXPECT_EQ(cache->getStats().device_heap_total_size, 2u);

    const size_t a_ref_before_duplicate = component_a->devicePools()[0]->refCount(refill_a[0]);
    const auto   before_duplicate       = cache->getKeySnapshot(/*limit=*/16);
    cache->insert(nullptr, CacheKeysType{703}, refill_slots);
    const auto after_duplicate = cache->getKeySnapshot(/*limit=*/16);
    EXPECT_EQ(after_duplicate.version, before_duplicate.version);
    EXPECT_EQ(component_a->devicePools()[0]->refCount(refill_a[0]), a_ref_before_duplicate);
    EXPECT_EQ(component_b->devicePools()[0]->refCount(original_blocks[1]), b_ref_before);
    EXPECT_EQ(cache->getStats().device_heap_total_size, 2u);

    const auto nonempty_replacement = allocator->cacheGroups()[0]->blockPool()->malloc(1);
    ASSERT_EQ(nonempty_replacement.size(), 1u);
    GroupSlot nonempty_incoming_a;
    nonempty_incoming_a.device_blocks = {nonempty_replacement[0]};
    const auto before_nonempty        = cache->getKeySnapshot(/*limit=*/16);
    cache->insert(nullptr, CacheKeysType{703}, {{nonempty_incoming_a, incoming_b}});
    EXPECT_EQ(cache->getKeySnapshot(/*limit=*/16).version, before_nonempty.version);
    EXPECT_EQ(node->group_slots[0].device_blocks, (BlockIndicesType{refill_a[0]}));
    EXPECT_EQ(component_a->devicePools()[0]->refCount(refill_a[0]), a_ref_before_duplicate);
    allocator->cacheGroups()[0]->blockPool()->requestFree(nonempty_replacement);

    ASSERT_EQ(cache->evictForTag("full_a", 1), 1);
    cache->waitForPendingTasks();
    ASSERT_TRUE(node->group_slots[0].is_empty());

    const BlockIdxType host_a = component_a->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(host_a, NULL_BLOCK_IDX);
    node->group_slots[0].host_block = host_a;
    const auto host_replacement     = allocator->cacheGroups()[0]->blockPool()->malloc(1);
    ASSERT_EQ(host_replacement.size(), 1u);
    GroupSlot blocked_incoming_a;
    blocked_incoming_a.device_blocks = {host_replacement[0]};
    const auto before_host           = cache->getKeySnapshot(/*limit=*/16);
    cache->insert(nullptr, CacheKeysType{703}, {{blocked_incoming_a, incoming_b}});
    EXPECT_EQ(cache->getKeySnapshot(/*limit=*/16).version, before_host.version);
    EXPECT_EQ(node->group_slots[0].host_block, host_a);
    EXPECT_FALSE(node->group_slots[0].has_value(Tier::DEVICE));
    EXPECT_EQ(component_a->hostPool()->refCount(host_a), 1u);
    node->group_slots[0].host_block = NULL_BLOCK_IDX;
    component_a->releaseSingleBlock(Tier::HOST, host_a);

    for (const auto state : {SlotTransferState::DEMOTING, SlotTransferState::LOADING_BACK}) {
        SCOPED_TRACE(state == SlotTransferState::DEMOTING ? "demoting" : "loading_back");
        node->group_slots[0].transfer_state = state;
        const auto before_in_flight         = cache->getKeySnapshot(/*limit=*/16);
        cache->insert(nullptr, CacheKeysType{703}, {{blocked_incoming_a, incoming_b}});
        EXPECT_EQ(cache->getKeySnapshot(/*limit=*/16).version, before_in_flight.version);
        EXPECT_TRUE(node->group_slots[0].is_empty());
        EXPECT_EQ(node->group_slots[0].transfer_state, state);
    }
    node->group_slots[0].transfer_state = SlotTransferState::IDLE;
    allocator->cacheGroups()[0]->blockPool()->requestFree(host_replacement);

    EXPECT_EQ(component_b->devicePools()[0]->refCount(original_blocks[1]), b_ref_before);
    expectDevicePattern(
        allocator->cacheGroups()[1]->convertIndexToAddr(b_layer, original_blocks[1]).kv_addr, b_bytes, 0x5a);
    allocator->setBlockTreeCache(nullptr);
}

TEST_F(BlockTreeCacheFactoryTest, InsertRejectsWrongComponentShapeAndNullBlocksBeforeMutation) {
    const auto config    = makeCompatibleFullGroupsConfig(CacheEvictPolicy::CHAIN);
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);
    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->componentGroups().size(), 1u);
    ASSERT_EQ(cache->componentGroups()[0]->devicePoolCount(), 2u);

    const auto groups = allocator->cacheGroups();
    ASSERT_EQ(groups.size(), 2u);
    const auto first  = groups[0]->blockPool()->malloc(1);
    const auto second = groups[1]->blockPool()->malloc(1);
    ASSERT_EQ(first.size(), 1u);
    ASSERT_EQ(second.size(), 1u);

    auto expect_rejected_without_mutation = [&](CacheKeyType key, std::vector<std::vector<GroupSlot>> slots) {
        const auto before = cache->getKeySnapshot(/*limit=*/32);
        cache->insert(nullptr, CacheKeysType{key}, slots);
        const auto after = cache->getKeySnapshot(/*limit=*/32);
        EXPECT_EQ(after.version, before.version);
        EXPECT_EQ(after.keys, before.keys);
        EXPECT_EQ(cache->getStats().tree_node_count, 0u);
    };

    expect_rejected_without_mutation(/*key=*/710, {});
    GroupSlot valid;
    valid.device_blocks = {first[0], second[0]};
    expect_rejected_without_mutation(/*key=*/711, {{valid, valid}});

    GroupSlot wrong_cardinality;
    wrong_cardinality.device_blocks = {first[0]};
    expect_rejected_without_mutation(/*key=*/712, {{wrong_cardinality}});

    GroupSlot partially_null;
    partially_null.device_blocks = {first[0], NULL_BLOCK_IDX};
    expect_rejected_without_mutation(/*key=*/713, {{partially_null}});

    GroupSlot null_only;
    null_only.device_blocks = {NULL_BLOCK_IDX, NULL_BLOCK_IDX};
    expect_rejected_without_mutation(/*key=*/714, {{null_only}});

    groups[0]->blockPool()->requestFree(first);
    groups[1]->blockPool()->requestFree(second);
}

TEST_F(BlockTreeCacheFactoryTest, NoneEvictionPolicyIsNotSilentlyTreatedAsChain) {
    const auto config    = makeCompatibleFullGroupsConfig(CacheEvictPolicy::NONE);
    auto       allocator = initAllocator<HybridPoolKVCacheAllocator>(config);
    auto       cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->componentGroups().size(), 2u);
    EXPECT_EQ(cache->componentGroups()[0]->tags(), (std::vector<std::string>{"full_a"}));
    EXPECT_EQ(cache->componentGroups()[1]->tags(), (std::vector<std::string>{"full_b"}));
}

TEST_F(BlockTreeCacheFactoryTest, SharedPoolCopyLayoutUsesPerLayerPhysicalStrideTable) {
    auto config = makeHybridConfig(/*independent_pools=*/false);
    ASSERT_EQ(config.layer_to_block_stride_bytes.size(), 4u);
    auto allocator = initAllocator<HybridTypeKVCacheAllocator>(config);
    auto cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->components().size(), 2u);
    for (const auto& component : cache->components()) {
        const auto& declared = config.topology().group(component.tag);
        ASSERT_EQ(component.memory_block_layer_tag_slots.size(), declared.layer_ids.size());
        for (const auto& slot : component.memory_block_layer_tag_slots) {
            ASSERT_GE(slot.layer_id, 0);
            ASSERT_LT(static_cast<size_t>(slot.layer_id), config.layer_to_block_stride_bytes.size());
            EXPECT_EQ(slot.stride_bytes,
                      static_cast<size_t>(config.layer_to_block_stride_bytes[static_cast<size_t>(slot.layer_id)]));
        }
    }

    const auto& linear = config.topology().group("linear");
    ASSERT_FALSE(linear.layer_ids.empty());
    EXPECT_NE(static_cast<size_t>(config.layer_to_block_stride_bytes[static_cast<size_t>(linear.layer_ids[0])]),
              linear.kv_block_stride_bytes + linear.kv_scale_stride_bytes);
}

TEST_F(BlockTreeCacheFactoryTest, SharedPoolRejectsMissingZeroAndNegativePhysicalStride) {
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
        expectFactoryRejects(config, allocator);
    }
}

TEST_F(BlockTreeCacheFactoryTest, LegacySingleGroupAllowsMissingPhysicalStrideTableFallback) {
    auto config = makeSingleConfig();
    config.layer_to_block_stride_bytes.clear();
    auto allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
    auto cache     = createBlockTreeCache(config, KVCacheConfig{}, allocator);

    ASSERT_NE(cache, nullptr);
    ASSERT_EQ(cache->components().size(), 1u);
    const size_t fallback_stride =
        config.topology().groupById(0).kv_block_stride_bytes + config.topology().groupById(0).kv_scale_stride_bytes;
    ASSERT_FALSE(cache->components()[0].memory_block_layer_tag_slots.empty());
    for (const auto& slot : cache->components()[0].memory_block_layer_tag_slots) {
        EXPECT_EQ(slot.stride_bytes, fallback_stride);
    }
}

TEST_F(BlockTreeCacheFactoryTest, RejectsDiskCacheWithoutMemoryCacheBeforePublication) {
    const auto    config    = makeSingleConfig();
    auto          allocator = initAllocator<SingleTypeKVCacheAllocator>(config);
    KVCacheConfig kv_cache_config;
    kv_cache_config.enable_memory_cache      = false;
    kv_cache_config.enable_memory_cache_disk = true;

    expectFactoryRejects(config, allocator, kv_cache_config);
}

}  // namespace rtp_llm
