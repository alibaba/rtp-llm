#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string_view>
#include <thread>

#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm::test {
namespace {

BlockDependency rootDep(uint32_t ordinal = 0) {
    BlockDependency dep;
    dep.ordinal = ordinal;
    return dep;
}

BlockDependency childDep(CacheKeyType parent, uint32_t ordinal) {
    BlockDependency dep;
    dep.has_parent = true;
    dep.parent_key = parent;
    dep.ordinal    = ordinal;
    return dep;
}

CacheConfig makeTaggedCacheConfig() {
    CacheConfig config;
    config.dtype              = DataType::TYPE_FP16;
    config.layer_num          = 2;
    config.block_num          = 16;
    config.seq_size_per_block = 4;

    auto linear = makeResolvedMhaSpec(config.dtype, 1, 1, 4, "linear");
    auto full   = makeResolvedMhaSpec(config.dtype, 1, 1, 4, "full");
    rtp_llm::test::assignCacheConfigFromGroupedSpecs(config,
                                                     config.layer_num,
                                                     {linear, full},
                                                     {{0}, {1}},
                                                     {CacheGroupType::FULL, CacheGroupType::FULL},
                                                     {"linear", "full"});
    setGroupBlockLayout(config, {16, 16}, {linear->block_size_bytes(), full->block_size_bytes()}, {0, 0});
    return config;
}

CacheConfig makeSlotCacheConfig(size_t group_count) {
    CacheConfig config;
    config.dtype              = DataType::TYPE_FP16;
    config.layer_num          = static_cast<uint32_t>(group_count);
    config.block_num          = 2048;
    config.seq_size_per_block = 1;

    std::vector<KVCacheSpecPtr>   specs;
    std::vector<std::vector<int>> layer_ids;
    std::vector<CacheGroupType>   group_types;
    std::vector<std::string>      tags;
    std::vector<uint32_t>         block_nums;
    std::vector<size_t>           kv_strides;
    std::vector<size_t>           scale_strides;
    for (size_t slot = 0; slot < group_count; ++slot) {
        const auto tag  = "group" + std::to_string(slot);
        auto       spec = makeResolvedMhaSpec(config.dtype, 1, 1, 1, tag);
        kv_strides.push_back(spec->block_size_bytes());
        specs.push_back(std::move(spec));
        layer_ids.push_back({static_cast<int>(slot)});
        group_types.push_back(CacheGroupType::FULL);
        tags.push_back(tag);
        block_nums.push_back(2048);
        scale_strides.push_back(0);
    }
    rtp_llm::test::assignCacheConfigFromGroupedSpecs(config, config.layer_num, specs, layer_ids, group_types, tags);
    setGroupBlockLayout(config, block_nums, kv_strides, scale_strides);
    return config;
}

BlockPoolPtr makeLargeTestPool() {
    auto pool_config      = createTestConfig();
    pool_config.block_num = 2048;
    for (auto& layout : pool_config.memory_layouts) {
        layout.block_num                = pool_config.block_num;
        layout.kv_block_pool_size_bytes = layout.layer_num * layout.block_num * layout.kv_block_stride_bytes;
        layout.kv_scale_offset_bytes    = layout.kv_cache_offset_bytes + layout.kv_block_pool_size_bytes;
        layout.kv_scale_pool_size_bytes = layout.layer_num * layout.block_num * layout.kv_scale_stride_bytes;
        layout.total_size_bytes         = layout.kv_block_pool_size_bytes + layout.kv_scale_pool_size_bytes;
    }
    pool_config.total_size_bytes = pool_config.memory_layouts.front().total_size_bytes;
    auto pool                    = std::make_shared<BlockPool>(pool_config, AllocationType::HOST);
    EXPECT_TRUE(pool->init());
    return pool;
}

class PositionalSharedBlockCacheForTest: public SharedBlockCache {
public:
    PositionalSharedBlockCacheForTest(): config_(makeSlotCacheConfig(4)) {
        std::map<std::string, BlockPoolPtr> tagged_pools;
        for (const auto& group : config_.groups()) {
            auto pool = makeLargeTestPool();
            pools_.push_back(pool);
            tagged_pools.emplace(group.tag, std::move(pool));
        }
        init(config_, tagged_pools);
    }

    void put(CacheKeyType                     cache_key,
             const std::vector<BlockIdxType>& block_ids,
             bool                             is_resident,
             NamespaceId                      namespace_id     = kDefaultNamespace,
             const BlockDependency&           dependency       = {},
             const std::vector<bool>&         matchable_groups = {}) {
        std::map<std::string, BlockIdxType> groups;
        std::map<std::string, bool>         group_matchable;
        for (size_t slot = 0; slot < block_ids.size(); ++slot) {
            const auto tag = "group" + std::to_string(slot);
            groups.emplace(tag, block_ids[slot]);
            group_matchable.emplace(tag, slot >= matchable_groups.size() || matchable_groups[slot]);
        }
        SharedBlockCache::put(cache_key, groups, group_matchable, is_resident, namespace_id, dependency);
    }

    BlockIdxType matchGroup(CacheKeyType cache_key, size_t group_slot) {
        return SharedBlockCache::matchGroup(cache_key, "group" + std::to_string(group_slot));
    }

    EvictResult selectAndEvictForGroup(size_t group_slot, size_t min_blocks) {
        return SharedBlockCache::selectAndEvictForGroup("group" + std::to_string(group_slot), min_blocks);
    }

    void setIndependentGroupEviction(bool enabled, const std::vector<int>& group_slots) {
        std::vector<std::string> tags;
        tags.reserve(group_slots.size());
        for (const auto group_slot : group_slots) {
            tags.push_back("group" + std::to_string(group_slot));
        }
        SharedBlockCache::setIndependentGroupEviction(enabled, tags);
    }

private:
    CacheConfig               config_;
    std::vector<BlockPoolPtr> pools_;
};

void putOne(PositionalSharedBlockCacheForTest& cache,
            CacheKeyType                       key,
            BlockIdxType                       block,
            const BlockDependency&             dep,
            NamespaceId                        namespace_id = SharedBlockCache::kGpuLogicalNamespace,
            bool                               resident     = false) {
    cache.put(key, std::vector<BlockIdxType>{block}, resident, namespace_id, dep);
}

BlockIdxType blockByTag(const std::map<std::string, BlockIdxType>& groups, std::string_view tag) {
    const auto it = groups.find(std::string(tag));
    RTP_LLM_CHECK_WITH_INFO(it != groups.end(), "missing tagged SharedBlockCache group=%s", std::string(tag).c_str());
    return it->second;
}

const SharedGroupBinding& bindingByTag(const UnifiedCacheItem& item, std::string_view tag) {
    const auto it = item.bindings_by_group.find(std::string(tag));
    RTP_LLM_CHECK_WITH_INFO(
        it != item.bindings_by_group.end(), "missing SharedBlockCache binding=%s", std::string(tag).c_str());
    return it->second;
}

CacheKeysType evictionKeys(const EvictResult& result) {
    CacheKeysType keys;
    keys.reserve(result.evictions.size());
    for (const auto& eviction : result.evictions) {
        keys.push_back(eviction.cache_key);
    }
    return keys;
}

const CacheEviction& evictionByKey(const EvictResult& result, CacheKeyType cache_key) {
    const auto it = std::find_if(result.evictions.begin(), result.evictions.end(), [cache_key](const auto& eviction) {
        return eviction.cache_key == cache_key;
    });
    RTP_LLM_CHECK_WITH_INFO(it != result.evictions.end(), "missing SharedBlockCache eviction key=%ld", cache_key);
    return *it;
}

class RecordingSharedBlockCache: public SharedBlockCache {
public:
    std::vector<std::string> operations;

protected:
    void blockCacheReferenceByTag(std::string_view tag, BlockIdxType block_id) override {
        operations.push_back("reference:" + std::string(tag));
        SharedBlockCache::blockCacheReferenceByTag(tag, block_id);
    }

    void blockCacheFreeByTag(std::string_view tag, BlockIdxType block_id) override {
        operations.push_back("free:" + std::string(tag));
        SharedBlockCache::blockCacheFreeByTag(tag, block_id);
    }
};

}  // namespace

TEST(SharedBlockCacheTest, TaggedBoundaryDistinguishesSameBlockIdAcrossShuffledTags) {
    auto config      = makeTaggedCacheConfig();
    auto linear_pool = createBlockPool();
    auto full_pool   = createBlockPool();
    ASSERT_TRUE(linear_pool->init());
    ASSERT_TRUE(full_pool->init());
    const auto linear_block = linear_pool->malloc(1).at(0);
    const auto full_block   = full_pool->malloc(1).at(0);
    ASSERT_EQ(linear_block, full_block);

    SharedBlockCache tagged_cache;
    tagged_cache.init(config, {{"full", full_pool}, {"linear", linear_pool}});
    tagged_cache.put(42, {{"full", full_block}, {"linear", linear_block}}, /*is_resident=*/false);

    EXPECT_EQ(tagged_cache.matchGroup(42, "linear"), linear_block);
    EXPECT_EQ(tagged_cache.matchGroup(42, "full"), full_block);
    const auto removed = tagged_cache.remove(42);
    ASSERT_TRUE(removed.has_value());
    EXPECT_EQ(bindingByTag(*removed, "linear").pool_block_id, linear_block);
    EXPECT_EQ(bindingByTag(*removed, "full").pool_block_id, full_block);
}

TEST(SharedBlockCacheTest, TaggedEvictionReportsShuffledTagIdentity) {
    auto config      = makeTaggedCacheConfig();
    auto linear_pool = createBlockPool();
    auto full_pool   = createBlockPool();
    ASSERT_TRUE(linear_pool->init());
    ASSERT_TRUE(full_pool->init());
    const auto linear_block = linear_pool->malloc(1).at(0);
    const auto full_block   = full_pool->malloc(1).at(0);
    ASSERT_EQ(linear_block, full_block);

    SharedBlockCache tagged_cache;
    tagged_cache.init(config, {{"full", full_pool}, {"linear", linear_pool}});
    tagged_cache.setPrefixTreeEnabled(false);
    tagged_cache.put(7, {{"full", full_block}, {"linear", linear_block}}, /*is_resident=*/false);

    const auto evicted = tagged_cache.selectAndEvict(/*min_blocks=*/2);
    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{7}));
    const auto& groups = evictionByKey(evicted, 7).blocks_by_group;
    EXPECT_EQ(blockByTag(groups, "linear"), linear_block);
    EXPECT_EQ(blockByTag(groups, "full"), full_block);
}

TEST(SharedBlockCacheTest, TaggedOperationTracePreservesVersionsReferencesDependenciesAndLifetimes) {
    auto config      = makeTaggedCacheConfig();
    auto linear_pool = createBlockPool();
    auto full_pool   = createBlockPool();
    ASSERT_TRUE(linear_pool->init());
    ASSERT_TRUE(full_pool->init());
    const auto linear_free_before = linear_pool->freeBlocksNum();
    const auto full_free_before   = full_pool->freeBlocksNum();
    const auto linear_block       = linear_pool->malloc(1).at(0);
    const auto full_block         = full_pool->malloc(1).at(0);
    const auto global_block       = linear_pool->malloc(1).at(0);
    const auto group_block        = linear_pool->malloc(1).at(0);
    linear_pool->requestFree({linear_block, global_block, group_block});
    full_pool->requestFree({full_block});
    SharedBlockCache cache;
    cache.init(config, {{"linear", linear_pool}, {"full", full_pool}});
    EXPECT_EQ(cache.version(), -1);

    cache.put(10, {{"linear", linear_block}}, {}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    EXPECT_EQ(cache.version(), 0);
    EXPECT_EQ(linear_pool->blockCacheRefBlocksNum(), 1u);
    EXPECT_EQ(cache.matchGroup(10, "linear"), linear_block);
    cache.put(10, {{"full", full_block}}, {}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(9, 2));
    EXPECT_EQ(cache.version(), 1);
    EXPECT_EQ(full_pool->blockCacheRefBlocksNum(), 1u);
    EXPECT_EQ(cache.matchGroup(10, "full"), full_block);
    const auto removed = cache.remove(10);
    ASSERT_TRUE(removed.has_value());
    EXPECT_TRUE(removed->dependency.has_parent);
    EXPECT_EQ(removed->dependency.parent_key, 9);
    EXPECT_EQ(removed->dependency.ordinal, 2u);
    EXPECT_FALSE(cache.contains(10));
    EXPECT_EQ(cache.version(), 1);
    EXPECT_EQ(linear_pool->blockCacheRefBlocksNum(), 1u);
    EXPECT_EQ(full_pool->blockCacheRefBlocksNum(), 1u);
    linear_pool->blockCacheFree(linear_block);
    full_pool->blockCacheFree(full_block);
    EXPECT_EQ(linear_pool->blockCacheRefBlocksNum(), 0u);
    EXPECT_EQ(full_pool->blockCacheRefBlocksNum(), 0u);

    cache.setPrefixTreeEnabled(false);
    cache.put(20, {{"linear", global_block}}, {}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(3));
    EXPECT_EQ(cache.version(), 2);
    const auto global = cache.selectAndEvict(1);
    ASSERT_EQ(evictionKeys(global), (CacheKeysType{20}));
    EXPECT_GE(evictionByKey(global, 20).lifetime_ms, 0);
    EXPECT_EQ(blockByTag(evictionByKey(global, 20).blocks_by_group, "linear"), global_block);
    EXPECT_EQ(evictionByKey(global, 20).dependency.ordinal, 3u);
    linear_pool->blockCacheFree(global_block);
    EXPECT_EQ(linear_pool->blockCacheRefBlocksNum(), 0u);

    cache.put(30, {{"linear", group_block}}, {}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(4));
    EXPECT_EQ(cache.version(), 3);
    EvictResult group;
    EXPECT_EQ(cache.evictAndFreeForGroup("linear", 1, &group), 1u);
    ASSERT_EQ(evictionKeys(group), (CacheKeysType{30}));
    EXPECT_GE(evictionByKey(group, 30).lifetime_ms, 0);
    EXPECT_EQ(blockByTag(evictionByKey(group, 30).blocks_by_group, "linear"), group_block);
    EXPECT_EQ(evictionByKey(group, 30).dependency.ordinal, 4u);
    EXPECT_EQ(linear_pool->blockCacheRefBlocksNum(), 0u);
    EXPECT_EQ(linear_pool->freeBlocksNum(), linear_free_before);
    EXPECT_EQ(full_pool->freeBlocksNum(), full_free_before);
}

TEST(SharedBlockCacheTest, IncrementalPutKeepsExistingBindingAndPromotesStateWithoutDuplicateReference) {
    auto config      = makeTaggedCacheConfig();
    auto linear_pool = createBlockPool();
    auto full_pool   = createBlockPool();
    ASSERT_TRUE(linear_pool->init());
    ASSERT_TRUE(full_pool->init());

    const auto linear_block      = linear_pool->malloc(1).at(0);
    const auto conflicting_block = linear_pool->malloc(1).at(0);
    linear_pool->requestFree({linear_block, conflicting_block});

    SharedBlockCache cache;
    cache.init(config, {{"linear", linear_pool}, {"full", full_pool}});
    cache.put(50,
              {{"linear", linear_block}},
              {{"linear", false}},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    EXPECT_EQ(linear_pool->blockCacheRefBlocksNum(), 1u);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(50, "linear")));

    cache.put(50,
              {{"linear", NULL_BLOCK_IDX}},
              {{"linear", true}},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    EXPECT_EQ(linear_pool->blockCacheRefBlocksNum(), 1u);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(50, "linear")));

    const auto version_before_ignored_rebind = cache.version();
    cache.put(50,
              {{"linear", conflicting_block}},
              {},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    EXPECT_EQ(cache.version(), version_before_ignored_rebind);
    EXPECT_EQ(linear_pool->blockCacheRefBlocksNum(), 1u);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(50, "linear")));

    cache.put(50,
              {{"linear", conflicting_block}},
              {{"linear", true}},
              /*is_resident=*/true,
              SharedBlockCache::kGpuCpCanonicalNamespace,
              rootDep(7));
    EXPECT_EQ(cache.version(), version_before_ignored_rebind + 1);
    EXPECT_EQ(linear_pool->blockCacheRefBlocksNum(), 1u);
    EXPECT_EQ(cache.matchGroup(50, "linear"), linear_block);

    cache.put(50,
              {{"linear", conflicting_block}},
              {{"linear", false}},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(49, 9));
    EXPECT_EQ(linear_pool->blockCacheRefBlocksNum(), 1u);
    EXPECT_EQ(cache.matchGroup(50, "linear"), linear_block);

    const auto removed = cache.remove(50);
    ASSERT_TRUE(removed.has_value());
    const auto& binding = bindingByTag(*removed, "linear");
    EXPECT_EQ(binding.pool_block_id, linear_block);
    EXPECT_TRUE(binding.matchable);
    EXPECT_EQ(binding.created_time_us, removed->created_time_us);
    EXPECT_TRUE(removed->is_resident);
    EXPECT_TRUE(removed->has_dependency);
    EXPECT_EQ(removed->dependency_namespace, SharedBlockCache::kGpuCpCanonicalNamespace);
    EXPECT_FALSE(removed->dependency.has_parent);
    EXPECT_EQ(removed->dependency.ordinal, 7u);
    linear_pool->blockCacheFree(linear_block);
}

TEST(SharedBlockCacheTest, MultiGroupReferenceAndFreeSideEffectsFollowTopologyOrder) {
    auto config      = makeTaggedCacheConfig();  // topology: linear, full; lexical map order: full, linear
    auto linear_pool = createBlockPool();
    auto full_pool   = createBlockPool();
    ASSERT_TRUE(linear_pool->init());
    ASSERT_TRUE(full_pool->init());

    const auto linear_block = linear_pool->malloc(1).at(0);
    const auto full_block   = full_pool->malloc(1).at(0);
    linear_pool->requestFree(linear_block);
    full_pool->requestFree(full_block);

    RecordingSharedBlockCache cache;
    cache.init(config, {{"full", full_pool}, {"linear", linear_pool}});
    cache.setPrefixTreeEnabled(false);
    cache.put(40, {{"full", full_block}, {"linear", linear_block}}, /*is_resident=*/false);
    EXPECT_EQ(cache.operations, (std::vector<std::string>{"reference:linear", "reference:full"}));

    cache.operations.clear();
    EXPECT_EQ(cache.evictAndFree(/*min_blocks=*/2), 2u);
    EXPECT_EQ(cache.operations, (std::vector<std::string>{"free:linear", "free:full"}));

    const auto next_linear_block = linear_pool->malloc(1).at(0);
    const auto next_full_block   = full_pool->malloc(1).at(0);
    linear_pool->requestFree(next_linear_block);
    full_pool->requestFree(next_full_block);
    cache.put(41, {{"full", NULL_BLOCK_IDX}, {"linear", NULL_BLOCK_IDX}}, /*is_resident=*/false);

    cache.operations.clear();
    cache.put(41, {{"full", next_full_block}, {"linear", next_linear_block}}, /*is_resident=*/false);
    EXPECT_EQ(cache.operations, (std::vector<std::string>{"reference:linear", "reference:full"}));

    cache.operations.clear();
    EXPECT_EQ(cache.evictAndFreeForGroup("linear", /*min_blocks=*/1), 1u);
    EXPECT_EQ(cache.operations, (std::vector<std::string>{"free:linear", "free:full"}));
}

TEST(SharedBlockCacheTest, TaggedRegistryRejectsUnknownDuplicateAndMissingTags) {
    auto config      = makeTaggedCacheConfig();
    auto linear_pool = createBlockPool();
    auto full_pool   = createBlockPool();

    SharedBlockCache unknown;
    EXPECT_THROW(unknown.init(config, {{"linear", linear_pool}, {"unknown", full_pool}}), RTPException);

    SharedBlockCache duplicate;
    EXPECT_THROW(duplicate.init(config, {{"linear", linear_pool}, {"linear", full_pool}}), RTPException);

    SharedBlockCache missing;
    EXPECT_THROW(missing.init(config, {{"linear", linear_pool}}), RTPException);
}

TEST(SharedBlockCacheTest, EmptyCacheKeepsLegacyVersion) {
    PositionalSharedBlockCacheForTest cache;
    EXPECT_EQ(cache.version(), -1);
}

TEST(SharedBlockCacheTest, PrefixTreeEvictsCollectedChainInParentFirstOrderWithDependencies) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 1, 101, rootDep(0));
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, childDep(2, 2));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{1, 2, 3}));
    ASSERT_EQ(blockByTag(evictionByKey(evicted, 1).blocks_by_group, "group0"), 101);
    ASSERT_FALSE(evictionByKey(evicted, 1).dependency.has_parent);
    ASSERT_TRUE(evictionByKey(evicted, 2).dependency.has_parent);
    ASSERT_EQ(evictionByKey(evicted, 2).dependency.parent_key, 1);
    ASSERT_TRUE(evictionByKey(evicted, 3).dependency.has_parent);
    ASSERT_EQ(evictionByKey(evicted, 3).dependency.parent_key, 2);
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, PrefixTreeStopsAtBranchPoint) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 1, 101, rootDep(0));
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, childDep(1, 2));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{2}));
    EXPECT_FALSE(cache.contains(2));
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(3));
}

TEST(SharedBlockCacheTest, PrefixTreeLinksChildInsertedBeforeParent) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 1, 101, rootDep(0));

    ASSERT_EQ(cache.matchGroup(2, 0), 102);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{1, 2}));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, PrefixTreeEvictsOrphanLeafWithMissingParentDependency) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 2, 102, childDep(1, 1));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{2}));
    ASSERT_TRUE(evictionByKey(evicted, 2).has_dependency);
    EXPECT_TRUE(evictionByKey(evicted, 2).dependency.has_parent);
    EXPECT_EQ(evictionByKey(evicted, 2).dependency.parent_key, 1);
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, PrefixTreeAttachesMultiplePendingChildrenAndStopsAtBranch) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, childDep(1, 2));
    putOne(cache, 1, 101, rootDep(0));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{2}));
    EXPECT_FALSE(cache.contains(2));
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(3));

    evicted = cache.selectAndEvict(/*min_blocks=*/1);
    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{1, 3}));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, PrefixTreeStopsAtResidentParent) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 1, 101, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/true);
    putOne(cache, 2, 102, childDep(1, 1));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{2}));
    ASSERT_TRUE(evictionByKey(evicted, 2).has_dependency);
    EXPECT_TRUE(evictionByKey(evicted, 2).dependency.has_parent);
    EXPECT_EQ(evictionByKey(evicted, 2).dependency.parent_key, 1);
    EXPECT_TRUE(cache.contains(1));
    EXPECT_FALSE(cache.contains(2));
}

TEST(SharedBlockCacheTest, MatchGroupTouchesPrefixTreeLeafLru) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 1, 101, rootDep(0));
    putOne(cache, 2, 102, childDep(1, 1));
    putOne(cache, 3, 103, rootDep(0));

    ASSERT_EQ(cache.matchGroup(2, 0), 102);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{3}));
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
    EXPECT_FALSE(cache.contains(3));
}

TEST(SharedBlockCacheTest, ResidentIsStickyAcrossPuts) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 1, 101, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/false);
    putOne(cache, 1, NULL_BLOCK_IDX, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/true);
    putOne(cache, 1, NULL_BLOCK_IDX, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/false);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    EXPECT_TRUE(evictionKeys(evicted).empty());
    EXPECT_TRUE(cache.contains(1));
}

TEST(SharedBlockCacheTest, ResidentIsStickyAcrossNamespaceAliases) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 1, 101, rootDep(0), SharedBlockCache::kGpuLogicalNamespace, /*resident=*/false);
    putOne(cache, 1, NULL_BLOCK_IDX, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace, /*resident=*/true);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    EXPECT_TRUE(evictionKeys(evicted).empty());
    EXPECT_TRUE(cache.contains(1));
}

TEST(SharedBlockCacheTest, PrefixTreeEvictionReportsNamespace) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 1, 101, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{1}));
    ASSERT_TRUE(evictionByKey(evicted, 1).has_dependency);
    EXPECT_EQ(evictionByKey(evicted, 1).dependency_namespace, SharedBlockCache::kGpuCpCanonicalNamespace);
}

TEST(SharedBlockCacheTest, PrefixTreeEvictionKeepsCanonicalDependencyWhenLogicalAliasUpdatesSameKey) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 8, 108, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace);
    putOne(cache, 8, NULL_BLOCK_IDX, childDep(7, 7), SharedBlockCache::kGpuLogicalNamespace);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{8}));
    ASSERT_TRUE(evictionByKey(evicted, 8).has_dependency);
    EXPECT_FALSE(evictionByKey(evicted, 8).dependency.has_parent);
    EXPECT_EQ(evictionByKey(evicted, 8).dependency.ordinal, 0u);
    ASSERT_TRUE(evictionByKey(evicted, 8).has_dependency);
    EXPECT_EQ(evictionByKey(evicted, 8).dependency_namespace, SharedBlockCache::kGpuCpCanonicalNamespace);
}

TEST(SharedBlockCacheTest, CanonicalAliasOwnsEvictionWhenLogicalAliasIsOlder) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 100, 1000, rootDep(0), SharedBlockCache::kGpuLogicalNamespace);
    putOne(cache, 101, 1010, childDep(100, 1), SharedBlockCache::kGpuLogicalNamespace);
    putOne(cache, 102, 1020, childDep(101, 2), SharedBlockCache::kGpuLogicalNamespace);
    putOne(cache, 103, 1030, childDep(102, 3), SharedBlockCache::kGpuLogicalNamespace);

    putOne(cache, 101, NULL_BLOCK_IDX, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace);
    putOne(cache, 103, NULL_BLOCK_IDX, childDep(101, 1), SharedBlockCache::kGpuCpCanonicalNamespace);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{101, 103}));
    ASSERT_TRUE(evictionByKey(evicted, 101).has_dependency);
    EXPECT_FALSE(evictionByKey(evicted, 101).dependency.has_parent);
    ASSERT_TRUE(evictionByKey(evicted, 103).has_dependency);
    EXPECT_TRUE(evictionByKey(evicted, 103).dependency.has_parent);
    EXPECT_EQ(evictionByKey(evicted, 103).dependency.parent_key, 101);
    EXPECT_EQ(evictionByKey(evicted, 101).dependency_namespace, SharedBlockCache::kGpuCpCanonicalNamespace);
    EXPECT_EQ(evictionByKey(evicted, 103).dependency_namespace, SharedBlockCache::kGpuCpCanonicalNamespace);
    EXPECT_TRUE(cache.contains(100));
    EXPECT_TRUE(cache.contains(102));
}

TEST(SharedBlockCacheTest, FlatFallbackKeepsCanonicalDependencyWhenLogicalAliasUpdatesSameKey) {
    PositionalSharedBlockCacheForTest cache;
    cache.setPrefixTreeEnabled(false);

    putOne(cache, 8, 108, rootDep(0), SharedBlockCache::kGpuCpCanonicalNamespace);
    putOne(cache, 8, NULL_BLOCK_IDX, childDep(7, 7), SharedBlockCache::kGpuLogicalNamespace);

    auto evicted = cache.selectAndEvict(/*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{8}));
    ASSERT_TRUE(evictionByKey(evicted, 8).has_dependency);
    EXPECT_FALSE(evictionByKey(evicted, 8).dependency.has_parent);
    EXPECT_EQ(evictionByKey(evicted, 8).dependency.ordinal, 0u);
    ASSERT_TRUE(evictionByKey(evicted, 8).has_dependency);
    EXPECT_EQ(evictionByKey(evicted, 8).dependency_namespace, SharedBlockCache::kGpuCpCanonicalNamespace);
}

TEST(SharedBlockCacheTest, NonMatchableSlotStillEvictsButDoesNotMatchGroup) {
    PositionalSharedBlockCacheForTest cache;
    cache.put(1,
              std::vector<BlockIdxType>{101, 201},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0),
              std::vector<bool>{true, false});

    EXPECT_EQ(cache.matchGroup(1, 0), 101);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(1, 1)));

    auto evicted = cache.selectAndEvict(/*min_blocks=*/2);
    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{1}));
    ASSERT_EQ(blockByTag(evictionByKey(evicted, 1).blocks_by_group, "group0"), 101);
    ASSERT_EQ(blockByTag(evictionByKey(evicted, 1).blocks_by_group, "group1"), 201);
}

TEST(SharedBlockCacheTest, StateIndependentEvictionDropsDeepestNonLeafStateFirst) {
    PositionalSharedBlockCacheForTest cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {3});

    cache.put(1,
              std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 301},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              std::vector<BlockIdxType>{103, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 303},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(2, 2));

    auto evicted = cache.selectAndEvictForGroup(/*group_slot=*/3, /*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{2}));
    ASSERT_EQ(blockByTag(evictionByKey(evicted, 2).blocks_by_group, "group3"), 302);
    ASSERT_EQ(evictionByKey(evicted, 2).kind, EvictionKind::IndependentGroup);
    EXPECT_EQ(evictionByKey(evicted, 2).group_tag, "group3");
    EXPECT_EQ(cache.matchGroup(2, 0), 102);
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(2, 3)));
    EXPECT_EQ(cache.matchGroup(3, 3), 303);
}

TEST(SharedBlockCacheTest, IncrementalPutAndEvictionRecordsPreserveVersionBindingAndLifetimeOrigins) {
    PositionalSharedBlockCacheForTest cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {3});

    cache.put(1, std::vector<BlockIdxType>{101}, false, SharedBlockCache::kGpuLogicalNamespace, rootDep(0));
    cache.put(2, std::vector<BlockIdxType>{102}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(1, 1));
    cache.put(3, std::vector<BlockIdxType>{103}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(2, 2));

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));

    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    const auto version_before_idempotent_put = cache.version();
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    EXPECT_EQ(cache.version(), version_before_idempotent_put);

    // group0 is absent from this update and must remain bound. A false matchable
    // input for group3 must not demote its existing true state.
    cache.SharedBlockCache::put(
        2, {{"group3", 302}}, {{"group3", false}}, false, SharedBlockCache::kGpuLogicalNamespace, childDep(1, 1));
    EXPECT_EQ(cache.version(), version_before_idempotent_put);
    EXPECT_EQ(cache.matchGroup(2, 0), 102);
    EXPECT_EQ(cache.matchGroup(2, 3), 302);

    const auto version_before_independent_eviction = cache.version();
    const auto independent                         = cache.selectAndEvictForGroup(/*group_slot=*/3, /*min_blocks=*/1);
    EXPECT_EQ(cache.version(), version_before_independent_eviction + 1);
    ASSERT_EQ(independent.evictions.size(), 1u);
    const auto& independent_record = independent.evictions.front();
    EXPECT_EQ(independent_record.cache_key, 2);
    EXPECT_EQ(independent_record.blocks_by_group, (std::map<std::string, BlockIdxType>{{"group3", 302}}));
    EXPECT_TRUE(independent_record.has_dependency);
    EXPECT_TRUE(independent_record.dependency.has_parent);
    EXPECT_EQ(independent_record.dependency.parent_key, 1);
    EXPECT_EQ(independent_record.dependency.ordinal, 1u);
    EXPECT_EQ(independent_record.dependency_namespace, SharedBlockCache::kGpuLogicalNamespace);
    EXPECT_EQ(independent_record.kind, EvictionKind::IndependentGroup);
    EXPECT_EQ(independent_record.group_tag, "group3");
    EXPECT_GE(independent_record.lifetime_ms, 30);

    const auto whole = cache.selectAndEvict(/*min_blocks=*/1);
    ASSERT_EQ(evictionKeys(whole), (CacheKeysType{1, 2, 3}));
    const auto& whole_record = evictionByKey(whole, 2);
    EXPECT_EQ(whole_record.blocks_by_group, (std::map<std::string, BlockIdxType>{{"group0", 102}}));
    EXPECT_TRUE(whole_record.has_dependency);
    EXPECT_TRUE(whole_record.dependency.has_parent);
    EXPECT_EQ(whole_record.dependency.parent_key, 1);
    EXPECT_EQ(whole_record.dependency.ordinal, 1u);
    EXPECT_EQ(whole_record.dependency_namespace, SharedBlockCache::kGpuLogicalNamespace);
    EXPECT_EQ(whole_record.kind, EvictionKind::WholeItem);
    EXPECT_TRUE(whole_record.group_tag.empty());
    EXPECT_GE(whole_record.lifetime_ms, independent_record.lifetime_ms + 30);
}

TEST(SharedBlockCacheTest, StateIndependentEvictionScansMultipleLeavesSafely) {
    PositionalSharedBlockCacheForTest cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {3});

    cache.put(1,
              std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 301},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              std::vector<BlockIdxType>{103, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 303},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(2, 2));
    cache.put(10,
              std::vector<BlockIdxType>{110, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 310},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(11,
              std::vector<BlockIdxType>{111, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 311},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(10, 1));
    cache.put(12,
              std::vector<BlockIdxType>{112, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 312},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(11, 2));

    auto evicted = cache.selectAndEvictForGroup(/*group_slot=*/3, /*min_blocks=*/2);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{2, 11}));
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(2, 3)));
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(11, 3)));
    EXPECT_EQ(cache.matchGroup(3, 3), 303);
    EXPECT_EQ(cache.matchGroup(12, 3), 312);
}

TEST(SharedBlockCacheTest, StateIndependentEvictionFallsBackToWholeChainWhenOnlyLeafStateRemains) {
    PositionalSharedBlockCacheForTest cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {3});

    cache.put(1,
              std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));

    auto evicted = cache.selectAndEvictForGroup(/*group_slot=*/3, /*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{1, 2}));
    ASSERT_EQ(evictionByKey(evicted, 2).kind, EvictionKind::WholeItem);
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupSkipsChainsWithoutTargetSlot) {
    PositionalSharedBlockCacheForTest cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {3});

    cache.put(1,
              std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(10,
              std::vector<BlockIdxType>{110, NULL_BLOCK_IDX, NULL_BLOCK_IDX, NULL_BLOCK_IDX},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(11,
              std::vector<BlockIdxType>{111, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 311},
              false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(10, 1));

    auto evicted = cache.selectAndEvictForGroup(/*group_slot=*/3, /*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{10, 11}));
    EXPECT_FALSE(cache.contains(10));
    EXPECT_FALSE(cache.contains(11));
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupPrunesBranchUntilTargetAncestorIsEvictable) {
    PositionalSharedBlockCacheForTest cache;
    cache.put(1,
              std::vector<BlockIdxType>{101, 201},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              std::vector<BlockIdxType>{103, NULL_BLOCK_IDX},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 2));

    auto evicted = cache.selectAndEvictForGroup(/*group_slot=*/1, /*min_blocks=*/1);

    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{2, 1, 3}));
    ASSERT_EQ(blockByTag(evictionByKey(evicted, 1).blocks_by_group, "group0"), 101);
    ASSERT_EQ(blockByTag(evictionByKey(evicted, 1).blocks_by_group, "group1"), 201);
    EXPECT_FALSE(evictionByKey(evicted, 2).blocks_by_group.count("group1"));
    EXPECT_FALSE(evictionByKey(evicted, 3).blocks_by_group.count("group1"));
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupDoesNotPruneWhenTargetAncestorBlockedByResidentSibling) {
    PositionalSharedBlockCacheForTest cache;
    cache.put(1,
              std::vector<BlockIdxType>{101, 201},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              std::vector<BlockIdxType>{103, NULL_BLOCK_IDX},
              /*is_resident=*/true,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 2));

    auto evicted = cache.selectAndEvictForGroup(/*group_slot=*/1, /*min_blocks=*/1);

    EXPECT_TRUE(evictionKeys(evicted).empty());
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
    EXPECT_TRUE(cache.contains(3));
}

TEST(SharedBlockCacheTest, SelectAndEvictForGroupDoesNotPruneWhenTargetAncestorBlockedByResidentDescendant) {
    PositionalSharedBlockCacheForTest cache;
    cache.put(1,
              std::vector<BlockIdxType>{101, 201},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1));
    cache.put(3,
              std::vector<BlockIdxType>{103, NULL_BLOCK_IDX},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 2));
    cache.put(4,
              std::vector<BlockIdxType>{104, NULL_BLOCK_IDX},
              /*is_resident=*/true,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(3, 3));

    auto evicted = cache.selectAndEvictForGroup(/*group_slot=*/1, /*min_blocks=*/1);

    EXPECT_TRUE(evictionKeys(evicted).empty());
    EXPECT_TRUE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
    EXPECT_TRUE(cache.contains(3));
    EXPECT_TRUE(cache.contains(4));
}

// ---------------------------------------------------------------------------
// Invalid-input boundaries. These only exercise rejection and no-op paths;
// valid-path locking, LRU, tree, version and reference behavior is untouched.
// ---------------------------------------------------------------------------

TEST(SharedBlockCacheTest, ZeroWorkGroupRequestsDoNothingAndKeepCacheIntact) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 1, 101, rootDep(0));
    const auto version_before = cache.version();

    // min_blocks == 0 is zero work: selectAndEvictForGroup returns before the
    // tag is resolved, so even an invalid tag is accepted and nothing happens.
    EXPECT_TRUE(cache.selectAndEvictForGroup(/*group_slot=*/0, /*min_blocks=*/0).evictions.empty());
    EXPECT_TRUE(cache.SharedBlockCache::selectAndEvictForGroup("", /*min_blocks=*/0).evictions.empty());
    EXPECT_TRUE(cache.SharedBlockCache::selectAndEvictForGroup("no_such_group", /*min_blocks=*/0).evictions.empty());
    EXPECT_TRUE(cache.selectAndEvict(/*min_blocks=*/0).evictions.empty());
    EXPECT_EQ(cache.evictAndFreeForGroup("group0", /*min_blocks=*/0), 0u);

    EXPECT_EQ(cache.version(), version_before);
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_EQ(cache.matchGroup(1, 0), 101);
}

TEST(SharedBlockCacheTest, InvalidGroupTagsAreRejectedOnRealWork) {
    PositionalSharedBlockCacheForTest cache;
    putOne(cache, 1, 101, rootDep(0));

    EXPECT_THROW(cache.SharedBlockCache::matchGroup(1, ""), RTPException);
    EXPECT_THROW(cache.SharedBlockCache::matchGroup(1, "no_such_group"), RTPException);
    EXPECT_THROW(cache.SharedBlockCache::selectAndEvictForGroup("", /*min_blocks=*/1), RTPException);
    EXPECT_THROW(cache.SharedBlockCache::selectAndEvictForGroup("no_such_group", /*min_blocks=*/1), RTPException);
    // evictAndFreeForGroup resolves the target tag unconditionally, so an
    // invalid tag is rejected there even when no eviction work is requested.
    EXPECT_THROW(cache.evictAndFreeForGroup("no_such_group", /*min_blocks=*/0), RTPException);
    EXPECT_THROW(cache.SharedBlockCache::setIndependentGroupEviction(/*enabled=*/true, {"no_such_group"}),
                 RTPException);

    EXPECT_EQ(cache.size(), 1u);
    EXPECT_EQ(cache.matchGroup(1, 0), 101);
}

TEST(SharedBlockCacheTest, TaggedRegistryRejectsEmptyTagNullPoolAndEmptyRegistry) {
    auto config      = makeTaggedCacheConfig();
    auto linear_pool = createBlockPool();
    auto full_pool   = createBlockPool();

    SharedBlockCache empty_tag;
    EXPECT_THROW(empty_tag.init(config, {{"linear", linear_pool}, {"", full_pool}}), RTPException);

    SharedBlockCache null_pool;
    EXPECT_THROW(null_pool.init(config, {{"linear", linear_pool}, {"full", nullptr}}), RTPException);

    SharedBlockCache empty_registry;
    EXPECT_THROW(empty_registry.init(config, {}), RTPException);
}

TEST(SharedBlockCacheTest, PutRejectsUnknownAndEmptyInputTags) {
    auto config      = makeTaggedCacheConfig();
    auto linear_pool = createBlockPool();
    auto full_pool   = createBlockPool();
    ASSERT_TRUE(linear_pool->init());
    ASSERT_TRUE(full_pool->init());

    SharedBlockCache cache;
    cache.init(config, {{"linear", linear_pool}, {"full", full_pool}});
    const auto linear_block = linear_pool->malloc(1).at(0);

    EXPECT_THROW(cache.put(2, {{"no_such_group", linear_block}}, /*is_resident=*/false), RTPException);
    EXPECT_THROW(cache.put(3, {{"", linear_block}}, /*is_resident=*/false), RTPException);

    // Rejection happens before any cache mutation.
    EXPECT_TRUE(cache.empty());
    EXPECT_EQ(cache.version(), -1);
}

TEST(SharedBlockCacheTest, PutWithNoGroupEntriesRecordsKeyWithoutGroupBlocks) {
    auto config      = makeTaggedCacheConfig();
    auto linear_pool = createBlockPool();
    auto full_pool   = createBlockPool();
    ASSERT_TRUE(linear_pool->init());
    ASSERT_TRUE(full_pool->init());

    SharedBlockCache cache;
    cache.init(config, {{"linear", linear_pool}, {"full", full_pool}});
    cache.put(9, {}, /*is_resident=*/false);

    EXPECT_TRUE(cache.contains(9));
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(9, "linear")));
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(9, "full")));

    const auto removed = cache.remove(9);
    ASSERT_TRUE(removed.has_value());
    EXPECT_TRUE(removed->bindings_by_group.empty());
    EXPECT_TRUE(cache.empty());
}

TEST(SharedBlockCacheTest, IndependentEvictionStillReclaimsNonMatchableTargetGroup) {
    PositionalSharedBlockCacheForTest cache;
    cache.setIndependentGroupEviction(/*enabled=*/true, {3});

    cache.put(1,
              std::vector<BlockIdxType>{101, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 301},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              rootDep(0));
    cache.put(2,
              std::vector<BlockIdxType>{102, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 302},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(1, 1),
              /*matchable_groups=*/std::vector<bool>{true, true, true, false});
    cache.put(3,
              std::vector<BlockIdxType>{103, NULL_BLOCK_IDX, NULL_BLOCK_IDX, 303},
              /*is_resident=*/false,
              SharedBlockCache::kGpuLogicalNamespace,
              childDep(2, 2));

    // Non-matchable metadata is unusable for prefix matching...
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(2, 3)));
    EXPECT_EQ(cache.matchGroup(2, 0), 102);

    // ...but the block is still really reserved, so independent group eviction
    // reclaims it exactly as it does a matchable one.
    auto evicted = cache.selectAndEvictForGroup(/*group_slot=*/3, /*min_blocks=*/1);
    ASSERT_EQ(evictionKeys(evicted), (CacheKeysType{2}));
    EXPECT_EQ(blockByTag(evictionByKey(evicted, 2).blocks_by_group, "group3"), 302);
    ASSERT_EQ(evictionByKey(evicted, 2).kind, EvictionKind::IndependentGroup);
    EXPECT_EQ(evictionByKey(evicted, 2).group_tag, "group3");
    EXPECT_TRUE(isNullBlockIdx(cache.matchGroup(2, 3)));
    EXPECT_EQ(cache.matchGroup(2, 0), 102);
    EXPECT_EQ(cache.matchGroup(3, 3), 303);
}

TEST(SharedBlockCachePerfTest, DISABLED_FlatFallbackLargeLru) {
    constexpr int kItemCount    = 20000;
    constexpr int kTargetStride = 5;
    constexpr int kEvictCount   = 2000;

    PositionalSharedBlockCacheForTest cache;
    cache.setPrefixTreeEnabled(false);
    for (int i = 0; i < kItemCount; ++i) {
        const auto key         = static_cast<CacheKeyType>(i + 1);
        const auto target_slot = i % kTargetStride == 0 ? static_cast<BlockIdxType>(i + 100001) : NULL_BLOCK_IDX;
        cache.put(key,
                  std::vector<BlockIdxType>{static_cast<BlockIdxType>(i + 1), target_slot},
                  /*is_resident=*/false,
                  SharedBlockCache::kGpuLogicalNamespace,
                  rootDep());
    }

    const auto start   = std::chrono::steady_clock::now();
    const auto evicted = cache.selectAndEvictForGroup(/*group_slot=*/1, kEvictCount);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);

    EXPECT_EQ(evictionKeys(evicted).size(), kEvictCount);
    std::cout << "[ PERF ] prefix_tree=off items=" << kItemCount << " evicted=" << evictionKeys(evicted).size()
              << " selection_us=" << elapsed.count() << std::endl;
}

TEST(SharedBlockCachePerfTest, DISABLED_PrefixTreeLongSessionChains) {
    constexpr int kFamilyCount = 16;
    constexpr int kChainDepth  = 512;

    PositionalSharedBlockCacheForTest cache;
    for (int family = 0; family < kFamilyCount; ++family) {
        CacheKeyType parent_key = 0;
        for (int depth = 0; depth < kChainDepth; ++depth) {
            const auto key         = static_cast<CacheKeyType>(family * kChainDepth + depth + 1);
            const bool target_leaf = family == kFamilyCount - 1 && depth == kChainDepth - 1;
            cache.put(key,
                      std::vector<BlockIdxType>{static_cast<BlockIdxType>(key + 10000),
                                                target_leaf ? static_cast<BlockIdxType>(key + 20000) : NULL_BLOCK_IDX},
                      /*is_resident=*/false,
                      SharedBlockCache::kGpuLogicalNamespace,
                      depth == 0 ? rootDep() : childDep(parent_key, static_cast<uint32_t>(depth)));
            parent_key = key;
        }
    }

    const auto start   = std::chrono::steady_clock::now();
    const auto evicted = cache.selectAndEvictForGroup(/*group_slot=*/1, /*min_blocks=*/1);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);

    EXPECT_EQ(evictionKeys(evicted).size(), kChainDepth);
    std::cout << "[ PERF ] prefix_tree=on items=" << kFamilyCount * kChainDepth << " chains=" << kFamilyCount
              << " depth=" << kChainDepth << " evicted=" << evictionKeys(evicted).size()
              << " selection_us=" << elapsed.count() << std::endl;
}

}  // namespace rtp_llm::test
