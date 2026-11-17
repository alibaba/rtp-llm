#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"

#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

// Build the device pool config from the lightweight cache-config test helpers
// (cache_config_test_utils + DeviceBlockPoolConfigHelper) rather than BlockPoolTestHelper,
// which drags in the heavy //rtp_llm/cpp/cache monolith (and its CUDA-13 prebuilt
// remote_kv_cache_manager dependency that does not link in the CUDA-12 toolchain).
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {
namespace {

class CountingMemoryUtil: public MemoryUtil {
public:
    bool regUserMr(void* address, uint64_t bytes, bool gpu, uint64_t alignment) override {
        ++registrations;
        last_address   = address;
        last_bytes     = bytes;
        last_gpu       = gpu;
        last_alignment = alignment;
        return register_ok;
    }
    bool deregUserMr(void* address, bool gpu) override {
        ++deregistrations;
        EXPECT_EQ(address, last_address);
        EXPECT_EQ(gpu, last_gpu);
        return true;
    }
    bool isMemoryMr(void*, uint64_t, bool, bool) override {
        return false;
    }
    bool findMemoryMr(void*, void*, uint64_t, bool, bool) override {
        return false;
    }
    bool isRdmaMode() override {
        return false;
    }
    size_t   registrations{0}, deregistrations{0};
    void*    last_address{nullptr};
    uint64_t last_bytes{0}, last_alignment{0};
    bool     last_gpu{false}, register_ok{true};
};

static_assert(std::is_same_v<decltype(DeviceBlockPoolConfigHelper::createConfigForGroup(
                                 std::declval<const CacheConfig&>(), std::declval<const GroupBase&>())),
                             DeviceBlockPoolConfig>);

std::shared_ptr<DeviceBlockPoolConfig> makeConfig() {
    constexpr int        kLayerNum       = 4;
    constexpr int        kBlockNum       = 10;
    constexpr size_t     kTokensPerBlock = 1;
    rtp_llm::CacheConfig cache_config    = rtp_llm::test::makeSimpleMhaCacheConfig(kLayerNum,
                                                                                kBlockNum,
                                                                                kTokensPerBlock,
                                                                                rtp_llm::TYPE_FP16,
                                                                                /*local_head_num_kv=*/1,
                                                                                /*size_per_head=*/64);
    cache_config.finalizeBlockNums(kBlockNum, rtp_llm::RuntimeConfig{});
    auto config = std::make_shared<DeviceBlockPoolConfig>(
        DeviceBlockPoolConfigHelper::createConfigForGroup(cache_config, cache_config.topology().groups().front()));
    config->pool_name                 = "device";
    config->use_device_malloc_backing = true;
    return config;
}

std::shared_ptr<DeviceBlockPoolConfig> makeMixedScaleConfig() {
    constexpr int    kBlockNum       = 8;
    constexpr size_t kTokensPerBlock = 1;

    rtp_llm::CacheConfig scaled_cfg =
        rtp_llm::test::makeSimpleMhaCacheConfig(2, kBlockNum, kTokensPerBlock, rtp_llm::TYPE_INT8, 1, 64);
    rtp_llm::CacheConfig plain_cfg =
        rtp_llm::test::makeSimpleMhaCacheConfig(3, kBlockNum, kTokensPerBlock, rtp_llm::TYPE_FP16, 1, 64);
    scaled_cfg.finalizeBlockNums(kBlockNum, rtp_llm::RuntimeConfig{});
    plain_cfg.finalizeBlockNums(kBlockNum, rtp_llm::RuntimeConfig{});

    DeviceBlockPoolConfig scaled_pool =
        DeviceBlockPoolConfigHelper::createConfigForGroup(scaled_cfg, scaled_cfg.topology().groups().front());
    DeviceBlockPoolConfig plain_pool =
        DeviceBlockPoolConfigHelper::createConfigForGroup(plain_cfg, plain_cfg.topology().groups().front());

    MemoryLayoutConfig l0    = scaled_pool.memory_layouts[0];
    MemoryLayoutConfig l1    = plain_pool.memory_layouts[0];
    l1.kv_cache_offset_bytes = l0.total_size_bytes + l1.kv_cache_offset_bytes;
    l1.kv_scale_offset_bytes = l0.total_size_bytes + l1.kv_scale_offset_bytes;

    auto config                       = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type                 = BlockPoolType::DEVICE;
    config->pool_name                 = "mixed_scale_device";
    config->physical_block_count      = l0.block_num;
    config->total_size_bytes          = l0.total_size_bytes + l1.total_size_bytes;
    config->memory_layouts            = {l0, l1};
    config->use_device_malloc_backing = true;
    return config;
}

CacheConfig makeMtpCacheConfig() {
    CacheConfig main = rtp_llm::test::makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/3, /*seq_size_per_block=*/1, TYPE_FP16, /*local_head_num_kv=*/1, 64);
    for (int module = 0; module < 2; ++module) {
        const auto draft = std::make_shared<CacheConfig>(rtp_llm::test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/4 + module, /*seq_size_per_block=*/1, TYPE_FP16, 1, 64));
        main.mtp_sub_configs.push_back(main.mergeMTPModule(*draft, module, main.layer_num));
    }
    main.finalizeBlockNums(/*baseline_block_num=*/3, rtp_llm::RuntimeConfig{});
    return main;
}

}  // namespace

TEST(DeviceBlockPoolTest, InitKeepsBlockZeroInvalid) {
    auto config                       = makeConfig();
    config->use_device_malloc_backing = false;
    DeviceBlockPool pool(config);

    ASSERT_TRUE(pool.init());
    EXPECT_FALSE(pool.isAllocated(0));
    EXPECT_FALSE(pool.validBlock(0));
    EXPECT_EQ(pool.totalBlocksNum(), config->physical_block_count - 1);

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_NE(*block, 0);
}

TEST(DeviceBlockPoolTest, ConstructorRejectsInvalidConfigMatrix) {
    std::shared_ptr<const DeviceBlockPoolConfig> null_config;
    EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(null_config)));

    {
        auto config       = makeConfig();
        config->pool_type = BlockPoolType::HOST;
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
    {
        auto config = makeConfig();
        config->memory_layouts.clear();
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
    {
        auto config = makeConfig();
        config->memory_layouts.push_back(config->memory_layouts.front());
        --config->memory_layouts.back().block_num;
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
    {
        auto config                              = makeConfig();
        config->memory_layouts.front().layer_num = 0;
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
    {
        auto config                                             = makeConfig();
        config->memory_layouts.front().kv_block_pool_size_bytes = 0;
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
    {
        auto config = makeConfig();
        ++config->physical_block_count;
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
    {
        auto config                                             = makeConfig();
        config->physical_block_count                            = 1;
        config->memory_layouts.front().block_num                = 1;
        config->memory_layouts.front().total_size_bytes         = config->memory_layouts.front().kv_block_stride_bytes;
        config->memory_layouts.front().kv_block_pool_size_bytes = config->memory_layouts.front().kv_block_stride_bytes;
        EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(config)));
    }
}

TEST(DeviceBlockPoolTest, MultiLayoutMtpConfigUsesMainBlockCountAndGlobalLayerMapping) {
    CacheConfig cache_config = makeMtpCacheConfig();
    ASSERT_EQ(cache_config.group("default").block_num, 3u);
    ASSERT_EQ(cache_config.mtp_sub_configs.size(), 2u);
    ASSERT_EQ(cache_config.mtp_sub_configs[0]->group("default").block_num, 3u);
    ASSERT_EQ(cache_config.mtp_sub_configs[1]->group("default").block_num, 3u);

    auto config = std::make_shared<DeviceBlockPoolConfig>(
        DeviceBlockPoolConfigHelper::createConfigForGroup(cache_config, cache_config.topology().groups().front()));
    config->use_device_malloc_backing = true;
    ASSERT_EQ(config->physical_block_count, 3u);
    ASSERT_EQ(config->memory_layouts.size(), 3u);
    EXPECT_EQ(config->memory_layouts[0].layer_num, 2u);
    EXPECT_EQ(config->memory_layouts[1].layer_num, 1u);
    EXPECT_EQ(config->memory_layouts[2].layer_num, 1u);
    for (const auto& layout : config->memory_layouts) {
        EXPECT_EQ(layout.block_num, 3u);
    }

    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());
    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());

    const auto main_addr = pool.convertIndexToAddr(/*global_layer_id=*/0, *block);
    const auto mtp1_addr = pool.convertIndexToAddr(/*global_layer_id=*/2, *block);
    const auto mtp2_addr = pool.convertIndexToAddr(/*global_layer_id=*/3, *block);
    ASSERT_NE(main_addr.kv_addr, nullptr);
    ASSERT_NE(mtp1_addr.kv_addr, nullptr);
    ASSERT_NE(mtp2_addr.kv_addr, nullptr);
    EXPECT_NE(main_addr.kv_addr, mtp1_addr.kv_addr);
    EXPECT_NE(mtp1_addr.kv_addr, mtp2_addr.kv_addr);
}

TEST(DeviceBlockPoolTest, ConvertIndexToBufferReturnsBlockInfo) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_NE(*block, 0);

    auto buffers = pool.convertIndexToBuffer(0, *block);
    ASSERT_FALSE(buffers.empty());
    for (const auto& buffer : buffers) {
        EXPECT_NE(buffer.addr, nullptr);
        EXPECT_GT(buffer.size_bytes, 0u);
    }
}

TEST(DeviceBlockPoolTest, PartitionedConvertIndexToBufferReturnsBlockInfo) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());

    auto buffers = pool.convertIndexToBuffer(0, *block, 1, 0);
    ASSERT_FALSE(buffers.empty());
    for (const auto& buffer : buffers) {
        EXPECT_NE(buffer.addr, nullptr);
        EXPECT_GT(buffer.size_bytes, 0u);
    }
}

TEST(DeviceBlockPoolTest, InvalidLayerAndBlockDoNotCorruptPoolState) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    pool.incRef(*block);
    const size_t free_before = pool.freeBlocksNum();
    const auto   refs_before = pool.refCount(*block);

    EXPECT_ANY_THROW(pool.convertIndexToBuffer(static_cast<int>(config->memory_layouts.front().layer_num), *block));
    EXPECT_ANY_THROW(pool.convertIndexToAddr(/*layer_id=*/0, static_cast<BlockIdxType>(config->physical_block_count)));

    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), refs_before);
    EXPECT_EQ(pool.freeBlocksNum(), free_before);
    pool.decRef(*block);
}

TEST(DeviceBlockPoolTest, LifecycleStartsAllocatedBlockWithZeroRefCount) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 0u);

    pool.incRef(*block);
    pool.decRef(*block);
    EXPECT_FALSE(pool.isAllocated(*block));
}

TEST(DeviceBlockPoolTest, LifecycleUsesIBlockPoolSemantics) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 0u);

    pool.incRef(*block);
    EXPECT_EQ(pool.refCount(*block), 1u);
    pool.decRef(*block);
    EXPECT_FALSE(pool.isAllocated(*block));
}

TEST(DeviceBlockPoolTest, RequestAndCacheRefsUpdateAvailableIndependently) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    const size_t total_blocks = pool.totalBlocksNum();
    EXPECT_EQ(pool.availableBlocksNum(), total_blocks - 1);

    pool.incTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_EQ(pool.availableBlocksNum(), total_blocks);

    pool.incRef(*block);
    EXPECT_EQ(pool.refCount(*block), 2u);
    EXPECT_EQ(pool.treeRefCount(*block), 1u);
    EXPECT_EQ(pool.referencedBlocksNum(), 1u);
    EXPECT_EQ(pool.availableBlocksNum(), total_blocks - 1);

    pool.decRef(*block);
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 1u);
    EXPECT_EQ(pool.availableBlocksNum(), total_blocks);

    pool.decTreeRef(*block, BlockTreeRefType::CACHE);
    EXPECT_FALSE(pool.isAllocated(*block));
    EXPECT_EQ(pool.availableBlocksNum(), total_blocks);
}

TEST(DeviceBlockPoolTest, ActiveBlocksIncludeRequestAndTreeOperationRefs) {
    std::shared_ptr<const DeviceBlockPoolConfig> config = makeConfig();
    DeviceBlockPool                              pool(config);
    ASSERT_TRUE(pool.init());

    std::optional<BlockIdxType> block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    size_t active_blocks = 0;

    pool.incTreeRef(*block, BlockTreeRefType::CACHE);
    active_blocks = pool.activeBlocksNum();
    EXPECT_EQ(active_blocks, 0u);

    pool.incRef(*block);
    pool.incTreeRef(*block, BlockTreeRefType::LOAD);
    active_blocks = pool.activeBlocksNum();
    EXPECT_EQ(active_blocks, 1u);

    pool.decRef(*block);
    active_blocks = pool.activeBlocksNum();
    EXPECT_EQ(active_blocks, 1u);

    pool.decTreeRef(*block, BlockTreeRefType::LOAD);
    active_blocks = pool.activeBlocksNum();
    EXPECT_EQ(active_blocks, 0u);

    pool.decTreeRef(*block, BlockTreeRefType::CACHE);
}

TEST(DeviceBlockPoolTest, ExplicitReleaseCannotConsumeTreeUmbrella) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    pool.incTreeRef(*block, BlockTreeRefType::LOAD);

    EXPECT_ANY_THROW(pool.decRef(*block));

    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 1u);
    EXPECT_EQ(pool.treeRefCount(*block), 1u);
    pool.decTreeRef(*block, BlockTreeRefType::LOAD);
}

TEST(DeviceBlockPoolTest, BatchIncRefRejectsInvalidTailWithoutMutatingPrefix) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    const BlockIdxType unallocated_block = *block + 1;

    EXPECT_ANY_THROW(pool.incRef(BlockIdList{*block, unallocated_block}));

    EXPECT_EQ(pool.refCount(*block), 0u);
    EXPECT_EQ(pool.referencedBlocksNum(), 0u);
    pool.incRef(*block);
    pool.decRef(*block);
}

TEST(DeviceBlockPoolTest, BatchDecRefRejectsUnheldTailWithoutMutatingPrefix) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto blocks = pool.malloc(2);
    ASSERT_TRUE(blocks.has_value());
    pool.incRef(blocks->front());

    EXPECT_ANY_THROW(pool.decRef(*blocks));

    EXPECT_EQ(pool.refCount(blocks->front()), 1u);
    EXPECT_EQ(pool.referencedBlocksNum(), 1u);
    pool.decRef(blocks->front());
    pool.incRef(blocks->back());
    pool.decRef(blocks->back());
}

TEST(DeviceBlockPoolTest, ExposesAllocatorFacingLayerTensorsAndBuffers) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto cache_tensors = pool.allLayerCacheBase();
    ASSERT_FALSE(cache_tensors.empty());
    EXPECT_TRUE(cache_tensors[0].defined());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    auto addr = pool.convertIndexToAddr(/*layer_id=*/0, *block);
    EXPECT_NE(addr.kv_addr, nullptr);

    auto infos = pool.convertIndexToBuffer(/*layer_id=*/0, *block);
    ASSERT_FALSE(infos.empty());
    EXPECT_NE(infos[0].addr, nullptr);
    EXPECT_GT(infos[0].size_bytes, 0u);
    EXPECT_TRUE(infos[0].is_cuda);
}

TEST(DeviceBlockPoolTest, RegUserMrWithoutCacheStoreIsNoOp) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    // No cache store wired: MR registration must be a no-op, not a crash.
    pool.regUserMr(/*model_id=*/0, /*cache_store=*/nullptr);
    EXPECT_EQ(pool.getMrCostTimeMs(), 0);

    // Idempotent / safe to call again and to deregister when nothing was registered.
    pool.regUserMr(/*model_id=*/0, /*cache_store=*/nullptr);
    pool.deregUserMr();
    EXPECT_EQ(pool.getMrCostTimeMs(), 0);
}

TEST(DeviceBlockPoolTest, AllLayerScaleCacheBaseStaysAlignedWithPartialScale) {
    auto            config = makeMixedScaleConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    const auto kv_bases    = pool.allLayerCacheBase();
    const auto scale_bases = pool.allLayerScaleCacheBase();

    ASSERT_EQ(kv_bases.size(), 5u);
    ASSERT_EQ(scale_bases.size(), 5u);

    for (const auto& kv : kv_bases) {
        EXPECT_TRUE(kv.defined());
        EXPECT_GT(kv.numel(), 0);
    }

    EXPECT_TRUE(scale_bases[0].defined());
    EXPECT_GT(scale_bases[0].numel(), 0);
    EXPECT_TRUE(scale_bases[1].defined());
    EXPECT_GT(scale_bases[1].numel(), 0);
    EXPECT_FALSE(scale_bases[2].defined() && scale_bases[2].numel() > 0);
    EXPECT_FALSE(scale_bases[3].defined() && scale_bases[3].numel() > 0);
    EXPECT_FALSE(scale_bases[4].defined() && scale_bases[4].numel() > 0);
}

TEST(DeviceBlockPoolTest, MallocAllocatesLowestBlockFirst) {
    auto            config = makeConfig();
    DeviceBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto b1 = pool.malloc();
    auto b2 = pool.malloc();
    auto b3 = pool.malloc();
    ASSERT_TRUE(b1.has_value() && b2.has_value() && b3.has_value());
    EXPECT_EQ(*b1, 1);
    EXPECT_EQ(*b2, 2);
    EXPECT_EQ(*b3, 3);
}

TEST(DeviceBlockPoolTest, SharedBackingPreservesMainMtpAndScaleViewsAcrossPoolInitialization) {
    auto mtp   = makeMtpCacheConfig();
    auto first = std::make_shared<DeviceBlockPoolConfig>(
        DeviceBlockPoolConfigHelper::createConfigForGroup(mtp, mtp.topology().groups().front()));
    auto                      second  = makeMixedScaleConfig();
    const std::vector<size_t> strides = {DeviceBlockPoolConfigHelper::sharedSlotStride(*first),
                                         DeviceBlockPoolConfigHelper::sharedSlotStride(*second)};
    auto arena = std::make_shared<SharedBlockArena>(16 * std::max(strides[0], strides[1]), strides);
    DeviceBlockPoolConfigHelper::bindSharedArena(*first, arena, 0);
    auto first_pool = std::make_unique<DeviceBlockPool>(first);
    ASSERT_TRUE(first_pool->init());
    auto block = first_pool->malloc();
    ASSERT_TRUE(block.has_value());
    first_pool->incRef(*block);
    const auto first_views = first_pool->allLayerCacheBase();
    ASSERT_EQ(first_views.size(), 4u);  // main two layers plus two MTP modules.
    for (size_t layer = 0; layer < first_views.size(); ++layer) {
        first_views[layer][*block].fill_(10 + layer);
    }
    DeviceBlockPoolConfigHelper::bindSharedArena(*second, arena, 1, first_pool->sharedBacking());
    auto second_pool = std::make_unique<DeviceBlockPool>(second);
    ASSERT_TRUE(second_pool->init());
    EXPECT_EQ(first_pool->getBaseAddress(), second_pool->getBaseAddress());
    EXPECT_EQ(first_pool->sharedBacking(), second_pool->sharedBacking());
    EXPECT_EQ(first_pool->getTotalSizeBytes(), arena->bytes());
    EXPECT_EQ(second_pool->getTotalSizeBytes(), arena->bytes());
    for (size_t layer = 0; layer < first_views.size(); ++layer) {
        EXPECT_TRUE(first_views[layer][*block].eq(10 + layer).all().item<bool>());
    }
    auto other = second_pool->malloc();
    ASSERT_TRUE(other.has_value());
    second_pool->incRef(*other);
    const size_t begin0 = *block * strides[0], begin1 = *other * strides[1];
    EXPECT_TRUE(begin0 + strides[0] <= begin1 || begin1 + strides[1] <= begin0);
    const auto second_views  = second_pool->allLayerCacheBase();
    const auto second_scales = second_pool->allLayerScaleCacheBase();
    second_views[0][*other].fill_(19);
    second_scales[0][*other].fill_(23);
    EXPECT_TRUE(second_views[0][*other].eq(19).all().item<bool>());
    EXPECT_TRUE(second_scales[0][*other].eq(23).all().item<bool>());
    for (size_t layer = 0; layer < first_views.size(); ++layer) {
        EXPECT_TRUE(first_views[layer][*block].eq(10 + layer).all().item<bool>());
        const auto buffer = first_pool->convertIndexToBuffer(static_cast<int>(layer), *block).front();
        EXPECT_EQ(buffer.addr, first_views[layer][*block].data_ptr());
    }
    first_pool->decRef(*block);
    second_pool->decRef(*other);
    EXPECT_EQ(arena->freeBytes(), arena->usableBytes());
    first_pool.reset();
    // The second pool and model views retain the common Storage.
    EXPECT_TRUE(second_views[0][*other].eq(19).all().item<bool>());
}

TEST(DeviceBlockPoolTest, SharedBackingRegistersWholeBudgetOnceAndRollsBackRegistrationFailure) {
    auto arena          = std::make_shared<SharedBlockArena>(4096, std::vector<size_t>{512, 1024});
    auto bytes          = torch::zeros({4096}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    auto backing        = std::make_shared<SharedDeviceBlockBacking>(bytes, arena);
    auto memory         = std::make_shared<CountingMemoryUtil>();
    memory->register_ok = false;
    EXPECT_ANY_THROW(backing->retainMemoryRegistration(memory));
    memory->register_ok = true;
    backing->retainMemoryRegistration(memory);
    backing->retainMemoryRegistration(memory);
    EXPECT_EQ(memory->registrations, 2u);  // one failed attempt, one actual registration.
    EXPECT_EQ(memory->last_address, bytes.data_ptr());
    EXPECT_EQ(memory->last_bytes, 4096u);
    EXPECT_TRUE(memory->last_gpu);
    EXPECT_EQ(memory->last_alignment, 0u);
    auto different_provider = std::make_shared<CountingMemoryUtil>();
    EXPECT_ANY_THROW(backing->retainMemoryRegistration(different_provider));
    backing->releaseMemoryRegistration();
    EXPECT_EQ(memory->deregistrations, 0u);
    backing->releaseMemoryRegistration();
    EXPECT_EQ(memory->deregistrations, 1u);
    EXPECT_ANY_THROW(backing->releaseMemoryRegistration());
    backing->retainMemoryRegistration(memory);
    backing.reset();  // exception cleanup still releases the one registration.
    EXPECT_EQ(memory->deregistrations, 2u);
}

TEST(DeviceBlockPoolTest, SharedBackingRejectsIndependentAllocatorAndWrongPitch) {
    auto       first  = makeConfig();
    const auto stride = DeviceBlockPoolConfigHelper::sharedSlotStride(*first);
    auto       arena  = std::make_shared<SharedBlockArena>(16 * stride, std::vector<size_t>{stride});
    DeviceBlockPoolConfigHelper::bindSharedArena(*first, arena, 0);
    DeviceBlockPool pool(first);
    ASSERT_TRUE(pool.init());
    auto second      = makeConfig();
    auto independent = std::make_shared<SharedBlockArena>(16 * stride, std::vector<size_t>{stride});
    DeviceBlockPoolConfigHelper::bindSharedArena(*second, independent, 0, pool.sharedBacking());
    DeviceBlockPool unsafe(second);
    EXPECT_ANY_THROW(unsafe.init());
    first->memory_layouts.front().block_slot_stride_bytes += 256;
    EXPECT_ANY_THROW(static_cast<void>(DeviceBlockPool(first)));
}

}  // namespace rtp_llm
