#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "autil/EnvUtil.h"

// Build the device pool config from the lightweight cache-config test helpers
// (cache_config_test_utils + DeviceBlockPoolConfigHelper) rather than BlockPoolTestHelper,
// which drags in the heavy //rtp_llm/cpp/cache monolith (and its CUDA-13 prebuilt
// remote_kv_cache_manager dependency that does not link in the CUDA-12 toolchain).
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace {

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

std::shared_ptr<DeviceBlockPoolConfig> makeOffloadConfig(bool padded = false, uint32_t total = 6, uint32_t hbm = 2) {
    KVCacheSpecPtr spec;
    if (padded) {
        KVCacheSpecDesc desc;
        desc.tag                          = "global_kv_1";
        desc.cache_type                   = OpaqueKV;
        desc.entry_elems                  = 288;
        desc.entry_dtype                  = TYPE_UINT8;
        desc.entry_count_mode             = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
        desc.compression_ratio            = 8;
        desc.block_stride_bytes_alignment = 8192;
        SpecBuildContext ctx;
        ctx.seq_size_per_block      = 256;
        ctx.kernel_tokens_per_block = 128;
        spec                        = SpecBuilder::build(desc, ctx);
    } else {
        spec                            = test::makeResolvedMlaSpec(TYPE_BF16, 512, 64, 256, "default");
        spec->kernel_seq_size_per_block = 128;
    }
    auto cache    = test::makeSingleGroupCacheConfig(spec, CacheGroupType::FULL, 2, total);
    cache.use_mla = true;
    cache.tiered_cache_configs.emplace(spec->tag, TieredCacheConfig{total, hbm, padded ? 17u : 129u});
    return std::make_shared<DeviceBlockPoolConfig>(
        DeviceBlockPoolConfigHelper::createConfigForGroup(cache, cache.group(spec->tag)));
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

TEST(DeviceBlockPoolTest, OffloadLengthAwareAdmissionAndGrowth) {
    autil::EnvGuard guard("RTP_LLM_SPARSE_KV_OFFLOAD_HBM_SHARE_DENOMINATOR", "3");
    DeviceBlockPool pool(makeOffloadConfig(false, 30, 10));
    ASSERT_TRUE(pool.init());
    for (int round = 0; round < 3; ++round) {
        // A long request arriving first leaves all nine usable HBM blocks free.
        auto long_request = pool.mallocForRequest(12, 12);
        ASSERT_TRUE(long_request.has_value());
        EXPECT_TRUE(std::all_of(long_request->begin(), long_request->end(), [](int b) { return b >= 10; }));
        pool.incRef(*long_request);
        auto short_request = pool.mallocForRequest(2, 2);
        ASSERT_TRUE(short_request.has_value());
        EXPECT_TRUE(std::all_of(short_request->begin(), short_request->end(), [](int b) { return b < 10; }));
        pool.incRef(*short_request);

        auto host_growth = pool.mallocForRequest(1, 13, long_request->back());
        ASSERT_TRUE(host_growth.has_value());
        EXPECT_GE(host_growth->front(), 10);
        pool.incRef(*host_growth);
        auto hbm_growth = pool.mallocForRequest(1, 3, short_request->back());
        ASSERT_TRUE(hbm_growth.has_value());
        EXPECT_LT(hbm_growth->front(), 10);
        pool.incRef(*hbm_growth);

        pool.decRef(*long_request);
        pool.decRef(*short_request);
        pool.decRef(*host_growth);
        pool.decRef(*hbm_growth);
        EXPECT_EQ(pool.freeBlocksNum(), 29u);
        EXPECT_EQ(pool.activeBlocksNum(), 0u);
    }
}

TEST(DeviceBlockPoolTest, OffloadAdmissionUsesFreeHbmAndInclusiveBoundary) {
    autil::EnvGuard guard("RTP_LLM_SPARSE_KV_OFFLOAD_HBM_SHARE_DENOMINATOR", "3");
    DeviceBlockPool pool(makeOffloadConfig(false, 20, 10));
    ASSERT_TRUE(pool.init());
    auto first = pool.mallocForRequest(3, 3);  // Exactly nine / three fits HBM.
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(*first, (BlockIdList{1, 2, 3}));
    pool.incRef(*first);
    auto next = pool.mallocForRequest(3, 3);  // Now only six HBM blocks are free.
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(*next, (BlockIdList{10, 11, 12}));
    pool.incRef(*next);

    // CACHE and in-flight LOAD holds must not inflate free HBM on request release.
    pool.incTreeRef(*first, BlockTreeRefType::CACHE);
    pool.incTreeRef(*first, BlockTreeRefType::LOAD);
    pool.decRef(*first);
    pool.decTreeRef(*first, BlockTreeRefType::CACHE);
    auto held = pool.mallocForRequest(3, 3);
    ASSERT_TRUE(held.has_value());
    EXPECT_GE(held->front(), 10);
    pool.incRef(*held);
    pool.decTreeRef(*first, BlockTreeRefType::LOAD);
    auto restored = pool.mallocForRequest(3, 3);
    ASSERT_TRUE(restored.has_value());
    EXPECT_TRUE(std::all_of(restored->begin(), restored->end(), [](int b) { return b < 10; }));
    pool.incRef(*restored);
    pool.decRef(*next);
    pool.decRef(*held);
    pool.decRef(*restored);
    EXPECT_EQ(pool.freeBlocksNum(), 19u);
}

TEST(DeviceBlockPoolTest, OffloadMixedLengthRoutingProtectsShortRequestsAcrossArrivalOrders) {
    // Scale 48 short + 16 long requests to one/twelve blocks per request.
    for (const char* denominator : {"0", "3"}) {
        autil::EnvGuard guard("RTP_LLM_SPARSE_KV_OFFLOAD_HBM_SHARE_DENOMINATOR", denominator);
        DeviceBlockPool pool(makeOffloadConfig(false, 241, 49));
        ASSERT_TRUE(pool.init());
        for (int round = 0; round < 3; ++round) {
            std::vector<BlockIdList> requests;
            int                      host_queries = 0;
            for (int i = 0; i < 64; ++i) {
                const bool   is_long = round == 1 ? i >= 48 : i < 16;
                const size_t count   = is_long ? 12 : 1;
                auto         blocks  = pool.mallocForRequest(count, count);
                ASSERT_TRUE(blocks.has_value());
                host_queries += std::any_of(blocks->begin(), blocks->end(), [](int block) { return block >= 49; });
                pool.incRef(*blocks);
                requests.push_back(std::move(*blocks));
            }
            EXPECT_EQ(host_queries, std::string(denominator) == "0" ? (round == 1 ? 16 : 60) : (round == 1 ? 18 : 38));
            EXPECT_EQ(pool.freeBlocksNum(), 0u);
            for (const auto& blocks : requests) {
                pool.decRef(blocks);
            }
            EXPECT_EQ(pool.freeBlocksNum(), 240u);
        }
    }
}

TEST(DeviceBlockPoolTest, OffloadAdmissionFallsBackAtomicallyAndReusesFreedHbm) {
    autil::EnvGuard guard("RTP_LLM_SPARSE_KV_OFFLOAD_HBM_SHARE_DENOMINATOR", "3");
    DeviceBlockPool pool(makeOffloadConfig(false, 20, 10));
    ASSERT_TRUE(pool.init());
    auto large = pool.mallocForRequest(12, 12);
    ASSERT_TRUE(large.has_value());
    EXPECT_EQ(*large, (BlockIdList{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 2}));
    pool.incRef(*large);
    (void)pool.takeOffloadVersionUpdates();
    EXPECT_FALSE(pool.mallocForRequest(8, 20).has_value());
    EXPECT_EQ(pool.freeBlocksNum(), 7u);
    EXPECT_TRUE(pool.takeOffloadVersionUpdates().empty());

    auto rest = pool.mallocForRequest(7, 7, 2);
    ASSERT_TRUE(rest.has_value());
    EXPECT_EQ(*rest, (BlockIdList{3, 4, 5, 6, 7, 8, 9}));
    pool.incRef(*rest);
    pool.decRef(1);
    auto recycled = pool.mallocForRequest(1, 8, 9);
    ASSERT_TRUE(recycled.has_value());
    EXPECT_EQ(*recycled, (BlockIdList{1}));
    pool.incRef(*recycled);
    EXPECT_EQ(pool.takeOffloadVersionUpdates().size(), 8u);
    pool.decRef(*recycled);
    pool.decRef(*rest);
    large->erase(std::find(large->begin(), large->end(), 1));
    pool.decRef(*large);
    EXPECT_EQ(pool.freeBlocksNum(), 19u);
}

TEST(DeviceBlockPoolTest, OffloadLengthAwareRoutingCanBeDisabledAndValidatesSetting) {
    {
        autil::EnvGuard guard("RTP_LLM_SPARSE_KV_OFFLOAD_HBM_SHARE_DENOMINATOR", "0");
        DeviceBlockPool pool(makeOffloadConfig(false, 20, 10));
        ASSERT_TRUE(pool.init());
        EXPECT_EQ(pool.mallocForRequest(4, 4).value(), (BlockIdList{1, 2, 3, 4}));
    }
    for (const char* value : {"", "1", "-1", "abc", "3x", "18446744073709551616"}) {
        autil::EnvGuard guard("RTP_LLM_SPARSE_KV_OFFLOAD_HBM_SHARE_DENOMINATOR", value);
        EXPECT_ANY_THROW(DeviceBlockPool{makeOffloadConfig(false, 20, 10)});
        // A configuration for a regular HBM-only pool does not consult this knob.
        EXPECT_NO_THROW(DeviceBlockPool{makeConfig()});
    }
}

TEST(DeviceBlockPoolTest, OffloadNativePlanesAndSegmentBoundaries) {
    for (const bool raw_device : {false, true}) {
        auto config                       = makeOffloadConfig();
        config->use_device_malloc_backing = raw_device;
        DeviceBlockPool pool(config);
        ASSERT_TRUE(pool.init());
        const auto& storages = pool.allLayerOffloadStorage();
        ASSERT_EQ(storages.size(), 2u);
        EXPECT_EQ(storages[0]->generations.data_ptr(), storages[1]->generations.data_ptr());
        EXPECT_EQ(storages[0]->generations.scalar_type(), torch::kInt64);
        EXPECT_EQ(storages[0]->generations.numel(), 6);
        EXPECT_FALSE(pool.validBlock(0));
        for (int owner = 0; owner < 2; ++owner) {
            const auto& storage = *storages[owner];
            EXPECT_EQ(storage.owner_layer, owner);
            EXPECT_EQ(storage.rows_per_block, 256u);
            EXPECT_EQ(storage.rows_per_page, 128u);
            const auto& gpu  = storage.resident_planes[0];
            const auto& host = storage.host_planes[0];
            EXPECT_EQ(gpu.sizes().vec(), (std::vector<int64_t>{6, 128, 576}));
            EXPECT_EQ(host.sizes().vec(), (std::vector<int64_t>{8, 128, 576}));
            EXPECT_EQ(gpu.scalar_type(), torch::kBFloat16);
            EXPECT_TRUE(gpu.is_cuda());
            EXPECT_FALSE(host.is_cuda());
            EXPECT_TRUE(host.is_pinned());
            EXPECT_EQ(gpu.narrow(0, 0, 2).count_nonzero().item<int64_t>(), 0);
            const size_t block_bytes = 256 * 576 * 2;
            for (int block = 0; block < 6; ++block) {
                const auto buffers = pool.convertIndexToBuffer(owner, block);
                ASSERT_EQ(buffers.size(), 1u);
                const auto& actual = buffers[0];
                EXPECT_EQ(actual.is_cuda, block < 2);
                const auto&  base  = block < 2 ? gpu : host;
                const size_t local = block < 2 ? block : block - 2;
                EXPECT_EQ(actual.addr, static_cast<char*>(base.data_ptr()) + local * block_bytes);
                EXPECT_EQ(actual.size_bytes, block_bytes);
            }
        }
        EXPECT_EQ(pool.blockSizeBytes(), 2u * 256 * 576 * 2);
        EXPECT_GT(pool.getTotalSizeBytes(), config->total_size_bytes);
        EXPECT_ANY_THROW(pool.allLayerCacheBase());
    }
}

TEST(DeviceBlockPoolTest, OffloadPreservesPagePaddingAndHostLocalOffsets) {
    DeviceBlockPool pool(makeOffloadConfig(true));
    ASSERT_TRUE(pool.init());
    const auto& storage = *pool.allLayerOffloadStorage()[0];
    EXPECT_EQ(storage.rows_per_block, 32u);
    EXPECT_EQ(storage.rows_per_page, 16u);
    const auto& host = storage.host_planes[0];
    EXPECT_EQ(host.sizes().vec(), (std::vector<int64_t>{8, 16, 288}));
    EXPECT_EQ(host.strides().vec(), (std::vector<int64_t>{8192, 288, 1}));
    EXPECT_EQ(host.scalar_type(), torch::kUInt8);
    EXPECT_EQ(pool.convertIndexToBuffer(0, 2)[0].addr, host.data_ptr());
    EXPECT_EQ(pool.convertIndexToBuffer(0, 3)[0].addr, static_cast<char*>(host.data_ptr()) + 16384);
    EXPECT_EQ(pool.convertIndexToBuffer(0, 3)[0].size_bytes, 16384u);
    EXPECT_EQ(pool.convertIndexToBuffer(0, 0, 4, 2)[0].size_bytes, 16384u);
}

TEST(DeviceBlockPoolTest, OffloadReallocationChangesGenerationButReferencesDoNot) {
    DeviceBlockPool pool(makeOffloadConfig());
    ASSERT_TRUE(pool.init());
    const auto blocks = pool.malloc(5);
    ASSERT_TRUE(blocks.has_value());
    pool.incRef(*blocks);
    EXPECT_EQ(pool.takeOffloadVersionUpdates().size(), 5u);
    pool.incRef(2);
    pool.decRef(2);
    EXPECT_TRUE(pool.takeOffloadVersionUpdates().empty());
    pool.decRef(2);
    const auto reused = pool.malloc();
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(*reused, 2);
    pool.incRef(*reused);
    const auto updates = pool.takeOffloadVersionUpdates();
    ASSERT_EQ(updates.size(), 1u);
    EXPECT_EQ(updates[0].first, 2);
    EXPECT_EQ(updates[0].second, 2);
    pool.markOffloadOverwrite({2});
    pool.markOffloadOverwrite({2});
    const auto overwritten = pool.takeOffloadVersionUpdates();
    ASSERT_EQ(overwritten.size(), 1u);
    EXPECT_EQ(overwritten[0].second, 4);
    pool.decRef(*blocks);
}

TEST(DeviceBlockPoolTest, OffloadMhaPlanesPreserveNativeTransferBuffers) {
    for (uint32_t total : {6u, 2u}) {
        auto spec  = test::makeResolvedMhaSpec(TYPE_INT8, 2, 8, 8, "sparse_heads");
        auto cache = test::makeSingleGroupCacheConfig(spec, CacheGroupType::FULL, 2, total);
        cache.tiered_cache_configs.emplace(spec->tag, TieredCacheConfig{total, 2, 16});
        auto config = std::make_shared<DeviceBlockPoolConfig>(
            DeviceBlockPoolConfigHelper::createConfigForGroup(cache, cache.group(spec->tag)));
        DeviceBlockPool pool(config);
        ASSERT_TRUE(pool.init());
        for (int layer = 0; layer < 2; ++layer) {
            const auto& storage = *pool.allLayerOffloadStorage()[layer];
            ASSERT_EQ(storage.resident_planes.size(), 4u);
            EXPECT_EQ(storage.resident_planes[0].sizes().vec(), (std::vector<int64_t>{4, 8, 2, 8}));
            EXPECT_EQ(storage.resident_planes[0].strides().vec(), (std::vector<int64_t>{256, 8, 64, 1}));
            EXPECT_EQ(storage.resident_planes[2].strides().vec(), (std::vector<int64_t>{32, 1, 8}));
            EXPECT_EQ(storage.host_planes[1].size(0), total - 2);
            for (const auto* planes : {&storage.resident_planes, &storage.host_planes}) {
                (*planes)[0].fill_(7);
                (*planes)[1].fill_(9);
                (*planes)[2].fill_(0.25);
                (*planes)[3].fill_(0.5);
            }
            for (uint32_t block = 1; block < total; ++block) {
                const auto buffers = pool.convertIndexToBuffer(layer, block);
                ASSERT_EQ(buffers.size(), 2u);
                EXPECT_EQ(buffers[0].size_bytes, 256u);
                EXPECT_EQ(buffers[1].size_bytes, 128u);
                const auto options = torch::TensorOptions().device(block < 2 ? torch::kCUDA : torch::kCPU);
                const auto kv      = torch::from_blob(buffers[0].addr, {2, 2, 8, 8}, options.dtype(torch::kInt8)).cpu();
                const auto scales  = torch::from_blob(buffers[1].addr, {2, 2, 8}, options.dtype(torch::kFloat32)).cpu();
                EXPECT_TRUE(kv[0].eq(7).all().item<bool>());
                EXPECT_TRUE(kv[1].eq(9).all().item<bool>());
                EXPECT_TRUE(scales[0].eq(0.25).all().item<bool>());
                EXPECT_TRUE(scales[1].eq(0.5).all().item<bool>());
                const auto whole = pool.convertIndexToBuffer(layer, block, 1, 0);
                ASSERT_EQ(whole.size(), 4u);
                EXPECT_EQ(whole[0].size_bytes + whole[1].size_bytes, buffers[0].size_bytes);
                EXPECT_EQ(whole[2].size_bytes + whole[3].size_bytes, buffers[1].size_bytes);
                const auto parts = pool.convertIndexToBuffer(layer, block, 2, 1);
                ASSERT_EQ(parts.size(), 4u);
                EXPECT_EQ(parts[0].addr, static_cast<char*>(buffers[0].addr) + 64);
                EXPECT_EQ(parts[1].addr, static_cast<char*>(buffers[0].addr) + 192);
                EXPECT_EQ(parts[2].addr, static_cast<char*>(buffers[1].addr) + 32);
                EXPECT_EQ(parts[3].addr, static_cast<char*>(buffers[1].addr) + 96);
            }
        }
    }
}

TEST(DeviceBlockPoolTest, OffloadDonorTransferChangesGeneration) {
    DeviceBlockPool pool(makeOffloadConfig());
    ASSERT_TRUE(pool.init());
    const auto blocks = pool.malloc(5);
    ASSERT_TRUE(blocks.has_value());
    pool.incRef(*blocks);
    (void)pool.takeOffloadVersionUpdates();
    BlockIndicesType replacements;
    int              required = -1;
    ASSERT_TRUE(pool.tryReplaceRequestReferences({{2, 1, 0}}, 1, replacements, required));
    EXPECT_EQ(required, 0);
    EXPECT_EQ(replacements, (BlockIndicesType{2}));
    const auto updates = pool.takeOffloadVersionUpdates();
    ASSERT_EQ(updates.size(), 1u);
    EXPECT_EQ(updates[0].first, 2);
    EXPECT_EQ(updates[0].second, 2);
    pool.decRef(*blocks);
}

TEST(DeviceBlockPoolTest, OffloadViewsRetainBackingAndSupportNoHostBlocks) {
    OffloadStoragePtr storage;
    {
        auto config                       = makeOffloadConfig(false, 2);
        config->use_device_malloc_backing = true;
        config->offload_version_owner     = false;
        DeviceBlockPool pool(config);
        ASSERT_TRUE(pool.init());
        storage = pool.allLayerOffloadStorage()[0];
        EXPECT_EQ(storage->host_planes[0].size(0), 0);
        EXPECT_TRUE(storage->host_planes[0].is_pinned());
        EXPECT_TRUE(pool.takeOffloadVersionUpdates().empty());
    }
    storage->resident_planes[0].fill_(17);
    EXPECT_TRUE(storage->resident_planes[0].eq(17).all().item<bool>());
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

    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(pool.decRef(*block));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;

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

    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(pool.incRef(BlockIdList{*block, unallocated_block}));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;

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

    const bool old_core_dump                     = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_ANY_THROW(pool.decRef(*blocks));
    StaticConfig::user_ft_core_dump_on_exception = old_core_dump;

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

}  // namespace rtp_llm
