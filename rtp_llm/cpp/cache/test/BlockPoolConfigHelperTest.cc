#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm::test {
namespace {

class BlockPoolConfigHelperTest: public ::testing::Test {
protected:
    void SetUp() override {
        old_core_dump_on_exception_ = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
    }
    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception = old_core_dump_on_exception_;
    }
private:
    bool old_core_dump_on_exception_{false};
};

template<typename Fn>
void expectRuntimeErrorContains(Fn&& fn, const std::string& expected) {
    try {
        fn();
        FAIL() << "expected std::runtime_error containing: " << expected;
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(expected), std::string::npos) << error.what();
    }
}

void appendMtp(CacheConfig& main, const CacheConfig& draft) {
    main.mtp_sub_configs.push_back(
        main.mergeMTPModule(draft, static_cast<int>(main.mtp_sub_configs.size()), main.layer_num));
}

DeviceBlockPoolConfig poolFor(const CacheConfig& config, const std::string& tag = "default") {
    return DeviceBlockPoolConfigHelper::createConfigForGroup(config, config.group(tag));
}

// Indexer storage is now an independent opaque pool, not MLA scale storage.
// Build both groups through the production spec builder and merge by tag.
CacheConfig sparseMlaConfig(int layers, uint32_t blocks, size_t entry_bytes, bool reverse = false) {
    KVCacheSpecDesc desc;
    desc.tag = "indexer_kv";
    desc.cache_type = KVCacheSpecType::OpaqueKV;
    desc.entry_dtype = DataType::TYPE_UINT8;
    desc.entry_elems = entry_bytes;
    desc.entry_count_mode = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    desc.compression_ratio = 1;
    SpecBuildContext ctx;
    ctx.seq_size_per_block = 4;
    ctx.kernel_tokens_per_block = 4;
    auto indexer = SpecBuilder::build(desc, ctx);
    auto mla = makeMlaSpec("default", 4, DataType::TYPE_BF16, 4, 4);
    std::vector<int> ids(layers);
    std::iota(ids.begin(), ids.end(), 0);
    CacheConfig config;
    config.layer_num = layers;
    config.seq_size_per_block = 4;
    config.use_mla = true;
    config.is_sparse = true;
    config.fromGroupedSpecs(reverse ? std::vector<KVCacheSpecPtr>{indexer, mla} :
                                     std::vector<KVCacheSpecPtr>{mla, indexer},
                            {ids, ids},
                            {CacheGroupType::FULL, CacheGroupType::FULL},
                            reverse ? std::vector<std::string>{"indexer_kv", "default"} :
                                      std::vector<std::string>{"default", "indexer_kv"});
    config.finalizeBlockNums(blocks, RuntimeConfig{});
    return config;
}

TEST_F(BlockPoolConfigHelperTest, SparseMtpUsesIndependentTagPoolsAndDraftStride) {
    auto main = sparseMlaConfig(2, 4, 8);
    auto draft = sparseMlaConfig(1, 3, 32, true);
    appendMtp(main, draft);
    const auto mla = poolFor(main);
    const auto indexer = poolFor(main, "indexer_kv");
    ASSERT_EQ(mla.memory_layouts.size(), 2u);
    ASSERT_EQ(indexer.memory_layouts.size(), 2u);
    EXPECT_EQ(mla.total_size_bytes, 3u * 4u * 64u);
    EXPECT_EQ(indexer.memory_layouts[0].kv_block_stride_bytes, 32u);
    EXPECT_EQ(indexer.memory_layouts[1].kv_block_stride_bytes, 128u);
    EXPECT_EQ(indexer.memory_layouts[1].kv_cache_offset_bytes, 2u * 4u * 32u);
    EXPECT_EQ(indexer.total_size_bytes, 2u * 4u * 32u + 4u * 128u);
    // The owning pool capacity, not the draft's stale capacity, sizes both segments.
    EXPECT_EQ(main.mtp_sub_configs[0]->group("indexer_kv").block_num, 3u);
    EXPECT_EQ(indexer.memory_layouts[1].block_num, 4u);
    for (const auto& layout : mla.memory_layouts) {
        EXPECT_FALSE(layout.hasScale());
    }
    for (const auto& layout : indexer.memory_layouts) {
        EXPECT_FALSE(layout.hasScale());
        EXPECT_FALSE(layout.enable_hybrid_attention);
    }
}

TEST_F(BlockPoolConfigHelperTest, MtpUsesItsOwnQuantizedScaleStride) {
    auto main = makeSimpleMhaCacheConfig(2, 4, 4, DataType::TYPE_BF16, 1, 2);
    auto draft = makeSimpleMhaCacheConfig(1, 3, 4, DataType::TYPE_INT8, 1, 2);
    appendMtp(main, draft);
    const auto pool = poolFor(main);
    ASSERT_EQ(pool.memory_layouts.size(), 2u);
    const auto& layout = pool.memory_layouts[1];
    EXPECT_EQ(layout.kv_block_stride_bytes, 16u);
    EXPECT_EQ(layout.kv_scale_stride_bytes, 32u);
    EXPECT_EQ(layout.kv_cache_offset_bytes, 2u * 4u * 32u);
    EXPECT_EQ(layout.kv_scale_offset_bytes, 320u);
    EXPECT_EQ(layout.kv_scale_pool_size_bytes, 128u);
    EXPECT_EQ(pool.total_size_bytes, 448u);
}

TEST_F(BlockPoolConfigHelperTest, MultipleMtpLayoutsKeepContiguousOffsets) {
    auto main = makeSimpleMhaCacheConfig(2, 4, 4, DataType::TYPE_BF16, 1, 2);
    appendMtp(main, makeSimpleMhaCacheConfig(1, 3, 4, DataType::TYPE_INT8, 1, 2));
    appendMtp(main, makeSimpleMhaCacheConfig(1, 2, 4, DataType::TYPE_BF16, 1, 4));
    const auto pool = poolFor(main);
    ASSERT_EQ(pool.memory_layouts.size(), 3u);
    EXPECT_EQ(pool.memory_layouts[2].kv_cache_offset_bytes, 448u);
    EXPECT_EQ(pool.memory_layouts[2].kv_block_stride_bytes, 64u);
    EXPECT_EQ(pool.memory_layouts[2].kv_scale_offset_bytes, 704u);
    EXPECT_FALSE(pool.memory_layouts[2].hasScale());
    EXPECT_EQ(pool.total_size_bytes, 704u);
}

TEST_F(BlockPoolConfigHelperTest, RejectsNullMtpSubConfig) {
    auto main = makeSimpleMhaCacheConfig(1, 4, 4, DataType::TYPE_BF16);
    main.mtp_sub_configs.push_back(nullptr);
    expectRuntimeErrorContains([&] { poolFor(main); }, "is null");
}

TEST_F(BlockPoolConfigHelperTest, RejectsMtpWithoutTopologyAtMergeBoundary) {
    auto main = makeSimpleMhaCacheConfig(1, 4, 4, DataType::TYPE_BF16);
    expectRuntimeErrorContains([&] { appendMtp(main, CacheConfig{}); }, "requires propose topology");
}

TEST_F(BlockPoolConfigHelperTest, RejectsLayerWithoutCacheGroup) {
    CacheConfig config;
    config.layer_num = 1;
    auto spec = makeMhaSpec("default", 4, DataType::TYPE_BF16, 1, 1);
    expectRuntimeErrorContains([&] {
        config.fromGroupedSpecs({spec}, {{}}, {CacheGroupType::FULL}, {"default"});
    }, "has no cache group");
}

TEST_F(BlockPoolConfigHelperTest, RejectsMtpWithUnmappedGroupAtMergeBoundary) {
    auto main = makeSimpleMhaCacheConfig(1, 4, 4, DataType::TYPE_BF16);
    auto draft = sparseMlaConfig(1, 4, 8);
    expectRuntimeErrorContains([&] { appendMtp(main, draft); }, "unmapped draft cache group");
}

TEST_F(BlockPoolConfigHelperTest, RejectsUnmergedMtpLayerCounts) {
    auto main = makeSimpleMhaCacheConfig(1, 4, 4, DataType::TYPE_BF16);
    main.mtp_sub_configs.push_back(std::make_shared<CacheConfig>(main));
    expectRuntimeErrorContains([&] { poolFor(main); }, "does not match topology layers");
}

TEST_F(BlockPoolConfigHelperTest, RejectsZeroOwningPoolCapacity) {
    auto config = makeSimpleMhaCacheConfig(1, 4, 4, DataType::TYPE_BF16);
    auto group = config.group("default");
    group.block_num = 0;
    expectRuntimeErrorContains([&] {
        DeviceBlockPoolConfigHelper::createConfigForGroup(config, group);
    }, "requires positive pool capacity");
}

TEST_F(BlockPoolConfigHelperTest, RejectsIncompatibleMtpTokenGeometry) {
    auto main = makeSimpleMhaCacheConfig(1, 4, 4, DataType::TYPE_BF16);
    auto draft = makeSimpleMhaCacheConfig(1, 4, 8, DataType::TYPE_BF16);
    expectRuntimeErrorContains([&] { appendMtp(main, draft); }, "incompatible block token spans");
}

}  // namespace
}  // namespace rtp_llm::test
