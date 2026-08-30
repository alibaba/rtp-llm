#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/extension.h>

#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/CacheGroupTagOrder.h"
#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {
namespace {

class TestKVCacheSpec: public KVCacheSpec {
public:
    TestKVCacheSpec(KVCacheSpecType type, size_t seq_size, size_t kernel_seq_size, size_t k_elems, size_t v_elems):
        k_elems_(k_elems), v_elems_(v_elems) {
        this->type                      = type;
        this->seq_size_per_block        = static_cast<uint32_t>(seq_size);
        this->kernel_seq_size_per_block = static_cast<uint32_t>(kernel_seq_size);
    }

    size_t block_size() const override {
        return k_elems_ + v_elems_;
    }
    size_t k_block_size() const override {
        return k_elems_;
    }
    size_t v_block_size() const override {
        return v_elems_;
    }
    size_t block_size_bytes() const override {
        return block_size() * sizeof(at::Half);
    }
    size_t k_block_size_bytes() const override {
        return k_elems_ * sizeof(at::Half);
    }
    size_t v_block_size_bytes() const override {
        return v_elems_ * sizeof(at::Half);
    }
    DataType memoryLayoutDType() const override {
        return DataType::TYPE_FP16;
    }
    KVCacheSpecPtr clone() const override {
        return std::make_shared<TestKVCacheSpec>(*this);
    }
    std::string debugString(size_t = 0) const override {
        return "TestKVCacheSpec";
    }

private:
    size_t k_elems_;
    size_t v_elems_;
};

CacheGroup makeGroup(const std::string& tag,
                     KVCacheSpecType    spec_type,
                     CacheGroupType     group_type,
                     size_t             physical_seq_size,
                     size_t             kernel_seq_size,
                     size_t             k_elems,
                     size_t             v_elems,
                     uint32_t           local_kv_heads = 1) {
    CacheGroup group;
    group.tag  = tag;
    group.spec = std::make_shared<TestKVCacheSpec>(spec_type, physical_seq_size, kernel_seq_size, k_elems, v_elems);
    group.policy.group_type = group_type;
    group.block_num         = 4;
    group.local_kv_head_num = local_kv_heads;
    return group;
}

GroupedCacheLayerLayout makeLayout(std::vector<CacheGroup>         groups,
                                   std::vector<std::string>        layer_tags,
                                   std::vector<BlockBufferPtrInfo> buffers) {
    EXPECT_EQ(groups.size(), buffers.size());
    auto config = std::make_shared<CacheConfig>(
        std::move(groups), std::vector<CacheLayer>{std::move(layer_tags)}, /*main_layer_num=*/1);
    GroupedCacheLayerLayout::GroupLayouts layouts;
    size_t                                buffer_ordinal = 0;
    for (const auto& group : config->groups()) {
        layouts.emplace(group.tag,
                        CacheLayerLayout(std::vector<BlockBufferPtrInfo>{std::move(buffers[buffer_ordinal++])}));
    }
    return GroupedCacheLayerLayout(std::move(config), std::move(layouts));
}

CacheConfig createSparseIndexerCacheConfig() {
    ModelConfig model_config;
    model_config.num_layers                   = 1;
    model_config.data_type                    = DataType::TYPE_FP16;
    model_config.attn_config.tokens_per_block = 512;

    KVCacheSpecDesc indexer_desc;
    indexer_desc.tag                 = "indexer_kv";
    indexer_desc.cache_type          = KVCacheSpecType::OpaqueKV;
    indexer_desc.entry_dtype         = DataType::TYPE_UINT8;
    indexer_desc.entry_elems         = 132;
    indexer_desc.entry_count_mode    = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    indexer_desc.compression_ratio   = 1;
    model_config.kv_cache_spec_descs = {{indexer_desc}};

    KVCacheConfig kv_cache_config;
    kv_cache_config.seq_size_per_block        = 512;
    kv_cache_config.kernel_seq_size_per_block = 64;
    kv_cache_config.test_block_num            = 3;
    return CacheConfigCreator::createConfig(model_config, ParallelismConfig{}, RuntimeConfig{}, kv_cache_config);
}

GroupedCacheLayerLayout makeIndexerLayout(const CacheConfig& config, torch::Tensor physical_storage) {
    const auto topology = std::shared_ptr<const CacheConfig>(&config, [](const CacheConfig*) {});
    GroupedCacheLayerLayout::GroupLayouts layouts;
    for (const auto& group : topology->groups()) {
        std::vector<BlockBufferPtrInfo> layers(topology->layers().size());
        if (group.tag == "indexer_kv") {
            layers[0].kv_addr = std::move(physical_storage);
        }
        layouts.emplace(group.tag, CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(topology, std::move(layouts));
}

CacheConfig createFp8SparseMlaCacheConfig() {
    ModelConfig model_config;
    model_config.num_layers                   = 1;
    model_config.data_type                    = DataType::TYPE_BF16;
    model_config.attn_config.use_mla          = true;
    model_config.attn_config.kv_lora_rank     = 512;
    model_config.attn_config.rope_head_dim    = 64;
    model_config.attn_config.tokens_per_block = 512;
    model_config.attn_config.kv_cache_dtype   = KvCacheDataType::FP8;

    KVCacheSpecDesc mla_desc;
    mla_desc.tag        = "default";
    mla_desc.cache_type = KVCacheSpecType::MultiHeadLatentAttention;

    KVCacheSpecDesc compressed_desc;
    compressed_desc.tag               = "indexer_kv";
    compressed_desc.cache_type        = KVCacheSpecType::OpaqueKV;
    compressed_desc.entry_dtype       = DataType::TYPE_UINT8;
    compressed_desc.entry_elems       = 132;
    compressed_desc.entry_count_mode  = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    compressed_desc.compression_ratio = 1;
    model_config.kv_cache_spec_descs  = {{mla_desc, compressed_desc}};

    KVCacheConfig kv_cache_config;
    kv_cache_config.seq_size_per_block        = 512;
    kv_cache_config.kernel_seq_size_per_block = 64;
    kv_cache_config.test_block_num            = 2;
    return CacheConfigCreator::createConfig(model_config, ParallelismConfig{}, RuntimeConfig{}, kv_cache_config);
}

GroupedCacheLayerLayout
makeFp8SparseMlaLayout(const CacheConfig& config, torch::Tensor mla_storage, torch::Tensor compressed_storage) {
    const auto topology = std::shared_ptr<const CacheConfig>(&config, [](const CacheConfig*) {});
    GroupedCacheLayerLayout::GroupLayouts layouts;
    for (const auto& group : topology->groups()) {
        std::vector<BlockBufferPtrInfo> layers(topology->layers().size());
        if (group.tag == "default") {
            layers[0].kv_addr = std::move(mla_storage);
        } else if (group.tag == "indexer_kv") {
            layers[0].kv_addr = std::move(compressed_storage);
        }
        layouts.emplace(group.tag, CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(topology, std::move(layouts));
}

TEST(KVCacheLayoutViewTest, MhaUsesGroupHeadsAndSpecPayloadForKernelView) {
    const auto         base  = torch::arange(3 * 64, torch::TensorOptions().dtype(torch::kFloat16)).reshape({3, 64});
    const auto         scale = torch::arange(3 * 16, torch::TensorOptions().dtype(torch::kFloat32)).reshape({3, 16});
    auto               group = makeGroup("full",
                           KVCacheSpecType::MultiHeadAttention,
                           CacheGroupType::FULL,
                           /*physical_seq_size=*/8,
                           /*kernel_seq_size=*/2,
                           /*k_elems=*/32,
                           /*v_elems=*/32,
                           /*local_kv_heads=*/1);
    torch_ext::KVCache cache(makeLayout({std::move(group)}, {"full"}, {{base, scale}}));

    const auto layer  = cache.getLayerCache(0);
    const auto by_tag = cache.getLayerCache(0, "full");
    EXPECT_EQ(layer.seq_size_per_block, 2);
    EXPECT_EQ(layer.kv_cache_base.sizes().vec(), (std::vector<int64_t>{12, 2, 1, 2, 4}));
    EXPECT_EQ(layer.kv_scale_base.sizes().vec(), (std::vector<int64_t>{12, 4}));
    EXPECT_EQ(layer.kv_cache_base.data_ptr(), base.data_ptr());
    EXPECT_EQ(by_tag.kv_cache_base.data_ptr(), layer.kv_cache_base.data_ptr());
    EXPECT_EQ(by_tag.tag, "full");
    EXPECT_EQ(cache.groupTags(), std::vector<std::string>{"full"});
    EXPECT_EQ(cache.layerCount(), 1u);
    EXPECT_EQ(cache.getSeqSizePerBlock("full"), 8);
    EXPECT_EQ(cache.getKernelSeqSizePerBlock("full"), 2);
}

TEST(KVCacheLayoutViewTest, MlaReshapesKvAndScaleWithoutChangingStorage) {
    const auto base =
        torch::arange(2 * 8 * 6, torch::TensorOptions().dtype(torch::kFloat32)).to(torch::kBFloat16).reshape({2, 8, 6});
    const auto scale =
        torch::arange(2 * 8 * 3, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kUInt8).reshape({2, 8, 3});
    auto               group = makeGroup("mla",
                           KVCacheSpecType::MultiHeadLatentAttention,
                           CacheGroupType::FULL,
                           8,
                           2,
                           /*k_elems=*/32,
                           /*v_elems=*/16);
    torch_ext::KVCache cache(makeLayout({std::move(group)}, {"mla"}, {{base, scale}}));

    const auto layer = cache.getLayerCache(0, "mla");
    EXPECT_EQ(layer.kv_cache_base.sizes().vec(), (std::vector<int64_t>{8, 2, 6}));
    EXPECT_EQ(layer.kv_scale_base.sizes().vec(), (std::vector<int64_t>{8, 2, 3}));
    EXPECT_EQ(layer.kv_cache_base.data_ptr(), base.data_ptr());
    EXPECT_EQ(layer.kv_scale_base.data_ptr(), scale.data_ptr());
}

TEST(KVCacheLayoutViewTest, LinearSwaAndStateStayPhysical) {
    const auto physical = torch::arange(3 * 64, torch::TensorOptions().dtype(torch::kFloat16)).reshape({3, 64});
    for (const auto& [tag, spec_type, policy] : std::vector<std::tuple<std::string, KVCacheSpecType, CacheGroupType>>{
             {"linear", KVCacheSpecType::LinearAttention, CacheGroupType::LINEAR},
             {"swa", KVCacheSpecType::MultiHeadAttention, CacheGroupType::SWA},
             {"state", KVCacheSpecType::OpaqueState, CacheGroupType::FULL}}) {
        auto               group = makeGroup(tag, spec_type, policy, 8, 2, 32, 32);
        torch_ext::KVCache cache(makeLayout({std::move(group)}, {tag}, {{physical, {}}}));
        const auto         layer = cache.getLayerCache(0);
        EXPECT_EQ(layer.seq_size_per_block, 8) << tag;
        EXPECT_EQ(layer.kv_cache_base.sizes().vec(), physical.sizes().vec()) << tag;
        EXPECT_EQ(layer.kv_cache_base.data_ptr(), physical.data_ptr()) << tag;
    }
}

TEST(KVCacheLayoutViewTest, SparseIndexerSpecOwnsPhysicalBlockAndOpaqueViewPreservesKernelPageGeometry) {
    constexpr int64_t kPhysicalBlocks         = 3;
    constexpr int64_t kPhysicalTokensPerBlock = 512;
    constexpr int64_t kKernelTokensPerBlock   = 64;
    constexpr int64_t kEntryBytes             = 132;
    constexpr int64_t kKernelPageBytes        = kKernelTokensPerBlock * kEntryBytes;
    constexpr int64_t kPhysicalBlockBytes     = kPhysicalTokensPerBlock * kEntryBytes;
    constexpr int64_t kBlocksPerPhysicalBlock = kPhysicalTokensPerBlock / kKernelTokensPerBlock;

    const auto  config = createSparseIndexerCacheConfig();
    const auto& group  = config.group("indexer_kv");
    const auto* spec   = dynamic_cast<const CompressedKVCacheSpec*>(group.spec.get());
    ASSERT_NE(spec, nullptr);
    EXPECT_EQ(spec->block_payload_bytes(), kPhysicalBlockBytes);
    EXPECT_EQ(spec->block_size_bytes(), kPhysicalBlockBytes);
    EXPECT_EQ(group.seqSizePerBlock(), kPhysicalTokensPerBlock);
    EXPECT_EQ(group.kernelSeqSizePerBlock(), kKernelTokensPerBlock);
    EXPECT_EQ(group.kv_block_stride_bytes, kPhysicalBlockBytes);

    const auto physical = torch::arange(kPhysicalBlocks * static_cast<int64_t>(group.kv_block_stride_bytes),
                                        torch::TensorOptions().dtype(torch::kUInt8))
                              .reshape({kPhysicalBlocks, static_cast<int64_t>(group.kv_block_stride_bytes)});
    torch_ext::KVCache cache(makeIndexerLayout(config, physical));
    const auto         layer = cache.getLayerCache(0, "indexer_kv");
    EXPECT_EQ(layer.seq_size_per_block, kKernelTokensPerBlock);
    EXPECT_EQ(layer.kv_cache_base.sizes().vec(),
              (std::vector<int64_t>{kPhysicalBlocks * kBlocksPerPhysicalBlock, kKernelPageBytes}));
    EXPECT_EQ(layer.kv_cache_base.data_ptr(), physical.data_ptr());
    EXPECT_EQ(layer.kv_cache_base.select(0, kBlocksPerPhysicalBlock).data_ptr(), physical.select(0, 1).data_ptr());
    EXPECT_TRUE(torch::equal(layer.kv_cache_base.flatten(), physical.flatten()));
}

TEST(KVCacheLayoutViewTest, Fp8MlaAndCompressedSpecsOwnPhysicalBlocksWhileViewsProjectKernelBlocks) {
    constexpr int64_t kPhysicalBlocks          = 2;
    constexpr int64_t kPhysicalTokensPerBlock  = 512;
    constexpr int64_t kKernelTokensPerBlock    = 64;
    constexpr int64_t kBlocksPerPhysicalBlock  = kPhysicalTokensPerBlock / kKernelTokensPerBlock;
    constexpr int64_t kFp8MlaWidth             = 656;
    constexpr int64_t kCompressedEntryBytes    = 132;
    constexpr int64_t kMlaPhysicalBytes        = kPhysicalTokensPerBlock * kFp8MlaWidth;
    constexpr int64_t kCompressedPhysicalBytes = kPhysicalTokensPerBlock * kCompressedEntryBytes;
    constexpr int64_t kCompressedKernelBytes   = kKernelTokensPerBlock * kCompressedEntryBytes;

    const auto  config           = createFp8SparseMlaCacheConfig();
    const auto& mla_group        = config.group("default");
    const auto& compressed_group = config.group("indexer_kv");

    ASSERT_EQ(mla_group.spec->type, KVCacheSpecType::MultiHeadLatentAttention);
    EXPECT_EQ(mla_group.spec->block_size_bytes(), kMlaPhysicalBytes);
    EXPECT_EQ(mla_group.kv_block_stride_bytes, kMlaPhysicalBytes);
    ASSERT_EQ(compressed_group.spec->type, KVCacheSpecType::OpaqueKV);
    EXPECT_EQ(compressed_group.spec->block_payload_bytes(), kCompressedPhysicalBytes);
    EXPECT_EQ(compressed_group.spec->block_size_bytes(), kCompressedPhysicalBytes);
    EXPECT_EQ(compressed_group.kv_block_stride_bytes, kCompressedPhysicalBytes);

    const auto mla_storage =
        torch::arange(kPhysicalBlocks * kMlaPhysicalBytes, torch::TensorOptions().dtype(torch::kInt64))
            .to(torch::kUInt8)
            .reshape({kPhysicalBlocks, kPhysicalTokensPerBlock, kFp8MlaWidth});
    const auto compressed_storage =
        torch::arange(kPhysicalBlocks * kCompressedPhysicalBytes, torch::TensorOptions().dtype(torch::kInt64))
            .to(torch::kUInt8)
            .reshape({kPhysicalBlocks, kCompressedPhysicalBytes});
    torch_ext::KVCache cache(makeFp8SparseMlaLayout(config, mla_storage, compressed_storage));

    const auto mla_view        = cache.getLayerCache(0, "default");
    const auto compressed_view = cache.getLayerCache(0, "indexer_kv");
    EXPECT_EQ(mla_view.seq_size_per_block, kKernelTokensPerBlock);
    EXPECT_EQ(mla_view.kv_cache_base.sizes().vec(),
              (std::vector<int64_t>{kPhysicalBlocks * kBlocksPerPhysicalBlock, kKernelTokensPerBlock, kFp8MlaWidth}));
    EXPECT_EQ(compressed_view.seq_size_per_block, kKernelTokensPerBlock);
    EXPECT_EQ(compressed_view.kv_cache_base.sizes().vec(),
              (std::vector<int64_t>{kPhysicalBlocks * kBlocksPerPhysicalBlock, kCompressedKernelBytes}));
    EXPECT_EQ(mla_view.kv_cache_base.data_ptr(), mla_storage.data_ptr());
    EXPECT_EQ(compressed_view.kv_cache_base.data_ptr(), compressed_storage.data_ptr());
    EXPECT_EQ(compressed_view.kv_cache_base.select(0, kBlocksPerPhysicalBlock).data_ptr(),
              compressed_storage.select(0, 1).data_ptr());
    EXPECT_TRUE(torch::equal(compressed_view.kv_cache_base.flatten(), compressed_storage.flatten()));
}

TEST(KVCacheLayoutViewTest, Fp8MhaSpecAndScaleStridesOwnPhysicalBlock) {
    ModelConfig model_config;
    model_config.num_layers                   = 1;
    model_config.data_type                    = DataType::TYPE_BF16;
    model_config.attn_config.head_num         = 4;
    model_config.attn_config.kv_head_num      = 2;
    model_config.attn_config.size_per_head    = 8;
    model_config.attn_config.tokens_per_block = 512;
    model_config.attn_config.kv_cache_dtype   = KvCacheDataType::FP8;

    KVCacheSpecDesc desc;
    desc.tag                         = "default";
    desc.cache_type                  = KVCacheSpecType::MultiHeadAttention;
    model_config.kv_cache_spec_descs = {{desc}};

    KVCacheConfig kv_cache_config;
    kv_cache_config.seq_size_per_block        = 512;
    kv_cache_config.kernel_seq_size_per_block = 64;
    kv_cache_config.test_block_num            = 2;
    const auto config =
        CacheConfigCreator::createConfig(model_config, ParallelismConfig{}, RuntimeConfig{}, kv_cache_config);
    const auto& group = config.group("default");

    constexpr size_t kPhysicalKvBytes    = 2 * 2 * 8 * 512;
    constexpr size_t kPhysicalScaleBytes = 2 * 2 * sizeof(float) * 512;
    EXPECT_EQ(group.spec->block_size_bytes(), kPhysicalKvBytes);
    EXPECT_EQ(group.spec->scale_block_size_bytes(), kPhysicalScaleBytes);
    EXPECT_EQ(group.kv_block_stride_bytes, kPhysicalKvBytes);
    EXPECT_EQ(group.kv_scale_stride_bytes, kPhysicalScaleBytes);
}

TEST(KVCacheLayoutViewTest, Dsv4Fp8CompressedPhysicalBlocksPreserveAlignedKernelPageGeometry) {
    constexpr int64_t kPhysicalTokensPerBlock = 16384;
    constexpr int64_t kKernelTokensPerBlock   = 128;
    constexpr int64_t kKernelPagesPerBlock    = kPhysicalTokensPerBlock / kKernelTokensPerBlock;
    constexpr int64_t kEntryBytes             = 584;
    constexpr int64_t kAlignmentBytes         = 576;
    constexpr int64_t kCsaCompressionRatio    = 4;
    constexpr int64_t kHcaCompressionRatio    = 128;
    constexpr int64_t kCsaKernelPayloadBytes  = kKernelTokensPerBlock / kCsaCompressionRatio * kEntryBytes;
    constexpr int64_t kHcaKernelPayloadBytes  = kKernelTokensPerBlock / kHcaCompressionRatio * kEntryBytes;
    constexpr int64_t kCsaKernelStrideBytes   = 19008;
    constexpr int64_t kHcaKernelStrideBytes   = 1152;

    ModelConfig model_config;
    model_config.num_layers                   = 1;
    model_config.data_type                    = DataType::TYPE_BF16;
    model_config.attn_config.tokens_per_block = kKernelTokensPerBlock;
    const auto make_desc                      = [](const std::string& tag, uint32_t compression_ratio) {
        KVCacheSpecDesc desc;
        desc.tag                          = tag;
        desc.cache_type                   = KVCacheSpecType::OpaqueKV;
        desc.entry_dtype                  = DataType::TYPE_UINT8;
        desc.entry_elems                  = kEntryBytes;
        desc.entry_count_mode             = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
        desc.compression_ratio            = compression_ratio;
        desc.block_stride_bytes_alignment = kAlignmentBytes;
        return desc;
    };
    model_config.kv_cache_spec_descs = {
        {make_desc("csa_kv", kCsaCompressionRatio), make_desc("hca_kv", kHcaCompressionRatio)}};

    KVCacheConfig kv_cache_config;
    kv_cache_config.seq_size_per_block        = kPhysicalTokensPerBlock;
    kv_cache_config.kernel_seq_size_per_block = kKernelTokensPerBlock;
    kv_cache_config.test_block_num            = 1;
    const auto config =
        CacheConfigCreator::createConfig(model_config, ParallelismConfig{}, RuntimeConfig{}, kv_cache_config);
    const auto& csa_group = config.group("csa_kv");
    const auto& hca_group = config.group("hca_kv");

    ASSERT_EQ(csa_group.spec->block_payload_bytes(), kCsaKernelPayloadBytes * kKernelPagesPerBlock);
    ASSERT_EQ(csa_group.kv_block_stride_bytes, kCsaKernelStrideBytes * kKernelPagesPerBlock);
    ASSERT_EQ(hca_group.spec->block_payload_bytes(), kHcaKernelPayloadBytes * kKernelPagesPerBlock);
    ASSERT_EQ(hca_group.kv_block_stride_bytes, kHcaKernelStrideBytes * kKernelPagesPerBlock);

    const auto csa_storage = torch::arange(csa_group.kv_block_stride_bytes, torch::TensorOptions().dtype(torch::kInt64))
                                 .to(torch::kUInt8)
                                 .reshape({1, static_cast<int64_t>(csa_group.kv_block_stride_bytes)});
    const auto hca_storage = torch::arange(hca_group.kv_block_stride_bytes, torch::TensorOptions().dtype(torch::kInt64))
                                 .to(torch::kUInt8)
                                 .reshape({1, static_cast<int64_t>(hca_group.kv_block_stride_bytes)});
    const auto topology = std::shared_ptr<const CacheConfig>(&config, [](const CacheConfig*) {});
    GroupedCacheLayerLayout::GroupLayouts layouts;
    for (const auto& group : topology->groups()) {
        BlockBufferPtrInfo buffer;
        buffer.kv_addr = group.tag == "csa_kv" ? csa_storage : hca_storage;
        layouts.emplace(group.tag, CacheLayerLayout(std::vector<BlockBufferPtrInfo>{std::move(buffer)}));
    }
    torch_ext::KVCache cache(GroupedCacheLayerLayout(topology, std::move(layouts)));
    const auto         csa_view = cache.getLayerCache(0, "csa_kv");
    const auto         hca_view = cache.getLayerCache(0, "hca_kv");

    EXPECT_EQ(csa_view.kv_cache_base.sizes().vec(),
              (std::vector<int64_t>{kKernelPagesPerBlock, kCsaKernelStrideBytes}));
    EXPECT_EQ(hca_view.kv_cache_base.sizes().vec(),
              (std::vector<int64_t>{kKernelPagesPerBlock, kHcaKernelStrideBytes}));
    EXPECT_EQ(csa_view.kv_cache_base.data_ptr(), csa_storage.data_ptr());
    EXPECT_EQ(hca_view.kv_cache_base.data_ptr(), hca_storage.data_ptr());
    EXPECT_EQ(csa_view.kv_cache_base.select(0, 1).data_ptr<uint8_t>(),
              csa_storage.data_ptr<uint8_t>() + kCsaKernelStrideBytes);
    EXPECT_EQ(hca_view.kv_cache_base.select(0, 1).data_ptr<uint8_t>(),
              hca_storage.data_ptr<uint8_t>() + kHcaKernelStrideBytes);
    EXPECT_TRUE(torch::equal(csa_view.kv_cache_base.flatten(), csa_storage.flatten()));
    EXPECT_TRUE(torch::equal(hca_view.kv_cache_base.flatten(), hca_storage.flatten()));
}

TEST(KVCacheLayoutViewTest, MultiGroupRequiresTagAndEnumerationSkipsPlaceholder) {
    const auto full       = torch::zeros({2, 64}, torch::TensorOptions().dtype(torch::kFloat16));
    const auto linear     = torch::ones({2, 9}, torch::TensorOptions().dtype(torch::kFloat16));
    auto       full_group = makeGroup("full", KVCacheSpecType::MultiHeadAttention, CacheGroupType::FULL, 8, 8, 32, 32);
    auto       linear_group = makeGroup("linear", KVCacheSpecType::LinearAttention, CacheGroupType::LINEAR, 8, 8, 9, 0);
    auto       empty_group  = makeGroup("empty", KVCacheSpecType::OpaqueState, CacheGroupType::LINEAR, 1, 1, 1, 0);
    torch_ext::KVCache cache(makeLayout({std::move(full_group), std::move(linear_group), std::move(empty_group)},
                                        {"full", "linear", "empty"},
                                        {{full, {}}, {linear, {}}, {{}, {}}}));

    EXPECT_ANY_THROW(cache.getLayerCache(0));
    const auto groups = cache.getLayerCacheGroups(0);
    ASSERT_EQ(groups.size(), 2u);
    EXPECT_EQ(groups[0].tag, "full");
    EXPECT_EQ(groups[1].tag, "linear");
    EXPECT_EQ(cache.getLayerCache(0, "linear").kv_cache_base.data_ptr(), linear.data_ptr());

    EXPECT_ANY_THROW(cache.getLayerCache(-1));
    EXPECT_ANY_THROW(cache.getLayerCache(1));
    EXPECT_ANY_THROW(cache.getLayerCache(0, "missing"));
    EXPECT_ANY_THROW(cache.getLayerCache(0, "empty"));
    EXPECT_ANY_THROW(cache.getSeqSizePerBlock("missing"));
}

TEST(KVCacheLayoutViewTest, SortedBoundaryTagOrderIsCanonicalAndValidated) {
    const std::vector<std::string> declared = {"swa_kv", "csa_kv", "indexer_kv"};
    const std::vector<std::string> expected = {"csa_kv", "indexer_kv", "swa_kv"};
    EXPECT_EQ(sortedCacheGroupTags(declared), expected);
    // Sorting must not disturb the caller's own record order.
    EXPECT_EQ(declared, (std::vector<std::string>{"swa_kv", "csa_kv", "indexer_kv"}));
    // Reordering the declaration cannot move an entry.
    EXPECT_EQ(sortedCacheGroupTags({"indexer_kv", "swa_kv", "csa_kv"}), expected);

    for (size_t ordinal = 0; ordinal < expected.size(); ++ordinal) {
        EXPECT_EQ(groupOrdinalForTag(expected, expected[ordinal]), ordinal);
    }
    EXPECT_ANY_THROW(groupOrdinalForTag(expected, "missing"));
    EXPECT_ANY_THROW(sortedCacheGroupTags({"full", ""}));
    EXPECT_ANY_THROW(sortedCacheGroupTags({"full", "full"}));
}

TEST(KVCacheLayoutViewTest, ModelCacheTagsAreSortedAndBindingIgnoresDeclarationOrder) {
    const auto csa       = torch::zeros({2, 64}, torch::TensorOptions().dtype(torch::kFloat16));
    const auto swa       = torch::ones({2, 64}, torch::TensorOptions().dtype(torch::kFloat16));
    const auto makeCache = [&](bool reversed) {
        auto csa_group   = makeGroup("csa_kv", KVCacheSpecType::MultiHeadAttention, CacheGroupType::FULL, 8, 8, 32, 32);
        auto swa_group   = makeGroup("swa_kv", KVCacheSpecType::MultiHeadAttention, CacheGroupType::SWA, 8, 8, 32, 32);
        auto placeholder = makeGroup("indexer_kv", KVCacheSpecType::OpaqueState, CacheGroupType::FULL, 1, 1, 1, 0);
        if (reversed) {
            return torch_ext::KVCache(makeLayout({std::move(placeholder), std::move(swa_group), std::move(csa_group)},
                                                 {"indexer_kv", "swa_kv", "csa_kv"},
                                                 {{{}, {}}, {swa, {}}, {csa, {}}}));
        }
        return torch_ext::KVCache(makeLayout({std::move(csa_group), std::move(swa_group), std::move(placeholder)},
                                             {"csa_kv", "swa_kv", "indexer_kv"},
                                             {{csa, {}}, {swa, {}}, {{}, {}}}));
    };

    const std::vector<std::string> sorted_tags = {"csa_kv", "indexer_kv", "swa_kv"};
    for (const bool reversed : {false, true}) {
        auto cache = makeCache(reversed);
        EXPECT_EQ(cache.groupTags(), sorted_tags) << "reversed=" << reversed;
        // The empty placeholder group is skipped by enumeration in both orders.
        const auto groups = cache.getLayerCacheGroups(0);
        ASSERT_EQ(groups.size(), 2u) << "reversed=" << reversed;
        std::vector<std::string> enumerated_tags;
        for (const auto& group : groups) {
            enumerated_tags.push_back(group.tag);
        }
        std::sort(enumerated_tags.begin(), enumerated_tags.end());
        EXPECT_EQ(enumerated_tags, (std::vector<std::string>{"csa_kv", "swa_kv"})) << "reversed=" << reversed;
        EXPECT_EQ(cache.getLayerCache(0, "csa_kv").kv_cache_base.data_ptr(), csa.data_ptr()) << "reversed=" << reversed;
        EXPECT_EQ(cache.getLayerCache(0, "swa_kv").kv_cache_base.data_ptr(), swa.data_ptr()) << "reversed=" << reversed;
        EXPECT_ANY_THROW(cache.getLayerCache(0, "indexer_kv")) << "reversed=" << reversed;
    }
}

}  // namespace
}  // namespace rtp_llm
