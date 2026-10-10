#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/extension.h>

#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {
namespace {

class TestKVCacheSpec: public KVCacheSpec {
public:
    TestKVCacheSpec(std::string tag, KVCacheSpecType type, size_t seq_size, size_t k_elems, size_t v_elems):
        KVCacheSpec(std::move(tag), static_cast<uint32_t>(seq_size), 1, 1), k_elems_(k_elems), v_elems_(v_elems) {
        this->type = type;
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

GroupBase makeGroup(const std::string& tag,
                    KVCacheSpecType    spec_type,
                    CacheGroupType     group_type,
                    size_t             physical_seq_size,
                    size_t             kernel_seq_size,
                    size_t             k_elems,
                    size_t             v_elems,
                    uint32_t           local_kv_heads = 1) {
    auto spec = std::make_shared<TestKVCacheSpec>(tag, spec_type, physical_seq_size, k_elems, v_elems);
    spec->kernel_seq_size_per_block = kernel_seq_size;
    spec->local_kv_head_num         = local_kv_heads;
    GroupBase group;
    group.tag               = tag;
    group.spec              = std::move(spec);
    group.policy.group_type = group_type;
    group.block_num         = 4;
    return group;
}

GroupedCacheLayerLayout makeLayout(std::vector<GroupBase>          groups,
                                   std::vector<std::string>        layer_tags,
                                   std::vector<BlockBufferPtrInfo> buffers) {
    EXPECT_EQ(groups.size(), buffers.size());
    auto topology = CacheTopology::create(std::move(groups), {{0, std::move(layer_tags)}});
    GroupedCacheLayerLayout::GroupLayouts layouts;
    for (size_t group_id = 0; group_id < topology->groups().size(); ++group_id) {
        layouts.emplace(topology->groupTags()[group_id],
                        CacheLayerLayout(std::vector<BlockBufferPtrInfo>{std::move(buffers[group_id])}));
    }
    return GroupedCacheLayerLayout(std::move(topology), std::move(layouts));
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
    EXPECT_EQ(cache.getEntriesPerBlock("full"), 0u);
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

TEST(KVCacheLayoutViewTest, FullOpaqueExpandsButLinearSwaAndStateStayPhysical) {
    const auto opaque       = torch::arange(3 * 64, torch::TensorOptions().dtype(torch::kUInt8)).reshape({3, 64});
    auto       opaque_group = makeGroup("opaque", KVCacheSpecType::OpaqueKV, CacheGroupType::FULL, 512, 128, 64, 0);
    torch_ext::KVCache opaque_cache(makeLayout({std::move(opaque_group)}, {"opaque"}, {{opaque, {}}}));
    const auto         opaque_layer = opaque_cache.getLayerCache(0);
    EXPECT_EQ(opaque_layer.seq_size_per_block, 128);
    EXPECT_EQ(opaque_layer.kv_cache_base.sizes().vec(), (std::vector<int64_t>{12, 16}));

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

TEST(KVCacheLayoutViewTest, CompressedSpecPreservesPaddingBetweenKernelPages) {
    KVCacheSpecDesc desc;
    desc.tag                          = "compressed";
    desc.cache_type                   = KVCacheSpecType::OpaqueKV;
    desc.entry_dtype                  = DataType::TYPE_UINT8;
    desc.entry_elems                  = 3;
    desc.entry_count_mode             = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    desc.compression_ratio            = 4;
    desc.block_stride_bytes_alignment = 8;
    SpecBuildContext ctx;
    ctx.seq_size_per_block      = 32;
    ctx.kernel_tokens_per_block = 8;
    GroupBase group;
    group.tag               = desc.tag;
    group.spec              = CompressedKVCacheSpec::build(desc, ctx);
    group.policy.group_type = CacheGroupType::FULL;
    group.block_num         = 3;
    ASSERT_EQ(group.kvBlockStrideBytes(), 32u);
    auto               base = torch::arange(96, torch::TensorOptions().dtype(torch::kUInt8)).reshape({3, 32});
    torch_ext::KVCache cache(makeLayout({group}, {desc.tag}, {{base, {}}}));
    const auto         view = cache.getLayerCache(0).kv_cache_base;
    ASSERT_EQ(view.sizes().vec(), (std::vector<int64_t>{12, 8}));
    EXPECT_EQ(cache.getEntriesPerBlock(desc.tag), 2u);
    EXPECT_EQ(view.data_ptr(), base.data_ptr());
    for (int page = 0; page < 12; ++page) {
        EXPECT_EQ(view[page][0].item<int>(), page * 8);
        EXPECT_EQ(view[page][7].item<int>(), page * 8 + 7);
    }
}

TEST(KVCacheLayoutViewTest, Cp5ByteSlicedRingEntryCountExcludesAlignmentPadding) {
    KVCacheSpecDesc desc;
    desc.tag                          = "decoder_swa_kv";
    desc.cache_type                   = KVCacheSpecType::OpaqueState;
    desc.entry_dtype                  = DataType::TYPE_UINT8;
    desc.entry_elems                  = 528;
    desc.explicit_entry_count         = 128;
    desc.block_stride_bytes_alignment = 512;
    desc.cp                           = CacheCpPolicyDesc{};
    desc.cp->prefill_slice_layout     = CpPrefillSliceLayout::BLOCK_STRIDE;
    ParallelismConfig parallelism;
    parallelism.role_type                          = RoleType::PREFILL;
    parallelism.tp_size                            = 5;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    SpecBuildContext ctx;
    ctx.seq_size_per_block      = 640;
    ctx.kernel_tokens_per_block = 128;
    ctx.parallelism_config      = &parallelism;
    GroupBase group;
    group.tag               = desc.tag;
    group.spec              = FixedStateCacheSpec::build(desc, ctx);
    group.policy.group_type = CacheGroupType::LINEAR;
    group.block_num         = 2;
    const auto stride       = group.kvBlockStrideBytes();
    // LCM(512, 5) alignment adds at least one 528-byte entry's worth of padding.
    EXPECT_GT(stride * 5 / desc.entry_elems, desc.explicit_entry_count);
    auto base = torch::zeros({2, static_cast<int64_t>(stride)}, torch::TensorOptions().dtype(torch::kUInt8));
    torch_ext::KVCache cache(makeLayout({group}, {desc.tag}, {{base, {}}}));
    EXPECT_EQ(cache.getEntriesPerBlock(desc.tag), 128u);
    EXPECT_EQ(cache.getLayerCache(0).kv_cache_base.size(1), stride);
}

TEST(KVCacheLayoutViewTest, AlignedSwaRingSharesFullStrideAcrossPrefillAndDecode) {
    for (const uint32_t cp_size : {3u, 5u, 7u}) {
        KVCacheSpecDesc desc;
        desc.tag                          = "swa_kv";
        desc.cache_type                   = KVCacheSpecType::OpaqueState;
        desc.entry_dtype                  = DataType::TYPE_UINT8;
        desc.entry_elems                  = 528;
        desc.explicit_entry_count         = 136;
        desc.block_stride_bytes_alignment = 16896;  // LCM(512, 528)
        desc.cp                           = CacheCpPolicyDesc{};
        desc.cp->align_payload            = true;
        desc.cp->prefill_slice_layout     = CpPrefillSliceLayout::BLOCK_STRIDE;
        ParallelismConfig prefill;
        prefill.role_type                          = RoleType::PREFILL;
        prefill.tp_size                            = cp_size;
        prefill.prefill_cp_config.kv_cache_sharded = true;
        ParallelismConfig decode;
        decode.role_type                          = RoleType::DECODE;
        decode.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
        decode.prefill_cp_config.kv_cache_sharded = true;
        decode.prefill_cp_config.prefill_cp_size  = cp_size;
        SpecBuildContext ctx;
        ctx.seq_size_per_block      = 128 * cp_size;
        ctx.kernel_tokens_per_block = 128;
        ctx.parallelism_config      = &prefill;
        auto prefill_spec           = FixedStateCacheSpec::build(desc, ctx);
        ctx.parallelism_config      = &decode;
        auto       decode_spec      = FixedStateCacheSpec::build(desc, ctx);
        const auto full_stride      = decode_spec->block_size_bytes();
        EXPECT_EQ(full_stride % 528, 0u) << cp_size;
        EXPECT_EQ(full_stride % cp_size, 0u) << cp_size;
        EXPECT_EQ(prefill_spec->block_size_bytes() * cp_size, full_stride) << cp_size;
        EXPECT_EQ(prefill_spec->block_payload_bytes(), decode_spec->block_payload_bytes());
        const size_t logical_entries = ((136 + cp_size - 1) / cp_size) * cp_size;
        EXPECT_EQ(decode_spec->block_payload_bytes(), logical_entries * 528);
        GroupBase group;
        group.tag               = desc.tag;
        group.policy.group_type = CacheGroupType::LINEAR;
        group.block_num         = 2;
        for (const auto& spec : {prefill_spec, decode_spec}) {
            group.spec              = spec;
            auto               base = torch::zeros({2, static_cast<int64_t>(spec->block_size_bytes())},
                                     torch::TensorOptions().dtype(torch::kUInt8));
            torch_ext::KVCache cache(makeLayout({group}, {desc.tag}, {{base, {}}}));
            EXPECT_EQ(cache.getEntriesPerBlock(desc.tag), logical_entries) << cp_size;
        }
        // EQUAL_BYTES transport must tile the complete destination exactly,
        // with every sender publishing its local physical stride.
        group.spec            = decode_spec;
        group.policy.cp_slice = CpBlockSliceMode::EQUAL_BYTES;
        CacheConfig config;
        config.seq_size_per_block = 128;
        config.layer_num          = 1;
        config.setTopology({group}, {{0, {desc.tag}}});
        CPSlotMapper      mapper(0, cp_size, 128);
        std::vector<char> destination(full_stride);
        BlockInfo         block;
        block.addr       = destination.data();
        block.size_bytes = full_stride;
        for (uint32_t rank = 0; rank < cp_size; ++rank) {
            const auto slices = mapper.sliceBlockForPeer(config, desc.tag, {block}, rank);
            ASSERT_EQ(slices.size(), 1u);
            EXPECT_EQ(slices[0].size_bytes, prefill_spec->block_size_bytes());
            EXPECT_EQ(slices[0].addr, destination.data() + rank * prefill_spec->block_size_bytes());
        }
    }
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

}  // namespace
}  // namespace rtp_llm
