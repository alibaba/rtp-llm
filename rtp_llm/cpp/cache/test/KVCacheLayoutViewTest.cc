#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/extension.h>

#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {
namespace {

class TestKVCacheSpec: public KVCacheSpec {
public:
    TestKVCacheSpec(std::string tag, KVCacheSpecType type, size_t seq_size, size_t k_elems, size_t v_elems):
        k_elems_(k_elems), v_elems_(v_elems) {
        this->tag                = std::move(tag);
        this->type               = type;
        this->seq_size_per_block = static_cast<uint32_t>(seq_size);
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
    GroupBase group;
    group.tag                = tag;
    group.spec               = std::make_shared<TestKVCacheSpec>(tag, spec_type, physical_seq_size, k_elems, v_elems);
    group.policy.group_type  = group_type;
    group.layer_ids          = {0};
    group.block_num          = 4;
    group.local_kv_head_num  = local_kv_heads;
    group.seq_size_per_block = physical_seq_size;
    group.kernel_seq_size_per_block = kernel_seq_size;
    return group;
}

GroupedCacheLayerLayout makeLayout(std::vector<GroupBase>          groups,
                                   std::vector<std::string>        layer_tags,
                                   std::vector<BlockBufferPtrInfo> buffers) {
    EXPECT_EQ(groups.size(), buffers.size());
    auto topology = CacheTopology::create(std::move(groups), {{0, std::move(layer_tags)}});
    GroupedCacheLayerLayout::GroupLayouts layouts;
    for (size_t group_index = 0; group_index < topology->groups().size(); ++group_index) {
        layouts.emplace(topology->groups().at(group_index).tag,
                        CacheLayerLayout(std::vector<BlockBufferPtrInfo>{std::move(buffers[group_index])}));
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
