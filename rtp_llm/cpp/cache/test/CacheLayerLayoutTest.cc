#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <torch/extension.h>

#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace {

GroupBase makeLayoutGroup(std::string tag, std::vector<int> layer_ids) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = tag;
    spec->seq_size_per_block = 512;

    GroupBase group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids                 = std::move(layer_ids);
    group.seq_size_per_block        = 512;
    group.kernel_seq_size_per_block = 128;
    return group;
}

CacheLayerLayout makeLayerLayout(size_t layer_count, const std::vector<int>& active_layers, int value) {
    std::vector<BlockBufferPtrInfo> layers(layer_count);
    for (int layer_id : active_layers) {
        layers[static_cast<size_t>(layer_id)].kv_addr = torch::full({2, 8}, value, torch::kInt32);
    }
    return CacheLayerLayout(std::move(layers));
}

TEST(CacheLayerLayoutTest, SingleGroupCoversAllLayersAndTagMatchesSlotApi) {
    auto topology =
        CacheTopology::create({makeLayoutGroup("full", {0, 1, 2})}, {{0, {"full"}}, {1, {"full"}}, {2, {"full"}}});
    GroupedCacheLayerLayout::GroupLayouts groups;
    groups.emplace("full", makeLayerLayout(3, {0, 1, 2}, 7));
    GroupedCacheLayerLayout layout(topology, std::move(groups));

    EXPECT_FALSE(layout.group("full").empty());
    EXPECT_EQ(layout.group("full").activeLayerCount(), 3u);
    EXPECT_EQ(layout.groupId("full"), 0u);
    EXPECT_EQ(layout.at("full", 1).kv_addr.data_ptr(), layout.at(0, 1).kv_addr.data_ptr());
    EXPECT_EQ(layout.at(1).kv_addr.data_ptr(), layout.at("full", 1).kv_addr.data_ptr());
}

TEST(CacheLayerLayoutTest, SupportsOneGroupPerLayerAndOneToManyTopology) {
    auto topology = CacheTopology::create({makeLayoutGroup("a", {0, 2}), makeLayoutGroup("b", {1, 2})},
                                          {{0, {"a"}}, {1, {"b"}}, {2, {"a", "b"}}});
    GroupedCacheLayerLayout::GroupLayouts groups;
    groups.emplace("a", makeLayerLayout(3, {0, 2}, 1));
    groups.emplace("b", makeLayerLayout(3, {1, 2}, 2));
    GroupedCacheLayerLayout layout(topology, std::move(groups));

    EXPECT_EQ(layout.group("a").activeLayerCount(), 2u);
    EXPECT_EQ(layout.group("b").activeLayerCount(), 2u);
    EXPECT_TRUE(layout.group("a").hasLayer(0));
    EXPECT_FALSE(layout.group("a").hasLayer(1));
    EXPECT_EQ(layout.at(0).kv_addr.data_ptr(), layout.at("a", 0).kv_addr.data_ptr());
    EXPECT_EQ(layout.at(1).kv_addr.data_ptr(), layout.at("b", 1).kv_addr.data_ptr());
    EXPECT_ANY_THROW(layout.at(2));
}

TEST(CacheLayerLayoutTest, EmptyPlaceholderIsSkippedAndProjectionRecountsActiveLayers) {
    auto topology = CacheTopology::create({makeLayoutGroup("active", {0, 1}), makeLayoutGroup("mtp", {})},
                                          {{0, {"active"}}, {1, {"active"}}});
    GroupedCacheLayerLayout::GroupLayouts groups;
    groups.emplace("active", makeLayerLayout(2, {0, 1}, 1));
    groups.emplace("mtp", makeLayerLayout(2, {}, 0));
    GroupedCacheLayerLayout layout(topology, std::move(groups));

    EXPECT_TRUE(layout.group("mtp").empty());
    EXPECT_EQ(layout.group("mtp").activeLayerCount(), 0u);
    EXPECT_FALSE(layout.hasGroupData("mtp"));

    std::vector<BlockBufferPtrInfo> projected_layers(1);
    projected_layers[0] = layout.at("active", 1);
    CacheLayerLayout projected(std::move(projected_layers));
    EXPECT_FALSE(projected.empty());
    EXPECT_EQ(projected.activeLayerCount(), 1u);
}

TEST(CacheLayerLayoutTest, InvalidTagSlotAndLayerFailFast) {
    auto topology = CacheTopology::create({makeLayoutGroup("full", {0})}, {{0, {"full"}}});
    GroupedCacheLayerLayout::GroupLayouts groups;
    groups.emplace("full", makeLayerLayout(1, {0}, 1));
    GroupedCacheLayerLayout layout(topology, std::move(groups));

    EXPECT_ANY_THROW(layout.group("missing"));
    EXPECT_ANY_THROW(layout.group(1));
    EXPECT_ANY_THROW(layout.group("full").at(1));
    EXPECT_ANY_THROW(layout.group("full").hasLayer(1));
}

}  // namespace
}  // namespace rtp_llm
