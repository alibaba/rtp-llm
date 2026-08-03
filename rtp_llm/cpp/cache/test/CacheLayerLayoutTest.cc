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
    EXPECT_EQ(layout.groupTagsForLayer(1), std::vector<std::string>{"full"});
    EXPECT_EQ(layout.group("full").at(1).kv_addr.data_ptr(), layout.at("full", 1).kv_addr.data_ptr());
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
    EXPECT_EQ(layout.groupTagsForLayer(0), std::vector<std::string>{"a"});
    EXPECT_EQ(layout.groupTagsForLayer(1), std::vector<std::string>{"b"});
    EXPECT_EQ(layout.groupTagsForLayer(2), (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(layout.group("a").at(0).kv_addr.data_ptr(), layout.at("a", 0).kv_addr.data_ptr());
    EXPECT_EQ(layout.group("b").at(1).kv_addr.data_ptr(), layout.at("b", 1).kv_addr.data_ptr());
    EXPECT_TRUE(layout.at("a", 2).kv_addr.defined());
    EXPECT_TRUE(layout.at("b", 2).kv_addr.defined());
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
    EXPECT_ANY_THROW(layout.group("full").at(1));
    EXPECT_ANY_THROW(layout.group("full").hasLayer(1));
    EXPECT_ANY_THROW(layout.groupTagsForLayer(1));
    EXPECT_ANY_THROW(layout.at("missing", 0));
}

TEST(CacheLayerLayoutTest, LayerTagEnumerationSupportsZeroGroupsWithoutAllocatingACollection) {
    auto topology = CacheTopology::create({makeLayoutGroup("full", {1})}, {{0, {}}, {1, {"full"}}});
    GroupedCacheLayerLayout::GroupLayouts groups;
    groups.emplace("full", makeLayerLayout(2, {1}, 1));
    GroupedCacheLayerLayout layout(topology, std::move(groups));

    const auto& tags = layout.groupTagsForLayer(0);
    EXPECT_TRUE(tags.empty());
}

TEST(CacheLayerLayoutTest, TagAccessIsIndependentOfTopologyTraversalOrder) {
    auto make_layout = [](std::vector<GroupBase> groups, std::vector<std::string> layer_tags) {
        auto topology = CacheTopology::create(std::move(groups), {{0, std::move(layer_tags)}});
        GroupedCacheLayerLayout::GroupLayouts layouts;
        layouts.emplace("a", makeLayerLayout(1, {0}, 1));
        layouts.emplace("b", makeLayerLayout(1, {0}, 2));
        return GroupedCacheLayerLayout(topology, std::move(layouts));
    };

    auto first =
        make_layout({makeLayoutGroup("a", {0}), makeLayoutGroup("b", {0})}, std::vector<std::string>{"a", "b"});
    auto second =
        make_layout({makeLayoutGroup("b", {0}), makeLayoutGroup("a", {0})}, std::vector<std::string>{"b", "a"});

    EXPECT_EQ(*first.at("a", 0).kv_addr.data_ptr<int>(), *second.at("a", 0).kv_addr.data_ptr<int>());
    EXPECT_EQ(*first.at("b", 0).kv_addr.data_ptr<int>(), *second.at("b", 0).kv_addr.data_ptr<int>());
}

}  // namespace
}  // namespace rtp_llm
