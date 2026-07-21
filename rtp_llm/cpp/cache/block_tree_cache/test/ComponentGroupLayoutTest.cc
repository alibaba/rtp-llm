#include <gtest/gtest.h>

#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"

namespace rtp_llm {
namespace {

Component makeComponent(int                        component_id,
                        int                        group_id,
                        const std::string&         tag,
                        const std::vector<int>&    model_layer_ids,
                        const std::vector<size_t>& layer_bytes) {
    Component component;
    component.component_id       = component_id;
    component.component_group_id = group_id;
    component.tag                = tag;
    component.model_layer_ids    = model_layer_ids;
    component.layer_bytes        = layer_bytes;
    return component;
}

TEST(ComponentGroupLayoutTest, SingleComponentOffsetsAndPayload) {
    const auto layout = ComponentGroupLayout::create({{100, 200, 150}});
    ASSERT_TRUE(layout.has_value());

    ASSERT_EQ(layout->slices().size(), 3u);
    EXPECT_EQ(layout->componentCount(), 1u);
    EXPECT_EQ(layout->payloadBytes(), 450u);

    const auto& slices = layout->slices();
    EXPECT_EQ(slices[0].component_idx, 0u);
    EXPECT_EQ(slices[0].layer_idx, 0u);
    EXPECT_EQ(slices[0].offset_bytes, 0u);
    EXPECT_EQ(slices[1].layer_idx, 1u);
    EXPECT_EQ(slices[1].offset_bytes, 100u);
    EXPECT_EQ(slices[2].layer_idx, 2u);
    EXPECT_EQ(slices[2].offset_bytes, 300u);
}

TEST(ComponentGroupLayoutTest, MultiComponentInterleavedModelLayersKeepCanonicalOrder) {
    const auto layout = ComponentGroupLayout::create({{10, 20}, {30, 40}});
    ASSERT_TRUE(layout.has_value());
    ASSERT_EQ(layout->slices().size(), 4u);
    EXPECT_EQ(layout->payloadBytes(), 100u);

    const auto& slices = layout->slices();
    EXPECT_EQ(slices[0].component_idx, 0u);
    EXPECT_EQ(slices[1].component_idx, 0u);
    EXPECT_EQ(slices[2].component_idx, 1u);
    EXPECT_EQ(slices[3].component_idx, 1u);
    EXPECT_EQ(slices[2].offset_bytes, 30u);
    EXPECT_EQ(slices[3].offset_bytes, 60u);
}

TEST(ComponentGroupLayoutTest, RejectsInvalidLayerBytes) {
    EXPECT_FALSE(ComponentGroupLayout::create({}).has_value());
    EXPECT_FALSE(ComponentGroupLayout::create({{}}).has_value());
    EXPECT_FALSE(ComponentGroupLayout::create({{8, 0}}).has_value());
}

TEST(ComponentGroupFinalizeLayoutTest, SealsMembershipAndLayoutTogether) {
    const std::vector<Component> components = {
        makeComponent(0, 0, "a", {0}, {64}),
        makeComponent(1, 0, "b", {0}, {32}),
    };

    auto group                = std::make_shared<FullComponentGroup>();
    group->component_group_id = 0;
    EXPECT_FALSE(group->hasLayout());
    EXPECT_TRUE(group->componentIndices().empty());

    ASSERT_TRUE(group->finalizeLayout({0, 1}, components));
    ASSERT_TRUE(group->hasLayout());
    EXPECT_EQ(group->componentIndices(), (std::vector<int>{0, 1}));
    EXPECT_EQ(group->layout().payloadBytes(), 96u);

    EXPECT_FALSE(group->finalizeLayout({1, 0}, components));
    EXPECT_EQ(group->componentIndices(), (std::vector<int>{0, 1}));
    EXPECT_EQ(group->layout().payloadBytes(), 96u);
}

TEST(ComponentGroupFinalizeLayoutTest, FailedFinalizeCommitsNothing) {
    const std::vector<Component> components = {
        makeComponent(0, 0, "a", {0}, {64}),
        makeComponent(1, 1, "b", {0}, {32}),
    };

    const std::vector<std::vector<int>> bad_memberships = {{0, 2}, {0, 1}, {}};
    for (const auto& membership : bad_memberships) {
        auto group                = std::make_shared<FullComponentGroup>();
        group->component_group_id = 0;
        EXPECT_FALSE(group->finalizeLayout(membership, components));
        EXPECT_FALSE(group->hasLayout());
        EXPECT_TRUE(group->componentIndices().empty());
    }
}

}  // namespace
}  // namespace rtp_llm
