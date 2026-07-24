#include <gtest/gtest.h>

#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"

namespace rtp_llm {
namespace {

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

TEST(ComponentGroupSetLayoutTest, SealsMembershipAndLayoutTogether) {
    auto layout = ComponentGroupLayout::create({{64}, {32}});
    ASSERT_TRUE(layout.has_value());

    auto group                = std::make_shared<FullComponentGroup>();
    group->component_group_id = 0;
    EXPECT_FALSE(group->hasLayout());
    EXPECT_TRUE(group->componentIndices().empty());

    ASSERT_TRUE(group->setLayout({0, 1}, std::move(*layout)));
    ASSERT_TRUE(group->hasLayout());
    EXPECT_EQ(group->componentIndices(), (std::vector<int>{0, 1}));
    EXPECT_EQ(group->layout().payloadBytes(), 96u);

    auto replacement = ComponentGroupLayout::create({{32}, {64}});
    ASSERT_TRUE(replacement.has_value());
    EXPECT_FALSE(group->setLayout({1, 0}, std::move(*replacement)));
    EXPECT_EQ(group->componentIndices(), (std::vector<int>{0, 1}));
    EXPECT_EQ(group->layout().payloadBytes(), 96u);
}

}  // namespace
}  // namespace rtp_llm
