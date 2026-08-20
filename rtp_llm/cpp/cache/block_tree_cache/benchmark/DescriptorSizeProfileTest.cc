#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/DescriptorSizeProfile.h"

#include <stdexcept>

#include <gtest/gtest.h>

namespace rtp_llm::benchmark {
namespace {

TEST(DescriptorSizeProfileTest, LoadsMinimalSizesAndBuildsSyntheticLayout) {
    const auto profile = DescriptorSizeProfile::fromString(R"json({
        "descriptor_size_bytes": {
            "full_context": 2930688,
            "swa": 1250304
        }
    })json");

    EXPECT_EQ(profile.descriptorSizeBytes("full_context"), 2930688);
    EXPECT_EQ(profile.descriptorSizeBytes("swa"), 1250304);
    EXPECT_EQ(profile.sha256_hex.size(), 64);

    const auto synthetic = profile.toSyntheticModelProfile();
    ASSERT_EQ(synthetic.groups.size(), 2);
    ASSERT_EQ(synthetic.group_sets.size(), 2);

    const auto* full = synthetic.findGroupSet("full_context");
    ASSERT_NE(full, nullptr);
    EXPECT_EQ(full->payload_bytes, 2930688);
    EXPECT_EQ(full->group_type, CacheGroupType::FULL);
    ASSERT_EQ(full->member_tags.size(), 1);
    const auto* full_member = synthetic.findGroup(full->member_tags.front());
    ASSERT_NE(full_member, nullptr);
    EXPECT_EQ(full_member->layer_count, 1);
    EXPECT_EQ(full_member->layer_stride_bytes, 2930688);

    const auto* swa = synthetic.findGroupSet("swa");
    ASSERT_NE(swa, nullptr);
    EXPECT_EQ(swa->payload_bytes, 1250304);
    EXPECT_EQ(swa->group_type, CacheGroupType::SWA);
    EXPECT_EQ(swa->sliding_window_size, 1);
}

TEST(DescriptorSizeProfileTest, ExtractsSizesFromLegacyModelProfile) {
    const auto profile = DescriptorSizeProfile::fromString(R"json({
        "profile_id": "legacy",
        "group_sets": [
            {"name": "full_context", "payload_bytes": 2930688},
            {"name": "swa", "payload_bytes": 1250304}
        ]
    })json");

    EXPECT_EQ(profile.descriptorSizeBytes("full_context"), 2930688);
    EXPECT_EQ(profile.descriptorSizeBytes("swa"), 1250304);
}

TEST(DescriptorSizeProfileTest, RejectsMissingOrZeroSizes) {
    EXPECT_THROW(DescriptorSizeProfile::fromString(R"json({
        "descriptor_size_bytes": {"full_context": 2930688}
    })json"), std::runtime_error);
    EXPECT_THROW(DescriptorSizeProfile::fromString(R"json({
        "descriptor_size_bytes": {"full_context": 0, "swa": 1250304}
    })json"), std::runtime_error);
}

}  // namespace
}  // namespace rtp_llm::benchmark
