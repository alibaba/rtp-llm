#include "rtp_llm/cpp/cache/DecodeReuseSources.h"
#include <gtest/gtest.h>
namespace rtp_llm {
namespace {
DecodeReuseSource source(int64_t begin, int64_t end, int64_t handoff, Tier tier) {
    return {std::make_shared<const DecodeCacheOrigin>(
                DecodeCacheOrigin{std::string(32, 'a'), handoff, begin, end, 4096, 4097}),
            tier};
}
TEST(DecodeReuseSourcesTest, ClipsAtActualComputedHandoffAndAdoptedPrefix) {
    const auto counts =
        countDecodeReuse({source(0, 1024, 768, Tier::REMOTE), source(1024, 2048, 768, Tier::REMOTE)}, 1500);
    EXPECT_EQ(counts.total(), 1500 - 768);
    EXPECT_EQ(counts.by_tier[3], counts.total());
}
TEST(DecodeReuseSourcesTest, ReplicasGroupsAndRetriesNeverMultiplyLogicalUnion) {
    DecodeReuseSources values(16, source(512, 1024, 256, Tier::REMOTE));
    values.push_back(source(1024, 1536, 256, Tier::REMOTE));
    EXPECT_EQ(countDecodeReuse(values, 4096).total(), 1024);
}
TEST(DecodeReuseSourcesTest, MixedTierUnionDoesNotDoubleCountOverlap) {
    const auto counts = countDecodeReuse({source(0, 1024, 0, Tier::DEVICE),
                                          source(512, 1536, 0, Tier::HOST),
                                          source(1024, 2048, 0, Tier::DISK),
                                          source(1536, 2048, 0, Tier::REMOTE)},
                                         2048);
    EXPECT_EQ(counts.total(), 2048);
    for (const auto count : counts.by_tier)
        EXPECT_EQ(count, 512);
}
TEST(DecodeReuseSourcesTest, UnknownNegativeAndUnadoptedDoNotCount) {
    EXPECT_EQ(countDecodeReuse({{}, source(-1, 1024, -1, Tier::REMOTE), source(0, 1024, 0, Tier::NONE)}, 2048).total(),
              0);
    EXPECT_EQ(countDecodeReuse({source(0, 1024, 0, Tier::REMOTE)}, 0).total(), 0);
}
TEST(DecodeReuseSourcesTest, ActualUseRequiresProofAndIsOncePerIndependentRequest) {
    const auto values =
        std::make_shared<const DecodeReuseSources>(DecodeReuseSources{source(0, 1024, 768, Tier::DEVICE)});
    DecodeReuseUseState first;
    first.adopt(values, 1024);
    EXPECT_FALSE(first.consume(false));
    ASSERT_TRUE(first.pending());
    auto used = first.consume(true);
    ASSERT_TRUE(used);
    EXPECT_EQ(used->total(), 256);
    EXPECT_FALSE(first.consume(true));
    first.adopt(values, 1024);
    EXPECT_FALSE(first.consume(true));
    DecodeReuseUseState second;
    second.adopt(values, 1024);
    ASSERT_TRUE(second.consume(true));
    first.reset();
    EXPECT_FALSE(first.pending());
}
}  // namespace
}  // namespace rtp_llm
