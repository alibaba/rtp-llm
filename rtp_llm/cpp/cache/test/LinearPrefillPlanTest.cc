#include "rtp_llm/cpp/cache/LinearPrefillPlan.h"

#include <map>
#include <set>
#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TEST(LinearPrefillPlanTest, LastOneThroughFiveStayBoundedForOneMillionTokens) {
    for (int n = 1; n <= 5; ++n) {
        for (int tail : {0, 1, 128}) {
            const int64_t length = 1048576 + tail;
            LinearPrefillPlan plan(length, 128, 1024, n);
            ASSERT_EQ(plan.newPositions().size(), n + (tail != 0));
            std::map<int64_t, int64_t> contents;
            std::set<int64_t> frozen;
            for (int64_t prefix = 0; prefix < length;) {
                const auto round = plan.round(prefix, std::min<int64_t>(65536, length - prefix));
                if (round.input_owner_position) {
                    ASSERT_EQ(contents.at(*round.input_owner_position), prefix);
                }
                std::set<int64_t> outputs;
                for (const auto& write : round.writes) {
                    ASSERT_TRUE(outputs.insert(write.owner_position).second);
                    ASSERT_EQ(frozen.count(write.owner_position), 0);
                    contents[write.owner_position] = write.tokens;
                    if (write.retained) {
                        frozen.insert(write.owner_position);
                    }
                }
                ASSERT_EQ(contents.at(round.output_owner_position), round.end_tokens);
                ASSERT_LE(contents.size(), plan.newPositions().size());
                prefix = round.end_tokens;
            }
            for (const auto slot : plan.checkpointPositions()) {
                EXPECT_EQ(contents.at(slot), (slot + 1) * 128);
                EXPECT_TRUE(plan.canPublish(slot, length));
            }
        }
    }
}

TEST(LinearPrefillPlanTest, SpanningChunksReadTheLastActualOutput) {
    LinearPrefillPlan plan(12 * 1024 + 128, 128, 1024, 5);
    EXPECT_EQ(plan.round(6 * 1024, 2048).output_owner_position, 63);  // P1 / C8
    EXPECT_EQ(plan.round(8 * 1024, 2048).input_owner_position, 63);
    EXPECT_EQ(plan.round(8 * 1024, 2048).output_owner_position, 79);  // P3 / C10
    EXPECT_EQ(plan.round(10 * 1024, 2048).input_owner_position, 79);
    EXPECT_EQ(plan.round(12 * 1024, 128).input_owner_position, 95);  // P5 / C12
    EXPECT_EQ(plan.round(12 * 1024, 128).output_owner_position, 96); // T
}

TEST(LinearPrefillPlanTest, UncomputedPrefixHolesAreNotAllocatedOrPublished) {
    LinearPrefillPlan plan(13 * 1024 + 128, 128, 1024, 5, 10 * 1024);
    EXPECT_EQ(plan.newPositions(), (std::vector<int64_t>{87, 95, 103, 104}));
    EXPECT_EQ(plan.round(10 * 1024, 2048).input_owner_position, 79);
    EXPECT_FALSE(plan.canPublish(71, plan.promptTokens())); // Missing C9
    EXPECT_FALSE(plan.canPublish(79, plan.promptTokens())); // Existing C10
    EXPECT_TRUE(plan.canPublish(87, plan.promptTokens()));
}

TEST(LinearPrefillPlanTest, SmallRequestsAndUnwrittenFinalSlots) {
    EXPECT_EQ(LinearPrefillPlan(1023, 128, 1024, 5).newPositions(), (std::vector<int64_t>{7}));
    EXPECT_EQ(LinearPrefillPlan(1024, 128, 1024, 5).newPositions(), (std::vector<int64_t>{7}));
    EXPECT_EQ(LinearPrefillPlan(1025, 128, 1024, 5).newPositions(), (std::vector<int64_t>{7, 8}));
    EXPECT_EQ(LinearPrefillPlan(2049, 128, 1024, 5).newPositions().size(), 3);
    LinearPrefillPlan plan(1048576, 128, 1024, 5);
    EXPECT_FALSE(plan.canPublish(plan.borrowedPosition(), 65536));
    EXPECT_TRUE(plan.canPublish(plan.borrowedPosition(), 1048576));
}

TEST(LinearPrefillPlanTest, GrantsCanChangeWithoutAllocatingAnotherBank) {
    LinearPrefillPlan plan(6 * 1024 + 128, 128, 1024, 5);
    const auto first = plan.round(0, 384);
    const auto next = plan.round(384, 640);
    EXPECT_EQ(first.output_owner_position, plan.borrowedPosition());
    EXPECT_EQ(next.input_owner_position, first.output_owner_position);
    EXPECT_EQ(plan.round(1024, 1024).output_owner_position, 15); // First retained point C2
}

TEST(LinearPrefillPlanTest, WithoutCheckpointsEveryChunkUsesOnlyTheRequestTailBlock) {
    for (const int64_t length : {100000, 131072}) {
        LinearPrefillPlan plan(length, 2048, 16384, 0);
        const auto tail = (length - 1) / 2048;
        ASSERT_EQ(plan.newPositions(), (std::vector<int64_t>{tail}));
        EXPECT_TRUE(plan.checkpointPositions().empty());

        int64_t prefix = 0;
        for (const int64_t grant : {int64_t{65536}, int64_t{16384}, length - 81920}) {
            const auto round = plan.round(prefix, grant);
            EXPECT_EQ(round.input_owner_position, prefix ? std::optional<int64_t>{tail} : std::nullopt);
            ASSERT_EQ(round.writes.size(), 1u);
            EXPECT_EQ(round.writes.front().execution_position, (prefix + grant - 1) / 2048);
            EXPECT_EQ(round.writes.front().owner_position, tail);
            EXPECT_FALSE(round.writes.front().retained);
            EXPECT_EQ(round.output_owner_position, tail);
            prefix += grant;
        }
        EXPECT_EQ(prefix, length);
        EXPECT_FALSE(plan.canPublish(tail, length));
    }
}

TEST(LinearPrefillPlanTest, InvalidGeometryAndGrantsFailBeforeKernelDispatch) {
    EXPECT_THROW(LinearPrefillPlan(1024, 128, 1024, -1), std::invalid_argument);
    EXPECT_THROW(LinearPrefillPlan(1024, 128, 1000, 5), std::invalid_argument);
    EXPECT_THROW(LinearPrefillPlan(2048, 128, 1024, 5, 128), std::invalid_argument);
    LinearPrefillPlan plan(2049, 128, 1024, 5);
    EXPECT_THROW(plan.round(0, 127), std::invalid_argument);
    EXPECT_THROW(plan.round(0, 2050), std::invalid_argument);
}

}  // namespace
}  // namespace rtp_llm
