#include <limits>
#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/GdnPrefillMemory.h"

namespace rtp_llm {

TEST(GdnPrefillMemoryTest, EmptyAndInvalidDimensions) {
    EXPECT_EQ(GdnPrefillMemory::estimate(0, 0, 0, 0, 0, 0, 0, 0), 0);
    EXPECT_THROW(GdnPrefillMemory::estimate(1, 0, 1, 1, 2, 2, 4, 2), std::invalid_argument);
    EXPECT_THROW(GdnPrefillMemory::estimate(1, 1, 0, 1, 2, 2, 4, 2), std::invalid_argument);
}

TEST(GdnPrefillMemoryTest, AccountsRaggedChunksAndLiveTensors) {
    // 620 bytes/token; ceil(65/64)+3*2 FP32 1x2x2 states.
    EXPECT_EQ(GdnPrefillMemory::estimate(65, 2, 1, 1, 2, 2, 4, 2), 65 * 620 + 8 * 16);
}

TEST(GdnPrefillMemoryTest, LongPrefillScalesWithTokensNotOnlyRequests) {
    const auto short_batch = GdnPrefillMemory::estimate(32768, 2, 16, 64, 128, 128, 4096, 2);
    const auto long_batch  = GdnPrefillMemory::estimate(65245, 2, 16, 64, 128, 128, 4096, 2);
    EXPECT_GT(long_batch, short_batch);
    EXPECT_GT(long_batch, static_cast<size_t>(65245) * 64 * 128 * 2);
    EXPECT_EQ(GdnPrefillMemory::estimate(65536, 64, 16, 64, 128, 128, 4096, 2), static_cast<size_t>(20736638976ULL));
}

TEST(GdnPrefillMemoryTest, MoreSequencesReserveRaggedAndInitialFinalStates) {
    const auto a = GdnPrefillMemory::estimate(128, 1, 1, 1, 2, 2, 4, 2);
    const auto b = GdnPrefillMemory::estimate(128, 4, 1, 1, 2, 2, 4, 2);
    EXPECT_EQ(b - a, 3 * 3 * 16);
}

TEST(GdnPrefillMemoryTest, EstimateOverflowFailsClosed) {
    const auto max = std::numeric_limits<size_t>::max();
    EXPECT_THROW(GdnPrefillMemory::estimate(max, 1, 1, 1, 2, 2, 4, 2), std::overflow_error);
    EXPECT_THROW(GdnPrefillMemory::add(max, 1), std::overflow_error);
    EXPECT_THROW(GdnPrefillMemory::multiply(max, 2), std::overflow_error);
}

TEST(GdnPrefillMemoryTest, TokenBoundaryAndFailureAreAtomic) {
    size_t tokens = 0, sequences = 0;
    EXPECT_TRUE(GdnPrefillMemory::tryAdd(tokens, sequences, 3, 2, 8, 4));
    EXPECT_EQ(tokens, 6);
    EXPECT_EQ(sequences, 2);
    EXPECT_FALSE(GdnPrefillMemory::tryAdd(tokens, sequences, 2, 1, 8, 4));
    EXPECT_EQ(tokens, 6);
    EXPECT_EQ(sequences, 2);
    EXPECT_TRUE(GdnPrefillMemory::tryAdd(tokens, sequences, 1, 1, 8, 4));
    EXPECT_EQ(tokens, 7);
}

TEST(GdnPrefillMemoryTest, NoStandaloneBypassOrMultiplicationOverflow) {
    size_t tokens = 0, sequences = 0;
    EXPECT_FALSE(GdnPrefillMemory::tryAdd(tokens, sequences, 8, 1, 8, 8));
    EXPECT_FALSE(GdnPrefillMemory::tryAdd(tokens, sequences, 100, 1, 8, 8));
    EXPECT_FALSE(GdnPrefillMemory::tryAdd(tokens, sequences, std::numeric_limits<size_t>::max(), 8, 8, 8));
    EXPECT_FALSE(GdnPrefillMemory::tryAdd(tokens, sequences, 1, 1, 0, 8));
    EXPECT_EQ(tokens, 0);
    EXPECT_EQ(sequences, 0);
}

TEST(GdnPrefillMemoryTest, ExpandedSequenceBudgetAlsoBoundsStateMemory) {
    size_t tokens = 0, sequences = 0;
    EXPECT_TRUE(GdnPrefillMemory::tryAdd(tokens, sequences, 1, 4, 100, 4));
    EXPECT_FALSE(GdnPrefillMemory::tryAdd(tokens, sequences, 1, 1, 100, 4));
    EXPECT_EQ(tokens, 4);
    EXPECT_EQ(sequences, 4);
}

TEST(GdnPrefillMemoryTest, AutomaticKVLeavesRuntimeReserve) {
    EXPECT_EQ(GdnPrefillMemory::kvCacheBytes(100, 30), 70);
    EXPECT_THROW(GdnPrefillMemory::kvCacheBytes(30, 30), std::invalid_argument);
    EXPECT_THROW(GdnPrefillMemory::kvCacheBytes(20, 30), std::invalid_argument);
}

TEST(GdnPrefillMemoryTest, ExplicitKVCannotBypassRuntimeReserve) {
    EXPECT_EQ(GdnPrefillMemory::kvCacheBytes(100, 30, 60), 60);
    EXPECT_EQ(GdnPrefillMemory::kvCacheBytes(100, 30, 70), 70);
    EXPECT_THROW(GdnPrefillMemory::kvCacheBytes(100, 30, 71), std::invalid_argument);
    EXPECT_THROW(GdnPrefillMemory::kvCacheBytes(100, 30, std::numeric_limits<size_t>::max()), std::invalid_argument);
}

}  // namespace rtp_llm
