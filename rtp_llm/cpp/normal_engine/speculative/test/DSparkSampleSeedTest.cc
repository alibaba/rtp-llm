#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/normal_engine/speculative/DSparkSampleSeed.h"

namespace rtp_llm {
namespace speculative {
namespace {

// A seeded DSpARK stream must sample the same value for the same absolute
// position no matter what the round structure looked like. These tests pin the
// three properties that guarantee it: purity, position addressing, and a
// separate draft/target namespace.

TEST(DSparkSampleSeedTest, SameSeedAndPositionAlwaysProducesTheSameDraw) {
    const uint64_t seed = 7;
    for (int64_t position = 0; position < 64; ++position) {
        const auto first  = dsparkUniformAt(seed, kDSparkTargetSeedNamespace, position);
        const auto second = dsparkUniformAt(seed, kDSparkTargetSeedNamespace, position);
        EXPECT_EQ(first, second) << "position=" << position;
    }
}

TEST(DSparkSampleSeedTest, DrawDependsOnTheAbsolutePositionNotTheRoundOffset) {
    // A round that verifies positions [10, 14) must draw exactly what a round
    // that verifies only position 12 draws for that position. A stateful
    // generator cannot satisfy this once the budget changes the round width.
    const uint64_t     seed = 11;
    std::vector<float> wide;
    for (int64_t position = 10; position < 14; ++position) {
        wide.push_back(dsparkUniformAt(seed, kDSparkTargetSeedNamespace, position));
    }
    const auto narrow = dsparkUniformAt(seed, kDSparkTargetSeedNamespace, 12);
    EXPECT_EQ(wide[2], narrow);
    EXPECT_NE(wide[0], wide[1]);
    EXPECT_NE(wide[1], wide[2]);
    EXPECT_NE(wide[2], wide[3]);
}

TEST(DSparkSampleSeedTest, DifferentSeedsProduceDifferentDraws) {
    std::set<float> draws;
    for (uint64_t seed = 0; seed < 32; ++seed) {
        draws.insert(dsparkUniformAt(seed, kDSparkTargetSeedNamespace, 0));
    }
    EXPECT_EQ(draws.size(), 32u);
}

TEST(DSparkSampleSeedTest, DraftAndTargetNamespacesNeverCollide) {
    const uint64_t seed = 1000000007ULL;
    for (int64_t position = 0; position < 256; ++position) {
        const auto target = dsparkUniformAt(seed, kDSparkTargetSeedNamespace, position);
        const auto draft  = dsparkUniformAt(seed, kDSparkDraftSeedNamespace, position);
        EXPECT_NE(target, draft) << "position=" << position;
    }
}

TEST(DSparkSampleSeedTest, DrawsStayInsideTheUnitInterval) {
    for (uint64_t seed : {0ULL, 1ULL, 7ULL, 0xFFFFFFFFFFFFFFFFULL}) {
        for (int64_t position : {0LL, 1LL, 4096LL, 1048576LL}) {
            const auto value = dsparkUniformAt(seed, kDSparkTargetSeedNamespace, position);
            EXPECT_GE(value, 0.0f);
            EXPECT_LT(value, 1.0f);
            EXPECT_TRUE(std::isfinite(value));
        }
    }
}

TEST(DSparkSampleSeedTest, NegativeSeedsAreAddressable) {
    // GenerateConfig::random_seed is a signed int; the counter form must still
    // be a pure function after the caller's two's-complement widening.
    const auto seed = static_cast<uint64_t>(static_cast<int64_t>(-7));
    EXPECT_EQ(dsparkUniformAt(seed, kDSparkTargetSeedNamespace, 3),
              dsparkUniformAt(seed, kDSparkTargetSeedNamespace, 3));
    EXPECT_NE(dsparkUniformAt(seed, kDSparkTargetSeedNamespace, 3),
              dsparkUniformAt(seed, kDSparkTargetSeedNamespace, 4));
}

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
