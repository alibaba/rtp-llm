#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"

#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

class DecodeBudgetConfigTest: public ::testing::Test {
protected:
    struct UnsetOverride {
        const char*                name;
        std::optional<std::string> previous;
        explicit UnsetOverride(const char* key): name(key) {
            if (const auto* value = std::getenv(key))
                previous = value;
            unsetenv(key);
        }
        ~UnsetOverride() {
            if (previous)
                setenv(name, previous->c_str(), 1);
            else
                unsetenv(name);
        }
    } bytes{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB"}, jobs{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS"};
    static constexpr size_t MiB = 1024 * 1024, GiB = 1024 * MiB;
};

TEST_F(DecodeBudgetConfigTest, UsesNativeCapacityWithoutRoundingSmallPoolsUp) {
    EXPECT_FALSE(remoteCacheDecodePublicationLimits(0));
    for (auto [capacity, expected] : {std::pair<size_t, size_t>{4 * GiB, 512 * MiB},
                                      {8 * GiB, GiB},
                                      {24 * GiB, 3 * GiB},
                                      {48 * GiB, 4 * GiB},
                                      {1024, 128},
                                      {std::numeric_limits<size_t>::max(), 4 * GiB}}) {
        const auto limits = remoteCacheDecodePublicationLimits(capacity);
        ASSERT_TRUE(limits) << capacity;
        EXPECT_EQ(limits->bytes, expected) << capacity;
        EXPECT_EQ(limits->jobs, 4u);
    }
}

TEST_F(DecodeBudgetConfigTest, StrictExplicitBudgetsRemainAvailable) {
    setenv(bytes.name, "256", 1);
    setenv(jobs.name, "8", 1);
    auto limits = remoteCacheDecodePublicationLimits(4 * GiB);
    ASSERT_TRUE(limits);
    EXPECT_EQ(limits->bytes, 256 * MiB);
    EXPECT_EQ(limits->jobs, 8u);
    for (const auto* invalid : {"", "0", "-1", "1.5", " 8", "8x", "18446744073709551616", "16385"}) {
        setenv(bytes.name, invalid, 1);
        EXPECT_FALSE(remoteCacheDecodePublicationLimits(4 * GiB)) << invalid;
    }
    unsetenv(bytes.name);
    setenv(jobs.name, "65", 1);
    EXPECT_FALSE(remoteCacheDecodePublicationLimits(4 * GiB));
}

}  // namespace
}  // namespace rtp_llm
