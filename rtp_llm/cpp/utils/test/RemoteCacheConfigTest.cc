#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace rtp_llm {
namespace {

struct ScopedEnvironment {
    const char*                name;
    std::optional<std::string> previous;
    explicit ScopedEnvironment(const char* key): name(key) {
        if (const auto* value = std::getenv(key)) {
            previous = value;
        }
        unsetenv(key);
    }
    ~ScopedEnvironment() {
        if (previous) {
            setenv(name, previous->c_str(), 1);
        } else {
            unsetenv(name);
        }
    }
};

class DecodePublicationConfigTest: public ::testing::Test {
protected:
    ScopedEnvironment enabled{"RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH"};
    ScopedEnvironment bytes{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB"};
    ScopedEnvironment jobs{"RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS"};
    static constexpr size_t MiB = 1024 * 1024, GiB = 1024 * MiB;
};

TEST_F(DecodePublicationConfigTest, DisabledUnlessExplicitlyEnabled) {
    EXPECT_FALSE(remoteCacheDecodePublicationEnabled());
    for (const auto* value : {"", "0", "true", "ON", " 1", "1 ", "01", "2"}) {
        setenv(enabled.name, value, 1);
        EXPECT_FALSE(remoteCacheDecodePublicationEnabled()) << value;
    }
    setenv(enabled.name, "1", 1);
    EXPECT_TRUE(remoteCacheDecodePublicationEnabled());
}

TEST_F(DecodePublicationConfigTest, InvalidUnusedBudgetsDoNotEnableFeature) {
    setenv(bytes.name, "invalid", 1);
    setenv(jobs.name, "invalid", 1);
    EXPECT_FALSE(remoteCacheDecodePublicationEnabled());
}

TEST_F(DecodePublicationConfigTest, DerivesBudgetFromActualUniquePhysicalPool) {
    EXPECT_FALSE(remoteCacheDecodePublicationLimits(0));
    for (auto [capacity, expected] : {std::pair<size_t, size_t>{4 * GiB, 512 * MiB},
                                     {8 * GiB, GiB},
                                     {24 * GiB, 3 * GiB},
                                     {48 * GiB, 4 * GiB},
                                     {1024, 128},
                                     {1, 1},
                                     {std::numeric_limits<size_t>::max(), 4 * GiB}}) {
        const auto limits = remoteCacheDecodePublicationLimits(capacity);
        ASSERT_TRUE(limits) << capacity;
        EXPECT_EQ(limits->bytes, expected) << capacity;
        EXPECT_EQ(limits->jobs, 4u);
    }
}

TEST_F(DecodePublicationConfigTest, AcceptsExplicitResourceBudgets) {
    setenv(bytes.name, "256", 1);
    setenv(jobs.name, "8", 1);
    const auto limits = remoteCacheDecodePublicationLimits(4 * GiB);
    ASSERT_TRUE(limits);
    EXPECT_EQ(limits->bytes, 256 * MiB);
    EXPECT_EQ(limits->jobs, 8u);
    setenv(bytes.name, "16384", 1);
    setenv(jobs.name, "64", 1);
    const auto maximum = remoteCacheDecodePublicationLimits(16 * GiB);
    ASSERT_TRUE(maximum);
    EXPECT_EQ(maximum->bytes, 16 * GiB);
    EXPECT_EQ(maximum->jobs, 64u);
}

TEST_F(DecodePublicationConfigTest, RejectsMalformedOrUnboundedByteBudgets) {
    for (const auto* invalid : {"", "0", "-1", "+1", "1.5", " 8", "8 ", "8x", "18446744073709551616", "16385"}) {
        setenv(bytes.name, invalid, 1);
        EXPECT_FALSE(remoteCacheDecodePublicationLimits(48 * GiB)) << invalid;
    }
}

TEST_F(DecodePublicationConfigTest, RejectsBudgetLargerThanActualPool) {
    setenv(bytes.name, "4097", 1);
    EXPECT_FALSE(remoteCacheDecodePublicationLimits(4 * GiB));
    setenv(bytes.name, "1", 1);
    EXPECT_FALSE(remoteCacheDecodePublicationLimits(1024));
}

TEST_F(DecodePublicationConfigTest, RejectsMalformedOrUnboundedJobBudgets) {
    for (const auto* invalid : {"", "0", "-1", "+1", "1.5", " 8", "8 ", "8x", "18446744073709551616", "65"}) {
        setenv(jobs.name, invalid, 1);
        EXPECT_FALSE(remoteCacheDecodePublicationLimits(4 * GiB)) << invalid;
    }
}

}  // namespace
}  // namespace rtp_llm
