#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include "rtp_llm/cpp/utils/test/DecodePublicationTestStartup.h"

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

TEST_F(DecodeBudgetConfigTest, PublicationSettingIsStrictAndReadOncePerProcess) {
    constexpr const char* key = "RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH";
    UnsetOverride restore(key);
    for (const char* initial : {static_cast<const char*>(nullptr), "0", "1", "", "true", " 1", "2"}) {
        SCOPED_TRACE(initial ? initial : "unset");
        if (initial) setenv(key, initial, 1);
        else unsetenv(key);
        ::rtp_llm::test::withDecodePublicationStartup([&] {
            const bool valid = !initial || std::string(initial) == "0" || std::string(initial) == "1";
            const bool enabled = initial && std::string(initial) == "1";
            // Exercise both getter orders; they must share one cached setting.
            if (enabled) {
                EXPECT_TRUE(remoteCacheDecodePublicationEnabled());
            }
            EXPECT_EQ(remoteCacheDecodePublicationConfigValid(), valid);
            EXPECT_EQ(remoteCacheDecodePublicationEnabled(), enabled);
            for (const char* later : {"0", "1", "invalid"}) {
                setenv(key, later, 1);
                EXPECT_EQ(remoteCacheDecodePublicationEnabled(), enabled);
                EXPECT_EQ(remoteCacheDecodePublicationConfigValid(), valid);
            }
            unsetenv(key);
            EXPECT_EQ(remoteCacheDecodePublicationEnabled(), enabled);
            EXPECT_EQ(remoteCacheDecodePublicationConfigValid(), valid);
        });
    }
}

TEST_F(DecodeBudgetConfigTest, UsesNativeCapacityWithoutRoundingSmallPoolsUp) {
    EXPECT_EQ(DecodePublicationLimits{}.bytes, 512 * MiB);
    EXPECT_EQ(DecodePublicationLimits{}.jobs, 4u);
    const auto defaults = remoteCacheDecodePublicationLimits();
    ASSERT_TRUE(defaults);
    EXPECT_EQ(defaults->bytes, 512 * MiB);
    EXPECT_EQ(defaults->jobs, 4u);
    EXPECT_FALSE(remoteCacheDecodePublicationLimits(0));
    for (auto [capacity, expected] : {std::pair<size_t, size_t>{4 * GiB, 512 * MiB},
                                      {8 * GiB, GiB},
                                      {24 * GiB, 3 * GiB},
                                      {48 * GiB, 4 * GiB},
                                      {64 * GiB, 4 * GiB},
                                      {8, 1},
                                      {8192, 1024},
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
