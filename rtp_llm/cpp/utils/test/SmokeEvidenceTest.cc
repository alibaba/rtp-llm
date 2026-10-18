#include "rtp_llm/cpp/utils/SmokeEvidence.h"
#include <gtest/gtest.h>
#include <cstdlib>

namespace rtp_llm {

TEST(SmokeEvidenceTest, OnlyExplicitOneEnablesStartupEvidence) {
    for (const char* value : {"", "0", "1", "true", "on", " 1", "1 "}) {
        const bool expected = std::strcmp(value, "1") == 0;
        EXPECT_EXIT(
            {
                setenv("RTP_LLM_SMOKE_EVIDENCE", value, 1);
                std::exit(smokeEvidenceEnabled() ? 1 : 0);
            },
            ::testing::ExitedWithCode(expected ? 1 : 0),
            "");
    }
    EXPECT_EXIT(
        {
            unsetenv("RTP_LLM_SMOKE_EVIDENCE");
            std::exit(smokeEvidenceEnabled() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST(SmokeEvidenceTest, StartupFlagIsNotChangedByLaterEnvironmentMutation) {
    EXPECT_EXIT(
        {
            setenv("RTP_LLM_SMOKE_EVIDENCE", "1", 1);
            const bool before = smokeEvidenceEnabled();
            setenv("RTP_LLM_SMOKE_EVIDENCE", "0", 1);
            std::exit(before && smokeEvidenceEnabled() ? 0 : 1);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST(SmokeEvidenceTest, TimestampUsesUnixNanosecondsForCrossRoleCorrelation) {
    const auto begin = std::chrono::system_clock::now();
    const auto event = smokeEvidenceTimeNs();
    const auto end   = std::chrono::system_clock::now();
    EXPECT_LE(std::chrono::duration_cast<std::chrono::nanoseconds>(begin.time_since_epoch()).count(), event);
    EXPECT_LE(event, std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count());
}

}  // namespace rtp_llm
