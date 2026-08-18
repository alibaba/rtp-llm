#include <cstddef>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/normal_engine/WarmupRoleGate.h"

namespace rtp_llm {
namespace {

// The role classification decides how much KV cache a whole role class allocates, so it is pinned
// per enum value rather than by spot-checking. ROLE_TYPE_COUNT makes adding a sixth role break this
// test until its expected warmup and trust behavior is recorded here.
TEST(WarmupRoleGateTest, PinsEveryRoleValue) {
    static_assert(static_cast<int>(RoleType::ROLE_TYPE_COUNT) == 5,
                  "update the warmup role classification when adding a RoleType");
    struct Expectation {
        RoleType role;
        bool     warmup_role;
        bool     measurement_trusted;
    };
    // PDFUSION runs the forward (lazy-init timing + post-forward sample) but discards the
    // measurement; PREFILL/DECODE both run it and use it; VIT/FRONTEND have no warmup phase.
    const Expectation expectations[] = {
        {RoleType::PDFUSION, true, false},
        {RoleType::PREFILL, true, true},
        {RoleType::DECODE, true, true},
        {RoleType::VIT, false, false},
        {RoleType::FRONTEND, false, false},
    };
    static_assert(sizeof(expectations) / sizeof(expectations[0]) == static_cast<std::size_t>(RoleType::ROLE_TYPE_COUNT),
                  "every RoleType must have a warmup classification expectation");

    bool seen[static_cast<std::size_t>(RoleType::ROLE_TYPE_COUNT)] = {};
    int  untrusted_warmup_roles                                    = 0;
    for (const auto& e : expectations) {
        const auto index = static_cast<std::size_t>(e.role);
        SCOPED_TRACE(index);
        ASSERT_LT(index, static_cast<std::size_t>(RoleType::ROLE_TYPE_COUNT));
        EXPECT_FALSE(seen[index]) << "duplicate RoleType expectation";
        seen[index] = true;
        EXPECT_EQ(isWarmupRole(e.role), e.warmup_role);
        EXPECT_EQ(warmupMeasurementTrustedForRole(e.role), e.measurement_trusted);
        if (e.warmup_role && !e.measurement_trusted) {
            ++untrusted_warmup_roles;
            EXPECT_EQ(e.role, RoleType::PDFUSION);
        }
        if (e.measurement_trusted) {
            EXPECT_TRUE(e.warmup_role);
        }
    }
    for (std::size_t index = 0; index < static_cast<std::size_t>(RoleType::ROLE_TYPE_COUNT); ++index) {
        EXPECT_TRUE(seen[index]) << "missing RoleType expectation for enum value " << index;
    }
    EXPECT_EQ(untrusted_warmup_roles, 1);
}

}  // namespace
}  // namespace rtp_llm
