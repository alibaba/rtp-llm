#include "rtp_llm/cpp/normal_engine/pipeline/PPPrefillGuardPolicy.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

struct GuardCase {
    bool        reject;
    const char* name;
    bool        compat_enabled;
    RoleType    role;
    bool        pd_separation;
    bool        fake;
    bool        perf;
    int64_t     max_new_tokens;
};

constexpr GuardCase kGuardCases[] = {
    {false, "CompatOffPdfusionMultiToken", false, RoleType::PDFUSION, false, false, false, 10},
    {false, "CompatOffPrefillLocalMultiToken", false, RoleType::PREFILL, false, false, false, 10},
    {false, "PdfusionOneToken", true, RoleType::PDFUSION, false, false, false, 1},
    {true, "PdfusionMultiToken", true, RoleType::PDFUSION, false, false, false, 10},
    {true, "PdfusionInjectedPdFlag", true, RoleType::PDFUSION, true, false, false, 10},
    {false, "PrefillLocalOneToken", true, RoleType::PREFILL, false, false, false, 1},
    {true, "PrefillLocalMultiToken", true, RoleType::PREFILL, false, false, false, 10},
    {false, "PrefillGenuinePdMultiToken", true, RoleType::PREFILL, true, false, false, 10},
    {true, "DecodeMultiToken", true, RoleType::DECODE, false, false, false, 10},
    {true, "DecodeInjectedPdFlag", true, RoleType::DECODE, true, false, false, 10},
    {false, "FakeStreamExempt", true, RoleType::PDFUSION, false, true, false, 10},
    {false, "PerfTestStreamExempt", true, RoleType::PDFUSION, false, false, true, 10},
    {false, "PrefillFakeStreamExempt", true, RoleType::PREFILL, false, true, false, 10},
};

class PPPrefillGuardPolicyTest: public testing::TestWithParam<GuardCase> {};

TEST_P(PPPrefillGuardPolicyTest, EnforcesPerRequestPolicy) {
    const auto& test_case = GetParam();
    // Exercise the production predicate without linking an engine or device.
    EXPECT_EQ(test_case.reject,
              dsv4PrefillCpGuardRejects(test_case.compat_enabled,
                                        test_case.role,
                                        test_case.pd_separation,
                                        test_case.fake,
                                        test_case.perf,
                                        test_case.max_new_tokens));
}

INSTANTIATE_TEST_SUITE_P(Profiles,
                         PPPrefillGuardPolicyTest,
                         testing::ValuesIn(kGuardCases),
                         [](const testing::TestParamInfo<GuardCase>& info) { return info.param.name; });

}  // namespace
}  // namespace rtp_llm
