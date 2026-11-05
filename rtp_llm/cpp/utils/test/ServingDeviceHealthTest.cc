#include "rtp_llm/cpp/utils/ServingDeviceHealth.h"

#include <gtest/gtest.h>
#include <new>
#include <stdexcept>
#include <tuple>

namespace rtp_llm {
namespace {

using Domain = DeviceRuntimeDomain;
using Op = DeviceRuntimeOperation;

struct RuntimeCase {
    const char* name;
    Domain domain;
    int code;
    int serving{0};
    int resource{0};
    int current{0};
    int selection_code{0};
    bool performed{true};
    bool failed{false};
};

const RuntimeCase kRuntimeCases[] = {
    {"cuda illegal address", Domain::CUDA, 700, 0, 0, 0, 0, true, true},
    {"cuda launch timeout", Domain::CUDA, 702, 0, 0, 0, 0, true, true},
    {"cuda assert", Domain::CUDA, 710, 0, 0, 0, 0, true, true},
    {"cuda hardware stack error", Domain::CUDA, 714, 0, 0, 0, 0, true, true},
    {"cuda illegal instruction", Domain::CUDA, 715, 0, 0, 0, 0, true, true},
    {"cuda misaligned address", Domain::CUDA, 716, 0, 0, 0, 0, true, true},
    {"cuda invalid address space", Domain::CUDA, 717, 0, 0, 0, 0, true, true},
    {"cuda invalid PC", Domain::CUDA, 718, 0, 0, 0, 0, true, true},
    {"cuda launch failure", Domain::CUDA, 719, 0, 0, 0, 0, true, true},
    {"hip illegal address", Domain::HIP, 700, 0, 0, 0, 0, true, true},
    {"hip assert", Domain::HIP, 710, 0, 0, 0, 0, true, true},
    {"hip launch failure", Domain::HIP, 719, 0, 0, 0, 0, true, true},
    {"hip unverified code 702", Domain::HIP, 702},
    {"hip unverified code 714", Domain::HIP, 714},
    {"hip unverified code 715", Domain::HIP, 715},
    {"hip unverified code 716", Domain::HIP, 716},
    {"hip unverified code 717", Domain::HIP, 717},
    {"hip unverified code 718", Domain::HIP, 718},
    {"success", Domain::CUDA, 0},
    {"cuda OOM", Domain::CUDA, 2},
    {"hip OOM", Domain::HIP, 2},
    {"invalid argument", Domain::CUDA, 1},
    {"not ready", Domain::CUDA, 600},
    {"launch resources", Domain::HIP, 701},
    {"cuda launch resources", Domain::CUDA, 701},
    {"cuda unknown error", Domain::CUDA, 999},
    {"hip unknown error", Domain::HIP, 999},
    {"non-runtime domain", Domain::UNKNOWN, 700},
    {"non-runtime domain with CUDA-only fatal code", Domain::UNKNOWN, 716},
    {"foreign device", Domain::CUDA, 700, 0, 1, 1},
    {"unknown serving device", Domain::HIP, 710, -1, 0, 0},
    {"wrong current device", Domain::CUDA, 700, 0, 0, 1, 0, false},
    {"unknown resource", Domain::HIP, 710, 0, -1, 0, 0, false},
    {"unknown current device", Domain::CUDA, 719, 0, 0, -1, 0, false},
    {"GetDevice fatal is not attribution", Domain::CUDA, 700, 0, 0, 0, 700, false},
    {"CUDA-only fatal on foreign device", Domain::CUDA, 716, 0, 1, 1},
    {"CUDA-only fatal with unknown serving device", Domain::CUDA, 716, -1, 0, 0},
    {"CUDA-only fatal with wrong current device", Domain::CUDA, 716, 0, 0, 1, 0, false},
    {"CUDA-only fatal with unknown resource", Domain::CUDA, 716, 0, -1, 0, 0, false},
    {"CUDA-only GetDevice fatal is not attribution", Domain::CUDA, 716, 0, 0, 0, 716, false},
};

class ServingDeviceOperationTest: public ::testing::TestWithParam<std::tuple<RuntimeCase, Op>> {};

TEST_P(ServingDeviceOperationTest, LatchesFailureOnlyForVerifiedNativeCompletion) {
    const auto& [test, operation] = GetParam();
    SCOPED_TRACE(test.name);
    ServingDeviceHealth health;
    int calls = 0;
    int32_t result = -1;
    const bool performed = runServingDeviceOperation(
        &health, test.domain, test.serving, test.resource, operation,
        [&](int& current) { current = test.current; return test.selection_code; },
        [&] { ++calls; return test.code; }, result);
    EXPECT_EQ(performed, test.performed);
    EXPECT_EQ(calls, int(test.performed));
    EXPECT_EQ(result, performed ? test.code : test.selection_code);
    EXPECT_EQ(health.healthy(), !test.failed);
    if (test.failed) {
        EXPECT_FALSE(health.observeRuntimeResult({test.domain, 0, test.resource, operation}));
        EXPECT_FALSE(health.healthy());
    }
}

INSTANTIATE_TEST_SUITE_P(Completion,
                        ServingDeviceOperationTest,
                        ::testing::Combine(::testing::ValuesIn(kRuntimeCases),
                                           ::testing::Values(Op::EVENT_WAIT, Op::STREAM_WAIT, Op::DEVICE_WAIT)));

TEST(ServingDeviceHealthTest, NonCompletionOperationCannotLatchHealth) {
    ServingDeviceHealth health;
    EXPECT_FALSE(health.observeRuntimeResult({Domain::CUDA, 700, 0, static_cast<Op>(255)}));
    EXPECT_TRUE(health.healthy());
}

class ServingDeviceExceptionTest: public ::testing::TestWithParam<bool> {};

TEST_P(ServingDeviceExceptionTest, OomAndNetworkFatalTextAreNotNativeResults) {
    ServingDeviceHealth health;
    int32_t result = 0;
    const auto run = [&] {
        runServingDeviceOperation(&health, Domain::CUDA, 0, 0, Op::EVENT_WAIT,
                                  [](int& current) { current = 0; return 0; },
                                  [&]() -> int {
                                      if (GetParam()) {
                                          throw std::bad_alloc();
                                      }
                                      throw std::runtime_error("network: CUDA error 700 illegal address");
                                  }, result);
    };
    if (GetParam()) {
        EXPECT_THROW(run(), std::bad_alloc);
    } else {
        EXPECT_THROW(run(), std::runtime_error);
    }
    EXPECT_TRUE(health.healthy());
}

INSTANTIATE_TEST_SUITE_P(Exceptions, ServingDeviceExceptionTest, ::testing::Bool());

}  // namespace
}  // namespace rtp_llm
