#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "kmonitor/client/KMonitorFactory.h"

#include <gtest/gtest.h>

#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rtp_llm {
namespace {

// 注册表与 factory 是进程级状态：每个用例先初始化，结束时注销本用例的 owner 并关闭 factory。
class KmonitorShutdownHookTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(initKmonitorFactory());
        ASSERT_TRUE(kmonitor::KMonitorFactory::IsStarted());
    }

    void TearDown() override {
        for (void* owner : owners_) {
            unregisterKmonitorShutdownHook(owner);
        }
        stopKmonitorFactory();
    }

    bool registerHook(void* owner, std::function<void()> hook) {
        owners_.push_back(owner);
        return registerKmonitorShutdownHook(owner, std::move(hook));
    }

    std::vector<void*> owners_;
};

TEST_F(KmonitorShutdownHookTest, StopRunsHooksBeforeFactoryShutdown) {
    int  owner                 = 0;
    int  calls                 = 0;
    bool factory_started_in_cb = false;
    ASSERT_TRUE(registerHook(&owner, [&] {
        ++calls;
        factory_started_in_cb = kmonitor::KMonitorFactory::IsStarted();
    }));

    stopKmonitorFactory();

    EXPECT_EQ(1, calls);
    EXPECT_TRUE(factory_started_in_cb);
    EXPECT_FALSE(kmonitor::KMonitorFactory::IsStarted());
}

TEST_F(KmonitorShutdownHookTest, UnregisteredHookIsNotRun) {
    int owner = 0;
    int calls = 0;
    ASSERT_TRUE(registerHook(&owner, [&] { ++calls; }));
    unregisterKmonitorShutdownHook(&owner);

    stopKmonitorFactory();

    EXPECT_EQ(0, calls);
}

TEST_F(KmonitorShutdownHookTest, RepeatedStopRunsRemainingHooksEachTime) {
    int owner = 0;
    int calls = 0;
    ASSERT_TRUE(registerHook(&owner, [&] { ++calls; }));

    stopKmonitorFactory();
    stopKmonitorFactory();

    EXPECT_EQ(2, calls);
}

TEST_F(KmonitorShutdownHookTest, RejectsRegistrationAfterStopUntilReinit) {
    int owner = 0;
    int calls = 0;
    stopKmonitorFactory();
    EXPECT_FALSE(registerHook(&owner, [&] { ++calls; }));

    ASSERT_TRUE(initKmonitorFactory());
    ASSERT_TRUE(registerHook(&owner, [&] { ++calls; }));
    stopKmonitorFactory();

    EXPECT_EQ(1, calls);
}

TEST_F(KmonitorShutdownHookTest, RejectsNullOwnerAndEmptyHook) {
    int owner = 0;
    EXPECT_FALSE(registerKmonitorShutdownHook(nullptr, [] {}));
    EXPECT_FALSE(registerHook(&owner, std::function<void()>()));
}

TEST_F(KmonitorShutdownHookTest, DuplicateOwnerKeepsExistingHook) {
    int owner          = 0;
    int existing_calls = 0;
    int new_calls      = 0;
    ASSERT_TRUE(registerHook(&owner, [&] { ++existing_calls; }));
    EXPECT_FALSE(registerHook(&owner, [&] { ++new_calls; }));

    stopKmonitorFactory();

    EXPECT_EQ(1, existing_calls);
    EXPECT_EQ(0, new_calls);
}

TEST_F(KmonitorShutdownHookTest, ThrowingHookDoesNotSkipOtherHooksOrFactoryShutdown) {
    // 回调按 owner 地址升序执行；用同一数组内的地址固定顺序，确保正常回调排在两个抛异常的回调之后。
    int owners[3]    = {};
    int normal_calls = 0;
    ASSERT_TRUE(registerHook(&owners[0], [] { throw std::runtime_error("hook failure"); }));
    ASSERT_TRUE(registerHook(&owners[1], [] { throw 1; }));
    ASSERT_TRUE(registerHook(&owners[2], [&] { ++normal_calls; }));

    EXPECT_NO_THROW(stopKmonitorFactory());

    EXPECT_EQ(1, normal_calls);
    EXPECT_FALSE(kmonitor::KMonitorFactory::IsStarted());
}

}  // namespace
}  // namespace rtp_llm
