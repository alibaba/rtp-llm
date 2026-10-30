#include "aios/network/rdma/accl/Timer.h"

#include "aios/network/rdma/accl/TimerManager.h"
#include "autil/Log.h"
#include "unittest/unittest.h"

using namespace testing;

ARPC_BEGIN_NAMESPACE(arpc);

class TimerTest: public TESTBASE {
public:
    TimerTest() {}
    ~TimerTest() {}

public:
    void setUp() {}
    void tearDown() {}

private:
    AUTIL_LOG_DECLARE();
};

AUTIL_LOG_SETUP(arpc, TimerTest);

TEST_F(TimerTest, testOnThread) {
    // check timeout interval set to 5ms
    auto manager = std::make_shared<TimerManager>();

    std::string resultString;
    auto        callback = [&resultString](std::string key, int64_t startTime) {
        AUTIL_LOG(
            WARN, "call back %s after %ld", key.c_str(), autil::TimeUtility::currentTimeInMicroSeconds() - startTime);
        resultString.append(key);
    };

    auto startTime = autil::TimeUtility::currentTimeInMicroSeconds();
    auto timer1    = manager->addTimer(10, [callback, startTime]() { callback("1", startTime); });  // 100ms
    auto timer2    = manager->addTimer(50, [callback, startTime]() { callback("2", startTime); });  // 500ms
    auto timer3    = manager->addTimer(90, [callback, startTime]() { callback("3", startTime); });  // 900ms

    ASSERT_TRUE(timer1 != nullptr);
    ASSERT_TRUE(timer2 != nullptr);
    ASSERT_TRUE(timer3 != nullptr);

    usleep(30 * 1000);  // after 30 ms
    EXPECT_EQ(std::string("1"), resultString);

    usleep(40 * 1000);  // after 70 ms
    EXPECT_EQ(std::string("12"), resultString);

    timer3->stop();
    auto timer4 = manager->addTimer(40, std::bind(callback, "4", startTime));   // 110ms
    auto timer5 = manager->addTimer(100, std::bind(callback, "5", startTime));  // 170ms

    ASSERT_TRUE(timer4 != nullptr);
    ASSERT_TRUE(timer5 != nullptr);

    usleep(20 * 1000);  // after 90 ms
    EXPECT_EQ(std::string("12"), resultString);

    usleep(40 * 1000);  // after 130 ms
    EXPECT_EQ(std::string("124"), resultString);

    ASSERT_EQ(1, manager->getActiveTimerCount());
    manager.reset();
    EXPECT_EQ(std::string("124"), resultString);

    usleep(40 * 1000);  // after 170 ms
    EXPECT_EQ(std::string("124"), resultString);
}

TEST_F(TimerTest, testMultiThreadCall) {
    auto                          manager = std::make_shared<TimerManager>();
    std::atomic<int64_t>          timerCallbackCount(0);
    std::vector<autil::ThreadPtr> threads;
    {
        auto addTimerFunc = [manager, &timerCallbackCount] {
            std::vector<std::shared_ptr<Timer>> timers;
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 100; j++) {
                    auto callbackFunc = [&timerCallbackCount] { timerCallbackCount++; };
                    auto timer        = manager->addTimer(10, std::move(callbackFunc));
                    EXPECT_TRUE(timer != nullptr);
                    timers.push_back(timer);
                }
                // call stop on active timer
                for (int j = timers.size() - 1; j < 50; j++) {
                    if (timers[i]) {
                        timers[i]->stop();
                    }
                }
            }
            usleep(20000);  // wai timeout and call stop timer
            for (auto& timer : timers) {
                if (timer) {
                    timer->stop();
                }
            }
        };
        for (int i = 0; i < 100; i++) {
            auto thread = autil::Thread::createThread(addTimerFunc, "TimerTest");
            threads.push_back(thread);
        }
    }

    for (auto& thread : threads) {
        thread->join();
    }
    threads.clear();

    usleep(10 * 1000);  // wait all timer done, stop or timeout
    EXPECT_EQ(0, manager->getActiveTimerCount());
    manager.reset();
    std::cout << "stop callback " << timerCallbackCount.load() << std::endl;
}

ARPC_END_NAMESPACE(arpc);