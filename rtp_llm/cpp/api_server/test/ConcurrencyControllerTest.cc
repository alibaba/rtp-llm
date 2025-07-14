#include <chrono>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/api_server/ConcurrencyControllerUtil.h"

namespace rtp_llm {

class ConcurrencyControllerTest: public ::testing::Test {};

TEST_F(ConcurrencyControllerTest, testSimple) {
    ConcurrencyController controller(2);
    ASSERT_EQ(controller.get_available_concurrency(), 2);

    ASSERT_TRUE(controller.increment());
    ASSERT_TRUE(controller.increment());
    ASSERT_FALSE(controller.increment());

    controller.decrement();
    ASSERT_EQ(controller.get_available_concurrency(), 1);
    controller.decrement();
    ASSERT_EQ(controller.get_available_concurrency(), 2);
}

TEST_F(ConcurrencyControllerTest, testBlocking) {
    ConcurrencyController controller(2, true);
    controller.increment();
    controller.increment();

    int         flag = 0;
    std::thread t([&controller, &flag] {
        controller.increment();
        flag = 1;
    });
    std::this_thread::sleep_for(std::chrono::seconds(1));
    ASSERT_EQ(flag, 0);

    controller.decrement();
    t.join();
    ASSERT_EQ(flag, 1);
}

}  // namespace rtp_llm
