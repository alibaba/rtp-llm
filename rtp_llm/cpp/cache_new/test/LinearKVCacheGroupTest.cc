#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "rtp_llm/cpp/cache_new/LinearKVCacheGroup.h"

namespace rtp_llm {
namespace test {

class LinearKVCacheGroupTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// ==================== 基础功能测试 ====================

TEST_F(LinearKVCacheGroupTest, AllocateTest) {
    // 测试构造函数
    // LinearKVCacheGroup group1;
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
