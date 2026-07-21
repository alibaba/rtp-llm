#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceFullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

class DeviceKVCacheGroupTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// ==================== Basic functionality tests ====================

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
