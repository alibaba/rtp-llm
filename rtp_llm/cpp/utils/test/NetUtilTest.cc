#include "gtest/gtest.h"
#include <algorithm>
#include <vector>
#include "rtp_llm/cpp/utils/NetUtil.h"

namespace rtp_llm {

class NetUtilTest: public ::testing::Test {
protected:
};

TEST_F(NetUtilTest, testGetAddressByName) {
    // localhost
    std::vector<std::string> ips;
    int                      err = getAddressByName("localhost", ips);
    ASSERT_TRUE(err == 0);
    ASSERT_FALSE(ips.empty());
    ASSERT_TRUE(std::find(ips.begin(), ips.end(), "127.0.0.1") != ips.end());

    // invaild host
    std::vector<std::string> ips2;
    err = getAddressByName("invaild-rtp-llm.dev", ips2);
    ASSERT_TRUE(err != 0);
    ASSERT_TRUE(ips2.empty());
}

}  // namespace rtp_llm