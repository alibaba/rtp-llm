
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/utils/Cm2Config.h"

using namespace std;

namespace rtp_llm {

class Cm2ConfigTest: public ::testing::Test {
protected:
};

TEST_F(Cm2ConfigTest, testSimple) {
    string           cm2_config_str = R"({"cm2_server_cluster_name":"com.aicheng.whale.pre.test_seperate7.decode",
            "cm2_server_leader_path":"/cm_server_common",
            "cm2_server_zookeeper_host":"search-zk-cm2-ea120.vip.tbsite.net:2187",
            "httpPort":31640,"tcpPort":31641})";
    Cm2ClusterConfig cluster_config;
    FromJsonString(cluster_config, cm2_config_str);
    ASSERT_EQ(cluster_config.cluster_name, "com.aicheng.whale.pre.test_seperate7.decode");
    ASSERT_EQ(cluster_config.zk_host, "search-zk-cm2-ea120.vip.tbsite.net:2187");
    ASSERT_EQ(cluster_config.zk_path, "/cm_server_common");
}

}  // namespace rtp_llm
