#include "aios/network/rdma/accl/RdmaMempoolFactory.h"

#include "unittest/unittest.h"

namespace arpc {

TEST(RdmaMempoolFactoryTest, testInitAndShutdown) {
    // not inited
    ASSERT_TRUE(RdmaMempoolFactory::GetMempool() == nullptr);

    // init
    ASSERT_TRUE(RdmaMempoolFactory::Init());
    ASSERT_TRUE(RdmaMempoolFactory::GetMempool() != nullptr);

    // double init
    ASSERT_TRUE(RdmaMempoolFactory::Init());
    ASSERT_TRUE(RdmaMempoolFactory::GetMempool() != nullptr);

    // shutdown call once, not shutdown
    RdmaMempoolFactory::Shutdown();
    ASSERT_TRUE(RdmaMempoolFactory::GetMempool() != nullptr);

    // shutdown times equal init times, shutdown
    RdmaMempoolFactory::Shutdown();
    ASSERT_TRUE(RdmaMempoolFactory::GetMempool() == nullptr);
}

TEST(RdmaMempoolFactoryTest, testStop) {
    // stop on null mempool
    RdmaMempoolFactory::Stop();

    // stop on real mempool
    ASSERT_TRUE(RdmaMempoolFactory::Init());
    ASSERT_TRUE(RdmaMempoolFactory::GetMempool() != nullptr);

    RdmaMempoolFactory::Stop();

    RdmaMempoolFactory::Shutdown();
    ASSERT_TRUE(RdmaMempoolFactory::GetMempool() == nullptr);
}

}  // namespace arpc