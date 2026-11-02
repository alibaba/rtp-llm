#include "gtest/gtest.h"
#include "internal_source/rtp_llm/cpp/disaggregate/cache_store/RdmaClientHostConnections.h"
#include <atomic>
#include <future>
namespace rtp_llm {
namespace {
using namespace accl::barex;
class QuietConnection final: public XConnection {
public:
 void Close() override {}
 void Send(const XMessage&, DoneCallback, SendMode) override {}
 XSimpleMempool* GetMempool() override { return nullptr; }
 XChannel* GetChannel() override { return nullptr; }
 std::pair<std::string,int> GetPeerAddrAndPort() override { return {"127.0.0.1",1}; }
 std::pair<std::string,int> GetLocalAddrAndPort() override { return {"127.0.0.1",2}; }
};
class PendingTransport final: public XClientTransport {
public:
 std::promise<void> started;
 std::atomic<int> calls{0};
 void Connect(TcpProtoType,const std::string&,int,const std::shared_ptr<XConnectionCallback>&) override {
  if (calls.fetch_add(1)==0) started.set_value();
 }
 bool Disconnect(const std::shared_ptr<XConnection>&) override { return true; }
};
TEST(RdmaConnectionPoolRegression, ReusesReturnedLeaseWhileNewHandshakeIsPending) {
 auto pool=std::make_shared<RdmaClientHostConnections>(nullptr,"127.0.0.1",1,2,200);
 auto first=std::make_shared<QuietConnection>(); auto second=std::make_shared<QuietConnection>();
 pool->connectSuccess(first);pool->connectSuccess(second);
 auto held=pool->getRdmaConnectionFromConnectionPool(pool->newly_connections_,true);
 ASSERT_NE(held,nullptr);ASSERT_EQ(held->getAllConnections().size(),2u);
 auto transport=std::make_shared<PendingTransport>();auto started=transport->started.get_future();
 auto pending=std::async(std::launch::async,[pool,transport]{return pool->getConnection(transport);});
 started.wait();pool->recycleConnection(held);
 auto reused=pending.get();
 ASSERT_NE(reused,nullptr) << "a completed lease was available before the connect deadline";
 auto connections=reused->getAllConnections();EXPECT_EQ(connections.size(),2u);
 EXPECT_NE(std::find(connections.begin(),connections.end(),first),connections.end());
 EXPECT_NE(std::find(connections.begin(),connections.end(),second),connections.end());
 pool->recycleConnection(reused);
}
} // namespace
} // namespace rtp_llm
