#include "gtest/gtest.h"
#include "internal_source/rtp_llm/cpp/disaggregate/cache_store/RdmaClientHostConnections.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection_callback.h"
#include <atomic>
#include <chrono>
#include <future>

namespace rtp_llm { namespace {
using namespace accl::barex;
class RetryQuietConnection final: public XConnection {
public:
 void Close() override {}
 void Send(const XMessage&,DoneCallback,SendMode) override {}
 XSimpleMempool* GetMempool() override { return nullptr; }
 XChannel* GetChannel() override { return nullptr; }
 std::pair<std::string,int> GetPeerAddrAndPort() override { return {"127.0.0.1",1}; }
 std::pair<std::string,int> GetLocalAddrAndPort() override { return {"127.0.0.1",2}; }
};
class RetryFaultTransport final: public XClientTransport {
public:
 explicit RetryFaultTransport(int failures, bool never_complete=false):failures_(failures),pending_(never_complete) {}
 std::atomic<int> calls{0};
 void Connect(TcpProtoType,const std::string&,int,const std::shared_ptr<XConnectionCallback>& callback) override {
  const int number = calls.fetch_add(1);
  auto c=std::make_shared<RetryQuietConnection>();
  if (pending_) return;
  if (number < failures_) callback->OnConnectFailed(c);
  else callback->OnConnectSuccess(c);
 }
 bool Disconnect(const std::shared_ptr<XConnection>&) override { return true; }
private:
 int failures_; bool pending_;
};

TEST(K31TerminalHandshakeRetry, RecoversFirstTerminalBatchWithinSameAcquire) {
 auto pool=std::make_shared<RdmaClientHostConnections>(nullptr,"127.0.0.1",1,2,1800);
 auto transport=std::make_shared<RetryFaultTransport>(2);
 auto started=std::chrono::steady_clock::now();
 auto lease=pool->getConnection(transport);
 auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-started).count();
 ASSERT_NE(lease,nullptr); EXPECT_EQ(lease->getConnectionsCount(),2); EXPECT_EQ(transport->calls.load(),4);
 EXPECT_GE(ms,500); EXPECT_LT(ms,1500); pool->recycleConnection(lease);
}
TEST(K31TerminalHandshakeRetry, PendingHandshakesAreNeverDuplicated) {
 auto pool=std::make_shared<RdmaClientHostConnections>(nullptr,"127.0.0.1",1,2,700);
 auto transport=std::make_shared<RetryFaultTransport>(0,true);
 auto lease=pool->getConnection(transport);
 EXPECT_EQ(lease,nullptr); EXPECT_EQ(transport->calls.load(),2);
}
TEST(K31TerminalHandshakeRetry, PermanentFailurePreservesAbsoluteBudgetAndBackoff) {
 auto pool=std::make_shared<RdmaClientHostConnections>(nullptr,"127.0.0.1",1,2,1600);
 auto transport=std::make_shared<RetryFaultTransport>(1000);
 auto started=std::chrono::steady_clock::now();
 auto lease=pool->getConnection(transport);
 auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-started).count();
 EXPECT_EQ(lease,nullptr); EXPECT_GE(transport->calls.load(),6); EXPECT_LE(transport->calls.load(),8);
 EXPECT_GE(ms,1550); EXPECT_LT(ms,1900);
}
TEST(K31TerminalHandshakeRetry, ShortBudgetDoesNotStartANewBatch) {
 auto pool=std::make_shared<RdmaClientHostConnections>(nullptr,"127.0.0.1",1,2,80);
 auto transport=std::make_shared<RetryFaultTransport>(1000);
 EXPECT_EQ(pool->getConnection(transport),nullptr); EXPECT_EQ(transport->calls.load(),2);
}
TEST(K31TerminalHandshakeRetry, SixtyFourColdAcquiresRecoverWithoutASecondCallerWave) {
 auto pool=std::make_shared<RdmaClientHostConnections>(nullptr,"127.0.0.1",1,2,1800);
 auto transport=std::make_shared<RetryFaultTransport>(128);
 std::vector<std::future<std::shared_ptr<RdmaConnection>>> futures;
 for(int i=0;i<64;++i)futures.emplace_back(std::async(std::launch::async,[&]{return pool->getConnection(transport);}));
 std::vector<std::shared_ptr<RdmaConnection>> leases;
 for(auto& future:futures){auto lease=future.get();EXPECT_NE(lease,nullptr);if(lease)leases.push_back(lease);}
 EXPECT_EQ(leases.size(),64u);EXPECT_LE(transport->calls.load(),384);
 for(auto& lease:leases)pool->recycleConnection(lease);
}
} }

namespace rtp_llm { namespace {
class OnePartialHandshakeTransport final: public XClientTransport {
public:
 int calls = 0;
 void Connect(TcpProtoType,const std::string&,int,const std::shared_ptr<XConnectionCallback>& callback) override {
  auto connection=std::make_shared<RetryQuietConnection>();
  if(calls++==0)callback->OnConnectSuccess(connection);
  else callback->OnConnectFailed(connection);
 }
 bool Disconnect(const std::shared_ptr<XConnection>&) override { return true; }
};
TEST(K31TerminalHandshakeRetry, PartialSuccessRemainsUsableAtDeadline) {
 auto pool=std::make_shared<RdmaClientHostConnections>(nullptr,"127.0.0.1",1,2,80);
 auto transport=std::make_shared<OnePartialHandshakeTransport>();
 auto lease=pool->getConnection(transport);
 ASSERT_NE(lease,nullptr);EXPECT_EQ(lease->getConnectionsCount(),1);EXPECT_EQ(transport->calls,2);
 pool->recycleConnection(lease);
}
} }

namespace rtp_llm { namespace {
TEST(K31TerminalHandshakeRetry, RecoversAfterThreeFailedBatchesWithinOriginalDeadline) {
 auto pool=std::make_shared<RdmaClientHostConnections>(nullptr,"127.0.0.1",1,2,2600);
 auto transport=std::make_shared<RetryFaultTransport>(6);
 const auto started=std::chrono::steady_clock::now();
 auto lease=pool->getConnection(transport);
 const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-started).count();
 ASSERT_NE(lease,nullptr);EXPECT_EQ(lease->getConnectionsCount(),2);
 EXPECT_EQ(transport->calls.load(),8);EXPECT_GE(ms,1500);EXPECT_LT(ms,2500);
 pool->recycleConnection(lease);
}
} }
