#include "aios/navi_sched/resource/TairKvcmGrpcClientR.h"
#include "aios/navi_sched/scheduler/TairKvcmGrpcProtocol.h"
#include "aios/navi_sched/test/TairKvcmGrpcTestServer.h"
#include "aios/navi_sched/test/TairKvcmGigTestClient.h"
#include "autil/TimeUtility.h"
#include "unittest/unittest.h"
#include "navi/tester/NaviResourceHelper.h"
#include "aios/network/gig/multi_call/subscribe/SubscribeServiceManager.h"
#include "aios/network/gig/multi_call/subscribe/VIPSubscribeService.h"
#include "autil/Scope.h"
#include <condition_variable>
#include <deque>

namespace navi_sched {
namespace {
namespace wire = kv_cache_manager::proto::meta;
const auto &LEADER = TAIR_KVCM_GRPC_GET_CLUSTER_INFO;
const auto &LOOKUP = TAIR_KVCM_GRPC_GET_REMOTE_LOCATIONS;
class CompletionPipe : public navi::AsyncPipe {
public:
    CompletionPipe() : navi::AsyncPipe(nullptr, navi::AS_INDEPENDENT) {}
    navi::ErrorCode setData(const navi::DataPtr &data) override {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push_back(std::dynamic_pointer_cast<TairKvcmGrpcCompletionData>(data));
        condition.notify_all(); return navi::EC_NONE;
    }
    std::shared_ptr<TairKvcmGrpcCompletionData> wait() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!condition.wait_for(lock, std::chrono::seconds(3), [&] { return !queue.empty(); })) return {};
        auto result = queue.front(); queue.pop_front(); lock.unlock();
        if (result) result->unpack(); // Same caller-thread boundary as Navi compute.
        return result;
    }
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::shared_ptr<TairKvcmGrpcCompletionData>> queue;
};
TairKvcmGrpcClientR::MessagePtr payload(const std::string &trace, bool discovery = false)
{
    if (discovery) return makeTairKvcmGrpcLeaderMessage(trace);
    auto request = std::make_shared<wire::GetCacheLocationsByBackendRequest>();
    request->set_trace_id(trace); request->set_instance_id("isolated_unary_test");
    request->set_query_type(wire::QT_BATCH_GET);
    return request;
}
test::KvcmGrpcMock::Reply respond(const std::string &method, const std::string &body)
{
    std::string trace;
    if (method == LEADER) {
        wire::GetClusterInfoRequest request;
        if (!request.ParseFromString(body)) throw std::runtime_error("invalid leader request");
        trace = request.trace_id();
    } else {
        wire::GetCacheLocationsByBackendRequest request;
        if (!request.ParseFromString(body)) throw std::runtime_error("invalid lookup request");
        trace = request.trace_id();
    }
    if (trace == "error") return {{}, grpc::Status(grpc::StatusCode::UNAVAILABLE, "synthetic")};
    if (trace == "hold") return {{}, grpc::Status::OK, true};
    wire::GetCacheLocationsByBackendResponse reply;
    reply.mutable_header()->mutable_status()->set_code(wire::OK);
    reply.mutable_header()->mutable_status()->set_message(trace == "oversize" ? std::string(1024, 'x') : trace);
    return {reply.SerializeAsString(), grpc::Status::OK};
}
std::string traceReply(const TairKvcmGrpcCompletionData &done)
{
    wire::GetCacheLocationsByBackendResponse reply;
    if (!reply.ParseFromString(test::bytes(done.response))) throw std::runtime_error("invalid test reply");
    return reply.header().status().message();
}
int64_t deadline() { return autil::TimeUtility::monotonicTimeUs() + 2000000; }
}
class TairKvcmGrpcClientTest : public TESTBASE {};

TEST_F(TairKvcmGrpcClientTest, resourceJsonSetsThreadCountAndRejectsZero)
{
    navi::NaviResourceHelper helper;
    ASSERT_TRUE(helper.snapshotConfig(navi::GigClientR::RESOURCE_ID,
        R"({"init_config":{"connection_config":{"grpc":{"thread_num":2}}}})"));
    ASSERT_TRUE(helper.config(TairKvcmGrpcClientR::RESOURCE_ID,
        R"({"address":"127.0.0.1","port":6381,"discovery":{"type":"direct"},"thread_num":2})"));
    auto client = helper.getOrCreateResPtr<TairKvcmGrpcClientR>(); ASSERT_TRUE(client);
    EXPECT_EQ(2u, client->_topology.threadNum);
    EXPECT_EQ((std::vector<std::string>{"127.0.0.1:6381"}), client->bootstrapTargets());
    auto pool = test::unaryPool(*client); ASSERT_TRUE(pool);
    EXPECT_EQ(2u, pool->_worker->_completionQueues.size()); EXPECT_TRUE(pool->_connectionMap.empty());
    navi::NaviResourceHelper invalid;
    ASSERT_TRUE(invalid.config(TairKvcmGrpcClientR::RESOURCE_ID, R"({"thread_num":0})"));
    EXPECT_FALSE(invalid.getOrCreateResPtr<TairKvcmGrpcClientR>());
}

TEST_F(TairKvcmGrpcClientTest, spectrumBootstrapReadsSnapshotIpsAndUsesConfiguredPort)
{
    navi::GigClientR gig;
    auto service = std::make_shared<multi_call::SearchService>();
    auto manager = std::make_shared<multi_call::SearchServiceManager>();
    auto snapshot = std::make_shared<multi_call::SearchServiceSnapshot>(
        multi_call::ConnectionManagerPtr(), multi_call::MetricReporterManagerPtr(), std::make_shared<multi_call::MiscConfig>());
    const auto add = [&](const std::string &id, const std::string &biz, const std::string &ip) {
        multi_call::TopoNode node; node.bizName = biz; node.clusterName = biz; node.serverBizName = "unrelated_server_biz";
        node.spec.ip = ip; node.spec.ports[multi_call::MC_PROTOCOL_HTTP] = 6382;
        snapshot->_providerMap[id] = {std::make_shared<multi_call::SearchServiceProvider>(multi_call::ConnectionManagerPtr(), node), nullptr};
    };
    add("kvcm1", "v-kvcm", "127.0.0.2"); add("kvcm2", "v-kvcm", "127.0.0.2"); add("worker", "v-worker", "127.0.0.3");
    manager->_searchServiceSnapshotPtr = snapshot; service->_serviceManager = manager; gig._searchService = service;
    TairKvcmGrpcClientR client; client._gigClientR = &gig; client._subscriptionReady = true;
    client._topology.address = "v-kvcm"; client._topology.port = 16381;
    EXPECT_EQ((std::vector<std::string>{"127.0.0.2:16381"}), client.bootstrapTargets());
    EXPECT_FALSE(client._transportReady);
    client._subscriptionReady = false; EXPECT_TRUE(client.bootstrapTargets().empty());
}

TEST_F(TairKvcmGrpcClientTest, allFlexlbDiscoveryTypesBuildNonHeartbeatGigSubscriptions)
{
    TairKvcmGrpcClientR client;
    client._topology.address = "kvcm-test"; client._topology.port = 16381;
    multi_call::MultiCallConfig parent;
    parent.subConf.vipConfig.jmenvDom = "existing-platform-jmenv";
    parent.subConf.vipConfig.clusters = {"existing-worker-domain"};
    parent.subConf.vipConfig.supportHeartbeat = true;
    for (const std::string type : {"dashscope", "spectrum", "vipserver", "static-env"}) {
        client._topology.discovery.type = type;
        client._topology.discovery.hosts = {"127.0.0.2:6382", "127.0.0.2:9000", "127.0.0.1:6382"};
        ASSERT_TRUE(client._topology.valid()); multi_call::SubscribeConfig subscription;
        ASSERT_TRUE(client.makeSubscription(parent, subscription));
        if (type == "dashscope" || type == "spectrum") {
            EXPECT_EQ("kvcm-test", subscription.spectrumConfig.serviceId);
            EXPECT_FALSE(subscription.spectrumConfig.supportHeartbeat);
        } else if (type == "vipserver") {
            EXPECT_EQ("existing-platform-jmenv", subscription.vipConfig.jmenvDom);
            EXPECT_EQ((std::vector<std::string>{"kvcm-test"}), subscription.vipConfig.clusters);
            EXPECT_FALSE(subscription.vipConfig.supportHeartbeat);
        } else {
            ASSERT_EQ(2u, subscription.localConfig.nodes.size());
            for (const auto &node : subscription.localConfig.nodes) {
                EXPECT_EQ("kvcm-test", node._clusterName); EXPECT_FALSE(node._supportHeartbeat);
                EXPECT_EQ(16381, node._grpcPort);
                EXPECT_TRUE(node.validate());
            }
        }
    }
    EXPECT_EQ((std::vector<std::string>{"existing-worker-domain"}), parent.subConf.vipConfig.clusters);
    EXPECT_TRUE(parent.subConf.vipConfig.supportHeartbeat);
    client._topology.discovery.type = "vipserver";
    parent.subConf.vipConfig.jmenvDom.clear(); multi_call::SubscribeConfig subscription;
    EXPECT_FALSE(client.makeSubscription(parent, subscription));
    client._topology.discovery.jmenvDom = "resource-platform-jmenv";
    ASSERT_TRUE(client.makeSubscription(parent, subscription));
    EXPECT_EQ("resource-platform-jmenv", subscription.vipConfig.jmenvDom);
    parent.subConf.vipConfig.jmenvDom = "different-existing-environment";
    EXPECT_FALSE(client.makeSubscription(parent, subscription));
}

TEST_F(TairKvcmGrpcClientTest, staticEnvIsActuallyPublishedThroughGigAndDoesNotContactBackends)
{
    auto client = test::gigQueryClient(16381); const auto service = client->_gigClientR->getSearchService();
    client->_topology.address = "kvcm-static"; client->_topology.discovery.type = "static-env";
    client->_topology.discovery.hosts = {"127.0.0.1:6382", "127.0.0.1:6382", "127.0.0.2:6382"};
    struct Ready { std::mutex mutex; std::condition_variable condition; bool value = false; };
    auto ready = std::make_shared<Ready>(); std::weak_ptr<multi_call::SearchService> weak = service;
    const auto listener = service->addSnapshotChangeListener([weak, ready] {
        auto service = weak.lock(); if (!service) return;
        const auto snapshot = service->getSearchServiceManager()->getSearchServiceSnapshot();
        if (!snapshot) return;
        size_t found = 0;
        for (const auto &entry : snapshot->getProviderMap())
            found += entry.second.first && entry.second.first->getClusterName() == "kvcm-static";
        if (found < 2) return;
        std::lock_guard<std::mutex> lock(ready->mutex); ready->value = true; ready->condition.notify_all();
    });
    autil::ScopeGuard remove([&] { service->removeSnapshotChangeListener(listener); });
    ASSERT_TRUE(client->subscribe());
    {
        std::unique_lock<std::mutex> lock(ready->mutex);
        ASSERT_TRUE(ready->condition.wait_for(lock, std::chrono::seconds(3), [&] { return ready->value; }));
    }
    EXPECT_EQ((std::vector<std::string>{"127.0.0.1:16381", "127.0.0.2:16381"}), client->bootstrapTargets());
    EXPECT_TRUE(test::unaryPool(*client)->_connectionMap.empty());
}

TEST_F(TairKvcmGrpcClientTest, failedNewSubscriptionPreservesExistingWorkerDiscovery)
{
    // Same init-failure pattern as GIG's own SubscribeServiceManagerTest;
    // the empty VIP environment never names or queries an online service.
    multi_call::VIPSubscribeService::destroy();
    autil::ScopeGuard cleanup([] { multi_call::VIPSubscribeService::destroy(); });
    multi_call::SubscribeServiceManager manager;
    multi_call::SubscribeConfig existing;
    multi_call::LocalJsonize node; node._bizName = "existing-worker"; node._clusterName = "existing-worker";
    node._ip = "127.0.0.1"; node._grpcPort = 19001; node._partCnt = 1; node._partId = 0; node._version = 0;
    existing.localConfig.nodes.push_back(node); ASSERT_TRUE(manager.addSubscribeService(existing));
    const auto original = manager._subServiceVec; ASSERT_EQ(1u, original.size());
    struct FailVipInit : multi_call::SubscribeConfig {
        bool validate(bool &cm2, bool &vip, bool &local, bool &istio) const override {
            cm2 = local = istio = false; vip = true; return true;
        }
    } failure;
    EXPECT_FALSE(manager.addSubscribeService(failure));
    ASSERT_EQ(original, manager._subServiceVec);
    multi_call::TopoNodeVec nodes; multi_call::HeartbeatSpecMap heartbeats;
    ASSERT_TRUE(manager._subServiceVec.front()->getClusterInfoMap(nodes, heartbeats));
    ASSERT_EQ(1u, nodes.size()); EXPECT_EQ("existing-worker", nodes.front().clusterName);
}

TEST_F(TairKvcmGrpcClientTest, validatesResourceTopologyWithoutWorkerAddress)
{
    TairKvcmGrpcClientR::Topology config;
    EXPECT_TRUE(config.valid()); EXPECT_FALSE(config.configured()); EXPECT_EQ(4u, config.threadNum);
    config.address = "v-kvcm-test"; EXPECT_TRUE(config.valid());
    config.port = 0; EXPECT_FALSE(config.valid()); config.port = 6381;
    config.discovery.type = "unsupported"; EXPECT_FALSE(config.valid());
    config.discovery.type = "direct"; config.address = "127.0.0.1"; EXPECT_TRUE(config.valid());
    config.address = "host/path"; EXPECT_FALSE(config.valid());
}

TEST_F(TairKvcmGrpcClientTest, preservesBinaryProtobufAndReusesGigConnectionAcrossRounds)
{
    test::KvcmGrpcMock server(respond); auto client = test::gigQueryClient(server.port);
    auto pipe = std::make_shared<CompletionPipe>(); const std::string trace("key\0value", 9);
    auto pool = test::unaryPool(*client); ASSERT_TRUE(pool);
    multi_call::ConnectionPtr connection;
    for (uint64_t round = 1; round <= 2; ++round) {
        const bool discovery = round == 1; auto message = payload(trace, discovery);
        ASSERT_TRUE(client->call(server.target(), discovery ? LEADER : LOOKUP, message,
            deadline(), 4096, pipe, 3, round, discovery));
        auto done = pipe->wait(); ASSERT_TRUE(done); EXPECT_TRUE(done->eventOk); EXPECT_TRUE(done->status.ok());
        EXPECT_EQ(trace, traceReply(*done)); EXPECT_EQ(round, done->round); EXPECT_EQ(3u, done->queryIndex);
        EXPECT_EQ(message->ByteSizeLong(), done->requestBytes);
        ASSERT_EQ(1u, pool->_connectionMap.size()); EXPECT_EQ(4u, pool->_worker->_completionQueues.size());
        if (round == 1) connection = pool->_connectionMap.begin()->second;
        else EXPECT_EQ(connection, pool->_connectionMap.begin()->second);
    }
    ASSERT_TRUE(server.waitRequests(2)); EXPECT_EQ(LEADER, server.requests[0].first); EXPECT_EQ(LOOKUP, server.requests[1].first);
}

TEST_F(TairKvcmGrpcClientTest, refusesExpiredOversizedInvalidAndWriteCallsWithoutIO)
{
    auto client = test::gigQueryClient(1); auto pipe = std::make_shared<CompletionPipe>(); auto message = payload("xx", true);
    EXPECT_FALSE(client->call("127.0.0.1:1", LEADER, message, 1, 4096, pipe, 0, 0, true));
    EXPECT_FALSE(client->call("127.0.0.1:1", LEADER, message, deadline(), 1, pipe, 0, 0, true));
    EXPECT_FALSE(client->call("bad/host:1", LEADER, message, deadline(), 4096, pipe, 0, 0, true));
    EXPECT_FALSE(client->call("127.0.0.1:1", LEADER, message, deadline(), 4096, {}, 0, 0, true));
    EXPECT_FALSE(client->call("127.0.0.1:1", "/kv_cache_manager.proto.meta.MetaService/StartWriteCache",
        message, deadline(), 4096, pipe, 0, 0, false));
    EXPECT_TRUE(test::unaryPool(*client)->_connectionMap.empty());
}

TEST_F(TairKvcmGrpcClientTest, asyncDeadlineAndTokenRemainOwnedUntilCompletion)
{
    test::KvcmGrpcMock server(respond); auto client = test::gigQueryClient(server.port);
    auto pipe = std::make_shared<CompletionPipe>(); auto token = std::make_shared<int>(5); std::weak_ptr<int> weak = token;
    const auto begin = autil::TimeUtility::monotonicTimeUs();
    ASSERT_TRUE(client->call(server.target(), LOOKUP, payload("hold"), begin + 300000, 4096, pipe, 0, 8, false, token));
    EXPECT_LT(autil::TimeUtility::monotonicTimeUs() - begin, 200000); token.reset(); EXPECT_FALSE(weak.expired());
    ASSERT_TRUE(server.waitRequests(1)); auto done = pipe->wait(); ASSERT_TRUE(done);
    EXPECT_EQ(grpc::StatusCode::DEADLINE_EXCEEDED, done->status.error_code());
    // GIG uses integer milliseconds, so the RPC may expire up to 1 ms before
    // the original monotonic deadline. It cannot extend the scheduling window.
    EXPECT_GE(done->completedUs + 1000, done->deadlineUs); EXPECT_FALSE(weak.expired());
    done.reset(); client.reset(); EXPECT_TRUE(weak.expired()); server.release();
}

TEST_F(TairKvcmGrpcClientTest, resourceDestructionDoesNotStopSharedGigOrLosePendingCompletion)
{
    test::KvcmGrpcMock server(respond); auto client = test::gigQueryClient(server.port);
    auto gig = std::dynamic_pointer_cast<navi::GigClientR>(client->_gigClientR->shared_from_this()); ASSERT_TRUE(gig);
    auto pipe = std::make_shared<CompletionPipe>();
    ASSERT_TRUE(client->call(server.target(), LOOKUP, payload("hold"),
        autil::TimeUtility::monotonicTimeUs() + 300000, 4096, pipe, 0, 1, false));
    ASSERT_TRUE(server.waitRequests(1));
    const auto begin = autil::TimeUtility::monotonicTimeUs(); client.reset();
    EXPECT_LT(autil::TimeUtility::monotonicTimeUs() - begin, 200000);
    TairKvcmGrpcClientR other; other._gigClientR = gig.get(); ASSERT_TRUE(other.configureTransport());
    ASSERT_TRUE(other.call(server.target(), LOOKUP, payload("still_alive"), deadline(), 4096, pipe, 1, 2, false));
    auto first = pipe->wait(); auto second = pipe->wait(); ASSERT_TRUE(first); ASSERT_TRUE(second);
    const auto pending = first->queryIndex == 0 ? first : second;
    const auto successful = first->queryIndex == 1 ? first : second;
    EXPECT_EQ(grpc::StatusCode::DEADLINE_EXCEEDED, pending->status.error_code());
    EXPECT_TRUE(successful->status.ok()); EXPECT_EQ("still_alive", traceReply(*successful)); server.release();
}

TEST_F(TairKvcmGrpcClientTest, reportsGrpcFailureOnceWithoutRetry)
{
    test::KvcmGrpcMock server(respond); auto client = test::gigQueryClient(server.port); auto pipe = std::make_shared<CompletionPipe>();
    ASSERT_TRUE(client->call(server.target(), LOOKUP, payload("error"), deadline(), 4096, pipe, 0, 1, false));
    auto done = pipe->wait(); ASSERT_TRUE(done); EXPECT_EQ(grpc::StatusCode::UNAVAILABLE, done->status.error_code());
    ASSERT_TRUE(server.waitRequests(1)); EXPECT_EQ(1u, server.requests.size());
}

TEST_F(TairKvcmGrpcClientTest, responseByteLimitIsEnforcedBeforeProtobufParsing)
{
    test::KvcmGrpcMock server(respond); auto client = test::gigQueryClient(server.port); auto pipe = std::make_shared<CompletionPipe>();
    ASSERT_TRUE(client->call(server.target(), LOOKUP, payload("oversize"), deadline(), 128, pipe, 0, 1, false));
    auto done = pipe->wait(); ASSERT_TRUE(done); EXPECT_EQ(grpc::StatusCode::RESOURCE_EXHAUSTED, done->status.error_code());
}

TEST_F(TairKvcmGrpcClientTest, concurrentCompletionsPreserveRequestIdentity)
{
    test::KvcmGrpcMock server(respond); auto client = test::gigQueryClient(server.port); auto pipe = std::make_shared<CompletionPipe>();
    for (size_t i = 0; i < 32; ++i)
        ASSERT_TRUE(client->call(server.target(), LOOKUP, payload(std::to_string(i)), deadline(), 4096, pipe, i, 7, false));
    std::set<size_t> indexes;
    for (size_t i = 0; i < 32; ++i) {
        auto done = pipe->wait(); ASSERT_TRUE(done); EXPECT_TRUE(done->status.ok()); EXPECT_EQ(7u, done->round);
        EXPECT_EQ(std::to_string(done->queryIndex), traceReply(*done)); EXPECT_TRUE(indexes.insert(done->queryIndex).second);
    }
    EXPECT_EQ(32u, indexes.size());
}
} // namespace navi_sched
