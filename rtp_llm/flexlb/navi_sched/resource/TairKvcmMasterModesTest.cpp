#include "aios/navi_sched/test/TairKvcmGrpcTestServer.h"
#include "aios/navi_sched/scheduler/TairKvcmGrpcProtocol.h"
#include "aios/navi_sched/graph/MasterScheduleGraph.h"
#include "aios/navi_sched/service/NaviSchedMaster.h"
#include "aios/navi_sched/kernel/MasterTairQueryKernel.h"
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/resource/MasterScheduleR.h"
#include "aios/network/gig/multi_call/service/GrpcConnectionPool.h"
#include "aios/network/gig/multi_call/service/SearchServiceManager.h"
#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "navi/engine/NaviSnapshot.h"
#include "navi/engine/Biz.h"
#include "unittest/unittest.h"
#include <algorithm>
#include <condition_variable>

namespace navi_sched {
namespace {
class ReplyPipe : public navi::AsyncPipe {
public:
    ReplyPipe() : navi::AsyncPipe(nullptr, navi::AS_INDEPENDENT) {}
    navi::ErrorCode setData(const navi::DataPtr &data) override {
        std::lock_guard<std::mutex> lock(mutex);
        reply = std::dynamic_pointer_cast<MasterResponseToClientData>(data);
        condition.notify_all(); return navi::EC_NONE;
    }
    navi::ErrorCode setEof() override { return navi::EC_NONE; }
    MasterResponseToClientDataPtr wait() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!condition.wait_for(lock, std::chrono::seconds(5), [&] { return bool(reply); })) return {};
        return reply;
    }
    std::mutex mutex; std::condition_variable condition; MasterResponseToClientDataPtr reply;
};
bool addNode(MasterStateR &state)
{
    if (!state.registerAgent("agent_mode_test", "model_biz", "127.0.0.1:19000", 100)) return false;
    auto handshake = std::make_shared<AgentUpdateToMasterData>();
    handshake->event = NaviSchedAgentEvent::AGENT_HANDSHAKE;
    handshake->control = std::make_shared<AgentControlToMasterData>();
    handshake->control->agentStartTimeUs = 100;
    handshake->control->serverInfo.ip = "127.0.0.1"; handshake->control->serverInfo.port = 18000;
    handshake->requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    if (!state.reportAgent("agent_mode_test", handshake)) return false;
    auto cache = std::make_shared<AgentUpdateToMasterData>(); cache->event = NaviSchedAgentEvent::CACHE_UPDATED;
    cache->cache = std::make_shared<AgentCacheToMasterData>();
    cache->cache->blockSize = 256; cache->cache->linearStep = 1; cache->cache->hybridTailRequired = 0;
    cache->cache->revision = 1; cache->cache->fullSnapshot = true;
    cache->cache->cachedBlockCount = 0; cache->cache->fullReadyBlockCount = 0; cache->cache->tailReadyBlockCount = 0;
    cache->cache->addedBlockHashes = std::make_shared<const NaviSchedBlockHashes>();
    cache->cache->addedFullReadyBlockHashes = std::make_shared<const NaviSchedBlockHashes>();
    cache->cache->addedTailReadyBlockHashes = std::make_shared<const NaviSchedBlockHashes>();
    if (!state.reportAgent("agent_mode_test", cache)) return false;
    auto pending = std::make_shared<AgentUpdateToMasterData>(); pending->event = NaviSchedAgentEvent::PENDING_CACHE_UPDATED;
    pending->pendingCache = std::make_shared<AgentPendingCacheToMasterData>(); pending->pendingCache->fullSnapshot = true;
    return state.reportAgent("agent_mode_test", pending);
}
}
class TairKvcmMasterModesTest : public TESTBASE {};

TEST_F(TairKvcmMasterModesTest, graphIncludesQueryOnlyWhenEnabledAndUsesCommonPrepare)
{
    for (bool enabled : {false, true}) {
        auto graph = MasterScheduleGraph::buildGraph("master_biz", "{}", "navi_sched.cost_scheduler_k", "{}", 1, enabled);
        ASSERT_TRUE(graph); size_t queries = 0, prepares = 0, contexts = 0;
        for (const auto &subgraph : graph->sub_graphs()) for (const auto &node : subgraph.nodes()) {
            queries += node.kernel_name() == MasterTairQueryKernel::KERNEL_ID;
            prepares += node.kernel_name() == "navi_sched.master_schedule_prepare_k";
            contexts += node.kernel_name() == "navi_sched.master_schedule_context_k";
        }
        EXPECT_EQ(enabled ? 1u : 0u, queries); EXPECT_EQ(1u, prepares); EXPECT_EQ(1u, contexts);
    }
}

TEST_F(TairKvcmMasterModesTest, actualMasterSchedulesWhenQueryIsDisabledOrHasNoNativeCapability)
{
    for (size_t unaryThreads : {size_t(0), size_t(2), size_t(4)}) {
        const bool enabled = unaryThreads != 0;
        SCOPED_TRACE(unaryThreads);
        NaviSchedMasterConfig config;
        config.service.name = enabled ? "master_tair_enabled_mode" : "master_tair_disabled_mode";
        config.service.enableKmon = false; config.service.enableSnapshot = false;
        config.service.waitNoQuery = false; config.service.gig.grpcPort = 0;
        config.runtime.bizName = "master_biz"; config.runtime.threadNum = 2; config.runtime.queueSize = 100;
        config.maxWaitScheduleCount = 1; config.maxWaitScheduleTimeMs = 1000;
        config.schedulerName = "cost";
        config.tairKvcmConfig = !enabled ? "{}" : unaryThreads == 2
            ? R"({"address":"127.0.0.1","port":1,"discovery":{"type":"direct"},"thread_num":2})"
            : R"({"address":"127.0.0.1","port":1,"discovery":{"type":"direct"}})";
        NaviSchedMaster master(config); ASSERT_TRUE(master.init());
        EXPECT_EQ(enabled, master._tairQueryEnabled);
        ASSERT_TRUE(master._masterStateR); ASSERT_TRUE(addNode(*master._masterStateR));
        for (uint64_t id : {1, 2}) {
            auto request = std::make_shared<ClientRequestToMasterData>(); request->requestId = id;
            request->bizName = "model_biz"; request->promptTokenCount = 512; request->blockSize = 256;
            request->blockHashes = std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
            request->timeoutMs = 5000; request->routeOnly = true;
            auto pipe = std::make_shared<ReplyPipe>(); ASSERT_TRUE(master._masterScheduleR->submit(request, pipe));
            auto reply = pipe->wait(); ASSERT_TRUE(reply);
            EXPECT_EQ("127.0.0.1:18000", reply->target.address);
            EXPECT_EQ("127.0.0.1:19000", reply->target.agentAddress);
            EXPECT_EQ(0, reply->predictedCacheHitTokenCount); EXPECT_NE("master_restarting", reply->reason);
        }
        {
            // createResource/runGraph delegate through Navi's configuration wrapper.
            // Inspect the actual runtime, not the wrapper's bootstrap-only snapshot.
            auto runtime = master.getNavi();
            while (auto real = runtime->getRealNavi()) runtime = std::move(real);
            auto snapshot = runtime->getSnapshot(); ASSERT_TRUE(snapshot);
            const auto bizNames = snapshot->getBizList();
            ASSERT_NE(bizNames.end(), std::find(bizNames.begin(), bizNames.end(), master._masterBizName));
            auto biz = snapshot->_bizManager->getBiz(nullptr, master._masterBizName);
            ASSERT_TRUE(biz);
            navi::ResourceMapPtr resources; ASSERT_TRUE(biz->getResourceMap(1, 0, resources)); ASSERT_TRUE(resources);
            EXPECT_EQ(enabled ? 1u : 0u, resources->getMap().count(MasterTairQueryR::RESOURCE_ID));
            EXPECT_EQ(enabled ? 1u : 0u, resources->getMap().count(TairKvcmGrpcClientR::RESOURCE_ID));
            const auto gigIt = resources->getMap().find(navi::GigClientR::RESOURCE_ID);
            ASSERT_NE(resources->getMap().end(), gigIt);
            const auto gig = std::dynamic_pointer_cast<navi::GigClientR>(gigIt->second); ASSERT_TRUE(gig);
            const auto unary = gig->getSearchService()->getSearchServiceManager()
                ->getConnectionManager()->getConnectionPool(multi_call::MC_PROTOCOL_GRPC);
            EXPECT_EQ(enabled, bool(unary));
            if (enabled) {
                auto resource = std::dynamic_pointer_cast<TairKvcmGrpcClientR>(resources->getMap().at(TairKvcmGrpcClientR::RESOURCE_ID));
                ASSERT_TRUE(resource); EXPECT_TRUE(resource->_transportReady);
                const auto pool = std::dynamic_pointer_cast<multi_call::GrpcConnectionPool>(
                    resource->_gigClientR->getSearchService()->getSearchServiceManager()
                        ->getConnectionManager()->getConnectionPool(multi_call::MC_PROTOCOL_GRPC));
                ASSERT_TRUE(pool); EXPECT_EQ(unaryThreads, pool->_worker->_completionQueues.size());
                EXPECT_TRUE(pool->_connectionMap.empty()); // No V6D capability, no RPC.
            }
        } // Release the inspected snapshot before stop waits for its readers.
        master.stop();
    }
}

TEST_F(TairKvcmMasterModesTest, unifiedTairCheckpointHitReachesActualClientReply)
{
    namespace wire = kv_cache_manager::proto::meta;
    test::KvcmGrpcMock *serverPtr = nullptr;
    std::atomic<size_t> lookups{0};
    test::KvcmGrpcMock server([&](const std::string &method, const std::string &payload) {
        if (method == TAIR_KVCM_GRPC_GET_CLUSTER_INFO) {
            wire::GetClusterInfoResponse reply;
            reply.mutable_header()->mutable_status()->set_code(wire::OK);
            reply.mutable_leader_endpoint()->set_host("127.0.0.1");
            reply.mutable_leader_endpoint()->set_meta_rpc_port(serverPtr->port);
            return test::KvcmGrpcMock::Reply{reply.SerializeAsString(), grpc::Status::OK};
        }
        wire::GetCacheLocationsByBackendRequest query;
        if (!query.ParseFromString(payload)) throw std::runtime_error("bad lookup");
        wire::GetCacheLocationsByBackendResponse reply;
        reply.mutable_header()->mutable_status()->set_code(wire::OK);
        for (int i = 0; i < query.block_keys_size(); ++i) {
            auto location = reply.add_key_locations()->add_locations();
            location->set_type(wire::ST_TAIRMEMPOOL);
            auto spec = location->add_location_specs();
            spec->set_name(query.location_spec_names(i));
            spec->set_uri("pace://loopback/remote");
        }
        ++lookups;
        return test::KvcmGrpcMock::Reply{reply.SerializeAsString(), grpc::Status::OK};
    });
    serverPtr = &server;
    NaviSchedMasterConfig config;
    config.service.name = "unified_tair_checkpoint_reply";
    config.service.enableKmon = config.service.enableSnapshot = config.service.waitNoQuery = false;
    config.service.gig.grpcPort = 0;
    config.runtime.bizName = "master_biz"; config.runtime.threadNum = 2; config.runtime.queueSize = 100;
    config.maxWaitScheduleCount = 1; config.maxWaitScheduleTimeMs = 1000; config.schedulerName = "cost";
    config.tairKvcmConfig = "{\"address\":\"127.0.0.1\",\"port\":" + std::to_string(server.port) +
        ",\"discovery\":{\"type\":\"direct\"}}";
    NaviSchedMaster master(config); ASSERT_TRUE(master.init());
    auto &state = *master._masterStateR;
    ASSERT_TRUE(state.registerAgent("unified_tair_agent", "model_biz", "127.0.0.1:19000", 100));
    auto sent = std::make_shared<AgentUpdateToMasterData>();
    sent->event = NaviSchedAgentEvent::AGENT_HANDSHAKE;
    sent->control = std::make_shared<AgentControlToMasterData>();
    auto &control = *sent->control;
    control.agentStartTimeUs = 100;
    control.serverInfo.ip = "127.0.0.1"; control.serverInfo.port = 18000;
    control.pdSeparated = control.unifiedLatencyPrediction = true;
    sent->workload = std::make_shared<NaviSchedWorkloadSnapshot>();
    sent->workload->revision = 1; sent->workload->eventTimeUs = 100;
    sent->workload->prefillCapacitySplit = true; sent->workload->maxRunningRequestCount = 256;
    ASSERT_TRUE(sent->workload->valid());
    control.tairKvcmConfig.managerEndpoint = "http://127.0.0.1:1";
    control.tairKvcmConfig.instanceId = "unified_checkpoint_64";
    control.tairKvcmConfig.hashBlockSize = 64; control.tairKvcmConfig.nodeCount = 1;
    for (int groupId = 0; groupId < 2; ++groupId) {
        NaviSchedTairKvcmGroupConfig group;
        group.groupId = groupId; group.hashBlockScale = 2; group.tail = groupId == 1;
        group.objectSizeBytes = 4096; group.locationSpecNames = {"group" + std::to_string(groupId)};
        control.tairKvcmConfig.groups.push_back(group);
    }
    sent->requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    sent->requestEvent->inflightStatsRevision = 1;
    ASSERT_TRUE(sent->valid());
    // Exercise the real typed handshake codec, with geometry and Native9 both OFF.
    autil::DataBuffer buffer; sent->serialize(buffer);
    auto received = std::make_shared<AgentUpdateToMasterData>(); received->deserialize(buffer);
    ASSERT_TRUE(received->valid());
    EXPECT_EQ(0, received->control->prefillExecutionPolicy.schedulerBlockSize);
    EXPECT_FALSE(received->control->executionGeometryEnabled);
    EXPECT_EQ(0, received->control->nativePrefillPipelineStages);
    ASSERT_TRUE(state.reportAgent("unified_tair_agent", received));
    auto cache = std::make_shared<AgentUpdateToMasterData>(); cache->event = NaviSchedAgentEvent::CACHE_UPDATED;
    cache->cache = std::make_shared<AgentCacheToMasterData>();
    cache->cache->blockSize = 64; cache->cache->linearStep = 1; cache->cache->hybridTailRequired = 0;
    cache->cache->revision = 1; cache->cache->fullSnapshot = true;
    cache->cache->cachedBlockCount = cache->cache->fullReadyBlockCount = cache->cache->tailReadyBlockCount = 0;
    cache->cache->addedBlockHashes = std::make_shared<const NaviSchedBlockHashes>();
    cache->cache->addedFullReadyBlockHashes = std::make_shared<const NaviSchedBlockHashes>();
    cache->cache->addedTailReadyBlockHashes = std::make_shared<const NaviSchedBlockHashes>();
    ASSERT_TRUE(state.reportAgent("unified_tair_agent", cache));
    auto v6dCache = std::make_shared<AgentUpdateToMasterData>();
    v6dCache->event = NaviSchedAgentEvent::CACHE_UPDATED;
    v6dCache->cache = std::make_shared<AgentCacheToMasterData>(*cache->cache);
    cache = std::move(v6dCache);
    cache->cache->cacheKeyType = NaviSchedCacheKeyType::V6D;
    cache->cache->linearStep = 2; cache->cache->hybridTailRequired = 1;
    ASSERT_TRUE(state.reportAgent("unified_tair_agent", cache));
    auto pending = std::make_shared<AgentUpdateToMasterData>(); pending->event = NaviSchedAgentEvent::PENDING_CACHE_UPDATED;
    pending->pendingCache = std::make_shared<AgentPendingCacheToMasterData>(); pending->pendingCache->fullSnapshot = true;
    ASSERT_TRUE(state.reportAgent("unified_tair_agent", pending));
    auto request = std::make_shared<ClientRequestToMasterData>();
    request->requestId = 101; request->bizName = "model_biz"; request->promptTokenCount = 192;
    request->blockSize = 64; request->explicitCacheCheckpointPosition = 130;
    request->blockHashes = std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
    request->timeoutMs = 5000; request->routeOnly = true;
    auto pipe = std::make_shared<ReplyPipe>(); ASSERT_TRUE(master._masterScheduleR->submit(request, pipe));
    auto reply = pipe->wait();
    EXPECT_EQ(1u, lookups.load());
    ASSERT_TRUE(reply);
    EXPECT_EQ("127.0.0.1:18000", reply->target.address);
    EXPECT_EQ(128, reply->predictedCacheHitTokenCount);
    EXPECT_NE("master_restarting", reply->reason);
    master.stop();
}

} // namespace navi_sched
