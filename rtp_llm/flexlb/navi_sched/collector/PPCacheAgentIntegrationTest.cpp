#include "aios/navi_sched/collector/NaviSchedCollector.h"
#include "aios/navi_sched/collector/PPCacheAgentConfig.h"
#include "aios/navi_sched/service/NaviSchedAgentImpl.h"
#include "aios/navi_sched/resource/AgentReportR.h"
#include "aios/navi_sched/data/AgentUpdateToMasterData.h"
#include "unittest/unittest.h"
#include <chrono>
#include <thread>
#include <iomanip>
#include <sstream>

using namespace navi_sched;
class PPCacheAgentIntegrationTest : public TESTBASE {
public:
    static std::shared_ptr<AgentReportR> report(NaviSchedAgent &agent)
    {
        return NaviSchedAgentImpl::getImpl(agent)->_agentReportR;
    }
    static std::vector<PPRequestShardConnection> pending(NaviSchedAgent &agent,
                                                         uint64_t requestId,
                                                         uint64_t attemptId)
    {
        return NaviSchedAgentImpl::getImpl(agent)->_ppCacheMerge->pendingRequestShards(
            {requestId, attemptId});
    }
    static bool collecting(NaviSchedAgent &agent)
    {
        return bool(NaviSchedAgentImpl::getImpl(agent)->_ppCacheGraph);
    }
};
namespace {
class PeerReadStatePipe : public navi::AsyncPipe {
public:
    PeerReadStatePipe() : navi::AsyncPipe(nullptr, navi::AS_REQUIRED) {}
    navi::ErrorCode setData(const navi::DataPtr &data) override
    {
        auto update = std::dynamic_pointer_cast<AgentUpdateToMasterData>(data);
        if (!update) return navi::EC_ABORT;
        available.store(update->peerCacheReadAvailable);
        return navi::EC_NONE;
    }
    navi::ErrorCode setEof() override { return navi::EC_NONE; }
    std::atomic<bool> available{false};
};
NaviSchedConfig service(const std::string &name)
{
    NaviSchedConfig config;
    config.name = name;
    config.enableKmon = false;
    config.enableSnapshot = false;
    config.waitNoQuery = false;
    config.gig.grpcIoThreadNum = 1;
    return config;
}
std::vector<PPCacheResidentObject> objects()
{
    std::vector<PPCacheResidentObject> result;
    for (uint32_t group = 0; group < 3; ++group) {
        for (uint64_t hash : {8, 9}) {
            std::ostringstream encoded;
            encoded << std::hex << std::setfill('0') << std::setw(64) << hash;
            result.push_back({"g" + std::to_string(group) + "-" + std::to_string(hash),
                              "{\"group_id\":" + std::to_string(group) +
                                  ",\"block_hashes\":[\"" + encoded.str() + "\"]}"});
        }
    }
    return result;
}
template <typename Predicate>
bool waitFor(Predicate predicate)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return predicate();
}
}

TEST_F(PPCacheAgentIntegrationTest, testDefaultAgentDoesNotStartPPCollection)
{
    NaviSchedAgentConfig config;
    config.service = service("agent_without_pp_collector_test");
    config.runtime.bizName = "agent_without_pp_collector_biz";
    config.runtime.threadNum = 2;
    NaviSchedAgent agent(config);
    ASSERT_TRUE(agent.init());
    EXPECT_FALSE(collecting(agent));
    EXPECT_EQ(0, agent.beginPPCacheRequest(10));
    EXPECT_FALSE(agent.finishPPCacheRequest(10, 1, PPShardRequestStatus::FAILED));
    EXPECT_GT(agent.getGrpcPort(), 0);
    agent.stop();
}

TEST_F(PPCacheAgentIntegrationTest, testRejectsNonPPAndIncompleteTopologyBeforeStarting)
{
    NaviSchedAgentConfig config;
    config.runtime.bizName = "invalid_pp_collector_biz";
    config.ppCacheCollector = std::make_shared<PPCacheAgentConfig>();
    config.ppCacheCollector->pipelineParallelSize = 1;
    config.ppCacheCollector->ppGroup = "pp-a";
    config.ppCacheCollector->policy.blockSize = 4096;
    config.ppCacheCollector->policy.linearStep = 1;
    NaviSchedAgent agent(config);
    EXPECT_FALSE(agent.init());
    EXPECT_LE(agent.getGrpcPort(), 0);
    EXPECT_FALSE(collecting(agent));
    config.ppCacheCollector->pipelineParallelSize = 4;
    NaviSchedAgent incomplete(config);
    EXPECT_FALSE(incomplete.init());
    EXPECT_LE(incomplete.getGrpcPort(), 0);
}

TEST_F(PPCacheAgentIntegrationTest, testFourCollectorsUpdateOnlyTheAgentsV6DCache)
{
    auto pp = std::make_shared<PPCacheAgentConfig>();
    pp->ppGroup = "pp-a";
    pp->pipelineParallelSize = 4;
    pp->policy.blockSize = 4096;
    pp->policy.linearStep = 2;
    pp->policy.hybridTailRequired = true;
    pp->policy.fullAttentionQueryDropTokens = 1;
    pp->policy.mambaReplayStopShift = true;
    pp->policy.mambaRetentionInterval = 8192;
    pp->policy.mambaAlign = true;
    pp->totalBlocks = 100;
    std::vector<std::unique_ptr<NaviSchedCollector>> collectors;
    for (uint32_t shard = 0; shard < 4; ++shard) {
        NaviSchedCollectorConfig config;
        config.service = service("agent_pp_shard_test_" + std::to_string(shard));
        config.runtime.bizName = "navi_sched_pp_cache_collector";
        config.runtime.threadNum = 1;
        config.ppGroup = pp->ppGroup;
        config.shardId = shard;
        config.shardCount = 4;
        config.groupRoles = {PPCacheGroupRole::FULL, PPCacheGroupRole::FULL,
                             PPCacheGroupRole::TAIL};
        config.groupBlockScales = {1, 1, 1};
        auto collector = std::make_unique<NaviSchedCollector>(config);
        ASSERT_TRUE(collector->init());
        ASSERT_TRUE(collector->replaceResidentObjects(objects()));
        pp->shards.push_back(
            {shard, config.runtime.bizName, "127.0.0.1", collector->getGrpcPort()});
        collectors.push_back(std::move(collector));
    }
    ASSERT_TRUE(pp->valid());
    NaviSchedAgentConfig config;
    config.service = service("pp_cache_receiving_agent_test");
    config.runtime.bizName = "pp_cache_receiving_agent_biz";
    config.runtime.threadNum = 4;
    config.ppCacheCollector = pp;
    NaviSchedAgent agent(config);
    ASSERT_TRUE(agent.init());
    const auto state = report(agent);
    ASSERT_TRUE(state);
    auto peerRead = std::make_shared<PeerReadStatePipe>();
    ASSERT_TRUE(state->subscribe("pp-read-state-test", peerRead));
    agent.reportCacheInserted(
        std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{99}));
    ASSERT_TRUE(waitFor([&]() {
        const auto cache = state->getCacheState(NaviSchedCacheKeyType::V6D);
        return cache.fullReadyBlockCount == 2 && cache.tailReadyBlockCount == 2;
    }));
    const auto cache = state->getCacheState(NaviSchedCacheKeyType::V6D);
    EXPECT_EQ(0, cache.cachedBlockCount);
    EXPECT_EQ(4096, cache.policy.blockSize);
    EXPECT_EQ(2, cache.policy.linearStep);
    EXPECT_TRUE(cache.policy.hybridTailRequired);
    EXPECT_EQ(1, cache.policy.fullAttentionQueryDropTokens);
    EXPECT_TRUE(cache.policy.mambaReplayStopShift);
    EXPECT_EQ(8192, cache.policy.mambaRetentionInterval);
    EXPECT_TRUE(cache.policy.mambaAlign);
    EXPECT_EQ(1, state->getCacheState().cachedBlockCount);
    EXPECT_FALSE(peerRead->available.load());
    for (size_t shard = 0; shard < collectors.size(); ++shard) {
        ASSERT_TRUE(collectors[shard]->reportPeerCacheReadAvailable(true));
        if (shard + 1 < collectors.size()) EXPECT_FALSE(peerRead->available.load());
    }
    ASSERT_TRUE(waitFor([&]() { return peerRead->available.load(); }));
    ASSERT_TRUE(collectors[1]->reportPeerCacheReadAvailable(false));
    ASSERT_TRUE(waitFor([&]() { return !peerRead->available.load(); }));
    EXPECT_EQ(2, state->getCacheState(NaviSchedCacheKeyType::V6D).fullReadyBlockCount);
    ASSERT_TRUE(collectors[1]->reportPeerCacheReadAvailable(true));
    ASSERT_TRUE(waitFor([&]() { return peerRead->available.load(); }));
    uint64_t previousAttempt = 0;
    EXPECT_EQ(0, agent.beginPPCacheRequest(0));
    for (auto status : {PPShardRequestStatus::SUCCEEDED, PPShardRequestStatus::FAILED,
                        PPShardRequestStatus::CANCELLED}) {
        const auto attempt = agent.beginPPCacheRequest(10);
        ASSERT_GT(attempt, previousAttempt);
        ASSERT_EQ(4, pending(agent, 10, attempt).size());
        EXPECT_EQ(0, agent.beginPPCacheRequest(10));
        EXPECT_FALSE(agent.finishPPCacheRequest(10, attempt + 1, status));
        EXPECT_FALSE(agent.finishPPCacheRequest(10, previousAttempt, status));
        EXPECT_FALSE(
            agent.finishPPCacheRequest(10, attempt, PPShardRequestStatus::PENDING));
        ASSERT_TRUE(agent.finishPPCacheRequest(10, attempt, status));
        EXPECT_FALSE(agent.finishPPCacheRequest(10, attempt, status));
        ASSERT_TRUE(waitFor([&]() { return pending(agent, 10, attempt).empty(); }));
        EXPECT_EQ(2,
                  state->getCacheState(NaviSchedCacheKeyType::V6D).fullReadyBlockCount);
        EXPECT_EQ(2,
                  state->getCacheState(NaviSchedCacheKeyType::V6D).tailReadyBlockCount);
        previousAttempt = attempt;
    }
    ASSERT_TRUE(collectors[1]->objectRemoved("g0-8"));
    ASSERT_TRUE(waitFor([&]() {
        return state->getCacheState(NaviSchedCacheKeyType::V6D).fullReadyBlockCount == 1;
    }));
    EXPECT_EQ(2, state->getCacheState(NaviSchedCacheKeyType::V6D).tailReadyBlockCount);
    const auto disconnectedAttempt = agent.beginPPCacheRequest(11);
    ASSERT_GT(disconnectedAttempt, previousAttempt);
    collectors[2]->stop();
    ASSERT_TRUE(waitFor([&]() {
        const auto cache = state->getCacheState(NaviSchedCacheKeyType::V6D);
        return cache.fullReadyBlockCount == 0 && cache.tailReadyBlockCount == 0;
    }));
    EXPECT_FALSE(peerRead->available.load());
    ASSERT_EQ(3, pending(agent, 11, disconnectedAttempt).size());
    ASSERT_TRUE(agent.finishPPCacheRequest(11, disconnectedAttempt,
                                           PPShardRequestStatus::CANCELLED));
    ASSERT_TRUE(
        waitFor([&]() { return pending(agent, 11, disconnectedAttempt).empty(); }));
    EXPECT_EQ(1, state->getCacheState().cachedBlockCount);
    const auto stoppedAttempt = agent.beginPPCacheRequest(12);
    ASSERT_GT(stoppedAttempt, disconnectedAttempt);
    ASSERT_TRUE(state->unsubscribe("pp-read-state-test", peerRead));
    agent.stop();
    EXPECT_EQ(0, agent.beginPPCacheRequest(12));
    EXPECT_FALSE(
        agent.finishPPCacheRequest(12, stoppedAttempt, PPShardRequestStatus::FAILED));
}
