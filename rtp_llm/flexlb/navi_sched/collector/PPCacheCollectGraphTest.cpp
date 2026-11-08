#include "aios/navi_sched/collector/NaviSchedCollector.h"
#include "aios/navi_sched/collector/PPCacheCollectGraph.h"
#include "aios/navi_sched/config/NaviSchedConfigBuilder.h"
#include "aios/navi_sched/service/NaviSchedRoleRuntimeAccess.h"
#include "unittest/unittest.h"
#include "autil/legacy/jsonizable.h"
#include <set>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace navi_sched;
class PPCacheCollectGraphTest : public TESTBASE {};

namespace {
class TestLogGuard {
public:
    ~TestLogGuard()
    {
        if (!::testing::Test::HasFailure()) return;
        for (const auto &entry : std::filesystem::directory_iterator("logs")) {
            if (!entry.is_regular_file()) continue;
            std::ifstream log(entry.path());
            std::string line;
            while (std::getline(log, line)) {
                if (line.find("ERROR") != std::string::npos)
                    std::cerr << entry.path() << ": " << line << "\n";
            }
        }
    }
};
std::vector<PPCacheResidentObject> residentObjects()
{
    std::vector<PPCacheResidentObject> objects;
    for (uint32_t group = 0; group < 2; ++group) {
        for (uint64_t hash : {8, 9}) {
            std::ostringstream encoded;
            encoded << std::hex << std::setfill('0') << std::setw(64) << hash;
            objects.push_back({"g" + std::to_string(group) + "-" + std::to_string(hash),
                               "{\"group_id\":" + std::to_string(group) +
                                   ",\"block_hashes\":[\"" + encoded.str() + "\"]}"});
        }
    }
    return objects;
}
class ReceiverRole : public NaviSchedRole {
public:
    explicit ReceiverRole(const NaviSchedConfig &config) : NaviSchedRole(config) {}
};
NaviSchedConfig serviceConfig(const std::string &name)
{
    NaviSchedConfig config;
    config.name = name;
    config.enableKmon = false;
    config.enableSnapshot = false;
    config.waitNoQuery = false;
    config.gig.grpcIoThreadNum = 1;
    return config;
}
}

TEST_F(PPCacheCollectGraphTest, testFourRealGrpcStreamsMergeAndReconnect)
{
    TestLogGuard logGuard;
    std::mutex mutex;
    std::condition_variable changed;
    std::set<uint64_t> full;
    std::vector<PPRequestCompletion> completions;
    auto merge = std::make_shared<PPCacheMergeR>();
    ASSERT_TRUE(merge->initialize("pp-a", 4, [&](const auto &changes) {
        std::lock_guard<std::mutex> lock(mutex);
        full.insert(changes.addedFull.begin(), changes.addedFull.end());
        for (auto hash : changes.removedFull) full.erase(hash);
        completions.insert(completions.end(), changes.completedRequests.begin(),
                           changes.completedRequests.end());
        changed.notify_all();
    }));
    auto requestFinished = std::make_shared<PPRequestFinishedR>();
    ASSERT_TRUE(requestFinished->initialize(
        "pp-a", 4,
        [merge](auto shard, auto generation) { merge->disconnect(shard, generation); }));
    auto waitFor = [&](const auto &predicate) {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(15), predicate);
    };
    std::filesystem::create_directories("logs");
    ReceiverRole receiver(serviceConfig("pp_merge_receiver_test"));
    ASSERT_TRUE(receiver.init());
    const auto navi = NaviSchedRoleRuntimeAccess::getNavi(receiver);
    NaviSchedRuntimeConfig runtime;
    runtime.bizName = "pp_merge_receiver_biz";
    runtime.threadNum = 4;
    NaviSchedLocalNode local;
    local.bizName = runtime.bizName;
    local.ip = "127.0.0.1";
    local.grpcPort = receiver.getGrpcPort();
    local.supportHeartbeat = false;
    runtime.localNodes.push_back(local);
    auto configMap = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
        autil::legacy::json::ParseJson(
            NaviSchedConfigBuilder::build(runtime,
                                          {"^navi_sched\\.pp_cache_merge_k$",
                                           "^navi_sched\\.pp_request_finished_source_k$"},
                                          true)));
    configMap["log_config"] = autil::legacy::json::ParseJson(
        R"({"file_appenders":[{"file_name":"logs/pp-grpc-integration.log","log_level":"DEBUG","async_flush":false}]})");
    const auto config = autil::legacy::ToJsonString(configMap, true);
    navi::ResourceMap resources;
    resources.addResource(merge);
    resources.addResource(requestFinished);
    const bool updated = navi->update(config, resources);
    if (!updated) {
        receiver.stop();
        for (const auto &entry : std::filesystem::directory_iterator("logs")) {
            if (!entry.is_regular_file()) continue;
            std::ifstream log(entry.path());
            std::string line;
            while (std::getline(log, line)) {
                if (line.find("ERROR") != std::string::npos)
                    std::cerr << entry.path() << ": " << line << "\n";
            }
        }
    }
    ASSERT_TRUE(updated);
    std::vector<std::unique_ptr<NaviSchedCollector>> collectors;
    std::vector<NaviSchedCollectorConfig> configs;
    std::vector<PPCacheShardEndpoint> endpoints;
    for (uint32_t shard = 0; shard < 4; ++shard) {
        NaviSchedCollectorConfig config;
        config.service = serviceConfig("pp_shard_test_" + std::to_string(shard));
        config.runtime.bizName = "navi_sched_pp_cache_collector";
        config.runtime.threadNum = 1;
        config.ppGroup = "pp-a";
        config.shardId = shard;
        config.shardCount = 4;
        config.groupRoles = {PPCacheGroupRole::FULL, PPCacheGroupRole::FULL};
        config.groupBlockScales = {1, 1};
        auto collector = std::make_unique<NaviSchedCollector>(config);
        ASSERT_TRUE(collector->init());
        config.service.gig.grpcPort = collector->getGrpcPort();
        endpoints.push_back(
            {shard, config.runtime.bizName, "127.0.0.1", collector->getGrpcPort()});
        configs.push_back(config);
        ASSERT_TRUE(collector->replaceResidentObjects(residentObjects()));
        collectors.push_back(std::move(collector));
    }
    PPCacheCollectGraph graph(navi, runtime.bizName, "pp-a", merge, requestFinished);
    ASSERT_TRUE(graph.start(endpoints));
    ASSERT_TRUE(waitFor([&]() { return full == std::set<uint64_t>({8, 9}); }));
    ASSERT_TRUE(merge->beginRequest({10, 1}).accepted);
    for (uint32_t shard = 0; shard < 4; ++shard) {
        PPCacheShardUpdate terminal;
        terminal.terminals = {{{10, 1},
                               shard == 2 ? PPShardRequestStatus::FAILED
                                          : PPShardRequestStatus::SUCCEEDED}};
        ASSERT_TRUE(collectors[shard]->publish(terminal));
    }
    ASSERT_TRUE(waitFor([&]() { return completions.size() == 1; }));
    {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_FALSE(completions[0].succeeded);
        EXPECT_EQ(PPShardRequestStatus::FAILED, completions[0].shardStatuses[2]);
    }
    uint64_t requestId = 11;
    for (auto status : {PPShardRequestStatus::SUCCEEDED, PPShardRequestStatus::FAILED,
                        PPShardRequestStatus::CANCELLED}) {
        const PPRequestAttempt request{requestId++, 1};
        ASSERT_TRUE(merge->beginRequest(request).accepted);
        ASSERT_EQ(4, graph.publishRequestFinished({request, status}));
        const auto expectedCount = size_t(request.requestId - 9);
        ASSERT_TRUE(waitFor([&]() { return completions.size() == expectedCount; }));
        {
            std::lock_guard<std::mutex> lock(mutex);
            EXPECT_EQ(std::vector<PPShardRequestStatus>(4, status),
                      completions.back().shardStatuses);
            EXPECT_EQ(status == PPShardRequestStatus::SUCCEEDED,
                      completions.back().succeeded);
            EXPECT_EQ(std::set<uint64_t>({8, 9}), full);
        }
        EXPECT_EQ(0, graph.publishRequestFinished({request, status}));
    }
    EXPECT_EQ(0, graph.publishRequestFinished({{999, 1}, PPShardRequestStatus::FAILED}));
    ASSERT_TRUE(collectors[1]->objectRemoved("g0-8"));
    ASSERT_TRUE(waitFor([&]() { return full == std::set<uint64_t>({9}); }));
    ASSERT_TRUE(merge->beginRequest({14, 1}).accepted);
    collectors[2]->stop();
    ASSERT_TRUE(waitFor([&]() { return full.empty(); }));
    ASSERT_EQ(3,
              graph.publishRequestFinished({{14, 1}, PPShardRequestStatus::CANCELLED}));
    ASSERT_TRUE(waitFor([&]() { return completions.size() == 5; }));
    {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_EQ(PPShardRequestStatus::DISCONNECTED,
                  completions.back().shardStatuses[2]);
        EXPECT_FALSE(completions.back().succeeded);
    }
    collectors[2] = std::make_unique<NaviSchedCollector>(configs[2]);
    ASSERT_TRUE(collectors[2]->init());
    ASSERT_TRUE(collectors[2]->replaceResidentObjects(residentObjects()));
    ASSERT_TRUE(waitFor([&]() { return full == std::set<uint64_t>({9}); }));
    graph.stop();
    {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_TRUE(full.empty());
    }
    receiver.stop();
}
