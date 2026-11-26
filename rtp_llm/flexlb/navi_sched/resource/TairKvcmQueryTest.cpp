#include "aios/navi_sched/scheduler/TairKvcmQuery.h"
#include "aios/navi_sched/scheduler/TairKvcmGrpcProtocol.h"
#include "aios/navi_sched/test/TairKvcmGrpcTestServer.h"
#include "aios/navi_sched/test/TairKvcmGigTestClient.h"
#include "aios/navi_sched/kernel/MasterTairQueryKernel.h"
#include "aios/navi_sched/resource/MasterPendingCacheR.h"
#include "aios/navi_sched/resource/MasterStateR.h"
#include "aios/navi_sched/scheduler/MasterSchedulePrepareKernel.h"
#include "navi/tester/KernelTesterBuilder.h"
#include "autil/TimeUtility.h"
#include "rapidjson/document.h"
#include "autil/legacy/json.h"
#include "autil/legacy/json.h"
#include "unittest/unittest.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include "kmonitor/client/test/MetricsTestUtil.h"
#include "autil/Scope.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <numeric>
#include <thread>

namespace navi_sched {
namespace {

class NotifyingPipe : public navi::AsyncPipe {
public:
    explicit NotifyingPipe(navi::AsyncPipePtr delegate)
        : navi::AsyncPipe(nullptr, navi::AS_INDEPENDENT), _delegate(std::move(delegate)) {}
    navi::ErrorCode setData(const navi::DataPtr &data) override
    {
        auto result = _delegate->setData(data);
        std::lock_guard<std::mutex> lock(_mutex);
        ++_count;
        if (auto rpc = std::dynamic_pointer_cast<TairKvcmGrpcCompletionData>(data))
            _observations.push_back({rpc->deadlineUs, rpc->queryToken.get()});
        _ready.notify_all();
        return result;
    }
    navi::ErrorCode setDataOverride(const navi::DataPtr &data, bool replace, bool ignore = false) override
    {
        return _delegate->setDataOverride(data, replace, ignore);
    }
    bool getData(navi::DataPtr &data, bool &eof) override { return _delegate->getData(data, eof); }
    bool waitFor(size_t count)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        return _ready.wait_for(lock, std::chrono::seconds(2), [&]() { return _count >= count; });
    }
    struct Observation { int64_t deadlineUs; const void *token; };
    std::vector<Observation> observations()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _observations;
    }
private:
    std::vector<Observation> _observations;
    navi::AsyncPipePtr _delegate;
    std::mutex _mutex;
    std::condition_variable _ready;
    size_t _count = 0;
};

std::shared_ptr<NaviSchedTairKvcmConfig> batchConfig(size_t nodeCount = 2)
{
    auto config = std::make_shared<NaviSchedTairKvcmConfig>();
    config->managerEndpoint = "http://127.0.0.1:6382";
    config->instanceId = "isolated_batch";
    config->hashBlockSize = 64;
    config->nodeCount = nodeCount;
    for (int32_t g = 0; g < 2; ++g) {
        NaviSchedTairKvcmGroupConfig group;
        group.groupId = g;
        group.hashBlockScale = 2;
        group.tail = g == 1;
        group.objectSizeBytes = 4096;
        for (size_t n = 0; n < nodeCount; ++n)
            group.locationSpecNames.push_back("group" + std::to_string(g) + "_pp" + std::to_string(n));
        config->groups.emplace_back(std::move(group));
    }
    return config;
}

std::vector<NaviSchedScheduleItem> batchRequests(size_t count = 1)
{
    std::vector<NaviSchedScheduleItem> requests(count);
    for (auto &item : requests) {
        item.request.blockSize = 64;
        item.request.promptTokenCount = 320;
        item.request.blockHashes = std::make_shared<NaviSchedBlockHashes>(
            NaviSchedBlockHashes{11, 12, 13, 14});
    }
    return requests;
}

std::shared_ptr<MasterStateR::NodeState> residentNode(
    MasterPendingCacheR &pending, const std::shared_ptr<const NaviSchedTairKvcmConfig> &config)
{
    auto node = std::make_shared<MasterStateR::NodeState>();
    node->_pendingCacheR = &pending;
    node->_info.online = true;
    node->_info.tairKvcmConfig = config;
    node->_cachePolicy.blockSize = 64;
    node->_cachePolicy.linearStep = 1;
    node->_cachePolicyReady.store(true);
    node->_cacheStateReady.store(true);
    node->_cachedBlockHashes = {11, 12};
    auto &local = node->_v6dCacheState;
    local._cachePolicy.blockSize = 64;
    local._cachePolicy.linearStep = 2;
    local._cachePolicy.hybridTailRequired = true;
    local._hybridTailRequired = true;
    local._cachePolicyReady.store(true);
    local._cacheStateReady.store(true);
    local._fullReadyBlockHashes = {13, 14};
    return node;
}

MasterScheduleContextDataPtr scheduleData(const std::shared_ptr<MasterStateR::NodeState> &node)
{
    auto data = std::make_shared<MasterScheduleContextData>();
    data->context->_requests = batchRequests();
    data->context->_scheduleBeginTimeUs = autil::TimeUtility::monotonicTimeUs();
    data->context->_scheduleDeadlineUs = data->context->_scheduleBeginTimeUs + 1000000;
    NaviSchedNodeSnapshot snapshot;
    snapshot.info = std::make_shared<const NaviSchedNodeInfo>(node->_info);
    data->context->setNodes({node}, {snapshot});
    return data;
}

} // namespace

class TairKvcmQueryTest : public TESTBASE {};

TEST_F(TairKvcmQueryTest, reportsQueryCoverageAndConfiguredLimits)
{
    kmonitor::KMonitorFactory::Shutdown();
    autil::ScopeGuard shutdown([] { kmonitor::KMonitorFactory::Shutdown(); });
    kmonitor::MetricsConfig config;
    config.set_tenant_name("default");
    config.set_service_name("tair_query_metric_test");
    config.set_sink_address("127.0.0.1:4141");
    config.set_manually_mode(true);
    config.set_inited(true);
    ASSERT_TRUE(kmonitor::KMonitorFactory::Init(config));
    kmonitor::KMonitorFactory::Start();
    auto reporter = std::make_shared<kmonitor::MetricsReporter>("", kmonitor::MetricsTags());
    kmonitor::MetricsReporterR resource(reporter);
    MasterTairQueryR queryResource;
    MasterTairQueryR::Batch batch;
    const_cast<kmonitor::MetricsReporterR *&>(queryResource._metricsReporterR) = &resource;
    batch.data = std::make_shared<MasterScheduleContextData>();
    batch.data->context->_requests = batchRequests();
    TairKvcmQueryResult result;
    result.status = TairKvcmQueryStatus::OK;
    result.remoteHits = {1, 0, 1};
    result.elapsedUs = 14000;
    batch.data->tairKvcmResults = {result};
    batch.queries.resize(1);
    batch.queries[0].retryCount = 1;
    batch.queries[0].rpcCount = 3;
    queryResource.reportBatch(batch);
    const auto check = [&](const char *name, const kmonitor::MetricsTags *tags, double value) {
        kmonitor::MetricsTestUtil::checkMetricWithValue(reporter.get(),
            std::string("tair_query_metric_test.master.") + name, tags, value);
    };
    kmonitor::MetricsTags tags;
    tags.AddTag("status", "OK");
    check("tair_query_remote_hit_objects", &tags, 2);
    check("tair_query_ms", &tags, 14);
    check("tair_query_rpc_count", &tags, 3);
    check("tair_query_retry_result_qps", &tags, 1);
    TairKvcmGrpcCompletionData completion;
    completion.startedUs = 1000;
    completion.completedUs = 6000;
    queryResource.reportRpcCompletion(completion, TairKvcmQueryStatus::BUSINESS_ERROR,
        37, false, kv_cache_manager::proto::meta::SERVER_NOT_LEADER);
    kmonitor::MetricsTags rpcTags;
    rpcTags.AddTag("stage", "lookup");
    rpcTags.AddTag("status", "BUSINESS_ERROR");
    rpcTags.AddTag("grpc_code", "0");
    rpcTags.AddTag("kvcm_code", "9");
    check("tair_grpc_completion_qps", &rpcTags, 1);
    check("tair_grpc_rpc_ms", &rpcTags, 5);
    check("tair_grpc_response_bytes_qps", &rpcTags, 37);
    for (auto status : {TairKvcmQueryStatus::WINDOW_EXPIRED,
                        TairKvcmQueryStatus::RPC_ERROR,
                        TairKvcmQueryStatus::BUSINESS_ERROR}) {
        batch.data->tairKvcmResults[0].status = status;
        batch.queries[0].sent = status != TairKvcmQueryStatus::WINDOW_EXPIRED;
        queryResource.reportBatch(batch);
        kmonitor::MetricsTags failedTags;
        failedTags.AddTag("status", tairKvcmQueryStatusName(status));
        check("tair_query_failed_qps", &failedTags, 1);
    }

}

TEST_F(TairKvcmQueryTest, runtimeSizeBoundsRoundTripWithoutTrafficLimits)
{
    SchedulerConfigR::Config config;
    ASSERT_TRUE(config.valid());
    autil::legacy::FromJsonString(config, R"({"tair_query_max_objects":65536,"tair_query_max_http_bytes":8388608})");
    SchedulerConfigR::Config restored;
    const auto json = autil::legacy::ToJsonString(config);
    autil::legacy::FromJsonString(restored, json);
    EXPECT_EQ(65536u, restored.tairQueryMaxObjects);
    EXPECT_EQ(8388608u, restored.tairQueryMaxHttpBytes);
    EXPECT_EQ(std::string::npos, json.find("tair_query_max_qps"));
    EXPECT_EQ(std::string::npos, json.find("tair_query_max_inflight"));
    restored.tairQueryMaxObjects = 0; EXPECT_FALSE(restored.valid());
}

TEST_F(TairKvcmQueryTest, queryTokensObserveOutstandingCallbacksWithoutRejectingTraffic)
{
    ASSERT_EQ(0u, tairKvcmQueriesInFlight());
    std::vector<std::shared_ptr<void>> queries;
    for (size_t i = 0; i < 100; ++i) queries.push_back(trackTairKvcmQuery());
    EXPECT_EQ(100u, tairKvcmQueriesInFlight());
    auto callback = queries.front();
    queries.clear();
    EXPECT_EQ(1u, tairKvcmQueriesInFlight());
    callback.reset(); EXPECT_EQ(0u, tairKvcmQueriesInFlight());
}

TEST_F(TairKvcmQueryTest, firstEffectiveWorkerDefinesPoolInstanceIdLikeFlexlb)
{
    MasterPendingCacheR pending;
    auto firstConfig = batchConfig();
    auto secondConfig = std::make_shared<NaviSchedTairKvcmConfig>(*firstConfig);
    firstConfig->instanceId = "first_pool_instance";
    secondConfig->instanceId = "second_pool_instance";
    auto first = residentNode(pending, firstConfig);
    auto second = residentNode(pending, secondConfig);
    auto data = scheduleData(first);
    NaviSchedNodeSnapshot firstSnapshot, secondSnapshot;
    firstSnapshot.info = std::make_shared<const NaviSchedNodeInfo>(first->_info);
    secondSnapshot.info = std::make_shared<const NaviSchedNodeInfo>(second->_info);
    data->context->setNodes({first, second}, {firstSnapshot, secondSnapshot});
    MasterTairQueryR resource;
    resource.buildQueries(*data, data->context->_requests, 1);
    ASSERT_EQ(1u, data->tairKvcmQueries.size());
    EXPECT_EQ("first_pool_instance", data->tairKvcmQueries[0].policy->instanceId);
    EXPECT_EQ((std::vector<std::vector<size_t>>{{0, 1}}), data->tairKvcmNodeIndexes);
    EXPECT_EQ("second_pool_instance", secondConfig->instanceId);
    auto reordered = scheduleData(second);
    reordered->context->setNodes({second, first}, {secondSnapshot, firstSnapshot});
    resource.buildQueries(*reordered, reordered->context->_requests, 2);
    ASSERT_EQ(1u, reordered->tairKvcmQueries.size());
    EXPECT_EQ("second_pool_instance", reordered->tairKvcmQueries[0].policy->instanceId);
}

TEST_F(TairKvcmQueryTest, identicalHandshakeLayoutsShareOneBatchLookup)
{
    MasterPendingCacheR pending;
    auto first = batchConfig();
    auto second = std::make_shared<NaviSchedTairKvcmConfig>(*first);
    auto a = residentNode(pending, first);
    auto b = residentNode(pending, second);
    auto data = scheduleData(a);
    NaviSchedNodeSnapshot sa, sb;
    sa.info = std::make_shared<const NaviSchedNodeInfo>(a->_info);
    sb.info = std::make_shared<const NaviSchedNodeInfo>(b->_info);
    data->context->setNodes({a, b}, {sa, sb});
    MasterTairQueryR kernel;
    kernel.buildQueries(*data, data->context->_requests, data->context->_scheduleId);
    ASSERT_EQ(1u, data->tairKvcmQueries.size());
    EXPECT_EQ((std::vector<std::vector<size_t>>{{0, 1}}), data->tairKvcmNodeIndexes);
    EXPECT_EQ(8u, data->tairKvcmQueries.front().objects.size());
}

TEST_F(TairKvcmQueryTest, batchLayoutRequiresAllPpObjectsAndKeepsFullAndMambaSeparate)
{
    const auto config = batchConfig();
    ASSERT_TRUE(config->valid());
    auto requests = batchRequests();
    TairKvcmQueryInput input;
    prepareTairKvcmBatchQuery(config, requests, input);
    ASSERT_EQ(config, input.policy);
    ASSERT_EQ(8u, input.objects.size());
    ASSERT_EQ(4u, input.logicalGroups.size());
    EXPECT_EQ(12, input.objects[0].blockKey);
    EXPECT_EQ("group0_pp0", input.objects[0].locationSpecName);
    EXPECT_EQ(12, input.objects[1].blockKey);
    EXPECT_EQ("group0_pp1", input.objects[1].locationSpecName);
    TairKvcmQueryResult result;
    result.status = TairKvcmQueryStatus::OK;
    result.remoteHits.assign(input.objects.size(), 1);
    auto ready = getTairKvcmReadyBlocks(input, result);
    EXPECT_EQ((absl::flat_hash_set<uint64_t>{11, 12, 13, 14}), ready.full);
    EXPECT_EQ((absl::flat_hash_set<uint64_t>{12, 14}), ready.tail);
    MasterPendingCacheR matcher;
    int64_t matched = -1;
    EXPECT_EQ(0u, matcher.getMatchedBlockCounts(1, requests, 64, ready.full,
                                               ready.full, ready.tail, true, &matched));
    EXPECT_EQ(4, matched);
    // One absent physical Mamba object invalidates that logical checkpoint.
    result.remoteHits[7] = 0;
    ready = getTairKvcmReadyBlocks(input, result);
    EXPECT_EQ((absl::flat_hash_set<uint64_t>{11, 12, 13, 14}), ready.full);
    EXPECT_EQ((absl::flat_hash_set<uint64_t>{12}), ready.tail);
    matcher.getMatchedBlockCounts(1, requests, 64, ready.full, ready.full,
                                   ready.tail, true, &matched);
    EXPECT_EQ(2, matched);
    // One absent Full object invalidates both base blocks in that group.
    result.remoteHits[0] = 0;
    ready = getTairKvcmReadyBlocks(input, result);
    EXPECT_EQ((absl::flat_hash_set<uint64_t>{13, 14}), ready.full);
    matcher.getMatchedBlockCounts(1, requests, 64, ready.full, ready.full,
                                   ready.tail, true, &matched);
    EXPECT_EQ(0, matched);
    result.status = TairKvcmQueryStatus::RPC_ERROR;
    EXPECT_TRUE(getTairKvcmReadyBlocks(input, result).full.empty());
    EXPECT_TRUE(getTairKvcmReadyBlocks(input, result).tail.empty());
}

TEST_F(TairKvcmQueryTest, everyFullGroupMustHitBeforeItsBaseBlockCanBeReused)
{
    auto config = batchConfig();
    auto full = config->groups[0];
    full.groupId = 2;
    full.hashBlockScale = 1;
    full.locationSpecNames = {"full2_pp0", "full2_pp1"};
    config->groups.push_back(full);
    ASSERT_TRUE(config->valid());
    TairKvcmQueryInput input;
    prepareTairKvcmBatchQuery(config, batchRequests(), input);
    ASSERT_EQ(16u, input.objects.size());
    TairKvcmQueryResult result;
    result.status = TairKvcmQueryStatus::OK;
    result.remoteHits.assign(input.objects.size(), 1);
    result.remoteHits[10] = 0; // Second Full group, key 12, physical node 0.
    auto ready = getTairKvcmReadyBlocks(input, result);
    EXPECT_EQ((absl::flat_hash_set<uint64_t>{11, 13, 14}), ready.full);
    EXPECT_EQ((absl::flat_hash_set<uint64_t>{12, 14}), ready.tail);
    result.remoteHits.pop_back();
    EXPECT_TRUE(getTairKvcmReadyBlocks(input, result).full.empty());
}

TEST_F(TairKvcmQueryTest, batchDeduplicatesSharedPrefixesAcrossRequestsAndUsesSignedKeys)
{
    auto config = batchConfig(32);
    ASSERT_TRUE(config->valid());
    auto requests = batchRequests(128);
    for (auto &item : requests)
        item.request.blockHashes = std::make_shared<NaviSchedBlockHashes>(
            NaviSchedBlockHashes{11, 12, 13, std::numeric_limits<uint64_t>::max()});
    TairKvcmQueryInput input;
    const auto start = autil::TimeUtility::monotonicTimeUs();
    prepareTairKvcmBatchQuery(config, requests, input);
    const auto prepareUs = autil::TimeUtility::monotonicTimeUs() - start;
    ASSERT_EQ(128u, input.objects.size());
    ASSERT_EQ(4u, input.logicalGroups.size());
    EXPECT_EQ(-1, input.objects[32].blockKey);
    std::string body;
    ASSERT_TRUE(makeTairKvcmGrpcQueryRequest(input, body, 4 * 1024 * 1024));
    std::cout << "[tair-batch-prepare] requests=128 physical_nodes=32 objects="
              << input.objects.size() << " prepare_us=" << prepareUs
              << " request_bytes=" << body.size() << std::endl;
    requests[0].request.blockSize = 128;
    prepareTairKvcmBatchQuery(config, requests, input);
    EXPECT_EQ(128u, input.objects.size());
    prepareTairKvcmBatchQuery(nullptr, requests, input);
    EXPECT_TRUE(input.objects.empty());
    EXPECT_FALSE(input.policy);
}

TEST_F(TairKvcmQueryTest, representativeBatchPreparationAndSerializationCost)
{
    const auto config = batchConfig(32);
    for (size_t count : {size_t(1), size_t(32), size_t(128)}) {
        for (size_t blocks : {size_t(128), size_t(512), size_t(2048)}) {
            auto requests = batchRequests(count);
            auto hashes = std::make_shared<NaviSchedBlockHashes>(blocks);
            std::iota(hashes->begin(), hashes->end(), uint64_t(1));
            for (auto &item : requests) {
                item.request.blockHashes = hashes;
                item.request.promptTokenCount = static_cast<int64_t>((blocks + 1) * 64);
            }
            TairKvcmQueryInput input;
            const auto begin = autil::TimeUtility::monotonicTimeUs();
            prepareTairKvcmBatchQuery(config, requests, input);
            const auto prepared = autil::TimeUtility::monotonicTimeUs();
            std::string body;
            if (!input.exceedsObjectBudget) {
                ASSERT_TRUE(makeTairKvcmGrpcQueryRequest(input, body, 4 * 1024 * 1024));
            }
            const auto end = autil::TimeUtility::monotonicTimeUs();
            EXPECT_EQ(blocks > 1024, input.exceedsObjectBudget);
            EXPECT_LE(input.objects.size(), TAIR_KVCM_MAX_QUERY_OBJECTS);
            std::cout << "[tair-query-realistic-cost] requests=" << count
                      << " base_hashes=" << blocks << " physical_nodes=32 objects="
                      << input.objects.size() << " capped=" << input.exceedsObjectBudget
                      << " prepare_us=" << prepared - begin << " serialize_us=" << end - prepared
                      << " request_bytes=" << body.size() << std::endl;
        }
    }
}

TEST_F(TairKvcmQueryTest, commonMatcherCombinesLocalAndRemoteFullAndMambaWithinOneBatch)
{
    const auto requests = batchRequests();
    MasterPendingCacheR matcher;
    const MasterPendingCacheR::PhysicalBlockSet localFull{11, 12};
    const MasterPendingCacheR::PhysicalBlockSet localTail{12};
    const MasterPendingCacheR::PhysicalBlockSet remoteFull{13, 14};
    const MasterPendingCacheR::PhysicalBlockSet remoteTail{14};
    const int64_t blockSize = 64;
    int64_t matched = -1;
    const auto match = [&](const MasterPendingCacheR::PhysicalBlockSet *full,
                           const MasterPendingCacheR::PhysicalBlockSet *tail,
                           const int64_t *startBlocks = nullptr) {
        return matcher.getMatchedBlockCounts(
            1, requests, blockSize, localFull, localFull, localTail, true, &matched,
            0, NaviSchedCacheKeyType::V6D, nullptr, nullptr, startBlocks, nullptr,
            2, false, nullptr, nullptr, nullptr, full, tail);
    };
    EXPECT_EQ(0u, match(nullptr, nullptr));
    EXPECT_EQ(2, matched);
    EXPECT_EQ(0u, match(&remoteFull, nullptr));
    EXPECT_EQ(2, matched); // Full alone cannot supply the last Mamba checkpoint.
    EXPECT_EQ(0u, match(&remoteFull, &remoteTail));
    EXPECT_EQ(4, matched);
    const int64_t startBlock = 2;
    EXPECT_EQ(0u, match(&remoteFull, &remoteTail, &startBlock));
    EXPECT_EQ(2, matched); // A suffix contributes only blocks after the local hit.
    EXPECT_EQ(0u, match(nullptr, nullptr));
    EXPECT_EQ(2, matched); // Remote observations never persist into another batch.
}

TEST_F(TairKvcmQueryTest, remoteMambaCanCompleteAFullPrefixAlreadyPresentLocally)
{
    const auto requests = batchRequests();
    MasterPendingCacheR matcher;
    const MasterPendingCacheR::PhysicalBlockSet localFull{11, 12, 13, 14};
    const MasterPendingCacheR::PhysicalBlockSet localTail{12};
    const MasterPendingCacheR::PhysicalBlockSet remoteTail{14};
    int64_t matched = -1;
    EXPECT_EQ(0u, matcher.getMatchedBlockCounts(
        1, requests, 64, localFull, localFull, localTail, true, &matched,
        0, NaviSchedCacheKeyType::V6D, nullptr, nullptr, nullptr, nullptr,
        2, false, nullptr, nullptr, nullptr, nullptr, &remoteTail));
    EXPECT_EQ(4, matched);
}

TEST_F(TairKvcmQueryTest, scheduleContextConsumesRemoteCacheBeforeFreezingItsCostInputs)
{
    MasterPendingCacheR pending;
    auto node = residentNode(pending, batchConfig());
    auto &localV6d = node->_v6dCacheState;
    NaviSchedNodeSnapshot snapshot;
    snapshot.info = std::make_shared<const NaviSchedNodeInfo>(node->_info);
    auto remote = std::make_shared<NaviSchedExternalCacheReadyBlocks>();
    remote->tail = {14};
    NaviSchedScheduleContext context;
    context._requests = batchRequests();
    context.setNodes({node}, {snapshot});
    ASSERT_TRUE(context.setRemoteCacheReadyBlocks(0, remote));
    EXPECT_EQ(4, context.getCacheHitBlockCount(0, 0));
    const auto inputs = context.getExecutionCacheInputs(0);
    ASSERT_TRUE(inputs);
    ASSERT_EQ(1u, inputs->size());
    EXPECT_EQ(256, inputs->front().computedTokens);
    EXPECT_EQ(128, inputs->front().externalReadTokens);
    const auto frozen = std::atomic_load(&context._scheduleNodes[0]._executionCacheInputs);
    ASSERT_TRUE(frozen->observation.beforeTairBlockCounts);
    EXPECT_EQ((std::vector<int64_t>{2}), *frozen->observation.beforeTairBlockCounts);
    EXPECT_FALSE(context.setRemoteCacheReadyBlocks(0, nullptr));
    // A different schedule round sees the original local state, without Tair.
    NaviSchedScheduleContext next;
    next._requests = batchRequests();
    next.setNodes({node}, {snapshot});
    EXPECT_EQ(2, next.getCacheHitBlockCount(0, 0));
    EXPECT_TRUE(localV6d._tailReadyBlockHashes.empty());
}


TEST_F(TairKvcmQueryTest, cacheOnlyCheckpointLookupPreservesStrictExecutionValidation)
{
    MasterPendingCacheR pending;
    auto node = residentNode(pending, batchConfig());
    auto requests = batchRequests();
    requests[0].request.explicitCacheCheckpointPosition = 200;
    auto remote = std::make_shared<NaviSchedExternalCacheReadyBlocks>(); remote->tail = {14};
    std::vector<TPUPrefillRequestInput> inputs;
    TPUPrefillCacheObservationBatch observation;
    observation.beforeTairBlockCounts.emplace();
    // Geometry still requires the real execution policy and must fail without it.
    EXPECT_FALSE(node->getExecutionCacheInputs(requests, inputs, nullptr, &observation, remote.get()));
    // Unified cache-only lookup shares the exact same batch match and observation.
    ASSERT_TRUE(node->getExecutionCacheInputs(requests, inputs, nullptr, &observation, remote.get(), false));
    ASSERT_EQ(1u, inputs.size()); EXPECT_EQ(256, inputs[0].computedTokens);
    EXPECT_EQ(128, inputs[0].externalReadTokens);
    ASSERT_TRUE(observation.beforeTairBlockCounts);
    EXPECT_EQ((std::vector<int64_t>{2}), *observation.beforeTairBlockCounts);
    requests[0].request.explicitCacheCheckpointPosition = 321;
    EXPECT_FALSE(node->getExecutionCacheInputs(requests, inputs, nullptr, &observation, remote.get(), false));
    requests[0].request.explicitCacheCheckpointPosition = -1;
    EXPECT_FALSE(node->getExecutionCacheInputs(requests, inputs, nullptr, &observation, remote.get(), false));
    requests[0].request.explicitCacheCheckpointPosition = 200;
    auto published = std::make_shared<MasterStateR::NodeState::PublishedSnapshot>();
    published->prefillExecutionPolicy.emplace();
    published->prefillExecutionPolicy->schedulerBlockSize = 128;
    node->_publishedSnapshot = published;
    ASSERT_TRUE(node->getExecutionCacheInputs(requests, inputs, nullptr, &observation, remote.get()));
    EXPECT_EQ(128, inputs[0].sharedPrefixBoundary);
    EXPECT_EQ(256, inputs[0].computedTokens);
}

TEST_F(TairKvcmQueryTest, masterConfigUpdatesQueryLimitsWithoutAgentSynchronization)
{
    SchedulerConfigR resource;
    const auto before = resource.get();
    SchedulerConfigR::Request request;
    request.update = true;
    request.configJson = R"({"tair_query_max_objects":65536,"tair_query_max_http_bytes":8388608})";
    SchedulerConfigResponseDataPtr response;
    bool waiting = true;
    ASSERT_TRUE(resource.processRequest(request, response, waiting));
    ASSERT_TRUE(response);
    ASSERT_TRUE(response->accepted) << response->error;
    EXPECT_FALSE(waiting);
    const auto after = resource.get();
    EXPECT_EQ(before.generation + 1, after.generation);
    EXPECT_EQ(before.agentConfigHash, after.agentConfigHash);
    EXPECT_EQ(65536u, after.config.tairQueryMaxObjects);
    EXPECT_EQ(8388608u, after.config.tairQueryMaxHttpBytes);
    request.configJson = R"({"tair_query_max_qps":0})";
    ASSERT_TRUE(resource.processRequest(request, response, waiting));
    ASSERT_TRUE(response);
    EXPECT_FALSE(response->accepted);
    EXPECT_EQ(after.generation, resource.get().generation);
}
namespace {
namespace wire = kv_cache_manager::proto::meta;
using test::KvcmGrpcMock;
std::string leaderReply(const std::string &host, int port)
{
    wire::GetClusterInfoResponse response;
    response.mutable_header()->mutable_status()->set_code(wire::OK);
    response.mutable_leader_endpoint()->set_host(host);
    response.mutable_leader_endpoint()->set_meta_rpc_port(port);
    response.mutable_leader_endpoint()->set_meta_http_port(1); // Must never be used.
    return response.SerializeAsString();
}
std::string lookupReply(const std::string &payload, bool missingLast = false,
                        wire::ErrorCode code = wire::OK)
{
    wire::GetCacheLocationsByBackendRequest request;
    if (!request.ParseFromString(payload)) throw std::runtime_error("bad protobuf request");
    wire::GetCacheLocationsByBackendResponse response;
    response.mutable_header()->mutable_status()->set_code(code);
    for (int i = 0; i < request.block_keys_size(); ++i) {
        auto entry = response.add_key_locations();
        if (missingLast && i + 1 == request.block_keys_size()) continue;
        auto location = entry->add_locations(); location->set_type(wire::ST_TAIRMEMPOOL);
        auto spec = location->add_location_specs();
        spec->set_name(request.location_spec_names(i)); spec->set_uri("pace://test/remote");
    }
    return response.SerializeAsString();
}
std::shared_ptr<TairKvcmGrpcClientR> clientFor(int port, const std::string &host = "127.0.0.1")
{
    return test::gigQueryClient(port, host);
}
navi::KernelTesterPtr queryTester(const std::shared_ptr<TairKvcmGrpcClientR> &client)
{
    navi::KernelTesterBuilder builder;
    if (client->_gigClientR) builder.resource(client->_gigClientR->shared_from_this());
    return builder.resource(std::static_pointer_cast<navi::Resource>(client))
        .kernel(MasterTairQueryKernel::KERNEL_ID)
        .input("context").output("queried_context").build();
}
}

TEST_F(TairKvcmQueryTest, grpcQueriesBeforeCostPreparationAndReusesLeaderAndConnection)
{
    KvcmGrpcMock *serverPtr = nullptr;
    KvcmGrpcMock server([&](const auto &method, const auto &payload) {
        return KvcmGrpcMock::Reply{method == TAIR_KVCM_GRPC_GET_CLUSTER_INFO
            ? leaderReply("127.0.0.1", serverPtr->port) : lookupReply(payload), grpc::Status::OK};
    });
    serverPtr = &server;
    auto client = clientFor(server.port);
    auto tester = queryTester(client); ASSERT_TRUE(tester);
    auto kernel = dynamic_cast<MasterTairQueryKernel *>(tester->getKernel()); ASSERT_TRUE(kernel);
    auto pipe = std::make_shared<NotifyingPipe>(kernel->_completionPipe); kernel->_completionPipe = pipe;
    MasterPendingCacheR pending;
    auto config = batchConfig();
    // The pool's query topology is Resource-owned, not taken from this Worker.
    config->managerEndpoint = "http://127.0.0.1:1";
    auto node = residentNode(pending, config);
    auto data = scheduleData(node); const auto deadlineUs = data->context->getScheduleDeadlineUs();
    
    ASSERT_TRUE(tester->setInput("context", data)); ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(pipe->waitFor(1)); ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(pipe->waitFor(2)); ASSERT_TRUE(tester->compute());
    navi::DataPtr output; bool eof = false;
    ASSERT_TRUE(tester->getOutput("queried_context", output, eof)); EXPECT_EQ(data, output);
    ASSERT_EQ(1u, data->tairKvcmResults.size());
    EXPECT_EQ(TairKvcmQueryStatus::OK, data->tairKvcmResults[0].status);
    EXPECT_FALSE(data->context->_prepared);
    EXPECT_EQ(deadlineUs, data->context->getScheduleDeadlineUs());
    MasterStateR state; MasterScheduleR schedule; MasterSchedulePrepareKernel prepare;
    const_cast<MasterStateR *&>(prepare._masterStateR) = &state;
    const_cast<MasterScheduleR *&>(prepare._masterScheduleR) = &schedule;
    ASSERT_TRUE(prepare.prepareContext(data));
    EXPECT_EQ(4, data->context->getCacheHitBlockCount(0, 0));
    auto inputs = data->context->getExecutionCacheInputs(0); ASSERT_TRUE(inputs);
    EXPECT_EQ(256, inputs->front().computedTokens);
    EXPECT_EQ(128, inputs->front().externalReadTokens);
    ASSERT_EQ(2u, server.requests.size());
    wire::GetClusterInfoRequest discovery; ASSERT_TRUE(discovery.ParseFromString(server.requests[0].second));
    EXPECT_TRUE(discovery.instance_id().empty());
    wire::GetCacheLocationsByBackendRequest lookup; ASSERT_TRUE(lookup.ParseFromString(server.requests[1].second));
    EXPECT_EQ(config->instanceId, lookup.instance_id());
    auto pool = test::unaryPool(*client); ASSERT_TRUE(pool);
    ASSERT_EQ(1u, pool->_connectionMap.size()); auto connection = pool->_connectionMap.begin()->second;
    kernel->_queryR->_service->discoveryStartedUs = 1;
    auto next = scheduleData(node);
    ASSERT_TRUE(tester->setInput("context", next)); ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(pipe->waitFor(3)); ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(tester->getOutput("queried_context", output, eof)); EXPECT_EQ(next, output);
    EXPECT_EQ(3u, server.requests.size()); EXPECT_EQ(TAIR_KVCM_GRPC_GET_REMOTE_LOCATIONS, server.requests[2].first);
    EXPECT_EQ(4, next->context->getCacheHitBlockCount(0, 0));
    ASSERT_EQ(1u, pool->_connectionMap.size()); EXPECT_EQ(connection, pool->_connectionMap.begin()->second);
    EXPECT_EQ(config->instanceId, node->_info.tairKvcmConfig->instanceId);
}

TEST_F(TairKvcmQueryTest, resourceNamespaceOverrideUsesTheNativeHashBlockSize)
{
    MasterPendingCacheR pending; auto config = batchConfig(); auto node = residentNode(pending, config);
    MasterTairQueryR queryResource; auto client = clientFor(1);
    client->_topology.instanceNamespace = "configured_base"; queryResource._grpcClientR = client.get();
    auto data = scheduleData(node); queryResource.buildQueries(*data, data->context->_requests, 1);
    ASSERT_EQ(1u, data->tairKvcmQueries.size());
    EXPECT_EQ("configured_base_64", data->tairKvcmQueries[0].policy->instanceId);
    EXPECT_EQ("isolated_batch", config->instanceId);
    EXPECT_TRUE(test::unaryPool(*client)->_connectionMap.empty());
}

TEST_F(TairKvcmQueryTest, grpcFailureKeepsLocalHitsAndQueuedRoundStillCompletes)
{
    KvcmGrpcMock server([](const auto &, const auto &) {
        return KvcmGrpcMock::Reply{{}, grpc::Status(grpc::StatusCode::UNAVAILABLE, "synthetic")};
    });
    auto client = clientFor(server.port); auto tester = queryTester(client); ASSERT_TRUE(tester);
    auto kernel = dynamic_cast<MasterTairQueryKernel *>(tester->getKernel()); ASSERT_TRUE(kernel);
    
    auto pipe = std::make_shared<NotifyingPipe>(kernel->_completionPipe); kernel->_completionPipe = pipe;
    MasterPendingCacheR pending; auto data = scheduleData(residentNode(pending, batchConfig()));
    auto queued = scheduleData(residentNode(pending, nullptr));
    ASSERT_TRUE(tester->setInput("context", data)); ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(tester->setInput("context", queued)); ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(pipe->waitFor(1)); ASSERT_TRUE(tester->compute());
    navi::DataPtr output; bool eof = false;
    ASSERT_TRUE(tester->getOutput("queried_context", output, eof)); EXPECT_EQ(data, output);
    EXPECT_EQ(TairKvcmQueryStatus::RPC_ERROR, data->tairKvcmResults[0].status);
    EXPECT_EQ(2, data->context->getCacheHitBlockCount(0, 0));
    ASSERT_TRUE(tester->compute()); ASSERT_TRUE(tester->getOutput("queried_context", output, eof));
    EXPECT_EQ(queued, output); EXPECT_EQ(2, queued->context->getCacheHitBlockCount(0, 0));
    EXPECT_EQ(navi::EC_NONE, tester->getErrorCode()); EXPECT_EQ(1u, server.requests.size());
    EXPECT_EQ(0u, tairKvcmQueriesInFlight());
}

TEST_F(TairKvcmQueryTest, absentTopologyMissingNativeCapabilityOrOfflineNodeCreatesNoGrpcCall)
{
    for (int scenario = 0; scenario < 3; ++scenario) {
        auto client = std::make_shared<TairKvcmGrpcClientR>();
        if (scenario != 0) client = clientFor(1);
        auto tester = queryTester(client); ASSERT_TRUE(tester);
        MasterPendingCacheR pending; auto config = scenario == 1 ? nullptr : batchConfig();
        auto node = residentNode(pending, config); if (scenario == 2) node->_info.online = false;
        auto data = scheduleData(node);
        ASSERT_TRUE(tester->setInput("context", data)); ASSERT_TRUE(tester->compute());
        navi::DataPtr output; bool eof = false;
        ASSERT_TRUE(tester->getOutput("queried_context", output, eof)); EXPECT_EQ(data, output);
        EXPECT_TRUE(data->tairKvcmQueries.empty());
        if (scenario == 0) EXPECT_FALSE(client->_transportReady);
        else EXPECT_TRUE(test::unaryPool(*client)->_connectionMap.empty());
    }
}

TEST_F(TairKvcmQueryTest, oversizedBatchSkipsBeforeGrpcThreadsOrChannelCreation)
{
    auto client = clientFor(1); auto tester = queryTester(client); ASSERT_TRUE(tester);
    MasterPendingCacheR pending; auto data = scheduleData(residentNode(pending, batchConfig(32)));
    NaviSchedBlockHashes hashes(2048); std::iota(hashes.begin(), hashes.end(), uint64_t(1));
    data->context->_requests[0].request.blockHashes = std::make_shared<const NaviSchedBlockHashes>(hashes);
    data->context->_requests[0].request.promptTokenCount = 64 * 2049;
    ASSERT_TRUE(tester->setInput("context", data)); ASSERT_TRUE(tester->compute());
    navi::DataPtr output; bool eof = false;
    ASSERT_TRUE(tester->getOutput("queried_context", output, eof));
    ASSERT_EQ(1u, data->tairKvcmResults.size()); EXPECT_EQ(TairKvcmQueryStatus::TOO_LARGE, data->tairKvcmResults[0].status);
    EXPECT_TRUE(test::unaryPool(*client)->_connectionMap.empty());
}

TEST_F(TairKvcmQueryTest, discoveryAndLookupUseDifferentIpsAndOnlyReturnedRpcPort)
{
    KvcmGrpcMock leader([](const auto &method, const auto &payload) {
        return KvcmGrpcMock::Reply{lookupReply(payload), grpc::Status::OK};
    }, "127.0.0.2");
    KvcmGrpcMock discovery([&](const auto &method, const auto &) {
        return KvcmGrpcMock::Reply{leaderReply("127.0.0.2", leader.port), grpc::Status::OK};
    });
    auto client = clientFor(discovery.port); auto tester = queryTester(client); ASSERT_TRUE(tester);
    auto kernel = dynamic_cast<MasterTairQueryKernel *>(tester->getKernel()); ASSERT_TRUE(kernel);
    auto pipe = std::make_shared<NotifyingPipe>(kernel->_completionPipe); kernel->_completionPipe = pipe;
    MasterPendingCacheR pending; auto data = scheduleData(residentNode(pending, batchConfig()));
    ASSERT_TRUE(tester->setInput("context", data)); ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(pipe->waitFor(1)); ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(pipe->waitFor(2)); ASSERT_TRUE(tester->compute());
    navi::DataPtr output; bool eof = false; ASSERT_TRUE(tester->getOutput("queried_context", output, eof));
    EXPECT_EQ(TairKvcmQueryStatus::OK, data->tairKvcmResults[0].status);
    ASSERT_EQ(1u, discovery.requests.size()); ASSERT_EQ(1u, leader.requests.size());
    EXPECT_EQ(TAIR_KVCM_GRPC_GET_CLUSTER_INFO, discovery.requests[0].first);
    EXPECT_EQ(TAIR_KVCM_GRPC_GET_REMOTE_LOCATIONS, leader.requests[0].first);
}

TEST_F(TairKvcmQueryTest, sharedDeadlineContinuesSchedulingAndLateReplyCannotEmitAgain)
{
    KvcmGrpcMock server([](const auto &, const auto &) {
        return KvcmGrpcMock::Reply{{}, grpc::Status::OK, true};
    });
    auto client = clientFor(server.port); auto tester = queryTester(client); ASSERT_TRUE(tester);
    auto kernel = dynamic_cast<MasterTairQueryKernel *>(tester->getKernel()); ASSERT_TRUE(kernel);
    auto pipe = std::make_shared<NotifyingPipe>(kernel->_completionPipe); kernel->_completionPipe = pipe;
    MasterPendingCacheR pending; auto data = scheduleData(residentNode(pending, batchConfig()));
    data->context->_scheduleDeadlineUs = autil::TimeUtility::monotonicTimeUs() + 50000;
    ASSERT_TRUE(tester->setInput("context", data)); ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(pipe->waitFor(1)); ASSERT_TRUE(tester->compute());
    navi::DataPtr output; bool eof = false; ASSERT_TRUE(tester->getOutput("queried_context", output, eof));
    EXPECT_EQ(data, output); EXPECT_EQ(TairKvcmQueryStatus::WINDOW_EXPIRED, data->tairKvcmResults[0].status);
    EXPECT_EQ(2, data->context->getCacheHitBlockCount(0, 0));
    auto late = std::make_shared<TairKvcmGrpcCompletionData>(); late->round = kernel->_round;
    ASSERT_EQ(navi::EC_NONE, pipe->setData(late)); ASSERT_TRUE(tester->compute());
    EXPECT_FALSE(tester->getOutput("queried_context", output, eof)); EXPECT_EQ(navi::EC_NONE, tester->getErrorCode());
    server.release();
}

TEST_F(TairKvcmQueryTest, leaderChangeRetriesWithinSameRoundAndKeepsDeadlineAndAdmission)
{
    // FlexLB defaults to one retry sharing the original deadline and wire inputs.
    {
        KvcmGrpcMock leader([](const auto &, const auto &payload) {
            return KvcmGrpcMock::Reply{lookupReply(payload), grpc::Status::OK};
        }, "127.0.0.2");
        KvcmGrpcMock discovery([&](const auto &, const auto &) {
            return KvcmGrpcMock::Reply{leaderReply("127.0.0.2", leader.port), grpc::Status::OK};
        });
        KvcmGrpcMock retired([&](const auto &, const auto &payload) {
            return KvcmGrpcMock::Reply{lookupReply(payload, false, wire::SERVER_NOT_LEADER), grpc::Status::OK};
        });
        auto client = clientFor(discovery.port); auto tester = queryTester(client); ASSERT_TRUE(tester);
        auto kernel = dynamic_cast<MasterTairQueryKernel *>(tester->getKernel()); ASSERT_TRUE(kernel);
        
        auto pipe = std::make_shared<NotifyingPipe>(kernel->_completionPipe); kernel->_completionPipe = pipe;
        auto service = kernel->_queryR->_service;
        service->leader = {"127.0.0.1", retired.port};
        service->refreshRequired = false;
        MasterPendingCacheR pending; auto node = residentNode(pending, batchConfig());
        auto first = scheduleData(node);
        const auto deadline = first->context->getScheduleDeadlineUs();
        ASSERT_TRUE(tester->setInput("context", first)); ASSERT_TRUE(tester->compute());
        for (size_t i = 1; i <= 3; ++i) {
            ASSERT_TRUE(pipe->waitFor(i));
            EXPECT_EQ(1u, tairKvcmQueriesInFlight());
            ASSERT_TRUE(tester->compute());
        }
        navi::DataPtr output; bool eof = false;
        ASSERT_TRUE(tester->getOutput("queried_context", output, eof)); EXPECT_EQ(first, output);
        EXPECT_EQ(TairKvcmQueryStatus::OK, first->tairKvcmResults[0].status);
        EXPECT_EQ(4, first->context->getCacheHitBlockCount(0, 0));
        EXPECT_EQ(deadline, first->context->getScheduleDeadlineUs());
        auto observations = pipe->observations(); ASSERT_EQ(3u, observations.size());
        ASSERT_NE(nullptr, observations.front().token);
        for (const auto &rpc : observations) {
            EXPECT_EQ(deadline, rpc.deadlineUs);
            EXPECT_EQ(observations.front().token, rpc.token);
        }
        EXPECT_EQ(0u, tairKvcmQueriesInFlight());
        ASSERT_EQ(1u, retired.requests.size()); ASSERT_EQ(1u, discovery.requests.size());
        ASSERT_EQ(1u, leader.requests.size());
        EXPECT_EQ(retired.requests[0].second, leader.requests[0].second);
        EXPECT_EQ("127.0.0.2", service->leader.host); EXPECT_EQ(leader.port, service->leader.port);
        EXPECT_TRUE(node->_v6dCacheState._tailReadyBlockHashes.empty());
        // Successful recovery does not trip the failure backoff or rediscover next round.
        auto second = scheduleData(node); 
        ASSERT_TRUE(tester->setInput("context", second)); ASSERT_TRUE(tester->compute());
        ASSERT_TRUE(pipe->waitFor(4)); ASSERT_TRUE(tester->compute());
        ASSERT_TRUE(tester->getOutput("queried_context", output, eof)); EXPECT_EQ(second, output);
        EXPECT_EQ(TairKvcmQueryStatus::OK, second->tairKvcmResults[0].status);
        EXPECT_EQ(1u, discovery.requests.size()); EXPECT_EQ(2u, leader.requests.size());
    }
}

TEST_F(TairKvcmQueryTest, transportErrorsIncludingCodeNineNeverRetryCurrentQuery)
{
    for (auto code : {grpc::StatusCode::UNAVAILABLE, grpc::StatusCode::FAILED_PRECONDITION,
                      grpc::StatusCode::DEADLINE_EXCEEDED}) {
        KvcmGrpcMock server([&](const auto &, const auto &) {
            return KvcmGrpcMock::Reply{{}, grpc::Status(code, "synthetic")};
        });
        auto tester = queryTester(clientFor(server.port)); ASSERT_TRUE(tester);
        auto kernel = dynamic_cast<MasterTairQueryKernel *>(tester->getKernel()); ASSERT_TRUE(kernel);
        kernel->_queryR->_service->leader = {"127.0.0.1", server.port};
        auto pipe = std::make_shared<NotifyingPipe>(kernel->_completionPipe); kernel->_completionPipe = pipe;
        MasterPendingCacheR pending; auto data = scheduleData(residentNode(pending, batchConfig()));
        ASSERT_TRUE(tester->setInput("context", data)); ASSERT_TRUE(tester->compute());
        ASSERT_TRUE(pipe->waitFor(1)); ASSERT_TRUE(tester->compute());
        navi::DataPtr output; bool eof = false;
        ASSERT_TRUE(tester->getOutput("queried_context", output, eof));
        EXPECT_EQ(code == grpc::StatusCode::DEADLINE_EXCEEDED
            ? TairKvcmQueryStatus::WINDOW_EXPIRED : TairKvcmQueryStatus::RPC_ERROR,
            data->tairKvcmResults[0].status);
        EXPECT_EQ(1u, server.requests.size());
        EXPECT_EQ(2, data->context->getCacheHitBlockCount(0, 0));
        EXPECT_EQ(std::vector<uint8_t>(8, 0), data->tairKvcmResults[0].remoteHits);
        EXPECT_EQ(0u, tairKvcmQueriesInFlight());
    }
}

TEST_F(TairKvcmQueryTest, failedRediscoveryRetryIsBoundedAndKeepsLocalCache)
{
    KvcmGrpcMock *serverPtr = nullptr;
    KvcmGrpcMock server([&](const auto &method, const auto &payload) {
        return KvcmGrpcMock::Reply{method == TAIR_KVCM_GRPC_GET_CLUSTER_INFO
            ? leaderReply("127.0.0.1", serverPtr->port)
            : lookupReply(payload, false, wire::SERVER_NOT_LEADER), grpc::Status::OK};
    });
    serverPtr = &server;
    auto client = clientFor(server.port); auto tester = queryTester(client); ASSERT_TRUE(tester);
    auto kernel = dynamic_cast<MasterTairQueryKernel *>(tester->getKernel()); ASSERT_TRUE(kernel);
    
    auto pipe = std::make_shared<NotifyingPipe>(kernel->_completionPipe); kernel->_completionPipe = pipe;
    MasterPendingCacheR pending; auto data = scheduleData(residentNode(pending, batchConfig()));
    ASSERT_TRUE(tester->setInput("context", data)); ASSERT_TRUE(tester->compute());
    // Cold start: discover, lookup fails, rediscover, one lookup retry fails.
    for (size_t i = 1; i <= 4; ++i) {
        ASSERT_TRUE(pipe->waitFor(i)); ASSERT_TRUE(tester->compute());
    }
    navi::DataPtr output; bool eof = false;
    ASSERT_TRUE(tester->getOutput("queried_context", output, eof));
    EXPECT_EQ(TairKvcmQueryStatus::BUSINESS_ERROR, data->tairKvcmResults[0].status);
    EXPECT_EQ(2, data->context->getCacheHitBlockCount(0, 0));
    EXPECT_EQ(4u, server.requests.size()); EXPECT_TRUE(kernel->_queryR->_service->refreshRequired);
    EXPECT_EQ(0u, tairKvcmQueriesInFlight());
    EXPECT_FALSE(tester->compute()); // No input or callback can activate another attempt.
    EXPECT_FALSE(tester->getOutput("queried_context", output, eof));
    EXPECT_EQ(navi::EC_NONE, tester->getErrorCode());
}

TEST_F(TairKvcmQueryTest, rediscoveryCannotExtendDeadlineAndLateReplyCannotEmitTwice)
{
    KvcmGrpcMock discovery([](const auto &, const auto &) {
        return KvcmGrpcMock::Reply{{}, grpc::Status::OK, true};
    });
    KvcmGrpcMock retired([](const auto &, const auto &payload) {
        return KvcmGrpcMock::Reply{lookupReply(payload, false, wire::SERVER_NOT_LEADER), grpc::Status::OK};
    });
    auto client = clientFor(discovery.port); auto tester = queryTester(client); ASSERT_TRUE(tester);
    auto kernel = dynamic_cast<MasterTairQueryKernel *>(tester->getKernel()); ASSERT_TRUE(kernel);
    
    auto pipe = std::make_shared<NotifyingPipe>(kernel->_completionPipe); kernel->_completionPipe = pipe;
    kernel->_queryR->_service->leader = {"127.0.0.1", retired.port};
    kernel->_queryR->_service->refreshRequired = false;
    MasterPendingCacheR pending; auto data = scheduleData(residentNode(pending, batchConfig()));
    const auto deadline = autil::TimeUtility::monotonicTimeUs() + 500000;
    data->context->_scheduleDeadlineUs = deadline;
    ASSERT_TRUE(tester->setInput("context", data)); ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(pipe->waitFor(1)); ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(discovery.waitRequests(1));
    ASSERT_TRUE(pipe->waitFor(2)); ASSERT_TRUE(tester->compute());
    navi::DataPtr output; bool eof = false;
    ASSERT_TRUE(tester->getOutput("queried_context", output, eof)); EXPECT_EQ(data, output);
    EXPECT_EQ(TairKvcmQueryStatus::WINDOW_EXPIRED, data->tairKvcmResults[0].status);
    EXPECT_EQ(2, data->context->getCacheHitBlockCount(0, 0));
    const auto observations = pipe->observations(); ASSERT_EQ(2u, observations.size());
    EXPECT_EQ(deadline, observations[0].deadlineUs); EXPECT_EQ(deadline, observations[1].deadlineUs);
    EXPECT_EQ(observations[0].token, observations[1].token);
    EXPECT_EQ(1u, retired.requests.size()); EXPECT_EQ(1u, discovery.requests.size());
    discovery.release();
    auto late = std::make_shared<TairKvcmGrpcCompletionData>(); late->round = kernel->_round;
    ASSERT_EQ(navi::EC_NONE, pipe->setData(late)); ASSERT_TRUE(tester->compute());
    EXPECT_FALSE(tester->getOutput("queried_context", output, eof));
    EXPECT_EQ(navi::EC_NONE, tester->getErrorCode());
}

TEST_F(TairKvcmQueryTest, invalidArgumentAndMalformedResponseDoNotRetry)
{
    for (bool malformed : {false, true}) {
        KvcmGrpcMock server([&](const auto &, const auto &payload) {
            return KvcmGrpcMock::Reply{malformed ? "bad protobuf"
                : lookupReply(payload, false, wire::INVALID_ARGUMENT), grpc::Status::OK};
        });
        auto client = clientFor(server.port); auto tester = queryTester(client); ASSERT_TRUE(tester);
        auto kernel = dynamic_cast<MasterTairQueryKernel *>(tester->getKernel()); ASSERT_TRUE(kernel);
        
        auto pipe = std::make_shared<NotifyingPipe>(kernel->_completionPipe); kernel->_completionPipe = pipe;
        kernel->_queryR->_service->leader = {"127.0.0.1", server.port};
        kernel->_queryR->_service->refreshRequired = false;
        MasterPendingCacheR pending; auto data = scheduleData(residentNode(pending, batchConfig()));
        ASSERT_TRUE(tester->setInput("context", data)); ASSERT_TRUE(tester->compute());
        ASSERT_TRUE(pipe->waitFor(1)); ASSERT_TRUE(tester->compute());
        navi::DataPtr output; bool eof = false;
        ASSERT_TRUE(tester->getOutput("queried_context", output, eof));
        EXPECT_EQ(malformed ? TairKvcmQueryStatus::INVALID_RESPONSE : TairKvcmQueryStatus::BUSINESS_ERROR,
                  data->tairKvcmResults[0].status);
        EXPECT_EQ(1u, server.requests.size()); EXPECT_EQ(2, data->context->getCacheHitBlockCount(0, 0));
    }
}

TEST_F(TairKvcmQueryTest, recoveryCannotExtendAnExpiredCommonDeadlineOrInvalidateNewerLeader)
{
    auto client = clientFor(1);
    MasterTairQueryR resource; resource._grpcClientR = client.get();
    MasterTairQueryR::Batch batch;
    MasterPendingCacheR pending; batch.data = scheduleData(residentNode(pending, batchConfig()));
    batch.data->context->_scheduleDeadlineUs = autil::TimeUtility::monotonicTimeUs() - 1;
    batch.queries.resize(1); batch.queries[0].stage = MasterTairQueryR::RpcStage::LOOKUP;
    EXPECT_FALSE(resource.retry(batch, 0, "not_leader"));
    EXPECT_EQ(0u, batch.retryCount(0)); EXPECT_TRUE(test::unaryPool(*client)->_connectionMap.empty());
    auto service = resource._service;
    service->leader = {"127.0.0.2", 6381}; service->revision = 2; service->refreshRequired = false;
    batch.queries[0].service = service; batch.queries[0].serviceRevision = 1;
    resource.invalidateLeader(batch.queries[0]);
    EXPECT_FALSE(service->refreshRequired); EXPECT_EQ(2u, service->revision);
}

} // namespace navi_sched
