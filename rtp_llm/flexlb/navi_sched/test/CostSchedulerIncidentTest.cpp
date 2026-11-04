#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyResidualModel.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerGraph.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerKernelInternal.h"
#include "rtp_llm/flexlb/navi_sched/test/ScheduleReplayRunner.h"
#include "autil/TimeUtility.h"
#include "rapidjson/document.h"
#include "unittest/unittest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <sched.h>
#include <sstream>
#include <string>
#include <vector>

namespace navi_sched {

TEST(CostSchedulerParallelismTest, AutoUsesAvailableCpusAndExplicitValueWins)
{
    cpu_set_t available;
    CPU_ZERO(&available);
    ASSERT_EQ(0, sched_getaffinity(0, sizeof(available), &available));
    EXPECT_EQ(static_cast<size_t>(CPU_COUNT(&available)),
              CostSchedulerGraphBuilder::getWorkerCount(0));
    for (int32_t workers : {1, 2, 10, 64, 128}) {
        EXPECT_EQ(static_cast<size_t>(workers),
                  CostSchedulerGraphBuilder::getWorkerCount(workers));
    }
}

TEST(CostGradientIterationMetricsTest, AccumulatesOnlyCompletedIterations)
{
    NaviSchedScheduleContext context;
    EXPECT_EQ(0u, context._costGradientIterationCount);
    EXPECT_EQ(0, context._costGradientIterationTotalUs);
    EXPECT_EQ(0, context._costGradientIterationMaxUs);
    context.recordCostGradientIteration(1200);
    context.recordCostGradientIteration(800);
    context.recordCostGradientIteration(2500);
    EXPECT_EQ(3u, context._costGradientIterationCount);
    EXPECT_EQ(4500, context._costGradientIterationTotalUs);
    EXPECT_EQ(2500, context._costGradientIterationMaxUs);
}

class CostSchedulerIncidentTest : public TESTBASE {
protected:
    // Forensic arithmetic only: production must reject these historical models.
    static double historicalPrediction(const NaviSchedLatencyFeatures &features,
                                       const std::vector<double> &parameters)
    {
        return NonLinearPrefillModelR::predict(
            NonLinearPrefillModelR::makeInput(features),
            NonLinearPrefillModelR::toValues(parameters), parameters.size(), nullptr);
    }

    static std::unique_ptr<float[]> prepareHistoricalCosts(
        NaviSchedScheduleContext &context, const int64_t *tokens,
        double lambda, double alpha, double alphaDecay, double minAlpha,
        size_t loops, int64_t budgetUs, CostSchedulerKernelInternal::ReplayInfo &trace)
    {
        auto productionCosts = CostSchedulerKernelInternal::prepare(
            context, tokens, lambda, alpha, alphaDecay, minAlpha, loops, budgetUs, trace);
        EXPECT_FALSE(productionCosts);
        const auto count = context.requestCount();
        const auto matrixSize = count * context.nodeCount();
        auto costs = std::make_unique<float[]>(2 * matrixSize);
        bool rejectedModel = false;
        for (size_t n = 0; n < context.nodeCount(); ++n) {
            auto &node = context._scheduleNodes[n];
            const auto &parameters = node.getLatencyParameters();
            rejectedModel |= !NonLinearPrefillModelR::validateParameters(parameters);
            std::vector<int64_t> hitBlocks(count);
            if (!context.getCacheHitBlockCounts(n, hitBlocks)) {
                ADD_FAILURE() << "missing historical matches";
                return nullptr;
            }
            for (size_t r = 0; r < count; ++r) {
                const auto hit = hitBlocks[r] * context.getRequest(r).request.blockSize;
                costs[n * count + r] = NonLinearPrefillModelR::calculateRequestLinearCost(
                    parameters, tokens[r], hit);
                costs[matrixSize + n * count + r] =
                    NonLinearPrefillModelR::calculateWaitingRequestLinearCost(parameters, tokens[r], hit);
            }
            NaviSchedLatencyFeatures floor;
            if (!node.getUnifiedDecayFloorFeatures(floor)) {
                ADD_FAILURE() << "missing historical floor features";
                return nullptr;
            }
            node._existingUnifiedLatencyMs = node.decayUnifiedLatency(
                node.applyLatencyResidual(historicalPrediction(node.getWorkloadFeatures(), parameters)),
                node.applyLatencyResidual(historicalPrediction(floor, parameters)));
        }
        EXPECT_TRUE(rejectedModel);
        trace.init(context.isCollectingReplay(), lambda, alpha, alphaDecay, minAlpha, loops, budgetUs);
        return costs;
    }

    static int64_t number(const rapidjson::Value &row, const char *key)
    {
        return std::stoll(row[key].GetString());
    }

    static std::vector<double> array(const rapidjson::Value &value)
    {
        std::vector<double> result;
        for (const auto &item : value.GetArray()) {
            result.push_back(item.GetDouble());
        }
        return result;
    }

    void loadReplay(NaviSchedScheduleReplay &replay,
                    const std::string &fixture = "incident_selected_transfer.json")
    {
        std::ifstream input(GET_PRIVATE_TEST_DATA_PATH() +
                            "/scheduler/" + fixture);
        ASSERT_TRUE(input.good());
        const std::string text{std::istreambuf_iterator<char>(input), {}};
        rapidjson::Document data;
        data.Parse(text.c_str());
        ASSERT_FALSE(data.HasParseError());
        replay.scheduleTimeUs = number(data["nodes"][0], "schedule_time_us");
        replay.bizName = "incident_selected_pair";
        for (const auto &row : data["nodes"].GetArray()) {
            NaviSchedNodeInfo node;
            node.agentId = row["ip"].GetString();
            node.agentIdHash = replay.nodes.size() + 1;
            node.serverAddress = row["agent"].GetString();
            node.bizName = replay.bizName;
            node.online = true;
            node.unifiedLatencyPrediction = true;
            node.latencyModelTrusted = true;
            node.latencyFormula = NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
            node.latencyBinaryVersion = NonLinearPrefillModelR::BINARY_VERSION;
            node.latencyModelRevision = number(row, "model_revision");
            node.latencyParameters = array(row["state"]["parameters_after"]);
            node.latencyResidualParameters = array(row["state"]["residual_parameters_after"]);
            node.latencyResidualVersion = number(row, "residual_version");
            node.latencyResidualLastUpdateTimeUs = number(row, "residual_last_update_time_us");
            node.workloadReportReceiveTimeUs = replay.scheduleTimeUs - std::llround(
                std::stod(row["report_age_ms"].GetString()) * 1000.0);
            const auto requestCount = number(row, "request_count");
            auto &w = node.workload;
            w.batchCount = number(row, "batch_count") - 1;
            w.requestCount = number(row, "preceding_request_count");
            w.remainingPrefillRequestCount = number(row, "waiting_batch_size") - requestCount;
            w.remainingPrefillContextTokenCount = number(row, "waiting_hit_tokens") -
                                                  number(row, "floor_hit_tokens");
            w.remainingPrefillTokenCount = number(row, "waiting_miss_tokens") -
                                           number(row, "floor_miss_tokens");
            w.remainingPrefillTokenSquareSum = number(row, "waiting_miss_square_sum") -
                                               number(row, "floor_miss_square_sum");
            w.contextRemainingPrefillTokenProductSum = number(row, "waiting_hit_miss_sum") -
                                                       number(row, "floor_hit_miss_sum");
            w.prefillCapacitySplit = true;
            w.waitingCapacityPrefillRequestCount = w.remainingPrefillRequestCount;
            w.waitingCapacityPrefillContextTokenCount = w.remainingPrefillContextTokenCount;
            w.waitingCapacityPrefillTokenCount = w.remainingPrefillTokenCount;
            w.waitingCapacityPrefillTokenSquareSum = w.remainingPrefillTokenSquareSum;
            w.waitingCapacityContextRemainingPrefillTokenProductSum =
                w.contextRemainingPrefillTokenProductSum;
            w.waitingRequestCount = w.remainingPrefillRequestCount;
            w.runningRequestCount = w.requestCount - w.waitingRequestCount;
            w.maxRunningRequestCount = w.runningRequestCount;
            w.runningCapacitySlotCount = w.runningRequestCount;
            node.maxRunningRequestCount = w.maxRunningRequestCount;
            node.workloadRunningCapacitySlotCount = w.runningCapacitySlotCount;
            node.workloadFeatureSnapshotValid = true;
            ASSERT_TRUE(node.workloadFeatures.appendPrefillWorkload(w));
            ASSERT_TRUE(node.reservedWorkloadFeatures.appendCapacitySplitAggregate(
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
            replay.nodes.push_back(node);
            for (auto it = row["requests"].MemberBegin(); it != row["requests"].MemberEnd(); ++it) {
                NaviSchedScheduleItem item;
                item.request.requestId = std::stoull(it->name.GetString());
                item.request.bizName = replay.bizName;
                item.request.promptTokenCount = it->value.GetInt64();
                item.request.blockSize = 16;
                item.request.timeoutMs = 600000;
                item.remainingTimeoutMs = 600000;
                auto hashes = std::make_shared<NaviSchedBlockHashes>();
                for (int64_t i = 0; i < item.request.promptTokenCount / 16; ++i) {
                    hashes->push_back(replay.requests.size() * 1000 + i + 1);
                }
                item.request.blockHashes = hashes;
                replay.requests.push_back(item);
            }
        }
        replay.nodeMatches.resize(replay.requests.size());
        size_t requestIndex = 0;
        size_t nodeIndex = 0;
        for (const auto &row : data["nodes"].GetArray()) {
            for (auto it = row["requests"].MemberBegin(); it != row["requests"].MemberEnd(); ++it) {
                auto &matches = replay.nodeMatches[requestIndex++];
                matches.resize(replay.nodes.size());
                for (auto &match : matches) {
                    match.matchedBlockCount = 240;
                }
                if (row.HasMember("chosen_hit_blocks")) {
                    matches[nodeIndex].matchedBlockCount =
                        row["chosen_hit_blocks"][it->name.GetString()].GetInt();
                }
            }
            ++nodeIndex;
        }
    }
};

TEST_F(CostSchedulerIncidentTest, testRecordedPredictionsAndBoundedTransfer)
{
    NaviSchedScheduleReplay replay;
    loadReplay(replay);
    ASSERT_FALSE(HasFatalFailure());
    // Cross-node matches are unknown. Try multiple feasible counterfactuals;
    // only the original assignment's hits are historical observations.
    for (int alternateHitBlocks : {0, 120, 240}) {
        for (size_t i = 0; i < 8; ++i) {
            replay.nodeMatches[i][1].matchedBlockCount = alternateHitBlocks;
        }
        NaviSchedScheduleContext context;
        ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
        ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
        std::vector<int64_t> tokens(9);
        CostSchedulerKernelInternal::fillRequestTokenCounts(context, tokens.data(), nullptr);
        CostSchedulerKernelInternal::ReplayInfo replayInfo;
        auto costs = prepareHistoricalCosts(
            context, tokens.data(), 0.5, 1, 1, 0, 10, 0, replayInfo);
        ASSERT_TRUE(costs);
        std::vector<uint32_t> original(9, 0);
        original.back() = 1;
        std::vector<uint32_t> moved(9, 1);
        const std::vector<double> inverseTokens(9, 1.0);
        CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
        const auto objective = [&](const std::vector<uint32_t> &assignment, double lambda) {
            return CostSchedulerKernelInternal::forwardDiscrete(
                context, costs.get(), assignment.data(), tokens.data(),
                inverseTokens.data(), 9.0, lambda, 0.0, workspace);
        };
        const auto recorded = objective(original, 0.0);
        EXPECT_NEAR((8 * 39395.926994583 + 36716.463361480) / 9, recorded, 0.02);
        std::vector<double> oneHot(18, 0.0);
        for (size_t i = 0; i < original.size(); ++i) {
            oneHot[original[i] * 9 + i] = 1.0;
        }
        double secondTerms[2] = {};
        CostSchedulerKernelInternal::forwardNodeRange(
            context, costs.get(), oneHot.data(), tokens.data(), inverseTokens.data(),
            9.0, 0.5, 0.0, 0, 2, nullptr, nullptr, secondTerms);
        EXPECT_NEAR(39395.926994583, secondTerms[0] / 8, 0.02);
        EXPECT_NEAR(36716.463361480, secondTerms[1], 0.02);
        EXPECT_NEAR(35892.04355845216, context.getNode(0).getExistingUnifiedLatencyMs(), 0.01);
        EXPECT_LT(objective(moved, 0.0), 38533.36449322181 + 0.02);
        for (double lambda : {0.0, 0.5, 1.0}) {
            SCOPED_TRACE(alternateHitBlocks);
            SCOPED_TRACE(lambda);
            EXPECT_LT(objective(moved, lambda), objective(original, lambda) - 1.0);
        }
    }
}

TEST_F(CostSchedulerIncidentTest, testSingleTransferImprovesAcrossAllDestinationCacheHits)
{
    NaviSchedScheduleReplay replay;
    loadReplay(replay);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(2, replay.nodes.size());
    ASSERT_EQ(9, replay.requests.size());
    ASSERT_EQ("33.242.152.12", replay.nodes[0].agentId);
    ASSERT_EQ("215.74.13.89", replay.nodes[1].agentId);
    size_t movedIndex = replay.requests.size();
    for (size_t i = 0; i < replay.requests.size(); ++i) {
        if (replay.requests[i].request.requestId == 3641248014938683055ULL) {
            movedIndex = i;
        }
    }
    ASSERT_LT(movedIndex, 8);
    const auto maxHitBlocks = replay.requests[movedIndex].request.promptTokenCount / 16;
    ASSERT_GT(maxHitBlocks, 240);
    for (int64_t hitBlocks = 0; hitBlocks <= maxHitBlocks; ++hitBlocks) {
        SCOPED_TRACE(hitBlocks);
        replay.nodeMatches[movedIndex][1].matchedBlockCount = hitBlocks;
        NaviSchedScheduleContext context;
        ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
        ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
        std::vector<int64_t> tokens(9);
        CostSchedulerKernelInternal::fillRequestTokenCounts(context, tokens.data(), nullptr);
        CostSchedulerKernelInternal::ReplayInfo info;
        auto costs = prepareHistoricalCosts(
            context, tokens.data(), 0.5, 512, 1, 0, 10, 0, info);
        ASSERT_TRUE(costs);
        const std::vector<double> inverseTokens(9, 1.0);
        std::vector<uint32_t> original(9, 0);
        original.back() = 1;
        auto moved = original;
        moved[movedIndex] = 1;
        CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
        const auto objective = [&](const std::vector<uint32_t> &assignment) {
            return CostSchedulerKernelInternal::forwardDiscrete(
                context, costs.get(), assignment.data(), tokens.data(), inverseTokens.data(),
                9.0, 0.5, 0.0, workspace);
        };
        const auto before = objective(original);
        const auto after = objective(moved);
        // This bound comes from Python's independent signed-quadratic extrema,
        // including the worst residual. It does not assume unknown cache hits.
        EXPECT_LE(after - before, -137.2331041989831 + 0.02);
        std::vector<double> oneHot(18, 0.0);
        for (size_t i = 0; i < moved.size(); ++i) {
            oneHot[moved[i] * 9 + i] = 1.0;
        }
        double secondTerms[2] = {};
        CostSchedulerKernelInternal::forwardNodeRange(
            context, costs.get(), oneHot.data(), tokens.data(), inverseTokens.data(),
            9.0, 0.5, 0.0, 0, 2, nullptr, nullptr, secondTerms);
        EXPECT_NEAR(38922.910120692235, secondTerms[0] / 7, 0.02);
        EXPECT_LE(secondTerms[1] / 2, 38047.597611029145 + 0.02);
    }
}

TEST_F(CostSchedulerIncidentTest, testTightResidualBoundStillFavorsAbsorbingNode)
{
    NaviSchedScheduleReplay replay;
    loadReplay(replay, "incident_tight_absorption_transfer.json");
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(1791637099392441, replay.scheduleTimeUs);
    ASSERT_EQ(2, replay.nodes.size());
    ASSERT_EQ(5, replay.requests.size());
    ASSERT_EQ("11.18.53.136", replay.nodes[0].agentId);
    ASSERT_EQ("215.91.17.72", replay.nodes[1].agentId);
    const std::vector<uint32_t> original = {0, 0, 1, 1, 1};
    const std::vector<double> inverseTokens(5, 1.0);
    const uint64_t ids[] = {3641248014938682563ULL, 3641248014938682565ULL,
                            3641248014938682566ULL};
    const double targetBounds[] = {37075.31472840711, 37074.29175425699,
                                    37077.56535175018};
    const double objectiveBounds[] = {-163.54550541598616, -164.55721136999313,
                                       -161.7629914198664};
    for (size_t index = 0; index < 3; ++index) {
        const auto movedIndex = index + 2;
        ASSERT_EQ(ids[index], replay.requests[movedIndex].request.requestId);
        const auto maxHit = replay.requests[movedIndex].request.promptTokenCount / 16;
        for (int64_t hit = 0; hit <= maxHit; ++hit) {
            SCOPED_TRACE(index);
            SCOPED_TRACE(hit);
            replay.nodeMatches[movedIndex][0].matchedBlockCount = hit;
            NaviSchedScheduleContext context;
            ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
            ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
            std::vector<int64_t> tokens(5);
            CostSchedulerKernelInternal::fillRequestTokenCounts(context, tokens.data(), nullptr);
            CostSchedulerKernelInternal::ReplayInfo info;
            auto costs = prepareHistoricalCosts(context, tokens.data(), 0.5, 512, 1, 0, 10, 0, info);
            ASSERT_TRUE(costs);
            const auto nodePredictions = [&](const std::vector<uint32_t> &assignment) {
                std::vector<double> weights(10, 0.0);
                for (size_t r = 0; r < assignment.size(); ++r) {
                    weights[assignment[r] * 5 + r] = 1.0;
                }
                std::vector<double> secondTerms(2);
                CostSchedulerKernelInternal::forwardNodeRange(
                    context, costs.get(), weights.data(), tokens.data(), inverseTokens.data(),
                    5.0, 0.5, 0.0, 0, 2, nullptr, nullptr, secondTerms.data());
                return secondTerms;
            };
            auto beforePredictions = nodePredictions(original);
            EXPECT_NEAR(36542.060874466, beforePredictions[0] / 2, 0.02);
            EXPECT_NEAR(40182.049195279, beforePredictions[1] / 3, 0.02);
            auto moved = original;
            moved[movedIndex] = 0;
            const auto afterPredictions = nodePredictions(moved);
            EXPECT_LE(afterPredictions[0] / 3, targetBounds[index] + 0.02);
            CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
            const auto objective = [&](const std::vector<uint32_t> &assignment) {
                return CostSchedulerKernelInternal::forwardDiscrete(
                    context, costs.get(), assignment.data(), tokens.data(), inverseTokens.data(),
                    5.0, 0.5, 0.0, workspace);
            };
            // Four requests on other nodes remain unchanged. Their terms cancel;
            // only the denominator changes from this pair's five to all nine.
            const auto delta = (objective(moved) - objective(original)) * 5 / 9;
            EXPECT_LE(delta, objectiveBounds[index] + 0.02);
            EXPECT_GE(-delta, -objectiveBounds[index] - 0.02);
        }
    }
}

TEST_F(CostSchedulerIncidentTest, testLongRequestTriggersCollapseWithoutParameterUpdate)
{
    std::ifstream input(GET_PRIVATE_TEST_DATA_PATH() +
                        "/scheduler/incident_long_request_trigger.json");
    ASSERT_TRUE(input.good());
    const std::string text{std::istreambuf_iterator<char>(input), {}};
    rapidjson::Document data;
    data.Parse(text.c_str());
    ASSERT_FALSE(data.HasParseError());
    const auto &row = data["trigger_row"];
    const auto &state = data["trigger_state"];
    const auto parameters = array(state["parameters_after"]);
    ASSERT_FALSE(NonLinearPrefillModelR::validateParameters(parameters));
    const auto residual = array(state["residual_parameters_after"]);
    ASSERT_EQ(259, number(row, "model_revision"));
    ASSERT_EQ(3840, number(data["trigger_cache_audit"], "decision_hit_blocks") *
                        number(data["trigger_cache_audit"], "block_size"));
    const auto makeFeatures = [&](bool floor, int64_t removedPrompt, int64_t addedPrompt) {
        const int64_t removedHit = removedPrompt ? 3840 : 0;
        const int64_t addedHit = addedPrompt ? 3840 : 0;
        const auto removedMiss = removedPrompt - removedHit;
        const auto addedMiss = addedPrompt - addedHit;
        const auto prefix = floor ? "floor_" : "waiting_";
        const auto get = [&](const char *key) {
            return number(row, (std::string(prefix) + key).c_str());
        };
        const auto n = get("batch_size") - (removedPrompt != 0) + (addedPrompt != 0);
        const auto hit = get("hit_tokens") - removedHit + addedHit;
        const auto miss = get("miss_tokens") - removedMiss + addedMiss;
        const auto square = get("miss_square_sum") - removedMiss * removedMiss + addedMiss * addedMiss;
        const auto product = get("hit_miss_sum") - removedHit * removedMiss + addedHit * addedMiss;
        NaviSchedLatencyFeatures result;
        if (floor) {
            EXPECT_TRUE(result.appendCapacitySplitAggregate(
                number(row, "floor_batch_count"), n, hit, miss, square, product, 0, 0, 0, 0, 0));
        } else {
            EXPECT_TRUE(result.appendCapacitySplitAggregate(
                number(row, "batch_count"), 0, 0, 0, 0, 0, n, hit, miss, square, product));
        }
        EXPECT_TRUE(result.appendPrecedingRequestCount(number(row, "preceding_request_count")));
        return result;
    };
    const auto corrected = [&](const NaviSchedLatencyFeatures &features) {
        return LatencyResidualModel::evaluate(
            historicalPrediction(features, parameters), residual,
            state["residual_time_after"].GetInt64(), number(row, "schedule_time_us")).predictionMs;
    };
    const auto decision = [&](int64_t removedPrompt, int64_t addedPrompt) {
        return std::max(corrected(makeFeatures(false, removedPrompt, addedPrompt)) - 12.194,
                        corrected(makeFeatures(true, removedPrompt, addedPrompt)));
    };
    EXPECT_NEAR(2011.354802765, historicalPrediction(makeFeatures(false, 0, 0), parameters), 1e-6);
    EXPECT_NEAR(3167.504962199, decision(0, 0), 1e-6);
    EXPECT_NEAR(51387.406327179, decision(5754, 0), 1e-6);
    EXPECT_NEAR(51599.816502316, decision(5754, 3890), 1e-6);
    EXPECT_NEAR(3175.981620366, decision(3890, 0), 1e-6);
    EXPECT_NEAR(-6737.608631974, NonLinearPrefillModelR::calculateWaitingRequestLinearCost(
                                   parameters, 5754, 3840), 1e-6);
    EXPECT_GT(NonLinearPrefillModelR::calculateWaitingRequestLinearCost(parameters, 3890, 3840), 0.0);
}

TEST_F(CostSchedulerIncidentTest, testSelectedPairRecoversKnownDiscreteOptimum)
{
    NaviSchedScheduleReplay replay;
    loadReplay(replay);
    ASSERT_FALSE(HasFatalFailure());
    NaviSchedScheduleContext context;
    ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
    ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
    std::vector<int64_t> tokens(9);
    const auto totalTokens = CostSchedulerKernelInternal::fillRequestTokenCounts(
        context, tokens.data(), nullptr);
    const std::vector<double> inverseTokens(9, 1.0);
    CostSchedulerKernelInternal::ReplayInfo replayInfo;
    auto costs = prepareHistoricalCosts(
        context, tokens.data(), 0.5, 1, 1, 0, 10, 0, replayInfo);
    ASSERT_TRUE(costs);
    auto best = std::numeric_limits<double>::infinity();
    size_t bestMask = 0;
    std::vector<double> exhaustiveValues(512);
    CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
    for (size_t mask = 0; mask < 512; ++mask) {
        std::vector<uint32_t> assignment(9);
        for (size_t i = 0; i < 9; ++i) {
            assignment[i] = (mask >> i) & 1;
        }
        const auto candidate = CostSchedulerKernelInternal::forwardDiscrete(
            context, costs.get(), assignment.data(), tokens.data(), inverseTokens.data(),
            9.0, 0.5, 0.0, workspace);
        exhaustiveValues[mask] = candidate;
        if (candidate < best) {
            best = candidate;
            bestMask = mask;
        }
    }
    EXPECT_NEAR(18659.546377861727, best, 1e-6);
    EXPECT_EQ(379, bestMask);
    // This small fixture has an enumerated optimum; do not generalize this
    // assertion to arbitrary nonconvex scheduling instances.
    for (uint64_t seed : {1, 7, 31}) {
        std::vector<double> weights(18);
        std::vector<uint32_t> selected;
        double value = 0;
        CostSchedulerKernelInternal::ReplayInfo trace;
        trace.init(true, 0.5, 512, 1, 0, 10, -1);
        ASSERT_TRUE(CostSchedulerKernelInternal::optimize(
            context, costs.get(), tokens.data(), totalTokens, 0.5, 512, 1, 0,
            10, -1, trace, 0, seed, weights.data(), selected, value,
            0.0, 5, 0.25, 0.7));
        size_t selectedMask = 0;
        for (size_t i = 0; i < selected.size(); ++i) {
            selectedMask |= selected[i] << i;
        }
        // Requests 5 and 7 have identical features, so the optimum is not unique.
        EXPECT_NEAR(best, exhaustiveValues[selectedMask], 1e-6);
        EXPECT_NEAR(best, value, 1e-6);
        ASSERT_EQ(1, trace.optimizationRuns.size());
        const auto &run = trace.optimizationRuns.front();
        ASSERT_EQ(10, run.actualLoopCount);
        std::string masks;
        for (const auto &step : run.steps) {
            EXPECT_DOUBLE_EQ(512.0, step.alpha);
            EXPECT_FALSE(step.discreteRepairActive);
            if (step.discreteEvaluated) {
                EXPECT_NEAR(0.0, step.discreteGap, 1e-12);
            }
            size_t mask = 0;
            for (size_t i = 0; i < 9; ++i) {
                const auto left = step.weightsAfterProjection[i];
                const auto right = step.weightsAfterProjection[9 + i];
                EXPECT_TRUE(left == 0.0 || left == 1.0);
                EXPECT_DOUBLE_EQ(1.0, left + right);
                mask |= static_cast<size_t>(right) << i;
            }
            masks += std::to_string(mask) + ",";
        }
        RecordProperty("projection_masks_seed_" + std::to_string(seed), masks);
        EXPECT_EQ("511,0,511,0,511,0,511,0,511,0,", masks);
        for (size_t i = 0; i < selected.size(); ++i) {
            auto oneRequestMove = selected;
            oneRequestMove[i] = 1 - oneRequestMove[i];
            const auto moved = CostSchedulerKernelInternal::forwardDiscrete(
                context, costs.get(), oneRequestMove.data(), tokens.data(), inverseTokens.data(),
                9.0, 0.5, 0.0, workspace);
            EXPECT_GE(moved, value - 1e-6);
        }
    }
}

TEST_F(CostSchedulerIncidentTest, testParallelGraphRefinesValidCounterpart)
{
    NaviSchedScheduleReplay replay;
    loadReplay(replay);
    ASSERT_FALSE(HasFatalFailure());
    // The graph must reject the untouched accident model at its normal ingress.
    auto rejectedContext = std::make_shared<NaviSchedScheduleContext>();
    ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, *rejectedContext));
    ASSERT_TRUE(ScheduleReplayRunner::prepareContext(*rejectedContext));
    auto rejectedState = CostSchedulerGraphState::create(rejectedContext, CostSchedulerConfig{}, 1);
    ASSERT_TRUE(rejectedState);
    EXPECT_FALSE(rejectedState->runRow(0, 0));
    EXPECT_TRUE(rejectedState->failed());
    // A synthetic valid counterpart exercises the real graph without bypassing
    // validation. Its optimum is enumerated, not borrowed from the accident.
    for (auto &node : replay.nodes) {
        for (auto index : NonLinearPrefillModelR::POSITIVE_PARAMETER_INDEXES) {
            node.latencyParameters[index] =
                std::max(NonLinearPrefillModelR::positiveParameterFloor(index, 1.0),
                         std::abs(node.latencyParameters[index]));
        }
        ASSERT_TRUE(NonLinearPrefillModelR::validateParameters(node.latencyParameters));
    }
    for (size_t workers : {size_t{1}, size_t{2}}) {
        for (int loops : {10, 100}) {
            SCOPED_TRACE(workers);
            SCOPED_TRACE(loops);
            auto context = std::make_shared<NaviSchedScheduleContext>();
            ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, *context));
            auto captured = std::make_shared<NaviSchedScheduleReplay>(replay);
            ScheduleReplayRunner::setReplay(*context, captured);
            ASSERT_TRUE(ScheduleReplayRunner::prepareContext(*context));
            std::vector<int64_t> tokens(9);
            CostSchedulerKernelInternal::fillRequestTokenCounts(*context, tokens.data(), nullptr);
            CostSchedulerKernelInternal::ReplayInfo referenceTrace;
            auto costs = CostSchedulerKernelInternal::prepare(
                *context, tokens.data(), 0.5, 512, 1, 0, loops, -1, referenceTrace);
            ASSERT_TRUE(costs);
            const std::vector<double> inverseTokens(9, 1.0);
            CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
            auto optimum = std::numeric_limits<double>::infinity();
            for (size_t mask = 0; mask < 512; ++mask) {
                std::vector<uint32_t> assignment(9);
                for (size_t i = 0; i < 9; ++i) {
                    assignment[i] = (mask >> i) & 1;
                }
                optimum = std::min(optimum, CostSchedulerKernelInternal::forwardDiscrete(
                    *context, costs.get(), assignment.data(), tokens.data(), inverseTokens.data(),
                    9.0, 0.5, 0.0, workspace));
            }
            CostSchedulerConfig config;
            config.alpha = 512;
            config.initialWeightSeed = 7;
            config.maxLoopCount = loops;
            config.optimizationTimeBudgetUs = -1;
            config.latencyWeightConcavity = 0;
            config.latencyTokenNormalizationPower = 0;
            auto state = CostSchedulerGraphState::create(context, config, workers);
            ASSERT_TRUE(state);
            for (int iteration = 0; iteration < loops; ++iteration) {
                for (size_t worker = 0; worker < workers; ++worker) {
                    ASSERT_TRUE(state->runRow(worker, iteration));
                }
                ASSERT_TRUE(state->finishRow(iteration));
                for (size_t worker = 0; worker < workers; ++worker) {
                    ASSERT_TRUE(state->runProjection(worker, iteration));
                }
                ASSERT_TRUE(state->finishProjection(iteration));
                EXPECT_EQ(static_cast<size_t>(iteration + 1),
                          context->_costGradientIterationCount);
                EXPECT_GE(context->_costGradientIterationTotalUs,
                          context->_costGradientIterationMaxUs);
                for (size_t request = 0; request < 9; ++request) {
                    EXPECT_GE(state->weights()[request], 0.0);
                    EXPECT_LE(state->weights()[request], 1.0);
                    EXPECT_DOUBLE_EQ(1.0, state->weights()[request] +
                                          state->weights()[9 + request]);
                }
            }
            for (size_t worker = 0; worker < workers; ++worker) {
                ASSERT_TRUE(state->runFinalRow(worker));
            }
            ASSERT_TRUE(state->finalize());
            EXPECT_EQ(static_cast<size_t>(loops), state->actualLoopCount());
            EXPECT_NEAR(optimum, state->forwardValue(), 1e-6);
            CostSchedulerKernelInternal::ReplayInfo trace;
            autil::legacy::FromJson(trace, autil::legacy::Any(captured->schedulerInfo));
            ASSERT_EQ(1, trace.optimizationRuns.size());
            const auto &steps = trace.optimizationRuns.front().steps;
            ASSERT_EQ(static_cast<size_t>(loops), steps.size());
            size_t checks = 0;
            for (const auto &step : steps) {
                EXPECT_DOUBLE_EQ(512.0, step.alpha);
                EXPECT_FALSE(step.discreteRepairActive);
                if (step.discreteEvaluated) {
                    ++checks;
                    EXPECT_NEAR(0.0, step.discreteGap, 1e-12);
                }
            }
            EXPECT_EQ(static_cast<size_t>(loops - 1), checks);
            ASSERT_GT(trace.optimizationRuns.front().refinementEvaluations, 0);
            auto replayContext = std::make_shared<NaviSchedScheduleContext>();
            ASSERT_TRUE(ScheduleReplayRunner::makeContext(*captured, *replayContext));
            auto replayOutput = std::make_shared<NaviSchedScheduleReplay>(*captured);
            ScheduleReplayRunner::setReplay(*replayContext, replayOutput);
            std::vector<NaviSchedDecision> recordedDecisions, replayedDecisions;
            ASSERT_TRUE(ScheduleReplayRunner::collectDecisions(*context, recordedDecisions));
            ASSERT_TRUE(ScheduleReplayRunner::runCostScheduler(
                *captured, replayContext, 0.5, 512, 1, 0, loops, 0, replayedDecisions));
            EXPECT_TRUE(ScheduleReplayRunner::decisionsEqual(recordedDecisions, replayedDecisions));
            CostSchedulerKernelInternal::ReplayInfo replayed;
            autil::legacy::FromJson(replayed, autil::legacy::Any(replayOutput->schedulerInfo));
            ASSERT_EQ(1, replayed.optimizationRuns.size());
            EXPECT_EQ(trace.optimizationRuns.front().refinementEvaluations,
                      replayed.optimizationRuns.front().refinementEvaluations);
            EXPECT_EQ(trace.optimizationRuns.front().refinementMoves,
                      replayed.optimizationRuns.front().refinementMoves);
            EXPECT_DOUBLE_EQ(trace.optimizationRuns.front().bestForwardValue,
                             replayed.optimizationRuns.front().bestForwardValue);
        }
    }
}

TEST_F(CostSchedulerIncidentTest, testRefinementMatchesFullObjectiveReference)
{
    using Internal = CostSchedulerKernelInternal;
    for (int freeSlots : {0, 4, 9}) {
        SCOPED_TRACE(freeSlots);
        NaviSchedScheduleReplay replay;
        loadReplay(replay);
        ASSERT_FALSE(HasFatalFailure());
        if (freeSlots == 4) {
            auto third = replay.nodes.front();
            third.agentId = "third_candidate";
            third.agentIdHash = 3;
            replay.nodes.push_back(std::move(third));
            for (auto &matches : replay.nodeMatches) {
                matches.push_back(matches.front());
                matches.back().matchedBlockCount = 60;
            }
        }
        for (auto &node : replay.nodes) {
            node.maxRunningRequestCount += freeSlots;
            node.workload.maxRunningRequestCount += freeSlots;
        }
        for (size_t i = 0; i < replay.requests.size(); ++i) {
            replay.nodeMatches[i][0].matchedBlockCount = i % 2 ? 120 : 240;
            replay.nodeMatches[i][1].matchedBlockCount = i % 2 ? 0 : 120;
        }
        NaviSchedScheduleContext context;
        ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
        ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
        const auto count = context.requestCount();
        ASSERT_EQ(freeSlots == 4 ? 3 : 2, context.nodeCount());
        for (size_t nodeIndex = 0; nodeIndex < context.nodeCount(); ++nodeIndex) {
            const auto &node = context.getNode(nodeIndex);
            EXPECT_EQ(freeSlots, node.getMaxRunningRequestCount() -
                                     node.getWorkloadRunningCapacitySlotCount());
        }
        const double power = freeSlots == 0 ? 0.0 : 0.5;
        const double concavity = freeSlots == 9 ? 128.0 : 0.0;
        const double lambda = freeSlots == 4 ? 0.0 : 0.5;
        std::vector<int64_t> tokens(count);
        std::vector<double> inverseTokens(count);
        Internal::fillRequestTokenCounts(context, tokens.data(), inverseTokens.data(), power);
        const auto denominator = Internal::calculateTokenNormalizationDenominator(
            tokens.data(), count, power);
        Internal::ReplayInfo trace;
        auto costs = prepareHistoricalCosts(context, tokens.data(), lambda, 512, 1, 0, 10, -1, trace);
        ASSERT_TRUE(costs);
        // Independent full-node formula, without the pair contribution cache.
        const auto objective = [&](const std::vector<uint32_t> &assignment) {
            std::vector<double> weights(count * context.nodeCount(), 0.0);
            for (size_t i = 0; i < count; ++i) {
                weights[assignment[i] * count + i] = 1.0;
            }
            return Internal::forwardNodeRange(
                context, costs.get(), weights.data(), tokens.data(), inverseTokens.data(),
                denominator, lambda, concavity, 0, context.nodeCount(), nullptr, nullptr, nullptr);
        };
        for (size_t mask = 0; mask < 512; ++mask) {
            SCOPED_TRACE(mask);
            std::vector<uint32_t> actual(count);
            for (size_t i = 0; i < count; ++i) {
                actual[i] = (mask >> i) & 1;
            }
            auto expected = actual;
            const auto originalValue = objective(actual);
            auto expectedValue = originalValue;
            // Also interrupt in the middle of a pass to verify replay budgets.
            const size_t limit = mask % 2 ? 13 : 4096;
            size_t evaluations = 0;
            size_t moves = 0;
            while (evaluations < limit) {
                auto best = expected;
                auto bestValue = expectedValue;
                const auto tolerance = 1e-12 * std::max(1.0, std::abs(expectedValue));
                for (size_t i = 0; i < count && evaluations < limit; ++i) {
                    for (size_t target = 0; target < context.nodeCount() && evaluations < limit; ++target) {
                        if (target == expected[i]) {
                            continue;
                        }
                        auto candidate = expected;
                        candidate[i] = target;
                        const auto value = objective(candidate);
                        ++evaluations;
                        if (value < bestValue - tolerance) {
                            bestValue = value;
                            best = std::move(candidate);
                        }
                    }
                }
                if (best == expected) {
                    break;
                }
                expected = std::move(best);
                expectedValue = bestValue;
                ++moves;
            }
            double value = 0;
            Internal::DiscreteRefinementStats stats;
            ASSERT_TRUE(Internal::refineDiscreteAssignment(
                context, costs.get(), actual.data(), tokens.data(), inverseTokens.data(),
                denominator, lambda, concavity, value, -1, limit, stats));
            EXPECT_EQ(expected, actual);
            EXPECT_NEAR(expectedValue, value, 1e-6);
            EXPECT_NEAR(objective(actual), value, 1e-6);
            EXPECT_LE(value, originalValue + 1e-6);
            EXPECT_EQ(evaluations, stats.evaluations);
            EXPECT_EQ(moves, stats.moves);
        }
    }
}

TEST_F(CostSchedulerIncidentTest, testRefinementBudgetAndInvalidInput)
{
    using Internal = CostSchedulerKernelInternal;
    EXPECT_EQ(0, Internal::discreteRefinementBudgetUs(0, 0));
    EXPECT_EQ(0, Internal::discreteRefinementBudgetUs(100, 101));
    EXPECT_EQ(1, Internal::discreteRefinementBudgetUs(100, 99));
    EXPECT_EQ(10000, Internal::discreteRefinementBudgetUs(10000, 0));
    EXPECT_EQ(-1, Internal::discreteRefinementBudgetUs(-1, 10000));
    NaviSchedScheduleReplay replay;
    loadReplay(replay);
    ASSERT_FALSE(HasFatalFailure());
    NaviSchedScheduleContext context;
    ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
    ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
    std::vector<int64_t> tokens(9);
    const std::vector<double> inverseTokens(9, 1.0);
    Internal::fillRequestTokenCounts(context, tokens.data());
    Internal::ReplayInfo trace;
    auto costs = prepareHistoricalCosts(context, tokens.data(), 0.5, 512, 1, 0, 10, -1, trace);
    ASSERT_TRUE(costs);
    const std::vector<uint32_t> original(9, 1);
    auto actual = original;
    Internal::DiscreteRefinementStats stats;
    double value = 0;
    const auto refine = [&](int64_t budget, size_t evaluations) {
        return Internal::refineDiscreteAssignment(
            context, costs.get(), actual.data(), tokens.data(), inverseTokens.data(),
            9.0, 0.5, 0.0, value, budget, evaluations, stats);
    };
    ASSERT_TRUE(refine(0, 4096));
    EXPECT_EQ(original, actual);
    EXPECT_EQ(0, stats.evaluations);
    EXPECT_EQ(0, stats.moves);
    EXPECT_NEAR(18766.183505953941, value, 1e-6);
    ASSERT_TRUE(refine(-1, 0));
    EXPECT_EQ(original, actual);
    EXPECT_EQ(0, stats.evaluations);
    ASSERT_TRUE(refine(-1, 1));
    EXPECT_EQ(1, stats.evaluations);
    EXPECT_TRUE(stats.budgetExhausted);
    EXPECT_LE(value, 18766.183505953941);
    actual[0] = 2;
    ASSERT_FALSE(refine(-1, 4096));
    EXPECT_EQ(0, stats.evaluations);
    EXPECT_EQ(2, actual[0]);
}

TEST_F(CostSchedulerIncidentTest, testPairDiscreteForwardPreservesUnrelatedQueues)
{
    using Internal = CostSchedulerKernelInternal;
    NaviSchedScheduleReplay replay;
    loadReplay(replay);
    ASSERT_FALSE(HasFatalFailure());
    auto third = replay.nodes.front();
    third.agentId = "untouched_candidate";
    third.agentIdHash = 3;
    replay.nodes.push_back(std::move(third));
    for (auto &matches : replay.nodeMatches) {
        matches.push_back(matches.front());
    }
    NaviSchedScheduleContext context;
    ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
    ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
    const auto count = context.requestCount();
    std::vector<int64_t> tokens(count);
    const std::vector<double> inverseTokens(count, 1.0);
    Internal::fillRequestTokenCounts(context, tokens.data());
    Internal::ReplayInfo trace;
    auto costs = prepareHistoricalCosts(context, tokens.data(), 0.5, 512, 1, 0, 10, -1, trace);
    ASSERT_TRUE(costs);
    std::vector<uint32_t> assignment(count, 2);
    assignment[0] = 0;
    assignment[1] = 1;
    Internal::DiscreteForwardWorkspace workspace;
    const auto full = Internal::forwardDiscrete(
        context, costs.get(), assignment.data(), tokens.data(), inverseTokens.data(),
        count, 0.5, 0.0, workspace);
    EXPECT_EQ(3, workspace.nodeBaselineLoads);
    const auto untouched = workspace;
    const auto expectedPair = workspace.nodeValues[0] + workspace.nodeValues[1];
    const auto pair = Internal::forwardDiscreteImpl(
        context, costs.get(), assignment.data(), tokens.data(), inverseTokens.data(),
        count, 0.5, 0.0, workspace, 0, 1);
    EXPECT_EQ(0, workspace.nodeBaselineLoads);
    EXPECT_DOUBLE_EQ(expectedPair, pair);
    EXPECT_DOUBLE_EQ(untouched.requestCostSums[2], workspace.requestCostSums[2]);
    EXPECT_DOUBLE_EQ(untouched.requestCostSumsWithoutReportedWorkload[2],
                     workspace.requestCostSumsWithoutReportedWorkload[2]);
    EXPECT_DOUBLE_EQ(untouched.totalPromptTokenSums[2], workspace.totalPromptTokenSums[2]);
    EXPECT_DOUBLE_EQ(untouched.requestWeightSums[2], workspace.requestWeightSums[2]);
    EXPECT_DOUBLE_EQ(untouched.inverseTokenWeightSums[2], workspace.inverseTokenWeightSums[2]);
    EXPECT_DOUBLE_EQ(untouched.nodeValues[2], workspace.nodeValues[2]);
    EXPECT_DOUBLE_EQ(full, Internal::forwardDiscrete(
        context, costs.get(), assignment.data(), tokens.data(), inverseTokens.data(),
        count, 0.5, 0.0, workspace));
}

TEST_F(CostSchedulerIncidentTest, testDiscreteBaselineRefreshesForNewContextAndFullEvaluation)
{
    using Internal = CostSchedulerKernelInternal;
    NaviSchedScheduleReplay replay;
    loadReplay(replay);
    ASSERT_FALSE(HasFatalFailure());
    NaviSchedScheduleContext first, second;
    ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, first));
    ASSERT_TRUE(ScheduleReplayRunner::prepareContext(first));
    const auto count = first.requestCount();
    std::vector<int64_t> tokens(count);
    const std::vector<double> inverseTokens(count, 1.0);
    Internal::fillRequestTokenCounts(first, tokens.data());
    Internal::ReplayInfo trace;
    auto firstCosts = prepareHistoricalCosts(first, tokens.data(), 0.5, 512, 1, 0, 10, -1, trace);
    ASSERT_TRUE(firstCosts);
    std::vector<uint32_t> assignment(count);
    for (size_t i = 0; i < count; ++i) {
        assignment[i] = i % 2;
    }
    Internal::DiscreteForwardWorkspace reused, fresh;
    const auto original = Internal::forwardDiscrete(
        first, firstCosts.get(), assignment.data(), tokens.data(), inverseTokens.data(),
        count, 0.5, 0.0, reused);
    for (auto &node : replay.nodes) {
        node.maxRunningRequestCount += 4;
        node.latencyParameters[NonLinearPrefillModelR::BATCH_COUNT_PARAMETER_INDEX] += 0.25;
    }
    ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, second));
    ASSERT_TRUE(ScheduleReplayRunner::prepareContext(second));
    auto secondCosts = prepareHistoricalCosts(second, tokens.data(), 0.5, 512, 1, 0, 10, -1, trace);
    ASSERT_TRUE(secondCosts);
    const auto expected = Internal::forwardDiscrete(
        second, secondCosts.get(), assignment.data(), tokens.data(), inverseTokens.data(),
        count, 0.5, 0.0, fresh);
    EXPECT_NE(original, expected);
    EXPECT_DOUBLE_EQ(expected, Internal::forwardDiscreteImpl(
        second, secondCosts.get(), assignment.data(), tokens.data(), inverseTokens.data(),
        count, 0.5, 0.0, reused, 0, 1));
    EXPECT_EQ(2, reused.nodeBaselineLoads);
    EXPECT_DOUBLE_EQ(expected, Internal::forwardDiscreteImpl(
        second, secondCosts.get(), assignment.data(), tokens.data(), inverseTokens.data(),
        count, 0.5, 0.0, reused, 0, 1));
    EXPECT_EQ(0, reused.nodeBaselineLoads);

    const auto originalFloorCost = reused.nodeBaselines[0].floorLinearCost;
    ASSERT_TRUE(second._scheduleNodes[0]._reservedWorkloadFeatures.appendBatchCount(3));
    auto changedCosts = prepareHistoricalCosts(second, tokens.data(), 0.5, 512, 1, 0, 10, -1, trace);
    ASSERT_TRUE(changedCosts);
    const auto changed = Internal::forwardDiscrete(
        second, changedCosts.get(), assignment.data(), tokens.data(), inverseTokens.data(),
        count, 0.5, 0.0, fresh);
    EXPECT_NE(originalFloorCost, fresh.nodeBaselines[0].floorLinearCost);
    EXPECT_DOUBLE_EQ(changed, Internal::forwardDiscrete(
        second, changedCosts.get(), assignment.data(), tokens.data(), inverseTokens.data(),
        count, 0.5, 0.0, reused));
    EXPECT_EQ(2, reused.nodeBaselineLoads);
    EXPECT_DOUBLE_EQ(fresh.nodeBaselines[0].floorLinearCost,
                     reused.nodeBaselines[0].floorLinearCost);
}

TEST_F(CostSchedulerIncidentTest, testPairLoadsCapacityOncePerNodeAndPreservesOrderedSums)
{
    using Internal = CostSchedulerKernelInternal;
    for (int freeSlots : {0, 4, 9}) {
        SCOPED_TRACE(freeSlots);
        NaviSchedScheduleReplay replay;
        loadReplay(replay);
        ASSERT_FALSE(HasFatalFailure());
        for (auto &node : replay.nodes) {
            node.maxRunningRequestCount += freeSlots;
            node.workload.maxRunningRequestCount += freeSlots;
        }
        NaviSchedScheduleContext context;
        ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
        ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
        const auto count = context.requestCount();
        ASSERT_EQ(9, count);
        std::vector<int64_t> tokens(count);
        std::vector<double> inverseTokens(count);
        Internal::fillRequestTokenCounts(context, tokens.data(), inverseTokens.data(), 0.5);
        const auto denominator = Internal::calculateTokenNormalizationDenominator(
            tokens.data(), count, 0.5);
        Internal::ReplayInfo trace;
        auto costs = prepareHistoricalCosts(
            context, tokens.data(), 0.5, 512, 1, 0, 10, -1, trace);
        ASSERT_TRUE(costs);
        std::vector<uint32_t> assignment(count);
        std::vector<std::vector<size_t>> membership(2);
        for (size_t i = 0; i < count; ++i) {
            assignment[i] = i % 2;
            membership[assignment[i]].push_back(i);
        }
        for (size_t moved = 0; moved < count; ++moved) {
            SCOPED_TRACE(moved);
            const auto source = assignment[moved];
            const auto target = 1 - source;
            auto candidate = assignment;
            candidate[moved] = target;
            Internal::DiscreteForwardWorkspace expected, actual;
            const auto expectedValue = Internal::forwardDiscrete(
                context, costs.get(), candidate.data(), tokens.data(), inverseTokens.data(),
                denominator, 0.5, 128.0, expected);
            EXPECT_EQ(count, expected.nodeCapacityLoads);
            const auto actualValue = Internal::forwardDiscreteImpl(
                context, costs.get(), assignment.data(), tokens.data(), inverseTokens.data(),
                denominator, 0.5, 128.0, actual, source, target, &membership, moved);
            EXPECT_EQ(2, actual.nodeCapacityLoads);
            EXPECT_DOUBLE_EQ(expectedValue, actualValue);
            EXPECT_EQ(expected.requestCostSums, actual.requestCostSums);
            EXPECT_EQ(expected.requestCostSumsWithoutReportedWorkload,
                      actual.requestCostSumsWithoutReportedWorkload);
            EXPECT_EQ(expected.totalPromptTokenSums, actual.totalPromptTokenSums);
            EXPECT_EQ(expected.requestWeightSums, actual.requestWeightSums);
            EXPECT_EQ(expected.inverseTokenWeightSums, actual.inverseTokenWeightSums);
            EXPECT_EQ(expected.nodeValues, actual.nodeValues);
        }
    }
}

TEST_F(CostSchedulerIncidentTest, testDiscreteSourceReusePreservesEveryTargetCost)
{
    using Internal = CostSchedulerKernelInternal;
    for (bool validModel : {false, true}) {
        SCOPED_TRACE(validModel);
        NaviSchedScheduleReplay fixture;
        loadReplay(fixture);
        ASSERT_FALSE(HasFatalFailure());
        if (validModel) {
            for (auto &node : fixture.nodes) {
                for (auto index : NonLinearPrefillModelR::POSITIVE_PARAMETER_INDEXES) {
                    node.latencyParameters[index] =
                        std::max(NonLinearPrefillModelR::positiveParameterFloor(index, 1.0),
                                 std::abs(node.latencyParameters[index]));
                }
                ASSERT_TRUE(NonLinearPrefillModelR::validateParameters(node.latencyParameters));
            }
        }
        for (int freeSlots : {0, 4, 9}) {
            for (bool concentrated : {false, true}) {
                auto replay = fixture;
                for (size_t n = 2; n < 4; ++n) {
                    auto node = fixture.nodes[n % 2];
                    node.agentId = "source_reuse_" + std::to_string(n);
                    node.agentIdHash = n + 1;
                    replay.nodes.push_back(std::move(node));
                }
                for (auto &matches : replay.nodeMatches) {
                    matches.push_back(matches[0]);
                    matches.push_back(matches[1]);
                }
                for (auto &node : replay.nodes) {
                    node.maxRunningRequestCount += freeSlots;
                    node.workload.maxRunningRequestCount += freeSlots;
                }
                NaviSchedScheduleContext context;
                ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
                ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
                const auto count = context.requestCount();
                std::vector<int64_t> tokens(count);
                std::vector<double> inverseTokens(count);
                Internal::fillRequestTokenCounts(context, tokens.data(), inverseTokens.data(), 0.5);
                const auto denominator = Internal::calculateTokenNormalizationDenominator(
                    tokens.data(), count, 0.5);
                Internal::ReplayInfo trace;
                auto costs = validModel
                    ? Internal::prepare(context, tokens.data(), 0.5, 512, 1, 0, 10, -1, trace)
                    : prepareHistoricalCosts(context, tokens.data(), 0.5, 512, 1, 0, 10, -1, trace);
                ASSERT_TRUE(costs);
                std::vector<uint32_t> assignment(count);
                std::vector<std::vector<size_t>> membership(4);
                for (size_t i = 0; i < count; ++i) {
                    assignment[i] = concentrated ? 0 : i % 4;
                    membership[assignment[i]].push_back(i);
                }
                for (size_t moved = 0; moved < count; ++moved) {
                    const auto source = assignment[moved];
                    Internal::DiscreteForwardWorkspace actual;
                    bool sourceReady = false;
                    for (size_t target = 0; target < 4; ++target) {
                        if (target == source) continue;
                        SCOPED_TRACE(std::to_string(freeSlots) + ":" +
                                     std::to_string(concentrated) + ":" +
                                     std::to_string(moved) + ":" + std::to_string(target));
                        auto candidate = assignment;
                        candidate[moved] = target;
                        Internal::DiscreteForwardWorkspace expected;
                        ASSERT_TRUE(std::isfinite(Internal::forwardDiscrete(
                            context, costs.get(), candidate.data(), tokens.data(),
                            inverseTokens.data(), denominator, 0.5, 128.0, expected)));
                        const auto value = Internal::forwardDiscreteImpl(
                            context, costs.get(), assignment.data(), tokens.data(),
                            inverseTokens.data(), denominator, 0.5, 128.0, actual,
                            source, target, &membership, moved, sourceReady);
                        EXPECT_EQ(sourceReady ? 1 : 2, actual.nodeCapacityLoads);
                        EXPECT_DOUBLE_EQ(expected.nodeValues[source] + expected.nodeValues[target], value);
                        for (size_t node : {size_t(source), target}) {
                            EXPECT_DOUBLE_EQ(expected.nodeValues[node], actual.nodeValues[node]);
                            EXPECT_DOUBLE_EQ(expected.requestCostSums[node], actual.requestCostSums[node]);
                            EXPECT_DOUBLE_EQ(expected.requestCostSumsWithoutReportedWorkload[node],
                                             actual.requestCostSumsWithoutReportedWorkload[node]);
                            EXPECT_DOUBLE_EQ(expected.requestWeightSums[node], actual.requestWeightSums[node]);
                            EXPECT_DOUBLE_EQ(expected.inverseTokenWeightSums[node], actual.inverseTokenWeightSums[node]);
                        }
                        sourceReady = true;
                    }
                }
            }
        }
    }
}

TEST_F(CostSchedulerIncidentTest, testRefinementFixedWorkLargeNodeCount)
{
    using Internal = CostSchedulerKernelInternal;
    NaviSchedScheduleReplay fixture;
    loadReplay(fixture);
    ASSERT_FALSE(HasFatalFailure());
    for (size_t nodeCount : {size_t{24}, size_t{432}}) {
        for (size_t count : {size_t{128}, size_t{512}, size_t{2048}}) {
            NaviSchedScheduleReplay replay = fixture;
            replay.nodes.clear();
            replay.requests.clear();
            replay.nodeMatches.clear();
            for (size_t n = 0; n < nodeCount; ++n) {
                auto node = fixture.nodes[n % 2];
                node.agentId = "fixed_work_" + std::to_string(n);
                node.agentIdHash = n + 1;
                replay.nodes.push_back(std::move(node));
            }
            for (size_t i = 0; i < count; ++i) {
                auto request = fixture.requests[i % fixture.requests.size()];
                request.request.requestId = i + 1;
                replay.requests.push_back(std::move(request));
                replay.nodeMatches.emplace_back(nodeCount, fixture.nodeMatches[i % fixture.requests.size()][1]);
            }
            NaviSchedScheduleContext context;
            ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
            ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
            std::vector<int64_t> tokens(count);
            const std::vector<double> inverseTokens(count, 1.0);
            Internal::fillRequestTokenCounts(context, tokens.data());
            Internal::ReplayInfo trace;
            auto costs = prepareHistoricalCosts(context, tokens.data(), 0.5, 512, 1, 0, 10, -1, trace);
            ASSERT_TRUE(costs);
            std::vector<int64_t> durations;
            for (size_t repeat = 0; repeat < 9; ++repeat) {
                // Include both a loaded source and dispersed ordered queues.
                std::vector<uint32_t> actual(count);
                for (size_t i = 0; i < count; ++i) {
                    actual[i] = i % 2 ? 0 : i % nodeCount;
                }
                Internal::DiscreteRefinementStats stats;
                double value = 0;
                const auto begin = autil::TimeUtility::monotonicTimeUs();
                ASSERT_TRUE(Internal::refineDiscreteAssignment(
                    context, costs.get(), actual.data(), tokens.data(), inverseTokens.data(),
                    count, 0.5, 0.0, value, -1, 4096, stats));
                const auto elapsed = autil::TimeUtility::monotonicTimeUs() - begin;
                EXPECT_EQ(4096, stats.evaluations);
                EXPECT_TRUE(stats.budgetExhausted);
                if (repeat == 0) {
                    const auto prefix = "nodes_" + std::to_string(nodeCount) +
                                        "_requests_" + std::to_string(count);
                    std::ostringstream selected, objective;
                    for (const auto node : actual) {
                        selected << node << ',';
                    }
                    objective << std::hexfloat << value;
                    RecordProperty(prefix + "_selected_nodes", selected.str());
                    RecordProperty(prefix + "_objective", objective.str());
                }
                std::vector<double> weights(count * nodeCount, 0.0);
                for (size_t i = 0; i < count; ++i) {
                    weights[actual[i] * count + i] = 1.0;
                }
                EXPECT_NEAR(value, Internal::forwardNodeRange(
                    context, costs.get(), weights.data(), tokens.data(), inverseTokens.data(),
                    count, 0.5, 0.0, 0, nodeCount, nullptr, nullptr, nullptr), 1e-6);
                if (repeat > 0) {
                    durations.push_back(elapsed);
                }
            }
            std::sort(durations.begin(), durations.end());
            const auto prefix = "nodes_" + std::to_string(nodeCount) + "_requests_" + std::to_string(count);
            RecordProperty(prefix + "_median_us", std::to_string(durations[4]));
            RecordProperty(prefix + "_min_us", std::to_string(durations.front()));
            RecordProperty(prefix + "_max_us", std::to_string(durations.back()));
        }
    }
}

TEST_F(CostSchedulerIncidentTest, testRefinementBoundedCost)
{
    using Internal = CostSchedulerKernelInternal;
    NaviSchedScheduleReplay fixture;
    loadReplay(fixture);
    ASSERT_FALSE(HasFatalFailure());
    for (size_t count : {size_t{32}, size_t{128}, size_t{512}}) {
        NaviSchedScheduleReplay replay = fixture;
        replay.nodes.clear();
        replay.requests.clear();
        replay.nodeMatches.clear();
        for (size_t n = 0; n < 24; ++n) {
            auto node = fixture.nodes[n % 2];
            node.agentId = "benchmark_" + std::to_string(n);
            node.agentIdHash = n + 1;
            replay.nodes.push_back(std::move(node));
        }
        for (size_t i = 0; i < count; ++i) {
            auto request = fixture.requests[i % fixture.requests.size()];
            request.request.requestId = i + 1;
            replay.requests.push_back(std::move(request));
            replay.nodeMatches.emplace_back(24, fixture.nodeMatches[i % fixture.requests.size()][1]);
        }
        NaviSchedScheduleContext context;
        ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
        ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
        std::vector<int64_t> tokens(count);
        const std::vector<double> inverseTokens(count, 1.0);
        Internal::fillRequestTokenCounts(context, tokens.data());
        Internal::ReplayInfo trace;
        auto costs = prepareHistoricalCosts(context, tokens.data(), 0.5, 512, 1, 0, 10, -1, trace);
        ASSERT_TRUE(costs);
        std::vector<int64_t> durations, evaluations;
        for (size_t repeat = 0; repeat < 31; ++repeat) {
            std::vector<uint32_t> actual(count, 0);
            Internal::DiscreteForwardWorkspace workspace;
            const auto original = Internal::forwardDiscrete(
                context, costs.get(), actual.data(), tokens.data(), inverseTokens.data(),
                count, 0.5, 0.0, workspace);
            Internal::DiscreteRefinementStats stats;
            double value = 0;
            const auto begin = autil::TimeUtility::monotonicTimeUs();
            ASSERT_TRUE(Internal::refineDiscreteAssignment(
                context, costs.get(), actual.data(), tokens.data(), inverseTokens.data(),
                count, 0.5, 0.0, value, Internal::MAX_DISCRETE_REFINEMENT_US,
                Internal::MAX_DISCRETE_REFINEMENT_EVALUATIONS, stats));
            const auto elapsed = autil::TimeUtility::monotonicTimeUs() - begin;
            EXPECT_LE(value, original);
            EXPECT_LE(stats.evaluations, Internal::MAX_DISCRETE_REFINEMENT_EVALUATIONS);
            EXPECT_NEAR(value, Internal::forwardDiscrete(
                context, costs.get(), actual.data(), tokens.data(), inverseTokens.data(),
                count, 0.5, 0.0, workspace), 1e-6);
            if (repeat > 0) {
                durations.push_back(elapsed);
                evaluations.push_back(stats.evaluations);
            }
        }
        std::sort(durations.begin(), durations.end());
        std::sort(evaluations.begin(), evaluations.end());
        const auto prefix = "requests_" + std::to_string(count);
        RecordProperty(prefix + "_median_us", std::to_string(durations[15]));
        RecordProperty(prefix + "_p95_us", std::to_string(durations[28]));
        RecordProperty(prefix + "_max_us", std::to_string(durations.back()));
        RecordProperty(prefix + "_median_evaluations", std::to_string(evaluations[15]));
    }
}

TEST_F(CostSchedulerIncidentTest, testRecoveryAssignmentHasBetterFeasibleTransfer)
{
    NaviSchedScheduleReplay replay;
    loadReplay(replay, "incident_recovery_transfer.json");
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(2, replay.nodes.size());
    ASSERT_EQ(26, replay.requests.size());
    ASSERT_EQ("11.18.53.136", replay.nodes[0].agentId);
    ASSERT_EQ("215.74.13.89", replay.nodes[1].agentId);
    ASSERT_EQ(1791637550762279, replay.scheduleTimeUs);
    // Keep all original hits. Only the unknown destination hits vary, including
    // zero and a full-block hit. The third node's unchanged contribution cancels
    // from the difference; convert the pair's denominator back to the full 27.
    for (int hitCase : {0, 120, 240, -1}) {
        for (size_t i = 0; i < 25; ++i) {
            replay.nodeMatches[i][1].matchedBlockCount = hitCase < 0
                ? replay.requests[i].request.promptTokenCount / 16
                : hitCase;
        }
        NaviSchedScheduleContext context;
        ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
        ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
        std::vector<int64_t> tokens(26);
        CostSchedulerKernelInternal::fillRequestTokenCounts(context, tokens.data(), nullptr);
        CostSchedulerKernelInternal::ReplayInfo info;
        auto costs = prepareHistoricalCosts(
            context, tokens.data(), 0.5, 512, 1, 0, 10, 0, info);
        ASSERT_TRUE(costs);
        std::vector<uint32_t> original(26, 0);
        original[25] = 1;
        auto moved = original;
        std::fill(moved.begin(), moved.begin() + 25, 1);
        const std::vector<double> inverseTokens(26, 1.0);
        CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
        const auto objective = [&](const std::vector<uint32_t> &assignment, double lambda) {
            return CostSchedulerKernelInternal::forwardDiscrete(
                context, costs.get(), assignment.data(), tokens.data(),
                inverseTokens.data(), 26.0, lambda, 0.0, workspace);
        };
        EXPECT_NEAR((25 * 13054.110323660 + 4453.464274475) / 26,
                    objective(original, 0.0), 0.02);
        EXPECT_LT((objective(moved, 0.5) - objective(original, 0.5)) * 26 / 27, -3267.60);
        EXPECT_LT((objective(moved, 0.0) - objective(original, 0.0)) * 26 / 27, -6301.40);
        auto refined = original;
        double refinedValue = 0;
        CostSchedulerKernelInternal::DiscreteRefinementStats refinement;
        ASSERT_TRUE(CostSchedulerKernelInternal::refineDiscreteAssignment(
            context, costs.get(), refined.data(), tokens.data(), inverseTokens.data(),
            26.0, 0.5, 0.0, refinedValue, -1, 4096, refinement));
        EXPECT_LE(refinedValue, objective(original, 0.5));
        RecordProperty("recovery_refined_hit_" + std::to_string(hitCase), std::to_string(refinedValue));
        RecordProperty("recovery_all_moved_hit_" + std::to_string(hitCase),
                       std::to_string(objective(moved, 0.5)));
        RecordProperty("recovery_original_hit_" + std::to_string(hitCase),
                       std::to_string(objective(original, 0.5)));
    }
}

TEST_F(CostSchedulerIncidentTest, testNegativeInsertionsAreNotConfinedToAbsorbingNode)
{
    std::ifstream input(GET_PRIVATE_TEST_DATA_PATH() + "/scheduler/incident_negative_insertions.json");
    ASSERT_TRUE(input.good());
    const std::string text{std::istreambuf_iterator<char>(input), {}};
    rapidjson::Document data;
    data.Parse(text.c_str());
    ASSERT_FALSE(data.HasParseError());
    ASSERT_EQ(6, data["examples"].Size());
    size_t longRequestCases = 0;
    for (const auto &example : data["examples"].GetArray()) {
        const auto &row = example["row"];
        const auto &state = example["state"];
        const auto parameters = array(state["parameters_after"]);
        const auto residual = array(state["residual_parameters_after"]);
        const auto hit = example["cache_hit_tokens"].GetInt64();
        const auto miss = example["prompt_tokens"].GetInt64() - hit;
        SCOPED_TRACE(row["schedule_batch_id"].GetString());
        const auto makeFeatures = [&](bool floor, bool remove) {
            const std::string prefix = floor ? "floor_" : "";
            const auto get = [&](const char *key) { return number(row, (prefix + key).c_str()); };
            const auto r = floor && remove ? 1 : 0;
            const auto w = !floor && remove ? 1 : 0;
            NaviSchedLatencyFeatures features;
            EXPECT_TRUE(features.appendCapacitySplitAggregate(
                get("batch_count") - (remove && number(row, "request_count") == 1),
                get("running_batch_size") - r,
                get("running_hit_tokens") - r * hit, get("running_miss_tokens") - r * miss,
                get("running_miss_square_sum") - r * miss * miss,
                get("running_hit_miss_sum") - r * hit * miss,
                get("waiting_batch_size") - w,
                get("waiting_hit_tokens") - w * hit, get("waiting_miss_tokens") - w * miss,
                get("waiting_miss_square_sum") - w * miss * miss,
                get("waiting_hit_miss_sum") - w * hit * miss));
            EXPECT_TRUE(features.appendPrecedingRequestCount(get("preceding_request_count")));
            if (!floor) {
                EXPECT_TRUE(features.appendRunningElapsedTimeMs(
                    std::stod(row["running_elapsed_time_ms"].GetString())));
            }
            return features;
        };
        const auto decision = [&](const std::vector<double> &p, bool remove) {
            const auto evaluate = [&](bool floor) {
                return LatencyResidualModel::evaluate(
                    historicalPrediction(makeFeatures(floor, remove), p), residual,
                    state["residual_time_after"].GetInt64(), number(row, "schedule_time_us")).predictionMs;
            };
            return std::max(evaluate(false) + std::stod(row["fixed_execution_ms"].GetString()),
                            evaluate(true));
        };
        const auto withRequest = decision(parameters, false);
        const auto withoutRequest = decision(parameters, true);
        EXPECT_NEAR(example["check"]["prediction_ms"].GetDouble(), withRequest, 1e-6);
        EXPECT_NEAR(example["check"]["prediction_without_removed_ms"].GetDouble(), withoutRequest, 1e-6);
        EXPECT_LT(withRequest, withoutRequest - 1.0);
        EXPECT_TRUE(example["completion_rejection"].IsNull());
        EXPECT_GT(example["actual_ms"].GetDouble(), withRequest);
        auto changed = parameters;
        changed[13] = 0;
        const auto &p13Intervention = example["interventions"][1];
        EXPECT_NEAR(p13Intervention["prediction_with_ms"].GetDouble(), decision(changed, false), 1e-6);
        EXPECT_NEAR(p13Intervention["prediction_without_ms"].GetDouble(), decision(changed, true), 1e-6);
        if (example["prompt_tokens"].GetInt64() == 8044) {
            ++longRequestCases;
            EXPECT_LT(withRequest - withoutRequest, -28000);
            EXPECT_GT(decision(changed, false) - decision(changed, true), 16000);
        } else {
            // The small MI308 response is a counterexample to treating P13
            // as the cause of every underprediction.
            EXPECT_EQ("11.225.199.135:27344", std::string(row["agent"].GetString()));
            EXPECT_LT(decision(changed, false) - decision(changed, true), -1);
        }
    }
    EXPECT_EQ(4, longRequestCases);
}

TEST_F(CostSchedulerIncidentTest, testRunningCrossTermUnderpredictsOnTwoOtherNodes)
{
    std::ifstream input(GET_PRIVATE_TEST_DATA_PATH() + "/scheduler/incident_running_cost.json");
    ASSERT_TRUE(input.good());
    const std::string text{std::istreambuf_iterator<char>(input), {}};
    rapidjson::Document data;
    data.Parse(text.c_str());
    ASSERT_FALSE(data.HasParseError());
    ASSERT_EQ(2, data["examples"].Size());
    for (const auto &example : data["examples"].GetArray()) {
        const auto &row = example["row"];
        const auto &state = example["state"];
        auto parameters = array(state["parameters_after"]);
        const auto residual = array(state["residual_parameters_after"]);
        const auto makeFeatures = [&](bool floor) {
            const std::string prefix = floor ? "floor_" : "";
            const auto get = [&](const char *key) {
                return number(row, (prefix + key).c_str());
            };
            NaviSchedLatencyFeatures features;
            EXPECT_TRUE(features.appendCapacitySplitAggregate(
                get("batch_count"), get("running_batch_size"), get("running_hit_tokens"),
                get("running_miss_tokens"), get("running_miss_square_sum"), get("running_hit_miss_sum"),
                get("waiting_batch_size"), get("waiting_hit_tokens"), get("waiting_miss_tokens"),
                get("waiting_miss_square_sum"), get("waiting_hit_miss_sum")));
            EXPECT_TRUE(features.appendPrecedingRequestCount(get("preceding_request_count")));
            if (!floor) {
                EXPECT_TRUE(features.appendRunningElapsedTimeMs(
                    std::stod(row["running_elapsed_time_ms"].GetString())));
            }
            return features;
        };
        const auto full = makeFeatures(false);
        const auto floor = makeFeatures(true);
        const auto adjusted = [&](const NaviSchedLatencyFeatures &features) {
            return LatencyResidualModel::evaluate(
                historicalPrediction(features, parameters), residual,
                state["residual_time_after"].GetInt64(), number(row, "schedule_time_us")).predictionMs;
        };
        const auto decision = [&]() {
            return std::max(adjusted(full) + std::stod(row["fixed_execution_ms"].GetString()),
                            adjusted(floor));
        };
        EXPECT_NEAR(std::stod(row["with_base_ms"].GetString()),
                    historicalPrediction(full, parameters), 1e-6);
        const auto original = decision();
        EXPECT_NEAR(example["prediction_ms"].GetDouble(), original, 1e-6);
        EXPECT_LT(NonLinearPrefillModelR::calculateRequestLinearCost(parameters, 8044, 3840), 0.0);
        EXPECT_FALSE(NonLinearPrefillModelR::validateParameters(parameters));
        EXPECT_FALSE(NonLinearPrefillModelR::hasNonNegativeWaitingWorkloadCost(parameters, full));
        const auto rawInput = NonLinearPrefillModelR::makeInput(full);
        double historicalWaitingCost = 0.0;
        for (size_t i = 0; i < NonLinearPrefillModelR::WAITING_PARAMETER_COUNT; ++i) {
            const auto index = NonLinearPrefillModelR::WAITING_PARAMETER_OFFSET + i;
            historicalWaitingCost += rawInput[7 + i] * parameters[index] *
                NonLinearPrefillModelR::getParameterScale(index);
        }
        EXPECT_GE(historicalWaitingCost, 0.0);
        const auto oldP5 = parameters[5];
        parameters[5] = 0.0;
        EXPECT_NEAR(example["zero_p5_prediction_ms"].GetDouble(), decision(), 1e-6);
        EXPECT_GT(decision() - original, 60000.0);
        parameters[5] = oldP5;
        EXPECT_NEAR(original, decision(), 1e-6);
    }
}

TEST_F(CostSchedulerIncidentTest, testHistoricalCapacityBoundaryReassignment)
{
    std::ifstream input(GET_PRIVATE_TEST_DATA_PATH() + "/scheduler/incident_capacity_boundary.json");
    ASSERT_TRUE(input.good());
    const std::string text{std::istreambuf_iterator<char>(input), {}};
    rapidjson::Document data;
    data.Parse(text.c_str());
    ASSERT_FALSE(data.HasParseError());
    ASSERT_EQ(20, data["cases"].Size());
    size_t checks = 0;
    size_t negative = 0;
    for (const auto &row : data["cases"].GetArray()) {
        SCOPED_TRACE(row["batch"].GetString());
        const auto &requests = row["requests"];
        const auto &state = row["state"];
        const auto &event = state["event"];
        const auto count = requests.Size();
        ASSERT_EQ(count, row["request_count"].GetUint());
        ASSERT_EQ(1500.0, event["nonlinear_scale"].GetDouble());
        LatencyResidualModel::RuntimeConfig config;
        config.idleDecayStartUs = event["residual_idle_decay_start_us"].GetInt64();
        config.idleDecayEndUs = event["residual_idle_decay_end_us"].GetInt64();
        const auto residual = array(state["residual"]);
        auto parameters = array(state["parameters"]);
        const auto makeFeatures = [&](bool floor, size_t remove, bool promote) {
            const auto &expected = row[floor ? "floor_features" : "features"];
            const auto capacity = row[floor ? "floor_capacity" : "capacity"].GetUint();
            NaviSchedLatencyFeatures result;
            EXPECT_TRUE(result.appendCapacitySplitAggregate(
                static_cast<int64_t>(expected[6].GetDouble()), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
            EXPECT_TRUE(result.appendPrecedingRequestCount(
                static_cast<int64_t>(expected[12].GetDouble())));
            EXPECT_TRUE(result.appendRunningElapsedTimeMs(
                expected[13].GetDouble() * NonLinearPrefillModelR::RUNNING_ELAPSED_TIME_SCALE_MS));
            size_t position = 0;
            for (size_t index = 0; index < count; ++index) {
                if (index == remove) {
                    continue;
                }
                const auto hit = requests[index]["hit"].GetInt64();
                const auto miss = requests[index]["prompt"].GetInt64() - hit;
                if ((promote ? position : index) < capacity) {
                    result.appendRequest(hit, miss);
                } else {
                    result.appendWaitingRequest(hit, miss);
                }
                ++position;
            }
            if (remove == count) {
                const auto actual = NonLinearPrefillModelR::makeInput(result);
                EXPECT_EQ(actual.size(), expected.Size());
                for (size_t index = 0; index < std::min<size_t>(actual.size(), expected.Size()); ++index) {
                    EXPECT_DOUBLE_EQ(expected[index].GetDouble(), actual[index]);
                }
            }
            return result;
        };
        const auto predict = [&](size_t remove, bool promote) {
            const auto evaluate = [&](bool floor) {
                return LatencyResidualModel::evaluate(
                    historicalPrediction(makeFeatures(floor, remove, promote), parameters),
                    residual, event["residual_time_after"].GetInt64(),
                    row["time_us"].GetInt64(), config).predictionMs;
            };
            return std::max(evaluate(false) + row["fixed_execution_ms"].GetDouble(), evaluate(true));
        };
        for (const auto *mode : {"recorded", "diagnostic_clamp_only_not_trained"}) {
            const bool clamp = std::string(mode) != "recorded";
            if (clamp) {
                for (size_t index : {1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14}) {
                    parameters[index] = std::max(0.0, parameters[index]);
                }
            }
            const auto before = predict(count, true);
            const auto &expected = row["models"][mode];
            ASSERT_EQ(count, expected.Size());
            for (size_t index = 0; index < count; ++index) {
                SCOPED_TRACE(index);
                const auto after = predict(index, true);
                EXPECT_NEAR(expected[index]["before_ms"].GetDouble(), before, 1e-6);
                EXPECT_NEAR(expected[index]["after_ms"].GetDouble(), after, 1e-6);
                EXPECT_NEAR(expected[index]["no_promotion_insertion_delta_ms"].GetDouble(),
                            before - predict(index, false), 1e-6);
                if (clamp) {
                    EXPECT_GT(before - after, 0.0);
                } else {
                    ++checks;
                    if (before - after < -1.0) {
                        ++negative;
                        EXPECT_LT(before - predict(index, false), -1.0);
                    }
                }
            }
        }
        if (!row["group_interventions"].IsNull()) {
            for (const auto *mode : {"running_only", "waiting_only"}) {
                parameters = array(state["parameters"]);
                const size_t offset = std::string(mode) == "running_only" ? 1 : 10;
                for (size_t index = offset; index < offset + 5; ++index) {
                    parameters[index] = std::max(0.0, parameters[index]);
                }
                const auto &expected = row["group_interventions"][mode];
                const auto before = predict(count, true);
                EXPECT_NEAR(expected["prediction_ms"].GetDouble(), before, 1e-6);
                for (const auto &check : expected["checks"].GetArray()) {
                    size_t index = 0;
                    for (; index < count; ++index) {
                        if (std::string(requests[index]["request_id"].GetString()) ==
                            check["removed"].GetString()) {
                            break;
                        }
                    }
                    ASSERT_LT(index, count);
                    const auto delta = before - predict(index, true);
                    EXPECT_NEAR(check["delta_ms"].GetDouble(), delta, 1e-6);
                    if (offset == 1) {
                        EXPECT_GT(delta, 0.0);
                    } else {
                        EXPECT_LT(delta, -1.0);
                    }
                }
            }
        }
    }
    EXPECT_EQ(299, checks);
    EXPECT_EQ(6, negative);
}

TEST_F(CostSchedulerIncidentTest, testIncidentObjectiveGradientMatchesFiniteDifference)
{
    NaviSchedScheduleReplay replay;
    loadReplay(replay);
    ASSERT_FALSE(HasFatalFailure());
    NaviSchedScheduleContext context;
    ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
    ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
    std::vector<int64_t> tokens(9);
    CostSchedulerKernelInternal::fillRequestTokenCounts(context, tokens.data(), nullptr);
    const std::vector<double> inverseTokens(9, 1.0);
    CostSchedulerKernelInternal::ReplayInfo replayInfo;
    auto costs = prepareHistoricalCosts(
        context, tokens.data(), 0.5, 1, 1, 0, 10, 0, replayInfo);
    ASSERT_TRUE(costs);
    std::vector<double> weights(18, 0.25);
    std::fill(weights.begin() + 9, weights.end(), 0.75);
    std::vector<CostSchedulerKernelInternal::NodeGradient> nodeGradients(2);
    std::vector<double> gradients(18);
    CostSchedulerKernelInternal::forwardNodeRange(
        context, costs.get(), weights.data(), tokens.data(), inverseTokens.data(),
        9.0, 0.5, 0.0, 0, 2, nodeGradients.data(), nullptr, nullptr);
    CostSchedulerKernelInternal::backwardNodeRange(
        context, costs.get(), weights.data(), tokens.data(), inverseTokens.data(),
        9, 0, 2, nodeGradients.data(), gradients.data());
    const auto forward = [&](const std::vector<double> &w) {
        return CostSchedulerKernelInternal::forwardNodeRange(
            context, costs.get(), w.data(), tokens.data(), inverseTokens.data(),
            9.0, 0.5, 0.0, 0, 2, nullptr, nullptr, nullptr);
    };
    for (size_t i = 0; i < weights.size(); ++i) {
        auto lower = weights;
        auto upper = weights;
        lower[i] -= 1e-6;
        upper[i] += 1e-6;
        const auto numeric = (forward(upper) - forward(lower)) / 2e-6;
        EXPECT_NEAR(numeric, gradients[i], std::max(1e-4, std::abs(numeric) * 1e-6))
            << "weight_index=" << i;
    }
}

TEST_F(CostSchedulerIncidentTest, testBetterDiscreteCandidateSurvivesBetweenRepairChecks)
{
    NaviSchedScheduleReplay replay;
    loadReplay(replay);
    ASSERT_FALSE(HasFatalFailure());
    for (auto &node : replay.nodes) {
        for (const auto index : NonLinearPrefillModelR::POSITIVE_PARAMETER_INDEXES) {
            node.latencyParameters[index] = std::max(
                NonLinearPrefillModelR::positiveParameterFloor(index, 1.0),
                std::abs(node.latencyParameters[index]));
        }
    }
    auto context = std::make_shared<NaviSchedScheduleContext>();
    ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, *context));
    ASSERT_TRUE(ScheduleReplayRunner::prepareContext(*context));
    CostSchedulerConfig config;
    config.optimizationTimeBudgetUs = -1;
    config.discreteCheckInterval = 5;
    auto state = CostSchedulerGraphState::create(context, config, 1);
    ASSERT_TRUE(state);
    ASSERT_TRUE(state->runRow(0, 0));
    const auto count = context->requestCount();
    ASSERT_EQ(9u, count);
    ASSERT_EQ(2u, context->nodeCount());
    CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
    double best = std::numeric_limits<double>::infinity();
    double worst = -std::numeric_limits<double>::infinity();
    std::vector<uint32_t> bestAssignment(count), worstAssignment(count);
    for (size_t mask = 0; mask < (1u << count); ++mask) {
        std::vector<uint32_t> assignment(count);
        for (size_t request = 0; request < count; ++request) {
            assignment[request] = (mask >> request) & 1;
        }
        const auto loss = CostSchedulerKernelInternal::forwardDiscrete(
            *context, state->_costs.get(), assignment.data(),
            state->_requestTokenCounts.get(), state->_inverseRequestTokenCounts.get(),
            state->_tokenNormalizationDenominator, config.lambda,
            config.latencyWeightConcavity, workspace);
        if (loss < best) {
            best = loss;
            bestAssignment = assignment;
        }
        if (loss > worst) {
            worst = loss;
            worstAssignment = assignment;
        }
    }
    ASSERT_LT(best, worst);
    state->_bestLoss = worst;
    std::copy(worstAssignment.begin(), worstAssignment.end(), state->_bestNodeIndexes.get());
    std::copy(bestAssignment.begin(), bestAssignment.end(), state->_candidateNodeIndexes.get());
    ASSERT_TRUE(state->evaluateDiscreteCandidate(1, best));
    EXPECT_NEAR(best, state->_bestLoss, 1e-12);
    EXPECT_EQ(bestAssignment, std::vector<uint32_t>(
        state->_bestNodeIndexes.get(), state->_bestNodeIndexes.get() + count));
    EXPECT_FALSE(state->_discreteRepairActive);
}

TEST_F(CostSchedulerIncidentTest, testBacktrackingAloneDoesNotFixDiscreteAssignment)
{
    NaviSchedScheduleReplay replay;
    loadReplay(replay);
    ASSERT_FALSE(HasFatalFailure());
    NaviSchedScheduleContext context;
    ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
    ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
    std::vector<int64_t> tokens(9);
    CostSchedulerKernelInternal::fillRequestTokenCounts(context, tokens.data(), nullptr);
    const std::vector<double> inverseTokens(9, 1.0);
    CostSchedulerKernelInternal::ReplayInfo info;
    auto costs = prepareHistoricalCosts(
        context, tokens.data(), 0.5, 512, 1, 0, 10, -1, info);
    ASSERT_TRUE(costs);
    CostSchedulerKernelInternal::DiscreteForwardWorkspace discreteWorkspace;
    const auto discrete = [&](const std::vector<uint32_t> &assignment) {
        return CostSchedulerKernelInternal::forwardDiscrete(
            context, costs.get(), assignment.data(), tokens.data(), inverseTokens.data(),
            9.0, 0.5, 0.0, discreteWorkspace);
    };
    const auto forward = [&](const std::vector<double> &weights,
                             CostSchedulerKernelInternal::NodeGradient *gradients) {
        return CostSchedulerKernelInternal::forwardNodeRange(
            context, costs.get(), weights.data(), tokens.data(), inverseTokens.data(),
            9.0, 0.5, 0.0, 0, 2, gradients, nullptr, nullptr);
    };
    double bestAcrossSeeds = std::numeric_limits<double>::infinity();
    for (uint64_t seed : {1, 7, 31}) {
        std::vector<double> weights(18), gradients(18), workspace(18);
        std::vector<uint32_t> selected(9), bestSelected(9);
        CostSchedulerKernelInternal::initializeWeights(weights.data(), 9, 2, seed);
        ASSERT_TRUE(CostSchedulerKernelInternal::projectRequestRange(
            weights.data(), 9, 2, 0, 9, workspace.data(), selected.data(), nullptr, 0, 0, seed));
        auto best = discrete(selected);
        bestSelected = selected;
        double alpha = 512;
        size_t evaluations = 0;
        std::vector<CostSchedulerKernelInternal::NodeGradient> nodeGradients(2);
        for (size_t iteration = 0; iteration < 10; ++iteration) {
            const auto oldValue = forward(weights, nodeGradients.data());
            CostSchedulerKernelInternal::backwardNodeRange(
                context, costs.get(), weights.data(), tokens.data(), inverseTokens.data(),
                9, 0, 2, nodeGradients.data(), gradients.data());
            CostSchedulerKernelInternal::GradientSpan span{gradients.data(), 0, 2};
            bool accepted = false;
            for (size_t trial = 0; trial < 32; ++trial) {
                auto candidate = weights;
                ASSERT_TRUE(CostSchedulerKernelInternal::projectRequestRange(
                    candidate.data(), 9, 2, 0, 9, workspace.data(), selected.data(), &span, 1, alpha, seed));
                const auto value = forward(candidate, nullptr);
                ++evaluations;
                double directionalChange = 0;
                for (size_t i = 0; i < weights.size(); ++i) {
                    directionalChange += gradients[i] * (candidate[i] - weights[i]);
                }
                if (value <= oldValue + 1e-4 * directionalChange + 1e-9) {
                    EXPECT_LE(value, oldValue + 1e-9);
                    weights = std::move(candidate);
                    const auto selectedValue = discrete(selected);
                    if (selectedValue < best) {
                        best = selectedValue;
                        bestSelected = selected;
                    }
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            ASSERT_TRUE(accepted);
        }
        RecordProperty("backtracking_objective_seed_" + std::to_string(seed), std::to_string(best));
        RecordProperty("backtracking_trials_seed_" + std::to_string(seed), std::to_string(evaluations));
        RecordProperty("backtracking_alpha_seed_" + std::to_string(seed), std::to_string(alpha));
        EXPECT_GE(best, 18659.546377861727 - 1e-6);
        bestAcrossSeeds = std::min(bestAcrossSeeds, best);
        // A separate discrete reference distinguishes continuous convergence
        // from rounding quality. This is an offline experiment, not a fix.
        auto local = best;
        for (size_t pass = 0; pass < 9; ++pass) {
            auto next = bestSelected;
            auto nextValue = local;
            for (size_t i = 0; i < bestSelected.size(); ++i) {
                auto candidate = bestSelected;
                candidate[i] = 1 - candidate[i];
                const auto value = discrete(candidate);
                if (value < nextValue) {
                    nextValue = value;
                    next = std::move(candidate);
                }
            }
            if (nextValue >= local) {
                break;
            }
            local = nextValue;
            bestSelected = std::move(next);
        }
        RecordProperty("backtracking_with_discrete_refinement_seed_" + std::to_string(seed), std::to_string(local));
        EXPECT_LE(local, best);
        EXPECT_NEAR(18659.546377861727, local, 1e-6);
    }
    EXPECT_NEAR(18766.183505953941, bestAcrossSeeds, 1e-6);
}

TEST_F(CostSchedulerIncidentTest, testSerialReplayCollectionDoesNotConsumeOptimizationBudget)
{
    NaviSchedScheduleReplay replay;
    loadReplay(replay);
    ASSERT_FALSE(HasFatalFailure());
    for (auto &node : replay.nodes) {
        for (const auto index : NonLinearPrefillModelR::POSITIVE_PARAMETER_INDEXES) {
            node.latencyParameters[index] = std::max(
                NonLinearPrefillModelR::positiveParameterFloor(index, 1.0),
                std::abs(node.latencyParameters[index]));
        }
    }
    constexpr int64_t PREPARATION_US = 60'000'000;
    constexpr int64_t BUDGET_US = 5'000'000;
    constexpr size_t LOOP_COUNT = 3;
    for (const bool replayPreparation : {false, true}) {
        SCOPED_TRACE(replayPreparation);
        NaviSchedScheduleContext context;
        ASSERT_TRUE(ScheduleReplayRunner::makeContext(replay, context));
        ASSERT_TRUE(ScheduleReplayRunner::prepareContext(context));
        std::vector<int64_t> tokens(context.requestCount());
        const auto totalTokens = CostSchedulerKernelInternal::fillRequestTokenCounts(
            context, tokens.data(), nullptr, 0.0);
        auto costs = CostSchedulerKernelInternal::prepare(
            context, tokens.data(), 0.5, 1.0, 1.0, 0.0,
            LOOP_COUNT, BUDGET_US, nullptr);
        ASSERT_TRUE(costs);
        CostSchedulerKernelInternal::ReplayInfo trace;
        trace.init(true, 0.5, 1.0, 1.0, 0.0, LOOP_COUNT, BUDGET_US);
        std::vector<double> weights(context.requestCount() * context.nodeCount());
        std::vector<uint32_t> selected;
        double loss = 0.0;
        // Inject elapsed preparation without sleeping; only its ownership differs.
        context._scheduleBeginTimeUs = autil::TimeUtility::monotonicTimeUs() - PREPARATION_US;
        context._scheduleDeadlineUs = context._scheduleBeginTimeUs + BUDGET_US;
        if (replayPreparation) {
            context.addReplayCollectionTimeUs(PREPARATION_US);
        }
        ASSERT_TRUE(CostSchedulerKernelInternal::optimize(
            context, costs.get(), tokens.data(), totalTokens,
            0.5, 1.0, 1.0, 0.0, LOOP_COUNT, BUDGET_US,
            &trace, 0.0, 1, weights.data(), selected, loss,
            0.0, 5, 0.25, 0.7, 1));
        ASSERT_EQ(1u, trace.optimizationRuns.size());
        const auto &run = trace.optimizationRuns.front();
        EXPECT_EQ(replayPreparation ? LOOP_COUNT : 1u, run.actualLoopCount);
        EXPECT_EQ(replayPreparation ? "max_loop_count" : "time_budget", run.stopReason);
        if (replayPreparation) {
            EXPECT_GT(run.refinementEvaluations, 0u);
            EXPECT_GE(context.getReplayCollectionTimeUs(), PREPARATION_US);
        } else {
            EXPECT_EQ(0u, run.refinementEvaluations);
            EXPECT_LT(context.getReplayCollectionTimeUs(), PREPARATION_US);
        }
        EXPECT_EQ(context.requestCount(), selected.size());
        EXPECT_TRUE(std::isfinite(loss));
    }
}

} // namespace navi_sched
