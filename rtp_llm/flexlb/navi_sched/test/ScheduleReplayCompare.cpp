#include "rtp_llm/flexlb/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "rtp_llm/flexlb/navi_sched/data/BlockHashUtil.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerKernel.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/FlexLBBaselineSchedulerKernel.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/MaxCacheHitSchedulerKernel.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/RandomSchedulerKernel.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerKernelInternal.h"
#include "rtp_llm/flexlb/navi_sched/test/ScheduleReplayRunner.h"
#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"
#include <autil/TimeUtility.h>
#include <autil/StringUtil.h>
#include <autil/legacy/jsonizable.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace navi_sched {

class ScheduleReplayCompare {
private:
    struct Statistics {
        std::string name;
        size_t replayCount = 0;
        size_t replayFailureCount = 0;
        size_t recordedMatchCount = 0;
        size_t requestCount = 0;
        size_t scheduledCount = 0;
        size_t rejectedCount = 0;
        size_t invalidTargetCount = 0;
        size_t scheduledBatchCount = 0;
        int64_t scheduleElapsedUs = 0;
        int64_t matchedBlockCount = 0;
        int64_t matchedTokenCount = 0;
        double queueMs = 0.0;
        double batchExecutionLatencyMs = 0.0;
    };

public:
    struct Options {
        bool printSummary = false;
        bool optimizerExperiment = false;
        bool requireRecordedMatch = false;
        std::string replayOutputPath;
        double costLambda = 0.5;
        double costAlpha = 1.0;
        double costAlphaDecay = 1.0;
        double costMinAlpha = 0.0;
        int32_t costMaxLoopCount = 10;
        double costLatencyWeightConcavity =
            std::numeric_limits<double>::quiet_NaN();
        double costLatencyTokenNormalizationPower =
            std::numeric_limits<double>::quiet_NaN();
    };

    static int run(const std::string &kernelId, const std::string &path,
                   const Options &options)
    {
        if (!supported(kernelId)) {
            std::cerr << "unsupported scheduler kernel: " << kernelId << std::endl;
            return 2;
        }
        std::ifstream input(path);
        if (!input) {
            std::cerr << "failed to open replay file: " << path << std::endl;
            return 1;
        }
        Statistics statistics{kernelId};
        size_t decodeFailureCount = 0;
        std::ofstream replayOutput;
        if (!options.replayOutputPath.empty()) {
            replayOutput.open(options.replayOutputPath);
            if (!replayOutput) {
                std::cerr << "failed to open replay output: "
                          << options.replayOutputPath << std::endl;
                return 1;
            }
        }
        auto replayOutputPtr = replayOutput.is_open() ? &replayOutput : nullptr;
        if (endsWith(path, ".json")) {
            const std::string json{std::istreambuf_iterator<char>(input),
                                   std::istreambuf_iterator<char>()};
            decodeAndEvaluate(json, kernelId, options, replayOutputPtr,
                              statistics, decodeFailureCount);
        } else {
            std::string line;
            while (std::getline(input, line)) {
                if (!line.empty()) {
                    decodeAndEvaluate(line, kernelId, options,
                                      replayOutputPtr, statistics,
                                      decodeFailureCount);
                }
            }
        }
        print(statistics);
        return statistics.replayCount > 0 && 0 == statistics.replayFailureCount &&
                       0 == decodeFailureCount &&
                       (!options.requireRecordedMatch ||
                        statistics.recordedMatchCount == statistics.replayCount)
                   ? 0
                   : 1;
    }

private:
    static bool endsWith(const std::string &value, const std::string &suffix)
    {
        return value.size() >= suffix.size() &&
               0 == value.compare(value.size() - suffix.size(), suffix.size(), suffix);
    }

    static void decodeAndEvaluate(const std::string &json, const std::string &kernelId,
                                  const Options &options,
                                  std::ostream *replayOutput,
                                  Statistics &statistics,
                                  size_t &decodeFailureCount)
    {
        NaviSchedScheduleReplay replay;
        try {
            NaviSchedScheduleReplayJson::fromJsonString(json, replay);
        } catch (const std::exception &e) {
            ++decodeFailureCount;
            std::cerr << "replay decode failed: " << e.what() << std::endl;
            return;
        }
        evaluate(replay, kernelId, options, replayOutput, statistics);
    }

    static bool supported(const std::string &kernelId)
    {
        return RandomSchedulerKernel::KERNEL_ID == kernelId ||
               MaxCacheHitSchedulerKernel::KERNEL_ID == kernelId ||
               FlexLBBaselineSchedulerKernel::KERNEL_ID == kernelId ||
               CostSchedulerKernel::KERNEL_ID == kernelId;
    }

    static void evaluate(const NaviSchedScheduleReplay &replay,
                         const std::string &kernelId,
                         const Options &options,
                         std::ostream *replayOutput,
                         Statistics &statistics)
    {
        if (RandomSchedulerKernel::KERNEL_ID == kernelId) {
            RandomSchedulerKernel scheduler;
            evaluate(replay, scheduler, options.printSummary,
                     replayOutput, statistics);
        } else if (MaxCacheHitSchedulerKernel::KERNEL_ID == kernelId) {
            MaxCacheHitSchedulerKernel scheduler;
            evaluate(replay, scheduler, options.printSummary,
                     replayOutput, statistics);
        } else if (FlexLBBaselineSchedulerKernel::KERNEL_ID == kernelId) {
            FlexLBBaselineSchedulerKernel scheduler;
            evaluate(replay, scheduler, options.printSummary,
                     replayOutput, statistics);
        } else {
            evaluateCost(replay, options, replayOutput, statistics);
        }
    }

    static void evaluateCost(const NaviSchedScheduleReplay &replay,
                             const Options &options,
                             std::ostream *replayOutput,
                             Statistics &statistics)
    {
        auto context = std::make_shared<NaviSchedScheduleContext>();
        std::vector<NaviSchedDecision> decisions;
        if (!ScheduleReplayRunner::makeContext(replay, *context)) {
            ++statistics.replayCount;
            ++statistics.replayFailureCount;
            return;
        }
        std::shared_ptr<NaviSchedScheduleReplay> collectedReplay;
        if (options.printSummary || replayOutput) {
            collectedReplay =
                std::make_shared<NaviSchedScheduleReplay>(replay);
            ScheduleReplayRunner::setReplay(*context, collectedReplay);
        }
        if (options.printSummary) {
            printContextSummary(*context);
        }
        auto scheduleBeginUs = autil::TimeUtility::currentTime();
        if (!ScheduleReplayRunner::runCostScheduler(
                replay, context, options.costLambda, options.costAlpha,
                options.costAlphaDecay, options.costMinAlpha,
                options.costMaxLoopCount,
                std::isfinite(options.costLatencyWeightConcavity)
                    ? options.costLatencyWeightConcavity
                    : 128.0,
                decisions,
                options.costLatencyTokenNormalizationPower,
                options.costLatencyWeightConcavity,
                [&](const NaviSchedScheduleContext &input) {
                    if (options.optimizerExperiment) {
                        runOptimizerExperiment(input, replay.schedulerInfo);
                    }
                    if (options.printSummary && !replay.schedulerInfo.empty()) {
                        printFrozenCostAudit(input, replay.schedulerInfo);
                    }
                })) {
            ++statistics.replayCount;
            ++statistics.replayFailureCount;
            return;
        }
        statistics.scheduleElapsedUs +=
            autil::TimeUtility::currentTime() - scheduleBeginUs;
        if (options.printSummary) {
            printSchedulerInfo(
                *context, CostSchedulerKernel::KERNEL_ID,
                ScheduleReplayRunner::getSchedulerInfo(*context));
        }
        if (replayOutput) {
            collectedReplay->decisions = decisions;
            if (const auto *schedulerInfo =
                    ScheduleReplayRunner::getSchedulerInfo(*context)) {
                collectedReplay->schedulerInfo = *schedulerInfo;
            }
            *replayOutput
                << NaviSchedScheduleReplayJson::toJsonString(*collectedReplay)
                << std::endl;
            if (!*replayOutput) {
                ++statistics.replayFailureCount;
                return;
            }
        }
        add(replay, decisions,
            ScheduleReplayRunner::decisionsEqual(
                replay.decisions, decisions, replay.bizName),
            *context, statistics);
    }

    template <typename Scheduler>
    static void evaluate(const NaviSchedScheduleReplay &replay,
                         const Scheduler &scheduler, bool printSummary,
                         std::ostream *replayOutput,
                         Statistics &statistics)
    {
        NaviSchedScheduleContext context;
        std::vector<NaviSchedDecision> decisions;
        if (!ScheduleReplayRunner::makeContext(replay, context)) {
            ++statistics.replayCount;
            ++statistics.replayFailureCount;
            return;
        }
        std::shared_ptr<NaviSchedScheduleReplay> collectedReplay;
        if (printSummary || replayOutput) {
            collectedReplay = std::make_shared<NaviSchedScheduleReplay>(replay);
            ScheduleReplayRunner::setReplay(context, collectedReplay);
        }
        if (printSummary) {
            printContextSummary(context);
        }
        auto scheduleBeginUs = autil::TimeUtility::currentTime();
        if (!ScheduleReplayRunner::runContext(context, scheduler, decisions)) {
            ++statistics.replayCount;
            ++statistics.replayFailureCount;
            return;
        }
        statistics.scheduleElapsedUs +=
            autil::TimeUtility::currentTime() - scheduleBeginUs;
        if (printSummary) {
            printSchedulerInfo(context, Scheduler::KERNEL_ID,
                               ScheduleReplayRunner::getSchedulerInfo(context));
        }
        if (replayOutput) {
            collectedReplay->decisions = decisions;
            *replayOutput
                << NaviSchedScheduleReplayJson::toJsonString(*collectedReplay)
                << std::endl;
            if (!*replayOutput) {
                ++statistics.replayFailureCount;
                return;
            }
        }
        add(replay, decisions,
            ScheduleReplayRunner::decisionsEqual(
                replay.decisions, decisions, replay.bizName), context,
            statistics);
    }

    static void printContextSummary(const NaviSchedScheduleContext &context)
    {
        std::cout << "context_summary:" << std::endl;
        std::cout << "  requests: " << context.requestCount() << std::endl;
        std::cout << "  nodes: " << context.nodeCount() << std::endl;
        std::cout << "  request_list";
        if (context.requestCount() > 0) {
            std::cout << " (biz="
                      << context.getRequest(0).request.bizName << ")";
        }
        std::cout << ":" << std::endl;
        std::cout << "    " << std::left << std::setw(12) << "request"
                  << std::right << std::setw(15) << "prompt_tokens"
                  << std::setw(12) << "block_size"
                  << std::setw(19) << "block_hash_count"
                  << std::setw(10) << "wait_ms"
                  << std::setw(22) << "remaining_timeout_ms"
                  << std::setw(19) << "timeout_triggered"
                  << std::setw(14) << "request_id" << std::endl;
        for (size_t requestIndex = 0; requestIndex < context.requestCount();
             ++requestIndex) {
            const auto &item = context.getRequest(requestIndex);
            const auto &request = item.request;
            auto label = "request[" + std::to_string(requestIndex) + "]";
            std::cout << "    " << std::left << std::setw(12) << label
                      << std::right << std::setw(15)
                      << request.promptTokenCount
                      << std::setw(12) << request.blockSize
                      << std::setw(19)
                      << BlockHashUtil::get(request.blockHashes).size()
                      << std::setw(10) << item.waitTimeMs
                      << std::setw(22) << item.remainingTimeoutMs
                      << std::setw(19) << std::boolalpha
                      << item.timeoutTriggered << std::noboolalpha
                      << std::setw(14) << request.requestId << std::endl;
        }
        std::cout << "  node_list:" << std::endl;
        for (size_t nodeIndex = 0; nodeIndex < context.nodeCount(); ++nodeIndex) {
            const auto &node = context.getNode(nodeIndex);
            std::cout << "    node[" << nodeIndex << "]: agent_id="
                      << node.getAgentId() << " total_blocks="
                      << node.getTotalBlocks() << " inflight="
                      << node.getInflightRequests() << " waiting_for_batch="
                      << node.getBatchWaitingRequests() << " queued="
                      << node.getQueuedBatchRequests() << " running="
                      << node.getRunningRequests() << " queue_wait_ms="
                      << std::fixed << std::setprecision(3)
                      << node.getEstimatedQueueWaitMs() << " model_formula="
                      << node.getLatencyFormula() << " model_revision="
                      << node.getLatencyModelRevision() << " model_parameters=";
            printValues(node.getLatencyParameters());
            std::cout << std::endl;
        }
        std::cout << "  cache_hit_blocks:" << std::endl;
        constexpr int requestColumnWidth = 24;
        constexpr int nodeColumnWidth = 10;
        std::cout << "    " << std::left << std::setw(requestColumnWidth) << "request";
        for (size_t nodeIndex = 0; nodeIndex < context.nodeCount(); ++nodeIndex) {
            auto label = "node[" + std::to_string(nodeIndex) + "]";
            std::cout << std::right << std::setw(nodeColumnWidth) << label;
        }
        std::cout << std::endl;
        for (size_t requestIndex = 0; requestIndex < context.requestCount();
             ++requestIndex) {
            const auto requestId = context.getRequest(requestIndex).request.requestId;
            auto label = "request[" + std::to_string(requestIndex) + "] id=" +
                         std::to_string(requestId);
            std::cout << "    " << std::left << std::setw(requestColumnWidth) << label;
            for (size_t nodeIndex = 0; nodeIndex < context.nodeCount(); ++nodeIndex) {
                std::cout << std::right << std::setw(nodeColumnWidth)
                          << context.getCacheHitBlockCount(requestIndex, nodeIndex);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    static void printValues(const std::vector<double> &values)
    {
        std::cout << '[';
        for (size_t index = 0; index < values.size(); ++index) {
            if (index > 0) {
                std::cout << ',';
            }
            std::cout << values[index];
        }
        std::cout << ']';
    }

    static void printSchedulerInfo(
        const NaviSchedScheduleContext &context, const std::string &kernelId,
        const autil::legacy::json::JsonMap *schedulerInfo)
    {
        std::cout << "scheduler_info:" << std::endl;
        if (!schedulerInfo || schedulerInfo->empty()) {
            std::cout << "  not collected" << std::endl << std::endl;
            return;
        }
        if (CostSchedulerKernel::KERNEL_ID == kernelId) {
            printCostSchedulerInfo(context, *schedulerInfo);
            return;
        }
        std::cout << autil::legacy::ToJsonString(*schedulerInfo, false)
                  << std::endl
                  << std::endl;
    }

    static void printCostSchedulerInfo(
        const NaviSchedScheduleContext &context,
        const autil::legacy::json::JsonMap &schedulerInfo)
    {
        CostSchedulerKernelInternal::ReplayInfo replayInfo;
        autil::legacy::FromJson(
            replayInfo, autil::legacy::Any(schedulerInfo));
        if (replayInfo.schedInfos.size() != context.nodeCount()) {
            std::cout << "  invalid node count: "
                      << replayInfo.schedInfos.size() << std::endl
                      << std::endl;
            return;
        }
        int64_t totalTokenCount = 0;
        for (size_t requestIndex = 0;
             requestIndex < context.requestCount(); ++requestIndex) {
            totalTokenCount += context.getRequest(
                requestIndex).request.promptTokenCount;
        }
        std::cout << "  scheduler=cost layout=node_major requests="
                  << context.requestCount() << " nodes=" << context.nodeCount()
                  << " total_tokens=" << totalTokenCount
                  << " lambda=" << replayInfo.lambda
                  << " alpha=" << replayInfo.alpha
                  << " alpha_decay=" << replayInfo.alphaDecay
                  << " min_alpha=" << replayInfo.minAlpha
                  << " max_loop_count=" << replayInfo.maxLoopCount
                  << std::endl;
        if (!replayInfo.optimizationRuns.empty()) {
            const auto &run = replayInfo.optimizationRuns.front();
            std::cout << "  cost_final_loss=" << run.finalForwardValue
                      << " cost_best_loss=" << run.bestForwardValue
                      << std::endl;
            printSelectedCostComponents(context, replayInfo, run);
        }
        std::cout << "  cell: c=cost, q=queue_wait_ms, w=weight" << std::endl;
        for (size_t nodeIndex = 0; nodeIndex < context.nodeCount(); ++nodeIndex) {
            const auto &nodeSchedInfos = replayInfo.schedInfos[nodeIndex];
            if (nodeSchedInfos.size() != context.requestCount()) {
                std::cout << "  invalid request count for node[" << nodeIndex
                          << "]: " << nodeSchedInfos.size() << std::endl
                          << std::endl;
                return;
            }
        }
        printCostSchedulerTable(context, replayInfo);
        std::cout << std::endl;
    }

    static void printSelectedCostComponents(
        const NaviSchedScheduleContext &context,
        const CostSchedulerKernelInternal::ReplayInfo &replayInfo,
        const CostSchedulerKernelInternal::ReplayInfo::OptimizationRun &run)
    {
        const auto requestCount = context.requestCount();
        const auto nodeCount = context.nodeCount();
        if (requestCount == 0 || nodeCount == 0 ||
            run.bestNodeIndexes.size() != requestCount ||
            replayInfo.schedInfos.size() != nodeCount) {
            return;
        }
        std::vector<int64_t> requestTokenCounts(requestCount);
        std::vector<double> requestWeights(requestCount);
        CostSchedulerKernelInternal::fillRequestTokenCounts(
            context, requestTokenCounts.data(), requestWeights.data(),
            replayInfo.latencyTokenNormalizationPower);
        const auto tokenNormalizationDenominator =
            CostSchedulerKernelInternal::
                calculateTokenNormalizationDenominator(
                    requestTokenCounts.data(), requestCount,
                    replayInfo.latencyTokenNormalizationPower);
        const auto matrixSize = requestCount * nodeCount;
        std::vector<float> costs(matrixSize * 2);
        std::vector<double> selectedWeights(requestCount * nodeCount, 0.0);
        std::vector<std::vector<size_t>> nodeRequests(nodeCount);
        for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
            if (replayInfo.schedInfos[nodeIndex].size() != requestCount) {
                return;
            }
            for (size_t requestIndex = 0; requestIndex < requestCount;
                 ++requestIndex) {
                costs[nodeIndex * requestCount + requestIndex] =
                    replayInfo.schedInfos[nodeIndex][requestIndex]
                        .schedInfo.cost;
                costs[matrixSize + nodeIndex * requestCount + requestIndex] =
                    replayInfo.schedInfos[nodeIndex][requestIndex]
                        .schedInfo.waitingCost;
            }
        }
        for (size_t requestIndex = 0; requestIndex < requestCount;
             ++requestIndex) {
            const auto nodeIndex = run.bestNodeIndexes[requestIndex];
            if (nodeIndex >= nodeCount) {
                return;
            }
            selectedWeights[nodeIndex * requestCount + requestIndex] = 1.0;
            nodeRequests[nodeIndex].emplace_back(requestIndex);
        }
        std::vector<double> executionTerms(nodeCount);
        std::vector<double> latencyTerms(nodeCount);
        const auto selectedObjective =
            CostSchedulerKernelInternal::forwardNodeRange(
                context, costs.data(), selectedWeights.data(),
                requestTokenCounts.data(),
                requestWeights.data(), tokenNormalizationDenominator,
                replayInfo.lambda, replayInfo.latencyWeightConcavity,
                0, nodeCount, nullptr, executionTerms.data(),
                latencyTerms.data());
        double executionNumerator = 0.0;
        double latencyNumerator = 0.0;
        double predictedCardTimeMs = 0.0;
        double predictedQueueRequestMs = 0.0;
        double predictedRequestExecutionMs = 0.0;
        for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
            executionNumerator += executionTerms[nodeIndex];
            latencyNumerator += latencyTerms[nodeIndex];
            if (nodeRequests[nodeIndex].empty()) {
                continue;
            }
            const auto batchExecutionMs =
                context.estimateBatchExecutionLatencyMs(
                    nodeIndex, nodeRequests[nodeIndex]);
            predictedCardTimeMs += batchExecutionMs;
            predictedRequestExecutionMs +=
                batchExecutionMs * nodeRequests[nodeIndex].size();
            predictedQueueRequestMs +=
                context.getNode(nodeIndex).getEstimatedQueueWaitMs() *
                nodeRequests[nodeIndex].size();
        }
        const auto executionContribution =
            replayInfo.lambda * executionNumerator /
            tokenNormalizationDenominator;
        const auto latencyContribution =
            (1.0 - replayInfo.lambda) * latencyNumerator / requestCount;
        std::cout << "  selected_objective=" << selectedObjective
                  << " execution_term_raw=" << executionNumerator
                  << " execution_term_weighted=" << executionContribution
                  << " latency_term_raw=" << latencyNumerator
                  << " latency_term_weighted=" << latencyContribution
                  << std::endl;
        std::cout << "  predicted_card_time_ms=" << predictedCardTimeMs
                  << " predicted_request_execution_sum_ms="
                  << predictedRequestExecutionMs
                  << " predicted_request_queue_sum_ms="
                  << predictedQueueRequestMs
                  << " predicted_request_total_mean_ms="
                  << (predictedRequestExecutionMs + predictedQueueRequestMs) /
                         requestCount
                  << std::endl;
    }

    static void printFrozenCostAudit(
        const NaviSchedScheduleContext &context,
        const autil::legacy::json::JsonMap &schedulerInfo)
    {
        CostSchedulerKernelInternal::ReplayInfo info;
        autil::legacy::FromJson(info, autil::legacy::Any(schedulerInfo));
        const auto requests = context.requestCount();
        const auto nodes = context.nodeCount();
        if (info.optimizationRuns.empty() || info.schedInfos.size() != nodes) {
            return;
        }
        std::vector<int64_t> tokens(requests);
        std::vector<double> inverseTokens(requests);
        CostSchedulerKernelInternal::fillRequestTokenCounts(
            context, tokens.data(), inverseTokens.data(),
            info.latencyTokenNormalizationPower);
        const auto denominator = CostSchedulerKernelInternal::calculateTokenNormalizationDenominator(
            tokens.data(), requests, info.latencyTokenNormalizationPower);
        std::vector<float> costs(2 * requests * nodes);
        if (!CostSchedulerKernelInternal::fillCostsForNodeRange(
                context, tokens.data(), 0, nodes, costs.data())) {
            return;
        }
        double maxCostDifference = 0.0;
        double maxReferenceCostDifference = 0.0;
        std::vector<int64_t> cacheHits;
        for (size_t node = 0; node < nodes; ++node) {
            if (info.schedInfos[node].size() != requests) {
                return;
            }
            if (!context.getCacheHitBlockCounts(node, cacheHits)) {
                return;
            }
            for (size_t request = 0; request < requests; ++request) {
                const auto &recorded = info.schedInfos[node][request].schedInfo;
                const auto index = node * requests + request;
                maxCostDifference = std::max(maxCostDifference,
                    std::abs(double(costs[index]) - recorded.cost));
                maxCostDifference = std::max(maxCostDifference,
                    std::abs(double(costs[requests * nodes + index]) - recorded.waitingCost));
                const auto &item = context.getRequest(request).request;
                const auto hitTokens = cacheHits[request] * item.blockSize;
                const auto &parameters = context.getNode(node).getLatencyParameters();
                const auto running = static_cast<float>(NonLinearPrefillModelR::calculateRequestLinearCost(
                    parameters, tokens[request], hitTokens));
                const auto waiting = static_cast<float>(NonLinearPrefillModelR::calculateWaitingRequestLinearCost(
                    parameters, tokens[request], hitTokens));
                maxReferenceCostDifference = std::max(maxReferenceCostDifference,
                    std::abs(double(costs[index]) - running));
                maxReferenceCostDifference = std::max(maxReferenceCostDifference,
                    std::abs(double(costs[requests * nodes + index]) - waiting));
            }
        }
        std::cout << "  frozen_cost_max_difference=" << maxCostDifference
                  << " reference_cost_max_difference=" << maxReferenceCostDifference << std::endl;
        const auto &run = info.optimizationRuns.front();
        for (size_t node = 0; node < nodes; ++node) {
            for (size_t count : {size_t(1), size_t(2), size_t(8), size_t(16),
                                 size_t(32), size_t(64), size_t(128), requests}) {
                if (count > requests) {
                    continue;
                }
                std::vector<size_t> assigned(count);
                for (size_t request = 0; request < count; ++request) {
                    assigned[request] = request;
                }
                std::cout << "  load_curve node=" << node
                          << " agent=" << context.getNode(node).getAgentId()
                          << " requests=" << count
                          << " predicted_execution_ms=" << std::setprecision(17)
                          << context.estimateBatchExecutionLatencyMs(node, assigned)
                          << std::setprecision(6) << std::endl;
            }
        }
        if (run.bestNodeIndexes.size() == requests) {
            std::vector<double> oneHot(nodes * requests, 0.0);
            bool valid = true;
            for (size_t request = 0; request < requests; ++request) {
                if (run.bestNodeIndexes[request] >= nodes) {
                    valid = false;
                    break;
                }
                oneHot[run.bestNodeIndexes[request] * requests + request] = 1.0;
            }
            if (valid) {
                CostSchedulerKernelInternal::DiscreteForwardWorkspace workspace;
                const auto discrete = CostSchedulerKernelInternal::forwardDiscrete(
                    context, costs.data(), run.bestNodeIndexes.data(), tokens.data(),
                    inverseTokens.data(), denominator, info.lambda,
                    info.latencyWeightConcavity, workspace);
                const auto reference = CostSchedulerKernelInternal::forwardNodeRange(
                    context, costs.data(), oneHot.data(), tokens.data(),
                    inverseTokens.data(), denominator, info.lambda,
                    info.latencyWeightConcavity, 0, nodes, nullptr, nullptr, nullptr);
                std::cout << "  frozen_discrete=" << std::setprecision(17) << discrete
                          << " reference_one_hot=" << reference
                          << " difference=" << discrete - reference
                          << std::setprecision(6) << std::endl;
            }
        }
        printSelectedCostComponents(context, info, run);
        printGradientAudit(context, info, info.optimizationRuns.front(), costs,
                           tokens, inverseTokens, denominator);
    }

    static void runOptimizerExperiment(
        const NaviSchedScheduleContext &context,
        const autil::legacy::json::JsonMap &schedulerInfo)
    {
        using Solver = CostSchedulerKernelInternal;
        Solver::ReplayInfo info;
        autil::legacy::FromJson(info, autil::legacy::Any(schedulerInfo));
        const auto requests = context.requestCount();
        const auto nodes = context.nodeCount();
        const auto count = requests * nodes;
        if (info.optimizationRuns.empty() || requests == 0 || nodes == 0 ||
            info.optimizationRuns.front().initialWeights.size() != count) {
            throw std::runtime_error("optimizer experiment requires recorded initial weights");
        }
        std::vector<int64_t> tokens(requests);
        std::vector<double> inverseTokens(requests);
        Solver::fillRequestTokenCounts(context, tokens.data(), inverseTokens.data(),
                                      info.latencyTokenNormalizationPower);
        const auto denominator = Solver::calculateTokenNormalizationDenominator(
            tokens.data(), requests, info.latencyTokenNormalizationPower);
        std::vector<float> costs(count * 2);
        if (!Solver::fillCostsForNodeRange(context, tokens.data(), 0, nodes, costs.data())) {
            throw std::runtime_error("optimizer experiment cost preparation failed");
        }
        for (const std::string mode : {"pg"}) {
            const std::vector<double> rates = mode == "pg"
                ? std::vector<double>{0.0002, 0.00015, 0.0001, 0.00008, 0.00006,
                                      0.00005, 0.00004, 0.00003, 0.000025, 0.00002,
                                      0.000015, 0.00001, 0.000008, 0.000006, 0.000005,
                                      0.000004, 0.000003, 0.000002, 0.000001,
                                      0.0000003, 0.0000001}
                : mode == "pg_backtrack" ? std::vector<double>{512.0}
                : std::vector<double>{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0};
            for (const auto loops : {size_t{5}, size_t{10}, size_t{20}, size_t{50},
                                     size_t{100}, size_t{200}, size_t{500}, size_t{1000}}) {
                if (mode != "pg" && loops != 10 && loops != 50) {
                    continue;
                }
                for (const auto rate : rates) {
                    auto weights = info.optimizationRuns.front().initialWeights;
                    std::vector<double> gradients(count), firstMoment(count, 0.0),
                        secondMoment(count, 0.0), projection(nodes * Solver::PROJECTION_REQUEST_TILE_SIZE);
                    std::vector<Solver::NodeGradient> nodeGradients(nodes);
                    std::vector<uint32_t> indexes(requests), bestIndexes;
                    Solver::DiscreteForwardWorkspace workspace;
                    auto bestDiscrete = std::numeric_limits<double>::infinity();
                    auto bestContinuous = std::numeric_limits<double>::infinity();
                    auto previousContinuous = std::numeric_limits<double>::infinity();
                    size_t increases = 0;
                    size_t lineEvaluations = 0;
                    double acceptedRate = rate;
                    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
                    for (size_t iteration = 0; iteration <= loops; ++iteration) {
                        const auto continuous = Solver::forwardNodeRange(
                            context, costs.data(), weights.data(), tokens.data(), inverseTokens.data(),
                            denominator, info.lambda, info.latencyWeightConcavity, 0, nodes,
                            nodeGradients.data(), nullptr, nullptr);
                        if (!std::isfinite(continuous) || !Solver::selectWeightRequestRange(
                                weights.data(), requests, nodes, 0, requests, indexes.data(),
                                info.initialWeightSeed)) {
                            throw std::runtime_error("optimizer experiment invalid iterate");
                        }
                        const auto discrete = Solver::forwardDiscrete(
                            context, costs.data(), indexes.data(), tokens.data(), inverseTokens.data(),
                            denominator, info.lambda, info.latencyWeightConcavity, workspace);
                        if (!std::isfinite(discrete)) {
                            throw std::runtime_error("optimizer experiment invalid discrete loss");
                        }
                        if (discrete < bestDiscrete) {
                            bestDiscrete = discrete;
                            bestIndexes = indexes;
                        }
                        bestContinuous = std::min(bestContinuous, continuous);
                        increases += continuous > previousContinuous + 1e-9;
                        previousContinuous = continuous;
                        if (iteration == loops) {
                            break;
                        }
                        Solver::backwardNodeRange(
                            context, costs.data(), weights.data(), tokens.data(), inverseTokens.data(),
                            requests, 0, nodes, nodeGradients.data(), gradients.data());
                        // Remove only the common component on each request's simplex.
                        if (mode == "adam_centered") {
                            for (size_t request = 0; request < requests; ++request) {
                                double mean = 0.0;
                                for (size_t node = 0; node < nodes; ++node) {
                                    mean += gradients[node * requests + request] / nodes;
                                }
                                for (size_t node = 0; node < nodes; ++node) {
                                    gradients[node * requests + request] -= mean;
                                }
                            }
                        }
                        const auto firstCorrection = 1.0 - std::pow(0.9, iteration + 1);
                        const auto secondCorrection = 1.0 - std::pow(0.999, iteration + 1);
                        for (size_t index = 0; index < count; ++index) {
                            auto direction = gradients[index];
                            if (mode == "adam" || mode == "adam_centered") {
                                firstMoment[index] = 0.9 * firstMoment[index] + 0.1 * direction;
                                secondMoment[index] = 0.999 * secondMoment[index] + 0.001 * direction * direction;
                                direction = (firstMoment[index] / firstCorrection) /
                                    (std::sqrt(secondMoment[index] / secondCorrection) + 1e-8);
                            }
                            gradients[index] = direction;
                        }
                        const auto previousWeights = weights;
                        auto stepRate = mode == "pg_backtrack"
                            ? std::min(rate, acceptedRate * 2.0) : rate;
                        bool accepted = false;
                        for (size_t trial = 0; trial < (mode == "pg_backtrack" ? 40 : 1); ++trial) {
                            for (size_t index = 0; index < count; ++index) {
                                weights[index] = previousWeights[index] - stepRate * gradients[index];
                            }
                            if (!Solver::projectRequestRange(weights.data(), requests, nodes, 0, requests,
                                                             projection.data())) {
                                throw std::runtime_error("optimizer experiment projection failed");
                            }
                            if (mode != "pg_backtrack") {
                                accepted = true;
                                break;
                            }
                            double directionalDerivative = 0.0;
                            for (size_t index = 0; index < count; ++index) {
                                directionalDerivative += gradients[index] * (weights[index] - previousWeights[index]);
                            }
                            const auto candidate = Solver::forwardNodeRange(
                                context, costs.data(), weights.data(), tokens.data(), inverseTokens.data(),
                                denominator, info.lambda, info.latencyWeightConcavity, 0, nodes,
                                nullptr, nullptr, nullptr);
                            ++lineEvaluations;
                            if (std::isfinite(candidate) && candidate <= continuous + 1e-4 * directionalDerivative) {
                                accepted = true;
                                acceptedRate = stepRate;
                                break;
                            }
                            stepRate *= 0.5;
                        }
                        if (!accepted) {
                            weights = previousWeights;
                        }
                        for (size_t request = 0; request < requests; ++request) {
                            double sum = 0.0;
                            for (size_t node = 0; node < nodes; ++node) {
                                const auto weight = weights[node * requests + request];
                                if (!std::isfinite(weight) || weight < -1e-10) {
                                    throw std::runtime_error("optimizer experiment infeasible weight");
                                }
                                sum += weight;
                            }
                            if (std::abs(sum - 1.0) > 1e-8) {
                                throw std::runtime_error("optimizer experiment invalid simplex sum");
                            }
                        }
                    }
                    const auto optimizeUs = autil::TimeUtility::monotonicTimeUs() - beginUs;
                    const auto beforeRefinement = bestDiscrete;
                    Solver::DiscreteRefinementStats stats;
                    if (!Solver::refineDiscreteAssignment(
                            context, costs.data(), bestIndexes.data(), tokens.data(), inverseTokens.data(),
                            denominator, info.lambda, info.latencyWeightConcavity, bestDiscrete, -1,
                            info.optimizationRuns.front().refinementEvaluations, stats)) {
                        throw std::runtime_error("optimizer experiment refinement failed");
                    }
                    std::vector<size_t> sizes(nodes, 0);
                    for (const auto node : bestIndexes) {
                        ++sizes[node];
                    }
                    std::cout << "optimizer_experiment mode=" << mode << " alpha=" << rate
                              << " loops=" << loops << " best_continuous=" << bestContinuous
                              << " raw_discrete=" << beforeRefinement << " refined_discrete=" << bestDiscrete
                              << " max_batch=" << *std::max_element(sizes.begin(), sizes.end())
                              << " used_nodes=" << std::count_if(sizes.begin(), sizes.end(), [](size_t n) { return n > 0; })
                              << " increases=" << increases << " optimize_us=" << optimizeUs
                              << " line_evaluations=" << lineEvaluations
                              << " refinement_evaluations=" << stats.evaluations
                              << " total_us=" << autil::TimeUtility::monotonicTimeUs() - beginUs
                              << std::endl;
                }
            }
        }
    }

    static void printGradientAudit(
        const NaviSchedScheduleContext &context,
        const CostSchedulerKernelInternal::ReplayInfo &info,
        const CostSchedulerKernelInternal::ReplayInfo::OptimizationRun &run,
        const std::vector<float> &costs, const std::vector<int64_t> &tokens,
        const std::vector<double> &inverseTokens, double denominator)
    {
        const auto requests = context.requestCount();
        const auto nodes = context.nodeCount();
        if (run.initialWeights.size() != requests * nodes) {
            return;
        }
        auto weights = run.initialWeights;
        std::vector<CostSchedulerKernelInternal::NodeGradient> nodeGradients(nodes);
        std::vector<double> gradients(requests * nodes);
        const auto evaluate = [&](size_t node, bool backward) {
            const auto value = CostSchedulerKernelInternal::forwardNodeRange(
                context, costs.data(), weights.data(), tokens.data(),
                inverseTokens.data(), denominator, info.lambda,
                info.latencyWeightConcavity, node, node + 1,
                backward ? &nodeGradients[node] : nullptr, nullptr, nullptr);
            if (backward) {
                CostSchedulerKernelInternal::backwardNodeRange(
                    context, costs.data(), weights.data(), tokens.data(),
                    inverseTokens.data(), requests, node, node + 1,
                    &nodeGradients[node], gradients.data() + node * requests);
            }
            return value;
        };
        constexpr double EPSILON = 1e-6;
        for (size_t node = 0; node < nodes; ++node) {
            for (bool zero : {false, true}) {
                if (zero) {
                    std::fill(weights.begin() + node * requests,
                              weights.begin() + (node + 1) * requests, 0.0);
                }
                const auto baseline = evaluate(node, true);
                for (size_t request : {size_t(0), requests / 2, requests - 1}) {
                    const auto index = node * requests + request;
                    const auto saved = weights[index];
                    weights[index] = saved + EPSILON;
                    const auto plus = evaluate(node, false);
                    weights[index] = saved;
                    double numeric = (plus - baseline) / EPSILON;
                    if (saved > EPSILON) {
                        weights[index] = saved - EPSILON;
                        const auto minus = evaluate(node, false);
                        weights[index] = saved;
                        numeric = (plus - minus) / (2.0 * EPSILON);
                    }
                    std::cout << "  gradient_audit node=" << node
                              << " agent=" << context.getNode(node).getAgentId()
                              << " zero=" << zero << " request=" << request
                              << " value=" << std::setprecision(17) << baseline
                              << " analytic=" << gradients[index]
                              << " numeric=" << numeric
                              << " difference=" << numeric - gradients[index]
                              << std::setprecision(6) << std::endl;
                }
            }
            std::copy(run.initialWeights.begin() + node * requests,
                      run.initialWeights.begin() + (node + 1) * requests,
                      weights.begin() + node * requests);
        }
    }

    static std::string makeCostNodeLabel(
        const NaviSchedScheduleContext &context, size_t nodeIndex)
    {
        std::ostringstream output;
        output << "node[" << nodeIndex << "] "
               << context.getNode(nodeIndex).getAgentId();
        return output.str();
    }

    static std::string makeCostRequestLabel(
        const NaviSchedScheduleContext &context, size_t requestIndex)
    {
        std::ostringstream output;
        output << "request[" << requestIndex << "] id="
               << context.getRequest(requestIndex).request.requestId;
        return output.str();
    }

    static std::string makeFixedValue(double value, int precision)
    {
        std::ostringstream output;
        output << std::fixed << std::setprecision(precision) << value;
        return output.str();
    }

    static std::string makeCostCell(
        const CostSchedulerKernelInternal::SchedInfo &schedInfo,
        size_t costWidth, size_t queueWidth, size_t weightWidth)
    {
        std::ostringstream output;
        output << std::right
               << "c=" << std::setw(costWidth)
               << makeFixedValue(schedInfo.cost, 2)
               << " q=" << std::setw(queueWidth)
               << makeFixedValue(schedInfo.queueWaitMs, 2)
               << " w=" << std::setw(weightWidth)
               << makeFixedValue(schedInfo.weight, 5);
        return output.str();
    }

    static void printCostSchedulerTable(
        const NaviSchedScheduleContext &context,
        const CostSchedulerKernelInternal::ReplayInfo &replayInfo)
    {
        constexpr size_t REQUESTS_PER_PANEL = 4;
        size_t nodeWidth = std::string("node").size();
        size_t costWidth = 1;
        size_t queueWidth = 1;
        size_t weightWidth = 1;
        for (size_t nodeIndex = 0; nodeIndex < context.nodeCount(); ++nodeIndex) {
            auto width = makeCostNodeLabel(context, nodeIndex).size();
            if (width > nodeWidth) {
                nodeWidth = width;
            }
            for (const auto &value : replayInfo.schedInfos[nodeIndex]) {
                const auto &schedInfo = value.schedInfo;
                width = makeFixedValue(schedInfo.cost, 2).size();
                if (width > costWidth) {
                    costWidth = width;
                }
                width = makeFixedValue(schedInfo.queueWaitMs, 2).size();
                if (width > queueWidth) {
                    queueWidth = width;
                }
                width = makeFixedValue(schedInfo.weight, 5).size();
                if (width > weightWidth) {
                    weightWidth = width;
                }
            }
        }
        auto valueWidth = costWidth + queueWidth + weightWidth + 8;
        for (size_t begin = 0; begin < context.requestCount();
             begin += REQUESTS_PER_PANEL) {
            auto end = begin + REQUESTS_PER_PANEL;
            if (end > context.requestCount()) {
                end = context.requestCount();
            }
            auto cellWidth = valueWidth;
            for (size_t requestIndex = begin; requestIndex < end;
                 ++requestIndex) {
                auto width = makeCostRequestLabel(context, requestIndex).size();
                if (width > cellWidth) {
                    cellWidth = width;
                }
            }
            std::cout << "  requests[" << begin << ".." << end - 1
                      << "]:" << std::endl;
            std::cout << "  " << std::left << std::setw(nodeWidth) << "node";
            for (size_t requestIndex = begin; requestIndex < end;
                 ++requestIndex) {
                std::cout << " | " << std::setw(cellWidth)
                          << makeCostRequestLabel(context, requestIndex);
            }
            std::cout << std::endl;
            for (size_t nodeIndex = 0; nodeIndex < context.nodeCount();
                 ++nodeIndex) {
                std::cout << "  " << std::left << std::setw(nodeWidth)
                          << makeCostNodeLabel(context, nodeIndex);
                for (size_t requestIndex = begin; requestIndex < end;
                     ++requestIndex) {
                    const auto &schedInfo =
                        replayInfo.schedInfos[nodeIndex][requestIndex].schedInfo;
                    std::cout << " | " << std::setw(cellWidth)
                              << makeCostCell(schedInfo, costWidth, queueWidth,
                                              weightWidth);
                }
                std::cout << std::endl;
            }
        }
    }

    static void add(const NaviSchedScheduleReplay &replay,
                    const std::vector<NaviSchedDecision> &decisions, bool matchesRecorded,
                    const NaviSchedScheduleContext &context, Statistics &statistics)
    {
        ++statistics.replayCount;
        if (matchesRecorded) {
            ++statistics.recordedMatchCount;
        }
        if (decisions.size() != replay.requests.size() ||
            replay.nodeMatches.size() != replay.requests.size()) {
            ++statistics.replayFailureCount;
            return;
        }
        statistics.requestCount += decisions.size();
        std::vector<std::vector<size_t>> nodeRequestIndexes(replay.nodes.size());
        for (size_t requestIndex = 0; requestIndex < decisions.size(); ++requestIndex) {
            const auto &decision = decisions[requestIndex];
            if (decision.target.address.empty() && decision.target.agentAddress.empty()) {
                ++statistics.rejectedCount;
                continue;
            }
            auto nodeIndex = findTargetNode(replay, decision.target);
            if (nodeIndex >= replay.nodes.size() ||
                replay.nodeMatches[requestIndex].size() != replay.nodes.size()) {
                ++statistics.invalidTargetCount;
                continue;
            }
            ++statistics.scheduledCount;
            const auto &match = replay.nodeMatches[requestIndex][nodeIndex];
            statistics.matchedBlockCount += match.matchedBlockCount;
            const auto &request = replay.requests[requestIndex].request;
            statistics.matchedTokenCount +=
                match.matchedBlockCount * request.blockSize;
            statistics.queueMs += context.getNode(nodeIndex).getEstimatedQueueWaitMs();
            nodeRequestIndexes[nodeIndex].emplace_back(requestIndex);
        }
        for (size_t nodeIndex = 0; nodeIndex < nodeRequestIndexes.size(); ++nodeIndex) {
            const auto &requestIndexes = nodeRequestIndexes[nodeIndex];
            if (requestIndexes.empty()) {
                continue;
            }
            ++statistics.scheduledBatchCount;
            statistics.batchExecutionLatencyMs +=
                context.estimateBatchExecutionLatencyMs(nodeIndex, requestIndexes);
        }
    }

    static size_t findTargetNode(const NaviSchedScheduleReplay &replay,
                                 const NaviSchedTarget &target)
    {
        for (size_t nodeIndex = 0; nodeIndex < replay.nodes.size(); ++nodeIndex) {
            const auto &node = replay.nodes[nodeIndex];
            if ((!target.agentAddress.empty() &&
                 target.agentAddress == node.collectorAddress) ||
                (!target.address.empty() && target.address == node.serverAddress)) {
                return nodeIndex;
            }
        }
        return replay.nodes.size();
    }

    static void print(const Statistics &statistics)
    {
        auto scheduledCount = static_cast<double>(statistics.scheduledCount);
        std::cout << std::fixed << std::setprecision(3) << "scheduler=" << statistics.name
                  << " replays=" << statistics.replayCount
                  << " replay_failures=" << statistics.replayFailureCount
                  << " recorded_matches=" << statistics.recordedMatchCount
                  << " requests=" << statistics.requestCount
                  << " scheduled=" << statistics.scheduledCount
                  << " rejected=" << statistics.rejectedCount
                  << " invalid_targets=" << statistics.invalidTargetCount
                  << " schedule_elapsed_us=" << statistics.scheduleElapsedUs
                  << " scheduled_batches=" << statistics.scheduledBatchCount
                  << " avg_cache_hit_blocks="
                  << (scheduledCount > 0.0 ? statistics.matchedBlockCount / scheduledCount
                                           : 0.0)
                  << " avg_cache_hit_tokens="
                  << (scheduledCount > 0.0 ? statistics.matchedTokenCount / scheduledCount
                                           : 0.0)
                  << " avg_estimated_queue_ms="
                  << (scheduledCount > 0.0 ? statistics.queueMs / scheduledCount : 0.0)
                  << " avg_estimated_batch_execution_ms="
                  << (statistics.scheduledBatchCount > 0
                          ? statistics.batchExecutionLatencyMs /
                                statistics.scheduledBatchCount
                          : 0.0)
                  << std::endl;
    }
};

} // namespace navi_sched

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " SCHEDULER_KERNEL REPLAY_JSON_OR_JSONL"
                  << " [--context-summary] [--require-recorded-match]"
                  << " [--optimizer-experiment]"
                  << " [--replay-output PATH]"
                  << " [--cost-lambda VALUE] [--cost-alpha VALUE]"
                  << " [--cost-alpha-decay VALUE] [--cost-min-alpha VALUE]"
                  << " [--cost-max-loop-count COUNT]"
                  << " [--cost-latency-weight-concavity VALUE]"
                  << " [--cost-latency-token-normalization-power VALUE]"
                  << std::endl;
        return 2;
    }
    navi_sched::ScheduleReplayCompare::Options options;
    for (int index = 3; index < argc; ++index) {
        const std::string option = argv[index];
        if ("--context-summary" == option) {
            options.printSummary = true;
        } else if ("--optimizer-experiment" == option) {
            options.optimizerExperiment = true;
        } else if ("--require-recorded-match" == option) {
            options.requireRecordedMatch = true;
        } else if ("--replay-output" == option && index + 1 < argc) {
            options.replayOutputPath = argv[++index];
        } else if ("--cost-lambda" == option && index + 1 < argc) {
            if (!autil::StringUtil::fromString(
                    std::string(argv[++index]), options.costLambda)) {
                std::cerr << "invalid cost lambda" << std::endl;
                return 2;
            }
        } else if ("--cost-alpha" == option && index + 1 < argc) {
            if (!autil::StringUtil::fromString(
                    std::string(argv[++index]), options.costAlpha)) {
                std::cerr << "invalid cost alpha" << std::endl;
                return 2;
            }
        } else if ("--cost-alpha-decay" == option && index + 1 < argc) {
            if (!autil::StringUtil::fromString(
                    std::string(argv[++index]), options.costAlphaDecay)) {
                std::cerr << "invalid cost alpha decay" << std::endl;
                return 2;
            }
        } else if ("--cost-min-alpha" == option && index + 1 < argc) {
            if (!autil::StringUtil::fromString(
                    std::string(argv[++index]), options.costMinAlpha)) {
                std::cerr << "invalid cost minimum alpha" << std::endl;
                return 2;
            }
        } else if ("--cost-max-loop-count" == option &&
                   index + 1 < argc) {
            if (!autil::StringUtil::fromString(
                    std::string(argv[++index]),
                    options.costMaxLoopCount)) {
                std::cerr << "invalid cost max loop count" << std::endl;
                return 2;
            }
        } else if ("--cost-latency-weight-concavity" == option &&
                   index + 1 < argc) {
            if (!autil::StringUtil::fromString(
                    std::string(argv[++index]),
                    options.costLatencyWeightConcavity) ||
                !std::isfinite(options.costLatencyWeightConcavity) ||
                options.costLatencyWeightConcavity < 0.0) {
                std::cerr << "invalid cost latency weight concavity"
                          << std::endl;
                return 2;
            }
        } else if ("--cost-latency-token-normalization-power" == option &&
                   index + 1 < argc) {
            if (!autil::StringUtil::fromString(
                    std::string(argv[++index]),
                    options.costLatencyTokenNormalizationPower) ||
                !std::isfinite(
                    options.costLatencyTokenNormalizationPower) ||
                options.costLatencyTokenNormalizationPower < 0.0 ||
                options.costLatencyTokenNormalizationPower > 1.0) {
                std::cerr << "invalid cost latency token normalization power"
                          << std::endl;
                return 2;
            }
        } else {
            std::cerr << "invalid option: " << option << std::endl;
            return 2;
        }
    }
    return navi_sched::ScheduleReplayCompare::run(
        argv[1], argv[2], options);
}
