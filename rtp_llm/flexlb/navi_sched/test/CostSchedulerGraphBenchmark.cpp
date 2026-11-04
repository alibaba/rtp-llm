#include "rtp_llm/flexlb/navi_sched/data/AgentLatencyToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"
#include "rtp_llm/flexlb/navi_sched/resource/SchedulerConfigR.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerGraph.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/MasterScheduleContextData.h"
#include "rtp_llm/flexlb/navi_sched/test/ScheduleReplayRunner.h"
#include "autil/EnvUtil.h"
#include "autil/HashAlgorithm.h"
#include "autil/Lock.h"
#include "autil/StringUtil.h"
#include "autil/TimeUtility.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/Navi.h"
#include "navi/test_cluster/NaviTestCluster.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace navi_sched {

class CostBenchmarkCoordinator {
public:
    static CostBenchmarkCoordinator &instance()
    {
        static CostBenchmarkCoordinator coordinator;
        return coordinator;
    }

    void setPipe(const navi::AsyncPipePtr &pipe)
    {
        autil::ScopedLock lock(_condition);
        _pipe = pipe;
        _condition.broadcast();
    }

    navi::AsyncPipePtr waitPipe()
    {
        autil::ScopedLock lock(_condition);
        while (!_pipe) {
            if (_condition.wait(30'000'000) != 0) return nullptr;
        }
        return _pipe;
    }

    void finish(const navi::DataPtr &data)
    {
        autil::ScopedLock lock(_condition);
        _lastData = data;
        ++_finishedCount;
        _condition.broadcast();
    }

    bool waitFinish(size_t expectedCount)
    {
        autil::ScopedLock lock(_condition);
        while (_finishedCount < expectedCount) {
            if (_condition.wait(30'000'000) != 0) return false;
        }
        auto context = MasterScheduleContextData::getContext(_lastData);
        return bool(context);
    }

private:
    autil::ThreadCond _condition;
    navi::AsyncPipePtr _pipe;
    navi::DataPtr _lastData;
    size_t _finishedCount = 0;
};

class CostBenchmarkSourceKernel final : public navi::Kernel {
public:
    static const std::string KERNEL_ID;

    void def(navi::KernelDefBuilder &builder) const override
    {
        builder.name(KERNEL_ID)
            .output("context", MASTER_SCHEDULE_CONTEXT_TYPE_ID);
    }

    navi::ErrorCode init(navi::KernelInitContext &ctx) override
    {
        _pipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
        if (!_pipe) {
            return navi::EC_INIT_SUB_GRAPH;
        }
        CostBenchmarkCoordinator::instance().setPipe(_pipe);
        return navi::EC_NONE;
    }

    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override
    {
        navi::DataPtr data;
        bool eof = false;
        if (!_pipe->getData(data, eof)) {
            return navi::EC_NONE;
        }
        if (!ctx.setOutput(
                navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
            return navi::EC_ABORT;
        }
        return navi::EC_NONE;
    }

private:
    navi::AsyncPipePtr _pipe;
};

const std::string CostBenchmarkSourceKernel::KERNEL_ID =
    "navi_sched.cost_benchmark_source_k";

class CostBenchmarkSinkKernel final : public navi::Kernel {
public:
    static const std::string KERNEL_ID;

    void def(navi::KernelDefBuilder &builder) const override
    {
        builder.name(KERNEL_ID)
            .input("result", MASTER_SCHEDULE_CONTEXT_TYPE_ID)
            .output("result", MASTER_SCHEDULE_CONTEXT_TYPE_ID);
    }

    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override
    {
        navi::DataPtr data;
        bool eof = false;
        if (!ctx.getInput(
                navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
            return navi::EC_NONE;
        }
        if (data) {
            CostBenchmarkCoordinator::instance().finish(data);
        }
        if (!ctx.setOutput(
                navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
            return navi::EC_ABORT;
        }
        return navi::EC_NONE;
    }
};

const std::string CostBenchmarkSinkKernel::KERNEL_ID =
    "navi_sched.cost_benchmark_sink_k";

class CostSchedulerGraphBenchmark {
public:
    static int prepareReplay(const std::string &path, size_t workers, size_t repeats)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input || input.tellg() > 768 * 1024 * 1024 ||
            workers == 0 || workers > 256 || repeats == 0 || repeats > 100) {
            return 2;
        }
        input.seekg(0);
        const std::string json{std::istreambuf_iterator<char>(input), {}};
        NaviSchedScheduleReplay replay;
        NaviSchedScheduleReplayJson::fromJsonString(json, replay);
        auto context = makeContext(replay);
        if (!context) {
            return 1;
        }
        std::vector<int64_t> tokens(context->requestCount());
        CostSchedulerKernelInternal::fillRequestTokenCounts(*context, tokens.data(), nullptr);
        const auto matrixSize = context->requestCount() * context->nodeCount();
        std::vector<float> baselineCosts(matrixSize * 2), timedCosts(matrixSize * 2);
        for (size_t trial = 0; trial < 3; ++trial) {
            const auto begin = autil::TimeUtility::monotonicTimeUs();
            if (!CostSchedulerKernelInternal::fillCostsForNodeRange(
                    *context, tokens.data(), 0, context->nodeCount(), baselineCosts.data())) {
                return 1;
            }
            const auto baselineEnd = autil::TimeUtility::monotonicTimeUs();
            CostSchedulerKernelInternal::PreparationTiming timing;
            if (!CostSchedulerKernelInternal::fillCostsForNodeRange(
                    *context, tokens.data(), 0, context->nodeCount(), timedCosts.data(), &timing)) {
                return 1;
            }
            const auto timedEnd = autil::TimeUtility::monotonicTimeUs();
            if (baselineCosts != timedCosts || timing.completedNodes != context->nodeCount()) {
                return 1;
            }
            // The independent feature-based evaluator is the numeric oracle;
            // the two timed runs alone would also agree on a shared regression.
            for (size_t nodeIndex = 0; nodeIndex < context->nodeCount(); ++nodeIndex) {
                std::vector<int64_t> matches;
                if (!context->getCacheHitBlockCounts(nodeIndex, matches)) return 1;
                const auto &parameters = context->getNode(nodeIndex).getLatencyParameters();
                for (size_t requestIndex = 0; requestIndex < context->requestCount(); ++requestIndex) {
                    const auto hit = matches[requestIndex] * context->getRequest(requestIndex).request.blockSize;
                    const auto offset = nodeIndex * context->requestCount() + requestIndex;
                    if (baselineCosts[offset] != static_cast<float>(
                            NonLinearPrefillModelR::calculateRequestLinearCost(
                                parameters, tokens[requestIndex], hit)) ||
                        baselineCosts[offset + matrixSize] != static_cast<float>(
                            NonLinearPrefillModelR::calculateWaitingRequestLinearCost(
                                parameters, tokens[requestIndex], hit))) return 1;
                }
            }
            std::cout << "sequential_cost_trial=" << trial
                      << " baseline_us=" << baselineEnd - begin
                      << " timed_us=" << timedEnd - baselineEnd
                      << " match_us=" << timing.matchUs
                      << " cost_eval_us=" << timing.costEvaluationUs
                      << " completed_nodes=" << timing.completedNodes
                      << " cost_mismatches=0" << std::endl;
        }
        std::mt19937_64 generator(12345);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        std::vector<double> randomWeights(1UL << 22);
        for (auto &value : randomWeights) value = distribution(generator);
        std::vector<int64_t> times;
        CostSchedulerConfig config;
        config.maxLoopCount = 1;
        // Replay restoration is outside the timed region. Preparation queries
        // restored live caches; no replay collection or optimization is timed.
        for (size_t run = 0; run <= repeats; ++run) {
            const auto begin = autil::TimeUtility::monotonicTimeUs();
            auto state = CostSchedulerGraphState::create(
                context, config, workers, randomWeights.data(), randomWeights.size());
            if (!state) return 1;
            const auto initialized = autil::TimeUtility::monotonicTimeUs();
            if (workers == 1) {
                if (!state->prepareWorker(0)) return 1;
            } else {
                std::vector<std::thread> threads;
                try {
                    for (size_t worker = 0; worker < workers; ++worker) {
                        threads.emplace_back([state, worker]() { state->prepareWorker(worker); });
                    }
                } catch (...) {
                    for (auto &thread : threads) thread.join();
                    throw;
                }
                for (auto &thread : threads) thread.join();
            }
            if (!state->finishPreparation()) return 1;
            const auto end = autil::TimeUtility::monotonicTimeUs();
            if (run == 0 &&
                (!CostSchedulerKernelInternal::fillCostsForNodeRange(
                    *context, tokens.data(), 0, context->nodeCount(), timedCosts.data()) ||
                 baselineCosts != timedCosts)) return 1;
            for (size_t request = 0; request < context->requestCount(); ++request) {
                double sum = 0.0;
                for (size_t node = 0; node < context->nodeCount(); ++node) {
                    sum += state->weights()[node * context->requestCount() + request];
                }
                if (std::abs(sum - 1.0) > 1e-12) return 1;
            }
            if (run > 0) {
                times.push_back(end - begin);
                std::cout << "prepare_replay_run=" << run
                          << " requests=" << context->requestCount()
                          << " nodes=" << context->nodeCount()
                          << " workers=" << workers
                          << " init_us=" << initialized - begin
                          << " workers_and_join_us=" << end - initialized
                          << " total_us=" << end - begin << std::endl;
            }
        }
        std::sort(times.begin(), times.end());
        std::cout << "prepare_replay_requests=" << context->requestCount()
                  << " nodes=" << context->nodeCount() << " workers=" << workers
                  << " repeats=" << repeats << " min_us=" << times.front()
                  << " median_us=" << times[times.size() / 2]
                  << " max_us=" << times.back() << std::endl;
        return 0;
    }

    static int graphReplay(const std::string &path, size_t loops, size_t workers,
                           size_t repeats, int64_t budgetUs)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input || input.tellg() > 768 * 1024 * 1024 ||
            loops == 0 || loops > CostSchedulerConfig::MAX_LOOP_COUNT ||
            workers == 0 || workers > 256 || repeats == 0 || repeats > 100 ||
            budgetUs < 8000 || budgetUs % 800 != 0) {
            return 2;
        }
        input.seekg(0);
        const std::string json{std::istreambuf_iterator<char>(input), {}};
        NaviSchedScheduleReplay replay;
        NaviSchedScheduleReplayJson::fromJsonString(json, replay);
        return run(replay.requests.size(), replay.nodes.size(), loops, workers,
                   repeats, budgetUs, &replay);
    }

    static int run(size_t requestCount, size_t nodeCount,
                   size_t loopCount, size_t parallelism,
                   size_t repeatCount, int64_t optimizationTimeBudgetUs,
                   const NaviSchedScheduleReplay *recorded = nullptr)
    {
        autil::EnvGuard pythonHome(
            "NAVI_PYTHON_HOME",
            "./aios/navi/config_loader/python:"
            "/opt/conda310/lib/python3.10");
        autil::EnvGuard threadNum(
            "NAVI_SCHED_TEST_THREAD_NUM",
            std::to_string(parallelism));
        navi::NaviTestCluster cluster;
        auto resources = std::make_shared<navi::ResourceMap>();
        if (!cluster.addServer(
                "host_0", "./aios/navi_sched/testdata/"
                "test_config_loader.py",
                "./aios/navi_sched/testdata/config", resources) ||
            !cluster.addBiz(
                "host_0", "master_biz", 1, 0,
                "./aios/navi_sched/testdata/config/master_biz.py") ||
            !cluster.start()) {
            return 1;
        }
        auto navi = cluster.getNavi("host_0");
        navi::ResourceMap schedulerResources;
        if (recorded) {
            if (!navi || !navi->createResource("master_biz", 1, 0,
                    {SchedulerConfigR::RESOURCE_ID}, schedulerResources)) {
                return 1;
            }
            auto *resource = schedulerResources.getResource<SchedulerConfigR>(
                SchedulerConfigR::RESOURCE_ID);
            if (!resource) return 1;
            CostSchedulerConfig config;
            autil::legacy::FromJsonString(config,
                autil::legacy::ToJsonString(recorded->schedulerInfo, true));
            config.maxLoopCount = loopCount;
            config.parallelism = parallelism;
            config.optimizationTimeBudgetUs = optimizationTimeBudgetUs;
            config.initialWeightSeed = 12345;
            auto patch = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                autil::legacy::ToJson(config));
            patch["max_wait_schedule_time_ms"] = optimizationTimeBudgetUs / 800;
            SchedulerConfigR::Request request;
            request.update = true;
            request.configJson = autil::legacy::ToJsonString(patch, true);
            SchedulerConfigResponseDataPtr response;
            bool waitingForAgents = false;
            if (!resource->processRequest(request, response, waitingForAgents) ||
                !response || !response->accepted || waitingForAgents ||
                autil::legacy::ToJsonString(resource->get().config.cost, true) !=
                    autil::legacy::ToJsonString(config, true)) {
                return 1;
            }
            std::cout << "graph_replay_effective_cost_config="
                      << autil::legacy::ToJsonString(resource->get().config.cost, true)
                      << std::endl;
        }
        auto graphDef = buildGraph(
            loopCount, parallelism, optimizationTimeBudgetUs);
        if (!navi || !graphDef) {
            return 1;
        }
        navi::RunGraphParams params;
        params.setTimeoutMs(60 * 60 * 1000);
        auto userResult = navi->runLocalGraph(
            graphDef.release(), params, {});
        if (!userResult) {
            return 1;
        }
        auto pipe = CostBenchmarkCoordinator::instance().waitPipe();
        if (!pipe) {
            std::cerr << "benchmark source pipe did not become ready" << std::endl;
            navi->stop();
            return 1;
        }
        struct PipeEofGuard {
            navi::AsyncPipePtr pipe;
            ~PipeEofGuard() { if (pipe) pipe->setEof(); }
        } pipeEofGuard{pipe};
        auto replay = recorded ? NaviSchedScheduleReplay() : makeReplay(requestCount, nodeCount);
        std::vector<int64_t> elapsedUs;
        elapsedUs.reserve(repeatCount);
        for (size_t runIndex = 0; runIndex <= repeatCount; ++runIndex) {
            auto context = makeContext(recorded ? *recorded : replay);
            if (!context) {
                return 1;
            }
            auto beginUs = autil::TimeUtility::monotonicTimeUs();
            if (navi::EC_NONE != pipe->setData(
                    std::make_shared<MasterScheduleContextData>(context))) {
                return 1;
            }
            if (!CostBenchmarkCoordinator::instance().waitFinish(
                    runIndex + 1)) {
                navi->stop();
                return 1;
            }
            auto runElapsedUs =
                autil::TimeUtility::monotonicTimeUs() - beginUs;
            if (recorded) {
                const auto resultBeginUs = autil::TimeUtility::monotonicTimeUs();
                std::vector<NaviSchedDecision> decisions;
                if (!ScheduleReplayRunner::collectDecisions(*context, decisions) ||
                    decisions.size() != recorded->requests.size()) {
                    return 1;
                }
                const auto resultEndUs = autil::TimeUtility::monotonicTimeUs();
                size_t scheduled = 0, rejected = 0;
                for (size_t index = 0; index < decisions.size(); ++index) {
                    const auto &decision = decisions[index];
                    if (decision.requestId != recorded->requests[index].request.requestId) {
                        return 1;
                    }
                    if ((decision.reason == "cost_selected" ||
                         decision.reason == "cold_start_inflight_selected") &&
                        !decision.target.address.empty() &&
                        decision.batchId > 0 && decision.expectedBatchSize > 0 &&
                        std::isfinite(decision.predictedExecutionLatencyMs)) {
                        ++scheduled;
                    } else if (decision.reason == "rate_limited" && decision.target.address.empty()) {
                        ++rejected;
                    } else {
                        std::cerr << "invalid graph replay result: request=" << decision.requestId
                                  << " reason=" << decision.reason << std::endl;
                        return 1;
                    }
                }
                std::cout << "graph_replay_result_run=" << runIndex
                          << " graph_us=" << runElapsedUs
                          << " result_prepare_us=" << resultEndUs - resultBeginUs
                          << " scheduled=" << scheduled << " rejected=" << rejected
                          << std::endl;
            }
            if (runIndex > 0) {
                elapsedUs.emplace_back(runElapsedUs);
            }
        }
        pipe->setEof();
        pipeEofGuard.pipe.reset();
        consume(userResult);
        std::sort(elapsedUs.begin(), elapsedUs.end());
        auto medianUs = elapsedUs[elapsedUs.size() / 2];
        std::cout << "requests=" << requestCount
                  << " nodes=" << nodeCount
                  << " loops=" << loopCount
                  << " parallelism=" << parallelism
                  << " budget_us=" << optimizationTimeBudgetUs
                  << " repeats=" << repeatCount
                  << " min_us=" << elapsedUs.front()
                  << " median_us=" << medianUs
                  << " max_us=" << elapsedUs.back() << std::endl;
        if (recorded) {
            std::cout << "graph_replay_restoration_outside_timing=1"
                      << " includes=navi_queue,preparation,optimization,node_selection"
                      << " excludes=result_preparation,master_admission,reservation,client_delivery,agent_updates"
                      << std::endl;
        }
        return 0;
    }

private:
    static std::unique_ptr<navi::GraphDef> buildGraph(
        size_t loopCount, size_t parallelism,
        int64_t optimizationTimeBudgetUs)
    {
        auto graphDef = std::make_unique<navi::GraphDef>();
        navi::GraphBuilder builder(graphDef.get());
        auto graphId = builder.newSubGraph("master_biz");
        builder.subGraph(graphId).location("master_biz", 1, 0);
        auto source = builder.node("cost_benchmark_source")
                          .kernel(CostBenchmarkSourceKernel::KERNEL_ID);
        auto sink = builder.node("cost_benchmark_sink")
                        .kernel(CostBenchmarkSinkKernel::KERNEL_ID);
        auto config = "{\"max_loop_count\":" +
                      std::to_string(loopCount) +
                      ",\"optimization_time_budget_us\":" +
                      std::to_string(optimizationTimeBudgetUs) +
                      ",\"parallelism\":" +
                      std::to_string(parallelism) + "}";
        if (!CostSchedulerGraphBuilder::add(
                builder, source, sink, config)) {
            return nullptr;
        }
        sink.out("result").asGraphOutput("result");
        if (!builder.ok()) {
            return nullptr;
        }
        navi::GraphBuilder::fillHashKey(graphDef.get());
        return graphDef;
    }

    static NaviSchedScheduleReplay makeReplay(
        size_t requestCount, size_t nodeCount)
    {
        NaviSchedScheduleReplay replay;
        replay.scheduleTimeUs = autil::TimeUtility::currentTime();
        replay.bizName = "model_biz";
        replay.requests.resize(requestCount);
        replay.nodeMatches.resize(requestCount);
        for (size_t requestIndex = 0; requestIndex < requestCount;
             ++requestIndex) {
            auto &request = replay.requests[requestIndex];
            request.request.requestId = requestIndex + 1;
            request.request.bizName = replay.bizName;
            request.request.promptTokenCount = 2048;
            request.request.timeoutMs = 10000;
            request.remainingTimeoutMs = 10000;
            replay.nodeMatches[requestIndex].resize(nodeCount);
        }
        const std::vector<double> parameters = {
            -4.0, 10.0, 1.4, 20.0, 0.1,
            0.09, 0.0, 1.4, 1.0, -4.0, 0.75,
        };
        replay.nodes.resize(nodeCount);
        for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
            auto &node = replay.nodes[nodeIndex];
            node.agentId = "agent_" + std::to_string(nodeIndex);
            node.agentIdHash = autil::HashAlgorithm::hashString64(
                node.agentId.data(), node.agentId.size());
            node.bizName = replay.bizName;
            node.serverAddress =
                "127.0.0.1:" + std::to_string(10000 + nodeIndex);
            node.online = true;
            node.queueTotalPredictedLatencyMs = nodeIndex % 10;
            node.queueReportReceiveTimeUs = replay.scheduleTimeUs;
            node.latencyFormula =
                NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
            node.latencyBinaryVersion =
                NonLinearPrefillModelR::BINARY_VERSION;
            node.latencyModelRevision = 1;
            node.latencyParameters = parameters;
        }
        return replay;
    }

    static NaviSchedScheduleContextPtr makeContext(
        const NaviSchedScheduleReplay &replay)
    {
        auto context = std::make_shared<NaviSchedScheduleContext>();
        if (!ScheduleReplayRunner::makeContext(replay, *context) ||
            !ScheduleReplayRunner::prepareContext(*context)) {
            return nullptr;
        }
        return context;
    }

    static void consume(const navi::NaviUserResultPtr &userResult)
    {
        bool eof = false;
        while (!eof) {
            navi::NaviUserData data;
            userResult->nextData(data, eof);
        }
    }
};

REGISTER_KERNEL(CostBenchmarkSourceKernel);
REGISTER_KERNEL(CostBenchmarkSinkKernel);

} // namespace navi_sched

int main(int argc, char **argv)
{
    if (argc == 7 && std::string(argv[1]) == "--graph-replay") {
        size_t loops = 0, workers = 0, repeats = 0;
        int64_t budgetUs = 0;
        if (!autil::StringUtil::fromString(std::string(argv[3]), loops) ||
            !autil::StringUtil::fromString(std::string(argv[4]), workers) ||
            !autil::StringUtil::fromString(std::string(argv[5]), repeats) ||
            !autil::StringUtil::fromString(std::string(argv[6]), budgetUs)) {
            return 2;
        }
        try {
            return navi_sched::CostSchedulerGraphBenchmark::graphReplay(
                argv[2], loops, workers, repeats, budgetUs);
        } catch (const std::exception &error) {
            std::cerr << error.what() << std::endl;
            return 1;
        }
    }
    if (argc == 5 && std::string(argv[1]) == "--prepare-replay") {
        size_t workers = 0, repeats = 0;
        if (!autil::StringUtil::fromString(std::string(argv[3]), workers) ||
            !autil::StringUtil::fromString(std::string(argv[4]), repeats)) {
            return 2;
        }
        try {
            return navi_sched::CostSchedulerGraphBenchmark::prepareReplay(argv[2], workers, repeats);
        } catch (const std::exception &error) {
            std::cerr << error.what() << std::endl;
            return 1;
        }
    }
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: " << argv[0]
                  << " REQUESTS NODES LOOPS PARALLELISM REPEATS"
                  << " [OPTIMIZATION_BUDGET_US]"
                  << std::endl;
        return 2;
    }
    size_t values[5];
    for (size_t index = 0; index < 5; ++index) {
        if (!autil::StringUtil::fromString(
                std::string(argv[index + 1]), values[index]) ||
            0 == values[index]) {
            return 2;
        }
    }
    int64_t optimizationTimeBudgetUs = -1;
    if (7 == argc &&
        !autil::StringUtil::fromString(
            std::string(argv[6]), optimizationTimeBudgetUs)) {
        return 2;
    }
    return navi_sched::CostSchedulerGraphBenchmark::run(
        values[0], values[1], values[2], values[3], values[4],
        optimizationTimeBudgetUs);
}
