#include "rtp_llm/flexlb/navi_sched/data/AgentUpdateToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/ClientRequestToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterResponseToClientData.h"
#include "rtp_llm/flexlb/navi_sched/graph/MasterScheduleGraph.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterScheduleR.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterStateR.h"
#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerKernel.h"
#include "autil/EnvUtil.h"
#include "autil/Lock.h"
#include "autil/StringUtil.h"
#include "autil/TimeUtility.h"
#include "navi/proto/GraphVis.pb.h"
#include "navi/test_cluster/NaviTestCluster.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

namespace navi_sched {
namespace {

const char *VIS_FILE = "master_schedule_graph.vis";
const char *SUMMARY_FILE = "master_schedule_graph_vis_summary.tsv";
const char *NO_VIS_SAMPLES_FILE = "master_schedule_no_vis_samples.tsv";
const char *BIZ_NAME = "model_biz";

class ScheduleResponsePipe final : public navi::AsyncPipe {
public:
    ScheduleResponsePipe()
        : navi::AsyncPipe(nullptr, navi::AS_REQUIRED)
    {
    }

    navi::ErrorCode setData(const navi::DataPtr &data) override
    {
        autil::ScopedLock lock(_condition);
        _data = data;
        _condition.broadcast();
        return navi::EC_NONE;
    }

    navi::ErrorCode setEof() override
    {
        autil::ScopedLock lock(_condition);
        _eof = true;
        _condition.broadcast();
        return navi::EC_NONE;
    }

    MasterResponseToClientDataPtr waitResponse()
    {
        autil::ScopedLock lock(_condition);
        while (!_eof) {
            _condition.wait();
        }
        return std::dynamic_pointer_cast<MasterResponseToClientData>(_data);
    }

private:
    autil::ThreadCond _condition;
    navi::DataPtr _data;
    bool _eof = false;
};

struct KernelMetricSample {
    struct Event {
        int32_t type = navi::GET_KERNEL_UNKNOWN;
        int64_t scheduleId = 0;
        int64_t beginNs = 0;
        int64_t endNs = 0;
        int64_t queueUs = 0;
        int64_t computeUs = 0;
    };

    std::string node;
    std::string kernel;
    int64_t scheduleCount = 0;
    int64_t queueUs = 0;
    int64_t computeUs = 0;
    std::vector<Event> events;
};

void collectMetrics(const navi::GraphVisNodeDef &visNode,
                    std::vector<KernelMetricSample> &samples)
{
    if (visNode.has_result() && visNode.result().has_graph_metric()) {
        for (const auto &metric :
             visNode.result().graph_metric().metrics()) {
            KernelMetricSample sample;
            sample.node = metric.node();
            sample.kernel = metric.kernel();
            sample.scheduleCount = metric.schedule_count();
            sample.queueUs = metric.queue_latency_us();
            sample.computeUs = metric.compute_latency_us();
            for (const auto &event : metric.events()) {
                KernelMetricSample::Event sampleEvent;
                sampleEvent.type = event.type();
                sampleEvent.scheduleId = event.schedule_id();
                sampleEvent.beginNs = event.begin_time_us();
                sampleEvent.endNs = event.end_time_us();
                sampleEvent.queueUs = std::max<int64_t>(
                    0, event.schedinfo().dequeue_time() -
                           event.schedinfo().enqueue_time()) /
                    1000;
                sampleEvent.computeUs = std::max<int64_t>(
                    0, sampleEvent.endNs - sampleEvent.beginNs) / 1000;
                sample.events.emplace_back(std::move(sampleEvent));
            }
            samples.emplace_back(std::move(sample));
        }
    }
    for (const auto &pair : visNode.sub_results()) {
        collectMetrics(pair.second, samples);
    }
}

int64_t percentile(std::vector<int64_t> values, size_t numerator)
{
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    return values[(values.size() - 1) * numerator / 100];
}

bool isUnsignedInteger(const std::string &value)
{
    if (value.empty()) {
        return false;
    }
    for (char character : value) {
        if (character < '0' || character > '9') {
            return false;
        }
    }
    return true;
}

std::string getStage(const std::string &node)
{
    static const std::vector<std::string> singleIndexPrefixes = {
        "navi_sched_cost_preparation_",
        "navi_sched_cost_final_row_",
    };
    for (const auto &prefix : singleIndexPrefixes) {
        if (0 == node.compare(0, prefix.size(), prefix) &&
            isUnsignedInteger(node.substr(prefix.size()))) {
            return node.substr(0, node.rfind('_'));
        }
    }
    static const std::vector<std::string> doubleIndexPrefixes = {
        "navi_sched_cost_projection_",
        "navi_sched_cost_row_",
    };
    for (const auto &prefix : doubleIndexPrefixes) {
        if (0 != node.compare(0, prefix.size(), prefix)) {
            continue;
        }
        auto workerSeparator = node.rfind('_');
        if (std::string::npos != workerSeparator &&
            isUnsignedInteger(node.substr(prefix.size(),
                                          workerSeparator - prefix.size())) &&
            isUnsignedInteger(node.substr(workerSeparator + 1))) {
            return node.substr(0, workerSeparator);
        }
    }
    return node;
}

bool writeVisSummary(int64_t wallElapsedUs, int64_t scheduleLatencyUs)
{
    std::ifstream input(VIS_FILE, std::ios::binary);
    navi::GraphVisDef vis;
    if (!input.is_open() || !vis.ParseFromIstream(&input)) {
        std::cerr << "failed to parse " << VIS_FILE << std::endl;
        return false;
    }
    std::vector<KernelMetricSample> samples;
    for (const auto &node : vis.nodes()) {
        collectMetrics(node, samples);
    }
    std::sort(samples.begin(), samples.end(),
              [](const auto &left, const auto &right) {
                  return left.node < right.node;
              });
    std::ofstream output(SUMMARY_FILE, std::ios::trunc);
    if (!output.is_open()) {
        std::cerr << "failed to write " << SUMMARY_FILE << std::endl;
        return false;
    }
    output << "wall_elapsed_us\t" << wallElapsedUs << '\n';
    output << "schedule_latency_us\t" << scheduleLatencyUs << '\n';
    std::map<std::string, std::vector<int64_t>> computeByStage;
    std::map<std::string, std::vector<int64_t>> queueByStage;
    for (const auto &sample : samples) {
        auto stage = getStage(sample.node);
        for (const auto &event : sample.events) {
            if (navi::GET_KERNEL_COMPUTE == event.type) {
                computeByStage[stage].push_back(event.computeUs);
                queueByStage[stage].push_back(event.queueUs);
                break;
            }
        }
    }
    output << "\nstage_summary\n";
    output << "stage\tinstances\tcompute_min_us\tcompute_p50_us"
              "\tcompute_p90_us\tcompute_max_us\tqueue_p50_us"
              "\tqueue_max_us\n";
    for (const auto &pair : computeByStage) {
        const auto &stage = pair.first;
        const auto &computeValues = pair.second;
        const auto &queueValues = queueByStage[stage];
        output << stage << '\t' << computeValues.size() << '\t'
               << percentile(computeValues, 0) << '\t'
               << percentile(computeValues, 50) << '\t'
               << percentile(computeValues, 90) << '\t'
               << percentile(computeValues, 100) << '\t'
               << percentile(queueValues, 50) << '\t'
               << percentile(queueValues, 100) << '\n';
    }
    output << "\nkernel_detail\n";
    output << "node\tkernel\tschedule_count\tqueue_us\tcompute_us"
              "\tevents(type:schedule_id:begin_ns:end_ns:queue_us:"
              "compute_us)\n";
    for (const auto &sample : samples) {
        output << sample.node << '\t' << sample.kernel << '\t'
               << sample.scheduleCount << '\t' << sample.queueUs << '\t'
               << sample.computeUs << '\t';
        for (size_t index = 0; index < sample.events.size();
             ++index) {
            if (index > 0) {
                output << ',';
            }
            const auto &event = sample.events[index];
            output << event.type << ':' << event.scheduleId << ':'
                   << event.beginNs << ':' << event.endNs << ':'
                   << event.queueUs << ':' << event.computeUs;
        }
        output << '\n';
    }
    std::cout << "master_schedule_wall_elapsed_us=" << wallElapsedUs
              << "\nmaster_schedule_latency_us=" << scheduleLatencyUs
              << "\nvis_file=" << VIS_FILE
              << "\nvis_summary=" << SUMMARY_FILE << std::endl;
    return true;
}

AgentUpdateToMasterDataPtr makeHandshake(size_t nodeIndex)
{
    auto update = std::make_shared<AgentUpdateToMasterData>();
    update->event = NaviSchedAgentEvent::AGENT_HANDSHAKE;
    update->control = std::make_shared<AgentControlToMasterData>();
    update->control->serverInfo.port =
        static_cast<int32_t>(10000 + nodeIndex);
    update->latency = std::make_shared<AgentLatencyToMasterData>();
    update->latency->formula =
        NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL;
    update->latency->binaryVersion = NonLinearPrefillModelR::BINARY_VERSION;
    update->latency->modelRevision = 1;
    update->latency->parameters = {
        -4.0, 10.0, 1.4, 20.0, 0.1, 0.09, 0.0, 1.4, 1.0, -4.0, 0.75,
    };
    return update;
}

bool registerNodes(MasterStateR &state, size_t nodeCount)
{
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        auto agentId = "agent_" + std::to_string(nodeIndex);
        auto address =
            "127.0.0.1:" + std::to_string(10000 + nodeIndex);
        if (!state.registerAgent(agentId, BIZ_NAME, address, 0) ||
            !state.reportAgent(agentId, makeHandshake(nodeIndex))) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<ClientRequestToMasterData> makeRequest(size_t requestIndex)
{
    auto request = std::make_shared<ClientRequestToMasterData>();
    request->requestId = requestIndex + 1;
    request->bizName = BIZ_NAME;
    request->promptTokenCount = 2048;
    request->timeoutMs = 60000;
    return request;
}

bool loadMasterResources(
    const navi::NaviPtr &navi, MasterScheduleRPtr &schedule,
    MasterStateRPtr &state)
{
    navi::ResourceMap resources;
    if (!navi || !navi->createResource(
            "master_biz", 1, 0,
            {MasterScheduleR::RESOURCE_ID, MasterStateR::RESOURCE_ID},
            resources)) {
        return false;
    }
    auto scheduleIt =
        resources.getMap().find(MasterScheduleR::RESOURCE_ID);
    auto stateIt = resources.getMap().find(MasterStateR::RESOURCE_ID);
    if (scheduleIt == resources.getMap().end() ||
        stateIt == resources.getMap().end()) {
        return false;
    }
    schedule = std::dynamic_pointer_cast<MasterScheduleR>(
        scheduleIt->second);
    state = std::dynamic_pointer_cast<MasterStateR>(stateIt->second);
    return schedule && state;
}

int run(size_t requestCount, size_t nodeCount, size_t loopCount,
        size_t parallelism, bool enableVis, size_t repeatCount)
{
    autil::EnvGuard pythonHome(
        "NAVI_PYTHON_HOME",
        "./aios/navi/config_loader/python:/opt/conda310/lib/python3.10");
    autil::EnvGuard threadNum(
        "NAVI_SCHED_TEST_THREAD_NUM", std::to_string(parallelism));
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
    MasterScheduleRPtr schedule;
    MasterStateRPtr state;
    auto navi = cluster.getNavi("host_0");
    if (!loadMasterResources(navi, schedule, state) ||
        !registerNodes(*state, nodeCount)) {
        return 1;
    }
    auto schedulerConfig =
        "{\"max_loop_count\":" + std::to_string(loopCount) +
        ",\"optimization_time_budget_us\":-1,\"parallelism\":" +
        std::to_string(parallelism) + "}";
    MasterScheduleGraph graph(
        navi, schedule, "master_biz",
        "{\"max_wait_schedule_count\":" +
            std::to_string(requestCount) +
            ",\"max_wait_schedule_time_ms\":1000}",
        CostSchedulerKernel::KERNEL_ID, schedulerConfig, enableVis);
    if (!graph.start()) {
        return 1;
    }
    usleep(100000);
    std::vector<int64_t> wallElapsedValues;
    std::vector<int64_t> scheduleLatencyValues;
    auto runCount = enableVis ? 1 : repeatCount + 1;
    for (size_t runIndex = 0; runIndex < runCount; ++runIndex) {
        std::vector<std::shared_ptr<ScheduleResponsePipe>> responsePipes;
        responsePipes.reserve(requestCount);
        auto beginUs = autil::TimeUtility::monotonicTimeUs();
        for (size_t requestIndex = 0; requestIndex < requestCount;
             ++requestIndex) {
            auto responsePipe = std::make_shared<ScheduleResponsePipe>();
            auto uniqueRequestIndex = runIndex * requestCount + requestIndex;
            if (!schedule->submit(
                    makeRequest(uniqueRequestIndex), responsePipe)) {
                return 1;
            }
            responsePipes.emplace_back(std::move(responsePipe));
        }
        uint64_t scheduleId = 0;
        int64_t scheduleLatencyUs = 0;
        for (const auto &responsePipe : responsePipes) {
            auto response = responsePipe->waitResponse();
            if (!response || "cost_selected" != response->reason ||
                (scheduleId != 0 && scheduleId != response->scheduleId)) {
                return 1;
            }
            scheduleId = response->scheduleId;
            scheduleLatencyUs = response->scheduleLatencyUs;
        }
        if (enableVis || runIndex > 0) {
            wallElapsedValues.emplace_back(
                autil::TimeUtility::monotonicTimeUs() - beginUs);
            scheduleLatencyValues.emplace_back(scheduleLatencyUs);
        }
    }
    graph.stop();
    if (enableVis) {
        return writeVisSummary(
                   wallElapsedValues.front(),
                   scheduleLatencyValues.front())
                   ? 0
                   : 1;
    }
    std::ofstream samplesOutput(NO_VIS_SAMPLES_FILE, std::ios::trunc);
    if (!samplesOutput.is_open()) {
        return 1;
    }
    samplesOutput << "run\twall_us\tschedule_us\n";
    for (size_t index = 0; index < wallElapsedValues.size(); ++index) {
        samplesOutput << index << '\t' << wallElapsedValues[index] << '\t'
                      << scheduleLatencyValues[index] << '\n';
    }
    std::cout << "master_schedule_no_vis_repeats=" << repeatCount
              << " wall_min_us=" << percentile(wallElapsedValues, 0)
              << " wall_median_us=" << percentile(wallElapsedValues, 50)
              << " wall_max_us=" << percentile(wallElapsedValues, 100)
              << " schedule_min_us=" << percentile(scheduleLatencyValues, 0)
              << " schedule_median_us="
              << percentile(scheduleLatencyValues, 50)
              << " schedule_max_us="
              << percentile(scheduleLatencyValues, 100)
              << " samples=" << NO_VIS_SAMPLES_FILE << std::endl;
    return 0;
}

} // namespace
} // namespace navi_sched

int main(int argc, char **argv)
{
    if (argc < 5 || argc > 7) {
        std::cerr << "usage: " << argv[0]
                  << " REQUESTS NODES LOOPS PARALLELISM"
                     " [vis|no_vis] [REPEATS]"
                  << std::endl;
        return 2;
    }
    size_t values[4];
    for (size_t index = 0; index < 4; ++index) {
        if (!autil::StringUtil::fromString(
                std::string(argv[index + 1]), values[index]) ||
            0 == values[index]) {
            return 2;
        }
    }
    auto mode = argc >= 6 ? std::string(argv[5]) : "vis";
    if ("vis" != mode && "no_vis" != mode) {
        return 2;
    }
    size_t repeatCount = 1;
    if (argc == 7 &&
        (!autil::StringUtil::fromString(
             std::string(argv[6]), repeatCount) ||
         0 == repeatCount)) {
        return 2;
    }
    if ("vis" == mode && argc == 7) {
        return 2;
    }
    return navi_sched::run(
        values[0], values[1], values[2], values[3],
        "vis" == mode, repeatCount);
}
