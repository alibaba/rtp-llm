#include "aios/navi_sched/test/simulator/TrafficReplayAccess.h"

#include "aios/navi_sched/service/NaviSchedClient.h"
#include "aios/navi_sched/service/NaviSchedRoleRuntimeAccess.h"
#include "aios/navi_sched/test/simulator/TrafficReplayGraph.h"
#include "autil/legacy/jsonizable.h"

namespace navi_sched {
namespace {

std::string makeConfig(const test::TrafficReplayConfig &config)
{
    autil::legacy::json::JsonMap value;
    value["traffic_log"] = config.trafficLog;
    value["target_biz_name"] = config.targetBizName;
    value["master_biz_name"] = config.masterBizName;
    value["service_name"] = config.serviceName;
    value["client_count"] = config.clientCount;
    value["client_index"] = config.clientIndex;
    value["time_compression"] = config.timeCompression;
    value["request_timeout_ms"] = config.requestTimeoutMs;
    value["start_delay_ms"] = config.startDelayMs;
    value["block_size"] = config.blockSize;
    value["collect_replay_sequences"] = config.collectReplaySequences;
    value["replay_output_path"] = config.replayOutputPath;
    value["latency_output_path"] = config.latencyOutputPath;
    return autil::legacy::ToJsonString(value, true);
}

}

class TrafficReplayAccess::Impl {
public:
    std::shared_ptr<test::TrafficReplayGraph> graph;
};

std::shared_ptr<TrafficReplayAccess> TrafficReplayAccess::create(
    NaviSchedClient &client, const test::TrafficReplayConfig &trafficConfig)
{
    if (trafficConfig.clientBizName.empty() || trafficConfig.trafficLog.empty() ||
        trafficConfig.targetBizName.empty() || trafficConfig.masterBizName.empty() ||
        trafficConfig.serviceName.empty() || trafficConfig.clientCount == 0 ||
        trafficConfig.clientIndex >= trafficConfig.clientCount ||
        trafficConfig.timeCompression <= 0.0 || trafficConfig.requestTimeoutMs <= 0 ||
        trafficConfig.startDelayMs <= 0 || trafficConfig.blockSize <= 0 ||
        !client.init()) {
        return nullptr;
    }
    auto bridge = std::make_shared<test::TrafficReplayBridgeR>();
    auto graph = std::make_shared<test::TrafficReplayGraph>(
        NaviSchedRoleRuntimeAccess::getNavi(client), bridge, trafficConfig.clientBizName,
        makeConfig(trafficConfig),
        NaviSchedRoleRuntimeAccess::getConfig(client).enableVis);
    if (!graph->start()) {
        return nullptr;
    }
    auto access = std::shared_ptr<TrafficReplayAccess>(new TrafficReplayAccess());
    access->_impl = std::make_unique<Impl>();
    access->_impl->graph = std::move(graph);
    return access;
}

TrafficReplayAccess::~TrafficReplayAccess() { stop(); }

void TrafficReplayAccess::stop()
{
    if (_impl && _impl->graph) {
        _impl->graph->stop();
        _impl->graph.reset();
    }
}

}
