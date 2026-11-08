#include "aios/navi_sched/collector/PPCacheAgentConfig.h"
#include "aios/navi_sched/bindings/NaviSchedBindings.h"
#include "aios/navi_sched/collector/NaviSchedCollector.h"
#include <pybind11/stl.h>

namespace py = pybind11;
namespace navi_sched {

void bindPPCacheCollector(py::module_ &module)
{
    py::class_<NaviSchedCachePolicy>(module, "PPCachePolicy")
        .def(py::init<>())
        .def_readwrite("block_size", &NaviSchedCachePolicy::blockSize)
        .def_readwrite("linear_step", &NaviSchedCachePolicy::linearStep)
        .def_readwrite("hybrid_tail_required", &NaviSchedCachePolicy::hybridTailRequired)
        .def_readwrite("full_attention_query_drop_tokens",
                       &NaviSchedCachePolicy::fullAttentionQueryDropTokens)
        .def_readwrite("mamba_replay_stop_shift",
                       &NaviSchedCachePolicy::mambaReplayStopShift)
        .def_readwrite("mamba_retention_interval",
                       &NaviSchedCachePolicy::mambaRetentionInterval)
        .def_readwrite("mamba_align", &NaviSchedCachePolicy::mambaAlign);
    py::class_<PPCacheShardEndpoint>(module, "PPCacheShardEndpoint")
        .def(py::init<>())
        .def_readwrite("shard_id", &PPCacheShardEndpoint::shardId)
        .def_readwrite("biz_name", &PPCacheShardEndpoint::bizName)
        .def_readwrite("host", &PPCacheShardEndpoint::host)
        .def_readwrite("grpc_port", &PPCacheShardEndpoint::grpcPort);
    py::class_<PPCacheAgentConfig, std::shared_ptr<PPCacheAgentConfig>>(
        module, "PPCacheAgentConfig")
        .def(py::init<>())
        .def_readwrite("pp_group", &PPCacheAgentConfig::ppGroup)
        .def_readwrite("pipeline_parallel_size",
                       &PPCacheAgentConfig::pipelineParallelSize)
        .def_readwrite("shards", &PPCacheAgentConfig::shards)
        .def_readwrite("policy", &PPCacheAgentConfig::policy)
        .def_readwrite("total_blocks", &PPCacheAgentConfig::totalBlocks)
        .def("valid", &PPCacheAgentConfig::valid);

    py::enum_<PPCacheGroupRole>(module, "PPCacheGroupRole")
        .value("FULL", PPCacheGroupRole::FULL)
        .value("TAIL", PPCacheGroupRole::TAIL);
    py::enum_<PPShardRequestStatus>(module, "PPCacheRequestStatus")
        .value("SUCCEEDED", PPShardRequestStatus::SUCCEEDED)
        .value("FAILED", PPShardRequestStatus::FAILED)
        .value("CANCELLED", PPShardRequestStatus::CANCELLED);
    py::class_<NaviSchedCollectorConfig>(module, "NaviSchedCollectorConfig")
        .def(py::init<>())
        .def_readwrite("service", &NaviSchedCollectorConfig::service)
        .def_readwrite("runtime", &NaviSchedCollectorConfig::runtime)
        .def_readwrite("pp_group", &NaviSchedCollectorConfig::ppGroup)
        .def_readwrite("shard_id", &NaviSchedCollectorConfig::shardId)
        .def_readwrite("shard_count", &NaviSchedCollectorConfig::shardCount)
        .def_readwrite("group_roles", &NaviSchedCollectorConfig::groupRoles)
        .def_readwrite("group_block_scales", &NaviSchedCollectorConfig::groupBlockScales);
    py::class_<NaviSchedCollector>(module, "NaviSchedCollector")
        .def(py::init<const NaviSchedCollectorConfig &>())
        .def("init", &NaviSchedCollector::init, py::call_guard<py::gil_scoped_release>())
        .def("get_grpc_port",
             [](const NaviSchedCollector &collector) { return collector.getGrpcPort(); })
        .def(
            "replace_resident_objects",
            [](NaviSchedCollector &collector,
               const std::vector<std::pair<std::string, std::string>> &objects) {
                std::vector<PPCacheResidentObject> resident;
                resident.reserve(objects.size());
                for (const auto &object : objects)
                    resident.push_back({object.first, object.second});
                return collector.replaceResidentObjects(resident);
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "object_resident",
            [](NaviSchedCollector &collector, const std::string &id,
               const std::string &identity) {
                return collector.objectResident({id, identity});
            },
            py::call_guard<py::gil_scoped_release>())
        .def("report_peer_cache_read_available",
             &NaviSchedCollector::reportPeerCacheReadAvailable, py::arg("available"),
             py::call_guard<py::gil_scoped_release>())
        .def("object_removed", &NaviSchedCollector::objectRemoved,
             py::call_guard<py::gil_scoped_release>())
        .def(
            "publish_terminal",
            [](NaviSchedCollector &collector, uint64_t requestId, uint64_t attemptId,
               PPShardRequestStatus status) {
                PPCacheShardUpdate update;
                update.terminals.push_back({{requestId, attemptId}, status});
                return collector.publish(update);
            },
            py::call_guard<py::gil_scoped_release>())
        .def("stop", &NaviSchedCollector::stop, py::call_guard<py::gil_scoped_release>());
}

}
