#include "navi/pybind/NaviPybind.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/builder/GraphDesc.h"
#include "navi/engine/Navi.h"
#include "navi/engine/Resource.h"
#include "navi/engine/ResourceMap.h"
#include "navi/engine/RunGraphParams.h"
#include "navi/proto/GraphDef.pb.h"
#include "navi/proto/GraphVis.pb.h"
#include "navi/test_cluster/NaviTestCluster.h"
#include "navi/config/NaviConfig.h"
#include "navi/pybind/NaviRuntimeConfig.h"
#include <google/protobuf/util/json_util.h>
#include <kmonitor/client/MetricsReporter.h>
#include <multi_call/config/MultiCallConfig.h>
#include <multi_call/rpc/GigRpcServer.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <autil/legacy/jsonizable.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <memory>

namespace py = pybind11;

using namespace navi;
using namespace multi_call;

namespace {

template <typename T>
T parseProtoFromJson(const std::string &jsonStr) {
    T proto;
    auto status = google::protobuf::util::JsonStringToMessage(jsonStr, &proto);
    if (!status.ok()) {
        throw py::value_error("Failed to parse proto json: " + status.ToString());
    }
    return proto;
}

template <typename T>
std::string protoToJson(const T &proto) {
    std::string jsonStr;
    google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace = true;
    options.always_print_primitive_fields = true;
    auto status = google::protobuf::util::MessageToJsonString(proto, &jsonStr, options);
    if (!status.ok()) {
        throw py::value_error("Failed to dump proto json: " + status.ToString());
    }
    return jsonStr;
}

py::dict naviResultToDict(NaviResult *result) {
    py::dict d;
    if (!result) {
        d["valid"] = false;
        return d;
    }
    d["valid"] = true;
    d["has_error"] = result->hasError();
    d["error_code"] = result->getErrorCode();
    d["error_message"] = result->getErrorMessage();
    d["error_prefix"] = result->getErrorPrefix();
    d["error_bt"] = result->getErrorBt();
    d["begin_time"] = result->beginTime;
    d["end_time"] = result->endTime;
    d["queue_us"] = result->queueUs;
    d["compute_us"] = result->computeUs;
    std::vector<std::string> traceVec;
    result->collectTrace(traceVec);
    d["traces"] = traceVec;
    auto visProto = result->getVisProto();
    d["vis_json"] = visProto ? protoToJson(*visProto) : "";
    return d;
}

} // namespace

void bindNaviRegistryConfigJsonManual(py::class_<NaviRegistryConfig> &cl) {
    auto parse_dict = [](const py::dict &dict, NaviRegistryConfig &obj) {
        if (dict.contains("name")) obj.name = dict["name"].cast<std::string>();
        if (dict.contains("build_mode")) obj.buildMode = dict["build_mode"].cast<std::string>();
        else if (dict.contains("buildMode")) obj.buildMode = dict["buildMode"].cast<std::string>();
    };

    cl.def(py::init([parse_dict](const py::dict &dict) {
        auto obj = std::make_unique<NaviRegistryConfig>();
        parse_dict(dict, *obj);
        return obj;
    }), py::arg("dict_data"));

    cl.def("to_dict", [](const NaviRegistryConfig &self) {
        py::dict d;
        d["name"] = self.name;
        d["build_mode"] = self.buildMode;
        return d;
    });

    cl.def_static("from_json", [parse_dict](const std::string &jsonStr) {
        py::object jsonMod = py::module_::import("json");
        py::dict dict = jsonMod.attr("loads")(jsonStr).cast<py::dict>();
        auto obj = std::make_unique<NaviRegistryConfig>();
        parse_dict(dict, *obj);
        return obj;
    }, py::arg("json_str"));

    cl.def("to_json", [](const NaviRegistryConfig &self) {
        py::dict d;
        d["name"] = self.name;
        d["build_mode"] = self.buildMode;
        py::object jsonMod = py::module_::import("json");
        return jsonMod.attr("dumps")(d).cast<std::string>();
    });
}

template <typename T>
void bindJsonSupport(py::class_<T, std::shared_ptr<T>> &cl) {
    cl.def(py::init([](const py::dict &dict) {
        auto obj = std::make_shared<T>();
        try {
            py::object jsonMod = py::module_::import("json");
            std::string jsonStr = jsonMod.attr("dumps")(dict).cast<std::string>();
            autil::legacy::FromJsonString(*obj, jsonStr);
        } catch (const std::exception &e) {
            throw py::value_error(std::string("Failed to parse dict: ") + e.what());
        }
        return obj;
    }), py::arg("dict_data"));

    cl.def("to_dict", [](const T &self) {
        std::string jsonStr = autil::legacy::ToJsonString(self);
        py::object jsonMod = py::module_::import("json");
        return jsonMod.attr("loads")(jsonStr);
    });

    cl.def_static("from_json", [](const std::string &jsonStr) {
        auto obj = std::make_shared<T>();
        try {
            autil::legacy::FromJsonString(*obj, jsonStr);
        } catch (const std::exception &e) {
            throw py::value_error(std::string("Failed to parse JSON string: ") + e.what());
        }
        return obj;
    }, py::arg("json_str"));

    cl.def("to_json", [](const T &self) {
        return autil::legacy::ToJsonString(self);
    });
}

template <typename T>
void bindJsonSupportValue(py::class_<T> &cl) {
    cl.def(py::init([](const py::dict &dict) {
        auto obj = std::make_unique<T>();
        try {
            py::object jsonMod = py::module_::import("json");
            std::string jsonStr = jsonMod.attr("dumps")(dict).cast<std::string>();
            autil::legacy::FromJsonString(*obj, jsonStr);
        } catch (const std::exception &e) {
            throw py::value_error(std::string("Failed to parse dict: ") + e.what());
        }
        return obj;
    }), py::arg("dict_data"));

    cl.def("to_dict", [](const T &self) {
        std::string jsonStr = autil::legacy::ToJsonString(self);
        py::object jsonMod = py::module_::import("json");
        return jsonMod.attr("loads")(jsonStr);
    });

    cl.def_static("from_json", [](const std::string &jsonStr) {
        auto obj = std::make_unique<T>();
        try {
            autil::legacy::FromJsonString(*obj, jsonStr);
        } catch (const std::exception &e) {
            throw py::value_error(std::string("Failed to parse JSON string: ") + e.what());
        }
        return obj;
    }, py::arg("json_str"));

    cl.def("to_json", [](const T &self) {
        return autil::legacy::ToJsonString(self);
    });
}

void bindGraphBuilder(py::module_ &mNavi) {
    py::enum_<NodeType>(mNavi, "NodeType")
        .value("NT_NORMAL", NT_NORMAL)
        .value("NT_SPLIT", NT_SPLIT)
        .value("NT_MERGE", NT_MERGE)
        .value("NT_RESOURCE", NT_RESOURCE)
        .value("NT_FORK", NT_FORK)
        .value("NT_TERMINATOR", NT_TERMINATOR)
        .value("NT_FAKE_SPLIT", NT_FAKE_SPLIT)
        .value("NT_FAKE_MERGE", NT_FAKE_MERGE)
        .value("NT_SCOPE_TERMINATOR", NT_SCOPE_TERMINATOR)
        .value("NT_SCOPE_TERMINATOR_SPLIT", NT_SCOPE_TERMINATOR_SPLIT)
        .value("NT_UNKNOWN", NT_UNKNOWN);

    py::enum_<ErrorHandleStrategy>(mNavi, "ErrorHandleStrategy")
        .value("EHS_ERROR_AS_FATAL", EHS_ERROR_AS_FATAL)
        .value("EHS_ERROR_AS_EOF", EHS_ERROR_AS_EOF);

    py::class_<ScopeId>(mNavi, "ScopeId")
        .def(py::init<>())
        .def(py::init<GraphId, int32_t>(), py::arg("graph_id"), py::arg("scope_index"))
        .def_readwrite("graph_id", &ScopeId::graphId)
        .def_readwrite("scope_index", &ScopeId::scopeIndex);

    py::class_<N>(mNavi, "Node")
        .def(py::init<>())
        .def("kernel", [](N &self, const std::string &name) -> N & {
            return self.kernel(name);
        }, py::arg("name"), py::return_value_policy::reference_internal)
        .def("device", [](N &self, const std::string &device) -> N & {
            return self.device(device);
        }, py::arg("device"), py::return_value_policy::reference_internal)
        .def("json_attrs", [](N &self, const std::string &attrs) -> N & {
            return self.jsonAttrs(attrs);
        }, py::arg("attrs"), py::return_value_policy::reference_internal)
        .def("json_attrs_from_map", [](N &self, const std::map<std::string, std::string> &attrs) -> N & {
            return self.jsonAttrsFromMap(attrs);
        }, py::arg("attrs"), py::return_value_policy::reference_internal)
        .def("binary_attrs_from_map", [](N &self, const std::map<std::string, std::string> &attrs) -> N & {
            return self.binaryAttrsFromMap(attrs);
        }, py::arg("attrs"), py::return_value_policy::reference_internal)
        .def("attr", [](N &self, const std::string &key, const std::string &value) -> N & {
            return self.attr(key, value);
        }, py::arg("key"), py::arg("value"), py::return_value_policy::reference_internal)
        .def("integer_attr", [](N &self, const std::string &key, int64_t value) -> N & {
            return self.integerAttr(key, value);
        }, py::arg("key"), py::arg("value"), py::return_value_policy::reference_internal)
        .def("integer_array_attr", [](N &self, const std::vector<int64_t> &attrs) -> N & {
            return self.integerArrayAttr(attrs);
        }, py::arg("attrs"), py::return_value_policy::reference_internal)
        .def("skip_config", [](N &self, bool skipConfig) -> N & {
            return self.skipConfig(skipConfig);
        }, py::arg("skip_config"), py::return_value_policy::reference_internal)
        .def("skip_init", [](N &self, bool skipInit) -> N & {
            return self.skipInit(skipInit);
        }, py::arg("skip_init"), py::return_value_policy::reference_internal)
        .def("stop_after_init", [](N &self, bool stopAfterInit) -> N & {
            return self.stopAfterInit(stopAfterInit);
        }, py::arg("stop_after_init"), py::return_value_policy::reference_internal)
        .def("skip_delete_kernel", [](N &self, bool skipDeleteKernel) -> N & {
            return self.skipDeleteKernel(skipDeleteKernel);
        }, py::arg("skip_delete_kernel"), py::return_value_policy::reference_internal)
        .def("get_integer_attr", [](const N &self, const std::string &key) -> py::object {
            int64_t value = 0;
            if (self.getIntegerAttr(key, value)) {
                return py::cast(value);
            }
            return py::none();
        }, py::arg("key"))
        .def("in_port", [](const N &self, const std::string &port) {
            return self.in(port);
        }, py::arg("port"))
        .def("out_port", [](const N &self, const std::string &port) {
            return self.out(port);
        }, py::arg("port"))
        .def("depend_on", &N::dependOn, py::arg("port"))
        .def("is_valid", &N::isValid)
        .def_property_readonly("name", [](const N &self) -> std::string {
            return self.name();
        })
        .def_property_readonly("kernel_name", [](const N &self) -> std::string {
            return self.kernel();
        })
        .def_property_readonly("graph_id", &N::graphId)
        .def_property_readonly("node_type", &N::getType)
        .def_property_readonly("node_id", &N::getNodeId);

    py::class_<P>(mNavi, "Port")
        .def(py::init<>())
        .def("from_port", [](P &self, P peer) {
            return self.from(peer);
        }, py::arg("peer"))
        .def("to", [](P &self, P peer) {
            return self.to(peer);
        }, py::arg("peer"))
        .def("from_fork_node_input", &P::fromForkNodeInput,
             py::arg("input"), py::arg("index") = INVALID_INDEX)
        .def("to_fork_node_output", &P::toForkNodeOutput,
             py::arg("output"), py::arg("index") = INVALID_INDEX)
        .def("as_graph_output", &P::asGraphOutput, py::arg("name"))
        .def("as_group", &P::asGroup)
        .def("next", &P::next)
        .def("index", &P::index, py::arg("index"))
        .def("auto_next", &P::autoNext)
        .def("node", &P::node)
        .def("has_connected", &P::hasConnected)
        .def("is_valid", &P::isValid)
        .def("is_group", &P::isGroup)
        .def_property_readonly("graph_id", &P::graphId)
        .def_property_readonly("node_name", &P::getNodeName)
        .def_property_readonly("port_name", &P::getPortName)
        .def_property_readonly("port_index", &P::getIndex);

    py::class_<E>(mNavi, "Edge")
        .def(py::init<>())
        .def("split", &E::split, py::arg("split_kernel"))
        .def("merge", &E::merge, py::arg("merge_kernel"))
        .def("require", [](E &self, bool require) {
            return self.require(require);
        }, py::arg("require"))
        .def("back_pressure", [](E &self, int64_t queueLow, int64_t queueHigh) {
            return self.backPressure(queueLow, queueHigh);
        }, py::arg("queue_low"), py::arg("queue_high"))
        .def("timeout_ms", [](E &self, int64_t timeoutMs) {
            return self.timeoutMs(timeoutMs);
        }, py::arg("timeout_ms"))
        .def("is_require", &E::isRequire)
        .def("is_valid", &E::isValid);

    py::class_<GraphBuilder>(mNavi, "GraphBuilder")
        .def(py::init<GraphDef *>(), py::arg("graph_def"))
        .def("new_sub_graph", [](GraphBuilder &self, const std::string &bizName) {
            return self.newSubGraph(bizName);
        }, py::arg("biz_name"))
        .def("add_scope", &GraphBuilder::addScope)
        .def("scope", [](GraphBuilder &self, ScopeId scope) -> GraphBuilder & {
            return self.scope(scope);
        }, py::arg("scope"), py::return_value_policy::reference_internal)
        .def("get_current_scope", &GraphBuilder::getCurrentScope)
        .def("get_scope_terminator", &GraphBuilder::getScopeTerminator,
             py::arg("scope") = ScopeId())
        .def("scope_error_handle_strategy", [](GraphBuilder &self, ScopeId scope, ErrorHandleStrategy strategy) -> GraphBuilder & {
            return self.scopeErrorHandleStrategy(scope, strategy);
        }, py::arg("scope"), py::arg("strategy"), py::return_value_policy::reference_internal)
        .def("location", [](GraphBuilder &self, const std::string &bizName, NaviPartId partCount, NaviPartId partId) -> GraphBuilder & {
            return self.location(bizName, partCount, partId);
        }, py::arg("biz_name"), py::arg("part_count"), py::arg("part_id"), py::return_value_policy::reference_internal)
        .def("direct_location", [](GraphBuilder &self, const std::string &bizName, const std::string &host, int32_t grpcStreamPort) -> GraphBuilder & {
            return self.directLocation(bizName, host, grpcStreamPort);
        }, py::arg("biz_name"), py::arg("host"), py::arg("grpc_stream_port"), py::return_value_policy::reference_internal)
        .def("part_ids", [](GraphBuilder &self, const std::vector<NaviPartId> &partIds) -> GraphBuilder & {
            return self.partIds(partIds);
        }, py::arg("part_ids"), py::return_value_policy::reference_internal)
        .def("gig_probe", [](GraphBuilder &self, bool enable) -> GraphBuilder & {
            return self.gigProbe(enable);
        }, py::arg("enable"), py::return_value_policy::reference_internal)
        .def("gig_tag", [](GraphBuilder &self, const std::string &tag, multi_call::TagMatchType type) -> GraphBuilder & {
            return self.gigTag(tag, type);
        }, py::arg("tag"), py::arg("type"), py::return_value_policy::reference_internal)
        .def("sub_graph", [](GraphBuilder &self, GraphId graphId) -> GraphBuilder & {
            return self.subGraph(graphId);
        }, py::arg("graph_id"), py::return_value_policy::reference_internal)
        .def("inline_mode", [](GraphBuilder &self, bool inlineMode) -> GraphBuilder & {
            return self.inlineMode(inlineMode);
        }, py::arg("inline_mode"), py::return_value_policy::reference_internal)
        .def("error_handle_strategy", [](GraphBuilder &self, ErrorHandleStrategy strategy) -> GraphBuilder & {
            return self.errorHandleStrategy(strategy);
        }, py::arg("strategy"), py::return_value_policy::reference_internal)
        .def("sub_graph_attr", [](GraphBuilder &self, const std::string &key, const std::string &value) -> GraphBuilder & {
            return self.subGraphAttr(key, value);
        }, py::arg("key"), py::arg("value"), py::return_value_policy::reference_internal)
        .def("ignore_isolate", [](GraphBuilder &self, bool ignoreIsolate) -> GraphBuilder & {
            return self.ignoreIsolate(ignoreIsolate);
        }, py::arg("ignore_isolate"), py::return_value_policy::reference_internal)
        .def("sub_graph_timeout_ms", [](GraphBuilder &self, int64_t timeoutMs) -> GraphBuilder & {
            return self.subGraphTimeoutMs(timeoutMs);
        }, py::arg("timeout_ms"), py::return_value_policy::reference_internal)
        .def("replace_r", [](GraphBuilder &self, const std::string &from, const std::string &to) -> GraphBuilder & {
            return self.replaceR(from, to);
        }, py::arg("from_name"), py::arg("to_name"), py::return_value_policy::reference_internal)
        .def("enable_back_pressure", [](GraphBuilder &self, int64_t queueLow, int64_t queueHigh) -> GraphBuilder & {
            return self.enableBackPressure(queueLow, queueHigh);
        }, py::arg("queue_low"), py::arg("queue_high"), py::return_value_policy::reference_internal)
        .def("node", [](GraphBuilder &self, const std::string &name) {
            return self.node(name);
        }, py::arg("name"))
        .def("node", [](GraphBuilder &self, GraphId graphId, const std::string &name) {
            return self.node(graphId, name);
        }, py::arg("graph_id"), py::arg("name"))
        .def("has_node", [](const GraphBuilder &self, const std::string &name) {
            return self.hasNode(name);
        }, py::arg("name"))
        .def("has_node", [](const GraphBuilder &self, GraphId graphId, const std::string &name) {
            return self.hasNode(graphId, name);
        }, py::arg("graph_id"), py::arg("name"))
        .def("get_output_node", &GraphBuilder::getOutputNode, py::arg("name"))
        .def("get_all_nodes", &GraphBuilder::getAllNodes)
        .def("current_sub_graph", &GraphBuilder::currentSubGraph)
        .def("has_sub_graph", &GraphBuilder::hasSubGraph, py::arg("graph_id"))
        .def("get_all_graph_id", &GraphBuilder::getAllGraphId)
        .def("get_all_scope", &GraphBuilder::getAllScope, py::arg("graph_id"))
        .def("ok", &GraphBuilder::ok)
        .def_static("fill_hash_key", &GraphBuilder::fillHashKey, py::arg("graph_def"));
}

void registerNaviRuntimeConfigPybind(py::module_ &m) {
    py::class_<NaviVipSubscription>(m, "NaviVipSubscription")
        .def(py::init<>())
        .def_readwrite("jmenv_domain", &NaviVipSubscription::jmenvDomain)
        .def_readwrite("services", &NaviVipSubscription::services)
        .def_readwrite("service_key", &NaviVipSubscription::serviceKey)
        .def_readwrite("grpc_port_diff", &NaviVipSubscription::grpcPortDiff);
    py::class_<NaviRuntimeConfig>(m, "NaviRuntimeConfig")
        .def(py::init<>())
        .def_readwrite("thread_num", &NaviRuntimeConfig::threadNum)
        .def_readwrite("queue_size", &NaviRuntimeConfig::queueSize)
        .def_readwrite("processing_size", &NaviRuntimeConfig::processingSize)
        .def_readwrite("disable_perf", &NaviRuntimeConfig::disablePerf)
        .def_readwrite("disable_symbol_table",
                       &NaviRuntimeConfig::disableSymbolTable)
        .def_readwrite("vip_subscription",
                       &NaviRuntimeConfig::vipSubscription);
}

void registerNaviPybind(py::module_ &m) {
    py::module_ mNavi = m.def_submodule("navi", "pybind11 binding for Navi structures");

    registerNaviRuntimeConfigPybind(mNavi);

    py::class_<kmonitor::MetricsTags>(mNavi, "MetricsTags")
        .def(py::init<>())
        .def(py::init<const std::map<std::string, std::string> &>(), py::arg("tags"))
        .def("add_tag", &kmonitor::MetricsTags::AddTag, py::arg("key"), py::arg("value"))
        .def("find_tag", &kmonitor::MetricsTags::FindTag, py::arg("key"))
        .def("del_tag", &kmonitor::MetricsTags::DelTag, py::arg("key"))
        .def("size", &kmonitor::MetricsTags::Size)
        .def("to_string", &kmonitor::MetricsTags::ToString)
        .def("to_dict", [](const kmonitor::MetricsTags &self) { return self.GetTagsMap(); });

    py::class_<kmonitor::MetricsReporter, std::shared_ptr<kmonitor::MetricsReporter>>(mNavi, "MetricsReporter")
        .def(py::init([](const std::string &serviceName,
                         const std::string &metricsPath,
                         const std::map<std::string, std::string> &tags,
                         const std::string &monitorNamePrefix) {
                 return std::make_shared<kmonitor::MetricsReporter>(
                     serviceName, metricsPath, kmonitor::MetricsTags(tags), monitorNamePrefix);
             }),
             py::arg("service_name") = "navi",
             py::arg("metrics_path") = "",
             py::arg("tags") = std::map<std::string, std::string>(),
             py::arg("monitor_name_prefix") = "")
        .def("get_sub_reporter",
             [](kmonitor::MetricsReporter &self,
                const std::string &subPath,
                const std::map<std::string, std::string> &tags) {
                 return self.getSubReporter(subPath, kmonitor::MetricsTags(tags));
             },
             py::arg("sub_path"),
             py::arg("tags") = std::map<std::string, std::string>());

    auto cGigRpcServerConfig = py::class_<GigRpcServerConfig>(mNavi, "GigRpcServerConfig")
        .def(py::init<>())
        .def_readwrite("heartbeat_thread_num", &GigRpcServerConfig::heartbeatThreadNum)
        .def_readwrite("heartbeat_queue_size", &GigRpcServerConfig::heartbeatQueueSize)
        .def_readwrite("log_prefix", &GigRpcServerConfig::logPrefix)
        .def_readwrite("snapshot_log_dir", &GigRpcServerConfig::snapshotLogDir)
        .def_readwrite("enable_agent_stat", &GigRpcServerConfig::enableAgentStat);
    bindJsonSupportValue(cGigRpcServerConfig);

    py::class_<GrpcServerDescription>(mNavi, "GrpcServerDescription")
        .def(py::init<>())
        .def_readwrite("io_thread_num", &GrpcServerDescription::ioThreadNum)
        .def_readwrite("port", &GrpcServerDescription::port)
        .def("valid", &GrpcServerDescription::valid);

    py::class_<GigRpcServer, std::shared_ptr<GigRpcServer>>(mNavi, "GigRpcServer")
        .def(py::init<>())
        .def("init", &GigRpcServer::init, py::arg("server_config") = GigRpcServerConfig())
        .def("release", &GigRpcServer::release)
        .def("start_grpc_server", &GigRpcServer::startGrpcServer, py::arg("desc"))
        .def("init_grpc_server", &GigRpcServer::initGrpcServer, py::arg("desc"))
        .def("has_grpc", &GigRpcServer::hasGrpc)
        .def("get_grpc_port", &GigRpcServer::getGrpcPort)
        .def("start_agent", &GigRpcServer::startAgent, py::arg("smooth_interval_us") = INVALID_SMOOTH_WEIGHT_INTERVAL)
        .def("stop_agent", &GigRpcServer::stopAgent, py::arg("smooth_interval_us") = INVALID_SMOOTH_WEIGHT_INTERVAL)
        .def("get_topo_info_str", &GigRpcServer::getTopoInfoStr);

    py::class_<GraphDef, std::shared_ptr<GraphDef>>(mNavi, "GraphDef")
        .def(py::init<>())
        .def_static("from_json", [](const std::string &jsonStr) {
            return std::make_shared<GraphDef>(parseProtoFromJson<GraphDef>(jsonStr));
        }, py::arg("json_str"))
        .def_static("from_bytes", [](const py::bytes &bytes) {
            auto graphDef = std::make_shared<GraphDef>();
            std::string str = bytes;
            if (!graphDef->ParseFromString(str)) {
                throw py::value_error("Failed to parse GraphDef from bytes");
            }
            return graphDef;
        }, py::arg("bytes"))
        .def("to_json", [](const GraphDef &self) { return protoToJson(self); })
        .def("to_bytes", [](const GraphDef &self) { return py::bytes(self.SerializeAsString()); });

    bindGraphBuilder(mNavi);

    py::class_<Resource, ResourcePtr>(mNavi, "Resource")
        .def_property_readonly("name", &Resource::getName);

    py::class_<ResourceMap, std::shared_ptr<ResourceMap>>(mNavi, "ResourceMap")
        .def(py::init<>())
        .def("add_resource",
             py::overload_cast<const ResourcePtr &>(&ResourceMap::addResource),
             py::arg("resource"))
        .def("clear", &ResourceMap::clear)
        .def("empty", &ResourceMap::empty)
        .def("size", &ResourceMap::size)
        .def("has_resource", &ResourceMap::hasResource, py::arg("name"));

    py::enum_<ResourceStage>(mNavi, "ResourceStage")
        .value("RS_EXTERNAL", RS_EXTERNAL)
        .value("RS_UNKNOWN", RS_UNKNOWN)
        .value("RS_SNAPSHOT", RS_SNAPSHOT)
        .value("RS_BIZ", RS_BIZ)
        .value("RS_BIZ_PART", RS_BIZ_PART)
        .value("RS_RUN_GRAPH_EXTERNAL", RS_RUN_GRAPH_EXTERNAL)
        .value("RS_GRAPH", RS_GRAPH)
        .value("RS_SUB_GRAPH", RS_SUB_GRAPH)
        .value("RS_SCOPE", RS_SCOPE)
        .value("RS_KERNEL", RS_KERNEL);

    py::enum_<ErrorCode>(mNavi, "ErrorCode")
        .value("EC_NONE", EC_NONE)
        .value("EC_ABORT", EC_ABORT)
        .value("EC_TIMEOUT", EC_TIMEOUT)
        .value("EC_NO_DATA", EC_NO_DATA)
        .value("EC_QUEUE_FULL", EC_QUEUE_FULL)
        .value("EC_KERNEL", EC_KERNEL)
        .value("EC_UNKNOWN", EC_UNKNOWN);

    py::class_<RunGraphParams>(mNavi, "RunGraphParams")
        .def(py::init<>())
        .def_property("session_id", &RunGraphParams::getSessionId, &RunGraphParams::setSessionId)
        .def_property("thread_limit", &RunGraphParams::getThreadLimit, &RunGraphParams::setThreadLimit)
        .def_property("timeout_ms", &RunGraphParams::getTimeoutMs, &RunGraphParams::setTimeoutMs)
        .def_property("global_start_time_us", &RunGraphParams::getGlobalStartTimeUs, &RunGraphParams::setGlobalStartTimeUs)
        .def_property("task_queue_name", &RunGraphParams::getTaskQueueName, &RunGraphParams::setTaskQueueName)
        .def_property("trace_format_pattern", &RunGraphParams::getTraceFormatPattern, &RunGraphParams::setTraceFormatPattern)
        .def_property("async_io", &RunGraphParams::asyncIo, &RunGraphParams::setAsyncIo)
        .def_property("disable_gig_degrade", &RunGraphParams::disableGigDegrade, &RunGraphParams::setDisableGigDegrade)
        .def_property("max_inline", &RunGraphParams::getMaxInline, &RunGraphParams::setMaxInline)
        .def_property("resource_stage", &RunGraphParams::getResourceStage, &RunGraphParams::setResourceStage)
        .def("set_trace_level", &RunGraphParams::setTraceLevel, py::arg("trace_level"))
        .def("add_trace_bt_filter", &RunGraphParams::addTraceBtFilter, py::arg("file"), py::arg("line"))
        .def("get_trace_bt_filter", &RunGraphParams::getTraceBtFilter)
        .def("set_collect_metric", &RunGraphParams::setCollectMetric, py::arg("collect"))
        .def("set_collect_perf", &RunGraphParams::setCollectPerf, py::arg("collect"))
        .def("set_collect_context_switch", &RunGraphParams::setCollectContextSwitch, py::arg("collect"))
        .def("set_collect_page_fault", &RunGraphParams::setCollectPageFault, py::arg("collect"))
        .def("set_collect_malloc", &RunGraphParams::setCollectMalloc, py::arg("collect"))
        .def("set_debug_param_str", &RunGraphParams::setDebugParamStr, py::arg("debug_param_str"))
        .def("override_edge_data",
             [](RunGraphParams &self,
                GraphId graphId,
                const std::string &outputNode,
                const std::string &outputPort,
                const std::vector<DataPtr> &datas,
                NaviPartId partId) {
                 OverrideData data;
                 data.graphId = graphId;
                 data.partId = partId;
                 data.outputNode = outputNode;
                 data.outputPort = outputPort;
                 data.datas = datas;
                 return self.overrideEdgeData(data);
             },
             py::arg("graph_id"),
             py::arg("output_node"),
             py::arg("output_port"),
             py::arg("datas"),
             py::arg("part_id") = INVALID_NAVI_PART_ID)
        .def("add_named_data",
             [](RunGraphParams &self,
                GraphId graphId,
                const std::string &name,
                const DataPtr &data) {
                 NamedData namedData;
                 namedData.graphId = graphId;
                 namedData.name = name;
                 namedData.data = data;
                 return self.addNamedData(namedData);
             },
             py::arg("graph_id"),
             py::arg("name"),
             py::arg("data"))
        .def("clear_override_datas", &RunGraphParams::clearOverrideDatas)
        .def("clear_named_data", &RunGraphParams::clearNamedData)
        .def("add_param_topo_tree_str", &RunGraphParams::addParamTopoTreeStr, py::arg("param_str"));

    py::class_<Data, DataPtr>(mNavi, "Data")
        .def_property_readonly("type_id", &Data::getTypeId);

    py::class_<NaviUserData>(mNavi, "NaviUserData")
        .def(py::init<>())
        .def_readwrite("name", &NaviUserData::name)
        .def_readwrite("part_id", &NaviUserData::partId)
        .def_readwrite("index", &NaviUserData::index)
        .def_readwrite("stream_id", &NaviUserData::streamId)
        .def_readwrite("data", &NaviUserData::data)
        .def_readwrite("eof", &NaviUserData::eof)
        .def("to_dict", [](const NaviUserData &self) {
            py::dict d;
            d["name"] = self.name;
            d["part_id"] = self.partId;
            d["index"] = self.index;
            d["stream_id"] = self.streamId;
            d["eof"] = self.eof;
            d["data_type_id"] = self.data ? self.data->getTypeId() : "";
            d["data"] = self.data;
            return d;
        });

    py::class_<NaviResult, NaviResultPtr>(mNavi, "NaviResult")
        .def("has_error", &NaviResult::hasError)
        .def("get_error_code", &NaviResult::getErrorCode)
        .def("get_error_message", &NaviResult::getErrorMessage)
        .def("get_error_prefix", &NaviResult::getErrorPrefix)
        .def("get_error_bt", &NaviResult::getErrorBt)
        .def("collect_trace", [](const NaviResult &self) {
            std::vector<std::string> traceVec;
            self.collectTrace(traceVec);
            return traceVec;
        })
        .def("get_vis_json", [](NaviResult &self) {
            auto visProto = self.getVisProto();
            return visProto ? protoToJson(*visProto) : std::string();
        })
        .def("to_dict", [](NaviResult &self) { return naviResultToDict(&self); })
        .def_readonly("id", &NaviResult::id)
        .def_readonly("begin_time", &NaviResult::beginTime)
        .def_readonly("end_time", &NaviResult::endTime)
        .def_readonly("queue_us", &NaviResult::queueUs)
        .def_readonly("compute_us", &NaviResult::computeUs)
        .def_static("get_error_string", &NaviResult::getErrorString, py::arg("error_code"));

    py::class_<NaviUserResult, NaviUserResultPtr>(mNavi, "NaviUserResult")
        .def("next_data", [](NaviUserResult &self) {
            NaviUserData data;
            bool eof = false;
            bool valid = self.nextData(data, eof);
            return py::make_tuple(valid, eof, data);
        })
        .def("get_navi_result", &NaviUserResult::getNaviResult)
        .def("abort",
             [](NaviUserResult &self, int32_t errorCode) { self.abort(static_cast<ErrorCode>(errorCode)); },
             py::arg("error_code") = static_cast<int32_t>(EC_ABORT))
        .def("output_count", &NaviUserResult::outputCount)
        .def("wait_init", &NaviUserResult::waitInit)
        .def("collect_all", [](NaviUserResult &self) {
            py::list dataList;
            while (true) {
                NaviUserData data;
                bool eof = false;
                bool valid = self.nextData(data, eof);
                if (valid) {
                    dataList.append(data);
                }
                if (eof) {
                    break;
                }
            }
            return py::make_tuple(dataList, naviResultToDict(self.getNaviResult().get()));
        });

    py::class_<Navi, NaviPtr>(mNavi, "Navi")
        .def(py::init<const std::string &>(), py::arg("name") = "")
        .def("init",
             [](Navi &self, const std::string &initConfigPath, const std::shared_ptr<GigRpcServer> &gigRpcServer) {
                 return self.init(initConfigPath, gigRpcServer.get());
             },
             py::arg("init_config_path") = "",
             py::arg("gig_rpc_server") = nullptr,
             py::keep_alive<1, 3>())
        .def("init_metrics", [](Navi &self, kmonitor::MetricsReporter &reporter) { self.initMetrics(reporter); })
        .def("update",
             py::overload_cast<const std::string &, const ResourceMap &>(&Navi::update),
             py::arg("config_str"),
             py::arg("root_resource_map"))
        .def("update_config",
             [](Navi &self, const NaviConfig &config, const ResourceMap &rootResourceMap) {
                 return self.update(config, rootResourceMap);
             },
             py::arg("config"),
             py::arg("root_resource_map"))
        .def("run_graph",
             [](Navi &self, const GraphDef &graphDef, const RunGraphParams &params) {
                 auto *graphDefCopy = new GraphDef();
                 graphDefCopy->CopyFrom(graphDef);
                 return self.runGraph(graphDefCopy, params);
             },
             py::arg("graph_def"),
             py::arg("params"))
        .def("run_local_graph",
             [](Navi &self, const GraphDef &graphDef, const RunGraphParams &params, const ResourceMap &resourceMap) {
                 auto *graphDefCopy = new GraphDef();
                 graphDefCopy->CopyFrom(graphDef);
                 return self.runLocalGraph(graphDefCopy, params, resourceMap);
             },
             py::arg("graph_def"),
             py::arg("params"),
             py::arg("resource_map"))
        .def("stop_snapshot", &Navi::stopSnapshot)
        .def("stop", &Navi::stop)
        .def("get_topo_info_str", &Navi::getTopoInfoStr)
        .def("get_instance_id", &Navi::getInstanceId)
        .def("set_wait_no_query", &Navi::setWaitNoQuery, py::arg("need_wait"))
        .def("create_resource",
             [](const Navi &self,
                const std::string &bizName,
                NaviPartId partCount,
                NaviPartId partId,
                const std::set<std::string> &resources) {
                 auto resourceMap = std::make_shared<ResourceMap>();
                 bool ret = self.createResource(bizName, partCount, partId, resources, *resourceMap);
                 return py::make_tuple(ret, resourceMap);
             },
             py::arg("biz_name"),
             py::arg("part_count"),
             py::arg("part_id"),
             py::arg("resources"));

    py::class_<NaviTestCluster, std::shared_ptr<NaviTestCluster>>(mNavi, "NaviTestCluster")
        .def(py::init<>())
        .def("add_server",
             [](NaviTestCluster &self,
                const std::string &serverName,
                const std::string &loader,
                const std::string &configPath,
                const ResourceMapPtr &rootResourceMap) {
                 return self.addServer(serverName, loader, configPath, rootResourceMap);
             },
             py::arg("server_name"),
             py::arg("loader"),
             py::arg("config_path"),
             py::arg("root_resource_map") = ResourceMapPtr())
        .def("add_biz",
             &NaviTestCluster::addBiz,
             py::arg("server_name"),
             py::arg("biz_name"),
             py::arg("part_count"),
             py::arg("part_id"),
             py::arg("biz_config"))
        .def("start", &NaviTestCluster::start)
        .def("get_navi", &NaviTestCluster::getNavi, py::arg("server_name"))
        .def("get_biz_list", &NaviTestCluster::getBizList)
        .def("get_cluster_gig_config_json", [](const NaviTestCluster &self) {
            auto config = self.getClusterGigConfig();
            return autil::legacy::FastToJsonString(config.nodes);
        });

    auto cFlowControlConfig = py::class_<FlowControlConfig, std::shared_ptr<FlowControlConfig>>(mNavi, "FlowControlConfig")
        .def(py::init<>())
        .def_readwrite("probe_percent", &FlowControlConfig::probePercent)
        .def_readwrite("error_ratio_limit", &FlowControlConfig::errorRatioLimit)
        .def_readwrite("latency_upper_limit_percent", &FlowControlConfig::latencyUpperLimitPercent)
        .def_readwrite("latency_upper_limit_ms", &FlowControlConfig::latencyUpperLimitMs)
        .def_readwrite("begin_server_degrade_latency", &FlowControlConfig::beginServerDegradeLatency)
        .def_readwrite("begin_degrade_latency", &FlowControlConfig::beginDegradeLatency)
        .def_readwrite("full_degrade_latency", &FlowControlConfig::fullDegradeLatency)
        .def_readwrite("et_trigger_percent", &FlowControlConfig::etTriggerPercent)
        .def_readwrite("et_wait_time_factor", &FlowControlConfig::etWaitTimeFactor)
        .def_readwrite("et_min_wait_time", &FlowControlConfig::etMinWaitTime)
        .def_readwrite("retry_trigger_percent", &FlowControlConfig::retryTriggerPercent)
        .def_readwrite("retry_wait_time_factor", &FlowControlConfig::retryWaitTimeFactor)
        .def_readwrite("retry_min_wait_time", &FlowControlConfig::retryMinWaitTime)
        .def_readwrite("retry_min_provider_weight", &FlowControlConfig::retryMinProviderWeight)
        .def_readwrite("retry_limit_per_second", &FlowControlConfig::retryLimitPerSecond)
        .def_readwrite("latency_time_window_size", &FlowControlConfig::latencyTimeWindowSize)
        .def_readwrite("begin_server_degrade_error_ratio", &FlowControlConfig::beginServerDegradeErrorRatio)
        .def_readwrite("begin_degrade_error_ratio", &FlowControlConfig::beginDegradeErrorRatio)
        .def_readwrite("full_degrade_error_ratio", &FlowControlConfig::fullDegradeErrorRatio)
        .def_readwrite("min_weight", &FlowControlConfig::minWeight)
        .def_readwrite("cooldown_after_reset_ms", &FlowControlConfig::cooldownAfterResetMs)
        .def_readwrite("override_client_config", &FlowControlConfig::overrideClientConfig)
        .def_property("request_info_field",
            [](const FlowControlConfig &self) { return self.compatibleFieldInfo.requestInfoField; },
            [](FlowControlConfig &self, const std::string &val) { self.compatibleFieldInfo.requestInfoField = val; })
        .def_property("ec_field",
            [](const FlowControlConfig &self) { return self.compatibleFieldInfo.ecField; },
            [](FlowControlConfig &self, const std::string &val) { self.compatibleFieldInfo.ecField = val; })
        .def_property("response_info_field",
            [](const FlowControlConfig &self) { return self.compatibleFieldInfo.responseInfoField; },
            [](FlowControlConfig &self, const std::string &val) { self.compatibleFieldInfo.responseInfoField = val; })
        .def_property(
            "sub_config_map",
            [](const FlowControlConfig &self)
                -> std::optional<std::map<std::string, std::shared_ptr<FlowControlConfig>>> {
                if (!self.subConfigMap) {
                    return std::nullopt;
                }
                return *(self.subConfigMap);
            },
            [](FlowControlConfig &self,
               const std::optional<std::map<std::string, std::shared_ptr<FlowControlConfig>>> &mapOpt) {
                if (!mapOpt) {
                    self.subConfigMap.reset();
                } else {
                    self.subConfigMap =
                        std::make_shared<std::map<std::string, std::shared_ptr<FlowControlConfig>>>(*mapOpt);
                }
            })
        .def("et_enabled", &FlowControlConfig::etEnabled)
        .def("retry_enabled", &FlowControlConfig::retryEnabled)
        .def("single_retry_enabled", &FlowControlConfig::singleRetryEnabled)
        .def("clone", &FlowControlConfig::clone, py::return_value_policy::take_ownership)
        .def("validate", &FlowControlConfig::validate);
    bindJsonSupport(cFlowControlConfig);

    auto cLogBtFilterParam = py::class_<LogBtFilterParam>(mNavi, "LogBtFilterParam")
        .def(py::init<>())
        .def(py::init<const std::string &, int>(), py::arg("file"), py::arg("line"))
        .def_readwrite("file", &LogBtFilterParam::file)
        .def_readwrite("line", &LogBtFilterParam::line);
    bindJsonSupportValue(cLogBtFilterParam);

    auto cFileLogConfig = py::class_<FileLogConfig>(mNavi, "FileLogConfig")
        .def(py::init<>())
        .def_readwrite("file_name", &FileLogConfig::fileName)
        .def_readwrite("level_str", &FileLogConfig::levelStr)
        .def_readwrite("log_pattern", &FileLogConfig::logPattern)
        .def_readwrite("auto_flush", &FileLogConfig::autoFlush)
        .def_readwrite("max_file_size", &FileLogConfig::maxFileSize)
        .def_readwrite("delay_time_str", &FileLogConfig::delayTimeStr)
        .def_readwrite("compress", &FileLogConfig::compress)
        .def_readwrite("cache_limit", &FileLogConfig::cacheLimit)
        .def_readwrite("log_keep_count", &FileLogConfig::logKeepCount)
        .def_readwrite("async_flush", &FileLogConfig::asyncFlush)
        .def_readwrite("flush_threshold", &FileLogConfig::flushThreshold)
        .def_readwrite("flush_interval", &FileLogConfig::flushInterval)
        .def_readwrite("bt_filters", &FileLogConfig::btFilters)
        .def("translate_delay_time", &FileLogConfig::translateDelayTime);
    bindJsonSupportValue(cFileLogConfig);

    auto cLogConfig = py::class_<LogConfig>(mNavi, "LogConfig")
        .def(py::init<>())
        .def_readwrite("max_message_length", &LogConfig::maxMessageLength)
        .def_readwrite("file_appenders", &LogConfig::fileAppenders)
        .def_static("get_default_log_config", &LogConfig::getDefaultLogConfig, py::arg("name"));
    bindJsonSupportValue(cLogConfig);

    auto cGraphConfig = py::class_<GraphConfig>(mNavi, "GraphConfig")
        .def(py::init<>())
        .def_readwrite("graph_file", &GraphConfig::graphFile);
    bindJsonSupportValue(cGraphConfig);

    auto cNaviRegistryConfig = py::class_<NaviRegistryConfig>(mNavi, "NaviRegistryConfig")
        .def(py::init<>())
        .def_readwrite("name", &NaviRegistryConfig::name)
        .def_readwrite("build_mode", &NaviRegistryConfig::buildMode)
        .def("load_config", &NaviRegistryConfig::loadConfig, py::arg("name"), py::arg("config_str"))
        .def_property_readonly("config_json", [](const NaviRegistryConfig &self) -> std::string {
            if (!self.config) {
                return "{}";
            }
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            self.config->Accept(writer);
            return buffer.GetString();
        });
    bindNaviRegistryConfigJsonManual(cNaviRegistryConfig);

    auto cUpdateHistoryConfig = py::class_<UpdateHistoryConfig>(mNavi, "UpdateHistoryConfig")
        .def(py::init<>())
        .def_readwrite("max_size", &UpdateHistoryConfig::maxSize)
        .def_readwrite("max_count", &UpdateHistoryConfig::maxCount);
    bindJsonSupportValue(cUpdateHistoryConfig);

    auto cQueryLRUHistoryConfig = py::class_<QueryLRUHistoryConfig>(mNavi, "QueryLRUHistoryConfig")
        .def(py::init<>())
        .def_readwrite("max_size", &QueryLRUHistoryConfig::maxSize)
        .def_readwrite("max_count", &QueryLRUHistoryConfig::maxCount);
    bindJsonSupportValue(cQueryLRUHistoryConfig);

    auto cQueryHistoryConfig = py::class_<QueryHistoryConfig>(mNavi, "QueryHistoryConfig")
        .def(py::init<>())
        .def_readwrite("sample_interval_us", &QueryHistoryConfig::sampleIntervalUs)
        .def_readwrite("normal", &QueryHistoryConfig::normal)
        .def_readwrite("error", &QueryHistoryConfig::error);
    bindJsonSupportValue(cQueryHistoryConfig);

    auto cHistoryConfig = py::class_<HistoryConfig>(mNavi, "HistoryConfig")
        .def(py::init<>())
        .def_readwrite("update", &HistoryConfig::update)
        .def_readwrite("query", &HistoryConfig::query);
    bindJsonSupportValue(cHistoryConfig);

    auto cBizPublishInfos = py::class_<BizPublishInfos>(mNavi, "BizPublishInfos")
        .def(py::init<>())
        .def_readwrite("biz_alias_names", &BizPublishInfos::bizAliasNames)
        .def_readwrite("version", &BizPublishInfos::version)
        .def_readwrite("disable_publish", &BizPublishInfos::disablePublish);
    bindJsonSupportValue(cBizPublishInfos);

    auto cResourceReplaceConfig = py::class_<ResourceReplaceConfig>(mNavi, "ResourceReplaceConfig")
        .def(py::init<>())
        .def_readwrite("from_", &ResourceReplaceConfig::from)
        .def_readwrite("to", &ResourceReplaceConfig::to);
    bindJsonSupportValue(cResourceReplaceConfig);

    auto cCacheConfig = py::class_<CacheConfig>(mNavi, "CacheConfig")
        .def(py::init<>())
        .def_readwrite("max_cache_size_mb", &CacheConfig::maxCacheSizeMB);
    bindJsonSupportValue(cCacheConfig);

    auto cNaviBizConfig = py::class_<NaviBizConfig, std::shared_ptr<NaviBizConfig>>(mNavi, "NaviBizConfig")
        .def(py::init<>())
        .def_readwrite("part_count", &NaviBizConfig::partCount)
        .def_readwrite("part_ids", &NaviBizConfig::partIds)
        .def_readwrite("config_path", &NaviBizConfig::configPath)
        .def_readwrite("meta_info", &NaviBizConfig::metaInfo)
        .def_readwrite("modules", &NaviBizConfig::modules)
        .def_readwrite("kernels", &NaviBizConfig::kernels)
        .def_readwrite("kernel_blacklist", &NaviBizConfig::kernelBlacklist)
        .def_readwrite("resources", &NaviBizConfig::resources)
        .def_readwrite("kernel_config_vec", &NaviBizConfig::kernelConfigVec)
        .def_readwrite("resource_config_vec", &NaviBizConfig::resourceConfigVec)
        .def_readwrite("graph_config_map", &NaviBizConfig::graphConfigMap)
        .def_readwrite("publish_infos", &NaviBizConfig::publishInfos)
        .def_readwrite("flow_control_config", &NaviBizConfig::flowControlConfig)
        .def_readwrite("resource_replace_infos", &NaviBizConfig::resourceReplaceInfos)
        .def_readwrite("cache_config", &NaviBizConfig::cacheConfig);
    bindJsonSupport(cNaviBizConfig);

    auto cConcurrencyConfig = py::class_<ConcurrencyConfig>(mNavi, "ConcurrencyConfig")
        .def(py::init<>())
        .def(py::init<int, size_t, size_t>(), py::arg("thread_num"), py::arg("queue_size"), py::arg("processing_size"))
        .def_readwrite("thread_num", &ConcurrencyConfig::threadNum)
        .def_readwrite("min_thread_num", &ConcurrencyConfig::minThreadNum)
        .def_readwrite("max_thread_num", &ConcurrencyConfig::maxThreadNum)
        .def_readwrite("max_thread_pool_queue_size", &ConcurrencyConfig::maxThreadPoolQueueSize)
        .def_readwrite("queue_size", &ConcurrencyConfig::queueSize)
        .def_readwrite("processing_size", &ConcurrencyConfig::processingSize)
        .def_readwrite("thread_nice_value", &ConcurrencyConfig::threadNiceValue);
    bindJsonSupportValue(cConcurrencyConfig);

    auto cEngineConfig = py::class_<EngineConfig>(mNavi, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("disable_perf", &EngineConfig::disablePerf)
        .def_readwrite("disable_symbol_table", &EngineConfig::disableSymbolTable)
        .def_readwrite("builtin_task_queue", &EngineConfig::builtinTaskQueue)
        .def_readwrite("extra_task_queue", &EngineConfig::extraTaskQueueMap);
    bindJsonSupportValue(cEngineConfig);

    auto cNaviConfig = py::class_<NaviConfig, std::shared_ptr<NaviConfig>>(mNavi, "NaviConfig")
        .def(py::init<>())
        .def("load_config", &NaviConfig::loadConfig, py::arg("config_str"))
        .def_readwrite("config_str", &NaviConfig::configStr)
        .def_readwrite("config_path", &NaviConfig::configPath)
        .def_readwrite("modules", &NaviConfig::modules)
        .def_readwrite("log_config", &NaviConfig::logConfig)
        .def_readwrite("engine_config", &NaviConfig::engineConfig)
        .def_readwrite("resources", &NaviConfig::resources)
        .def_readwrite("kernels", &NaviConfig::kernels)
        .def_readwrite("snapshot_resource_config_vec", &NaviConfig::snapshotResourceConfigVec)
        .def_readwrite("default_biz_name", &NaviConfig::defaultBizName)
        .def_readwrite("biz_map", &NaviConfig::bizMap)
        .def_readwrite("sleep_before_update_us", &NaviConfig::sleepBeforeUpdateUs)
        .def_readwrite("sleep_after_update_us", &NaviConfig::sleepAfterUpdateUs)
        .def_readwrite("ignore_produce_error", &NaviConfig::ignoreProduceError)
        .def_readwrite("collect_init_metric", &NaviConfig::collectInitMetric)
        .def_readwrite("disable_gig_all_replica_stopped", &NaviConfig::disableGigAllReplicaStopped)
        .def_readwrite("init_timeout_ms", &NaviConfig::initTimeoutMs)
        .def_readwrite("gig_smooth_interval_us", &NaviConfig::gigSmoothIntervalUs)
        .def_readwrite("history_config", &NaviConfig::historyConfig)
        .def_static(
            "pretty_dump_to_str",
            [](const NaviConfig &config) {
                std::string configStr;
                bool success = NaviConfig::prettyDumpToStr(config, configStr);
                return std::make_pair(success, configStr);
            },
            py::arg("config"));
    bindJsonSupport(cNaviConfig);
}
