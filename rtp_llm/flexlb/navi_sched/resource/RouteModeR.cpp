#include "rtp_llm/flexlb/navi_sched/resource/RouteModeR.h"

#include "rtp_llm/flexlb/navi_sched/kernel/RouteModeKernel.h"
#include "rtp_llm/flexlb/navi_sched/proto/RouteMode.pb.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/rpc_server/ArpcRegistryParam.h"

#include <algorithm>
#include <chrono>
#include <charconv>

namespace navi_sched {

const std::string RouteModeR::RESOURCE_ID = "navi_sched.route_mode_r";
const std::string RouteModeR::RPC_REQUEST_NAME = "navi_sched.route_mode_request";

void RouteModeR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ)
        .dependOn(navi::NaviRpcServerR::RESOURCE_ID, true,
                  BIND_RESOURCE_TO(_naviRpcServerR));
}

navi::ErrorCode RouteModeR::init(navi::ResourceInitContext &ctx)
{
    _bizName = ctx.getBizName();
    navi::ArpcRegistryParam param;
    param.arpcParam.method = "route_mode";
    param.arpcParam.httpAliasVec.emplace_back("/route-mode");
    param.arpcParam.logQuery = true;
    param.graphParam.requestDataName = RPC_REQUEST_NAME;
    if (!buildGraph(param.graphParam.graph)) {
        NAVI_KERNEL_LOG(ERROR, "build route mode rpc graph failed");
        return navi::EC_ABORT;
    }
    auto service =
        _naviRpcServerR->registerArpcService<RouteModeService>(param, this);
    if (!service) {
        NAVI_KERNEL_LOG(ERROR, "register route mode rpc service failed");
        return navi::EC_ABORT;
    }
    return navi::EC_NONE;
}

bool RouteModeR::buildGraph(navi::GraphDef &graphDef) const
{
    navi::GraphBuilder builder(&graphDef);
    auto graphId = builder.newSubGraph(_bizName);
    builder.subGraph(graphId).location(_bizName, 1, 0);
    builder.node("route_mode")
        .kernel(RouteModeKernel::KERNEL_ID)
        .out(RouteModeKernel::OUTPUT_PORT)
        .asGraphOutput("out");
    navi::GraphBuilder::fillHashKey(&graphDef);
    return builder.ok();
}

RouteModeR::Snapshot RouteModeR::get() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _snapshot;
}

bool RouteModeR::update(const std::string &masterAddress, Snapshot &snapshot)
{
    if (!validMasterAddress(masterAddress)) {
        snapshot = get();
        return false;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    if (_snapshot.masterAddress != masterAddress) {
        _snapshot.masterAddress = masterAddress;
        ++_snapshot.generation;
        _condition.notify_all();
    }
    snapshot = _snapshot;
    return true;
}

bool RouteModeR::validMasterAddress(const std::string &address)
{
    if (address.empty()) {
        return true;
    }
    std::string port;
    if ('[' == address.front()) {
        auto bracket = address.find(']');
        if (std::string::npos == bracket || bracket <= 1 ||
            bracket + 2 >= address.size() || ':' != address[bracket + 1]) {
            return false;
        }
        port = address.substr(bracket + 2);
    } else {
        auto separator = address.rfind(':');
        if (std::string::npos == separator || 0 == separator ||
            separator + 1 >= address.size() ||
            address.find(':') != separator) {
            return false;
        }
        port = address.substr(separator + 1);
    }
    int32_t value = 0;
    auto result = std::from_chars(port.data(), port.data() + port.size(), value);
    return result.ec == std::errc() && result.ptr == port.data() + port.size() &&
           value > 0 && value <= 65533;
}

RouteModeR::Snapshot RouteModeR::waitAfter(uint64_t generation,
                                           int64_t timeoutMs) const
{
    std::unique_lock<std::mutex> lock(_mutex);
    _condition.wait_for(
        lock,
        std::chrono::milliseconds(std::clamp<int64_t>(timeoutMs, 1, 30000)),
        [this, generation]() { return _snapshot.generation > generation; });
    return _snapshot;
}

REGISTER_RESOURCE(RouteModeR);

} // namespace navi_sched
